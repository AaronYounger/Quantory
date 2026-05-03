from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from dataclasses import dataclass, field
from typing import Optional


# ── HMM Config ───────────────────────────────────────────────────────────────

@dataclass
class HMMConfig:
    """
    Configuration for HMM regime detection.
    All user inputs live here.
    """
    symbol:          str                    # symbol to run HMM on e.g. "SPY"
    feature_cols:    list[str]              # columns to use as HMM inputs
    n_states:        int = 3               # number of hidden states, default 3
    n_iter:          int = 1000            # max iterations for fitting
    covariance_type: str = "full"          # full covariance matrix


# ── HMM Model ────────────────────────────────────────────────────────────────

class HMMRegimeDetector:

    # Valid number of states
    MIN_STATES = 2
    MAX_STATES = 5

    def __init__(self, df: pd.DataFrame, config: HMMConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
            config: HMMConfig with all user inputs
        """
        self.df            = df.copy()
        self.config        = config
        self.model         = None
        self.scaler        = None
        self.symbol_df     = None
        self.hmm_df        = None
        self.state_labels  = {}   # state number → user defined label
        self.fitted        = False

    # ── Step 1 — Validate ────────────────────────────────────────────────────

    def _validate(self):
        """Validates config inputs before fitting."""

        # Symbol check
        if self.config.symbol not in self.df["symbol"].values:
            available = self.df["symbol"].unique().tolist()
            raise ValueError(
                f"Symbol '{self.config.symbol}' not found. "
                f"Available: {available}"
            )

        # State count check
        if not (self.MIN_STATES <= self.config.n_states <= self.MAX_STATES):
            raise ValueError(
                f"n_states must be between {self.MIN_STATES} and {self.MAX_STATES}. "
                f"Standard choice is 3 (bull/transition/bear). "
                f"Got: {self.config.n_states}"
            )

        # Feature columns check
        if not self.config.feature_cols:
            raise ValueError(
                "At least one feature column must be selected. "
                "Recommended: returns and volatility columns."
            )

        missing = [
            c for c in self.config.feature_cols
            if c not in self.df.columns
        ]
        if missing:
            raise ValueError(
                f"Feature columns not found: {missing}. "
                f"Build them first in FeatureEngine."
            )

    # ── Step 2 — Prepare data ────────────────────────────────────────────────

    def _prepare_data(self):
        """
        Filters to selected symbol and prepares feature matrix.
        Scales features before fitting.
        """
        self.symbol_df = (
            self.df[self.df["symbol"] == self.config.symbol]
            .sort_values("date")
            .copy()
        )

        # Build feature matrix
        feature_df = self.symbol_df[self.config.feature_cols].dropna()

        if len(feature_df) < 50:
            raise ValueError(
                f"Not enough data to fit HMM. "
                f"Need at least 50 rows, got {len(feature_df)}."
            )

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feature_df)

        return X, feature_df.index

    # ── Step 3 — Fit model ───────────────────────────────────────────────────

    def fit(self):
        """
        Fits GaussianHMM to selected symbol and features.
        Assigns raw state numbers to each date.
        """
        self._validate()

        print(f"\n── Fitting HMM ─────────────────────────────────")
        print(f"  Symbol:    {self.config.symbol}")
        print(f"  Features:  {self.config.feature_cols}")
        print(f"  States:    {self.config.n_states}")
        print(f"  Note: 3 states is standard (bull/transition/bear)")

        X, valid_index = self._prepare_data()

        # Fit model
        self.model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=42
        )

        self.model.fit(X)

        # Check convergence
        print(f"\n  Converged:       {self.model.monitor_.converged}")
        print(f"  Iterations used: {self.model.monitor_.iter}")

        if not self.model.monitor_.converged:
            print(
                f"  ⚠ Warning: Model did not converge. "
                f"Try increasing n_iter or reducing n_states."
            )

        # Predict states
        states = self.model.predict(X)

        # Build hmm_df aligned to valid dates
        self.hmm_df = self.symbol_df.loc[valid_index].copy()
        self.hmm_df["state"] = states

        # Get state probabilities
        state_probs = self.model.predict_proba(X)
        for i in range(self.config.n_states):
            self.hmm_df[f"state_prob_{i}"] = state_probs[:, i]

        self.fitted = True

        # Show state summary to help user label
        self._show_state_summary()

        return self

    # ── Step 4 — Show state summary ──────────────────────────────────────────

    def _show_state_summary(self):
        """
        Shows mean of each feature per state.
        Helps user understand what each state represents
        so they can label them appropriately.
        """
        print(f"\n── State Summary ───────────────────────────────")
        print(f"  Use this to label your states.\n")

        summary = (
            self.hmm_df
            .groupby("state")[self.config.feature_cols]
            .mean()
            .round(6)
        )

        counts = self.hmm_df["state"].value_counts().sort_index()
        summary["count"] = counts
        summary["pct"]   = (counts / len(self.hmm_df) * 100).round(1)

        print(summary.to_string())
        print(f"\n  State counts and percentages shown above.")
        print(f"  Call label_states() to assign names to each state.")

    # ── Step 5 — Label states ────────────────────────────────────────────────

    def label_states(self, labels: dict[int, str]):
        """
        User assigns human readable labels to each state number.
        Call after fit() and reviewing the state summary.

        Args:
            labels: dict mapping state number to label name
                    example: {0: "Bear", 1: "Transition", 2: "Bull"}

        Example output added to df:
            df["regime_hmm"] = "Bull" / "Bear" / "Transition"
        """
        self._check_fitted()

        # Validate all states are labeled
        expected = set(range(self.config.n_states))
        provided = set(labels.keys())

        if expected != provided:
            missing  = expected - provided
            extra    = provided - expected
            raise ValueError(
                f"Must label all {self.config.n_states} states. "
                f"Missing states: {missing}. "
                f"Extra states: {extra}. "
                f"Valid state numbers: {sorted(expected)}"
            )

        self.state_labels = labels

        # Apply labels to hmm_df
        self.hmm_df["regime"] = self.hmm_df["state"].map(labels)

        # Rename probability columns to use labels
        for state_num, label in labels.items():
            old_col = f"state_prob_{state_num}"
            new_col = f"regime_prob_{label}"
            if old_col in self.hmm_df.columns:
                self.hmm_df[new_col] = self.hmm_df[old_col]
                self.hmm_df = self.hmm_df.drop(columns=[old_col])

        print(f"\n── States Labeled ──────────────────────────────")
        for state, label in labels.items():
            count = (self.hmm_df["state"] == state).sum()
            pct   = count / len(self.hmm_df) * 100
            print(f"  State {state} → {label:15} ({count} days, {pct:.1f}%)")

        return self

    # ── Step 6 — Decompose states ────────────────────────────────────────────

    def decompose_states(self, columns: list[str] = None):
        """
        Shows mean of selected columns per regime state.
        Helps characterize what each regime looks like.

        Args:
            columns: columns to decompose by
                     if None uses the HMM feature columns
        """
        self._check_fitted()
        self._check_labeled()

        if columns is None:
            columns = self.config.feature_cols

        missing = [c for c in columns if c not in self.hmm_df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. "
                f"Available: {self.hmm_df.columns.tolist()}"
            )

        print(f"\n── Regime Decomposition ────────────────────────")

        decomp = (
            self.hmm_df
            .groupby("regime")[columns]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )

        print(decomp.to_string())

        # Also show counts
        counts = self.hmm_df["regime"].value_counts()
        print(f"\n── Regime Counts ───────────────────────────────")
        for regime, count in counts.items():
            pct = count / len(self.hmm_df) * 100
            print(f"  {regime:15} → {count} days ({pct:.1f}%)")

        return decomp

    # ── Step 7 — Add regimes to main dataset ─────────────────────────────────

    def add_to_dataset(self, target_symbols: list[str] = None):
        """
        Merges regime labels back into the main dataset.
        Regime labels from one symbol (e.g. SPY) get applied
        to all symbols or a selected list.

        This is the critical step that connects HMM output
        to strategies and portfolio switching.

        Args:
            target_symbols: symbols to apply regimes to
                            if None applies to all symbols
        """
        self._check_fitted()
        self._check_labeled()

        # Get regime columns to merge
        regime_cols = ["date", "regime"] + [
            c for c in self.hmm_df.columns
            if c.startswith("regime_prob_")
        ]

        regime_merge = self.hmm_df[regime_cols].copy()

        # Filter to target symbols if specified
        if target_symbols is not None:
            mask = self.df["symbol"].isin(target_symbols)
            self.df = self.df.copy()
            self.df.loc[mask] = (
                self.df[mask]
                .merge(regime_merge, on="date", how="left")
            )
        else:
            self.df = self.df.merge(
                regime_merge,
                on="date",
                how="left"
            )

        # Confirm columns added
        regime_label_cols = [
            c for c in self.df.columns
            if "regime" in c.lower()
        ]

        print(f"\n── Regimes Added to Dataset ────────────────────")
        print(f"  Columns added: {regime_label_cols}")
        print(f"  Rows with regime: {self.df['regime'].notna().sum()}")

        return self.df

    # ── Step 8 — Plot regime overlay ─────────────────────────────────────────

    def plot(self, price_col: str = "close"):
        """
        Plots price/return with regime overlay shading.
        Similar to your existing chart.

        Args:
            price_col: column to plot on y axis
                       default "close" but can use any column
                       e.g. "cumulative_return"
        """
        self._check_fitted()
        self._check_labeled()

        if price_col not in self.hmm_df.columns:
            raise ValueError(
                f"Column '{price_col}' not found. "
                f"Available: {self.hmm_df.columns.tolist()}"
            )

        # Color palette for regimes
        default_colors = [
            "green", "orange", "red",
            "blue",  "purple"
        ]

        regime_colors = {
            label: default_colors[i]
            for i, label in enumerate(self.state_labels.values())
        }

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot price/return
        ax.plot(
            self.hmm_df["date"],
            self.hmm_df[price_col],
            color="black",
            linewidth=1.5,
            label=price_col,
            zorder=5
        )

        y_min = self.hmm_df[price_col].min()
        y_max = self.hmm_df[price_col].max()

        # Shade regimes
        for regime, color in regime_colors.items():
            mask = self.hmm_df["regime"] == regime
            ax.fill_between(
                self.hmm_df["date"],
                y_min,
                y_max,
                where=mask,
                color=color,
                alpha=0.15,
                label=regime
            )

        ax.set_title(
            f"HMM Regime Detection — {self.config.symbol} "
            f"({self.config.n_states} states)"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(price_col)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ── Step 9 — Plot transition matrix ──────────────────────────────────────

    def plot_transition_matrix(self):
        """
        Plots the HMM transition probability matrix as a heatmap.
        Shows probability of switching from one regime to another.
        """
        self._check_fitted()
        self._check_labeled()

        import matplotlib.colors as mcolors

        labels   = [self.state_labels[i] for i in range(self.config.n_states)]
        trans_mx = self.model.transmat_

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(trans_mx, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(self.config.n_states))
        ax.set_yticks(range(self.config.n_states))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_xlabel("To Regime")
        ax.set_ylabel("From Regime")
        ax.set_title("HMM Transition Probability Matrix")

        for i in range(self.config.n_states):
            for j in range(self.config.n_states):
                ax.text(
                    j, i,
                    f"{trans_mx[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if trans_mx[i, j] < 0.7 else "white",
                    fontsize=10
                )

        plt.tight_layout()
        plt.show()

    # ── Step 10 — Get current regime ─────────────────────────────────────────

    def get_current_regime(self):
        """
        Returns the most recent regime label and probabilities.
        Useful for live regime based switching.
        """
        self._check_fitted()
        self._check_labeled()

        latest = self.hmm_df.iloc[-1]

        prob_cols = [
            c for c in self.hmm_df.columns
            if c.startswith("regime_prob_")
        ]

        print(f"\n── Current Regime ──────────────────────────────")
        print(f"  Date:    {latest['date']}")
        print(f"  Regime:  {latest['regime']}")
        print(f"\n  Probabilities:")
        for col in prob_cols:
            label = col.replace("regime_prob_", "")
            print(f"    {label:15} → {latest[col]:.2%}")

        return {
            "date":    latest["date"],
            "regime":  latest["regime"],
            "probs":   {
                col.replace("regime_prob_", ""): latest[col]
                for col in prob_cols
            }
        }

    # ── Step 11 — Predict regime for new data ────────────────────────────────

    def predict(self, new_df: pd.DataFrame):
        """
        Predicts regimes for new data using fitted model.
        Useful for out of sample regime detection.

        Args:
            new_df: new dataframe with same feature columns
        """
        self._check_fitted()
        self._check_labeled()

        missing = [
            c for c in self.config.feature_cols
            if c not in new_df.columns
        ]
        if missing:
            raise ValueError(
                f"Feature columns missing from new data: {missing}"
            )

        feature_df = new_df[self.config.feature_cols].dropna()
        X          = self.scaler.transform(feature_df)
        states     = self.model.predict(X)
        probs      = self.model.predict_proba(X)

        result              = new_df.loc[feature_df.index].copy()
        result["state"]     = states
        result["regime"]    = pd.Series(states).map(self.state_labels).values

        for i, label in self.state_labels.items():
            result[f"regime_prob_{label}"] = probs[:, i]

        return result

    # ── Master run method ────────────────────────────────────────────────────

    def run(self,
            labels: dict[int, str],
            decompose_cols: list[str] = None,
            price_col: str = "close",
            plot: bool = True):
        """
        Master method — fits model, labels states, decomposes,
        and plots in one call.

        Args:
            labels:        state number → label name
                           example: {0: "Bear", 1: "Transition", 2: "Bull"}
            decompose_cols: columns to decompose regimes by
                           if None uses feature columns
            price_col:     column to plot on y axis
            plot:          whether to show charts
        """
        self.fit()
        self.label_states(labels)
        self.decompose_states(decompose_cols)

        if plot:
            self.plot(price_col)
            self.plot_transition_matrix()

        current = self.get_current_regime()

        return {
            "model":           self.model,
            "hmm_df":          self.hmm_df,
            "state_labels":    self.state_labels,
            "current_regime":  current,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.fitted:
            raise ValueError(
                "Model not fitted yet. Call fit() first."
            )

    def _check_labeled(self):
        if not self.state_labels:
            raise ValueError(
                "States not labeled yet. Call label_states() first."
            )