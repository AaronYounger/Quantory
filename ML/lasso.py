from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ModeType = Literal["pooled", "per_symbol"]


@dataclass
class LassoConfig:
    """
    Configuration for Lasso feature selection.
    """

    feature_cols: list[str]
    dependent_var: str
    mode: ModeType = "pooled"
    sample_size: Optional[int] = None
    cv_folds: int = 5
    symbols: Optional[list[str]] = None


class LassoSelector:
    CV_FOLDS_MIN = 2
    CV_FOLDS_MAX = 10

    def __init__(self, df: pd.DataFrame, config: LassoConfig):
        self.df = df.copy()
        self.config = config
        self.results = {}
        self.selected_features: list[str] = []
        self.eliminated_features: list[str] = []
        self.fitted = False

    def _validate(self):
        if not self.config.feature_cols:
            raise ValueError("At least one feature column required.")

        missing_features = [c for c in self.config.feature_cols if c not in self.df.columns]
        if missing_features:
            raise ValueError(
                f"Feature columns not found: {missing_features}. "
                "Build them first in FeatureEngine."
            )

        if self.config.dependent_var not in self.df.columns:
            raise ValueError(
                f"Dependent variable '{self.config.dependent_var}' not found. "
                "Build it first using FeatureEngine target methods."
            )

        if not (self.CV_FOLDS_MIN <= self.config.cv_folds <= self.CV_FOLDS_MAX):
            raise ValueError(
                f"cv_folds must be between {self.CV_FOLDS_MIN} "
                f"and {self.CV_FOLDS_MAX}. Got: {self.config.cv_folds}"
            )

        if self.config.mode == "per_symbol":
            if "symbol" not in self.df.columns:
                raise ValueError("Per-symbol mode requires a 'symbol' column.")
            if self.config.symbols:
                available = self.df["symbol"].unique().tolist()
                invalid = [s for s in self.config.symbols if s not in available]
                if invalid:
                    raise ValueError(
                        f"Symbols not found: {invalid}. "
                        f"Available: {available}"
                    )

    def _prepare_pooled_data(self):
        if "asset_type" in self.df.columns:
            data = self.df[self.df["asset_type"] == "equity"].copy()
        else:
            data = self.df.copy()

        cols = ["symbol", "date"] + self.config.feature_cols + [self.config.dependent_var]
        data = data[cols].dropna()

        if self.config.sample_size is not None:
            available_symbols = data["symbol"].unique()
            if len(available_symbols) > self.config.sample_size:
                sampled_symbols = np.random.choice(
                    available_symbols,
                    size=self.config.sample_size,
                    replace=False,
                )
                data = data[data["symbol"].isin(sampled_symbols)]
                print(
                    f"  Sampled {self.config.sample_size} symbols "
                    f"from {len(available_symbols)} available."
                )

        if len(data) < 50:
            raise ValueError(
                f"Not enough observations after dropping NaN. "
                f"Got {len(data)} rows. Need at least 50."
            )

        X = data[self.config.feature_cols].values
        y = data[self.config.dependent_var].values

        print(f"  Observations: {len(X):,}")
        print(f"  Features:     {len(self.config.feature_cols)}")

        return X, y

    def _prepare_per_symbol_data(self, symbol: str):
        data = self.df[self.df["symbol"] == symbol].copy()
        cols = self.config.feature_cols + [self.config.dependent_var]
        data = data[cols].dropna()

        if len(data) < 50:
            raise ValueError(
                f"Not enough data for {symbol}. "
                f"Need at least 50 rows, got {len(data)}."
            )

        X = data[self.config.feature_cols].values
        y = data[self.config.dependent_var].values
        return X, y

    def _run_lasso_cv(self, X: np.ndarray, y: np.ndarray):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=42,
        )

        lasso_cv = LassoCV(
            cv=cv,
            max_iter=10000,
            random_state=42,
            n_jobs=-1,
        )
        lasso_cv.fit(X_scaled, y)

        coefs = pd.Series(lasso_cv.coef_, index=self.config.feature_cols)
        selected = coefs[coefs != 0].sort_values(key=abs, ascending=False)
        eliminated = coefs[coefs == 0]

        return lasso_cv, scaler, coefs, selected, eliminated

    def _run_pooled(self):
        print("\n-- Lasso Feature Selection (Pooled) --")
        print(f"  Dependent Variable: {self.config.dependent_var}")
        print(f"  CV Folds:           {self.config.cv_folds}")

        X, y = self._prepare_pooled_data()
        model, scaler, coefs, selected, eliminated = self._run_lasso_cv(X, y)

        self.selected_features = selected.index.tolist()
        self.eliminated_features = eliminated.index.tolist()

        self.results["pooled"] = {
            "model": model,
            "scaler": scaler,
            "optimal_alpha": model.alpha_,
            "coefficients": coefs,
            "selected": selected,
            "eliminated": eliminated,
            "n_selected": len(selected),
            "n_eliminated": len(eliminated),
            "r_squared": model.score(scaler.transform(X), y),
        }

        self._print_pooled_results()

    def _print_pooled_results(self):
        r = self.results["pooled"]

        print("\n-- Results --")
        print(f"  Optimal Alpha:     {r['optimal_alpha']:.6f}")
        print(f"  R Squared:         {r['r_squared']:.4f}")
        print(f"  Features Selected: {r['n_selected']} / {len(self.config.feature_cols)}")
        print(f"  Features Removed:  {r['n_eliminated']}")

        print("\n-- Selected Features (ranked by importance) --")
        for feat, coef in r["selected"].items():
            direction = "up" if coef > 0 else "down"
            print(f"  {direction:4} {feat:30} coef: {coef:.6f}")

        if len(r["eliminated"]) > 0:
            print("\n-- Eliminated Features --")
            for feat in r["eliminated"].index:
                print(f"  x {feat}")

    def _run_per_symbol(self):
        symbols = self.config.symbols or self.df["symbol"].unique().tolist()

        print("\n-- Lasso Feature Selection (Per Symbol) --")
        print(f"  Dependent Variable: {self.config.dependent_var}")
        print(f"  CV Folds:           {self.config.cv_folds}")
        print(f"  Symbols:            {symbols}")

        symbol_results = {}
        all_selected: dict[str, int] = {}

        for symbol in symbols:
            try:
                print(f"\n  Running for {symbol}...")
                X, y = self._prepare_per_symbol_data(symbol)
                model, scaler, coefs, selected, eliminated = self._run_lasso_cv(X, y)

                symbol_results[symbol] = {
                    "model": model,
                    "scaler": scaler,
                    "optimal_alpha": model.alpha_,
                    "coefficients": coefs,
                    "selected": selected,
                    "eliminated": eliminated,
                    "r_squared": model.score(scaler.transform(X), y),
                }

                for feat in selected.index:
                    all_selected[feat] = all_selected.get(feat, 0) + 1

                print(
                    f"    Alpha: {model.alpha_:.6f} | "
                    f"Selected: {len(selected)} features"
                )
            except Exception as e:
                print(f"  x Failed for {symbol}: {e}")

        selection_freq = pd.Series(all_selected, dtype=float).sort_values(ascending=False)
        n_successful = len(symbol_results)
        if n_successful == 0:
            raise ValueError("Lasso failed for every symbol in per_symbol mode.")
        selection_freq_pct = (selection_freq / n_successful * 100).round(1)

        self.results["per_symbol"] = {
            "symbol_results": symbol_results,
            "selection_frequency": selection_freq,
            "selection_frequency_pct": selection_freq_pct,
        }

        self.selected_features = selection_freq[selection_freq_pct >= 50].index.tolist()
        self.eliminated_features = [f for f in self.config.feature_cols if f not in selection_freq.index]

        self._print_per_symbol_results()

    def _print_per_symbol_results(self):
        r = self.results["per_symbol"]

        print("\n-- Per Symbol Summary --")
        print(f"  Symbols run: {len(r['symbol_results'])}")

        print("\n-- Feature Selection Frequency --")
        print("  (% of successful symbols where feature was selected)\n")
        for feat, pct in r["selection_frequency_pct"].items():
            bar = "#" * int(pct / 5)
            print(f"  {feat:30} {pct:5.1f}% {bar}")

        print("\n-- Recommended Features (selected in 50%+ of symbols) --")
        for feat in self.selected_features:
            print(f"  yes {feat}")

    def plot(self):
        if not self.fitted:
            raise ValueError("Run fit() first.")

        if self.config.mode == "pooled":
            self._plot_pooled()
        else:
            self._plot_per_symbol()

    def _plot_pooled(self):
        coefs = self.results["pooled"]["coefficients"]
        coefs = coefs[coefs != 0].sort_values(key=abs, ascending=True)
        colors = ["steelblue" if c > 0 else "tomato" for c in coefs]

        fig, ax = plt.subplots(figsize=(10, max(4, len(coefs) * 0.4)))
        ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"Lasso Feature Importance (Pooled)\n"
            f"Dependent: {self.config.dependent_var} | "
            f"Alpha: {self.results['pooled']['optimal_alpha']:.6f}"
        )
        ax.set_xlabel("Coefficient")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

    def _plot_per_symbol(self):
        freq_pct = self.results["per_symbol"]["selection_frequency_pct"]
        freq_pct = freq_pct.sort_values(ascending=True)
        colors = ["steelblue" if pct >= 50 else "lightgrey" for pct in freq_pct]

        fig, ax = plt.subplots(figsize=(10, max(4, len(freq_pct) * 0.4)))
        ax.barh(freq_pct.index, freq_pct.values, color=colors, edgecolor="white")
        ax.axvline(50, color="red", linewidth=1, linestyle="--", label="50% threshold")
        ax.set_title(
            f"Lasso Feature Selection Frequency (Per Symbol)\n"
            f"Dependent: {self.config.dependent_var}"
        )
        ax.set_xlabel("% of Successful Symbols where Feature Selected")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

    def get_selected_features(self):
        if not self.fitted:
            raise ValueError("Run fit() first.")

        print("\n-- Selected Features --")
        print("  Ready to pass into Fama MacBeth:\n")
        for feat in self.selected_features:
            print(f"  yes {feat}")

        return self.selected_features

    def fit(self, plot: bool = True):
        self._validate()
        np.random.seed(42)

        if self.config.mode == "pooled":
            self._run_pooled()
        else:
            self._run_per_symbol()

        self.fitted = True

        if plot:
            self.plot()

        return self