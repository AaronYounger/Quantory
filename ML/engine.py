from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Optional

from ML.HMM import HMMRegimeDetector, HMMConfig
from ML.lasso import LassoSelector, LassoConfig
from ML.fama_macbeth import FamaMacBeth, FamaMacBethConfig
from ML.factor_regression import FactorRegression, FactorRegressionConfig


class MLEngine:

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: enriched dataframe from FeatureEngine
                MLEngine will enrich this further with
                regime labels, factor loadings etc.
        """
        self.df           = df.copy()   # central df — gets enriched by each tool
        self.model_log    = []          # track what has been run
        self.results      = {}          # store results from each tool

        # Stored model instances
        self._hmm          = None
        self._lasso        = None
        self._fama_macbeth = None
        self._factor_reg   = None

        # Stored outputs for inter-tool connections
        self._lasso_selected_features = None
        self._fama_macbeth_significant = None
        self._hmm_labels   = None

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, tool: str, config: dict):
        """Logs each tool run with timestamp and config."""
        self.model_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "tool":      tool,
            "config":    config,
        })

    # ── Get available columns ─────────────────────────────────────────────────

    def get_available_columns(self):
        """
        Shows all columns available for ML tools.
        Frontend uses this to populate feature selection dropdowns.
        """
        base_cols = [
            "date", "symbol", "open", "high", "low",
            "close", "volume", "asset_type", "sector",
            "industry", "market_cap", "country", "exchange",
            "employees", "currency", "shares_outstanding",
            "dividend_yield"
        ]

        available = [
            c for c in self.df.columns
            if c not in base_cols
        ]

        print("Available columns for ML tools:")
        for col in available:
            print(f"  → {col}")

        return available

    def get_available_symbols(self):
        """
        Shows all symbols available in the dataset.
        Frontend uses this to populate symbol dropdowns.
        """
        if "asset_type" in self.df.columns:
            symbols = (
                self.df[self.df["asset_type"] == "equity"]["symbol"]
                .unique()
                .tolist()
            )
        else:
            symbols = self.df["symbol"].unique().tolist()

        print("Available symbols:")
        for s in symbols:
            print(f"  → {s}")

        return symbols

    # ── HMM Regime Detection ─────────────────────────────────────────────────

    def run_hmm(self,
                symbol: str,
                feature_cols: list[str],
                labels: dict[int, str],
                n_states: int = 3,
                n_iter: int = 1000,
                decompose_cols: list[str] = None,
                price_col: str = "close",
                apply_to_all: bool = True,
                plot: bool = True):
        """
        Runs HMM regime detection and adds regime columns to central df.

        Args:
            symbol:        symbol to run HMM on e.g. "SPY"
            feature_cols:  columns to use as HMM inputs
            labels:        state number to label mapping
                           e.g. {0: "Bear", 1: "Transition", 2: "Bull"}
            n_states:      number of hidden states, default 3
                           standard is 3 (bull/transition/bear)
            n_iter:        max fitting iterations, default 1000
            decompose_cols: columns to decompose regimes by
                           if None uses feature_cols
            price_col:     column to plot on y axis
            apply_to_all:  apply regime labels to all symbols in dataset
                           if False only applies to selected symbol
            plot:          whether to show charts
        """
        print(f"\n{'='*50}")
        print(f"HMM REGIME DETECTION")
        print(f"{'='*50}")

        config = HMMConfig(
            symbol=symbol,
            feature_cols=feature_cols,
            n_states=n_states,
            n_iter=n_iter,
        )

        hmm = HMMRegimeDetector(df=self.df, config=config)

        hmm.run(
            labels=labels,
            decompose_cols=decompose_cols,
            price_col=price_col,
            plot=plot
        )

        # Add regime columns to central df
        target_symbols = None if apply_to_all else [symbol]
        self.df = hmm.add_to_dataset(target_symbols=target_symbols)

        # Store
        self._hmm        = hmm
        self._hmm_labels = labels
        self.results["hmm"] = {
            "model":          hmm.model,
            "hmm_df":         hmm.hmm_df,
            "state_labels":   labels,
            "current_regime": hmm.get_current_regime(),
        }

        self._log("HMM", {
            "symbol":       symbol,
            "feature_cols": feature_cols,
            "n_states":     n_states,
            "labels":       labels,
        })

        print(f"\n✓ Regime columns added to dataset.")
        print(f"  Regime labels: {list(labels.values())}")

        return self.results["hmm"]

    # ── Lasso Feature Selection ───────────────────────────────────────────────

    def run_lasso(self,
                  feature_cols: list[str],
                  dependent_var: str,
                  mode: Literal["pooled", "per_symbol"] = "pooled",
                  sample_size: Optional[int] = None,
                  cv_folds: int = 5,
                  symbols: Optional[list[str]] = None,
                  plot: bool = True):
        """
        Runs Lasso feature selection.
        Selected features are automatically available for Fama MacBeth.

        Args:
            feature_cols:  features to test
            dependent_var: target variable e.g. "fwd_return_21d"
            mode:          "pooled" or "per_symbol"
            sample_size:   for pooled mode, None = use all symbols
            cv_folds:      cross validation folds (2 to 10)
            symbols:       for per_symbol mode, which stocks to run
            plot:          whether to show chart
        """
        print(f"\n{'='*50}")
        print(f"LASSO FEATURE SELECTION")
        print(f"{'='*50}")

        config = LassoConfig(
            feature_cols=feature_cols,
            dependent_var=dependent_var,
            mode=mode,
            sample_size=sample_size,
            cv_folds=cv_folds,
            symbols=symbols,
        )

        lasso = LassoSelector(df=self.df, config=config)
        lasso.fit(plot=plot)

        # Store selected features for auto flow to Fama MacBeth
        self._lasso_selected_features = lasso.selected_features
        self._lasso = lasso

        self.results["lasso"] = {
            "selected_features":   lasso.selected_features,
            "eliminated_features": lasso.eliminated_features,
            "results":             lasso.results,
        }

        self._log("Lasso", {
            "feature_cols":  feature_cols,
            "dependent_var": dependent_var,
            "mode":          mode,
            "cv_folds":      cv_folds,
        })

        print(f"\n✓ Lasso complete.")
        print(f"  Selected features stored — ready for Fama MacBeth.")
        print(f"  Call run_fama_macbeth() to use them automatically.")

        return self.results["lasso"]

    # ── Fama MacBeth ──────────────────────────────────────────────────────────

    def run_fama_macbeth(self,
                         dependent_var: str,
                         feature_cols: Optional[list[str]] = None,
                         frequency: Literal["daily", "weekly", "monthly"] = "monthly",
                         min_obs: int = 10,
                         include_marginal: bool = False,
                         plot: bool = True):
        """
        Runs Fama MacBeth two stage regression.

        If feature_cols is None and Lasso has been run:
        → automatically uses Lasso selected features

        If feature_cols is None and Lasso has NOT been run:
        → raises error asking user to specify features

        Args:
            dependent_var:     target variable e.g. "fwd_return_21d"
            feature_cols:      features to test
                               if None uses Lasso selected features
            frequency:         "daily", "weekly", or "monthly"
                               monthly recommended for FF alignment
            min_obs:           minimum observations per cross section
            include_marginal:  include marginal factors in significant list
            plot:              whether to show chart
        """
        print(f"\n{'='*50}")
        print(f"FAMA MACBETH REGRESSION")
        print(f"{'='*50}")

        # ── Auto flow from Lasso ─────────────────────────────────────────────
        if feature_cols is None:
            if self._lasso_selected_features is not None:
                feature_cols = self._lasso_selected_features
                print(
                    f"  Auto using Lasso selected features: {feature_cols}"
                )
            else:
                raise ValueError(
                    "feature_cols not specified and Lasso has not been run. "
                    "Either run run_lasso() first or specify feature_cols manually."
                )

        config = FamaMacBethConfig(
            feature_cols=feature_cols,
            dependent_var=dependent_var,
            frequency=frequency,
            min_obs=min_obs,
        )

        fm = FamaMacBeth(df=self.df, config=config)
        fm.fit(plot=plot)

        # Store significant factors
        self._fama_macbeth_significant = fm.get_significant_factors(
            include_marginal=include_marginal
        )
        self._fama_macbeth = fm

        self.results["fama_macbeth"] = {
            "results":             fm.results,
            "significant_factors": self._fama_macbeth_significant,
            "frequency":           frequency,
        }

        self._log("Fama MacBeth", {
            "feature_cols":  feature_cols,
            "dependent_var": dependent_var,
            "frequency":     frequency,
        })

        print(f"\n✓ Fama MacBeth complete.")
        print(f"  Significant factors stored — ready for screener or portfolio.")

        return self.results["fama_macbeth"]

    # ── Factor Regression ─────────────────────────────────────────────────────

    def run_factor_regression(self,
                               mode: Literal["stocks", "strategy", "portfolio"] = "stocks",
                               factor_model: Literal["3_factors", "5_factors", "6_factors"] = "6_factors",
                               symbols: Optional[list[str]] = None,
                               return_col: Optional[str] = None,
                               rolling: bool = False,
                               roll_window: int = 36,
                               add_to_dataset: bool = True,
                               plot: bool = True):
        """
        Runs factor regression.
        For stocks mode adds factor loadings to central df automatically.

        Args:
            mode:           "stocks", "strategy", or "portfolio"
            factor_model:   "3_factors", "5_factors", or "6_factors"
            symbols:        for stocks mode, which stocks to run
                            None = all equities
            return_col:     for strategy/portfolio mode
                            column name of returns e.g. "total_value"
            rolling:        whether to run rolling regression
            roll_window:    rolling window in months (12 to 60)
            add_to_dataset: add factor loadings to central df
                            only for stocks mode
            plot:           whether to show charts
        """
        print(f"\n{'='*50}")
        print(f"FACTOR REGRESSION ({mode.upper()})")
        print(f"{'='*50}")

        config = FactorRegressionConfig(
            mode=mode,
            factor_model=factor_model,
            symbols=symbols,
            return_col=return_col,
            rolling=rolling,
            roll_window=roll_window,
            add_to_dataset=add_to_dataset,
        )

        fr = FactorRegression(df=self.df, config=config)
        fr.fit(plot=plot)

        # Update central df if loadings were added
        if mode == "stocks" and add_to_dataset:
            self.df = fr.df
            print(f"\n✓ Factor loadings added to central dataset.")
            print(f"  Available for screener ranking.")

        self._factor_reg = fr

        self.results[f"factor_regression_{mode}"] = {
            "summary":         fr.summary_df,
            "results":         fr.results,
            "rolling_results": fr.rolling_results,
        }

        self._log("Factor Regression", {
            "mode":         mode,
            "factor_model": factor_model,
            "symbols":      symbols,
            "rolling":      rolling,
        })

        print(f"\n✓ Factor regression complete.")

        return self.results[f"factor_regression_{mode}"]

    # ── Get enriched df ───────────────────────────────────────────────────────

    def get_df(self):
        """
        Returns the central enriched dataframe.
        Contains all ML outputs:
        → regime columns from HMM
        → factor loading columns from factor regression
        Ready to pass to screener or strategy builder.
        """
        return self.df.copy()

    # ── Get significant factors ───────────────────────────────────────────────

    def get_significant_factors(self):
        """
        Returns factors validated by Fama MacBeth.
        Use these for screener ranking or portfolio construction.
        """
        if self._fama_macbeth_significant is None:
            raise ValueError(
                "Fama MacBeth has not been run yet. "
                "Call run_fama_macbeth() first."
            )

        print(f"\n── Significant Factors ─────────────────────────")
        for f in self._fama_macbeth_significant:
            print(f"  ✓ {f}")

        return self._fama_macbeth_significant

    # ── Get current regime ────────────────────────────────────────────────────

    def get_current_regime(self):
        """
        Returns current market regime from HMM.
        Use this for live regime based strategy switching.
        """
        if self._hmm is None:
            raise ValueError(
                "HMM has not been run yet. "
                "Call run_hmm() first."
            )

        return self._hmm.get_current_regime()

    # ── Get factor loadings ───────────────────────────────────────────────────

    def get_factor_loadings(self):
        """
        Returns factor loadings per stock.
        Use these for screener ranking by factor exposure.
        """
        if self._factor_reg is None:
            raise ValueError(
                "Factor regression has not been run yet. "
                "Call run_factor_regression() first."
            )

        if self._factor_reg.config.mode != "stocks":
            raise ValueError(
                "Factor loadings only available for stocks mode."
            )

        return self._factor_reg.get_factor_loadings()

    # ── Model log ─────────────────────────────────────────────────────────────

    def get_model_log(self):
        """Shows all ML tools that have been run."""
        if not self.model_log:
            print("No ML tools run yet.")
            return pd.DataFrame()

        log_df = pd.DataFrame(self.model_log)
        print("\n── ML Model Log ────────────────────────────────")
        print(log_df[["timestamp", "tool"]].to_string(index=False))
        return log_df

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self):
        """
        Prints a summary of all ML tools run and their key outputs.
        """
        print(f"\n{'='*50}")
        print(f"ML ENGINE SUMMARY")
        print(f"{'='*50}")

        if not self.results:
            print("No ML tools have been run yet.")
            return

        # HMM summary
        if "hmm" in self.results:
            r = self.results["hmm"]
            print(f"\n── HMM Regime Detection ────────────────────────")
            print(f"  States:          {list(r['state_labels'].values())}")
            print(f"  Current Regime:  {r['current_regime']['regime']}")

        # Lasso summary
        if "lasso" in self.results:
            r = self.results["lasso"]
            print(f"\n── Lasso Feature Selection ─────────────────────")
            print(f"  Selected:   {r['selected_features']}")
            print(f"  Eliminated: {r['eliminated_features']}")

        # Fama MacBeth summary
        if "fama_macbeth" in self.results:
            r = self.results["fama_macbeth"]
            print(f"\n── Fama MacBeth ────────────────────────────────")
            print(f"  Frequency:           {r['frequency']}")
            print(f"  Significant Factors: {r['significant_factors']}")

        # Factor regression summaries
        for key in self.results:
            if key.startswith("factor_regression"):
                mode = key.replace("factor_regression_", "")
                r    = self.results[key]
                print(f"\n── Factor Regression ({mode}) ───────────────────")
                if r["summary"] is not None:
                    print(r["summary"][["label", "alpha_annual_pct", "r_squared_adj"]].to_string(index=False))

        # Enriched df columns
        print(f"\n── Enriched Dataset ────────────────────────────")
        ml_cols = [
            c for c in self.df.columns
            if any(x in c for x in ["regime", "factor_"])
        ]
        if ml_cols:
            print(f"  ML columns added: {ml_cols}")
        else:
            print(f"  No ML columns added to dataset yet.")