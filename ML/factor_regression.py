from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import getFamaFrenchFactors as gff
from dataclasses import dataclass, field
from typing import Literal, Optional
from scipy import stats


ModeType     = Literal["stocks", "strategy", "portfolio"]
FactorModel  = Literal["3_factors", "5_factors", "6_factors"]


# ── Factor Regression Config ─────────────────────────────────────────────────

@dataclass
class FactorRegressionConfig:
    """
    Configuration for Factor Regression.
    All user inputs live here.
    """
    mode:           ModeType      = "stocks"      # stocks, strategy, or portfolio
    factor_model:   FactorModel   = "6_factors"   # which FF model to use
    rolling:        bool          = False          # full period or rolling
    roll_window:    int           = 36             # rolling window in months
    add_to_dataset: bool          = False          # add loadings to main df
                                                   # only relevant for stocks mode

    # For stocks mode
    symbols:        Optional[list[str]] = None     # which stocks to run
                                                   # None = all equities

    # For strategy/portfolio mode
    return_col:     Optional[str] = None           # column name of returns
                                                   # e.g. "total_value"


# ── Factor Regression ─────────────────────────────────────────────────────────

class FactorRegression:

    # Significance thresholds
    SIGNIFICANT_ALPHA = 0.05
    MARGINAL_ALPHA    = 0.10

    # Roll window limits
    ROLL_WINDOW_MIN = 12   # minimum 12 months
    ROLL_WINDOW_MAX = 60   # maximum 60 months

    def __init__(self,
                 df: pd.DataFrame,
                 config: FactorRegressionConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
                    for stocks mode: must have symbol, date, close columns
                    for strategy/portfolio mode: must have date and return_col
            config: FactorRegressionConfig with all user inputs
        """
        self.df            = df.copy()
        self.config        = config
        self.ff_factors    = None    # FF factor data
        self.results       = {}      # full period results per symbol
        self.rolling_results = {}    # rolling results per symbol
        self.summary_df    = None    # summary table across all symbols
        self.fitted        = False

    # ── Step 1 — Validate ────────────────────────────────────────────────────

    def _validate(self):
        """Validates all inputs before running."""

        # Roll window
        if self.config.rolling:
            if not (self.ROLL_WINDOW_MIN <= self.config.roll_window <= self.ROLL_WINDOW_MAX):
                raise ValueError(
                    f"roll_window must be between {self.ROLL_WINDOW_MIN} "
                    f"and {self.ROLL_WINDOW_MAX} months. "
                    f"Got: {self.config.roll_window}"
                )

        # Mode specific validation
        if self.config.mode == "stocks":
            if "symbol" not in self.df.columns:
                raise ValueError("df must have a 'symbol' column for stocks mode.")
            if "close" not in self.df.columns:
                raise ValueError("df must have a 'close' column for stocks mode.")

            if self.config.symbols:
                available = self.df["symbol"].unique().tolist()
                invalid   = [s for s in self.config.symbols if s not in available]
                if invalid:
                    raise ValueError(
                        f"Symbols not found: {invalid}. "
                        f"Available: {available}"
                    )

        elif self.config.mode in ["strategy", "portfolio"]:
            if self.config.return_col is None:
                raise ValueError(
                    f"return_col must be specified for {self.config.mode} mode. "
                    f"e.g. return_col='total_value'"
                )
            if self.config.return_col not in self.df.columns:
                raise ValueError(
                    f"Column '{self.config.return_col}' not found in df."
                )

    # ── Step 2 — Fetch FF factors ─────────────────────────────────────────────

    def _fetch_factors(self):
        """
        Fetches Fama French factors using getFamaFrenchFactors.
        Combines 5 factor model with momentum for 6 factor model.
        Always fetches monthly data.

        Note: daily data is resampled to monthly internally.
        FF factors are only available at monthly frequency.
        """
        print(f"  Fetching Fama French factors ({self.config.factor_model})...")

        if self.config.factor_model == "3_factors":
            ff = gff.famaFrench3Factor()
            ff = ff.set_index("date_ff_factors").to_period("M")
            # 3 factor: Mkt-RF, SMB, HML, RF

        elif self.config.factor_model == "5_factors":
            ff = gff.famaFrench5Factor()
            ff = ff.set_index("date_ff_factors").to_period("M")
            # 5 factor: Mkt-RF, SMB, HML, RMW, CMA, RF

        elif self.config.factor_model == "6_factors":
            ff5 = gff.famaFrench5Factor().set_index("date_ff_factors").to_period("M")
            mom = gff.momentumFactor().set_index("date_ff_factors").to_period("M")
            ff  = ff5.join(mom, how="left")
            # 6 factor: Mkt-RF, SMB, HML, RMW, CMA, MOM, RF

        # Move RF to last column
        rf  = ff[["RF"]]
        ff  = ff.drop(columns=["RF"])
        ff  = ff.join(rf)

        self.ff_factors = ff
        print(f"  Factors fetched: {ff.columns.drop('RF').tolist()}")
        print(f"  Date range: {ff.index[0]} → {ff.index[-1]}")

        return ff

    # ── Step 3 — Prepare stock returns ────────────────────────────────────────

    def _prepare_stock_returns(self):
        """
        Prepares monthly return series per stock.
        Resamples daily OHLCV data to monthly.

        Note: FF factors are monthly so daily data
        is resampled to monthly close prices internally.
        """
        if "asset_type" in self.df.columns:
            data = self.df[self.df["asset_type"] == "equity"].copy()
        else:
            data = self.df.copy()

        symbols = self.config.symbols or data["symbol"].unique().tolist()

        print(f"  Resampling daily data to monthly...")
        print(f"  Note: FF factors are monthly — daily prices resampled to month end close.")

        monthly_returns = {}

        for symbol in symbols:
            sym_df = (
                data[data["symbol"] == symbol]
                .set_index("date")
                .sort_index()["close"]
            )

            # Resample to monthly
            monthly_close = sym_df.resample("ME").last()
            monthly_close.index = monthly_close.index.to_period("M")
            monthly_ret = monthly_close.pct_change().dropna()

            if len(monthly_ret) < self.config.roll_window + 6:
                print(f"  ⚠ Skipping {symbol} — not enough monthly observations.")
                continue

            monthly_returns[symbol] = monthly_ret

        if not monthly_returns:
            raise ValueError(
                "No symbols had enough data for factor regression. "
                "Check your date range and symbols."
            )

        rets_df = pd.DataFrame(monthly_returns)
        print(f"  Monthly returns prepared: {rets_df.shape[1]} symbols, {len(rets_df)} months")

        return rets_df

    # ── Step 4 — Prepare strategy/portfolio returns ───────────────────────────

    def _prepare_single_returns(self):
        """
        Prepares monthly return series for a single strategy or portfolio.
        Converts portfolio value to returns then resamples to monthly.
        """
        data = self.df.set_index("date").sort_index()

        print(f"  Resampling {self.config.mode} returns to monthly...")
        print(f"  Note: FF factors are monthly — daily values resampled to month end.")

        # Resample to monthly
        monthly_values = data[self.config.return_col].resample("ME").last()
        monthly_values.index = monthly_values.index.to_period("M")
        monthly_ret = monthly_values.pct_change().dropna()

        if len(monthly_ret) < 12:
            raise ValueError(
                f"Not enough monthly observations for factor regression. "
                f"Need at least 12 months, got {len(monthly_ret)}."
            )

        print(f"  Monthly returns prepared: {len(monthly_ret)} months")

        return monthly_ret

    # ── Step 5 — Run single regression ───────────────────────────────────────

    def _run_regression(self, rets: pd.Series, label: str):
        """
        Runs full period OLS factor regression for one return series.
        Returns factor loadings, t-stats, p-values, alpha, and R².
        """
        ff = self.ff_factors

        # Align on common dates
        common_idx = rets.index.intersection(ff.index)

        # ── Sample size warnings ──────────────────────────────────────────────
        if len(common_idx) < 12:
            raise ValueError(
                f"Not enough common dates between {label} returns "
                f"and FF factors. Got {len(common_idx)} months."
            )
        elif len(common_idx) < 36:
            print(
                f"  ⚠ Warning: {label} has only {len(common_idx)} monthly observations. "
                f"Recommend at least 36 months for reliable results."
            )
        elif len(common_idx) < 60:
            print(
                f"  ℹ Note: {label} has {len(common_idx)} monthly observations. "
                f"60+ months gives more reliable results."
            )
        # ─────────────────────────────────────────────────────────────────────

        rets_aligned = rets.loc[common_idx]
        ff_aligned   = ff.loc[common_idx]

        # Excess returns
        rets_excess = rets_aligned - ff_aligned["RF"].values

        # Factor columns (drop RF)
        factor_cols  = [c for c in ff_aligned.columns if c != "RF"]
        X            = ff_aligned[factor_cols].copy()
        X["Alpha"]   = 1.0

        # OLS regression
        model  = sm.OLS(rets_excess, X).fit()

        # Extract results
        loadings = model.params
        tstats   = model.tvalues
        pvalues  = model.pvalues

        # Alpha metrics
        alpha_monthly    = loadings["Alpha"]
        alpha_annualized = ((1 + alpha_monthly) ** 12) - 1

        # Significance flags
        def _sig_flag(p):
            if p < self.SIGNIFICANT_ALPHA:
                return "✓ Yes"
            elif p < self.MARGINAL_ALPHA:
                return "~ Marginal"
            else:
                return "✗ No"

        # Build results dict
        result = {
            "label":             label,
            "n_months":          len(common_idx),
            "r_squared":         round(model.rsquared, 4),
            "r_squared_adj":     round(model.rsquared_adj, 4),
            "alpha_monthly_pct": round(alpha_monthly * 100, 4),
            "alpha_annual_pct":  round(alpha_annualized * 100, 4),
            "alpha_tstat":       round(tstats["Alpha"], 3),
            "alpha_pvalue":      round(pvalues["Alpha"], 4),
            "alpha_significant": _sig_flag(pvalues["Alpha"]),
        }

        # Factor loadings
        for factor in factor_cols:
            result[f"{factor}_loading"]  = round(loadings[factor], 4)
            result[f"{factor}_tstat"]    = round(tstats[factor], 3)
            result[f"{factor}_pvalue"]   = round(pvalues[factor], 4)
            result[f"{factor}_sig"]      = _sig_flag(pvalues[factor])

        return result, model

    # ── Step 6 — Run rolling regression ──────────────────────────────────────

    def _run_rolling_regression(self, rets: pd.Series, label: str):
        """
        Runs rolling window factor regression.
        Shows how factor exposures change over time.
        """
        ff = self.ff_factors

        common_idx   = rets.index.intersection(ff.index)
        rets_aligned = rets.loc[common_idx]
        ff_aligned   = ff.loc[common_idx]

        rets_excess = rets_aligned - ff_aligned["RF"].values
        factor_cols = [c for c in ff_aligned.columns if c != "RF"]

        n_periods = len(common_idx)
        window    = self.config.roll_window

        if n_periods <= window:
            print(f"  ⚠ {label}: sample smaller than roll window. Skipping rolling.")
            return None

        windows = [
            (start, start + window)
            for start in range(n_periods - window + 1)
        ]

        roll_results = []

        for win in windows:
            try:
                X = ff_aligned[factor_cols].iloc[win[0]:win[1]].copy()
                X["Alpha"] = 1.0
                y = rets_excess.iloc[win[0]:win[1]]

                model    = sm.OLS(y, X).fit()
                row      = dict(model.params)
                row["date"] = common_idx[win[1] - 1]
                roll_results.append(row)

            except Exception:
                continue

        roll_df = pd.DataFrame(roll_results).set_index("date")

        return roll_df

    # ── Step 7 — Run stocks mode ──────────────────────────────────────────────

    def _run_stocks(self):
        """Runs factor regression for each stock."""

        print(f"\n── Factor Regression (Stocks Mode) ────────────")
        print(f"  Factor Model: {self.config.factor_model}")
        print(f"  Rolling:      {self.config.rolling}")

        rets_df = self._prepare_stock_returns()
        rows    = []

        for symbol in rets_df.columns:
            try:
                print(f"  Running for {symbol}...")
                result, model = self._run_regression(rets_df[symbol], symbol)
                self.results[symbol] = {
                    "result": result,
                    "model":  model
                }
                rows.append(result)

                if self.config.rolling:
                    roll_df = self._run_rolling_regression(rets_df[symbol], symbol)
                    if roll_df is not None:
                        self.rolling_results[symbol] = roll_df

            except Exception as e:
                print(f"  ✗ Failed for {symbol}: {e}")

        self.summary_df = pd.DataFrame(rows)
        self._print_stocks_results()

        # Add factor loadings back to main dataset if requested
        if self.config.add_to_dataset:
            self._add_loadings_to_dataset()

    def _print_stocks_results(self):
        """Prints factor regression results table for stocks."""

        print(f"\n── Results ─────────────────────────────────────")

        factor_cols = [c for c in self.ff_factors.columns if c != "RF"]

        # Build display table
        display_cols = ["label", "alpha_monthly_pct", "alpha_annual_pct",
                        "alpha_tstat", "alpha_significant", "r_squared_adj"]

        for factor in factor_cols:
            display_cols.append(f"{factor}_loading")
            display_cols.append(f"{factor}_sig")

        available = [c for c in display_cols if c in self.summary_df.columns]
        print(self.summary_df[available].to_string(index=False))

    # ── Step 8 — Run strategy/portfolio mode ──────────────────────────────────

    def _run_single(self):
        """Runs factor regression for a single strategy or portfolio."""

        label = self.config.mode.capitalize()

        print(f"\n── Factor Regression ({label} Mode) ────────────")
        print(f"  Factor Model: {self.config.factor_model}")
        print(f"  Rolling:      {self.config.rolling}")

        rets = self._prepare_single_returns()

        result, model = self._run_regression(rets, label)

        self.results[label] = {
            "result": result,
            "model":  model
        }

        self.summary_df = pd.DataFrame([result])

        self._print_single_results(result, model)

        if self.config.rolling:
            roll_df = self._run_rolling_regression(rets, label)
            if roll_df is not None:
                self.rolling_results[label] = roll_df

    def _print_single_results(self, result: dict, model):
        """Prints factor regression results for strategy/portfolio."""

        factor_cols = [c for c in self.ff_factors.columns if c != "RF"]

        print(f"\n── Factor Decomposition ────────────────────────")
        print(f"  Months:           {result['n_months']}")
        print(f"  R² Adjusted:      {result['r_squared_adj']}")
        print(f"\n── Alpha ───────────────────────────────────────")
        print(f"  Monthly Alpha:    {result['alpha_monthly_pct']}%")
        print(f"  Annualized Alpha: {result['alpha_annual_pct']}%")
        print(f"  T-Stat:           {result['alpha_tstat']}")
        print(f"  P-Value:          {result['alpha_pvalue']}")
        print(f"  Significant:      {result['alpha_significant']}")
        print(f"\n── Factor Loadings ─────────────────────────────")

        for factor in factor_cols:
            loading = result.get(f"{factor}_loading", "N/A")
            tstat   = result.get(f"{factor}_tstat", "N/A")
            sig     = result.get(f"{factor}_sig", "N/A")
            print(
                f"  {factor:10} loading: {loading:8} "
                f"t-stat: {tstat:8} {sig}"
            )

    # ── Step 9 — Add loadings to dataset ─────────────────────────────────────

    def _add_loadings_to_dataset(self):
        """
        Adds factor loadings as columns to the main dataset.
        Only for stocks mode.
        Factor loadings become features for screener ranking.
        """
        factor_cols = [c for c in self.ff_factors.columns if c != "RF"]

        print(f"\n── Adding Factor Loadings to Dataset ───────────")

        for _, row in self.summary_df.iterrows():
            symbol = row["label"]
            mask   = self.df["symbol"] == symbol

            # Add each factor loading as a column
            for factor in factor_cols:
                col_name = f"factor_{factor.lower().replace('-', '_')}_loading"
                self.df.loc[mask, col_name] = row.get(f"{factor}_loading", np.nan)

            # Add alpha
            self.df.loc[mask, "factor_alpha_monthly"]  = row.get("alpha_monthly_pct", np.nan)
            self.df.loc[mask, "factor_alpha_annual"]   = row.get("alpha_annual_pct", np.nan)
            self.df.loc[mask, "factor_r2_adj"]         = row.get("r_squared_adj", np.nan)

        loading_cols = [
            f"factor_{f.lower().replace('-', '_')}_loading"
            for f in factor_cols
        ] + ["factor_alpha_monthly", "factor_alpha_annual", "factor_r2_adj"]

        print(f"  Columns added: {loading_cols}")
        print(f"  These can now be used in the screener for factor based ranking.")

        return self.df

    # ── Step 10 — Plot ────────────────────────────────────────────────────────

    def plot(self, symbol: str = None):
        """
        Plots factor regression results.
        Stocks mode → bar chart of factor loadings per stock
        Strategy/Portfolio → loading bar chart + rolling chart if enabled

        Args:
            symbol: for stocks mode, which symbol to plot rolling for
                    if None plots summary bar chart
        """
        if not self.fitted:
            raise ValueError("Run fit() first.")

        if self.config.mode == "stocks":
            if symbol is None:
                self._plot_stocks_summary()
            else:
                self._plot_stock_loadings(symbol)
                if self.config.rolling and symbol in self.rolling_results:
                    self._plot_rolling(symbol)
        else:
            label = self.config.mode.capitalize()
            self._plot_single_loadings(label)
            if self.config.rolling and label in self.rolling_results:
                self._plot_rolling(label)

    def _plot_stocks_summary(self):
        """Bar chart of alpha per stock."""
        df = self.summary_df.copy()

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.5)))

        # Alpha bar chart
        colors = [
            "steelblue" if a > 0 else "tomato"
            for a in df["alpha_annual_pct"]
        ]
        axes[0].barh(
            df["label"],
            df["alpha_annual_pct"],
            color=colors,
            edgecolor="white"
        )
        axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title(f"Annualized Alpha by Stock\n({self.config.factor_model})")
        axes[0].set_xlabel("Annualized Alpha (%)")
        axes[0].grid(True, alpha=0.3, axis="x")

        # R² adjusted bar chart
        axes[1].barh(
            df["label"],
            df["r_squared_adj"],
            color="steelblue",
            edgecolor="white"
        )
        axes[1].set_title("Adjusted R² by Stock")
        axes[1].set_xlabel("R² Adjusted")
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.show()

    def _plot_stock_loadings(self, symbol: str):
        """Bar chart of factor loadings for a single stock."""
        if symbol not in self.results:
            raise ValueError(f"No results for {symbol}. Run fit() first.")

        result      = self.results[symbol]["result"]
        factor_cols = [c for c in self.ff_factors.columns if c != "RF"]

        loadings = pd.Series({
            f: result.get(f"{f}_loading", 0)
            for f in factor_cols
        })

        colors = ["steelblue" if v > 0 else "tomato" for v in loadings]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(loadings.index, loadings.values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"{symbol} Factor Loadings\n"
            f"Alpha: {result['alpha_annual_pct']}% annualized | "
            f"R²adj: {result['r_squared_adj']}"
        )
        ax.set_xlabel("Loading")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

    def _plot_single_loadings(self, label: str):
        """Bar chart of factor loadings for strategy/portfolio."""
        result      = self.results[label]["result"]
        factor_cols = [c for c in self.ff_factors.columns if c != "RF"]

        loadings = pd.Series({
            f: result.get(f"{f}_loading", 0)
            for f in factor_cols
        })

        colors = ["steelblue" if v > 0 else "tomato" for v in loadings]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(loadings.index, loadings.values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"{label} Factor Decomposition\n"
            f"Alpha: {result['alpha_annual_pct']}% annualized | "
            f"R²adj: {result['r_squared_adj']}"
        )
        ax.set_xlabel("Loading")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

    def _plot_rolling(self, label: str):
        """Line chart of rolling factor loadings over time."""
        roll_df     = self.rolling_results[label]
        factor_cols = [c for c in roll_df.columns if c != "Alpha"]

        n = len(factor_cols)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, factor in zip(axes, factor_cols):
            ax.plot(
                roll_df.index.to_timestamp(),
                roll_df[factor],
                linewidth=1.5,
                label=factor
            )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{label} — Rolling {factor} Loading ({self.config.roll_window}M window)")
            ax.set_ylabel("Loading")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    # ── Step 11 — Get factor loadings for screener ────────────────────────────

    def get_factor_loadings(self):
        """
        Returns factor loadings per stock as a clean dataframe.
        Use this to feed into the screener for factor based ranking.
        Only relevant for stocks mode.
        """
        if not self.fitted:
            raise ValueError("Run fit() first.")

        if self.config.mode != "stocks":
            raise ValueError(
                "get_factor_loadings() is only for stocks mode. "
                "For strategy/portfolio use get_results()."
            )

        factor_cols = [c for c in self.ff_factors.columns if c != "RF"]

        cols = ["label", "alpha_monthly_pct", "alpha_annual_pct",
                "alpha_significant", "r_squared_adj"]

        for factor in factor_cols:
            cols.append(f"{factor}_loading")
            cols.append(f"{factor}_sig")

        available = [c for c in cols if c in self.summary_df.columns]
        loadings  = self.summary_df[available].rename(columns={"label": "symbol"})

        print(f"\n── Factor Loadings (ready for screener) ────────")
        print(loadings.to_string(index=False))

        return loadings

    def get_results(self, label: str = None):
        """
        Returns full regression results.

        Args:
            label: symbol name for stocks mode
                   or "Strategy"/"Portfolio" for other modes
                   if None returns summary df
        """
        if not self.fitted:
            raise ValueError("Run fit() first.")

        if label is not None:
            if label not in self.results:
                available = list(self.results.keys())
                raise ValueError(
                    f"Label '{label}' not found. "
                    f"Available: {available}"
                )
            return self.results[label]

        return self.summary_df

    # ── Master run method ─────────────────────────────────────────────────────

    def fit(self, plot: bool = True):
        """
        Runs factor regression.

        Args:
            plot: whether to show charts
        """
        self._validate()
        self._fetch_factors()

        if self.config.mode == "stocks":
            self._run_stocks()
        else:
            self._run_single()

        self.fitted = True

        if plot:
            self.plot()

        return self