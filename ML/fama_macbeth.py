from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import Literal, Optional


# ── Fama MacBeth Config ──────────────────────────────────────────────────────

@dataclass
class FamaMacBethConfig:
    """
    Configuration for Fama MacBeth regression.
    All user inputs live here.
    """
    feature_cols:  list[str]
    dependent_var: str
    frequency:     Literal["daily", "weekly", "monthly"] = "monthly"
    min_obs:       int = 10


# ── Fama MacBeth ─────────────────────────────────────────────────────────────

class FamaMacBeth:

    SIGNIFICANT_ALPHA = 0.05
    MARGINAL_ALPHA    = 0.10

    # Frequency to resample rule mapping
    FREQ_RESAMPLE_MAP = {
        "daily":   None,    # no resampling needed
        "weekly":  "W-FRI", # resample to weekly
        "monthly": "ME",    # resample to month end
    }

    # Frequency labels for display
    FREQ_LABELS = {
        "daily":   "Daily",
        "weekly":  "Weekly",
        "monthly": "Monthly",
    }

    def __init__(self, df: pd.DataFrame, config: FamaMacBethConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
            config: FamaMacBethConfig with all user inputs
        """
        self.df      = df.copy()
        self.config  = config
        self.results = None
        self.fitted  = False

    # ── Step 1 — Validate ────────────────────────────────────────────────────

    def _validate(self):
        """Validates all inputs before running."""

        if not self.config.feature_cols:
            raise ValueError("At least one feature column required.")

        missing_features = [
            c for c in self.config.feature_cols
            if c not in self.df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Feature columns not found: {missing_features}. "
                f"Build them first in FeatureEngine."
            )

        if self.config.dependent_var not in self.df.columns:
            raise ValueError(
                f"Dependent variable '{self.config.dependent_var}' not found. "
                f"Build it first using FeatureEngine.add_forward_return() "
                f"or other target methods."
            )

        if "asset_type" in self.df.columns:
            equity_df = self.df[self.df["asset_type"] == "equity"]
        else:
            equity_df = self.df

        n_symbols = equity_df["symbol"].nunique()
        if n_symbols < 5:
            raise ValueError(
                f"Fama MacBeth requires at least 5 symbols. "
                f"Got: {n_symbols}."
            )

        # ── Frequency alignment warning ──────────────────────────────────────
        dep_var = self.config.dependent_var.lower()

        if self.config.frequency == "monthly":
            print(
                f"  ✓ Running at monthly frequency. "
                f"Aligned with FF factor regression."
            )
            # Warn if dependent variable looks daily or weekly
            if any(x in dep_var for x in ["1d", "2d", "3d", "4d", "5d", "7d"]):
                print(
                    f"  ⚠ Warning: dependent variable '{self.config.dependent_var}' "
                    f"appears to be a short window. "
                    f"For monthly Fama MacBeth a 21 day forward return "
                    f"(fwd_return_21d) is recommended for alignment "
                    f"with FF factor loadings."
                )

        elif self.config.frequency == "daily":
            print(
                f"  ⚠ Note: Running at daily frequency. "
                f"Risk premiums will be in daily terms. "
                f"If you plan to use these results alongside factor regression "
                f"switch to monthly for consistent interpretation. "
                f"Daily and monthly risk premiums are not directly comparable."
            )

        elif self.config.frequency == "weekly":
            print(
                f"  ⚠ Note: Running at weekly frequency. "
                f"Risk premiums will be in weekly terms. "
                f"If you plan to use these results alongside factor regression "
                f"switch to monthly for consistent interpretation."
            )

    # ── Step 2 — Prepare data ────────────────────────────────────────────────

    def _prepare_data(self):
        """
        Filters to equity symbols and selects relevant columns.
        Resamples to selected frequency if not daily.
        """
        if "asset_type" in self.df.columns:
            data = self.df[self.df["asset_type"] == "equity"].copy()
        else:
            data = self.df.copy()

        cols = (
            ["symbol", "date"] +
            self.config.feature_cols +
            [self.config.dependent_var]
        )

        data = data[cols].dropna()
        data = data.sort_values(["date", "symbol"])

        # ── Resample if not daily ────────────────────────────────────────────
        resample_rule = self.FREQ_RESAMPLE_MAP[self.config.frequency]

        if resample_rule is not None:
            print(
                f"  Resampling data to {self.config.frequency}. "
                f"Taking last value per period per symbol."
            )

            data["date"] = pd.to_datetime(data["date"])

            data = (
                data
                .groupby("symbol")
                .apply(
                    lambda g: g.set_index("date")
                    .resample(resample_rule)
                    .last()
                    .reset_index()
                )
                .reset_index(drop=True)
            )

            data = data.dropna()
            data = data.sort_values(["date", "symbol"])

            print(
                f"  Resampled: {data['date'].nunique()} "
                f"{self.config.frequency} periods."
            )

        return data

    # ── Step 3 — Stage 1: Cross sectional regressions ────────────────────────

    def _stage_1(self, data: pd.DataFrame):
        """
        Stage 1 of Fama MacBeth.
        For each time period run a cross sectional OLS regression.
        Returns a dataframe of betas per period.
        """
        period_betas = []
        dates        = sorted(data["date"].unique())

        print(f"  Running Stage 1 cross sectional regressions...")
        print(f"  Frequency: {self.FREQ_LABELS[self.config.frequency]}")
        print(f"  Periods:   {len(dates)}")

        for date in dates:
            period_data = data[data["date"] == date]

            if len(period_data) < self.config.min_obs:
                continue

            y = period_data[self.config.dependent_var].values
            X = period_data[self.config.feature_cols].values

            X_with_const = np.column_stack([np.ones(len(X)), X])

            try:
                coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

                row = {"date": date, "alpha": coeffs[0]}
                for i, feat in enumerate(self.config.feature_cols):
                    row[feat] = coeffs[i + 1]

                period_betas.append(row)

            except Exception:
                continue

        betas_df = pd.DataFrame(period_betas).set_index("date")

        print(f"  Periods used: {len(betas_df)}")

        return betas_df

    # ── Step 4 — Stage 2: Time series averages ───────────────────────────────

    def _stage_2(self, betas_df: pd.DataFrame):
        """
        Stage 2 of Fama MacBeth.
        Calculates mean, standard error, t-stat, and p-value
        for each factor.
        """
        print(f"  Running Stage 2 time series averaging...")

        rows = []

        for factor in self.config.feature_cols:
            if factor not in betas_df.columns:
                continue

            factor_betas = betas_df[factor].dropna()
            n            = len(factor_betas)

            if n < 2:
                continue

            mean    = factor_betas.mean()
            std_err = factor_betas.std() / np.sqrt(n)
            t_stat  = mean / std_err if std_err != 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

            if p_value < self.SIGNIFICANT_ALPHA:
                significant = "✓ Yes"
            elif p_value < self.MARGINAL_ALPHA:
                significant = "~ Marginal"
            else:
                significant = "✗ No"

            rows.append({
                "factor":        factor,
                "frequency":     self.FREQ_LABELS[self.config.frequency],
                "risk_premium":  round(mean * 100, 4),
                "std_error":     round(std_err * 100, 4),
                "t_stat":        round(t_stat, 3),
                "p_value":       round(p_value, 4),
                "n_periods":     n,
                "significant":   significant,
            })

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values(
            "t_stat", key=abs, ascending=False
        ).reset_index(drop=True)

        return results_df

    # ── Step 5 — Print results ───────────────────────────────────────────────

    def _print_results(self):
        """Prints Fama MacBeth results table."""

        print(f"\n── Fama MacBeth Results ────────────────────────")
        print(f"  Dependent Variable: {self.config.dependent_var}")
        print(f"  Frequency:          {self.FREQ_LABELS[self.config.frequency]}")
        print(f"  Factors Tested:     {len(self.config.feature_cols)}")
        print(f"\n")

        display = self.results.copy()
        display["risk_premium"] = display["risk_premium"].apply(
            lambda x: f"{x:.4f}%"
        )
        display["std_error"] = display["std_error"].apply(
            lambda x: f"{x:.4f}%"
        )
        display["t_stat"]  = display["t_stat"].apply(lambda x: f"{x:.3f}")
        display["p_value"] = display["p_value"].apply(lambda x: f"{x:.4f}")

        print(display.to_string(index=False))

        n_significant = (self.results["significant"] == "✓ Yes").sum()
        n_marginal    = (self.results["significant"] == "~ Marginal").sum()
        n_not         = (self.results["significant"] == "✗ No").sum()

        print(f"\n── Significance Summary ────────────────────────")
        print(f"  ✓ Significant (p < 0.05):  {n_significant} factors")
        print(f"  ~ Marginal   (p < 0.10):   {n_marginal} factors")
        print(f"  ✗ Not significant:          {n_not} factors")

        print(f"\n── Significant Factors ─────────────────────────")
        sig = self.results[self.results["significant"] == "✓ Yes"]
        if len(sig) > 0:
            for _, row in sig.iterrows():
                direction = "positive" if row["risk_premium"] > 0 else "negative"
                print(
                    f"  ✓ {row['factor']:30} "
                    f"premium: {row['risk_premium']:.4f}% "
                    f"per {self.config.frequency} ({direction})"
                )
        else:
            print("  No factors with significant risk premiums found.")

    # ── Step 6 — Plot ────────────────────────────────────────────────────────

    def plot(self):
        """Plots risk premiums and t-statistics."""
        if not self.fitted:
            raise ValueError("Run fit() first.")

        df     = self.results.copy()
        df     = df.sort_values("t_stat", key=abs, ascending=True)
        colors = []

        for _, row in df.iterrows():
            if row["significant"] == "✗ No":
                colors.append("lightgrey")
            elif row["risk_premium"] > 0:
                colors.append("steelblue")
            else:
                colors.append("tomato")

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.5)))

        # Risk premium chart
        axes[0].barh(
            df["factor"],
            df["risk_premium"],
            color=colors,
            edgecolor="white"
        )
        axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title(
            f"Risk Premiums ({self.FREQ_LABELS[self.config.frequency]})\n"
            f"Dependent: {self.config.dependent_var}"
        )
        axes[0].set_xlabel(
            f"Risk Premium (% per {self.config.frequency})"
        )
        axes[0].grid(True, alpha=0.3, axis="x")

        for i, (_, row) in enumerate(df.iterrows()):
            axes[0].text(
                0, i,
                f" {row['significant']}",
                va="center",
                fontsize=8
            )

        # T-stat chart
        tstat_colors = []
        for _, row in df.iterrows():
            if row["significant"] == "✗ No":
                tstat_colors.append("lightgrey")
            elif row["t_stat"] > 0:
                tstat_colors.append("steelblue")
            else:
                tstat_colors.append("tomato")

        axes[1].barh(
            df["factor"],
            df["t_stat"],
            color=tstat_colors,
            edgecolor="white"
        )
        axes[1].axvline(0,      color="black",  linewidth=0.8, linestyle="--")
        axes[1].axvline(1.96,   color="green",  linewidth=1.0,
                        linestyle="--", label="p=0.05 (±1.96)")
        axes[1].axvline(-1.96,  color="green",  linewidth=1.0, linestyle="--")
        axes[1].axvline(1.645,  color="orange", linewidth=1.0,
                        linestyle="--", label="p=0.10 (±1.645)")
        axes[1].axvline(-1.645, color="orange", linewidth=1.0, linestyle="--")
        axes[1].set_title("T-Statistics")
        axes[1].set_xlabel("T-Stat")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.show()

    # ── Step 7 — Get significant factors ─────────────────────────────────────

    def get_significant_factors(self, include_marginal: bool = False):
        """
        Returns list of factors with significant risk premiums.

        Args:
            include_marginal: whether to include marginal factors
        """
        if not self.fitted:
            raise ValueError("Run fit() first.")

        if include_marginal:
            mask = self.results["significant"].isin(["✓ Yes", "~ Marginal"])
        else:
            mask = self.results["significant"] == "✓ Yes"

        significant = self.results[mask]["factor"].tolist()

        print(f"\n── Significant Factors ─────────────────────────")
        print(
            f"  Frequency: {self.FREQ_LABELS[self.config.frequency]} | "
            f"Ready to use in screener or portfolio:\n"
        )
        for f in significant:
            row = self.results[self.results["factor"] == f].iloc[0]
            print(
                f"  ✓ {f:30} "
                f"premium: {row['risk_premium']:.4f}% "
                f"per {self.config.frequency} | "
                f"t-stat: {row['t_stat']:.3f}"
            )

        return significant

    # ── Master run method ─────────────────────────────────────────────────────

    def fit(self, plot: bool = True):
        """
        Runs full Fama MacBeth two stage regression.

        Args:
            plot: whether to show results chart
        """
        self._validate()

        print(f"\n── Running Fama MacBeth ────────────────────────")
        print(f"  Dependent Variable: {self.config.dependent_var}")
        print(f"  Frequency:          {self.FREQ_LABELS[self.config.frequency]}")
        print(f"  Features:           {self.config.feature_cols}")

        data     = self._prepare_data()
        betas_df = self._stage_1(data)

        self.results = self._stage_2(betas_df)
        self.fitted  = True

        self._print_results()

        if plot:
            self.plot()

        return self