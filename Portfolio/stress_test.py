from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Literal, Optional


RANDOM_STATE = 42
DistributionType = Literal["auto", "normal", "student_t", "skewed_t", "gev"]


class PortfolioStressTester:
    """
    Monte Carlo stress testing for portfolios.
    Separate from strategy stress tester since
    portfolio has additional decomposition needs.
    """

    def __init__(self, portfolio: object):
        """
        Args:
            portfolio: fitted Portfolio object
        """
        self.portfolio       = portfolio
        self.portfolio_df    = portfolio.portfolio_df.copy()
        self.returns         = (
            self.portfolio_df["total_value"]
            .pct_change()
            .dropna()
        )
        self.initial_capital = portfolio.config.initial_capital
        self.results         = {}

    # ── Distribution analysis ─────────────────────────────────────────────────

    def analyze_distribution(self):
        """Analyzes return distribution of portfolio."""
        kurtosis = self.returns.kurtosis()
        skewness = self.returns.skew()

        if abs(kurtosis) < 1 and abs(skewness) < 0.5:
            recommendation = "normal"
            reason = "Returns behave relatively normally"
        elif kurtosis < 6:
            recommendation = "student_t"
            reason = "Mild fat tails detected"
        elif kurtosis < 10:
            recommendation = "skewed_t"
            reason = "Fat tails with asymmetry detected"
        else:
            recommendation = "gev"
            reason = "Extreme fat tails detected"

        print(f"\n── Portfolio Return Distribution ───────────────")
        print(f"  Kurtosis:       {round(kurtosis, 4)}")
        print(f"  Skewness:       {round(skewness, 4)}")
        print(f"  Recommendation: {recommendation}")
        print(f"  Reason:         {reason}")

        return {
            "kurtosis":       round(kurtosis, 4),
            "skewness":       round(skewness, 4),
            "recommendation": recommendation,
            "reason":         reason,
        }

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def monte_carlo(self,
                    num_sims: int = 10000,
                    num_days: int = 252,
                    distribution: DistributionType = "auto"):
        """
        Monte Carlo simulation on portfolio returns.
        Auto selects distribution based on kurtosis.

        Args:
            num_sims:     number of simulations (100 to 100000)
            num_days:     days to simulate (21 to 1260)
            distribution: distribution type, "auto" recommended
        """
        # Validate
        if not (100 <= num_sims <= 100000):
            raise ValueError(f"num_sims must be between 100 and 100000. Got: {num_sims}")
        if not (21 <= num_days <= 1260):
            raise ValueError(f"num_days must be between 21 and 1260. Got: {num_days}")

        print(f"\n── Portfolio Monte Carlo ────────────────────────")

        np.random.seed(RANDOM_STATE)

        # Fit distribution
        if distribution == "auto":
            analysis     = self.analyze_distribution()
            distribution = analysis["recommendation"]

        if distribution == "normal":
            mu, sigma    = stats.norm.fit(self.returns)
            fitted_dist  = stats.norm(loc=mu, scale=sigma)
        elif distribution == "student_t":
            df, mu, sigma = stats.t.fit(self.returns)
            fitted_dist   = stats.t(df=df, loc=mu, scale=sigma)
        elif distribution == "skewed_t":
            a, df, mu, sigma = stats.nct.fit(self.returns)
            fitted_dist      = stats.nct(nc=a, df=df, loc=mu, scale=sigma)
        elif distribution == "gev":
            c, mu, sigma = stats.genextreme.fit(self.returns)
            fitted_dist  = stats.genextreme(c=c, loc=mu, scale=sigma)

        # Simulate
        sim_returns   = fitted_dist.rvs(
            size=(num_days, num_sims),
            random_state=RANDOM_STATE
        )
        paths         = self.initial_capital * np.exp(
            np.cumsum(sim_returns, axis=0)
        )
        ending_values = paths[-1, :]

        # ── Key metrics ───────────────────────────────────────────────────
        mean_val    = ending_values.mean()
        median_val  = np.percentile(ending_values, 50)
        std_val     = ending_values.std()

        prob_profit  = (ending_values > self.initial_capital).mean() * 100
        prob_double  = (ending_values > self.initial_capital * 2).mean() * 100
        prob_loss    = (ending_values < self.initial_capital).mean() * 100

        flat_returns = sim_returns.flatten()
        var_95  = np.percentile(flat_returns, 5)
        var_99  = np.percentile(flat_returns, 1)
        cvar_95 = flat_returns[flat_returns <= var_95].mean()
        cvar_99 = flat_returns[flat_returns <= var_99].mean()

        percentiles = {
            "p5":  np.percentile(ending_values, 5),
            "p25": np.percentile(ending_values, 25),
            "p50": median_val,
            "p75": np.percentile(ending_values, 75),
            "p95": np.percentile(ending_values, 95),
        }

        print(f"  Distribution:      {distribution}")
        print(f"  Simulations:       {num_sims:,}")
        print(f"  Days:              {num_days}")
        print(f"  Initial Capital:   ${self.initial_capital:,.2f}")
        print(f"\n── Outcome Metrics ─────────────────────────────")
        print(f"  Mean Value:        ${mean_val:,.2f}")
        print(f"  Median Value:      ${median_val:,.2f}")
        print(f"  Std Dev:           ${std_val:,.2f}")
        print(f"\n── Probabilities ───────────────────────────────")
        print(f"  Profit:            {prob_profit:.2f}%")
        print(f"  Doubling:          {prob_double:.2f}%")
        print(f"  Loss:              {prob_loss:.2f}%")
        print(f"\n── Tail Risk ───────────────────────────────────")
        print(f"  VaR  95%:          {round(var_95 * 100, 4)}%")
        print(f"  VaR  99%:          {round(var_99 * 100, 4)}%")
        print(f"  CVaR 95%:          {round(cvar_95 * 100, 4)}%")
        print(f"  CVaR 99%:          {round(cvar_99 * 100, 4)}%")

        self.results["monte_carlo"] = {
            "distribution":  distribution,
            "paths":         paths,
            "ending_values": ending_values,
            "percentiles":   percentiles,
            "mean":          mean_val,
            "median":        median_val,
            "std":           std_val,
            "prob_profit":   prob_profit,
            "prob_double":   prob_double,
            "prob_loss":     prob_loss,
            "var_95":        var_95,
            "var_99":        var_99,
            "cvar_95":       cvar_95,
            "cvar_99":       cvar_99,
        }

        return self.results["monte_carlo"]

    # ── Plot Monte Carlo ──────────────────────────────────────────────────────

    def plot_monte_carlo(self, n_paths: int = 200):
        """Plots Monte Carlo simulation paths and ending distribution."""
        if "monte_carlo" not in self.results:
            raise ValueError("Run monte_carlo() first.")

        r     = self.results["monte_carlo"]
        paths = r["paths"]
        percs = r["percentiles"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Paths
        for i in range(min(n_paths, paths.shape[1])):
            axes[0].plot(paths[:, i], alpha=0.05,
                         color="steelblue", linewidth=0.5)

        axes[0].axhline(percs["p50"], color="blue",  linewidth=1.5,
                        linestyle="--", label=f"Median: ${percs['p50']:,.0f}")
        axes[0].axhline(percs["p95"], color="green", linewidth=1.5,
                        linestyle="--", label=f"Best 5%: ${percs['p95']:,.0f}")
        axes[0].axhline(percs["p5"],  color="red",   linewidth=1.5,
                        linestyle="--", label=f"Worst 5%: ${percs['p5']:,.0f}")
        axes[0].axhline(self.initial_capital, color="black",
                        linewidth=1, label=f"Initial: ${self.initial_capital:,.0f}")
        axes[0].set_title(
            f"Portfolio Monte Carlo — {self.portfolio.config.name}\n"
            f"Distribution: {r['distribution']} | "
            f"Profit Prob: {r['prob_profit']:.1f}% | "
            f"Double Prob: {r['prob_double']:.1f}%"
        )
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Ending value distribution
        axes[1].hist(r["ending_values"], bins=100,
                     color="steelblue", edgecolor="white", alpha=0.7)
        axes[1].axvline(self.initial_capital, color="black",
                        linewidth=1.5, linestyle="--", label="Initial Capital")
        axes[1].axvline(percs["p5"],  color="red",   linewidth=1.5,
                        linestyle="--", label=f"Worst 5%: ${percs['p5']:,.0f}")
        axes[1].axvline(percs["p50"], color="blue",  linewidth=1.5,
                        linestyle="--", label=f"Median: ${percs['p50']:,.0f}")
        axes[1].axvline(percs["p95"], color="green", linewidth=1.5,
                        linestyle="--", label=f"Best 5%: ${percs['p95']:,.0f}")
        axes[1].set_title("Distribution of Ending Values")
        axes[1].set_xlabel("Portfolio Value ($)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()