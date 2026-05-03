from __future__ import annotations

import pandas as pd
from .stress_test import (
    StressTester,
    DistributionType,
    PARAM_LIMITS,
    RANDOM_STATE
)


class BacktestEngine:

    def __init__(self, strategy_results: dict):
        """
        Args:
            strategy_results: output from StrategyEngine.run_*() methods
                             contains metrics, portfolio, trades, signals
        """
        self.strategy_results = strategy_results
        self.tester           = StressTester(strategy_results)
        self.run_log          = []

    # ── Parameter info ───────────────────────────────────────────────────────

    def get_param_limits(self):
        """
        Returns parameter limits for all stress test methods.
        Frontend uses this to set input field constraints.
        """
        print("\n── Stress Test Parameter Limits ────────────────")
        for param, limits in PARAM_LIMITS.items():
            print(
                f"  {param:15} → "
                f"min: {limits['min']}, "
                f"max: {limits['max']}, "
                f"default: {limits['default']} "
                f"| {limits['description']}"
            )
        return PARAM_LIMITS

    # ── Distribution analysis ────────────────────────────────────────────────

    def analyze_distribution(self):
        """
        Analyzes return distribution before running stress tests.
        Recommended first step so you know which distribution fits.
        """
        from .stress_test import DistributionEngine
        returns  = self.tester.returns
        analysis = DistributionEngine.analyze(returns)

        print("\n── Return Distribution Analysis ────────────────")
        print(f"  Kurtosis:        {analysis['kurtosis']}")
        print(f"  Skewness:        {analysis['skewness']}")
        print(f"  Recommendation:  {analysis['recommendation']}")
        print(f"  Reason:          {analysis['reason']}")

        return analysis

    # ── Method 1 — In Sample ─────────────────────────────────────────────────

    def run_in_sample(self, plot: bool = True):
        """
        Baseline in sample performance.
        No parameters needed — uses full backtest period.

        Args:
            plot: whether to show equity curve
        """
        results = self.tester.in_sample()

        if plot:
            self.tester.portfolio_df["total_value"].plot(
                figsize=(12, 4),
                title="In Sample Equity Curve",
                ylabel="Portfolio Value ($)",
                grid=True
            )
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.show()

        self.run_log.append("in_sample")
        return results

    # ── Method 2 — In Sample / Out of Sample ─────────────────────────────────

    def run_in_out_sample(self,
                          split: float = 0.7,
                          split_date: str = None,
                          plot: bool = True):
        """
        In sample / out of sample split.

        Args:
            split:      fraction for training (0.1 to 0.9)
                        default 0.7
            split_date: optional specific split date "YYYY-MM-DD"
                        overrides split fraction if provided
            plot:       whether to show chart
        """
        results = self.tester.in_out_sample(split, split_date)

        if plot:
            self.tester.plot_in_out_sample()

        self.run_log.append("in_out_sample")
        return results

    # ── Method 3 — Walk Forward ──────────────────────────────────────────────

    def run_walk_forward(self,
                         train_pct: float = 0.6,
                         test_pct: float = 0.2,
                         n_splits: int = 5,
                         plot: bool = True):
        """
        Walk forward cross validation.

        Args:
            train_pct: fraction per training window (0.1 to 0.9)
            test_pct:  fraction per test window (0.1 to 0.9)
                       train_pct + test_pct must be less than 1.0
            n_splits:  number of folds (2 to 10)
            plot:      whether to show chart
        """
        results = self.tester.walk_forward(train_pct, test_pct, n_splits)

        if plot:
            self.tester.plot_walk_forward()

        self.run_log.append("walk_forward")
        return results

    # ── Method 4 — Monte Carlo Resample ──────────────────────────────────────

    def run_monte_carlo_resample(self,
                                  num_sims: int = 10000,
                                  num_days: int = 252,
                                  plot: bool = True):
        """
        Monte Carlo return resampling.
        Randomly shuffles actual returns to show range of outcomes.

        Args:
            num_sims: number of simulations (100 to 100000)
            num_days: days to simulate (21 to 1260)
            plot:     whether to show chart
        """
        results = self.tester.monte_carlo_resample(num_sims, num_days)

        if plot:
            self.tester.plot_monte_carlo("resample")

        self.run_log.append("monte_carlo_resample")
        return results

    # ── Method 5 — Monte Carlo Distribution ──────────────────────────────────

    def run_monte_carlo_distribution(self,
                                      distribution: DistributionType = "auto",
                                      num_sims: int = 10000,
                                      num_days: int = 252,
                                      plot: bool = True):
        """
        Monte Carlo distribution based simulation.
        Fits statistical distribution to returns and simulates paths.

        Args:
            distribution: "auto", "normal", "student_t", "skewed_t", "gev"
                          "auto" selects based on kurtosis
            num_sims:     number of simulations (100 to 100000)
            num_days:     days to simulate (21 to 1260)
            plot:         whether to show chart
        """
        results = self.tester.monte_carlo_distribution(
            distribution, num_sims, num_days
        )

        if plot:
            self.tester.plot_monte_carlo("distribution")

        self.run_log.append("monte_carlo_distribution")
        return results

    # ── Method 6 — Random Entry ───────────────────────────────────────────────

    def run_random_entry(self,
                         num_sims: int = 1000,
                         plot: bool = True):
        """
        Random entry Monte Carlo.
        Tests if strategy has genuine skill vs random luck.

        Args:
            num_sims: number of random strategies (100 to 100000)
            plot:     whether to show chart
        """
        results = self.tester.monte_carlo_random_entry(num_sims)

        if plot:
            self.tester.plot_random_entry()

        self.run_log.append("random_entry")
        return results

    # ── Run all stress tests ─────────────────────────────────────────────────

    def run_all(self,
                split: float = 0.7,
                train_pct: float = 0.6,
                test_pct: float = 0.2,
                n_splits: int = 5,
                num_sims: int = 10000,
                num_days: int = 252,
                distribution: DistributionType = "auto",
                plot: bool = True):
        """
        Runs all stress tests in sequence.
        Good for comprehensive strategy evaluation.

        Args:
            split:        in/out of sample split fraction (0.1 to 0.9)
            train_pct:    walk forward train fraction (0.1 to 0.9)
            test_pct:     walk forward test fraction (0.1 to 0.9)
            n_splits:     walk forward folds (2 to 10)
            num_sims:     Monte Carlo simulations (100 to 100000)
            num_days:     Monte Carlo days (21 to 1260)
            distribution: Monte Carlo distribution type
            plot:         whether to show charts
        """
        print("\n" + "=" * 50)
        print("FULL STRESS TEST SUITE")
        print("=" * 50)

        # Step 1 — distribution analysis first
        self.analyze_distribution()

        # Step 2 — run all methods
        self.run_in_sample(plot=plot)
        self.run_in_out_sample(split=split, plot=plot)
        self.run_walk_forward(
            train_pct=train_pct,
            test_pct=test_pct,
            n_splits=n_splits,
            plot=plot
        )
        self.run_monte_carlo_resample(
            num_sims=num_sims,
            num_days=num_days,
            plot=plot
        )
        self.run_monte_carlo_distribution(
            distribution=distribution,
            num_sims=num_sims,
            num_days=num_days,
            plot=plot
        )
        self.run_random_entry(
            num_sims=num_sims,
            plot=plot
        )

        print("\n" + "=" * 50)
        print("STRESS TEST COMPLETE")
        print(f"Methods run: {self.run_log}")
        print("=" * 50)

        return self.tester.stress_results

    # ── Get results ──────────────────────────────────────────────────────────

    def get_results(self, method: str = None):
        """
        Returns stress test results.

        Args:
            method: optional specific method name
                    e.g. "walk_forward", "monte_carlo_resample"
                    if None returns all results
        """
        if method is not None:
            if method not in self.tester.stress_results:
                available = list(self.tester.stress_results.keys())
                raise ValueError(
                    f"Method '{method}' not run yet. "
                    f"Available results: {available}"
                )
            return self.tester.stress_results[method]

        return self.tester.stress_results

    def get_log(self):
        """Returns list of stress tests that have been run."""
        print(f"\nStress tests completed: {self.run_log}")
        return self.run_log

    def get_summary(self):
        """
        Returns a summary table of key metrics across all
        stress tests that have been run.
        """
        if not self.tester.stress_results:
            print("No stress tests run yet.")
            return pd.DataFrame()

        rows = []

        if "in_sample" in self.tester.stress_results:
            m = self.tester.stress_results["in_sample"]["metrics"]
            rows.append({
                "method":           "In Sample",
                "total_return":     m.get("total_return"),
                "sharpe_ratio":     m.get("sharpe_ratio"),
                "max_drawdown":     m.get("max_drawdown"),
                "var_95":           m.get("var_95"),
                "cvar_95":          m.get("cvar_95"),
            })

        if "in_out_sample" in self.tester.stress_results:
            r = self.tester.stress_results["in_out_sample"]
            rows.append({
                "method":       "In Sample Split",
                "total_return": r["in_sample"]["metrics"].get("total_return"),
                "sharpe_ratio": r["in_sample"]["metrics"].get("sharpe_ratio"),
                "max_drawdown": r["in_sample"]["metrics"].get("max_drawdown"),
                "var_95":       r["in_sample"]["metrics"].get("var_95"),
                "cvar_95":      r["in_sample"]["metrics"].get("cvar_95"),
            })
            rows.append({
                "method":       "Out of Sample Split",
                "total_return": r["out_sample"]["metrics"].get("total_return"),
                "sharpe_ratio": r["out_sample"]["metrics"].get("sharpe_ratio"),
                "max_drawdown": r["out_sample"]["metrics"].get("max_drawdown"),
                "var_95":       r["out_sample"]["metrics"].get("var_95"),
                "cvar_95":      r["out_sample"]["metrics"].get("cvar_95"),
            })

        if "walk_forward" in self.tester.stress_results:
            r = self.tester.stress_results["walk_forward"]
            rows.append({
                "method":       "Walk Forward (avg test)",
                "total_return": round(r["avg_test_return"], 2),
                "sharpe_ratio": round(r["avg_test_sharpe"], 3),
                "max_drawdown": None,
                "var_95":       None,
                "cvar_95":      None,
            })

        if "monte_carlo_resample" in self.tester.stress_results:
            r = self.tester.stress_results["monte_carlo_resample"]
            rows.append({
                "method":       "MC Resample (median)",
                "total_return": round((r["percentiles"]["p50"] - self.tester.initial_capital) / self.tester.initial_capital * 100, 2),
                "sharpe_ratio": None,
                "max_drawdown": None,
                "var_95":       None,
                "cvar_95":      None,
            })

        if "monte_carlo_distribution" in self.tester.stress_results:
            r = self.tester.stress_results["monte_carlo_distribution"]
            rows.append({
                "method":       f"MC Distribution ({r['distribution']})",
                "total_return": round((r["percentiles"]["p50"] - self.tester.initial_capital) / self.tester.initial_capital * 100, 2),
                "sharpe_ratio": None,
                "max_drawdown": None,
                "var_95":       round(r["var_95"] * 100, 2),
                "cvar_95":      round(r["cvar_95"] * 100, 2),
            })

        if "random_entry" in self.tester.stress_results:
            r = self.tester.stress_results["random_entry"]
            rows.append({
                "method":       "Random Entry",
                "total_return": round(r["actual_return"], 2),
                "sharpe_ratio": round(r["actual_sharpe"], 3),
                "max_drawdown": None,
                "var_95":       None,
                "cvar_95":      None,
            })

        summary_df = pd.DataFrame(rows)

        print("\n── Stress Test Summary ─────────────────────────")
        print(summary_df.to_string(index=False))

        return summary_df