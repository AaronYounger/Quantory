from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import Literal

DistributionType = Literal["auto", "normal", "student_t", "skewed_t", "gev"]

# ── Fixed random state ───────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Parameter limits ─────────────────────────────────────────────────────────
PARAM_LIMITS = {
    "split": {
        "min":     0.1,
        "max":     0.9,
        "default": 0.7,
        "type":    float,
        "description": "Fraction of data for in/out of sample split"
    },
    "n_splits": {
        "min":     2,
        "max":     10,
        "default": 5,
        "type":    int,
        "description": "Number of walk forward folds"
    },
    "num_sims": {
        "min":     100,
        "max":     100000,
        "default": 10000,
        "type":    int,
        "description": "Number of Monte Carlo simulations"
    },
    "num_days": {
        "min":     21,
        "max":     1260,
        "default": 252,
        "type":    int,
        "description": "Number of days to simulate forward"
    },
    "train_pct": {
        "min":     0.1,
        "max":     0.9,
        "default": 0.6,
        "type":    float,
        "description": "Fraction of data per training window"
    },
    "test_pct": {
        "min":     0.1,
        "max":     0.9,
        "default": 0.2,
        "type":    float,
        "description": "Fraction of data per test window"
    },
}


# ── Distribution Engine ──────────────────────────────────────────────────────

class DistributionEngine:
    """
    Automatically selects and fits the best distribution
    to a return series based on statistical properties.
    """

    @staticmethod
    def analyze(returns: pd.Series) -> dict:
        """
        Analyzes statistical properties of return series.
        Returns kurtosis, skewness and recommended distribution.
        """
        kurtosis = returns.kurtosis()
        skewness = returns.skew()

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
            reason = "Extreme fat tails detected — crash like behavior"

        return {
            "kurtosis":       round(kurtosis, 4),
            "skewness":       round(skewness, 4),
            "recommendation": recommendation,
            "reason":         reason,
        }

    @staticmethod
    def fit(returns: pd.Series, distribution: DistributionType = "auto"):
        """
        Fits selected distribution to return series.
        Returns fitted distribution object ready for sampling.
        """
        if distribution == "auto":
            analysis     = DistributionEngine.analyze(returns)
            distribution = analysis["recommendation"]
            print(f"Auto selected: {distribution} — {analysis['reason']}")
            print(f"Kurtosis: {analysis['kurtosis']} Skewness: {analysis['skewness']}")

        if distribution == "normal":
            mu, sigma = stats.norm.fit(returns)
            return stats.norm(loc=mu, scale=sigma), distribution

        elif distribution == "student_t":
            df, mu, sigma = stats.t.fit(returns)
            return stats.t(df=df, loc=mu, scale=sigma), distribution

        elif distribution == "skewed_t":
            a, df, mu, sigma = stats.nct.fit(returns)
            return stats.nct(nc=a, df=df, loc=mu, scale=sigma), distribution

        elif distribution == "gev":
            c, mu, sigma = stats.genextreme.fit(returns)
            return stats.genextreme(c=c, loc=mu, scale=sigma), distribution

        else:
            raise ValueError(
                f"Invalid distribution '{distribution}'. "
                f"Valid options: auto, normal, student_t, skewed_t, gev"
            )


# ── Stress Tester ────────────────────────────────────────────────────────────

class StressTester:

    def __init__(self, strategy_results: dict):
        """
        Args:
            strategy_results: output from any strategy .run() method
        """
        self.strategy_results = strategy_results
        self.portfolio_df     = strategy_results["portfolio"].copy()
        self.metrics          = strategy_results["metrics"]
        self.returns          = self.portfolio_df["total_value"].pct_change().dropna()
        self.initial_capital  = self.metrics["initial_capital"]
        self.stress_results   = {}

    # ── Parameter validation ─────────────────────────────────────────────────

    def _validate_param(self, name: str, value):
        """Validates a single parameter against its limits."""
        if name not in PARAM_LIMITS:
            return value

        limits  = PARAM_LIMITS[name]
        min_val = limits["min"]
        max_val = limits["max"]
        typ     = limits["type"]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Parameter '{name}' must be a number. "
                f"Got: {type(value).__name__}"
            )

        value = typ(value)

        if value < min_val or value > max_val:
            raise ValueError(
                f"Parameter '{name}' must be between "
                f"{min_val} and {max_val}. "
                f"Got: {value}"
            )

        return value

    def _validate_walk_forward_params(self, train_pct: float, 
                                       test_pct: float, 
                                       n_splits: int):
        """
        Validates walk forward specific parameters.
        Train + test cannot sum to 1.0 or more.
        """
        if train_pct + test_pct >= 1.0:
            raise ValueError(
                f"train_pct ({train_pct}) + test_pct ({test_pct}) "
                f"must be less than 1.0. "
                f"Current sum: {round(train_pct + test_pct, 2)}. "
                f"Reduce one or both values."
            )

        # Check enough data exists for n_splits
        n            = len(self.portfolio_df)
        train_size   = int(n * train_pct)
        test_size    = int(n * test_pct)
        step_size    = int((n - train_size) / max(n_splits, 1))

        if step_size < 1:
            raise ValueError(
                f"Not enough data for {n_splits} splits. "
                f"Reduce n_splits or adjust train_pct/test_pct."
            )

    # ── Helper — calculate metrics ───────────────────────────────────────────

    def _calculate_metrics(self, portfolio_values: pd.Series) -> dict:
        """Calculates performance metrics from any equity curve."""
        returns = portfolio_values.pct_change().dropna()

        if len(returns) < 2:
            return {}

        total_return      = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        n_days            = len(portfolio_values)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        annualized_vol    = returns.std() * np.sqrt(252)
        sharpe            = annualized_return / annualized_vol if annualized_vol != 0 else 0

        downside     = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252)
        sortino      = annualized_return / downside_std if downside_std != 0 else 0

        rolling_max  = portfolio_values.expanding().max()
        drawdown     = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        var_95  = np.percentile(returns, 5)
        var_99  = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        return {
            "total_return":      round(total_return * 100, 2),
            "annualized_return": round(annualized_return * 100, 2),
            "annualized_vol":    round(annualized_vol * 100, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "sortino_ratio":     round(sortino, 3),
            "max_drawdown":      round(max_drawdown * 100, 2),
            "var_95":            round(var_95 * 100, 2),
            "var_99":            round(var_99 * 100, 2),
            "cvar_95":           round(cvar_95 * 100, 2),
            "cvar_99":           round(cvar_99 * 100, 2),
        }

    # ── Method 1 — In Sample Only ────────────────────────────────────────────

    def in_sample(self):
        """Baseline — full backtest period performance."""
        print("\n── In Sample Analysis ──────────────────────────")

        pv      = self.portfolio_df["total_value"]
        metrics = self._calculate_metrics(pv)

        print(f"  Period:            Full backtest")
        print(f"  Total Return:      {metrics['total_return']}%")
        print(f"  Annualized Return: {metrics['annualized_return']}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']}%")
        print(f"  VaR  95%:          {metrics['var_95']}%")
        print(f"  VaR  99%:          {metrics['var_99']}%")
        print(f"  CVaR 95%:          {metrics['cvar_95']}%")
        print(f"  CVaR 99%:          {metrics['cvar_99']}%")

        self.stress_results["in_sample"] = {
            "metrics":   metrics,
            "portfolio": pv,
        }

        return self.stress_results["in_sample"]

    # ── Method 2 — In Sample / Out of Sample Split ───────────────────────────

    def in_out_sample(self,
                      split: float = 0.7,
                      split_date: str = None):
        """
        Splits backtest into train and test periods.

        Args:
            split:      fraction for training (0.1 to 0.9)
            split_date: optional specific date "YYYY-MM-DD"
                        overrides split fraction if provided
        """
        # Validate
        split = self._validate_param("split", split)

        print("\n── In Sample / Out of Sample Split ────────────")

        pv    = self.portfolio_df["total_value"].reset_index(drop=True)
        dates = self.portfolio_df["date"].reset_index(drop=True)

        if split_date is not None:
            split_idx = int((dates <= pd.to_datetime(split_date)).sum())
        else:
            split_idx = int(len(pv) * split)

        if split_idx < 2 or split_idx >= len(pv) - 2:
            raise ValueError(
                f"Split results in too few data points on one side. "
                f"Adjust split value or split_date."
            )

        in_sample_pv     = pv.iloc[:split_idx]
        out_sample_pv    = pv.iloc[split_idx:].reset_index(drop=True)
        in_sample_dates  = dates.iloc[:split_idx]
        out_sample_dates = dates.iloc[split_idx:].reset_index(drop=True)

        in_metrics  = self._calculate_metrics(in_sample_pv)
        out_metrics = self._calculate_metrics(out_sample_pv)

        print(f"\n  In Sample ({round(split*100)}% of data):")
        print(f"    Total Return:      {in_metrics['total_return']}%")
        print(f"    Sharpe Ratio:      {in_metrics['sharpe_ratio']}")
        print(f"    Max Drawdown:      {in_metrics['max_drawdown']}%")
        print(f"    VaR  95%:          {in_metrics['var_95']}%")
        print(f"    CVaR 95%:          {in_metrics['cvar_95']}%")

        print(f"\n  Out of Sample ({round((1-split)*100)}% of data):")
        print(f"    Total Return:      {out_metrics['total_return']}%")
        print(f"    Sharpe Ratio:      {out_metrics['sharpe_ratio']}")
        print(f"    Max Drawdown:      {out_metrics['max_drawdown']}%")
        print(f"    VaR  95%:          {out_metrics['var_95']}%")
        print(f"    CVaR 95%:          {out_metrics['cvar_95']}%")

        return_deg = in_metrics["annualized_return"] - out_metrics["annualized_return"]
        sharpe_deg = in_metrics["sharpe_ratio"] - out_metrics["sharpe_ratio"]

        print(f"\n  Performance Degradation:")
        print(f"    Return drop:  {round(return_deg, 2)}%")
        print(f"    Sharpe drop:  {round(sharpe_deg, 3)}")

        if sharpe_deg > 0.5:
            print(f"  ⚠ Warning: Significant performance drop out of sample.")
            print(f"    Strategy may be overfit to historical data.")
        else:
            print(f"  ✓ Strategy performance relatively stable out of sample.")

        self.stress_results["in_out_sample"] = {
            "in_sample":  {
                "metrics":   in_metrics,
                "portfolio": in_sample_pv,
                "dates":     in_sample_dates
            },
            "out_sample": {
                "metrics":   out_metrics,
                "portfolio": out_sample_pv,
                "dates":     out_sample_dates
            },
            "split_idx":   split_idx,
            "degradation": {
                "return": return_deg,
                "sharpe": sharpe_deg
            }
        }

        return self.stress_results["in_out_sample"]

    # ── Method 3 — Walk Forward Cross Validation ─────────────────────────────

    def walk_forward(self,
                     train_pct: float = 0.6,
                     test_pct: float = 0.2,
                     n_splits: int = 5):
        """
        Multiple rolling train/test windows across full period.

        Args:
            train_pct: fraction per training window (0.1 to 0.9)
            test_pct:  fraction per test window (0.1 to 0.9)
                       train_pct + test_pct must be less than 1.0
            n_splits:  number of folds (2 to 10)
        """
        # Validate
        train_pct = self._validate_param("train_pct", train_pct)
        test_pct  = self._validate_param("test_pct",  test_pct)
        n_splits  = self._validate_param("n_splits",  n_splits)
        self._validate_walk_forward_params(train_pct, test_pct, n_splits)

        print("\n── Walk Forward Cross Validation ───────────────")
        print(f"  Train: {round(train_pct*100)}% | Test: {round(test_pct*100)}% | Folds: {n_splits}")

        pv    = self.portfolio_df["total_value"].reset_index(drop=True)
        dates = self.portfolio_df["date"].reset_index(drop=True)
        n     = len(pv)

        train_size = int(n * train_pct)
        test_size  = int(n * test_pct)
        step_size  = int((n - train_size) / n_splits)

        fold_results = []

        for i in range(n_splits):
            train_start = i * step_size
            train_end   = train_start + train_size
            test_start  = train_end
            test_end    = min(test_start + test_size, n)

            if test_end > n or test_start >= n:
                break

            train_pv = pv.iloc[train_start:train_end].reset_index(drop=True)
            test_pv  = pv.iloc[test_start:test_end].reset_index(drop=True)

            train_metrics = self._calculate_metrics(train_pv)
            test_metrics  = self._calculate_metrics(test_pv)

            fold_results.append({
                "fold":             i + 1,
                "train_start_date": dates.iloc[train_start],
                "train_end_date":   dates.iloc[train_end - 1],
                "test_start_date":  dates.iloc[test_start],
                "test_end_date":    dates.iloc[test_end - 1],
                "train_return":     train_metrics.get("total_return", 0),
                "test_return":      test_metrics.get("total_return", 0),
                "train_sharpe":     train_metrics.get("sharpe_ratio", 0),
                "test_sharpe":      test_metrics.get("sharpe_ratio", 0),
                "train_drawdown":   train_metrics.get("max_drawdown", 0),
                "test_drawdown":    test_metrics.get("max_drawdown", 0),
                "train_var_95":     train_metrics.get("var_95", 0),
                "test_var_95":      test_metrics.get("var_95", 0),
            })

            print(f"\n  Fold {i+1}:")
            print(f"    Train: {dates.iloc[train_start].date()} → {dates.iloc[train_end-1].date()}")
            print(f"    Test:  {dates.iloc[test_start].date()} → {dates.iloc[test_end-1].date()}")
            print(f"    Train Sharpe: {train_metrics.get('sharpe_ratio',0)} | Test Sharpe: {test_metrics.get('sharpe_ratio',0)}")
            print(f"    Train Return: {train_metrics.get('total_return',0)}% | Test Return: {test_metrics.get('total_return',0)}%")
            print(f"    Train VaR 95%: {train_metrics.get('var_95',0)}% | Test VaR 95%: {test_metrics.get('var_95',0)}%")

        fold_df = pd.DataFrame(fold_results)

        avg_test_sharpe  = fold_df["test_sharpe"].mean()
        avg_test_return  = fold_df["test_return"].mean()
        sharpe_stability = fold_df["test_sharpe"].std()

        print(f"\n  Walk Forward Summary:")
        print(f"    Avg Test Sharpe:     {round(avg_test_sharpe, 3)}")
        print(f"    Avg Test Return:     {round(avg_test_return, 2)}%")
        print(f"    Sharpe Stability:    {round(sharpe_stability, 3)}")
        print(f"    (lower stability = more consistent across periods)")

        if sharpe_stability > 0.5:
            print(f"  ⚠ Warning: High variance across folds — strategy may be unstable.")
        else:
            print(f"  ✓ Strategy relatively consistent across walk forward folds.")

        self.stress_results["walk_forward"] = {
            "folds":            fold_df,
            "avg_test_sharpe":  avg_test_sharpe,
            "avg_test_return":  avg_test_return,
            "sharpe_stability": sharpe_stability,
        }

        return self.stress_results["walk_forward"]

    # ── Method 4 — Monte Carlo Return Resampling ─────────────────────────────

    def monte_carlo_resample(self,
                             num_sims: int = 10000,
                             num_days: int = 252):
        """
        Randomly shuffles actual strategy returns thousands of times.

        Args:
            num_sims: number of simulations (100 to 100000)
            num_days: days to simulate (21 to 1260)
        """
        # Validate
        num_sims = self._validate_param("num_sims", num_sims)
        num_days = self._validate_param("num_days", num_days)

        print("\n── Monte Carlo Return Resampling ───────────────")

        np.random.seed(RANDOM_STATE)
        returns = self.returns.values

        paths = np.zeros((num_days, num_sims))

        for i in range(num_sims):
            sampled      = np.random.choice(returns, size=num_days, replace=True)
            paths[:, i]  = self.initial_capital * np.cumprod(1 + sampled)

        ending_values = paths[-1, :]

        percentiles = {
            "p5":  np.percentile(ending_values, 5),
            "p25": np.percentile(ending_values, 25),
            "p50": np.percentile(ending_values, 50),
            "p75": np.percentile(ending_values, 75),
            "p95": np.percentile(ending_values, 95),
        }

        prob_profit = (ending_values > self.initial_capital).mean() * 100
        prob_loss   = (ending_values < self.initial_capital).mean() * 100
        expected    = ending_values.mean()
        worst_5pct  = percentiles["p5"]
        best_5pct   = percentiles["p95"]

        print(f"  Simulations:       {num_sims:,}")
        print(f"  Days:              {num_days}")
        print(f"  Initial Capital:   ${self.initial_capital:,.2f}")
        print(f"\n  Outcome Distribution:")
        print(f"    Expected Value:  ${expected:,.2f}")
        print(f"    Median (p50):    ${percentiles['p50']:,.2f}")
        print(f"    Best 5%  (p95):  ${best_5pct:,.2f}")
        print(f"    Worst 5% (p5):   ${worst_5pct:,.2f}")
        print(f"\n  Probabilities:")
        print(f"    Profit:          {prob_profit:.1f}%")
        print(f"    Loss:            {prob_loss:.1f}%")

        self.stress_results["monte_carlo_resample"] = {
            "paths":         paths,
            "ending_values": ending_values,
            "percentiles":   percentiles,
            "prob_profit":   prob_profit,
            "prob_loss":     prob_loss,
            "expected":      expected,
        }

        return self.stress_results["monte_carlo_resample"]

    # ── Method 5 — Monte Carlo Distribution Based ────────────────────────────

    def monte_carlo_distribution(self,
                                  distribution: DistributionType = "auto",
                                  num_sims: int = 10000,
                                  num_days: int = 252):
        """
        Fits a statistical distribution to returns and simulates paths.

        Args:
            distribution: "auto", "normal", "student_t", "skewed_t", "gev"
            num_sims:     number of simulations (100 to 100000)
            num_days:     days to simulate (21 to 1260)
        """
        # Validate
        num_sims = self._validate_param("num_sims", num_sims)
        num_days = self._validate_param("num_days", num_days)

        print("\n── Monte Carlo Distribution Based ──────────────")

        np.random.seed(RANDOM_STATE)

        fitted_dist, dist_name = DistributionEngine.fit(
            self.returns, distribution
        )

        sim_returns = fitted_dist.rvs(
            size=(num_days, num_sims),
            random_state=RANDOM_STATE
        )

        paths = self.initial_capital * np.exp(
            np.cumsum(sim_returns, axis=0)
        )

        ending_values = paths[-1, :]

        percentiles = {
            "p5":  np.percentile(ending_values, 5),
            "p25": np.percentile(ending_values, 25),
            "p50": np.percentile(ending_values, 50),
            "p75": np.percentile(ending_values, 75),
            "p95": np.percentile(ending_values, 95),
        }

        prob_profit = (ending_values > self.initial_capital).mean() * 100
        prob_loss   = (ending_values < self.initial_capital).mean() * 100
        expected    = ending_values.mean()

        sim_returns_flat = sim_returns.flatten()
        var_95  = np.percentile(sim_returns_flat, 5)
        var_99  = np.percentile(sim_returns_flat, 1)
        cvar_95 = sim_returns_flat[sim_returns_flat <= var_95].mean()
        cvar_99 = sim_returns_flat[sim_returns_flat <= var_99].mean()

        print(f"  Distribution:      {dist_name}")
        print(f"  Simulations:       {num_sims:,}")
        print(f"  Days:              {num_days}")
        print(f"  Initial Capital:   ${self.initial_capital:,.2f}")
        print(f"\n  Outcome Distribution:")
        print(f"    Expected Value:  ${expected:,.2f}")
        print(f"    Median (p50):    ${percentiles['p50']:,.2f}")
        print(f"    Best 5%  (p95):  ${percentiles['p95']:,.2f}")
        print(f"    Worst 5% (p5):   ${percentiles['p5']:,.2f}")
        print(f"\n  Probabilities:")
        print(f"    Profit:          {prob_profit:.1f}%")
        print(f"    Loss:            {prob_loss:.1f}%")
        print(f"\n  Tail Risk:")
        print(f"    VaR  95%:        {round(var_95 * 100, 2)}%")
        print(f"    VaR  99%:        {round(var_99 * 100, 2)}%")
        print(f"    CVaR 95%:        {round(cvar_95 * 100, 2)}%")
        print(f"    CVaR 99%:        {round(cvar_99 * 100, 2)}%")

        self.stress_results["monte_carlo_distribution"] = {
            "distribution":  dist_name,
            "paths":         paths,
            "ending_values": ending_values,
            "percentiles":   percentiles,
            "prob_profit":   prob_profit,
            "prob_loss":     prob_loss,
            "expected":      expected,
            "var_95":        var_95,
            "var_99":        var_99,
            "cvar_95":       cvar_95,
            "cvar_99":       cvar_99,
        }

        return self.stress_results["monte_carlo_distribution"]

    # ── Method 6 — Random Entry Monte Carlo ──────────────────────────────────

    def monte_carlo_random_entry(self, num_sims: int = 1000):
        """
        Generates random entry/exit strategies and compares to actual.
        Tests if strategy has genuine skill vs random luck.

        Args:
            num_sims: number of random strategies (100 to 100000)
        """
        # Validate
        num_sims = self._validate_param("num_sims", num_sims)

        print("\n── Random Entry Monte Carlo ────────────────────")

        np.random.seed(RANDOM_STATE)

        returns        = self.returns.values
        n              = len(returns)
        actual_sharpe  = self.metrics.get("sharpe_ratio", 0)
        actual_return  = self.metrics.get("total_return", 0)

        random_sharpes = []
        random_returns = []

        for i in range(num_sims):
            in_position = False
            cash        = self.initial_capital
            shares      = 0.0
            pv_random   = [self.initial_capital]

            for j, ret in enumerate(returns):
                price = self.initial_capital * np.cumprod(1 + returns[:j+1])[-1]

                if not in_position and np.random.random() > 0.95:
                    shares      = cash / price
                    cash        = 0
                    in_position = True
                elif in_position and np.random.random() > 0.95:
                    cash        = shares * price
                    shares      = 0
                    in_position = False

                total = cash + shares * price
                pv_random.append(total)

            pv_series  = pd.Series(pv_random)
            r          = pv_series.pct_change().dropna()
            total_ret  = (pv_series.iloc[-1] - self.initial_capital) / self.initial_capital
            ann_ret    = (1 + total_ret) ** (252 / len(pv_series)) - 1
            ann_vol    = r.std() * np.sqrt(252)
            sharpe     = ann_ret / ann_vol if ann_vol != 0 else 0

            random_sharpes.append(sharpe)
            random_returns.append(total_ret * 100)

        random_sharpes  = np.array(random_sharpes)
        random_returns  = np.array(random_returns)
        sharpe_pct_rank = (random_sharpes < actual_sharpe).mean() * 100
        return_pct_rank = (random_returns < actual_return).mean() * 100

        print(f"  Simulations:          {num_sims:,}")
        print(f"\n  Actual Strategy:")
        print(f"    Sharpe Ratio:       {actual_sharpe}")
        print(f"    Total Return:       {actual_return}%")
        print(f"\n  vs Random Strategies:")
        print(f"    Sharpe Percentile:  {round(sharpe_pct_rank, 1)}%")
        print(f"    Return Percentile:  {round(return_pct_rank, 1)}%")
        print(f"\n  Random Strategy Stats:")
        print(f"    Avg Random Sharpe:  {round(random_sharpes.mean(), 3)}")
        print(f"    Avg Random Return:  {round(random_returns.mean(), 2)}%")

        if sharpe_pct_rank > 95:
            print(f"\n  ✓ Strong skill signal — beats {round(sharpe_pct_rank,1)}% of random strategies")
        elif sharpe_pct_rank > 75:
            print(f"\n  ~ Moderate skill signal — beats {round(sharpe_pct_rank,1)}% of random strategies")
        else:
            print(f"\n  ⚠ Weak skill signal — only beats {round(sharpe_pct_rank,1)}% of random strategies")

        self.stress_results["random_entry"] = {
            "random_sharpes":  random_sharpes,
            "random_returns":  random_returns,
            "actual_sharpe":   actual_sharpe,
            "actual_return":   actual_return,
            "sharpe_pct_rank": sharpe_pct_rank,
            "return_pct_rank": return_pct_rank,
        }

        return self.stress_results["random_entry"]

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_monte_carlo(self, method: str = "resample", n_paths: int = 200):
        """
        Plots Monte Carlo simulation paths and ending value distribution.

        Args:
            method:  "resample" or "distribution"
            n_paths: number of paths to display
        """
        key = f"monte_carlo_{method}"
        if key not in self.stress_results:
            raise ValueError(f"Run monte_carlo_{method}() first.")

        results = self.stress_results[key]
        paths   = results["paths"]
        percs   = results["percentiles"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        for i in range(min(n_paths, paths.shape[1])):
            axes[0].plot(
                paths[:, i],
                alpha=0.05,
                color="steelblue",
                linewidth=0.5
            )

        axes[0].axhline(percs["p50"], color="blue",  linewidth=1.5, linestyle="--", label=f"Median: ${percs['p50']:,.0f}")
        axes[0].axhline(percs["p95"], color="green", linewidth=1.5, linestyle="--", label=f"Best 5%: ${percs['p95']:,.0f}")
        axes[0].axhline(percs["p5"],  color="red",   linewidth=1.5, linestyle="--", label=f"Worst 5%: ${percs['p5']:,.0f}")
        axes[0].axhline(self.initial_capital, color="black", linewidth=1, linestyle="-", label=f"Initial: ${self.initial_capital:,.0f}")
        axes[0].set_title(f"Monte Carlo Simulation ({method})")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(results["ending_values"], bins=100, color="steelblue", edgecolor="white", alpha=0.7)
        axes[1].axvline(self.initial_capital, color="black", linewidth=1.5, linestyle="--", label="Initial Capital")
        axes[1].axvline(percs["p5"],  color="red",   linewidth=1.5, linestyle="--", label=f"Worst 5%: ${percs['p5']:,.0f}")
        axes[1].axvline(percs["p95"], color="green", linewidth=1.5, linestyle="--", label=f"Best 5%: ${percs['p95']:,.0f}")
        axes[1].set_title("Distribution of Ending Values")
        axes[1].set_xlabel("Portfolio Value ($)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_walk_forward(self):
        """Plots walk forward Sharpe and returns across folds."""
        if "walk_forward" not in self.stress_results:
            raise ValueError("Run walk_forward() first.")

        fold_df = self.stress_results["walk_forward"]["folds"]
        x       = fold_df["fold"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(x, fold_df["train_sharpe"], marker="o", label="Train Sharpe", color="steelblue")
        axes[0].plot(x, fold_df["test_sharpe"],  marker="o", label="Test Sharpe",  color="orange")
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Walk Forward — Sharpe Ratio per Fold")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(x - 0.2, fold_df["train_return"], 0.4, label="Train Return", color="steelblue", alpha=0.7)
        axes[1].bar(x + 0.2, fold_df["test_return"],  0.4, label="Test Return",  color="orange",    alpha=0.7)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Walk Forward — Return per Fold")
        axes[1].set_ylabel("Total Return (%)")
        axes[1].set_xlabel("Fold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_random_entry(self):
        """Plots random entry distribution vs actual strategy."""
        if "random_entry" not in self.stress_results:
            raise ValueError("Run monte_carlo_random_entry() first.")

        results = self.stress_results["random_entry"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(results["random_sharpes"], bins=50, color="steelblue", edgecolor="white", alpha=0.7, label="Random Strategies")
        axes[0].axvline(results["actual_sharpe"], color="red", linewidth=2, label=f"Actual: {results['actual_sharpe']}")
        axes[0].set_title("Sharpe Ratio vs Random Strategies")
        axes[0].set_xlabel("Sharpe Ratio")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(results["random_returns"], bins=50, color="steelblue", edgecolor="white", alpha=0.7, label="Random Strategies")
        axes[1].axvline(results["actual_return"], color="red", linewidth=2, label=f"Actual: {results['actual_return']}%")
        axes[1].set_title("Total Return vs Random Strategies")
        axes[1].set_xlabel("Total Return (%)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_in_out_sample(self):
        """Plots in sample vs out of sample equity curves."""
        if "in_out_sample" not in self.stress_results:
            raise ValueError("Run in_out_sample() first.")

        results      = self.stress_results["in_out_sample"]
        in_pv        = results["in_sample"]["portfolio"]
        out_pv       = results["out_sample"]["portfolio"]
        in_dates     = results["in_sample"]["dates"]
        out_dates    = results["out_sample"]["dates"]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(in_dates.values,  in_pv.values,  linewidth=1.5, color="steelblue", label="In Sample")
        ax.plot(out_dates.values, out_pv.values, linewidth=1.5, color="orange",    label="Out of Sample")
        ax.axvline(in_dates.iloc[-1], color="black", linewidth=1, linestyle="--", label="Split Point")
        ax.axhline(self.initial_capital, color="black", linewidth=0.8, linestyle=":", label="Initial Capital")
        ax.set_title("In Sample vs Out of Sample Performance")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()