from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EfficientFrontier:
    """
    Generates the efficient frontier for a set of assets/strategies.
    Shows the optimal risk/return tradeoff.
    Separate from MVO inside Portfolio so users can
    visualize the frontier before committing to weights.
    """

    def __init__(self, returns_df: pd.DataFrame):
        """
        Args:
            returns_df: aligned daily returns per component
                        one column per strategy/asset
        """
        self.returns_df   = returns_df.dropna()
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix   = returns_df.cov() * 252
        self.n            = len(returns_df.columns)
        self.names        = returns_df.columns.tolist()
        self.frontier     = None

    def _portfolio_stats(self, weights):
        """Returns annualized return, vol, and Sharpe for given weights."""
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = ret / vol if vol > 0 else 0
        return ret, vol, sharpe

    def _max_sharpe(self, min_weight=0.0, max_weight=1.0):
        """Finds maximum Sharpe portfolio."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds      = [(min_weight, max_weight)] * self.n
        w0          = np.array([1.0 / self.n] * self.n)

        result = minimize(
            lambda w: -self._portfolio_stats(w)[2],
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        return result.x if result.success else w0

    def _min_volatility(self, min_weight=0.0, max_weight=1.0):
        """Finds minimum volatility portfolio."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds      = [(min_weight, max_weight)] * self.n
        w0          = np.array([1.0 / self.n] * self.n)

        result = minimize(
            lambda w: self._portfolio_stats(w)[1],
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        return result.x if result.success else w0

    def generate(self,
                 n_portfolios: int = 5000,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):
        """
        Generates efficient frontier by simulating random portfolios.

        Args:
            n_portfolios: number of random portfolios to simulate
            min_weight:   minimum weight per component
            max_weight:   maximum weight per component
        """
        print(f"  Generating efficient frontier ({n_portfolios:,} portfolios)...")

        np.random.seed(42)
        results = []

        for _ in range(n_portfolios):
            # Random weights within bounds
            w = np.random.dirichlet(np.ones(self.n))
            w = np.clip(w, min_weight, max_weight)
            w = w / w.sum()

            ret, vol, sharpe = self._portfolio_stats(w)
            results.append({
                "return":  ret * 100,
                "vol":     vol * 100,
                "sharpe":  sharpe,
                "weights": w,
            })

        self.frontier = pd.DataFrame(results)

        # Find special portfolios
        max_sharpe_w = self._max_sharpe(min_weight, max_weight)
        min_vol_w    = self._min_volatility(min_weight, max_weight)

        max_sharpe_stats = self._portfolio_stats(max_sharpe_w)
        min_vol_stats    = self._portfolio_stats(min_vol_w)

        self.max_sharpe_portfolio = {
            "weights": {n: round(w, 4) for n, w in zip(self.names, max_sharpe_w)},
            "return":  round(max_sharpe_stats[0] * 100, 2),
            "vol":     round(max_sharpe_stats[1] * 100, 2),
            "sharpe":  round(max_sharpe_stats[2], 3),
        }

        self.min_vol_portfolio = {
            "weights": {n: round(w, 4) for n, w in zip(self.names, min_vol_w)},
            "return":  round(min_vol_stats[0] * 100, 2),
            "vol":     round(min_vol_stats[1] * 100, 2),
            "sharpe":  round(min_vol_stats[2], 3),
        }

        print(f"\n── Max Sharpe Portfolio ────────────────────────")
        print(f"  Return: {self.max_sharpe_portfolio['return']}%")
        print(f"  Vol:    {self.max_sharpe_portfolio['vol']}%")
        print(f"  Sharpe: {self.max_sharpe_portfolio['sharpe']}")
        print(f"  Weights:")
        for n, w in self.max_sharpe_portfolio["weights"].items():
            print(f"    {n:30} {w:.2%}")

        print(f"\n── Min Volatility Portfolio ────────────────────")
        print(f"  Return: {self.min_vol_portfolio['return']}%")
        print(f"  Vol:    {self.min_vol_portfolio['vol']}%")
        print(f"  Sharpe: {self.min_vol_portfolio['sharpe']}")

        return self.frontier

    def plot(self):
        """Plots the efficient frontier scatter."""
        if self.frontier is None:
            raise ValueError("Run generate() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter all portfolios colored by Sharpe
        sc = ax.scatter(
            self.frontier["vol"],
            self.frontier["return"],
            c=self.frontier["sharpe"],
            cmap="viridis",
            alpha=0.4,
            s=10
        )
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

        # Max Sharpe
        ax.scatter(
            self.max_sharpe_portfolio["vol"],
            self.max_sharpe_portfolio["return"],
            color="red", s=150, zorder=5,
            label=f"Max Sharpe ({self.max_sharpe_portfolio['sharpe']})"
        )

        # Min Vol
        ax.scatter(
            self.min_vol_portfolio["vol"],
            self.min_vol_portfolio["return"],
            color="blue", s=150, zorder=5,
            label=f"Min Vol ({self.min_vol_portfolio['vol']}%)"
        )

        # Individual components
        for name in self.names:
            ret = self.mean_returns[name] * 100
            vol = np.sqrt(self.cov_matrix.loc[name, name]) * 100
            ax.scatter(vol, ret, s=100, zorder=6, marker="D")
            ax.annotate(name, (vol, ret), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        ax.set_title("Efficient Frontier")
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()