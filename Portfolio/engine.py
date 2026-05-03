from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional

from Portfolio.portfolio import Portfolio, PortfolioConfig, PortfolioComponent
from Portfolio.optimization import EfficientFrontier
from Portfolio.stress_test import PortfolioStressTester
from Portfolio.decomposition import PortfolioDecomposition


class PortfolioEngine:

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: enriched dataframe from FeatureEngine/MLEngine
        """
        self.df         = df.copy()
        self.portfolios = {}   # name → Portfolio object
        self.active     = None # currently selected portfolio name

    # ── Portfolio management ──────────────────────────────────────────────────

    def create_portfolio(self,
                         name: str,
                         initial_capital: float = 100000.0,
                         weight_method: Literal["equal", "user_input", "mvo"] = "equal",
                         user_weights: dict = None,
                         min_weight: float = 0.05,
                         max_weight: float = 0.40):
        """
        Creates a new named portfolio.

        Args:
            name:            portfolio name
            initial_capital: starting capital
            weight_method:   "equal", "user_input", or "mvo"
            user_weights:    dict of component → weight
                             only needed if weight_method = "user_input"
            min_weight:      MVO minimum weight per component
            max_weight:      MVO maximum weight per component
        """
        if name in self.portfolios:
            raise ValueError(
                f"Portfolio '{name}' already exists. "
                f"Use a different name or delete existing portfolio first."
            )

        config = PortfolioConfig(
            name=name,
            initial_capital=initial_capital,
            weight_method=weight_method,
            user_weights=user_weights or {},
            min_weight=min_weight,
            max_weight=max_weight,
        )

        portfolio = Portfolio(df=self.df, config=config)
        self.portfolios[name] = portfolio
        self.active = name

        print(f"\n── Portfolio Created: {name} ────────────────────")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Weight Method:   {weight_method}")
        print(f"  Status:          Empty — add strategies or assets")

        return portfolio

    def select_portfolio(self, name: str):
        """Selects active portfolio."""
        if name not in self.portfolios:
            available = list(self.portfolios.keys())
            raise ValueError(
                f"Portfolio '{name}' not found. "
                f"Available portfolios: {available}"
            )
        self.active = name
        print(f"  Active portfolio: {name}")
        return self.portfolios[name]

    def delete_portfolio(self, name: str):
        """Deletes a portfolio."""
        if name not in self.portfolios:
            raise ValueError(f"Portfolio '{name}' not found.")
        del self.portfolios[name]
        if self.active == name:
            self.active = None
        print(f"  Portfolio '{name}' deleted.")

    def list_portfolios(self):
        """Shows all created portfolios."""
        if not self.portfolios:
            print("\n── No Portfolios Created ───────────────────────")
            print("  Use create_portfolio() to get started.")
            return []

        print(f"\n── Portfolios ──────────────────────────────────")
        for name, port in self.portfolios.items():
            status  = "built" if port.fitted else "not built"
            active  = "← active" if name == self.active else ""
            n_comp  = len(port.components)
            print(
                f"  {name:30} "
                f"components: {n_comp:2} "
                f"status: {status:10} {active}"
            )

        return list(self.portfolios.keys())

    # ── Add components ────────────────────────────────────────────────────────

    def add_strategy(self,
                     name: str,
                     strategy_results: dict,
                     portfolio_name: str = None):
        """
        Adds a strategy to a portfolio.

        Args:
            name:             component label
            strategy_results: output from StrategyEngine.run_*()
            portfolio_name:   which portfolio to add to
                              if None uses active portfolio
        """
        port = self._get_portfolio(portfolio_name)
        port.add_strategy(name, strategy_results)

    def add_buy_and_hold(self,
                         symbol: str,
                         name: str = None,
                         portfolio_name: str = None):
        """
        Adds a buy and hold asset to a portfolio.

        Args:
            symbol:         ticker e.g. "AAPL"
            name:           optional label, defaults to symbol
            portfolio_name: which portfolio to add to
        """
        port  = self._get_portfolio(portfolio_name)
        label = name or symbol
        port.add_buy_and_hold(label, symbol)

    def list_components(self, portfolio_name: str = None):
        """Shows components in a portfolio."""
        port = self._get_portfolio(portfolio_name)
        return port.list_components()

    # ── Build portfolio ───────────────────────────────────────────────────────

    def build(self,
              portfolio_name: str = None,
              plot: bool = True):
        """
        Builds portfolio — calculates weights, aligns returns,
        constructs equity curve.

        Args:
            portfolio_name: which portfolio to build
            plot:           whether to show chart
        """
        port = self._get_portfolio(portfolio_name)
        return port.build(plot=plot)

    # ── Efficient frontier ────────────────────────────────────────────────────

    def efficient_frontier(self,
                            portfolio_name: str = None,
                            n_portfolios: int = 5000,
                            min_weight: float = 0.0,
                            max_weight: float = 1.0,
                            plot: bool = True):
        """
        Generates and plots efficient frontier for a portfolio.
        Run before building to visualize optimal allocations.

        Args:
            portfolio_name: which portfolio
            n_portfolios:   random portfolios to simulate
            min_weight:     minimum weight per component
            max_weight:     maximum weight per component
            plot:           whether to show chart
        """
        port = self._get_portfolio(portfolio_name)

        if port.returns_df is None:
            # Need aligned returns without full build
            port.returns_df = port._get_component_returns()

        ef = EfficientFrontier(port.returns_df)
        ef.generate(n_portfolios, min_weight, max_weight)

        if plot:
            ef.plot()

        return ef

    # ── Stress testing ────────────────────────────────────────────────────────

    def run_monte_carlo(self,
                        portfolio_name: str = None,
                        num_sims: int = 10000,
                        num_days: int = 252,
                        distribution: str = "auto",
                        plot: bool = True):
        """
        Runs Monte Carlo simulation on portfolio returns.

        Args:
            portfolio_name: which portfolio
            num_sims:       number of simulations (100 to 100000)
            num_days:       days to simulate (21 to 1260)
            distribution:   "auto", "normal", "student_t", "skewed_t", "gev"
            plot:           whether to show chart
        """
        port = self._get_built_portfolio(portfolio_name)
        st   = PortfolioStressTester(port)

        results = st.monte_carlo(
            num_sims=num_sims,
            num_days=num_days,
            distribution=distribution
        )

        if plot:
            st.plot_monte_carlo()

        return results

    # ── Decomposition ─────────────────────────────────────────────────────────

    def decompose(self,
                  portfolio_name: str = None,
                  by: Literal["component", "factors", "all"] = "all",
                  factor_model: str = "6_factors",
                  rolling: bool = False,
                  roll_window: int = 36):
        """
        Decomposes portfolio returns.

        Args:
            portfolio_name: which portfolio
            by:             "component", "factors", or "all"
            factor_model:   FF factor model
            rolling:        rolling factor regression
            roll_window:    rolling window in months
        """
        port = self._get_built_portfolio(portfolio_name)
        decomp = PortfolioDecomposition(port)

        if by == "component":
            return decomp.decompose_by_component()
        elif by == "factors":
            return decomp.decompose_by_factors(
                factor_model=factor_model,
                rolling=rolling,
                roll_window=roll_window
            )
        else:
            return decomp.run_all(
                factor_model=factor_model,
                rolling=rolling,
                roll_window=roll_window
            )

    # ── Compare portfolios ────────────────────────────────────────────────────

    def compare(self, portfolio_names: list[str] = None):
        """
        Compares performance metrics across multiple portfolios.

        Args:
            portfolio_names: which portfolios to compare
                             if None compares all built portfolios
        """
        names = portfolio_names or list(self.portfolios.keys())
        rows  = []

        for name in names:
            port = self.portfolios.get(name)
            if port is None or not port.fitted:
                print(f"  ⚠ Skipping '{name}' — not built yet.")
                continue

            m = port.metrics
            rows.append({
                "portfolio":       name,
                "initial_capital": m["initial_capital"],
                "final_value":     m["final_value"],
                "total_return":    m["total_return"],
                "annualized_return": m["annualized_return"],
                "annualized_vol":  m["annualized_vol"],
                "sharpe_ratio":    m["sharpe_ratio"],
                "sortino_ratio":   m["sortino_ratio"],
                "max_drawdown":    m["max_drawdown"],
                "var_95":          m["var_95"],
                "cvar_95":         m["cvar_95"],
                "weight_method":   m["weight_method"],
                "n_components":    m["n_components"],
            })

        if not rows:
            print("No built portfolios to compare.")
            return pd.DataFrame()

        comparison_df = pd.DataFrame(rows)

        print(f"\n── Portfolio Comparison ────────────────────────")
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_comparison(self, portfolio_names: list[str] = None):
        """
        Plots equity curves of multiple portfolios on same chart.

        Args:
            portfolio_names: which portfolios to plot
        """
        names = portfolio_names or list(self.portfolios.keys())

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        for name in names:
            port = self.portfolios.get(name)
            if port is None or not port.fitted:
                continue

            pv    = port.portfolio_df["total_value"]
            dates = port.portfolio_df["date"]

            # Equity curve
            axes[0].plot(dates, pv, linewidth=1.5, label=name)

            # Drawdown
            rolling_max = pv.expanding().max()
            drawdown    = (pv - rolling_max) / rolling_max
            axes[1].plot(dates, drawdown, linewidth=1.2, label=name)

        axes[0].set_title("Portfolio Equity Curves Comparison")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Portfolio Drawdowns Comparison")
        axes[1].set_ylabel("Drawdown %")
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_portfolio(self, name: str = None) -> Portfolio:
        """Gets portfolio by name or returns active portfolio."""
        target = name or self.active

        if target is None:
            raise ValueError(
                "No active portfolio. "
                "Create one with create_portfolio() first."
            )

        if target not in self.portfolios:
            raise ValueError(
                f"Portfolio '{target}' not found. "
                f"Available: {list(self.portfolios.keys())}"
            )

        return self.portfolios[target]

    def _get_built_portfolio(self, name: str = None) -> Portfolio:
        """Gets a built portfolio — raises error if not built."""
        port = self._get_portfolio(name)

        if not port.fitted:
            raise ValueError(
                f"Portfolio '{port.config.name}' has not been built yet. "
                f"Call build() first."
            )

        return port