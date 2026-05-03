from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Component Types ───────────────────────────────────────────────────────────

ComponentType = Literal["strategy", "buy_and_hold"]
WeightMethod  = Literal["equal", "user_input", "mvo"]


# ── Portfolio Component ───────────────────────────────────────────────────────

@dataclass
class PortfolioComponent:
    """
    A single component within a portfolio.
    Either a strategy or a buy and hold asset.
    """
    name:           str
    component_type: ComponentType
    weight:         float = 0.0        # set by optimization

    # For strategy type
    strategy_results: Optional[dict] = None
    # expects output from StrategyEngine.run_*()
    # must have "portfolio" key with total_value column

    # For buy and hold type
    symbol:         Optional[str] = None
    # symbol must exist in enriched_df


# ── Portfolio Config ──────────────────────────────────────────────────────────

@dataclass
class PortfolioConfig:
    """
    Configuration for a portfolio.
    All user inputs live here.
    """
    name:             str
    initial_capital:  float = 100000.0
    weight_method:    WeightMethod = "equal"

    # For user input weights
    # key = component name, value = weight
    user_weights:     dict = field(default_factory=dict)

    # For MVO
    min_weight:       float = 0.05   # minimum weight per component
    max_weight:       float = 0.40   # maximum weight per component


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:

    def __init__(self,
                 df: pd.DataFrame,
                 config: PortfolioConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
                    needed for buy and hold assets
            config: PortfolioConfig with all user inputs
        """
        self.df         = df.copy()
        self.config     = config
        self.components = []          # list of PortfolioComponent
        self.weights    = {}          # name → weight
        self.returns_df = None        # aligned returns per component
        self.portfolio_df = None      # daily portfolio values
        self.metrics    = {}
        self.fitted     = False

    # ── Step 1 — Add components ───────────────────────────────────────────────

    def add_strategy(self, name: str, strategy_results: dict):
        """
        Adds a strategy to the portfolio.

        Args:
            name:             label for this component
            strategy_results: output from StrategyEngine.run_*()
                              must have "portfolio" key
        """
        if "portfolio" not in strategy_results:
            raise ValueError(
                f"strategy_results for '{name}' must have a 'portfolio' key. "
                f"Use output from StrategyEngine.run_single_asset() or similar."
            )

        component = PortfolioComponent(
            name=name,
            component_type="strategy",
            strategy_results=strategy_results,
        )
        self.components.append(component)
        print(f"  ✓ Added strategy: {name}")

    def add_buy_and_hold(self, name: str, symbol: str):
        """
        Adds a buy and hold asset to the portfolio.

        Args:
            name:   label for this component e.g. "AAPL Hold"
            symbol: ticker that must exist in enriched_df
        """
        if symbol not in self.df["symbol"].values:
            available = self.df["symbol"].unique().tolist()
            raise ValueError(
                f"Symbol '{symbol}' not found in dataset. "
                f"Available: {available}"
            )

        component = PortfolioComponent(
            name=name,
            component_type="buy_and_hold",
            symbol=symbol,
        )
        self.components.append(component)
        print(f"  ✓ Added buy and hold: {name} ({symbol})")

    def list_components(self):
        """Shows all components currently in the portfolio."""
        if not self.components:
            print(f"  Portfolio '{self.config.name}' is empty.")
            print(f"  Add components using add_strategy() or add_buy_and_hold()")
            return []

        print(f"\n── Portfolio: {self.config.name} ───────────────────")
        for i, c in enumerate(self.components):
            weight = self.weights.get(c.name, 0.0)
            print(
                f"  {i+1}. {c.name:30} "
                f"type: {c.component_type:12} "
                f"weight: {weight:.2%}"
            )

        return self.components

    # ── Step 2 — Get returns per component ───────────────────────────────────

    def _get_component_returns(self):
        """
        Extracts return series per component.
        Aligns all components on common dates automatically.
        """
        returns_dict = {}

        for component in self.components:

            if component.component_type == "strategy":
                # Get portfolio value series from strategy results
                port_df = component.strategy_results["portfolio"].copy()
                port_df = port_df.sort_values("date")
                returns = port_df.set_index("date")["total_value"].pct_change()
                returns_dict[component.name] = returns

            elif component.component_type == "buy_and_hold":
                # Get price series from enriched df
                sym_df = (
                    self.df[self.df["symbol"] == component.symbol]
                    .sort_values("date")
                    .set_index("date")["close"]
                )
                returns = sym_df.pct_change()
                returns_dict[component.name] = returns

        # Combine and align on common dates
        returns_df = pd.DataFrame(returns_dict)

        # Forward fill small gaps then drop remaining NaN
        returns_df = returns_df.ffill(limit=5).dropna()

        print(f"  Date range: {returns_df.index[0]} → {returns_df.index[-1]}")
        print(f"  Trading days: {len(returns_df)}")

        return returns_df

    # ── Step 3 — Calculate weights ────────────────────────────────────────────

    def _calculate_weights(self):
        """
        Calculates component weights based on weight_method.
        """
        n = len(self.components)
        names = [c.name for c in self.components]

        if self.config.weight_method == "equal":
            weight = 1.0 / n
            self.weights = {name: round(weight, 6) for name in names}
            print(f"  Equal weight: {round(weight * 100, 2)}% per component")

        elif self.config.weight_method == "user_input":
            # Validate user weights
            if not self.config.user_weights:
                raise ValueError(
                    "user_weights not set in PortfolioConfig. "
                    "Provide a dict of component name → weight."
                )

            missing = [n for n in names if n not in self.config.user_weights]
            if missing:
                raise ValueError(
                    f"Missing weights for components: {missing}. "
                    f"Provide weight for every component."
                )

            total = sum(self.config.user_weights[n] for n in names)
            if not np.isclose(total, 1.0, atol=0.01):
                raise ValueError(
                    f"User weights must sum to 1.0. "
                    f"Current sum: {round(total, 4)}"
                )

            self.weights = {n: self.config.user_weights[n] for n in names}
            print(f"  User defined weights applied.")

        elif self.config.weight_method == "mvo":
            self.weights = self._run_mvo()
            print(f"  MVO weights calculated.")

        # Print weights
        print(f"\n── Weights ─────────────────────────────────────")
        for name, w in self.weights.items():
            bar = "█" * int(w * 40)
            print(f"  {name:30} {w:.2%} {bar}")

    def _run_mvo(self):
        """
        Mean Variance Optimization.
        Finds maximum Sharpe portfolio subject to
        min/max weight constraints.
        Uses scipy optimize under the hood.
        """
        from scipy.optimize import minimize

        returns_df = self.returns_df
        names      = [c.name for c in self.components]
        n          = len(names)

        # Expected returns and covariance
        mean_returns = returns_df.mean() * 252
        cov_matrix   = returns_df.cov() * 252

        def _neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol    = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            )
            return -(port_return / port_vol) if port_vol > 0 else 0

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds — min and max weight per component
        bounds = [
            (self.config.min_weight, self.config.max_weight)
            for _ in range(n)
        ]

        # Initial guess — equal weight
        w0 = np.array([1.0 / n] * n)

        result = minimize(
            _neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if not result.success:
            print(
                f"  ⚠ MVO optimization did not converge. "
                f"Falling back to equal weight."
            )
            return {name: 1.0 / n for name in names}

        optimal_weights = result.x
        weights_dict    = {
            name: round(w, 6)
            for name, w in zip(names, optimal_weights)
        }

        # Print MVO details
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol    = np.sqrt(
            np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
        )
        port_sharpe = port_return / port_vol if port_vol > 0 else 0

        print(f"  MVO Result:")
        print(f"    Expected Return: {round(port_return * 100, 2)}%")
        print(f"    Expected Vol:    {round(port_vol * 100, 2)}%")
        print(f"    Expected Sharpe: {round(port_sharpe, 3)}")
        print(f"    Min Weight:      {self.config.min_weight:.2%}")
        print(f"    Max Weight:      {self.config.max_weight:.2%}")

        return weights_dict

    # ── Step 4 — Build portfolio equity curve ─────────────────────────────────

    def _build_portfolio(self):
        """
        Combines component returns using weights
        to produce portfolio equity curve.
        """
        returns_df = self.returns_df
        weights    = np.array([self.weights[c.name] for c in self.components])

        # Weighted portfolio returns
        port_returns = returns_df.values @ weights

        # Portfolio equity curve
        port_values  = self.config.initial_capital * np.cumprod(1 + port_returns)

        # Component equity curves (for decomposition)
        component_values = {}
        for component in self.components:
            w = self.weights[component.name]
            alloc = self.config.initial_capital * w
            comp_values = alloc * np.cumprod(
                1 + returns_df[component.name].values
            )
            component_values[component.name] = comp_values

        self.portfolio_df = pd.DataFrame({
            "date":        returns_df.index,
            "total_value": port_values,
            **component_values
        }).reset_index(drop=True)

        return self.portfolio_df

    # ── Step 5 — Calculate metrics ────────────────────────────────────────────

    def _calculate_metrics(self):
        """Calculates portfolio performance metrics."""
        pv      = self.portfolio_df["total_value"]
        returns = pv.pct_change().dropna()

        total_return      = (pv.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        n_days            = len(pv)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        annualized_vol    = returns.std() * np.sqrt(252)
        sharpe            = annualized_return / annualized_vol if annualized_vol != 0 else 0

        downside     = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252)
        sortino      = annualized_return / downside_std if downside_std != 0 else 0

        rolling_max  = pv.expanding().max()
        drawdown     = (pv - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        var_95  = np.percentile(returns, 5)
        var_99  = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Per component metrics
        component_metrics = []
        for component in self.components:
            col     = component.name
            if col not in self.portfolio_df.columns:
                continue

            comp_pv  = self.portfolio_df[col]
            alloc    = self.config.initial_capital * self.weights[col]
            comp_ret = (comp_pv.iloc[-1] - alloc) / alloc
            comp_returns = comp_pv.pct_change().dropna()
            comp_vol = comp_returns.std() * np.sqrt(252)
            comp_sharpe = (
                ((1 + comp_ret) ** (252 / len(comp_pv)) - 1) / comp_vol
                if comp_vol != 0 else 0
            )

            component_metrics.append({
                "name":        col,
                "type":        component.component_type,
                "weight":      self.weights[col],
                "allocation":  round(alloc, 2),
                "final_value": round(comp_pv.iloc[-1], 2),
                "return":      round(comp_ret * 100, 2),
                "sharpe":      round(comp_sharpe, 3),
            })

        self.metrics = {
            "name":               self.config.name,
            "initial_capital":    self.config.initial_capital,
            "final_value":        round(pv.iloc[-1], 2),
            "total_return":       round(total_return * 100, 2),
            "annualized_return":  round(annualized_return * 100, 2),
            "annualized_vol":     round(annualized_vol * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "max_drawdown":       round(max_drawdown * 100, 2),
            "var_95":             round(var_95 * 100, 4),
            "var_99":             round(var_99 * 100, 4),
            "cvar_95":            round(cvar_95 * 100, 4),
            "cvar_99":            round(cvar_99 * 100, 4),
            "weight_method":      self.config.weight_method,
            "n_components":       len(self.components),
            "component_metrics":  pd.DataFrame(component_metrics),
        }

        return self.metrics

    # ── Step 6 — Print results ────────────────────────────────────────────────

    def _print_results(self):
        """Prints portfolio performance summary."""

        print(f"\n── Portfolio: {self.config.name} ───────────────────")
        print(f"  Initial Capital:   ${self.metrics['initial_capital']:,.2f}")
        print(f"  Final Value:       ${self.metrics['final_value']:,.2f}")
        print(f"  Total Return:      {self.metrics['total_return']}%")
        print(f"  Annualized Return: {self.metrics['annualized_return']}%")
        print(f"  Annualized Vol:    {self.metrics['annualized_vol']}%")
        print(f"  Sharpe Ratio:      {self.metrics['sharpe_ratio']}")
        print(f"  Sortino Ratio:     {self.metrics['sortino_ratio']}")
        print(f"  Max Drawdown:      {self.metrics['max_drawdown']}%")
        print(f"  VaR  95%:          {self.metrics['var_95']}%")
        print(f"  VaR  99%:          {self.metrics['var_99']}%")
        print(f"  CVaR 95%:          {self.metrics['cvar_95']}%")
        print(f"  CVaR 99%:          {self.metrics['cvar_99']}%")
        print(f"  Weight Method:     {self.metrics['weight_method']}")

        print(f"\n── Component Breakdown ─────────────────────────")
        print(self.metrics["component_metrics"].to_string(index=False))

    # ── Step 7 — Plot ─────────────────────────────────────────────────────────

    def plot(self):
        """
        Plots portfolio equity curve, component breakdown,
        drawdown, and weight allocation.
        """
        if not self.fitted:
            raise ValueError("Run build() first.")

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

        dates = self.portfolio_df["date"]
        pv    = self.portfolio_df["total_value"]

        # ── Equity curve ──────────────────────────────────────────────────
        axes[0].plot(
            dates, pv,
            color="black", linewidth=2,
            label=f"Portfolio: {self.config.name}",
            zorder=5
        )
        axes[0].axhline(
            self.config.initial_capital,
            color="black", linewidth=0.8,
            linestyle="--", label="Initial Capital"
        )

        # Plot component curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.components)))
        for component, color in zip(self.components, colors):
            col = component.name
            if col in self.portfolio_df.columns:
                axes[0].plot(
                    dates,
                    self.portfolio_df[col],
                    linewidth=1.2,
                    linestyle="--",
                    color=color,
                    alpha=0.7,
                    label=col
                )

        axes[0].set_title(f"Portfolio Equity Curve — {self.config.name}")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # ── Drawdown ──────────────────────────────────────────────────────
        rolling_max = pv.expanding().max()
        drawdown    = (pv - rolling_max) / rolling_max

        axes[1].fill_between(
            dates, drawdown, 0,
            color="red", alpha=0.4, label="Drawdown"
        )
        axes[1].set_title("Portfolio Drawdown")
        axes[1].set_ylabel("Drawdown %")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ── Weight allocation pie chart ───────────────────────────────────
        names   = list(self.weights.keys())
        weights = list(self.weights.values())
        colors_pie = plt.cm.tab10(np.linspace(0, 1, len(names)))

        axes[2].pie(
            weights,
            labels=names,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90
        )
        axes[2].set_title(
            f"Weight Allocation ({self.config.weight_method})"
        )

        plt.tight_layout()
        plt.show()

    # ── Master build method ───────────────────────────────────────────────────

    def build(self, plot: bool = True):
        """
        Builds the portfolio — aligns returns, calculates weights,
        constructs equity curve, and computes metrics.

        Args:
            plot: whether to show performance chart
        """
        if not self.components:
            raise ValueError(
                f"Portfolio '{self.config.name}' has no components. "
                f"Add strategies or assets first using "
                f"add_strategy() or add_buy_and_hold()."
            )

        print(f"\n── Building Portfolio: {self.config.name} ──────────")
        print(f"  Components:      {len(self.components)}")
        print(f"  Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"  Weight Method:   {self.config.weight_method}")

        # Step 1 — get aligned returns
        print(f"\n  Aligning component returns...")
        self.returns_df = self._get_component_returns()

        # Step 2 — calculate weights
        print(f"\n  Calculating weights...")
        self._calculate_weights()

        # Step 3 — build equity curve
        print(f"\n  Building portfolio equity curve...")
        self._build_portfolio()

        # Step 4 — calculate metrics
        self._calculate_metrics()
        self._print_results()

        self.fitted = True

        if plot:
            self.plot()

        return {
            "metrics":   self.metrics,
            "portfolio": self.portfolio_df,
            "weights":   self.weights,
        }