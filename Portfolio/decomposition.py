from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PortfolioDecomposition:
    """
    Decomposes portfolio returns by:
    1. Component contribution (strategy/asset level)
    2. Factor regression (market/style factor level)
    """

    def __init__(self, portfolio: object):
        """
        Args:
            portfolio: fitted Portfolio object
        """
        self.portfolio    = portfolio
        self.portfolio_df = portfolio.portfolio_df.copy()
        self.weights      = portfolio.weights
        self.components   = portfolio.components
        self.results      = {}

    # ── Component decomposition ───────────────────────────────────────────────

    def decompose_by_component(self):
        """
        Decomposes portfolio returns by component contribution.
        Shows how much each strategy/asset contributed to total return.
        """
        print(f"\n── Component Return Decomposition ──────────────")

        total_return = (
            self.portfolio_df["total_value"].iloc[-1] -
            self.portfolio.config.initial_capital
        ) / self.portfolio.config.initial_capital

        rows = []
        for component in self.components:
            col = component.name
            if col not in self.portfolio_df.columns:
                continue

            alloc     = self.portfolio.config.initial_capital * self.weights[col]
            comp_pv   = self.portfolio_df[col]
            comp_ret  = (comp_pv.iloc[-1] - alloc) / alloc
            weighted_contribution = comp_ret * self.weights[col]

            rows.append({
                "component":              col,
                "type":                   component.component_type,
                "weight":                 round(self.weights[col] * 100, 2),
                "allocation":             round(alloc, 2),
                "component_return":       round(comp_ret * 100, 2),
                "weighted_contribution":  round(weighted_contribution * 100, 2),
            })

        decomp_df = pd.DataFrame(rows)

        print(decomp_df.to_string(index=False))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

        # Component returns bar chart
        axes[0].barh(
            decomp_df["component"],
            decomp_df["component_return"],
            color=colors,
            edgecolor="white"
        )
        axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Component Returns")
        axes[0].set_xlabel("Total Return (%)")
        axes[0].grid(True, alpha=0.3, axis="x")

        # Weighted contribution bar chart
        axes[1].barh(
            decomp_df["component"],
            decomp_df["weighted_contribution"],
            color=colors,
            edgecolor="white"
        )
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Weighted Contribution to Portfolio Return")
        axes[1].set_xlabel("Weighted Contribution (%)")
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            f"Portfolio Decomposition — {self.portfolio.config.name}\n"
            f"Total Portfolio Return: {round(total_return * 100, 2)}%"
        )
        plt.tight_layout()
        plt.show()

        self.results["component"] = decomp_df
        return decomp_df

    # ── Factor decomposition ──────────────────────────────────────────────────

    def decompose_by_factors(self,
                              factor_model: str = "6_factors",
                              rolling: bool = False,
                              roll_window: int = 36):
        """
        Decomposes portfolio returns using factor regression.
        Shows how much return comes from market, momentum, value etc.

        Args:
            factor_model: "3_factors", "5_factors", or "6_factors"
            rolling:      whether to run rolling factor regression
            roll_window:  rolling window in months
        """
        from ML.factor_regression import FactorRegression, FactorRegressionConfig

        print(f"\n── Factor Return Decomposition ─────────────────")

        # Build portfolio returns df for factor regression
        port_returns_df = self.portfolio_df[["date", "total_value"]].copy()
        port_returns_df["date"] = pd.to_datetime(port_returns_df["date"])

        config = FactorRegressionConfig(
            mode="portfolio",
            factor_model=factor_model,
            return_col="total_value",
            rolling=rolling,
            roll_window=roll_window,
            add_to_dataset=False,
        )

        fr = FactorRegression(df=port_returns_df, config=config)
        fr.fit(plot=True)

        self.results["factors"] = fr.results
        return fr.results

    # ── Run all decompositions ────────────────────────────────────────────────

    def run_all(self,
                factor_model: str = "6_factors",
                rolling: bool = False,
                roll_window: int = 36):
        """
        Runs both component and factor decomposition.

        Args:
            factor_model: FF factor model to use
            rolling:      rolling factor regression
            roll_window:  rolling window in months
        """
        self.decompose_by_component()
        self.decompose_by_factors(
            factor_model=factor_model,
            rolling=rolling,
            roll_window=roll_window
        )
        return self.results