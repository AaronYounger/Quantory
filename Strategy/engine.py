from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .conditions import Condition, LogicType
from .single_asset import SingleAssetStrategy, SingleAssetConfig
from .multi_asset import MultiAssetStrategy, MultiAssetConfig
from .quantile import QuantileStrategy, QuantileConfig


StrategyType = Literal["single_asset", "multi_asset", "quantile"]


class StrategyEngine:

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: enriched dataframe from FeatureEngine
        """
        self.df = df.copy()
        self.results = {}
        self.strategy_log = []

    # ── List available symbols ───────────────────────────────────────────────

    def get_available_symbols(self):
        """
        Shows all symbols available for strategy building.
        Frontend uses this to populate dropdowns.
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

    # ── List available columns for conditions ────────────────────────────────

    def get_available_columns(self):
        """
        Shows all columns available for building conditions.
        Frontend uses this to populate condition dropdowns.
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

        print("Available columns for conditions:")
        for col in available:
            print(f"  → {col}")

        return available

    # ── Run single asset strategy ────────────────────────────────────────────

    def run_single_asset(self,
                         strategy_name: str,
                         symbol: str,
                         entry_conditions: list[Condition],
                         exit_conditions: list[Condition],
                         initial_capital: float = 100000.0,
                         entry_logic: LogicType = "and",
                         exit_logic: LogicType = "and",
                         commission_pct: float = 0.001,
                         commission_fixed: float = 1.0,
                         slippage_pct: float = 0.0005,
                         plot: bool = True):
        """
        Builds and runs a single asset strategy.

        Args:
            strategy_name:     name to identify this strategy
            symbol:            ticker to trade e.g. "AAPL"
            entry_conditions:  list of Condition objects for entry
            exit_conditions:   list of Condition objects for exit
            initial_capital:   starting capital, default $100,000
            entry_logic:       "and" or "or" between entry conditions
            exit_logic:        "and" or "or" between exit conditions
            commission_pct:    % commission per trade, default 0.1%
            commission_fixed:  fixed commission per trade, default $1
            slippage_pct:      slippage per trade, default 0.05%
            plot:              whether to show performance chart
        """
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name}")
        print(f"Type: Single Asset")
        print(f"{'='*50}")

        config = SingleAssetConfig(
            symbol=symbol,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            initial_capital=initial_capital,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            commission_pct=commission_pct,
            commission_fixed=commission_fixed,
            slippage_pct=slippage_pct,
        )

        strategy = SingleAssetStrategy(df=self.df, config=config)
        results  = strategy.run(plot=plot)

        # Store results
        self.results[strategy_name] = {
            "type":    "single_asset",
            "config":  config,
            "results": results,
        }

        # Log
        self.strategy_log.append({
            "name":    strategy_name,
            "type":    "single_asset",
            "symbol":  symbol,
            "capital": initial_capital,
        })

        return results

    # ── Run multi asset strategy ─────────────────────────────────────────────

    def run_multi_asset(self,
                        strategy_name: str,
                        symbols: list[str],
                        entry_conditions: list[Condition],
                        exit_conditions: list[Condition],
                        initial_capital: float = 100000.0,
                        entry_logic: LogicType = "and",
                        exit_logic: LogicType = "and",
                        commission_pct: float = 0.001,
                        commission_fixed: float = 1.0,
                        slippage_pct: float = 0.0005,
                        plot: bool = True):
        """
        Builds and runs a multi asset strategy.

        Args:
            strategy_name:     name to identify this strategy
            symbols:           list of tickers to trade
            entry_conditions:  list of Condition objects for entry
            exit_conditions:   list of Condition objects for exit
            initial_capital:   starting capital, default $100,000
            entry_logic:       "and" or "or" between entry conditions
            exit_logic:        "and" or "or" between exit conditions
            commission_pct:    % commission per trade, default 0.1%
            commission_fixed:  fixed commission per trade, default $1
            slippage_pct:      slippage per trade, default 0.05%
            plot:              whether to show performance chart
        """
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name}")
        print(f"Type: Multi Asset")
        print(f"{'='*50}")

        config = MultiAssetConfig(
            symbols=symbols,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            initial_capital=initial_capital,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            commission_pct=commission_pct,
            commission_fixed=commission_fixed,
            slippage_pct=slippage_pct,
        )

        strategy = MultiAssetStrategy(df=self.df, config=config)
        results  = strategy.run(plot=plot)

        # Store results
        self.results[strategy_name] = {
            "type":    "multi_asset",
            "config":  config,
            "results": results,
        }

        # Log
        self.strategy_log.append({
            "name":    strategy_name,
            "type":    "multi_asset",
            "symbols": symbols,
            "capital": initial_capital,
        })

        return results

    # ── Run quantile strategy ────────────────────────────────────────────────

    def run_quantile(self,
                     strategy_name: str,
                     historical_rankings: pd.DataFrame,
                     target_quantile: str,
                     rebalance_type: str = "partial",
                     initial_capital: float = 100000.0,
                     commission_pct: float = 0.001,
                     commission_fixed: float = 1.0,
                     slippage_pct: float = 0.0005,
                     plot: bool = True):
        """
        Builds and runs a quantile/factor strategy.

        Args:
            strategy_name:        name to identify this strategy
            historical_rankings:  output from screener.historical_rank()
            target_quantile:      which quantile to trade e.g. "Q1"
            rebalance_type:       "partial" or "full"
            initial_capital:      starting capital, default $100,000
            commission_pct:       % commission per trade, default 0.1%
            commission_fixed:     fixed commission per trade, default $1
            slippage_pct:         slippage per trade, default 0.05%
            plot:                 whether to show performance chart
        """
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name}")
        print(f"Type: Quantile")
        print(f"{'='*50}")

        config = QuantileConfig(
            historical_rankings=historical_rankings,
            target_quantile=target_quantile,
            rebalance_type=rebalance_type,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            commission_fixed=commission_fixed,
            slippage_pct=slippage_pct,
        )

        strategy = QuantileStrategy(df=self.df, config=config)
        results  = strategy.run(plot=plot)

        # Store results
        self.results[strategy_name] = {
            "type":    "quantile",
            "config":  config,
            "results": results,
        }

        # Log
        self.strategy_log.append({
            "name":     strategy_name,
            "type":     "quantile",
            "quantile": target_quantile,
            "capital":  initial_capital,
        })

        return results

    # ── Compare strategies ───────────────────────────────────────────────────

    def compare_strategies(self, strategy_names: list[str] = None):
        """
        Compares performance metrics across multiple strategies.
        If no names provided compares all stored strategies.

        Args:
            strategy_names: optional list of strategy names to compare
        """
        if not self.results:
            raise ValueError(
                "No strategies run yet. "
                "Run at least one strategy first."
            )

        names = strategy_names or list(self.results.keys())
        rows  = []

        for name in names:
            if name not in self.results:
                print(f"Warning: Strategy '{name}' not found, skipping.")
                continue

            metrics = self.results[name]["results"]["metrics"]
            row     = {"strategy": name, "type": self.results[name]["type"]}
            row.update(metrics)
            rows.append(row)

        comparison_df = pd.DataFrame(rows)

        # Select key metrics for display
        display_cols = [
            "strategy", "type",
            "total_return", "annualized_return",
            "annualized_vol", "sharpe_ratio",
            "sortino_ratio", "max_drawdown",
            "win_rate", "total_trades",
            "total_commission", "total_slippage"
        ]

        display_cols = [
            c for c in display_cols
            if c in comparison_df.columns
        ]

        print(f"\n── Strategy Comparison ─────────────────────────")
        print(comparison_df[display_cols].to_string(index=False))

        return comparison_df

    # ── Plot equity curves together ──────────────────────────────────────────

    def plot_comparison(self, strategy_names: list[str] = None):
        """
        Plots equity curves of multiple strategies on same chart.

        Args:
            strategy_names: optional list of strategy names to plot
        """
        import matplotlib.pyplot as plt

        if not self.results:
            raise ValueError("No strategies run yet.")

        names = strategy_names or list(self.results.keys())

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for name in names:
            if name not in self.results:
                continue

            portfolio = self.results[name]["results"]["portfolio"]

            # Equity curve
            axes[0].plot(
                portfolio["date"],
                portfolio["total_value"],
                linewidth=1.5,
                label=name
            )

            # Drawdown
            pv          = portfolio["total_value"]
            rolling_max = pv.expanding().max()
            drawdown    = (pv - rolling_max) / rolling_max

            axes[1].plot(
                portfolio["date"],
                drawdown,
                linewidth=1.2,
                label=name
            )

        axes[0].set_title("Strategy Equity Curves")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Strategy Drawdowns")
        axes[1].set_ylabel("Drawdown %")
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    # ── Get strategy log ─────────────────────────────────────────────────────

    def get_strategy_log(self):
        """Shows all strategies that have been run."""
        if not self.strategy_log:
            print("No strategies run yet.")
            return pd.DataFrame()

        log_df = pd.DataFrame(self.strategy_log)
        print("\n── Strategy Log ────────────────────────────────")
        print(log_df.to_string(index=False))
        return log_df

    # ── Get strategy results ─────────────────────────────────────────────────

    def get_results(self, strategy_name: str):
        """
        Returns full results for a specific strategy.

        Args:
            strategy_name: name of strategy to retrieve
        """
        if strategy_name not in self.results:
            available = list(self.results.keys())
            raise ValueError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {available}"
            )
        return self.results[strategy_name]["results"]