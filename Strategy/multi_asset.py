from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .conditions import Condition, LogicType
from .signals import build_signals


@dataclass
class MultiAssetConfig:
    """
    Configuration for a multi asset strategy.
    All user inputs live here.
    """
    symbols:            list[str]
    entry_conditions:   list[Condition]
    exit_conditions:    list[Condition]
    initial_capital:    float = 100000.0
    entry_logic:        LogicType = "and"
    exit_logic:         LogicType = "and"
    commission_pct:     float = 0.001
    commission_fixed:   float = 1.0
    slippage_pct:       float = 0.0005


class MultiAssetStrategy:

    def __init__(self, df: pd.DataFrame, config: MultiAssetConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
            config: MultiAssetConfig with all user inputs
        """
        self.df = df.copy()
        self.config = config
        self.signals_df = None
        self.trades_df = None
        self.portfolio_df = None
        self.metrics = {}

    # ── Step 1 — Validate inputs ─────────────────────────────────────────────

    def _validate(self):
        """Checks all symbols exist and required columns are present."""
        available = self.df["symbol"].unique().tolist()
        invalid = [s for s in self.config.symbols if s not in available]
        if invalid:
            raise ValueError(
                f"Symbols not found: {invalid}. "
                f"Available symbols: {available}"
            )

        required_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0.")

        if len(self.config.symbols) < 2:
            raise ValueError(
                "Multi asset strategy requires at least 2 symbols. "
                "Use SingleAssetStrategy for one symbol."
            )

    # ── Step 2 — Generate signals ────────────────────────────────────────────

    def _generate_signals(self):
        """Builds entry/exit signals for all symbols."""
        self.signals_df = build_signals(
            df=self.df,
            entry_conditions=self.config.entry_conditions,
            exit_conditions=self.config.exit_conditions,
            selected_symbols=self.config.symbols,
            entry_logic=self.config.entry_logic,
            exit_logic=self.config.exit_logic,
        )
        return self.signals_df

    # ── Step 3 — Calculate commission ────────────────────────────────────────

    def _calculate_commission(self, trade_value: float) -> float:
        return max(
            self.config.commission_fixed,
            trade_value * self.config.commission_pct
        )

    # ── Step 4 — Calculate slippage ──────────────────────────────────────────

    def _calculate_slippage(self, price: float, direction: str) -> float:
        if direction == "buy":
            return price * (1 + self.config.slippage_pct)
        else:
            return price * (1 - self.config.slippage_pct)

    # ── Step 5 — Run backtest ────────────────────────────────────────────────

    def _run_backtest(self):
        """
        Loops through each date.
        Allocates capital equally across all active signals.
        Tracks portfolio value across all symbols combined.
        """
        df = self.signals_df.copy()
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

        dates = sorted(df["date"].unique())

        # Per symbol tracking
        symbol_state = {
            symbol: {
                "cash":     self.config.initial_capital / len(self.config.symbols),
                "shares":   0.0,
                "in_position": False,
            }
            for symbol in self.config.symbols
        }

        portfolio_records = []
        trade_records = []

        for date in dates:
            day_df = df[df["date"] == date]

            # Count active signals today for equal weight allocation
            entry_signals = day_df[
                day_df["entry_signal"] & 
                ~day_df["symbol"].map(lambda s: symbol_state[s]["in_position"])
            ]["symbol"].tolist()

            for _, row in day_df.iterrows():
                symbol = row["symbol"]
                state  = symbol_state[symbol]
                exec_price = row["execution_price"]

                if pd.isna(exec_price):
                    continue

                # ── Entry ──────────────────────────────────────────────────
                if row["entry_signal"] and not state["in_position"]:
                    buy_price  = self._calculate_slippage(exec_price, "buy")
                    commission = self._calculate_commission(state["cash"])
                    shares     = (state["cash"] - commission) / buy_price

                    state["shares"]      = shares
                    state["cash"]        = 0.0
                    state["in_position"] = True

                    trade_records.append({
                        "date":       date,
                        "symbol":     symbol,
                        "type":       "BUY",
                        "price":      buy_price,
                        "shares":     shares,
                        "commission": commission,
                        "slippage":   exec_price * self.config.slippage_pct,
                        "value":      shares * buy_price,
                    })

                # ── Exit ───────────────────────────────────────────────────
                elif row["exit_signal"] and state["in_position"]:
                    sell_price  = self._calculate_slippage(exec_price, "sell")
                    trade_value = state["shares"] * sell_price
                    commission  = self._calculate_commission(trade_value)

                    state["cash"]        = trade_value - commission
                    state["in_position"] = False

                    trade_records.append({
                        "date":       date,
                        "symbol":     symbol,
                        "type":       "SELL",
                        "price":      sell_price,
                        "shares":     state["shares"],
                        "commission": commission,
                        "slippage":   exec_price * self.config.slippage_pct,
                        "value":      trade_value,
                    })

                    state["shares"] = 0.0

            # ── Track total portfolio value for this date ──────────────────
            total_value    = 0.0
            total_cash     = 0.0
            total_position = 0.0

            per_symbol = {}
            for symbol in self.config.symbols:
                state       = symbol_state[symbol]
                close_price = day_df[day_df["symbol"] == symbol]["close"].values

                if len(close_price) == 0:
                    continue

                position_value = state["shares"] * close_price[0]
                symbol_total   = state["cash"] + position_value

                total_cash     += state["cash"]
                total_position += position_value
                total_value    += symbol_total

                per_symbol[f"{symbol}_value"] = round(symbol_total, 2)

            portfolio_records.append({
                "date":           date,
                "total_cash":     round(total_cash, 2),
                "total_position": round(total_position, 2),
                "total_value":    round(total_value, 2),
                **per_symbol
            })

        self.portfolio_df = pd.DataFrame(portfolio_records)
        self.trades_df    = pd.DataFrame(trade_records)

    # ── Step 6 — Calculate performance metrics ───────────────────────────────

    def _calculate_metrics(self):
        """Calculates performance metrics from combined portfolio history."""
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

        # Win rate across all symbols
        sells = self.trades_df[self.trades_df["type"] == "SELL"]
        buys  = self.trades_df[self.trades_df["type"] == "BUY"]

        win_rate = 0.0
        if len(sells) > 0 and len(buys) > 0:
            min_trades = min(len(sells), len(buys))
            profits    = sells["value"].values[:min_trades] - buys["value"].values[:min_trades]
            win_rate   = (profits > 0).sum() / len(profits)

        # Per symbol breakdown
        symbol_breakdown = {}
        for symbol in self.config.symbols:
            col = f"{symbol}_value"
            if col in self.portfolio_df.columns:
                sym_pv     = self.portfolio_df[col]
                sym_start  = self.config.initial_capital / len(self.config.symbols)
                sym_return = (sym_pv.iloc[-1] - sym_start) / sym_start
                symbol_breakdown[symbol] = round(sym_return * 100, 2)

        self.metrics = {
            "symbols":            self.config.symbols,
            "initial_capital":    self.config.initial_capital,
            "final_value":        round(pv.iloc[-1], 2),
            "total_return":       round(total_return * 100, 2),
            "annualized_return":  round(annualized_return * 100, 2),
            "annualized_vol":     round(annualized_vol * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "max_drawdown":       round(max_drawdown * 100, 2),
            "win_rate":           round(win_rate * 100, 2),
            "total_trades":       len(buys),
            "total_commission":   round(self.trades_df["commission"].sum(), 2),
            "total_slippage":     round(self.trades_df["slippage"].sum(), 2),
            "symbol_returns":     symbol_breakdown,
        }

        return self.metrics

    # ── Step 7 — Print results ───────────────────────────────────────────────

    def _print_results(self):
        """Prints performance summary."""
        print(f"\n── Multi Asset Strategy Results ────────────────")
        print(f"  Symbols:           {self.config.symbols}")
        print(f"  Initial Capital:   ${self.config.initial_capital:,.2f}")
        print(f"  Final Value:       ${self.metrics['final_value']:,.2f}")
        print(f"  Total Return:      {self.metrics['total_return']}%")
        print(f"  Annualized Return: {self.metrics['annualized_return']}%")
        print(f"  Annualized Vol:    {self.metrics['annualized_vol']}%")
        print(f"  Sharpe Ratio:      {self.metrics['sharpe_ratio']}")
        print(f"  Sortino Ratio:     {self.metrics['sortino_ratio']}")
        print(f"  Max Drawdown:      {self.metrics['max_drawdown']}%")
        print(f"  Win Rate:          {self.metrics['win_rate']}%")
        print(f"  Total Trades:      {self.metrics['total_trades']}")
        print(f"  Total Commission:  ${self.metrics['total_commission']:,.2f}")
        print(f"  Total Slippage:    ${self.metrics['total_slippage']:,.2f}")
        print(f"\n── Per Symbol Returns ──────────────────────────")
        for symbol, ret in self.metrics["symbol_returns"].items():
            print(f"  {symbol}: {ret}%")

    # ── Step 8 — Plot ────────────────────────────────────────────────────────

    def plot(self):
        """Plots combined equity curve, per symbol breakdown, and drawdown."""
        import matplotlib.pyplot as plt

        n_plots = 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)

        # Combined equity curve
        axes[0].plot(
            self.portfolio_df["date"],
            self.portfolio_df["total_value"],
            linewidth=1.5,
            label="Total Portfolio",
            color="steelblue"
        )
        axes[0].axhline(
            self.config.initial_capital,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="Initial Capital"
        )
        axes[0].set_title("Combined Portfolio Equity Curve")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Per symbol value
        for symbol in self.config.symbols:
            col = f"{symbol}_value"
            if col in self.portfolio_df.columns:
                axes[1].plot(
                    self.portfolio_df["date"],
                    self.portfolio_df[col],
                    linewidth=1.2,
                    label=symbol
                )
        axes[1].set_title("Per Symbol Portfolio Value")
        axes[1].set_ylabel("Value ($)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        pv          = self.portfolio_df["total_value"]
        rolling_max = pv.expanding().max()
        drawdown    = (pv - rolling_max) / rolling_max

        axes[2].fill_between(
            self.portfolio_df["date"],
            drawdown,
            0,
            color="red",
            alpha=0.4,
            label="Drawdown"
        )
        axes[2].set_title("Drawdown")
        axes[2].set_ylabel("Drawdown %")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    # ── Master run method ────────────────────────────────────────────────────

    def run(self, plot: bool = True):
        """
        Runs the full multi asset strategy pipeline.

        Args:
            plot: whether to show performance chart
        """
        print(f"Running Multi Asset Strategy for {self.config.symbols}...")

        self._validate()
        self._generate_signals()
        self._run_backtest()
        self._calculate_metrics()
        self._print_results()

        if plot:
            self.plot()

        return {
            "metrics":   self.metrics,
            "portfolio": self.portfolio_df,
            "trades":    self.trades_df,
            "signals":   self.signals_df,
        }