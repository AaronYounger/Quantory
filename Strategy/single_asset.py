from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .conditions import Condition, LogicType
from .signals import build_signals


@dataclass
class SingleAssetConfig:
    """
    Configuration for a single asset strategy.
    All user inputs live here.
    """
    symbol:             str
    entry_conditions:   list[Condition]
    exit_conditions:    list[Condition]
    initial_capital:    float = 100000.0
    entry_logic:        LogicType = "and"
    exit_logic:         LogicType = "and"
    commission_pct:     float = 0.001      # 0.1% per trade
    commission_fixed:   float = 1.0        # $1 minimum
    slippage_pct:       float = 0.0005     # 0.05% per trade


class SingleAssetStrategy:

    def __init__(self, df: pd.DataFrame, config: SingleAssetConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
            config: SingleAssetConfig with all user inputs
        """
        self.df = df.copy()
        self.config = config
        self.signals_df = None
        self.trades_df = None
        self.portfolio_df = None
        self.metrics = {}

    # ── Step 1 — Validate inputs ─────────────────────────────────────────────

    def _validate(self):
        """Checks symbol exists and required columns are present."""
        if self.config.symbol not in self.df["symbol"].values:
            available = self.df["symbol"].unique().tolist()
            raise ValueError(
                f"Symbol '{self.config.symbol}' not found. "
                f"Available symbols: {available}"
            )

        required_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0.")

    # ── Step 2 — Generate signals ────────────────────────────────────────────

    def _generate_signals(self):
        """Builds entry/exit signals and position state."""
        self.signals_df = build_signals(
            df=self.df,
            entry_conditions=self.config.entry_conditions,
            exit_conditions=self.config.exit_conditions,
            selected_symbols=[self.config.symbol],
            entry_logic=self.config.entry_logic,
            exit_logic=self.config.exit_logic,
        )
        return self.signals_df

    # ── Step 3 — Calculate commission ───────────────────────────────────────

    def _calculate_commission(self, trade_value: float) -> float:
        """
        Commission = max(fixed, percentage of trade value)
        """
        return max(
            self.config.commission_fixed,
            trade_value * self.config.commission_pct
        )

    # ── Step 4 — Calculate slippage ──────────────────────────────────────────

    def _calculate_slippage(self, price: float, direction: str) -> float:
        """
        Slippage makes buys more expensive and sells cheaper.
        direction: "buy" or "sell"
        """
        if direction == "buy":
            return price * (1 + self.config.slippage_pct)
        else:
            return price * (1 - self.config.slippage_pct)

    # ── Step 5 — Run backtest ────────────────────────────────────────────────

    def _run_backtest(self):
        """
        Loops through signals df row by row.
        Executes trades, tracks portfolio value.
        """
        df = self.signals_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Portfolio tracking
        cash = self.config.initial_capital
        shares = 0.0
        in_position = False

        portfolio_records = []
        trade_records = []

        for i, row in df.iterrows():

            exec_price = row["execution_price"]
            date = row["date"]

            # Skip if no execution price (last row)
            if pd.isna(exec_price):
                portfolio_records.append({
                    "date":            date,
                    "cash":            cash,
                    "shares":          shares,
                    "position_value":  shares * row["close"],
                    "total_value":     cash + shares * row["close"],
                    "in_position":     in_position,
                })
                continue

            # ── Entry ──────────────────────────────────────────────────────
            if row["entry_signal"] and not in_position:
                buy_price  = self._calculate_slippage(exec_price, "buy")
                commission = self._calculate_commission(cash)
                shares     = (cash - commission) / buy_price
                cash       = 0.0
                in_position = True

                trade_records.append({
                    "date":       date,
                    "type":       "BUY",
                    "price":      buy_price,
                    "shares":     shares,
                    "commission": commission,
                    "slippage":   exec_price * self.config.slippage_pct,
                    "value":      shares * buy_price,
                })

            # ── Exit ───────────────────────────────────────────────────────
            elif row["exit_signal"] and in_position:
                sell_price    = self._calculate_slippage(exec_price, "sell")
                trade_value   = shares * sell_price
                commission    = self._calculate_commission(trade_value)
                cash          = trade_value - commission
                in_position   = False

                trade_records.append({
                    "date":       date,
                    "type":       "SELL",
                    "price":      sell_price,
                    "shares":     shares,
                    "commission": commission,
                    "slippage":   exec_price * self.config.slippage_pct,
                    "value":      trade_value,
                })

                shares = 0.0

            # ── Track portfolio ────────────────────────────────────────────
            position_value = shares * row["close"]
            total_value    = cash + position_value

            portfolio_records.append({
                "date":           date,
                "cash":           cash,
                "shares":         shares,
                "position_value": position_value,
                "total_value":    total_value,
                "in_position":    in_position,
            })

        self.portfolio_df = pd.DataFrame(portfolio_records)
        self.trades_df    = pd.DataFrame(trade_records)

    # ── Step 6 — Calculate performance metrics ───────────────────────────────

    def _calculate_metrics(self):
        """
        Calculates performance metrics from portfolio history.
        """
        pv = self.portfolio_df["total_value"]
        returns = pv.pct_change().dropna()

        # Total return
        total_return = (pv.iloc[-1] - self.config.initial_capital) / self.config.initial_capital

        # Annualized return
        n_days = len(pv)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0

        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252)
        sortino = annualized_return / downside_std if downside_std != 0 else 0

        # Max drawdown
        rolling_max = pv.expanding().max()
        drawdown = (pv - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        if len(self.trades_df) > 0:
            sells = self.trades_df[self.trades_df["type"] == "SELL"]
            buys  = self.trades_df[self.trades_df["type"] == "BUY"]
            if len(sells) > 0 and len(buys) > 0:
                min_trades = min(len(sells), len(buys))
                profits = sells["value"].values[:min_trades] - buys["value"].values[:min_trades]
                win_rate = (profits > 0).sum() / len(profits)
            else:
                win_rate = 0
        else:
            win_rate = 0

        # Total trades
        total_trades = len(self.trades_df[self.trades_df["type"] == "BUY"])

        # Total commission paid
        total_commission = self.trades_df["commission"].sum()

        # Total slippage paid
        total_slippage = self.trades_df["slippage"].sum()

        self.metrics = {
            "symbol":             self.config.symbol,
            "initial_capital":    self.config.initial_capital,
            "final_value":        round(pv.iloc[-1], 2),
            "total_return":       round(total_return * 100, 2),
            "annualized_return":  round(annualized_return * 100, 2),
            "annualized_vol":     round(annualized_vol * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "max_drawdown":       round(max_drawdown * 100, 2),
            "win_rate":           round(win_rate * 100, 2),
            "total_trades":       total_trades,
            "total_commission":   round(total_commission, 2),
            "total_slippage":     round(total_slippage, 2),
        }

        return self.metrics

    # ── Step 7 — Print results ───────────────────────────────────────────────

    def _print_results(self):
        """Prints performance summary."""
        print(f"\n── Single Asset Strategy Results ───────────────")
        print(f"  Symbol:            {self.metrics['symbol']}")
        print(f"  Initial Capital:   ${self.metrics['initial_capital']:,.2f}")
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

    # ── Step 8 — Plot ────────────────────────────────────────────────────────

    def plot(self):
        """Plots equity curve and drawdown."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Equity curve
        axes[0].plot(
            self.portfolio_df["date"],
            self.portfolio_df["total_value"],
            linewidth=1.5,
            label="Portfolio Value",
            color="steelblue"
        )
        axes[0].axhline(
            self.config.initial_capital,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="Initial Capital"
        )
        axes[0].set_title(f"{self.config.symbol} — Equity Curve")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Daily returns
        returns = self.portfolio_df["total_value"].pct_change()
        axes[1].plot(
            self.portfolio_df["date"],
            returns,
            linewidth=0.8,
            color="orange",
            label="Daily Returns"
        )
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Daily Returns")
        axes[1].set_ylabel("Return")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        pv = self.portfolio_df["total_value"]
        rolling_max = pv.expanding().max()
        drawdown = (pv - rolling_max) / rolling_max

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
        Runs the full single asset strategy pipeline.

        Args:
            plot: whether to show performance chart
        """
        print(f"Running Single Asset Strategy for {self.config.symbol}...")

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