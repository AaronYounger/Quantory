from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal

from .conditions import Condition, LogicType


RebalanceType = Literal["partial", "full"]
RebalanceFrequency = Literal["daily", "weekly", "monthly", "quarterly"]


@dataclass
class QuantileConfig:
    """
    Configuration for a quantile/factor strategy.
    All user inputs live here.
    """
    historical_rankings:  pd.DataFrame        # from screener.historical_rank()
    target_quantile:      str                 # e.g. "Q1"
    rebalance_type:       RebalanceType       # "partial" or "full"
    initial_capital:      float = 100000.0
    commission_pct:       float = 0.001
    commission_fixed:     float = 1.0
    slippage_pct:         float = 0.0005


class QuantileStrategy:

    def __init__(self, df: pd.DataFrame, config: QuantileConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine
            config: QuantileConfig with all user inputs
        """
        self.df = df.copy()
        self.config = config
        self.portfolio_df = None
        self.trades_df = None
        self.metrics = {}

        # Current holdings — symbol → shares
        self.holdings: dict[str, float] = {}
        self.cash = config.initial_capital

    # ── Step 1 — Validate inputs ─────────────────────────────────────────────

    def _validate(self):
        """Checks historical rankings and target quantile are valid."""
        required_cols = ["symbol", "quantile", "rebalance_date"]
        missing = [
            c for c in required_cols
            if c not in self.config.historical_rankings.columns
        ]
        if missing:
            raise ValueError(
                f"historical_rankings missing columns: {missing}. "
                f"Make sure you used screener.historical_rank()"
            )

        available_quantiles = (
            self.config.historical_rankings["quantile"]
            .unique()
            .tolist()
        )
        if self.config.target_quantile not in available_quantiles:
            raise ValueError(
                f"Invalid target quantile '{self.config.target_quantile}'. "
                f"Available quantiles: {available_quantiles}"
            )

        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0.")

    # ── Step 2 — Get price for symbol on date ────────────────────────────────

    def _get_price(self, symbol: str, date, price_col: str = "open") -> float:
        """Gets execution price for a symbol on a given date."""
        mask = (
            (self.df["symbol"] == symbol) &
            (self.df["date"] == date)
        )
        prices = self.df.loc[mask, price_col]
        return prices.iloc[0] if len(prices) > 0 else None

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

    # ── Step 5 — Get target symbols at rebalance date ────────────────────────

    def _get_target_symbols(self, rebalance_date) -> list[str]:
        """Returns symbols in target quantile at this rebalance date."""
        mask = (
            (self.config.historical_rankings["rebalance_date"] == rebalance_date) &
            (self.config.historical_rankings["quantile"] == self.config.target_quantile)
        )
        return (
            self.config.historical_rankings
            .loc[mask, "symbol"]
            .tolist()
        )

    # ── Step 6 — Execute full rebalance ──────────────────────────────────────

    def _full_rebalance(self, target_symbols: list[str], 
                        rebalance_date, trade_records: list):
        """
        Sells ALL current holdings then buys target symbols equally.
        """
        next_open_date = self._get_next_trading_date(rebalance_date)
        if next_open_date is None:
            return

        # Sell everything
        for symbol, shares in list(self.holdings.items()):
            if shares <= 0:
                continue

            sell_price = self._get_price(symbol, next_open_date, "open")
            if sell_price is None:
                continue

            sell_price  = self._calculate_slippage(sell_price, "sell")
            trade_value = shares * sell_price
            commission  = self._calculate_commission(trade_value)
            self.cash  += trade_value - commission

            trade_records.append({
                "date":       next_open_date,
                "symbol":     symbol,
                "type":       "SELL",
                "price":      sell_price,
                "shares":     shares,
                "commission": commission,
                "slippage":   sell_price * self.config.slippage_pct,
                "value":      trade_value,
                "rebalance":  True,
            })

        self.holdings = {}

        # Buy target symbols equally
        if not target_symbols:
            return

        capital_per_symbol = self.cash / len(target_symbols)

        for symbol in target_symbols:
            buy_price = self._get_price(symbol, next_open_date, "open")
            if buy_price is None:
                continue

            buy_price  = self._calculate_slippage(buy_price, "buy")
            commission = self._calculate_commission(capital_per_symbol)
            shares     = (capital_per_symbol - commission) / buy_price
            self.cash -= capital_per_symbol

            self.holdings[symbol] = shares

            trade_records.append({
                "date":       next_open_date,
                "symbol":     symbol,
                "type":       "BUY",
                "price":      buy_price,
                "shares":     shares,
                "commission": commission,
                "slippage":   buy_price * self.config.slippage_pct,
                "value":      shares * buy_price,
                "rebalance":  True,
            })

    # ── Step 7 — Execute partial rebalance ───────────────────────────────────

    def _partial_rebalance(self, target_symbols: list[str],
                           rebalance_date, trade_records: list):
        """
        Only trades stocks that changed.
        Sells dropped stocks, buys new entrants.
        Keeps stocks that remain in target quantile.
        """
        next_open_date = self._get_next_trading_date(rebalance_date)
        if next_open_date is None:
            return

        current_symbols = set(self.holdings.keys())
        target_set      = set(target_symbols)

        # Stocks to sell — dropped out of target quantile
        to_sell = current_symbols - target_set

        # Stocks to buy — new entrants to target quantile
        to_buy = target_set - current_symbols

        # Sell dropped stocks
        sell_proceeds = 0.0
        for symbol in to_sell:
            shares = self.holdings.get(symbol, 0)
            if shares <= 0:
                continue

            sell_price = self._get_price(symbol, next_open_date, "open")
            if sell_price is None:
                continue

            sell_price   = self._calculate_slippage(sell_price, "sell")
            trade_value  = shares * sell_price
            commission   = self._calculate_commission(trade_value)
            proceeds     = trade_value - commission
            self.cash   += proceeds
            sell_proceeds += proceeds

            trade_records.append({
                "date":       next_open_date,
                "symbol":     symbol,
                "type":       "SELL",
                "price":      sell_price,
                "shares":     shares,
                "commission": commission,
                "slippage":   sell_price * self.config.slippage_pct,
                "value":      trade_value,
                "rebalance":  True,
            })

            del self.holdings[symbol]

        # Buy new entrants equally from available cash
        if to_buy and self.cash > 0:
            capital_per_symbol = self.cash / len(to_buy)

            for symbol in to_buy:
                buy_price = self._get_price(symbol, next_open_date, "open")
                if buy_price is None:
                    continue

                buy_price  = self._calculate_slippage(buy_price, "buy")
                commission = self._calculate_commission(capital_per_symbol)
                shares     = (capital_per_symbol - commission) / buy_price
                self.cash -= capital_per_symbol

                self.holdings[symbol] = shares

                trade_records.append({
                    "date":       next_open_date,
                    "symbol":     symbol,
                    "type":       "BUY",
                    "price":      buy_price,
                    "shares":     shares,
                    "commission": commission,
                    "slippage":   buy_price * self.config.slippage_pct,
                    "value":      shares * buy_price,
                    "rebalance":  True,
                })

    # ── Step 8 — Get next trading date ───────────────────────────────────────

    def _get_next_trading_date(self, date):
        """
        Gets the next available trading date after rebalance date.
        This is where trades actually execute (next open).
        """
        valid_dates = sorted(self.df["date"].unique())
        future_dates = [d for d in valid_dates if d > date]
        return future_dates[0] if future_dates else None

    # ── Step 9 — Run backtest ────────────────────────────────────────────────

    def _run_backtest(self):
        """
        Loops through all dates.
        Rebalances at each rebalance date.
        Tracks portfolio value daily.
        """
        all_dates       = sorted(self.df["date"].unique())
        rebalance_dates = sorted(
            self.config.historical_rankings["rebalance_date"].unique()
        )

        trade_records     = []
        portfolio_records = []

        for date in all_dates:

            # Check if this is a rebalance date
            if date in rebalance_dates:
                target_symbols = self._get_target_symbols(date)

                if self.config.rebalance_type == "full":
                    self._full_rebalance(target_symbols, date, trade_records)
                else:
                    self._partial_rebalance(target_symbols, date, trade_records)

            # Track portfolio value
            position_value = 0.0
            per_symbol     = {}

            for symbol, shares in self.holdings.items():
                close_price = self._get_price(symbol, date, "close")
                if close_price is not None:
                    sym_value              = shares * close_price
                    position_value        += sym_value
                    per_symbol[f"{symbol}_value"] = round(sym_value, 2)

            total_value = self.cash + position_value

            portfolio_records.append({
                "date":           date,
                "cash":           round(self.cash, 2),
                "position_value": round(position_value, 2),
                "total_value":    round(total_value, 2),
                "n_holdings":     len(self.holdings),
                **per_symbol
            })

        self.portfolio_df = pd.DataFrame(portfolio_records)
        self.trades_df    = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

    # ── Step 10 — Calculate metrics ──────────────────────────────────────────

    def _calculate_metrics(self):
        """Calculates performance metrics from portfolio history."""
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

        total_trades     = len(self.trades_df[self.trades_df["type"] == "BUY"]) if len(self.trades_df) > 0 else 0
        total_commission = self.trades_df["commission"].sum() if len(self.trades_df) > 0 else 0
        total_slippage   = self.trades_df["slippage"].sum() if len(self.trades_df) > 0 else 0

        # Average holdings per rebalance
        avg_holdings = self.portfolio_df["n_holdings"].mean()

        # Turnover per rebalance
        n_rebalances = len(
            self.config.historical_rankings["rebalance_date"].unique()
        )
        avg_turnover = total_trades / n_rebalances if n_rebalances > 0 else 0

        self.metrics = {
            "target_quantile":    self.config.target_quantile,
            "rebalance_type":     self.config.rebalance_type,
            "initial_capital":    self.config.initial_capital,
            "final_value":        round(pv.iloc[-1], 2),
            "total_return":       round(total_return * 100, 2),
            "annualized_return":  round(annualized_return * 100, 2),
            "annualized_vol":     round(annualized_vol * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "max_drawdown":       round(max_drawdown * 100, 2),
            "total_trades":       total_trades,
            "total_commission":   round(total_commission, 2),
            "total_slippage":     round(total_slippage, 2),
            "avg_holdings":       round(avg_holdings, 1),
            "n_rebalances":       n_rebalances,
            "avg_turnover":       round(avg_turnover, 1),
        }

        return self.metrics

    # ── Step 11 — Print results ──────────────────────────────────────────────

    def _print_results(self):
        """Prints performance summary."""
        print(f"\n── Quantile Strategy Results ───────────────────")
        print(f"  Target Quantile:   {self.metrics['target_quantile']}")
        print(f"  Rebalance Type:    {self.metrics['rebalance_type']}")
        print(f"  Initial Capital:   ${self.metrics['initial_capital']:,.2f}")
        print(f"  Final Value:       ${self.metrics['final_value']:,.2f}")
        print(f"  Total Return:      {self.metrics['total_return']}%")
        print(f"  Annualized Return: {self.metrics['annualized_return']}%")
        print(f"  Annualized Vol:    {self.metrics['annualized_vol']}%")
        print(f"  Sharpe Ratio:      {self.metrics['sharpe_ratio']}")
        print(f"  Sortino Ratio:     {self.metrics['sortino_ratio']}")
        print(f"  Max Drawdown:      {self.metrics['max_drawdown']}%")
        print(f"  Total Trades:      {self.metrics['total_trades']}")
        print(f"  Total Commission:  ${self.metrics['total_commission']:,.2f}")
        print(f"  Total Slippage:    ${self.metrics['total_slippage']:,.2f}")
        print(f"  Avg Holdings:      {self.metrics['avg_holdings']} stocks")
        print(f"  Rebalances:        {self.metrics['n_rebalances']}")
        print(f"  Avg Turnover:      {self.metrics['avg_turnover']} trades/rebalance")

    # ── Step 12 — Plot ───────────────────────────────────────────────────────

    def plot(self):
        """Plots equity curve, holdings count, and drawdown."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Equity curve
        axes[0].plot(
            self.portfolio_df["date"],
            self.portfolio_df["total_value"],
            linewidth=1.5,
            label=f"Quantile Strategy ({self.config.target_quantile})",
            color="steelblue"
        )
        axes[0].axhline(
            self.config.initial_capital,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="Initial Capital"
        )
        axes[0].set_title(
            f"Quantile Strategy — {self.config.target_quantile} "
            f"({self.config.rebalance_type} rebalance)"
        )
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Number of holdings over time
        axes[1].plot(
            self.portfolio_df["date"],
            self.portfolio_df["n_holdings"],
            linewidth=1.2,
            color="green",
            label="Number of Holdings"
        )
        axes[1].set_title("Number of Holdings Over Time")
        axes[1].set_ylabel("Holdings")
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
        Runs the full quantile strategy pipeline.

        Args:
            plot: whether to show performance chart
        """
        print(
            f"Running Quantile Strategy — "
            f"{self.config.target_quantile} "
            f"({self.config.rebalance_type} rebalance)..."
        )

        self._validate()
        self._run_backtest()
        self._calculate_metrics()
        self._print_results()

        if plot:
            self.plot()

        return {
            "metrics":   self.metrics,
            "portfolio": self.portfolio_df,
            "trades":    self.trades_df,
        }