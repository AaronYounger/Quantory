from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Literal, Optional

from Strategy.conditions import Condition, LogicType
from Strategy.signals import build_signals


# ── Transition Type ───────────────────────────────────────────────────────────

TransitionType = Literal["immediate", "gradual"]


# ── Sub Strategy Config ───────────────────────────────────────────────────────

@dataclass
class SubStrategy:
    """
    Defines a strategy for a specific regime.
    Each regime gets its own SubStrategy.
    """
    regime:            str                  # regime label e.g. "Bull"
    strategy_type:     Literal["single_asset", "multi_asset", "hold_cash"]
    entry_conditions:  list[Condition] = field(default_factory=list)
    exit_conditions:   list[Condition] = field(default_factory=list)
    symbols:           list[str] = field(default_factory=list)
    entry_logic:       LogicType = "and"
    exit_logic:        LogicType = "and"

    # hold_cash means do nothing in this regime
    # useful for Bear regime — just exit and hold cash


# ── Regime Strategy Config ────────────────────────────────────────────────────

@dataclass
class RegimeStrategyConfig:
    """
    Configuration for regime switching strategy.
    All user inputs live here.
    """
    regime_col:        str                  # column with regime labels
                                            # e.g. "regime" from HMM
    sub_strategies:    list[SubStrategy]    # one per regime
    initial_capital:   float = 100000.0
    transition_type:   TransitionType = "immediate"
    transition_days:   int = 5              # for gradual transition
                                            # how many days to phase in
    commission_pct:    float = 0.001
    commission_fixed:  float = 1.0
    slippage_pct:      float = 0.0005


# ── Regime Strategy ───────────────────────────────────────────────────────────

class RegimeStrategy:

    def __init__(self, df: pd.DataFrame, config: RegimeStrategyConfig):
        """
        Args:
            df:     enriched dataframe from FeatureEngine + MLEngine
                    must have regime column from HMM
            config: RegimeStrategyConfig with all user inputs
        """
        self.df            = df.copy()
        self.config        = config
        self.portfolio_df  = None
        self.trades_df     = None
        self.regime_perf   = None
        self.metrics       = {}
        self.fitted        = False

        # Map regime label to sub strategy
        self.regime_map = {
            sub.regime: sub
            for sub in config.sub_strategies
        }

    # ── Step 1 — Validate ────────────────────────────────────────────────────

    def _validate(self):
        """Validates all inputs before running."""

        # Regime column check
        if self.config.regime_col not in self.df.columns:
            raise ValueError(
                f"Regime column '{self.config.regime_col}' not found. "
                f"Run MLEngine.run_hmm() first to generate regime labels."
            )

        # Check all regimes in dataset are covered
        dataset_regimes = (
            self.df[self.config.regime_col]
            .dropna()
            .unique()
            .tolist()
        )
        covered_regimes = list(self.regime_map.keys())
        uncovered = [r for r in dataset_regimes if r not in covered_regimes]

        if uncovered:
            raise ValueError(
                f"Regimes found in dataset but not covered by sub strategies: {uncovered}. "
                f"Add a SubStrategy for each regime or ensure HMM labels match."
            )

        # Check symbols exist
        available = self.df["symbol"].unique().tolist()
        for sub in self.config.sub_strategies:
            if sub.strategy_type != "hold_cash":
                invalid = [s for s in sub.symbols if s not in available]
                if invalid:
                    raise ValueError(
                        f"Symbols not found for {sub.regime} strategy: {invalid}. "
                        f"Available: {available}"
                    )

        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0.")

    # ── Step 2 — Build signals per sub strategy ───────────────────────────────

    def _build_all_signals(self):
        """
        Builds entry/exit signals for each sub strategy.
        Returns dict of regime → signals df.
        """
        signals_map = {}

        for sub in self.config.sub_strategies:
            if sub.strategy_type == "hold_cash":
                signals_map[sub.regime] = None
                continue

            try:
                signals_df = build_signals(
                    df=self.df,
                    entry_conditions=sub.entry_conditions,
                    exit_conditions=sub.exit_conditions,
                    selected_symbols=sub.symbols,
                    entry_logic=sub.entry_logic,
                    exit_logic=sub.exit_logic,
                )
                signals_map[sub.regime] = signals_df
                print(f"  ✓ Signals built for {sub.regime} regime")

            except Exception as e:
                print(f"  ✗ Failed to build signals for {sub.regime}: {e}")
                signals_map[sub.regime] = None

        return signals_map

    # ── Step 3 — Commission and slippage ─────────────────────────────────────

    def _calculate_commission(self, trade_value: float) -> float:
        return max(
            self.config.commission_fixed,
            trade_value * self.config.commission_pct
        )

    def _calculate_slippage(self, price: float, direction: str) -> float:
        if direction == "buy":
            return price * (1 + self.config.slippage_pct)
        else:
            return price * (1 - self.config.slippage_pct)

    # ── Step 4 — Get current regime ──────────────────────────────────────────

    def _get_regime(self, date, regime_series: pd.Series):
        """Gets regime label for a specific date."""
        try:
            return regime_series.loc[date]
        except KeyError:
            return None

    # ── Step 5 — Liquidate all positions ─────────────────────────────────────

    def _liquidate(self, holdings: dict, date, trade_records: list) -> float:
        """
        Sells all current holdings.
        Called when regime switches to move to new strategy.
        Returns cash proceeds.
        """
        total_proceeds = 0.0

        for symbol, shares in list(holdings.items()):
            if shares <= 0:
                continue

            # Get next day open price
            mask        = (self.df["symbol"] == symbol) & (self.df["date"] == date)
            price_rows  = self.df[mask]["open"]

            if len(price_rows) == 0:
                continue

            sell_price  = self._calculate_slippage(price_rows.iloc[0], "sell")
            trade_value = shares * sell_price
            commission  = self._calculate_commission(trade_value)
            proceeds    = trade_value - commission
            total_proceeds += proceeds

            trade_records.append({
                "date":       date,
                "symbol":     symbol,
                "type":       "SELL",
                "reason":     "regime_switch",
                "price":      sell_price,
                "shares":     shares,
                "commission": commission,
                "value":      trade_value,
            })

        holdings.clear()
        return total_proceeds

    # ── Step 6 — Execute sub strategy for one day ────────────────────────────

    def _execute_day(self,
                     date,
                     regime: str,
                     signals_map: dict,
                     holdings: dict,
                     cash: float,
                     trade_records: list):
        """
        Executes the appropriate sub strategy for a given date and regime.
        Returns updated cash and holdings.
        """
        sub = self.regime_map.get(regime)

        if sub is None or sub.strategy_type == "hold_cash":
            return cash, holdings

        signals_df = signals_map.get(regime)
        if signals_df is None:
            return cash, holdings

        # Filter signals to this date
        day_signals = signals_df[signals_df["date"] == date]

        if day_signals.empty:
            return cash, holdings

        for _, row in day_signals.iterrows():
            symbol     = row["symbol"]
            exec_price = row.get("execution_price", None)

            if pd.isna(exec_price) or exec_price is None:
                continue

            # Entry
            if row["entry_signal"] and symbol not in holdings:
                n_symbols  = len(sub.symbols)
                allocation = cash / max(n_symbols, 1)

                if allocation <= 0:
                    continue

                buy_price  = self._calculate_slippage(exec_price, "buy")
                commission = self._calculate_commission(allocation)
                shares     = (allocation - commission) / buy_price
                cash      -= allocation

                holdings[symbol] = shares

                trade_records.append({
                    "date":       date,
                    "symbol":     symbol,
                    "type":       "BUY",
                    "reason":     f"{regime}_strategy",
                    "price":      buy_price,
                    "shares":     shares,
                    "commission": commission,
                    "value":      shares * buy_price,
                })

            # Exit
            elif row["exit_signal"] and symbol in holdings:
                shares     = holdings[symbol]
                sell_price = self._calculate_slippage(exec_price, "sell")
                trade_value = shares * sell_price
                commission  = self._calculate_commission(trade_value)
                cash       += trade_value - commission

                trade_records.append({
                    "date":       date,
                    "symbol":     symbol,
                    "type":       "SELL",
                    "reason":     f"{regime}_strategy",
                    "price":      sell_price,
                    "shares":     shares,
                    "commission": commission,
                    "value":      trade_value,
                })

                del holdings[symbol]

        return cash, holdings

    # ── Step 7 — Run backtest ────────────────────────────────────────────────

    def _run_backtest(self, signals_map: dict):
        """
        Main backtest loop.
        Detects regime each day and routes to correct sub strategy.
        Liquidates and switches when regime changes.
        """
        all_dates    = sorted(self.df["date"].unique())
        regime_series = (
            self.df[["date", self.config.regime_col]]
            .drop_duplicates("date")
            .set_index("date")[self.config.regime_col]
        )

        cash             = self.config.initial_capital
        holdings         = {}
        current_regime   = None
        portfolio_records = []
        trade_records    = []
        regime_days      = {}

        for date in all_dates:
            regime = self._get_regime(date, regime_series)

            if regime is None:
                continue

            # ── Regime switch detected ────────────────────────────────────
            if regime != current_regime:
                if current_regime is not None:
                    print(
                        f"  Regime switch: {current_regime} → {regime} "
                        f"on {date}"
                    )

                    # Liquidate current positions
                    if self.config.transition_type == "immediate":
                        proceeds = self._liquidate(holdings, date, trade_records)
                        cash    += proceeds

                current_regime = regime

            # Track days per regime
            regime_days[regime] = regime_days.get(regime, 0) + 1

            # ── Execute sub strategy ──────────────────────────────────────
            cash, holdings = self._execute_day(
                date, regime, signals_map,
                holdings, cash, trade_records
            )

            # ── Track portfolio value ─────────────────────────────────────
            position_value = 0.0
            for symbol, shares in holdings.items():
                mask        = (
                    (self.df["symbol"] == symbol) &
                    (self.df["date"]   == date)
                )
                close_rows  = self.df[mask]["close"]
                if len(close_rows) > 0:
                    position_value += shares * close_rows.iloc[0]

            total_value = cash + position_value

            portfolio_records.append({
                "date":           date,
                "regime":         regime,
                "cash":           round(cash, 2),
                "position_value": round(position_value, 2),
                "total_value":    round(total_value, 2),
                "n_holdings":     len(holdings),
            })

        self.portfolio_df = pd.DataFrame(portfolio_records)
        self.trades_df    = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        print(f"\n── Regime Day Counts ───────────────────────────")
        for regime, days in regime_days.items():
            pct = days / len(all_dates) * 100
            print(f"  {regime:15} → {days} days ({pct:.1f}%)")

    # ── Step 8 — Calculate metrics ────────────────────────────────────────────

    def _calculate_metrics(self):
        """
        Calculates overall and per regime performance metrics.
        """
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

        # Per regime metrics
        regime_metrics = []
        for regime in self.portfolio_df["regime"].unique():
            regime_pv = self.portfolio_df[
                self.portfolio_df["regime"] == regime
            ]["total_value"]

            if len(regime_pv) < 2:
                continue

            r_returns  = regime_pv.pct_change().dropna()
            r_total    = (regime_pv.iloc[-1] - regime_pv.iloc[0]) / regime_pv.iloc[0]
            r_vol      = r_returns.std() * np.sqrt(252)
            r_sharpe   = (r_returns.mean() * 252) / r_vol if r_vol != 0 else 0
            r_max_dd   = ((regime_pv - regime_pv.expanding().max()) / regime_pv.expanding().max()).min()

            regime_metrics.append({
                "regime":       regime,
                "days":         len(regime_pv),
                "total_return": round(r_total * 100, 2),
                "annualized_vol": round(r_vol * 100, 2),
                "sharpe":       round(r_sharpe, 3),
                "max_drawdown": round(r_max_dd * 100, 2),
            })

        self.regime_perf = pd.DataFrame(regime_metrics)

        self.metrics = {
            "initial_capital":    self.config.initial_capital,
            "final_value":        round(pv.iloc[-1], 2),
            "total_return":       round(total_return * 100, 2),
            "annualized_return":  round(annualized_return * 100, 2),
            "annualized_vol":     round(annualized_vol * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "max_drawdown":       round(max_drawdown * 100, 2),
            "total_trades":       len(self.trades_df),
            "regime_switches":    len(self.trades_df[
                self.trades_df["reason"] == "regime_switch"
            ]) if len(self.trades_df) > 0 else 0,
        }

        return self.metrics

    # ── Step 9 — Print results ────────────────────────────────────────────────

    def _print_results(self):
        """Prints overall and per regime performance."""

        print(f"\n── Regime Strategy Results ─────────────────────")
        print(f"  Initial Capital:   ${self.metrics['initial_capital']:,.2f}")
        print(f"  Final Value:       ${self.metrics['final_value']:,.2f}")
        print(f"  Total Return:      {self.metrics['total_return']}%")
        print(f"  Annualized Return: {self.metrics['annualized_return']}%")
        print(f"  Annualized Vol:    {self.metrics['annualized_vol']}%")
        print(f"  Sharpe Ratio:      {self.metrics['sharpe_ratio']}")
        print(f"  Sortino Ratio:     {self.metrics['sortino_ratio']}")
        print(f"  Max Drawdown:      {self.metrics['max_drawdown']}%")
        print(f"  Total Trades:      {self.metrics['total_trades']}")
        print(f"  Regime Switches:   {self.metrics['regime_switches']}")

        print(f"\n── Per Regime Performance ──────────────────────")
        print(self.regime_perf.to_string(index=False))

    # ── Step 10 — Plot ────────────────────────────────────────────────────────

    def plot(self):
        """
        Plots equity curve with regime overlay shading.
        Similar to HMM regime overlay chart.
        Also shows per regime performance breakdown.
        """
        if not self.fitted:
            raise ValueError("Run fit() first.")

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Regime color map
        regimes = self.portfolio_df["regime"].unique()
        colors  = ["green", "orange", "red", "blue", "purple"]
        regime_colors = {
            r: colors[i % len(colors)]
            for i, r in enumerate(regimes)
        }

        dates  = self.portfolio_df["date"]
        pv     = self.portfolio_df["total_value"]
        y_min  = pv.min()
        y_max  = pv.max()

        # ── Equity curve with regime overlay ─────────────────────────────
        axes[0].plot(dates, pv, color="black", linewidth=1.5,
                     label="Portfolio Value", zorder=5)
        axes[0].axhline(
            self.config.initial_capital,
            color="black", linewidth=0.8,
            linestyle="--", label="Initial Capital"
        )

        for regime, color in regime_colors.items():
            mask = self.portfolio_df["regime"] == regime
            axes[0].fill_between(
                dates, y_min, y_max,
                where=mask,
                color=color, alpha=0.15,
                label=regime
            )

        axes[0].set_title("Regime Switching Strategy — Equity Curve")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # ── Drawdown ──────────────────────────────────────────────────────
        rolling_max = pv.expanding().max()
        drawdown    = (pv - rolling_max) / rolling_max

        axes[1].fill_between(dates, drawdown, 0,
                              color="red", alpha=0.4, label="Drawdown")

        for regime, color in regime_colors.items():
            mask = self.portfolio_df["regime"] == regime
            axes[1].fill_between(
                dates, drawdown.min(), 0,
                where=mask,
                color=color, alpha=0.08
            )

        axes[1].set_title("Drawdown with Regime Overlay")
        axes[1].set_ylabel("Drawdown %")
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)

        # ── Per regime Sharpe bar chart ───────────────────────────────────
        regime_colors_list = [
            regime_colors.get(r, "steelblue")
            for r in self.regime_perf["regime"]
        ]

        axes[2].bar(
            self.regime_perf["regime"],
            self.regime_perf["sharpe"],
            color=regime_colors_list,
            edgecolor="white"
        )
        axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[2].set_title("Sharpe Ratio per Regime")
        axes[2].set_ylabel("Sharpe Ratio")
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    # ── Master run method ─────────────────────────────────────────────────────

    def run(self, plot: bool = True):
        """
        Runs the full regime switching strategy pipeline.

        Args:
            plot: whether to show performance chart
        """
        print(f"\n── Regime Switching Strategy ───────────────────")
        print(f"  Regime Column:    {self.config.regime_col}")
        print(f"  Sub Strategies:   {list(self.regime_map.keys())}")
        print(f"  Transition Type:  {self.config.transition_type}")
        print(f"  Initial Capital:  ${self.config.initial_capital:,.2f}")

        self._validate()

        print(f"\n  Building signals per sub strategy...")
        signals_map = self._build_all_signals()

        print(f"\n  Running backtest...")
        self._run_backtest(signals_map)

        self._calculate_metrics()
        self._print_results()

        self.fitted = True

        if plot:
            self.plot()

        return {
            "metrics":    self.metrics,
            "portfolio":  self.portfolio_df,
            "trades":     self.trades_df,
            "regime_perf": self.regime_perf,
        }