import pandas as pd
from datetime import datetime
from Features.transforms import (
    PrerequisiteMetrics,
    MarketFeatures,
    SectorFeatures,
    FundamentalFeatures,
    TechnicalFeatures,
    StatisticalFeatures,
    TargetFeatures,
)

class FeatureEngine:

    def __init__(self, df: pd.DataFrame):
        self.df          = df.copy()
        self.feature_log = []
        self._init_classes()

    def _init_classes(self):
        self.prerequisite = PrerequisiteMetrics(self.df)
        self.market       = MarketFeatures(self.df)
        self.sector       = SectorFeatures(self.df)
        self.fundamental  = FundamentalFeatures(self.df)
        self.technical    = TechnicalFeatures(self.df)
        self.statistical  = StatisticalFeatures(self.df)
        self.targets      = TargetFeatures(self.df)

    def _sync(self, updated_df: pd.DataFrame, level: str, feature: str, parameters: dict):
        self.df = updated_df.copy()
        self._init_classes()
        self.feature_log.append({
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "level":      level,
            "feature":    feature,
            "parameters": parameters,
        })

    def get_feature_log(self):
        if not self.feature_log:
            print("No features built yet.")
            return pd.DataFrame()
        log_df = pd.DataFrame(self.feature_log)
        print("\n── Feature Log ─────────────────────────────────")
        print(log_df.to_string(index=False))
        return log_df

    def get_columns(self):
        base_cols = [
            "date", "symbol", "open", "high",
            "low", "close", "volume", "asset_type",
            "sector", "industry", "market_cap",
            "country", "exchange", "employees",
            "currency", "shares_outstanding", "dividend_yield",
            "market_cap_bucket"
        ]
        built_cols = [c for c in self.df.columns if c not in base_cols]
        print("\n── Base Columns ────────────────────────────────")
        print(base_cols)
        print("\n── User Built Columns ──────────────────────────")
        if built_cols:
            for col in built_cols:
                print(f"  → {col}")
        else:
            print("  None yet.")
        return built_cols

    def get_df(self):
        return self.df.copy()

    # ── Prerequisite methods ──────────────────────────────────────────────────

    def add_return(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_return(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_return",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_log_return(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_log_return(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_log_return",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_cumulative_return(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_cumulative_return(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_cumulative_return",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_cumprod_return(self, col_name: str, return_col: str):
        updated_df = self.prerequisite.add_cumprod_return(col_name, return_col)
        self._sync(updated_df, "Prerequisite", "add_cumprod_return",
                   {"col_name": col_name, "return_col": return_col})
        return self.df

    def add_rolling_volatility(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_rolling_volatility(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_rolling_volatility",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_static_volatility(self, col_name: str):
        updated_df = self.prerequisite.add_static_volatility(col_name)
        self._sync(updated_df, "Prerequisite", "add_static_volatility",
                   {"col_name": col_name})
        return self.df

    def add_relative_volume(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_relative_volume(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_relative_volume",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_volume_zscore(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_volume_zscore(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_volume_zscore",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_vwap(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_vwap(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_vwap",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_rolling_high(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_rolling_high(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_rolling_high",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_rolling_low(self, col_name: str, window: int):
        updated_df = self.prerequisite.add_rolling_low(col_name, window)
        self._sync(updated_df, "Prerequisite", "add_rolling_low",
                   {"col_name": col_name, "window": window})
        return self.df

    # ── Market methods ────────────────────────────────────────────────────────

    def add_benchmark(self, benchmark: str, provider: str = "yfinance"):
        updated_df = self.market.add_benchmark(benchmark, provider)
        self._sync(updated_df, "Market", "add_benchmark",
                   {"benchmark": benchmark})
        return self.df

    def select_benchmark(self, benchmark: str):
        return self.market.select_benchmark(benchmark)

    def get_market_snapshot(self, table_metrics: list):
        return self.market.get_snapshot(table_metrics)

    def plot_market(self, plot_metrics: list):
        return self.market.plot(plot_metrics)

    def generate_market(self, table_metrics: list, plot_metrics: list):
        return self.market.generate(table_metrics, plot_metrics)

    # ── Sector methods ────────────────────────────────────────────────────────

    def add_sectors(self, sectors: list, provider: str = "yfinance"):
        updated_df = self.sector.add_sectors(sectors, provider)
        self._sync(updated_df, "Sector", "add_sectors",
                   {"sectors": sectors})
        return self.df

    def select_sectors(self, sectors: list):
        return self.sector.select_sectors(sectors)

    def get_sector_snapshot(self, table_metrics: list):
        return self.sector.get_snapshot(table_metrics)

    def plot_sector(self, plot_metrics: list):
        return self.sector.plot(plot_metrics)

    def generate_sector(self, table_metrics: list, plot_metrics: list):
        return self.sector.generate(table_metrics, plot_metrics)

    # ── Fundamental methods ───────────────────────────────────────────────────

    def fetch_fundamentals(self, metrics: list):
        self.fundamental.fetch(metrics)
        return self.fundamental.fundamental_df

    def fetch_fundamentals_by_category(self, categories: list):
        self.fundamental.fetch_by_category(categories)
        return self.fundamental.fundamental_df

    def get_fundamental_snapshot(self, metrics: list = None):
        return self.fundamental.get_snapshot(metrics)

    def rank_by_fundamental(self, metric: str, ascending: bool = False):
        return self.fundamental.rank_by(metric, ascending)

    def plot_fundamental(self, metric: str):
        return self.fundamental.plot(metric)

    def generate_fundamental(self, table_metrics: list, plot_metric: str):
        return self.fundamental.generate(table_metrics, plot_metric)

    # ── Technical methods ─────────────────────────────────────────────────────

    def add_sma(self, col_name: str, window: int):
        updated_df = self.technical.add_sma(col_name, window)
        self._sync(updated_df, "Technical", "add_sma",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_ema(self, col_name: str, window: int):
        updated_df = self.technical.add_ema(col_name, window)
        self._sync(updated_df, "Technical", "add_ema",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_macd(self, col_name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        updated_df = self.technical.add_macd(col_name, fast, slow, signal)
        self._sync(updated_df, "Technical", "add_macd",
                   {"col_name": col_name, "fast": fast, "slow": slow, "signal": signal})
        return self.df

    def add_adx(self, col_name: str, window: int = 14):
        updated_df = self.technical.add_adx(col_name, window)
        self._sync(updated_df, "Technical", "add_adx",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_rsi(self, col_name: str, window: int = 14):
        updated_df = self.technical.add_rsi(col_name, window)
        self._sync(updated_df, "Technical", "add_rsi",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_roc(self, col_name: str, window: int = 10):
        updated_df = self.technical.add_roc(col_name, window)
        self._sync(updated_df, "Technical", "add_roc",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_stochastic(self, col_name: str, window: int = 14):
        updated_df = self.technical.add_stochastic(col_name, window)
        self._sync(updated_df, "Technical", "add_stochastic",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_cci(self, col_name: str, window: int = 20):
        updated_df = self.technical.add_cci(col_name, window)
        self._sync(updated_df, "Technical", "add_cci",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_williams_r(self, col_name: str, window: int = 14):
        updated_df = self.technical.add_williams_r(col_name, window)
        self._sync(updated_df, "Technical", "add_williams_r",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_atr(self, col_name: str, window: int = 14):
        updated_df = self.technical.add_atr(col_name, window)
        self._sync(updated_df, "Technical", "add_atr",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_bollinger_bands(self, col_name: str, window: int = 20):
        updated_df = self.technical.add_bollinger_bands(col_name, window)
        self._sync(updated_df, "Technical", "add_bollinger_bands",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_obv(self, col_name: str):
        updated_df = self.technical.add_obv(col_name)
        self._sync(updated_df, "Technical", "add_obv",
                   {"col_name": col_name})
        return self.df

    def add_cmf(self, col_name: str, window: int = 20):
        updated_df = self.technical.add_cmf(col_name, window)
        self._sync(updated_df, "Technical", "add_cmf",
                   {"col_name": col_name, "window": window})
        return self.df

    # ── Statistical methods ───────────────────────────────────────────────────

    def add_rolling_correlation(self, col_name: str, window: int, benchmark: str):
        updated_df = self.statistical.add_rolling_correlation(col_name, window, benchmark)
        self._sync(updated_df, "Statistical", "add_rolling_correlation",
                   {"col_name": col_name, "window": window, "benchmark": benchmark})
        return self.df

    def add_rolling_beta(self, col_name: str, window: int, benchmark: str):
        updated_df = self.statistical.add_rolling_beta(col_name, window, benchmark)
        self._sync(updated_df, "Statistical", "add_rolling_beta",
                   {"col_name": col_name, "window": window, "benchmark": benchmark})
        return self.df

    def add_rolling_alpha(self, col_name: str, window: int, benchmark: str, risk_free_rate: float = 0.05):
        updated_df = self.statistical.add_rolling_alpha(col_name, window, benchmark, risk_free_rate)
        self._sync(updated_df, "Statistical", "add_rolling_alpha",
                   {"col_name": col_name, "window": window, "benchmark": benchmark,
                    "risk_free_rate": risk_free_rate})
        return self.df

    def add_rolling_skewness(self, col_name: str, window: int):
        updated_df = self.statistical.add_rolling_skewness(col_name, window)
        self._sync(updated_df, "Statistical", "add_rolling_skewness",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_rolling_kurtosis(self, col_name: str, window: int):
        updated_df = self.statistical.add_rolling_kurtosis(col_name, window)
        self._sync(updated_df, "Statistical", "add_rolling_kurtosis",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_rolling_sharpe(self, col_name: str, window: int, risk_free_rate: float = 0.05):
        updated_df = self.statistical.add_rolling_sharpe(col_name, window, risk_free_rate)
        self._sync(updated_df, "Statistical", "add_rolling_sharpe",
                   {"col_name": col_name, "window": window, "risk_free_rate": risk_free_rate})
        return self.df

    def add_rolling_sortino(self, col_name: str, window: int, risk_free_rate: float = 0.05):
        updated_df = self.statistical.add_rolling_sortino(col_name, window, risk_free_rate)
        self._sync(updated_df, "Statistical", "add_rolling_sortino",
                   {"col_name": col_name, "window": window, "risk_free_rate": risk_free_rate})
        return self.df

    def add_max_drawdown(self, col_name: str, window: int):
        updated_df = self.statistical.add_max_drawdown(col_name, window)
        self._sync(updated_df, "Statistical", "add_max_drawdown",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_current_drawdown(self, col_name: str, window: int):
        updated_df = self.statistical.add_current_drawdown(col_name, window)
        self._sync(updated_df, "Statistical", "add_current_drawdown",
                   {"col_name": col_name, "window": window})
        return self.df

    # ── Target methods ────────────────────────────────────────────────────────

    def add_forward_return(self, col_name: str, window: int):
        updated_df = self.targets.add_forward_return(col_name, window)
        self._sync(updated_df, "Target", "add_forward_return",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_forward_volatility(self, col_name: str, window: int):
        updated_df = self.targets.add_forward_volatility(col_name, window)
        self._sync(updated_df, "Target", "add_forward_volatility",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_forward_drawdown(self, col_name: str, window: int):
        updated_df = self.targets.add_forward_drawdown(col_name, window)
        self._sync(updated_df, "Target", "add_forward_drawdown",
                   {"col_name": col_name, "window": window})
        return self.df

    def add_forward_column(self, col_name: str, source_col: str, window: int):
        updated_df = self.targets.add_forward_column(col_name, source_col, window)
        self._sync(updated_df, "Target", "add_forward_column",
                   {"col_name": col_name, "source_col": source_col, "window": window})
        return self.df