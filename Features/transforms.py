import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openbb import obb
obb.user.preferences.output_type = "dataframe"
from pykalman import KalmanFilter


# ── Prerequisite Metrics ──────────────────────────────────────────────────────

class PrerequisiteMetrics:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ── Price Based ───────────────────────────────────────────────────────────

    def add_return(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .pct_change(window)
        )
        return self.df

    def add_log_return(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: np.log(x).diff(window))
        )
        return self.df

    def add_cumulative_return(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: x / x.shift(window) - 1)
        )
        return self.df

    def add_cumprod_return(self, col_name: str, return_col: str):
        if return_col not in self.df.columns:
            available = [
                col for col in self.df.columns
                if "return" in col.lower()
            ]
            raise ValueError(
                f"Column '{return_col}' not found. "
                f"Available return columns: {available}"
            )
        self.df[col_name] = (
            self.df.groupby("symbol")[return_col]
            .transform(lambda x: (1 + x).cumprod() - 1)
        )
        return self.df

    # ── Volatility Based ──────────────────────────────────────────────────────

    def add_rolling_volatility(self, col_name: str, window: int):
        daily_returns = (
            self.df.groupby("symbol")["close"]
            .pct_change()
        )
        self.df[col_name] = (
            daily_returns
            .groupby(self.df["symbol"])
            .transform(lambda x: x.rolling(window).std() * np.sqrt(252))
        )
        return self.df

    def add_static_volatility(self, col_name: str):
        daily_returns = (
            self.df.groupby("symbol")["close"]
            .pct_change()
        )
        self.df[col_name] = (
            daily_returns
            .groupby(self.df["symbol"])
            .transform(lambda x: x.std() * np.sqrt(252))
        )
        return self.df

    # ── Volume Based ──────────────────────────────────────────────────────────

    def add_relative_volume(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["volume"]
            .transform(lambda x: x / x.rolling(window).mean())
        )
        return self.df

    def add_volume_zscore(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["volume"]
            .transform(
                lambda x: (x - x.rolling(window).mean()) /
                x.rolling(window).std()
            )
        )
        return self.df

    # ── Price Reference ───────────────────────────────────────────────────────

    def add_vwap(self, col_name: str, window: int):
        def _vwap(group):
            typical_price = (
                group["high"] + group["low"] + group["close"]
            ) / 3
            vwap = (
                (typical_price * group["volume"]).rolling(window).sum() /
                group["volume"].rolling(window).sum()
            )
            return vwap

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_vwap)
        )
        return self.df

    def add_rolling_high(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["high"]
            .transform(lambda x: x.rolling(window).max())
        )
        return self.df

    def add_rolling_low(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["low"]
            .transform(lambda x: x.rolling(window).min())
        )
        return self.df


# ── Market Features ───────────────────────────────────────────────────────────

class MarketFeatures:

    VALID_BENCHMARKS = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "DIA": "Dow Jones Industrial Average"
    }

    def __init__(self, df: pd.DataFrame):
        self.df               = df.copy()
        self.benchmark_df     = None
        self.benchmark        = None
        self.available_metrics = []

    def list_benchmarks(self):
        print("Available Benchmarks:")
        print("─" * 30)
        for ticker, name in self.VALID_BENCHMARKS.items():
            already_loaded = ticker in self.df["symbol"].values
            status = "✓ loaded" if already_loaded else "not loaded"
            print(f"  {ticker} — {name} ({status})")
        return self.VALID_BENCHMARKS

    def add_benchmark(self, benchmark: str, provider: str = "yfinance"):
        from Data.Get_Data import fetch_symbols

        if benchmark not in self.VALID_BENCHMARKS:
            valid = ", ".join(self.VALID_BENCHMARKS.keys())
            raise ValueError(
                f"Invalid benchmark '{benchmark}'. "
                f"Valid options are: {valid}"
            )

        if benchmark in self.df["symbol"].values:
            print(f"{benchmark} already in dataset, skipping download.")
            return self.df

        print(f"Adding benchmark: {self.VALID_BENCHMARKS[benchmark]} ({benchmark})")

        start_date = self.df["date"].min().strftime("%Y-%m-%d")
        end_date   = self.df["date"].max().strftime("%Y-%m-%d")

        benchmark_df = fetch_symbols([benchmark], start_date, end_date, provider)
        benchmark_df["asset_type"] = "benchmark"

        self.df = pd.concat([self.df, benchmark_df], ignore_index=True)
        self.df = self.df.sort_values(["symbol", "date"]).reset_index(drop=True)

        print(f"✓ {benchmark} added successfully.")
        return self.df

    def select_benchmark(self, benchmark: str):
        if benchmark not in self.VALID_BENCHMARKS:
            valid = ", ".join(self.VALID_BENCHMARKS.keys())
            raise ValueError(
                f"Invalid benchmark '{benchmark}'. "
                f"Valid options are: {valid}"
            )

        if benchmark not in self.df["symbol"].values:
            raise ValueError(
                f"{benchmark} not in dataset. "
                f"Call add_benchmark('{benchmark}') first."
            )

        self.benchmark    = benchmark
        self.benchmark_df = self.df[self.df["symbol"] == benchmark].copy()

        base_cols = [
            "date", "symbol", "open", "high",
            "low", "close", "volume", "asset_type"
        ]
        self.available_metrics = [
            col for col in self.benchmark_df.columns
            if col not in base_cols
        ]

        print(f"Benchmark selected: {self.VALID_BENCHMARKS[benchmark]} ({benchmark})")
        print(f"Available metrics: {self.available_metrics}")
        return self.benchmark_df

    def get_available_metrics(self):
        self._check_benchmark_selected()
        if not self.available_metrics:
            print(
                "No metrics available yet. "
                "Build metrics first in PrerequisiteMetrics."
            )
        else:
            print("Available metrics to select:")
            for m in self.available_metrics:
                print(f"  → {m}")
        return self.available_metrics

    def get_snapshot(self, table_metrics: list):
        self._check_benchmark_selected()
        self._validate_metrics(table_metrics)
        snapshot = (
            self.benchmark_df[["date"] + table_metrics]
            .iloc[-1]
        )
        return pd.DataFrame(snapshot).T.reset_index(drop=True)

    def plot(self, plot_metrics: list):
        self._check_benchmark_selected()
        self._validate_metrics(plot_metrics)

        ts = self.benchmark_df[["date"] + plot_metrics].copy()
        n  = len(plot_metrics)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, metric in zip(axes, plot_metrics):
            ax.plot(ts["date"], ts[metric], linewidth=1.5, label=metric)
            ax.set_title(
                f"{self.VALID_BENCHMARKS[self.benchmark]} "
                f"({self.benchmark}) — {metric}"
            )
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.legend(loc="upper left")

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    def generate(self, table_metrics: list, plot_metrics: list):
        self._check_benchmark_selected()
        print(f"\n── Snapshot Table ──────────────────────────────")
        snapshot = self.get_snapshot(table_metrics)
        print(snapshot.to_string(index=False))
        print(f"\n── Time Series Graph ───────────────────────────")
        self.plot(plot_metrics)
        return snapshot

    def _check_benchmark_selected(self):
        if self.benchmark_df is None:
            raise ValueError(
                "No benchmark selected. "
                "Call select_benchmark() first."
            )

    def _validate_metrics(self, metrics: list):
        missing = [
            m for m in metrics
            if m not in self.benchmark_df.columns
        ]
        if missing:
            raise ValueError(
                f"Metrics not found: {missing}. "
                f"Available metrics: {self.available_metrics}"
            )


# ── Sector Features ───────────────────────────────────────────────────────────

class SectorFeatures:

    SECTOR_ETFS = {
        "XLK":  "Technology",
        "XLF":  "Financials",
        "XLE":  "Energy",
        "XLV":  "Healthcare",
        "XLI":  "Industrials",
        "XLY":  "Consumer Discretionary",
        "XLP":  "Consumer Staples",
        "XLB":  "Materials",
        "XLU":  "Utilities",
        "XLRE": "Real Estate",
        "XLC":  "Communication Services"
    }

    def __init__(self, df: pd.DataFrame):
        self.df               = df.copy()
        self.sector_df        = None
        self.selected_sectors = []
        self.available_metrics = []

    def list_sectors(self):
        print("Available Sectors:")
        print("─" * 40)
        for ticker, name in self.SECTOR_ETFS.items():
            already_loaded = ticker in self.df["symbol"].values
            status = "✓ loaded" if already_loaded else "not loaded"
            print(f"  {ticker} — {name} ({status})")
        return self.SECTOR_ETFS

    def add_sectors(self, sectors: list, provider: str = "yfinance"):
        from Data.Get_Data import fetch_symbols

        invalid = [s for s in sectors if s not in self.SECTOR_ETFS]
        if invalid:
            valid = ", ".join(self.SECTOR_ETFS.keys())
            raise ValueError(
                f"Invalid sectors: {invalid}. "
                f"Valid options are: {valid}"
            )

        to_fetch       = [s for s in sectors if s not in self.df["symbol"].values]
        already_loaded = [s for s in sectors if s in self.df["symbol"].values]

        if already_loaded:
            print(f"Already loaded: {already_loaded}, skipping.")

        if not to_fetch:
            print("All selected sectors already in dataset.")
            return self.df

        start_date = self.df["date"].min().strftime("%Y-%m-%d")
        end_date   = self.df["date"].max().strftime("%Y-%m-%d")

        sector_df = fetch_symbols(to_fetch, start_date, end_date, provider)
        sector_df["asset_type"] = "sector_etf"

        self.df = pd.concat([self.df, sector_df], ignore_index=True)
        self.df = self.df.sort_values(["symbol", "date"]).reset_index(drop=True)

        print(f"✓ Sectors added: {to_fetch}")
        return self.df

    def select_sectors(self, sectors: list):
        not_loaded = [s for s in sectors if s not in self.df["symbol"].values]
        if not_loaded:
            raise ValueError(
                f"Sectors not in dataset: {not_loaded}. "
                f"Call add_sectors({not_loaded}) first."
            )

        self.selected_sectors = sectors
        self.sector_df = self.df[self.df["symbol"].isin(sectors)].copy()

        base_cols = [
            "date", "symbol", "open", "high",
            "low", "close", "volume", "asset_type"
        ]
        self.available_metrics = [
            col for col in self.sector_df.columns
            if col not in base_cols
        ]

        print(f"Selected sectors: {sectors}")
        print(f"Available metrics: {self.available_metrics}")
        return self.sector_df

    def get_available_metrics(self):
        self._check_sectors_selected()
        if not self.available_metrics:
            print(
                "No metrics available yet. "
                "Build metrics first in PrerequisiteMetrics."
            )
        else:
            print("Available metrics to select:")
            for m in self.available_metrics:
                print(f"  → {m}")
        return self.available_metrics

    def get_snapshot(self, table_metrics: list):
        self._check_sectors_selected()
        self._validate_metrics(table_metrics)

        rows = []
        for sector in self.selected_sectors:
            sector_data = self.sector_df[self.sector_df["symbol"] == sector]
            row = {"sector": self.SECTOR_ETFS[sector], "ticker": sector}
            for metric in table_metrics:
                row[metric] = sector_data[metric].iloc[-1]
            rows.append(row)

        snapshot = pd.DataFrame(rows)
        print("\n── Sector Snapshot ─────────────────────────────")
        print(snapshot.to_string(index=False))
        return snapshot

    def plot(self, plot_metrics: list):
        self._check_sectors_selected()
        self._validate_metrics(plot_metrics)

        n = len(plot_metrics)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, metric in zip(axes, plot_metrics):
            for sector in self.selected_sectors:
                sector_data = self.sector_df[self.sector_df["symbol"] == sector]
                ax.plot(
                    sector_data["date"],
                    sector_data[metric],
                    linewidth=1.5,
                    label=f"{self.SECTOR_ETFS[sector]} ({sector})"
                )
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.legend(loc="upper left")

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    def generate(self, table_metrics: list, plot_metrics: list):
        self._check_sectors_selected()
        snapshot = self.get_snapshot(table_metrics)
        print("\n── Sector Time Series ──────────────────────────")
        self.plot(plot_metrics)
        return snapshot

    def _check_sectors_selected(self):
        if self.sector_df is None:
            raise ValueError(
                "No sectors selected. "
                "Call select_sectors() first."
            )

    def _validate_metrics(self, metrics: list):
        missing = [
            m for m in metrics
            if m not in self.sector_df.columns
        ]
        if missing:
            raise ValueError(
                f"Metrics not found: {missing}. "
                f"Available metrics: {self.available_metrics}"
            )


# ── Fundamental Features ──────────────────────────────────────────────────────

class FundamentalFeatures:

    FUNDAMENTAL_METRICS = {
        "Valuation": {
            "pe_ratio":              "Price to Earnings",
            "forward_pe":            "Forward P/E",
            "peg_ratio":             "PEG Ratio",
            "price_to_book":         "Price to Book",
            "enterprise_to_ebitda":  "EV to EBITDA",
            "enterprise_to_revenue": "EV to Revenue",
        },
        "Profitability": {
            "gross_margin":      "Gross Margin",
            "operating_margin":  "Operating Margin",
            "ebitda_margin":     "EBITDA Margin",
            "profit_margin":     "Profit Margin",
            "return_on_assets":  "Return on Assets",
            "return_on_equity":  "Return on Equity",
        },
        "Growth": {
            "earnings_growth":           "Earnings Growth",
            "earnings_growth_quarterly": "Quarterly Earnings Growth",
            "revenue_growth":            "Revenue Growth",
        },
        "Financial Health": {
            "debt_to_equity": "Debt to Equity",
            "current_ratio":  "Current Ratio",
            "quick_ratio":    "Quick Ratio",
        },
        "Dividends": {
            "dividend_yield":        "Dividend Yield",
            "dividend_yield_5y_avg": "5Y Avg Dividend Yield",
            "payout_ratio":          "Payout Ratio",
        },
        "Income Statement": {
            "total_revenue":            "Total Revenue",
            "gross_profit":             "Gross Profit",
            "operating_income":         "Operating Income",
            "net_income":               "Net Income",
            "ebitda":                   "EBITDA",
            "ebit":                     "EBIT",
            "basic_earnings_per_share": "Basic EPS",
        },
        "Size": {
            "market_cap":       "Market Cap",
            "enterprise_value": "Enterprise Value",
            "book_value":       "Book Value",
        },
        "Risk": {
            "overall_risk":             "Overall Risk",
            "audit_risk":               "Audit Risk",
            "board_risk":               "Board Risk",
            "compensation_risk":        "Compensation Risk",
            "shareholder_rights_risk":  "Shareholder Rights Risk",
        }
    }

    METRICS_ENDPOINT_FIELDS = [
        "pe_ratio", "forward_pe", "peg_ratio", "price_to_book",
        "enterprise_to_ebitda", "enterprise_to_revenue",
        "gross_margin", "operating_margin", "ebitda_margin",
        "profit_margin", "return_on_assets", "return_on_equity",
        "earnings_growth", "earnings_growth_quarterly", "revenue_growth",
        "debt_to_equity", "current_ratio", "quick_ratio",
        "dividend_yield", "dividend_yield_5y_avg", "payout_ratio",
        "market_cap", "enterprise_value", "book_value",
        "overall_risk", "audit_risk", "board_risk",
        "compensation_risk", "shareholder_rights_risk",
    ]

    INCOME_ENDPOINT_FIELDS = [
        "total_revenue", "gross_profit", "operating_income",
        "net_income", "ebitda", "ebit", "basic_earnings_per_share",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df             = df.copy()
        self.fundamental_df = None

        if "asset_type" in self.df.columns:
            self.symbols = (
                self.df[self.df["asset_type"] == "equity"]["symbol"]
                .unique()
                .tolist()
            )
        else:
            self.symbols = self.df["symbol"].unique().tolist()

    def list_metrics(self, category: str = None):
        if category is None:
            print("Available Fundamental Categories and Metrics:")
            print("─" * 45)
            for cat, metrics in self.FUNDAMENTAL_METRICS.items():
                print(f"\n{cat}:")
                for key, name in metrics.items():
                    print(f"    {key} — {name}")
        else:
            if category not in self.FUNDAMENTAL_METRICS:
                raise ValueError(
                    f"Invalid category '{category}'. "
                    f"Valid options: {list(self.FUNDAMENTAL_METRICS.keys())}"
                )
            print(f"\n{category}:")
            print("─" * 45)
            for key, name in self.FUNDAMENTAL_METRICS[category].items():
                print(f"    {key} — {name}")
        return self.FUNDAMENTAL_METRICS

    def list_categories(self):
        categories = list(self.FUNDAMENTAL_METRICS.keys())
        print("Available Categories:")
        for cat in categories:
            print(f"  → {cat}")
        return categories

    def get_metrics_for_category(self, category: str):
        if category not in self.FUNDAMENTAL_METRICS:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Valid options: {list(self.FUNDAMENTAL_METRICS.keys())}"
            )
        return self.FUNDAMENTAL_METRICS[category]

    def fetch(self, metrics: list):
        all_valid = {
            k: v
            for cat in self.FUNDAMENTAL_METRICS.values()
            for k, v in cat.items()
        }

        invalid = [m for m in metrics if m not in all_valid]
        if invalid:
            raise ValueError(
                f"Invalid metrics: {invalid}. "
                f"Call list_metrics() to see valid options."
            )

        metrics_fields = [m for m in metrics if m in self.METRICS_ENDPOINT_FIELDS]
        income_fields  = [m for m in metrics if m in self.INCOME_ENDPOINT_FIELDS]

        rows = []
        for symbol in self.symbols:
            row = {"symbol": symbol}
            try:
                if metrics_fields:
                    metrics_data = obb.equity.fundamental.metrics(
                        symbol=symbol,
                        provider="yfinance"
                    )
                    m = metrics_data.iloc[0]
                    for field in metrics_fields:
                        row[field] = m.get(field, None)

                if income_fields:
                    income_data = obb.equity.fundamental.income(
                        symbol=symbol,
                        provider="yfinance"
                    )
                    i = income_data.iloc[0]
                    for field in income_fields:
                        row[field] = i.get(field, None)

                print(f"✓ Fetched fundamentals for {symbol}")

            except Exception as e:
                print(f"✗ Failed for {symbol}: {e}")

            rows.append(row)

        self.fundamental_df = pd.DataFrame(rows)
        return self.fundamental_df

    def fetch_by_category(self, categories: list):
        invalid = [c for c in categories if c not in self.FUNDAMENTAL_METRICS]
        if invalid:
            raise ValueError(
                f"Invalid categories: {invalid}. "
                f"Valid options: {list(self.FUNDAMENTAL_METRICS.keys())}"
            )

        metrics = []
        for category in categories:
            metrics.extend(list(self.FUNDAMENTAL_METRICS[category].keys()))

        return self.fetch(metrics)

    def get_snapshot(self, metrics: list = None):
        self._check_fetched()

        if metrics is None:
            return self.fundamental_df

        missing = [m for m in metrics if m not in self.fundamental_df.columns]
        if missing:
            raise ValueError(
                f"Metrics not fetched yet: {missing}. "
                f"Call fetch({missing}) first."
            )

        return self.fundamental_df[["symbol"] + metrics]

    def rank_by(self, metric: str, ascending: bool = False):
        self._check_fetched()

        if metric not in self.fundamental_df.columns:
            raise ValueError(
                f"Metric '{metric}' not fetched. "
                f"Call fetch(['{metric}']) first."
            )

        all_valid = {
            k: v
            for cat in self.FUNDAMENTAL_METRICS.values()
            for k, v in cat.items()
        }
        metric_name = all_valid.get(metric, metric)

        ranked = (
            self.fundamental_df[["symbol", metric]]
            .dropna()
            .sort_values(metric, ascending=ascending)
            .reset_index(drop=True)
        )
        ranked.index += 1
        ranked.index.name = "rank"

        print(f"\n── Ranked by {metric_name} ─────────────────────────")
        print(ranked.to_string())
        return ranked

    def plot(self, metric: str):
        self._check_fetched()

        if metric not in self.fundamental_df.columns:
            raise ValueError(
                f"Metric '{metric}' not fetched. "
                f"Call fetch(['{metric}']) first."
            )

        all_valid = {
            k: v
            for cat in self.FUNDAMENTAL_METRICS.values()
            for k, v in cat.items()
        }
        metric_name = all_valid.get(metric, metric)

        data = self.fundamental_df[["symbol", metric]].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(data["symbol"], data[metric], color="steelblue", edgecolor="white")
        ax.set_title(f"{metric_name} by Stock")
        ax.set_xlabel("Symbol")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    def generate(self, table_metrics: list, plot_metric: str):
        self._check_fetched()
        print("\n── Fundamental Snapshot ────────────────────────")
        snapshot = self.get_snapshot(table_metrics)
        print(snapshot.to_string(index=False))
        print(f"\n── Chart ───────────────────────────────────────")
        self.plot(plot_metric)
        return snapshot

    def _check_fetched(self):
        if self.fundamental_df is None:
            raise ValueError(
                "No fundamental data fetched. "
                "Call fetch() or fetch_by_category() first."
            )


# ── Technical Features ────────────────────────────────────────────────────────

class TechnicalFeatures:

    TECHNICAL_METRICS = {
        "Trend": {
            "sma":  "Simple Moving Average",
            "ema":  "Exponential Moving Average",
            "macd": "Moving Average Convergence Divergence",
            "adx":  "Average Directional Index",
        },
        "Momentum": {
            "rsi":        "Relative Strength Index",
            "roc":        "Rate of Change",
            "stochastic": "Stochastic Oscillator",
            "cci":        "Commodity Channel Index",
            "williams_r": "Williams Percent Range",
        },
        "Volatility": {
            "atr":             "Average True Range",
            "bollinger_bands": "Bollinger Bands",
        },
        "Volume": {
            "obv": "On Balance Volume",
            "cmf": "Chaikin Money Flow",
        }
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def list_indicators(self, category: str = None):
        if category is None:
            print("Available Technical Indicators:")
            print("─" * 40)
            for cat, indicators in self.TECHNICAL_METRICS.items():
                print(f"\n{cat}:")
                for key, name in indicators.items():
                    print(f"    {key} — {name}")
        else:
            if category not in self.TECHNICAL_METRICS:
                raise ValueError(
                    f"Invalid category '{category}'. "
                    f"Valid options: {list(self.TECHNICAL_METRICS.keys())}"
                )
            print(f"\n{category}:")
            print("─" * 40)
            for key, name in self.TECHNICAL_METRICS[category].items():
                print(f"    {key} — {name}")
        return self.TECHNICAL_METRICS

    def list_categories(self):
        return list(self.TECHNICAL_METRICS.keys())

    # ── Trend ─────────────────────────────────────────────────────────────────

    def add_sma(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: x.rolling(window).mean())
        )
        return self.df

    def add_ema(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: x.ewm(span=window, adjust=False).mean())
        )
        return self.df

    def add_macd(self, col_name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        def _macd(x):
            fast_ema    = x.ewm(span=fast, adjust=False).mean()
            slow_ema    = x.ewm(span=slow, adjust=False).mean()
            macd_line   = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram   = macd_line - signal_line
            return macd_line, signal_line, histogram

        results = (
            self.df.groupby("symbol")["close"]
            .apply(lambda x: pd.DataFrame({
                f"{col_name}_line":   _macd(x)[0],
                f"{col_name}_signal": _macd(x)[1],
                f"{col_name}_hist":   _macd(x)[2],
            }))
            .reset_level(level=0, drop=True)
        )

        self.df[f"{col_name}_line"]   = results[f"{col_name}_line"]
        self.df[f"{col_name}_signal"] = results[f"{col_name}_signal"]
        self.df[f"{col_name}_hist"]   = results[f"{col_name}_hist"]

        return self.df

    def add_adx(self, col_name: str, window: int = 14):
        def _adx(group):
            high  = group["high"]
            low   = group["low"]
            close = group["close"]

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs()
            ], axis=1).max(axis=1)

            dm_plus  = high.diff()
            dm_minus = -low.diff()

            dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

            atr      = tr.ewm(span=window, adjust=False).mean()
            di_plus  = 100 * dm_plus.ewm(span=window, adjust=False).mean() / atr
            di_minus = 100 * dm_minus.ewm(span=window, adjust=False).mean() / atr

            dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).abs()
            adx = dx.ewm(span=window, adjust=False).mean()

            return adx

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_adx)
        )
        return self.df

    # ── Momentum ──────────────────────────────────────────────────────────────

    def add_rsi(self, col_name: str, window: int = 14):
        def _rsi(x):
            delta    = x.diff()
            gain     = delta.where(delta > 0, 0)
            loss     = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(span=window, adjust=False).mean()
            avg_loss = loss.ewm(span=window, adjust=False).mean()
            rs  = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(_rsi)
        )
        return self.df

    def add_roc(self, col_name: str, window: int = 10):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(
                lambda x: (x - x.shift(window)) / x.shift(window) * 100
            )
        )
        return self.df

    def add_stochastic(self, col_name: str, window: int = 14):
        def _stochastic(group):
            low_n  = group["low"].rolling(window).min()
            high_n = group["high"].rolling(window).max()
            k = 100 * (group["close"] - low_n) / (high_n - low_n)
            d = k.rolling(3).mean()
            return k, d

        k_values = []
        d_values = []

        for symbol in self.df["symbol"].unique():
            mask  = self.df["symbol"] == symbol
            group = self.df[mask]
            k, d  = _stochastic(group)
            k_values.append(k)
            d_values.append(d)

        self.df[f"{col_name}_k"] = pd.concat(k_values)
        self.df[f"{col_name}_d"] = pd.concat(d_values)

        return self.df

    def add_cci(self, col_name: str, window: int = 20):
        def _cci(group):
            typical_price = (
                group["high"] + group["low"] + group["close"]
            ) / 3
            sma      = typical_price.rolling(window).mean()
            mean_dev = typical_price.rolling(window).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            cci = (typical_price - sma) / (0.015 * mean_dev)
            return cci

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_cci)
        )
        return self.df

    def add_williams_r(self, col_name: str, window: int = 14):
        def _williams_r(group):
            high_n = group["high"].rolling(window).max()
            low_n  = group["low"].rolling(window).min()
            wr     = -100 * (high_n - group["close"]) / (high_n - low_n)
            return wr

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_williams_r)
        )
        return self.df

    # ── Volatility ────────────────────────────────────────────────────────────

    def add_atr(self, col_name: str, window: int = 14):
        def _atr(group):
            high  = group["high"]
            low   = group["low"]
            close = group["close"]

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs()
            ], axis=1).max(axis=1)

            atr = tr.ewm(span=window, adjust=False).mean()
            return atr

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_atr)
        )
        return self.df

    def add_bollinger_bands(self, col_name: str, window: int = 20):
        def _bbands(group):
            sma   = group["close"].rolling(window).mean()
            std   = group["close"].rolling(window).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            width = (upper - lower) / sma
            pct   = (group["close"] - lower) / (upper - lower)
            return upper, lower, width, pct

        upper_vals = []
        lower_vals = []
        width_vals = []
        pct_vals   = []

        for symbol in self.df["symbol"].unique():
            mask  = self.df["symbol"] == symbol
            group = self.df[mask]
            upper, lower, width, pct = _bbands(group)
            upper_vals.append(upper)
            lower_vals.append(lower)
            width_vals.append(width)
            pct_vals.append(pct)

        self.df[f"{col_name}_upper"] = pd.concat(upper_vals)
        self.df[f"{col_name}_lower"] = pd.concat(lower_vals)
        self.df[f"{col_name}_width"] = pd.concat(width_vals)
        self.df[f"{col_name}_pct"]   = pd.concat(pct_vals)

        return self.df

    # ── Volume ────────────────────────────────────────────────────────────────

    def add_obv(self, col_name: str):
        def _obv(group):
            direction = np.sign(group["close"].diff())
            obv       = (direction * group["volume"]).fillna(0).cumsum()
            return obv

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_obv)
        )
        return self.df

    def add_cmf(self, col_name: str, window: int = 20):
        def _cmf(group):
            clv = (
                (group["close"] - group["low"]) -
                (group["high"]  - group["close"])
            ) / (group["high"] - group["low"])

            money_flow_vol = clv * group["volume"]
            cmf = (
                money_flow_vol.rolling(window).sum() /
                group["volume"].rolling(window).sum()
            )
            return cmf

        self.df[col_name] = (
            self.df.groupby("symbol", group_keys=False)
            .apply(_cmf)
        )
        return self.df


# ── Statistical Features ──────────────────────────────────────────────────────

class StatisticalFeatures:

    STATISTICAL_METRICS = {
        "Correlation Based": {
            "rolling_correlation": "Rolling Correlation vs Benchmark",
            "rolling_beta":        "Rolling Beta vs Benchmark",
            "rolling_alpha":       "Rolling Alpha vs Benchmark",
        },
        "Distribution Based": {
            "rolling_skewness": "Rolling Skewness",
            "rolling_kurtosis": "Rolling Kurtosis",
            "rolling_sharpe":   "Rolling Sharpe Ratio",
            "rolling_sortino":  "Rolling Sortino Ratio",
        },
        "Drawdown Based": {
            "max_drawdown":     "Maximum Drawdown over Window",
            "current_drawdown": "Current Drawdown from Peak",
        }
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def list_metrics(self, category: str = None):
        if category is None:
            print("Available Statistical Metrics:")
            print("─" * 40)
            for cat, metrics in self.STATISTICAL_METRICS.items():
                print(f"\n{cat}:")
                for key, name in metrics.items():
                    print(f"    {key} — {name}")
        else:
            if category not in self.STATISTICAL_METRICS:
                raise ValueError(
                    f"Invalid category '{category}'. "
                    f"Valid options: {list(self.STATISTICAL_METRICS.keys())}"
                )
            print(f"\n{category}:")
            print("─" * 40)
            for key, name in self.STATISTICAL_METRICS[category].items():
                print(f"    {key} — {name}")
        return self.STATISTICAL_METRICS

    def list_categories(self):
        return list(self.STATISTICAL_METRICS.keys())

    def _get_returns(self, symbol: str):
        return (
            self.df[self.df["symbol"] == symbol]
            .set_index("date")["close"]
            .pct_change()
            .dropna()
        )

    def _validate_benchmark(self, benchmark: str):
        if benchmark not in self.df["symbol"].values:
            raise ValueError(
                f"Benchmark '{benchmark}' not in dataset. "
                f"Add it first using MarketFeatures.add_benchmark()"
            )

    def _get_equity_symbols(self):
        if "asset_type" in self.df.columns:
            return (
                self.df[self.df["asset_type"] == "equity"]["symbol"]
                .unique()
                .tolist()
            )
        return self.df["symbol"].unique().tolist()

    def add_rolling_correlation(self, col_name: str, window: int, benchmark: str):
        self._validate_benchmark(benchmark)
        benchmark_returns = self._get_returns(benchmark)
        equity_symbols    = self._get_equity_symbols()

        for symbol in equity_symbols:
            try:
                stock_returns = self._get_returns(symbol)
                aligned = pd.concat(
                    [stock_returns, benchmark_returns],
                    axis=1, join="inner"
                )
                aligned.columns = ["stock", "benchmark"]

                rolling_corr = aligned["stock"].rolling(window).corr(
                    aligned["benchmark"]
                )

                mask = self.df["symbol"] == symbol
                self.df.loc[mask, f"{col_name}_{benchmark.lower()}"] = (
                    self.df.loc[mask, "date"].map(rolling_corr)
                )
                print(f"✓ Rolling correlation calculated for {symbol}")

            except Exception as e:
                print(f"✗ Failed for {symbol}: {e}")

        return self.df

    def add_rolling_beta(self, col_name: str, window: int, benchmark: str):
        self._validate_benchmark(benchmark)
        benchmark_returns = self._get_returns(benchmark)
        equity_symbols    = self._get_equity_symbols()

        for symbol in equity_symbols:
            try:
                stock_returns = self._get_returns(symbol)
                aligned = pd.concat(
                    [stock_returns, benchmark_returns],
                    axis=1, join="inner"
                )
                aligned.columns = ["stock", "benchmark"]

                rolling_cov  = aligned["stock"].rolling(window).cov(aligned["benchmark"])
                rolling_var  = aligned["benchmark"].rolling(window).var()
                beta_series  = rolling_cov / rolling_var

                mask = self.df["symbol"] == symbol
                self.df.loc[mask, f"{col_name}_{benchmark.lower()}"] = (
                    self.df.loc[mask, "date"].map(beta_series)
                )
                print(f"✓ Rolling beta calculated for {symbol}")

            except Exception as e:
                print(f"✗ Failed for {symbol}: {e}")

        return self.df

    def add_rolling_alpha(self, col_name: str, window: int, benchmark: str, risk_free_rate: float = 0.05):
        self._validate_benchmark(benchmark)
        benchmark_returns = self._get_returns(benchmark)
        equity_symbols    = self._get_equity_symbols()
        daily_rf          = risk_free_rate / 252

        for symbol in equity_symbols:
            try:
                stock_returns = self._get_returns(symbol)
                aligned = pd.concat(
                    [stock_returns, benchmark_returns],
                    axis=1, join="inner"
                )
                aligned.columns = ["stock", "benchmark"]

                rolling_cov = aligned["stock"].rolling(window).cov(aligned["benchmark"])
                rolling_var = aligned["benchmark"].rolling(window).var()
                beta_series = rolling_cov / rolling_var

                alpha_series = (
                    aligned["stock"].rolling(window).mean() -
                    (daily_rf + beta_series *
                    (aligned["benchmark"].rolling(window).mean() - daily_rf))
                ) * 252

                mask = self.df["symbol"] == symbol
                self.df.loc[mask, f"{col_name}_{benchmark.lower()}"] = (
                    self.df.loc[mask, "date"].map(alpha_series)
                )
                print(f"✓ Rolling alpha calculated for {symbol}")

            except Exception as e:
                print(f"✗ Failed for {symbol}: {e}")

        return self.df

    def add_rolling_skewness(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: x.pct_change().rolling(window).skew())
        )
        return self.df

    def add_rolling_kurtosis(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: x.pct_change().rolling(window).kurt())
        )
        return self.df

    def add_rolling_sharpe(self, col_name: str, window: int, risk_free_rate: float = 0.05):
        daily_rf = risk_free_rate / 252

        def _sharpe(x):
            excess = x - daily_rf
            return (
                excess.rolling(window).mean() /
                x.rolling(window).std()
            ) * np.sqrt(252)

        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: _sharpe(x.pct_change()))
        )
        return self.df

    def add_rolling_sortino(self, col_name: str, window: int, risk_free_rate: float = 0.05):
        daily_rf = risk_free_rate / 252

        def _sortino(x):
            excess   = x - daily_rf
            downside = x.copy()
            downside[downside > 0] = 0
            downside_std = downside.rolling(window).std()
            return (
                excess.rolling(window).mean() / downside_std
            ) * np.sqrt(252)

        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(lambda x: _sortino(x.pct_change()))
        )
        return self.df

    def add_max_drawdown(self, col_name: str, window: int):
        def _max_drawdown(x):
            def _dd(prices):
                peak = prices.expanding().max()
                dd   = (prices - peak) / peak
                return dd.min()
            return x.rolling(window).apply(_dd, raw=False)

        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(_max_drawdown)
        )
        return self.df

    def add_current_drawdown(self, col_name: str, window: int):
        def _current_drawdown(x):
            rolling_peak = x.rolling(window).max()
            return (x - rolling_peak) / rolling_peak

        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(_current_drawdown)
        )
        return self.df


# ── Target Features ───────────────────────────────────────────────────────────

class TargetFeatures:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_forward_return(self, col_name: str, window: int):
        if window < 15:
            print(
                f"  ℹ Note: window={window} days is sub-monthly. "
                f"For use with Fama MacBeth and Factor Regression "
                f"a window of 21 days (monthly) is recommended "
                f"for alignment with FF factors."
            )
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .pct_change(window)
            .shift(-window)
        )
        return self.df

    def add_forward_volatility(self, col_name: str, window: int):
        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .pct_change()
            .transform(lambda x: x.rolling(window).std() * np.sqrt(252))
            .shift(-window)
        )
        return self.df

    def add_forward_drawdown(self, col_name: str, window: int):
        def _fwd_drawdown(x):
            rolling_max = x.rolling(window).max()
            dd = (x - rolling_max) / rolling_max
            return dd.rolling(window).min().shift(-window)

        self.df[col_name] = (
            self.df.groupby("symbol")["close"]
            .transform(_fwd_drawdown)
        )
        return self.df

    def add_forward_column(self, col_name: str, source_col: str, window: int):
        if source_col not in self.df.columns:
            raise ValueError(
                f"Column '{source_col}' not found. "
                f"Build it first in FeatureEngine."
            )
        self.df[col_name] = (
            self.df.groupby("symbol")[source_col]
            .shift(-window)
        )
        return self.df