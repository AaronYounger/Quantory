import pandas as pd
from openbb import obb
obb.user.preferences.output_type = "dataframe"


class openbb_get_data:
    def __init__(self, provider: str = "yfinance"):
        self.provider = provider
        self.symbols = ['AAPL', 'NVDA', 'AMZN', 'GOOGL', 'MSFT']
        self.start_date = "2024-01-01"
        self.end_date = "2026-03-01"

    def get_price_data(self):
        all_data = []
        for symbol in self.symbols:
            result = obb.equity.price.historical(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                provider=self.provider,
                auto_adjust=True
            )
            df = result
            df['symbol'] = symbol
            all_data.append(df)
        return pd.concat(all_data, ignore_index=False)

    def clean_price_data(self, df: pd.DataFrame):
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df = df.reset_index().rename(columns={"index": "date"})
        df.columns = [col.lower() for col in df.columns]
        df = df.sort_values(["symbol", "date"])
        return df

    def add_columns(self, df: pd.DataFrame):
        df = df.copy()
        if "dividend" in df.columns:
            df = df.drop(columns=["dividend"])
        if "split_ratio" in df.columns:
            df = df.drop(columns=['split_ratio'])
        return df

    def get_sector_data(self):
        rows = []
        for symbol in self.symbols:
            row = {"symbol": symbol}
            try:
                profile = obb.equity.profile(
                    symbol=symbol,
                    provider=self.provider
                )
                p = profile.iloc[0]
                row["sector"]             = p.get("sector", None)
                row["industry"]           = p.get("industry_category", None)
                row["market_cap"]         = p.get("market_cap", None)
                row["country"]            = p.get("hq_country", None)
                row["exchange"]           = p.get("stock_exchange", None)
                row["employees"]          = p.get("employees", None)
                row["currency"]           = p.get("currency", None)
                row["shares_outstanding"] = p.get("shares_outstanding", None)
                row["dividend_yield"]     = p.get("dividend_yield", None)
                row["asset_type"]         = "equity"

                mc = row.get("market_cap", None)
                if mc is not None:
                    if mc >= 200e9:
                        row["market_cap_bucket"] = "Mega"
                    elif mc >= 10e9:
                        row["market_cap_bucket"] = "Large"
                    elif mc >= 2e9:
                        row["market_cap_bucket"] = "Mid"
                    elif mc >= 300e6:
                        row["market_cap_bucket"] = "Small"
                    else:
                        row["market_cap_bucket"] = "Micro"
                else:
                    row["market_cap_bucket"] = None

            except Exception as e:
                print(f"Sector fetch failed for {symbol}: {e}")

            rows.append(row)
        return pd.DataFrame(rows)

    def build_dataset(self):
        df = self.get_price_data()
        df = self.clean_price_data(df)
        df = self.add_columns(df)

        sector_df = self.get_sector_data()
        df = df.merge(sector_df, on="symbol", how="left")

        return df


# ── Shared fetch utility ──────────────────────────────────────────────────────

def fetch_symbols(
    symbols: list,
    start_date: str,
    end_date: str,
    provider: str = "yfinance"
) -> pd.DataFrame:
    """
    Shared utility for fetching any list of symbols.
    Used by MarketFeatures and SectorFeatures.
    """
    all_data = []
    for symbol in symbols:
        try:
            result = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
                auto_adjust=True
            )
            df = result
            df["symbol"] = symbol
            all_data.append(df)
            print(f"✓ Fetched {symbol}")
        except Exception as e:
            print(f"✗ Failed to fetch {symbol}: {e}")

    if not all_data:
        raise ValueError("No data fetched. Check your symbols and date range.")

    df = pd.concat(all_data, ignore_index=False)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df.reset_index().rename(columns={"index": "date"})
    df.columns = [col.lower() for col in df.columns]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df