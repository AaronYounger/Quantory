import pandas as pd
import numpy as np


class ScreenEngine:

    def __init__(self, df: pd.DataFrame, fundamental_df: pd.DataFrame = None):
        self.raw_df         = df.copy()
        self.fundamental_df = fundamental_df.copy() if fundamental_df is not None else None
        self.snapshot_df    = None
        self.screened_df    = None
        self.ranked_df      = None

    def _snapshot(self):
        snapshot = (
            self.raw_df
            .sort_values("date")
            .groupby("symbol")
            .last()
            .reset_index()
        )

        if self.fundamental_df is not None:
            snapshot = snapshot.merge(
                self.fundamental_df,
                on="symbol",
                how="left"
            )

        self.snapshot_df = snapshot
        return snapshot

    def get_available_columns(self):
        if self.snapshot_df is None:
            self._snapshot()

        base_cols = [
            "date", "open", "high", "low",
            "close", "volume", "asset_type",
            "sector", "industry", "market_cap",
            "country", "exchange", "employees",
            "currency", "shares_outstanding"
        ]

        available = [
            c for c in self.snapshot_df.columns
            if c not in base_cols and c != "symbol"
        ]

        print("Available columns for screening:")
        for col in available:
            print(f"  → {col}")

        return available

    def _apply_single_filter(self, df: pd.DataFrame, column: str, operator: str, value):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Call get_available_columns() to see valid options."
            )

        if operator == ">":
            return df[column] > value
        elif operator == "<":
            return df[column] < value
        elif operator == ">=":
            return df[column] >= value
        elif operator == "<=":
            return df[column] <= value
        elif operator == "==":
            return df[column] == value
        elif operator == "!=":
            return df[column] != value
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def apply_filters(self, filter_groups: list):
        if self.snapshot_df is None:
            self._snapshot()

        df = self.snapshot_df.copy()
        combined_mask = pd.Series(False, index=df.index)

        for group in filter_groups:
            group_mask = pd.Series(True, index=df.index)
            for f in group:
                condition  = self._apply_single_filter(
                    df, f["column"], f["operator"], f["value"]
                )
                group_mask = group_mask & condition
            combined_mask = combined_mask | group_mask

        self.screened_df = df[combined_mask].reset_index(drop=True)

        print(f"\n── Screen Results ──────────────────────────────")
        print(f"  {len(self.screened_df)} stocks passed filters out of {len(df)}")

        return self.screened_df

    def composite_score(self, weights: dict):
        if self.screened_df is None:
            raise ValueError(
                "No screened results. Call apply_filters() first."
            )

        total_weight = sum(v["weight"] for v in weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(
                f"Weights must sum to 1.0. Current sum: {total_weight:.2f}"
            )

        missing = [c for c in weights.keys() if c not in self.screened_df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. "
                f"Call get_available_columns() to see valid options."
            )

        df = self.screened_df.copy()
        df["composite_score"] = 0.0

        for col, config in weights.items():
            weight    = config["weight"]
            ascending = config["ascending"]

            col_min = df[col].min()
            col_max = df[col].max()

            if col_max == col_min:
                normalized = pd.Series(0.5, index=df.index)
            else:
                normalized = (df[col] - col_min) / (col_max - col_min)

            if ascending:
                normalized = 1 - normalized

            df["composite_score"] += normalized * weight

        df = df.sort_values("composite_score", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        df = df.set_index("rank")

        self.ranked_df = df

        print(f"\n── Composite Score Rankings ────────────────────")
        print(df[["symbol", "composite_score"] + list(weights.keys())].to_string())

        return self.ranked_df

    def rank_by(self, metric: str, ascending: bool = False, top_n: int = None):
        if self.screened_df is None:
            raise ValueError(
                "No screened results. Call apply_filters() first."
            )

        if metric not in self.screened_df.columns:
            raise ValueError(
                f"Column '{metric}' not found. "
                f"Call get_available_columns() to see valid options."
            )

        df = (
            self.screened_df
            .sort_values(metric, ascending=ascending)
            .reset_index(drop=True)
        )

        df["rank"] = range(1, len(df) + 1)
        df = df.set_index("rank")

        if top_n is not None:
            df = df.head(top_n)

        self.ranked_df = df

        print(f"\n── Ranked by {metric} ──────────────────────────")
        print(df[["symbol", metric]].to_string())

        return self.ranked_df

    def quantile_rank(self, metric: str, n_quantiles: int = 5, ascending: bool = False):
        if self.screened_df is None:
            raise ValueError(
                "No screened results. Call apply_filters() first."
            )

        if metric not in self.screened_df.columns:
            raise ValueError(f"Column '{metric}' not found.")

        df = self.screened_df.copy()

        df["quantile"] = pd.qcut(
            df[metric],
            q=n_quantiles,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
            duplicates="drop"
        )

        if ascending:
            df["quantile"] = df["quantile"].cat.rename_categories(
                {f"Q{i+1}": f"Q{n_quantiles-i}" for i in range(n_quantiles)}
            )

        df = df.sort_values("quantile")
        self.ranked_df = df

        print(f"\n── Quantile Rankings by {metric} ───────────────")
        print(df[["symbol", metric, "quantile"]].to_string(index=False))

        print(f"\n── Quantile Summary ────────────────────────────")
        summary = (
            df.groupby("quantile")[metric]
            .agg(["count", "mean", "min", "max"])
            .round(4)
        )
        print(summary.to_string())

        return self.ranked_df

    def run(self,
            filter_groups: list,
            weights: dict = None,
            sort_by: str = None,
            ascending: bool = False,
            top_n: int = None,
            n_quantiles: int = None):

        self.apply_filters(filter_groups)

        if weights is not None:
            return self.composite_score(weights)
        elif n_quantiles is not None and sort_by is not None:
            return self.quantile_rank(sort_by, n_quantiles, ascending)
        elif sort_by is not None:
            return self.rank_by(sort_by, ascending, top_n)
        else:
            return self.screened_df

    def get_rebalance_dates(self, frequency: str, start_date: str, end_date: str):
        freq_map = {
            "daily":     "B",
            "weekly":    "W-FRI",
            "monthly":   "MS",
            "quarterly": "QS",
        }

        if frequency not in freq_map:
            raise ValueError(
                f"Invalid frequency '{frequency}'. "
                f"Valid options: {list(freq_map.keys())}"
            )

        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq_map[frequency]
        )

        valid_dates     = pd.to_datetime(self.raw_df["date"].unique())
        rebalance_dates = [d for d in date_range if d in valid_dates]

        if not rebalance_dates:
            raise ValueError(
                f"No valid rebalance dates found between "
                f"{start_date} and {end_date}. "
                f"Check your date range matches the dataset."
            )

        print(f"Generated {len(rebalance_dates)} rebalance dates ({frequency})")
        return rebalance_dates

    def historical_rank(self,
                        metric: str,
                        rebalance_dates: list,
                        n_quantiles: int = 5,
                        ascending: bool = False,
                        filter_groups: list = None):

        if metric not in self.raw_df.columns:
            raise ValueError(
                f"Metric '{metric}' not found in dataset. "
                f"Build it first in FeatureEngine."
            )

        results = []

        for date in rebalance_dates:
            try:
                snapshot = (
                    self.raw_df[self.raw_df["date"] <= date]
                    .sort_values("date")
                    .groupby("symbol")
                    .last()
                    .reset_index()
                )

                if self.fundamental_df is not None:
                    snapshot = snapshot.merge(
                        self.fundamental_df,
                        on="symbol",
                        how="left"
                    )

                if filter_groups is not None:
                    combined_mask = pd.Series(False, index=snapshot.index)
                    for group in filter_groups:
                        group_mask = pd.Series(True, index=snapshot.index)
                        for f in group:
                            if f["column"] in snapshot.columns:
                                condition  = self._apply_single_filter(
                                    snapshot,
                                    f["column"],
                                    f["operator"],
                                    f["value"]
                                )
                                group_mask = group_mask & condition
                        combined_mask = combined_mask | group_mask
                    snapshot = snapshot[combined_mask].reset_index(drop=True)

                if len(snapshot) < n_quantiles:
                    print(
                        f"Warning: Not enough stocks on {date} "
                        f"to form {n_quantiles} quantiles. Skipping."
                    )
                    continue

                snapshot["quantile"] = pd.qcut(
                    snapshot[metric],
                    q=n_quantiles,
                    labels=[f"Q{i+1}" for i in range(n_quantiles)],
                    duplicates="drop"
                )

                if ascending:
                    snapshot["quantile"] = snapshot["quantile"].cat.rename_categories(
                        {f"Q{i+1}": f"Q{n_quantiles-i}" for i in range(n_quantiles)}
                    )

                snapshot["rebalance_date"] = date

                results.append(snapshot[[
                    "symbol", "quantile", "rebalance_date", metric
                ]])

            except Exception as e:
                print(f"Warning: Failed to rank on {date}: {e}")
                continue

        if not results:
            raise ValueError(
                "No historical rankings could be generated. "
                "Check your metric and date range."
            )

        historical_df = pd.concat(results, ignore_index=True)

        print(f"\n── Historical Rankings Generated ───────────────")
        print(f"  Dates:     {len(rebalance_dates)}")
        print(f"  Metric:    {metric}")
        print(f"  Quantiles: {n_quantiles}")
        print(f"  Preview:")
        print(historical_df.head(10).to_string(index=False))

        return historical_df