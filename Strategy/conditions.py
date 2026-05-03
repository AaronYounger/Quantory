from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Operator  = Literal[">", "<", ">=", "<=", "==", "!=", "cross_above", "cross_below"]
LogicType = Literal["and", "or"]


@dataclass
class Condition:
    left:           str
    operator:       Operator
    right:          Any
    right_is_column: bool = False


def select_assets(
    df: pd.DataFrame,
    selected_symbols: list[str],
    symbol_col: str = "symbol"
) -> pd.DataFrame:
    if symbol_col not in df.columns:
        raise KeyError(f"Missing symbol column: {symbol_col}")

    available_symbols = sorted(df[symbol_col].dropna().unique().tolist())

    if not selected_symbols:
        raise ValueError("No assets were selected.")

    invalid_symbols = [
        sym for sym in selected_symbols
        if sym not in available_symbols
    ]
    if invalid_symbols:
        raise ValueError(f"Invalid symbols selected: {invalid_symbols}")

    filtered_df = df[df[symbol_col].isin(selected_symbols)].copy()

    if filtered_df.empty:
        raise ValueError("No rows found for selected symbols after filtering.")

    return filtered_df


def _compare_series(
    left: pd.Series,
    operator: Operator,
    right,
):
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "cross_above":
        if not isinstance(right, pd.Series):
            raise ValueError("cross_above requires right to be a column/Series.")
        return (left.shift(1) <= right.shift(1)) & (left > right)
    if operator == "cross_below":
        if not isinstance(right, pd.Series):
            raise ValueError("cross_below requires right to be a column/Series.")
        return (left.shift(1) >= right.shift(1)) & (left < right)

    raise ValueError(f"Unsupported operator: {operator}")


def evaluate_condition(
    df: pd.DataFrame,
    condition: Condition,
    group_col: str = "symbol",
    date_col:  str = "date",
):
    required_cols = [condition.left, group_col, date_col]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df_sorted = df.sort_values([group_col, date_col]).copy()

    if condition.right_is_column and condition.right not in df_sorted.columns:
        raise KeyError(f"Missing right-side column: {condition.right}")

    result = (
        df_sorted.groupby(group_col, group_keys=False)
        .apply(
            lambda g: _compare_series(
                g[condition.left],
                condition.operator,
                g[condition.right] if condition.right_is_column else condition.right,
            )
        )
    )

    if isinstance(result.index, pd.MultiIndex):
        result = result.reset_index(level=0, drop=True)

    result = result.reindex(df_sorted.index)

    out = pd.Series(
        index=df_sorted.index,
        data=result.astype(bool),
        name="condition_result"
    )
    out = out.reindex(df.index)

    return out.fillna(False)


def combine_conditions(
    condition_results: list[pd.Series],
    logic: LogicType = "and",
) -> pd.Series:
    if not condition_results:
        raise ValueError("At least one condition result is required.")

    combined = condition_results[0].copy()

    for cond in condition_results[1:]:
        if logic == "and":
            combined = combined & cond
        elif logic == "or":
            combined = combined | cond
        else:
            raise ValueError("logic must be 'and' or 'or'")

    return combined.fillna(False)