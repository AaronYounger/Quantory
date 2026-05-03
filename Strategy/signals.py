from __future__ import annotations

import pandas as pd
from .conditions import (
    Condition,
    LogicType,
    select_assets,
    evaluate_condition,
    combine_conditions,
)


def build_signals(
    df: pd.DataFrame,
    entry_conditions: list[Condition],
    exit_conditions:  list[Condition],
    selected_symbols: list[str] | None = None,
    entry_logic:      LogicType = "and",
    exit_logic:       LogicType = "and",
    group_col:        str = "symbol",
    date_col:         str = "date",
) -> pd.DataFrame:
    out = df.copy()

    if selected_symbols is not None:
        out = select_assets(out, selected_symbols, symbol_col=group_col)

    entry_results = [
        evaluate_condition(out, cond, group_col=group_col, date_col=date_col)
        for cond in entry_conditions
    ]

    exit_results = [
        evaluate_condition(out, cond, group_col=group_col, date_col=date_col)
        for cond in exit_conditions
    ]

    out["entry_signal"] = combine_conditions(entry_results, logic=entry_logic)
    out["exit_signal"]  = combine_conditions(exit_results,  logic=exit_logic)

    # ── Position state tracker ────────────────────────────────────────────────
    def _track_positions(group):
        position  = 0
        positions = []
        for entry, exit_ in zip(group["entry_signal"], group["exit_signal"]):
            if entry and position == 0:
                position = 1
            elif exit_ and position == 1:
                position = 0
            positions.append(position)
        group["position"] = positions
        return group

    out = (
        out.groupby(group_col, group_keys=False)
        .apply(_track_positions)
    )

    # ── Execution price — next day open ───────────────────────────────────────
    out["execution_price"] = (
        out.groupby(group_col)["open"]
        .shift(-1)
    )

    return out