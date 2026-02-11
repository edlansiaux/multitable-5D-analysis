"""
Schema detection utilities for Step 1 (Meta-Profiling, Section 5.1).

Uses sampling-based profiling: <=1% or 10 000 rows, HyperLogLog for
cardinality estimation, and value-overlap statistics for FK candidates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


def profile_table(
    df: "pd.DataFrame", sample_frac: float = 0.01, max_sample: int = 10_000
) -> dict:
    """
    Lightweight profiling of a single table.

    Returns a dict with column types, cardinalities, null rates, and
    datetime-column detection.
    """
    n = min(int(len(df) * sample_frac), max_sample, len(df))
    sample = df.sample(n, random_state=42) if n < len(df) else df

    profile: dict[str, Any] = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": {},
    }
    for col in sample.columns:
        col_info: dict[str, Any] = {}
        col_info["dtype"] = str(sample[col].dtype)
        col_info["null_rate"] = float(sample[col].isna().mean())
        col_info["nunique_sample"] = int(sample[col].nunique())
        col_info["is_numeric"] = pd.api.types.is_numeric_dtype(sample[col])
        col_info["is_datetime"] = pd.api.types.is_datetime64_any_dtype(sample[col])
        col_info["is_high_cardinality"] = col_info["nunique_sample"] > 100
        profile["columns"][col] = col_info
    return profile


def detect_foreign_key_candidates(
    tables: dict[str, "pd.DataFrame"],
    overlap_min: float = 0.5,
    sample_size: int = 10_000,
) -> list[dict]:
    """
    Heuristic FK detection via value-overlap statistics (Section 5.1).
    """
    candidates = []
    names = list(tables.keys())
    for i, t1 in enumerate(names):
        df1 = tables[t1]
        for t2 in names[i + 1:]:
            df2 = tables[t2]
            for c1 in df1.columns:
                for c2 in df2.columns:
                    s1 = set(df1[c1].dropna().head(sample_size))
                    s2 = set(df2[c2].dropna().head(sample_size))
                    if not s1 or not s2:
                        continue
                    overlap = len(s1 & s2) / min(len(s1), len(s2))
                    if overlap >= overlap_min:
                        candidates.append(
                            {
                                "table1": t1,
                                "col1": c1,
                                "table2": t2,
                                "col2": c2,
                                "overlap": overlap,
                            }
                        )
    return candidates


def dimensional_profile(tables: dict[str, "pd.DataFrame"]) -> dict[str, float]:
    """
    Compute a quantitative 5D dimensional profile (Step 1 output).

    Scores each dimension from 0 to 1 based on heuristics.
    """
    total_rows = sum(len(df) for df in tables.values())
    total_cols = sum(len(df.columns) for df in tables.values())
    max_card = max(
        (df[c].nunique() for df in tables.values() for c in df.columns), default=0
    )
    num_tables = len(tables)
    has_temporal = any(
        any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
        for df in tables.values()
    )

    return {
        "D1_volume": min(total_rows / 1e7, 1.0),
        "D2_variables": min(total_cols / 500, 1.0),
        "D3_cardinality": min(max_card / 10_000, 1.0),
        "D4_tables": min(num_tables / 30, 1.0),
        "D5_temporal": 1.0 if has_temporal else 0.0,
    }
