"""
Step 1: Predictive Meta-Profiling (Section 5.1).

Automatically characterises data across the five dimensions using lightweight
sampling-based profiling.
"""

from __future__ import annotations

from typing import Any

from rht.utils.schema import (
    profile_table,
    dimensional_profile,
    detect_foreign_key_candidates,
)


class MetaProfiler:
    """Meta-profiling engine for the first step of the pipeline."""

    def profile(self, tables: dict[str, Any]) -> dict:
        """
        Profile the given tables.

        Returns
        -------
        dict with keys:
          - "table_profiles": per-table column-level profiles
          - "dimensional_profile": 5D score vector
          - "fk_candidates": heuristic FK candidates
          - "recommended_config": pipeline configuration hints
        """
        import pandas as pd

        table_profiles = {}
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                table_profiles[name] = profile_table(df)

        dim_profile = dimensional_profile(
            {k: v for k, v in tables.items() if isinstance(v, pd.DataFrame)}
        )

        fk_candidates = detect_foreign_key_candidates(
            {k: v for k, v in tables.items() if isinstance(v, pd.DataFrame)}
        )

        config_hints = self._recommend_config(dim_profile)

        return {
            "table_profiles": table_profiles,
            "dimensional_profile": dim_profile,
            "fk_candidates": fk_candidates,
            "recommended_config": config_hints,
        }

    @staticmethod
    def _recommend_config(dim_profile: dict[str, float]) -> dict[str, Any]:
        """
        Rule-based recommender that selects hyperparameters based on
        the dimensional profile (Section 5.1, Method paragraph).
        """
        config: dict[str, Any] = {}
        if dim_profile["D1_volume"] > 0.7:
            config["batch_strategy"] = "streaming"
            config["num_message_passing_layers"] = 2
        else:
            config["batch_strategy"] = "full"
            config["num_message_passing_layers"] = 3

        if dim_profile["D3_cardinality"] > 0.5:
            config["use_hierarchical_encoding"] = True
            config["memory_bank_size"] = 2048
        else:
            config["use_hierarchical_encoding"] = False

        config["use_temporal"] = dim_profile["D5_temporal"] > 0
        return config
