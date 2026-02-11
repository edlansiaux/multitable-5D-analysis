"""Temporal utility functions."""

from __future__ import annotations

import torch


def timestamps_to_seconds(timestamps, reference=None):
    """Convert pandas Timestamps or datetime64 to float seconds since reference."""
    import pandas as pd
    import numpy as np

    ts = pd.to_datetime(timestamps)
    ref = ts.min() if reference is None else pd.Timestamp(reference)
    delta = (
        (ts - ref).total_seconds()
        if hasattr(ts, "total_seconds")
        else (ts - ref).dt.total_seconds()
    )
    return torch.tensor(delta.values, dtype=torch.float32)
