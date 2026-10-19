"""Independent pandas implementation of the core window features.

The production path is SQL (sql/features/03-04). This module recomputes a subset with
NumPy/pandas so tests/test_sql_pandas_parity.py can check that both agree. Two
implementations written differently are a cheap guard against an off-by-one frame.

Rules (identical to the SQL):
  * velocity counts transactions with  t_i - window <= t_j <= t_i - 1   (strictly earlier seconds)
  * prior stats use every earlier row ordered by (TransactionDT, TransactionID), never the current one
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_count(df: pd.DataFrame, key: str, window_s: int) -> pd.Series:
    """Earlier transactions by the same key in the previous `window_s` seconds."""
    out = pd.Series(0, index=df.index, dtype=np.int64)
    for _, g in df.groupby(key, sort=False):
        t = np.sort(g.TransactionDT.to_numpy())
        order = np.argsort(g.TransactionDT.to_numpy(), kind="stable")
        lo = np.searchsorted(t, t - window_s, side="left")
        hi = np.searchsorted(t, t - 1, side="right")
        counts = np.empty(len(t), dtype=np.int64)
        counts[order] = hi - lo
        out.loc[g.index] = counts
    return out


def prior_stats(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Count / mean / sample std of earlier amounts per key (expanding, shifted by one)."""
    d = df.sort_values(["TransactionDT", "TransactionID"])
    amt = d.TransactionAmt.astype(np.float64)
    g = amt.groupby(d[key], sort=False)
    n = g.cumcount()
    s = g.cumsum() - amt
    ss = (amt**2).groupby(d[key], sort=False).cumsum() - amt**2
    mean = s / n.replace(0, np.nan)
    var = (ss - n * mean**2) / (n - 1).where(n > 1)
    out = pd.DataFrame({"prior_n": n, "prior_mean": mean, "prior_std": np.sqrt(var.clip(lower=0))})
    return out.reindex(df.index)
