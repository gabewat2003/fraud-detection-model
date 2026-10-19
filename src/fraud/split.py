"""Time-based split on TransactionDT.

Fraud patterns drift as attackers adapt, and a random split would put the same card in
train and test. Splitting on time mimics deployment: fit on the past, score the future.
Four splits keep every tuning decision away from the final test numbers:

    train  -> fit
    valid  -> early stopping, Optuna, rule thresholds
    calib  -> isotonic calibration, policy thresholds
    test   -> final numbers, touched once
"""

from __future__ import annotations

import pandas as pd

from fraud.config import SplitConfig, split_config

SPLITS = ["train", "valid", "calib", "test"]


def compute_cutoffs(conn, cfg: SplitConfig | None = None) -> dict[str, float]:
    cfg = cfg or split_config()
    row = conn.execute(
        "SELECT quantile_cont(TransactionDT, ?), quantile_cont(TransactionDT, ?), "
        "quantile_cont(TransactionDT, ?) FROM base",
        [cfg.train_end, cfg.valid_end, cfg.calib_end],
    ).fetchone()
    return {"train_end": float(row[0]), "valid_end": float(row[1]), "calib_end": float(row[2])}


def split_case_sql(cutoffs: dict[str, float]) -> str:
    return (
        f"CASE WHEN TransactionDT <= {cutoffs['train_end']} THEN 'train' "
        f"WHEN TransactionDT <= {cutoffs['valid_end']} THEN 'valid' "
        f"WHEN TransactionDT <= {cutoffs['calib_end']} THEN 'calib' "
        f"ELSE 'test' END"
    )


def assign_split(dt: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    out = pd.Series("test", index=dt.index, dtype=object)
    out[dt <= cutoffs["calib_end"]] = "calib"
    out[dt <= cutoffs["valid_end"]] = "valid"
    out[dt <= cutoffs["train_end"]] = "train"
    return out


def new_uid_flag(df: pd.DataFrame) -> pd.Series:
    """Test rows whose account proxy never appeared before the test window (cold start)."""
    seen = set(df.loc[df.split != "test", "uid"].dropna())
    return (df.split == "test") & df.uid.notna() & ~df.uid.isin(seen)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split", observed=True)
    out = pd.DataFrame(
        {
            "rows": g.size(),
            "fraud_rate": g.isFraud.mean(),
            "fraud_usd": g.apply(lambda d: d.TransactionAmt[d.isFraud == 1].sum()),
            "first_day": g.tx_day.min(),
            "last_day": g.tx_day.max(),
        }
    )
    return out.reindex(SPLITS)
