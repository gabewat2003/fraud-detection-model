"""Metrics a fraud team actually uses, plus day-block bootstrap confidence intervals.

Accuracy is useless here (approve-everything scores ~96.5%). What matters:
  * PR-AUC (average precision), compared against the base rate, which is what random scores get
  * precision / recall in the top k% of scores: tied to how many cases reviewers can see
  * recall at a fixed false-positive rate: "at a 1% false-decline rate, what share do we catch?"
  * dollar-weighted recall: share of fraud DOLLARS caught, not fraud count
ROC-AUC is reported for completeness but flatters models on imbalanced data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

TOP_K = (0.01, 0.02, 0.05)
FPR_TARGETS = (0.005, 0.01)


def top_k_mask(p: np.ndarray, k: float) -> np.ndarray:
    n = max(1, int(round(len(p) * k)))
    idx = np.argsort(-p, kind="stable")[:n]
    mask = np.zeros(len(p), dtype=bool)
    mask[idx] = True
    return mask


def precision_recall_at_k(y: np.ndarray, p: np.ndarray, k: float) -> tuple[float, float]:
    m = top_k_mask(p, k)
    return float(y[m].mean()), float(y[m].sum() / max(y.sum(), 1))


def dollar_recall_at_k(y: np.ndarray, p: np.ndarray, amt: np.ndarray, k: float) -> float:
    m = top_k_mask(p, k)
    fraud_usd = (amt * y).sum()
    return float((amt[m] * y[m]).sum() / fraud_usd) if fraud_usd > 0 else float("nan")


def recall_at_fpr(y: np.ndarray, p: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def average_precision_manual(y: np.ndarray, p: np.ndarray) -> float:
    """AP as the step-wise area under the PR curve, written out (sklearn cross-check)."""
    precision, recall, _ = precision_recall_curve(y, p)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def metrics(y, p, amt, calibrated: np.ndarray | None = None) -> dict:
    y, p, amt = np.asarray(y), np.asarray(p, dtype=float), np.asarray(amt, dtype=float)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
    }
    for k in TOP_K:
        prec, rec = precision_recall_at_k(y, p, k)
        tag = f"{int(k * 100)}pct"
        out[f"precision_at_{tag}"] = prec
        out[f"recall_at_{tag}"] = rec
        out[f"dollar_recall_at_{tag}"] = dollar_recall_at_k(y, p, amt, k)
    for f in FPR_TARGETS:
        out[f"recall_at_fpr_{f:g}"] = recall_at_fpr(y, p, f)
    if calibrated is not None:
        out["brier"] = float(brier_score_loss(y, np.asarray(calibrated, dtype=float)))
    return out


def bootstrap_by_day(
    df: pd.DataFrame, stat, n_boot: int = 500, seed: int = 0, day_col: str = "tx_day"
) -> tuple[float, float]:
    """95% CI by resampling whole days (transactions within a day are correlated)."""
    rng = np.random.default_rng(seed)
    groups = {d: g.index.to_numpy() for d, g in df.groupby(day_col)}
    days = np.array(list(groups))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        vals.append(stat(df.loc[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def reliability_table(y, p, bins: int = 10, min_n: int = 20) -> pd.DataFrame:
    """Mean predicted vs observed fraud rate in fixed-width probability bins.

    Fixed-width bins (not quantiles): with a 3.5% base rate, quantile bins put nine of ten
    bins near zero and hide the high-score region, which is where decisions are made.
    """
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    d["bin"] = np.minimum((d.p * bins).astype(int), bins - 1)
    t = (
        d.groupby("bin")
        .agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    return t[t.n >= min_n]


def pr_curve_points(y, p, max_points: int = 400) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(y, p)
    d = pd.DataFrame({"recall": recall, "precision": precision})
    if len(d) > max_points:
        d = d.iloc[np.linspace(0, len(d) - 1, max_points).astype(int)]
    return d
