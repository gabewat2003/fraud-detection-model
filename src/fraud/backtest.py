"""Rolling-origin backtest: how well does the model hold up as fraud drifts?

Cut the timeline into 30-day windows. For each window k (after a minimum history), train
on every earlier window and score window k with the champion's parameters and a fixed
tree count. Rank metrics (PR-AUC, recall and dollar recall in the top 1%) need no
calibration, so each fold is cheap and comparable.

Frequency encodings are left out: they are counted over the main training window, which
overlaps later backtest folds and would leak.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from fraud import charts
from fraud import featuresets as fs
from fraud.config import Paths, model_config
from fraud.evaluate import metrics
from fraud.io import load_json, save_json

LEAKY_FOR_BACKTEST = set(fs.FREQ) | {"freq_p_email", "freq_device"}


def windows(tx_day: pd.Series, window_days: int) -> pd.Series:
    return ((tx_day - tx_day.min()) // window_days).astype(int)


def run(paths: Paths) -> dict:
    cfg = model_config()
    meta = load_json(paths.models / "metadata.json")
    feats = [f for f in meta["features"] if f not in LEAKY_FOR_BACKTEST]
    df = pd.read_parquet(
        paths.features, columns=list(dict.fromkeys(["tx_day", "isFraud", "TransactionAmt", *feats]))
    )
    df["window"] = windows(df.tx_day, cfg.backtest.window_days)
    days = df.groupby("window").tx_day.nunique()
    full = days[days >= cfg.backtest.window_days // 2].index  # drop a short trailing stub

    params = {k: v for k, v in meta["params"].items() if k not in ("n_estimators",)}
    params.update(n_estimators=meta["best_iteration"], n_jobs=-1, verbose=-1)
    rows = []
    for k in [w for w in full if w >= cfg.backtest.min_train_windows]:
        tr, te = df[df.window < k], df[df.window == k]
        cats = fs.fit_categories(tr, feats)
        model = lgb.LGBMClassifier(**params)
        model.fit(fs.prepare_frame(tr, feats, cats), tr.isFraud)
        p = model.predict_proba(fs.prepare_frame(te, feats, cats))[:, 1]
        m = metrics(te.isFraud.to_numpy(), p, te.TransactionAmt.to_numpy())
        rows.append({"test_window": int(k), "train_windows": int(k), "train_rows": len(tr), "test_rows": len(te),
                     "fraud_rate": m["base_rate"], "pr_auc": m["pr_auc"], "recall_at_1pct": m["recall_at_1pct"],
                     "dollar_recall_at_1pct": m["dollar_recall_at_1pct"], "roc_auc": m["roc_auc"]})  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(paths.reports / "backtest.csv", index=False)
    long = table.melt(id_vars="test_window", value_vars=["pr_auc", "recall_at_1pct", "dollar_recall_at_1pct"],
                      var_name="metric", value_name="value")  # fmt: skip
    long["metric"] = long.metric.map({"pr_auc": "PR-AUC", "recall_at_1pct": "Recall in top 1%",
                                      "dollar_recall_at_1pct": "Fraud $ recall in top 1%"})  # fmt: skip
    charts.save(charts.backtest(long), paths.figures, "backtest")
    out = {"folds": rows}
    save_json(out, paths.reports / "backtest.json")
    return out
