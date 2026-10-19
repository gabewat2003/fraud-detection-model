"""Population Stability Index (PSI) for the top features and the model score.

PSI compares a feature's distribution in a later window against the training window:
    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)
Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 large shift worth investigating.
Numeric features use training-window deciles; missing values get their own bin (in this
data, missingness itself drifts). Categorical features use their categories directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud import charts
from fraud import featuresets as fs
from fraud.backtest import windows
from fraud.config import Paths, model_config
from fraud.io import load_json, save_json

EPS = 1e-4
TOP_N = 20


def _shares(codes: pd.Series, levels) -> np.ndarray:
    s = codes.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    return np.clip(s, EPS, None)


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = expected.dropna().to_numpy(dtype=float)
    if len(e) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1))[1:-1])

    def code(s: pd.Series) -> pd.Series:
        c = pd.Series(np.searchsorted(edges, s.to_numpy(dtype=float), side="right"), index=s.index)
        return c.where(s.notna(), -1)

    levels = [-1, *range(len(edges) + 1)]
    pe, pa = _shares(code(expected), levels), _shares(code(actual), levels)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e, a = expected.astype(object).fillna("<NA>"), actual.astype(object).fillna("<NA>")
    levels = sorted(set(e) | set(a), key=str)
    pe, pa = _shares(e, levels), _shares(a, levels)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi(expected: pd.Series, actual: pd.Series) -> float:
    if fs.is_categorical(expected):
        return psi_categorical(expected, actual)
    return psi_numeric(expected, actual)


def run(paths: Paths) -> dict:
    cfg = model_config()
    imp_path = paths.reports / "shap_importance.csv"
    if imp_path.exists():
        top = pd.read_csv(imp_path).feature.head(TOP_N).tolist()
    else:
        top = load_json(paths.models / "metadata.json")["features"][:TOP_N]
    df = pd.read_parquet(
        paths.features, columns=list(dict.fromkeys(["TransactionID", "tx_day", "split", *top]))
    )
    scored = pd.read_parquet(paths.scored)
    champ = load_json(paths.models / "metadata.json")["model_name"]
    df = df.merge(scored[["TransactionID", f"p_{champ}"]].rename(columns={f"p_{champ}": "model_score"}),
                  on="TransactionID")  # fmt: skip
    df["window"] = windows(df.tx_day, cfg.backtest.window_days)
    ref = df[df.split == "train"]

    rows = []
    for col in [*top, "model_score"]:
        for w, g in df.groupby("window"):
            if g.tx_day.nunique() < 5:
                continue
            rows.append({"feature": col, "window": int(w), "psi": psi(ref[col], g[col])})
    table = pd.DataFrame(rows)
    table.to_csv(paths.reports / "drift_psi.csv", index=False)
    charts.save(charts.psi_heatmap(table), paths.figures, "drift_psi")

    last = table[table.window == table.window.max()].sort_values("psi", ascending=False)
    out = {
        "windows": sorted(table.window.unique().tolist()),
        "train_windows": sorted(df[df.split == "train"].window.unique().tolist()),
        "last_window_large_shift": last[last.psi > 0.2].feature.tolist(),
        "last_window_top": last.head(10).to_dict(orient="records"),
        "score_psi_by_window": table[table.feature == "model_score"]
        .set_index("window")
        .psi.to_dict(),
    }
    save_json(out, paths.reports / "drift.json")
    return out
