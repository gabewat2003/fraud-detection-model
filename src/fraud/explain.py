"""Explainability: TreeSHAP attributions and plain-English reason codes.

LightGBM computes exact TreeSHAP values natively (`pred_contrib=True`), the same algorithm
as `shap.TreeExplainer`, without the extra dependency at serving time. Contributions are
in log-odds. Masked vendor columns can't be explained to a customer or an auditor, so
reason codes only name interpretable features and fold the rest into one
"masked vendor signals" line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud import charts
from fraud import featuresets as fs
from fraud.config import Paths, model_config
from fraud.io import load_json, save_json
from fraud.train import load_champion

GROUP_ORDER = [
    "masked",
    "velocity",
    "history",
    "text_norm",
    "known_raw",
    "frequency",
    "simple",
    "text_raw",
]


def contributions(booster, X: pd.DataFrame) -> pd.DataFrame:
    contrib = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index).assign(
        _bias=contrib[:, -1]
    )


def _fmt(name: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if name in (
            "TransactionAmt",
            "uid_prior_mean_amt",
            "uid_prior_max_amt",
            "card_amt_24h",
            "card_prior_mean_amt",
        ):
            return f"${v:,.2f}"
        if name.startswith("amt_vs"):
            return f"{v:.1f}x"
        if name.endswith("secs_since_prev"):
            return f"{v / 60:,.0f} min" if v < 86400 else f"{v / 86400:,.1f} days"
        return f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"
    return str(value)


def reason_codes(contrib: pd.Series, values: pd.Series, k: int = 3) -> list[dict]:
    """Top-k features pushing this transaction TOWARD fraud, interpretable ones by name."""
    contrib = contrib.drop(labels="_bias", errors="ignore")
    groups = {c: fs.group_of(c) for c in contrib.index}
    masked = [c for c in contrib.index if groups[c] not in fs.INTERPRETABLE_GROUPS]
    named = contrib.drop(labels=masked)
    items = [(c, float(v)) for c, v in named.items() if v > 0]
    masked_total = float(contrib[masked].clip(lower=0).sum()) if masked else 0.0
    if masked_total > 0:
        items.append(("masked_vendor_signals", masked_total))
    items.sort(key=lambda t: -t[1])
    out = []
    for name, v in items[:k]:
        if name == "masked_vendor_signals":
            text = "masked vendor risk signals (V/C/D/M/id columns)"
        else:
            text = f"{fs.DESCRIPTIONS.get(name, name)}: {_fmt(name, values.get(name))}"
        out.append({"feature": name, "contribution": round(v, 4), "text": text})
    return out


def run(paths: Paths) -> dict:
    booster, _, meta = load_champion(paths)
    feats, cats = meta["features"], meta["categories"]
    df = pd.read_parquet(
        paths.features, columns=list(dict.fromkeys(["TransactionID", "split", "isFraud", *feats]))
    )
    test = df[df.split == "test"]
    n = min(model_config().explain.shap_sample, len(test))
    sample = test.sample(n=n, random_state=model_config().seed)
    X = fs.prepare_frame(sample, feats, cats)
    contrib = contributions(booster, X)

    imp = contrib.drop(columns="_bias").abs().mean().rename("mean_abs_shap").reset_index()
    imp.columns = ["feature", "mean_abs_shap"]
    imp["group"] = imp.feature.map(fs.group_of)
    imp["interpretable"] = imp.group.isin(fs.INTERPRETABLE_GROUPS)
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    imp["gain"] = imp.feature.map(gain).fillna(0)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(paths.reports / "shap_importance.csv", index=False)

    groups_present = [g for g in GROUP_ORDER if g in set(imp.group)]
    top = imp.head(25)
    charts.save(
        charts.importance(top, [g for g in groups_present if g in set(top.group)]),
        paths.figures,
        "shap_top_features",
    )
    interp = imp[imp.interpretable].head(20)
    charts.save(charts.importance(interp, [g for g in groups_present if g in set(interp.group)]),
                paths.figures, "shap_interpretable_features")  # fmt: skip
    by_group = imp.groupby("group").mean_abs_shap.sum()
    by_group = (by_group / by_group.sum()).rename("share").reset_index()
    charts.save(charts.group_importance(by_group), paths.figures, "shap_by_group")

    # example reason codes for the highest-scored frauds and false positives
    p = booster.predict(X)
    sample = sample.assign(p=p)
    examples = []
    for label, rows in (("caught_fraud", sample[sample.isFraud == 1].nlargest(3, "p")),
                        ("false_positive", sample[sample.isFraud == 0].nlargest(3, "p"))):  # fmt: skip
        for idx, r in rows.iterrows():
            examples.append({"type": label, "TransactionID": int(r.TransactionID), "p": float(r.p),
                             "reasons": reason_codes(contrib.loc[idx], sample.loc[idx])})  # fmt: skip
    out = {
        "n_explained": int(n),
        "group_share": by_group.set_index("group").share.to_dict(),
        "interpretable_share": float(
            imp[imp.interpretable].mean_abs_shap.sum() / imp.mean_abs_shap.sum()
        ),
        "top_features": imp.head(15)[["feature", "group", "mean_abs_shap"]].to_dict(
            orient="records"
        ),
        "examples": examples,
    }
    save_json(out, paths.reports / "explain.json")
    return out


def manifest_lookup(paths: Paths) -> dict:
    return {m["name"]: m for m in load_json(paths.processed / "feature_manifest.json")}
