"""Train the rules baseline, logistic regression and LightGBM variants; calibrate; score.

Experiments (all on the same time split):
  rules             hand-written rules, thresholds fit on valid
  lr                logistic regression on a small explainable feature set
  lgbm              LightGBM, unweighted                          (variant A)
  lgbm_spw          LightGBM, scale_pos_weight                    (variant B)
  lgbm_cost         LightGBM, sample weight = cost of an error    (variant C)
  lgbm_tuned        LightGBM with Optuna params (if configs/model_tuned.yaml exists)
  abl_raw_text      ablation: raw device/browser/email strings instead of normalized
  abl_no_behavior   ablation: drop velocity/history/frequency features
  abl_interpretable ablation: drop all masked vendor columns (V*, C*, D*, M*, id_*)

Every model is calibrated with isotonic regression on the calib split. The champion is
the full-feature LightGBM variant with the best validation PR-AUC.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud import featuresets as fs
from fraud import rules, split, tracking
from fraud.config import CONFIG_DIR, Paths, costs, model_config
from fraud.evaluate import metrics
from fraud.io import load_features, load_json, save_json


def tuned_path(paths: Paths):
    """Real-data tuning result is committed in configs/; other profiles keep theirs local."""
    return (
        CONFIG_DIR / "model_tuned.yaml"
        if paths.profile == "full"
        else paths.models / "model_tuned.yaml"
    )


SCORED_KEEP = [
    "TransactionID", "split", "tx_day", "TransactionDT", "isFraud", "TransactionAmt", "ProductCD",
    "has_identity", "is_new_uid", "uid", "card_key", "DeviceType", "device_family",
    "p_email_provider", "is_first_txn_uid", "card_tx_1h", "uid_tx_24h", "amt_vs_uid_mean",
    "email_match", "hour",
]  # fmt: skip


@dataclass
class Experiment:
    name: str
    kind: str  # rules | lr | lgbm
    feature_set: str = "full"
    weighting: str | None = None  # None | spw | cost
    params: dict = field(default_factory=dict)
    candidate: bool = False  # eligible to be champion


def experiments(paths: Paths, include_ablations: bool = True) -> list[Experiment]:
    exps = [
        Experiment("rules", "rules"),
        Experiment("lr", "lr"),
        Experiment("lgbm", "lgbm", candidate=True),
        Experiment("lgbm_spw", "lgbm", weighting="spw", candidate=True),
        Experiment("lgbm_cost", "lgbm", weighting="cost", candidate=True),
    ]
    if tuned_path(paths).exists():
        tuned = yaml.safe_load(tuned_path(paths).read_text(encoding="utf-8"))["params"]
        exps.append(Experiment("lgbm_tuned", "lgbm", params=tuned, candidate=True))
    if include_ablations:
        exps += [
            Experiment("abl_raw_text", "lgbm", feature_set="raw_text"),
            Experiment("abl_no_behavior", "lgbm", feature_set="no_behavioral"),
            Experiment("abl_interpretable", "lgbm", feature_set="interpretable"),
        ]
    return exps


def load_modeling_frame(paths: Paths) -> pd.DataFrame:
    df = load_features(paths)
    df["is_new_uid"] = split.new_uid_flag(df)
    return df


def cost_weights(y: np.ndarray, amt: np.ndarray) -> np.ndarray:
    """Weight each row by what a mistake on it costs: amount + fee for fraud, friction otherwise."""
    c = costs()
    w = np.where(y == 1, amt + c.chargeback_fee, c.friction_cost)
    return w / w.mean()


def lgbm_params(overrides: dict | None = None) -> dict:
    cfg = model_config()
    p = cfg.lgbm.model_dump()
    p.pop("early_stopping_rounds")
    p.update(overrides or {})
    return {**p, "objective": "binary", "random_state": cfg.seed, "n_jobs": -1, "verbose": -1}


def fit_lgbm(X_tr, y_tr, X_va, y_va, params: dict, sample_weight=None) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_X=(X_va,),
        eval_y=(y_va,),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(model_config().lgbm.early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )  # fmt: skip
    return model


def fit_lr(train: pd.DataFrame) -> Pipeline:
    num = [c for c in fs.LR_FEATURES if c in train]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),  # fmt: skip
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ProductCD"]),
        ]
    )
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    X = train[num + ["ProductCD"]].copy()
    X["ProductCD"] = X.ProductCD.astype(str)
    model.fit(X, train.isFraud)
    return model


def _lr_input(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in fs.LR_FEATURES if c in df]
    X = df[num + ["ProductCD"]].copy()
    X["ProductCD"] = X.ProductCD.astype(str)
    return X


def calibrate(p_calib: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_calib, y_calib)
    return iso


def run(paths: Paths, include_ablations: bool = True, only: list[str] | None = None) -> dict:
    paths.ensure()
    cfg = model_config()
    df = load_modeling_frame(paths)
    masks = {s: (df.split == s).to_numpy() for s in split.SPLITS}
    y = df.isFraud.to_numpy()
    amt = df.TransactionAmt.to_numpy(dtype=float)
    train, valid = df[masks["train"]], df[masks["valid"]]

    scored = df[[c for c in SCORED_KEEP if c in df]].copy()
    results, artifacts = {}, {}
    exps = [e for e in experiments(paths, include_ablations) if only is None or e.name in only]

    for exp in exps:
        t0 = time.time()
        info: dict = {"kind": exp.kind, "feature_set": exp.feature_set, "weighting": exp.weighting}
        with tracking.run(exp.name, f"fraud-{paths.profile}",
                          params={"kind": exp.kind, "feature_set": exp.feature_set,
                                  "weighting": exp.weighting, **exp.params}):  # fmt: skip
            if exp.kind == "rules":
                params = rules.fit(valid)
                p = rules.score(df, params)
                info["rules"] = params
                artifacts[exp.name] = {"rules": params}
            elif exp.kind == "lr":
                model = fit_lr(train)
                p = model.predict_proba(_lr_input(df))[:, 1]
                artifacts[exp.name] = {"model": model}
            else:
                feats = fs.feature_columns(list(df.columns), exp.feature_set)
                cats = fs.fit_categories(train, feats)
                X = fs.prepare_frame(df, feats, cats)
                params = lgbm_params(exp.params)
                weight = None
                if exp.weighting == "spw":
                    params["scale_pos_weight"] = cfg.scale_pos_weight_variant
                elif exp.weighting == "cost":
                    weight = cost_weights(y[masks["train"]], amt[masks["train"]])
                model = fit_lgbm(X[masks["train"]], y[masks["train"]], X[masks["valid"]],
                                 y[masks["valid"]], params, sample_weight=weight)  # fmt: skip
                p = model.predict_proba(X)[:, 1]
                info.update(best_iteration=int(model.best_iteration_ or params["n_estimators"]),
                            n_features=len(feats))  # fmt: skip
                artifacts[exp.name] = {"model": model, "features": feats, "categories": cats,
                                       "params": params}  # fmt: skip
                tracking.log_metrics({"best_iteration": info["best_iteration"]})

            iso = calibrate(p[masks["calib"]], y[masks["calib"]])
            pc = iso.predict(p)
            artifacts[exp.name]["calibrator"] = iso
            scored[f"p_{exp.name}"] = p.astype(np.float32)
            scored[f"pc_{exp.name}"] = pc.astype(np.float32)

            for s in ("valid", "test"):
                m = masks[s]
                info[s] = metrics(y[m], p[m], amt[m], calibrated=pc[m])
                tracking.log_metrics(info[s], prefix=f"{s}_")
            new = masks["test"] & df.is_new_uid.to_numpy()
            if new.sum() > 0 and y[new].any():
                info["test_new_uid"] = metrics(y[new], p[new], amt[new], calibrated=pc[new])
            info["seconds"] = round(time.time() - t0, 1)
            results[exp.name] = info

    champion = max((e.name for e in exps if e.candidate and e.name in results),
                   key=lambda n: results[n]["valid"]["pr_auc"])  # fmt: skip
    _save_models(paths, artifacts, champion, results)
    scored.to_parquet(paths.scored, index=False)

    summary = split.split_summary(df).reset_index().to_dict(orient="records")
    out = {"champion": champion, "experiments": results, "splits": summary,
           "new_uid_test_share": float(df.is_new_uid[masks["test"]].mean())}  # fmt: skip
    save_json(out, paths.reports / "train_results.json")
    return out


def _save_models(paths: Paths, artifacts: dict, champion: str, results: dict) -> None:
    for name, a in artifacts.items():
        if "model" in a and isinstance(a["model"], lgb.LGBMClassifier):
            a["model"].booster_.save_model(str(paths.models / f"{name}.txt"))
        with open(paths.models / f"{name}.pkl", "wb") as f:
            pickle.dump(
                {
                    k: v
                    for k, v in a.items()
                    if k != "model" or not isinstance(v, lgb.LGBMClassifier)
                },
                f,
            )

    a = artifacts[champion]
    cutoffs = load_json(paths.processed / "split_cutoffs.json")
    meta = {
        "model_name": champion,
        "model_file": f"{champion}.txt",
        "calibrator_file": f"{champion}.pkl",
        "features": a["features"],
        "categories": a["categories"],
        "params": {k: v for k, v in a["params"].items() if k != "n_jobs"},
        "best_iteration": results[champion]["best_iteration"],
        "split_cutoffs": cutoffs,
        "valid_pr_auc": results[champion]["valid"]["pr_auc"],
        "test_pr_auc": results[champion]["test"]["pr_auc"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(meta, paths.models / "metadata.json")


def load_champion(paths: Paths) -> tuple[lgb.Booster, IsotonicRegression, dict]:
    meta = load_json(paths.models / "metadata.json")
    booster = lgb.Booster(model_file=str(paths.models / meta["model_file"]))
    with open(paths.models / meta["calibrator_file"], "rb") as f:
        iso = pickle.load(f)["calibrator"]
    return booster, iso, meta


def valid_ap(model, X, y) -> float:
    return float(average_precision_score(y, model.predict_proba(X)[:, 1]))
