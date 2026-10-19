"""Budgeted Optuna search over LightGBM tree parameters, scored on validation PR-AUC.

The learning rate is held fixed (configs/model.yaml: tune.learning_rate) and the number of
trees comes from early stopping, so the search only covers tree shape and regularization.
The best parameters go to configs/model_tuned.yaml, which `fraud train` picks up as the
`lgbm_tuned` experiment. Test data is never touched here.
"""

from __future__ import annotations

import time

import optuna
import yaml
from sklearn.metrics import average_precision_score

from fraud import featuresets as fs
from fraud import tracking
from fraud.config import Paths, model_config
from fraud.train import fit_lgbm, lgbm_params, load_modeling_frame, tuned_path


def search_space(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 384, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def run(paths: Paths, n_trials: int | None = None, timeout_minutes: float | None = None) -> dict:
    cfg = model_config()
    n_trials = n_trials or cfg.tune.n_trials
    timeout = 60 * (timeout_minutes or cfg.tune.timeout_minutes)

    df = load_modeling_frame(paths)
    feats = fs.feature_columns(list(df.columns), "full")
    train, valid = df[df.split == "train"], df[df.split == "valid"]
    cats = fs.fit_categories(train, feats)
    X_tr, X_va = fs.prepare_frame(train, feats, cats), fs.prepare_frame(valid, feats, cats)
    y_tr, y_va = train.isFraud.to_numpy(), valid.isFraud.to_numpy()
    del df

    def objective(trial: optuna.Trial) -> float:
        params = lgbm_params({**search_space(trial), "learning_rate": cfg.tune.learning_rate})
        model = fit_lgbm(X_tr, y_tr, X_va, y_va, params)
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or params["n_estimators"]))
        return float(average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    t0 = time.time()
    with tracking.run("optuna_search", f"fraud-{paths.profile}", params={"n_trials": n_trials}):
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        tracking.log_metrics({"best_valid_pr_auc": study.best_value})

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    trials.to_csv(paths.reports / "tuning_trials.csv", index=False)

    best = {**study.best_params, "learning_rate": cfg.tune.learning_rate}
    tuned_path(paths).write_text(
        "# Written by `fraud tune` (Optuna, validation PR-AUC). Picked up by `fraud train`.\n"
        + yaml.safe_dump({"params": best, "valid_pr_auc": float(study.best_value),
                          "n_trials": len(study.trials), "profile": paths.profile}, sort_keys=False),
        encoding="utf-8",
    )  # fmt: skip
    return {"best_valid_pr_auc": study.best_value, "best_params": best,
            "n_trials": len(study.trials), "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
