"""Optional MLflow experiment tracking (local SQLite store). Disabled with FRAUD_MLFLOW=0."""

from __future__ import annotations

import contextlib
import math
import os
import warnings

from fraud.config import ROOT


def enabled() -> bool:
    return os.environ.get("FRAUD_MLFLOW", "1") != "0"


@contextlib.contextmanager
def run(name: str, experiment: str, params: dict | None = None, tags: dict | None = None):
    if not enabled():
        yield None
        return
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    warnings.filterwarnings("ignore", module="mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as r:
        if params:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
        if tags:
            mlflow.set_tags(tags)
        yield r


def log_metrics(metrics: dict, prefix: str = "") -> None:
    if not enabled():
        return
    import mlflow

    clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()
             if isinstance(v, (int, float)) and not math.isnan(float(v))}  # fmt: skip
    if clean and mlflow.active_run():
        mlflow.log_metrics(clean)


def log_artifact(path) -> None:
    if not enabled():
        return
    import mlflow

    if mlflow.active_run():
        mlflow.log_artifact(str(path))
