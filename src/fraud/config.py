"""Project configuration: paths per profile plus the YAML configs in configs/."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# The repo root holds configs/ and sql/. An editable install finds it from this file;
# a regular install (the Docker image) lives in site-packages, so it sets FRAUD_ROOT.
ROOT = Path(os.environ.get("FRAUD_ROOT") or Path(__file__).resolve().parents[2])
CONFIG_DIR = ROOT / "configs"
SQL_DIR = ROOT / "sql"


@dataclass(frozen=True)
class Paths:
    """Where each pipeline stage reads and writes.

    The "full" profile runs on the Kaggle CSVs. The "sample" profile runs the identical
    pipeline on a generated IEEE-shaped dataset so tests and CI never need the real data.
    """

    profile: str
    raw: Path
    processed: Path
    models: Path
    reports: Path

    @property
    def db(self) -> Path:
        return self.processed / "fraud.duckdb"

    @property
    def features(self) -> Path:
        return self.processed / "features.parquet"

    @property
    def scored(self) -> Path:
        return self.processed / "scored.parquet"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def serving_db(self) -> Path:
        return self.models / "serving.duckdb"

    def ensure(self) -> Paths:
        for p in (self.processed, self.models, self.reports, self.figures, self.reports / "sql"):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(profile: str = "full") -> Paths:
    if profile == "full":
        return Paths(
            "full", ROOT / "data/raw", ROOT / "data/processed", ROOT / "models", ROOT / "reports"
        )
    if profile == "sample":
        return Paths(
            "sample",
            ROOT / "data/sample",
            ROOT / "data/processed/sample",
            ROOT / "models/sample",
            ROOT / "reports/sample",
        )
    raise ValueError(f"unknown profile {profile!r} (expected 'full' or 'sample')")


def paths_at(base: Path) -> Paths:
    """Self-contained layout under one directory (used by tests)."""
    return Paths("custom", base / "raw", base / "processed", base / "models", base / "reports")


class Sensitivity(BaseModel):
    friction_cost: list[float]
    lost_margin_rate: list[float]
    review_cap: list[float]
    # second grid: how much of the saving depends on the manual-review assumptions
    reviewer_accuracy: list[float] = [1.0]
    review_cost: list[float] = [3.0]


class Costs(BaseModel):
    chargeback_fee: float
    friction_cost: float
    lost_margin_rate: float = 0.0
    review_cost: float
    review_cap: float
    reviewer_accuracy: float = Field(1.0, ge=0.0, le=1.0)
    sensitivity: Sensitivity | None = None


class SplitConfig(BaseModel):
    train_end: float
    valid_end: float
    calib_end: float


class LGBMConfig(BaseModel):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_lambda: float
    early_stopping_rounds: int


class TuneConfig(BaseModel):
    n_trials: int
    timeout_minutes: float
    learning_rate: float


class ExplainConfig(BaseModel):
    shap_sample: int


class BacktestConfig(BaseModel):
    window_days: int
    min_train_windows: int


class ModelConfig(BaseModel):
    seed: int
    lgbm: LGBMConfig
    scale_pos_weight_variant: float
    tune: TuneConfig
    explain: ExplainConfig
    backtest: BacktestConfig


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def costs() -> Costs:
    return Costs(**_load("costs.yaml"))


@cache
def split_config() -> SplitConfig:
    return SplitConfig(**_load("split.yaml"))


@cache
def model_config() -> ModelConfig:
    return ModelConfig(**_load("model.yaml"))
