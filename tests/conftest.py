import os

import pandas as pd
import pytest

os.environ["FRAUD_MLFLOW"] = "0"

from fraud import features, io, report, synth, train  # noqa: E402
from fraud.config import paths_at  # noqa: E402


@pytest.fixture(scope="session")
def sample_paths(tmp_path_factory):
    """Synthetic IEEE-shaped data pushed through ingest + feature SQL once per test session."""
    base = tmp_path_factory.mktemp("pipeline")
    paths = paths_at(base).ensure()
    synth.write(paths.raw, seed=1, n_cards=300)
    io.ingest(paths)
    features.build(paths)
    return paths


@pytest.fixture(scope="session")
def feature_df(sample_paths):
    return pd.read_parquet(sample_paths.features)


@pytest.fixture(scope="session")
def trained_paths(sample_paths):
    """Sample pipeline after training (no ablations) and the policy step."""
    train.run(sample_paths, include_ablations=False)
    report.policy_report(sample_paths)
    return sample_paths


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
