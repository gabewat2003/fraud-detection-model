"""Scoring service: online features must equal batch features (no train/serve skew)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud import featuresets as fs
from fraud.serving import Scorer, example_payload


@pytest.fixture(scope="module")
def scorer(trained_paths):
    return Scorer(trained_paths.models)


@pytest.fixture(scope="module")
def client(trained_paths, monkeypatch_module):
    monkeypatch_module.setenv("FRAUD_MODELS_DIR", str(trained_paths.models))
    from api.main import app

    with TestClient(app) as c:
        yield c


def _test_ids(paths, n=15):
    scored = pd.read_parquet(paths.scored)
    test = scored[scored.split == "test"]
    return [
        *test.nlargest(3, "p_lgbm").TransactionID,
        *test.sample(n - 3, random_state=0).TransactionID,
    ]


def test_online_features_equal_batch_features(trained_paths, scorer):
    batch = pd.read_parquet(trained_paths.features).set_index("TransactionID")
    for tid in _test_ids(trained_paths):
        online = scorer.build_features(example_payload(trained_paths.db, int(tid)))
        a = fs.prepare_frame(online, scorer.features, scorer.categories).reset_index(drop=True)
        b = fs.prepare_frame(batch.loc[[tid]].reset_index(), scorer.features, scorer.categories)
        pd.testing.assert_frame_equal(a, b.reset_index(drop=True), check_exact=True)


def test_online_scores_match_batch_scores(trained_paths, scorer):
    scored = pd.read_parquet(trained_paths.scored).set_index("TransactionID")
    champ = scorer.meta["model_name"]
    for tid in _test_ids(trained_paths):
        r = scorer.score(example_payload(trained_paths.db, int(tid)))
        # features are identical; the tiny tolerance covers LightGBM's text model serialization
        assert r["p_raw"] == pytest.approx(float(scored.loc[tid, f"p_{champ}"]), abs=1e-4)


def test_transaction_without_identity_scores(scorer, trained_paths):
    payload = {
        k: v
        for k, v in example_payload(trained_paths.db, int(_test_ids(trained_paths)[0])).items()
        if not (k.startswith("id_") or k in ("DeviceType", "DeviceInfo"))
    }
    r = scorer.score(payload)
    assert 0 <= r["p_calibrated"] <= 1
    assert r["decision"] in {"approve", "review", "decline"}


def test_health_and_score_endpoint(client, trained_paths):
    assert client.get("/health").json()["status"] == "ok"
    payload = example_payload(trained_paths.db, int(_test_ids(trained_paths)[0]))
    payload = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in payload.items()}
    r = client.post("/score", json={"transaction": payload})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in {"approve", "review", "decline"}
    assert len(body["reason_codes"]) <= 3


def test_cost_override_changes_break_even(client, trained_paths):
    payload = example_payload(trained_paths.db, int(_test_ids(trained_paths)[0]))
    low = client.post("/score", json={"transaction": payload, "costs": {"friction_cost": 1}}).json()
    high = client.post(
        "/score", json={"transaction": payload, "costs": {"friction_cost": 50}}
    ).json()
    assert low["break_even_probability"] < high["break_even_probability"]


def test_invalid_amount_rejected(client):
    r = client.post("/score", json={"transaction": {"TransactionDT": 1, "TransactionAmt": -5,
                                                    "ProductCD": "W", "card1": 1}})  # fmt: skip
    assert r.status_code == 422
