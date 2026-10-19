import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraud import evaluate as E


def test_average_precision_recomputed_independently():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.04).astype(int)
    p = np.clip(rng.normal(0.2 + 0.4 * y, 0.2), 0, 1)
    assert E.average_precision_manual(y, p) == pytest.approx(
        average_precision_score(y, p), abs=1e-12
    )


def test_precision_recall_at_k_toy():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    prec, rec = E.precision_recall_at_k(y, p, 0.3)  # top 3 rows
    assert prec == pytest.approx(2 / 3)
    assert rec == pytest.approx(2 / 3)


def test_dollar_recall_weights_by_amount():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3])
    amt = np.array([10.0, 990.0, 5.0, 5.0])
    assert E.dollar_recall_at_k(y, p, amt, 0.25) == pytest.approx(
        0.01
    )  # count recall would say 50%


def test_recall_at_fpr():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.8, 0.3, 0.2, 0.1])
    # no false positives allowed: only the 0.9 fraud is caught
    assert E.recall_at_fpr(y, p, 0.0) == pytest.approx(0.5)
    # one of four negatives (the 0.8) may be flagged: threshold 0.4 then catches both frauds
    assert E.recall_at_fpr(y, p, 0.25) == pytest.approx(1.0)


def test_random_scores_get_base_rate_pr_auc():
    rng = np.random.default_rng(1)
    y = (rng.random(200_000) < 0.035).astype(int)
    assert E.metrics(y, rng.random(len(y)), np.ones(len(y)))["pr_auc"] == pytest.approx(
        0.035, abs=0.003
    )


def test_bootstrap_by_day_brackets_point_estimate():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"tx_day": rng.integers(0, 30, 3000), "v": rng.normal(5, 1, 3000)})
    lo, hi = E.bootstrap_by_day(df, lambda d: d.v.mean(), n_boot=200)
    assert lo < df.v.mean() < hi
