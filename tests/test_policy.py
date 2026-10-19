"""Decision layer: hand-checked cost arithmetic (the README's "spreadsheet check")."""

import numpy as np
import pytest

from fraud import policy as P
from fraud.config import Costs

C = Costs(chargeback_fee=15, friction_cost=5, review_cost=3, review_cap=0.01)


def test_three_transactions_by_hand():
    #            p     amount   E[approve]=p(a+15)   E[decline]=(1-p)5   best
    #  t1:     0.01    $20      0.35                 4.95                approve
    #  t2:     0.30    $10      7.50                 3.50                decline
    #  t3:     0.02    $1000    20.30                4.90                decline
    p = np.array([0.01, 0.30, 0.02])
    amt = np.array([20.0, 10.0, 1000.0])
    e_a, e_d, e_r = P.expected_costs(p, amt, C)
    np.testing.assert_allclose(e_a, [0.35, 7.5, 20.3])
    np.testing.assert_allclose(e_d, [4.95, 3.5, 4.9])
    np.testing.assert_allclose(e_r, [3, 3, 3])
    assert P.amount_aware(p, amt, C).tolist() == [P.APPROVE, P.DECLINE, P.DECLINE]


def test_break_even_falls_with_amount():
    be = P.break_even_probability([10, 100, 1000], C)
    np.testing.assert_allclose(be, [5 / 30, 5 / 120, 5 / 1020])
    assert be[0] > be[1] > be[2]


def test_amount_aware_matches_break_even_rule():
    rng = np.random.default_rng(0)
    p, amt = rng.random(5000), rng.lognormal(4, 1, 5000)
    expected = np.where(p > P.break_even_probability(amt, C), P.DECLINE, P.APPROVE)
    assert (P.amount_aware(p, amt, C) == expected).all()


def test_realized_cost_per_outcome():
    dec = np.array([P.APPROVE, P.APPROVE, P.DECLINE, P.DECLINE, P.REVIEW, P.REVIEW])
    y = np.array([1, 0, 1, 0, 1, 0])
    amt = np.array([100.0, 100, 100, 100, 100, 100])
    assert P.realized_cost(dec, y, amt, C).tolist() == [115, 0, 0, 5, 3, 3]


def test_summary_totals_add_up():
    rng = np.random.default_rng(1)
    n = 2000
    dec, y, amt = rng.integers(0, 3, n), (rng.random(n) < 0.05).astype(int), rng.lognormal(4, 1, n)
    s = P.summarize(dec, y, amt, C)
    assert s["total_cost"] == pytest.approx(
        s["cost_fraud_loss"] + s["cost_friction"] + s["cost_review"]
    )
    assert s["fraud_usd_caught"] + s["fraud_usd_missed"] == pytest.approx(amt[y == 1].sum())


def test_review_threshold_respects_capacity():
    rng = np.random.default_rng(2)
    p, amt = rng.beta(0.5, 5, 10_000), rng.lognormal(4, 1, 10_000)
    t = P.fit_review_threshold(p, amt, C)
    dec = P.three_tier(p, amt, C, t)
    assert (dec == P.REVIEW).sum() <= int(C.review_cap * len(p))
    assert (dec == P.REVIEW).sum() >= int(C.review_cap * len(p)) - 1


def test_review_only_where_it_beats_both_automatic_actions():
    p, amt = np.array([0.001, 0.5, 0.999]), np.array([50.0, 50.0, 50.0])
    dec = P.three_tier(p, amt, C.model_copy(update={"review_cap": 1.0}), 0.0)
    # p=0.001: approve costs 0.065; p=0.5: decline (2.5) beats review (3); p=0.999: decline
    assert dec.tolist() == [P.APPROVE, P.DECLINE, P.DECLINE]
    dec = P.three_tier(
        np.array([0.2]), np.array([50.0]), C.model_copy(update={"review_cap": 1.0}), 0.0
    )
    assert dec.tolist() == [P.REVIEW]  # approve 13, decline 4, review 3


def test_cost_curve_agrees_with_summary():
    rng = np.random.default_rng(3)
    p, amt = rng.random(3000), rng.lognormal(4, 1, 3000)
    y = (rng.random(3000) < p * 0.2).astype(int)
    curve = P.cost_curve(p, y, amt, C)
    row = curve.iloc[len(curve) // 2]
    direct = P.summarize(P.decline_above(p, row.threshold), y, amt, C)["total_cost"]
    assert row.total_cost == pytest.approx(direct)
    t = P.fit_cost_optimal_threshold(p, y, amt, C)
    assert P.summarize(P.decline_above(p, t), y, amt, C)["total_cost"] == pytest.approx(
        curve.total_cost.min()
    )


def test_rules_policy():
    dec = P.rules_policy(np.array([0, 1, 1, 2, 3]), np.array([10.0, 10, 500, 1, 1]), amount_bar=100)
    assert dec.tolist() == [P.APPROVE, P.APPROVE, P.REVIEW, P.DECLINE, P.DECLINE]


def test_policy_report_outputs(trained_paths):
    from fraud.io import load_json

    out = load_json(trained_paths.reports / "policy_results.json")
    by_name = {r["policy"]: r for r in out["policies"]}
    assert by_name["Approve everything"]["savings_vs_approve_all"] == 0
    three = by_name["Model: three-tier with review cap"]
    assert three["total_cost"] <= by_name["Approve everything"]["total_cost"]
    assert (trained_paths.models / "policy.json").exists()
