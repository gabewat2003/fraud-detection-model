"""Turn calibrated fraud probabilities into approve / review / decline decisions.

Cost model (assumptions in configs/costs.yaml, varied in the sensitivity grid):
    approve a fraud   -> lose amount + chargeback fee
    decline a good    -> friction (support contact, churn risk) + the margin on the lost sale
    review anything   -> reviewer cost; the reviewer is assumed to resolve it correctly
    approve a good / decline a fraud -> 0

For one transaction with fraud probability p and amount a, with lost margin rate m:
    E[cost | approve] = p * (a + fee)
    E[cost | decline] = (1 - p) * (friction + m * a)
    E[cost | review]  = review_cost
Declining beats approving when p > (friction + m*a) / (friction + m*a + a + fee). Both
sides grow with the amount, so the break-even probability falls from roughly
friction / (friction + fee) on tiny orders toward m / (1 + m) on large ones.

The lost-margin term matters: with a flat friction cost only, declining a $500 order
looks as cheap as declining a $5 one, and the "optimal" policy declines a large share of
good customers (the first version of this project declined 15% of them).

Every threshold below is fit on the calib split and then applied unchanged to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.config import Costs

APPROVE, REVIEW, DECLINE = 0, 1, 2
LABELS = {APPROVE: "approve", REVIEW: "review", DECLINE: "decline"}


# ----------------------------------------------------------------------- cost algebra


def expected_costs(p, amt, c: Costs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, amt = np.asarray(p, dtype=float), np.asarray(amt, dtype=float)
    e_approve = p * (amt + c.chargeback_fee)
    e_decline = (1 - p) * decline_cost(amt, c)
    e_review = np.full_like(p, c.review_cost)
    return e_approve, e_decline, e_review


def decline_cost(amt, c: Costs) -> np.ndarray:
    """Cost of wrongly declining a legitimate order: friction plus the lost margin."""
    return c.friction_cost + c.lost_margin_rate * np.asarray(amt, dtype=float)


def break_even_probability(amt, c: Costs) -> np.ndarray:
    amt = np.asarray(amt, dtype=float)
    d = decline_cost(amt, c)
    return d / (d + amt + c.chargeback_fee)


def realized_cost(dec, y, amt, c: Costs) -> np.ndarray:
    """What each decision actually cost, given the true label."""
    dec, y, amt = np.asarray(dec), np.asarray(y), np.asarray(amt, dtype=float)
    cost = np.zeros(len(dec))
    cost[(dec == APPROVE) & (y == 1)] = amt[(dec == APPROVE) & (y == 1)] + c.chargeback_fee
    good_declined = (dec == DECLINE) & (y == 0)
    cost[good_declined] = decline_cost(amt[good_declined], c)
    cost[dec == REVIEW] = c.review_cost
    return cost


def summarize(dec, y, amt, c: Costs) -> dict:
    dec, y, amt = np.asarray(dec), np.asarray(y), np.asarray(amt, dtype=float)
    fraud = y == 1
    stopped = fraud & (dec != APPROVE)
    good_declined = (~fraud) & (dec == DECLINE)
    cost = realized_cost(dec, y, amt, c)
    return {
        "fraud_usd_caught": float(amt[stopped].sum()),
        "fraud_usd_missed": float(amt[fraud & (dec == APPROVE)].sum()),
        "fraud_dollar_recall": float(amt[stopped].sum() / max(amt[fraud].sum(), 1e-9)),
        "fraud_count_recall": float(stopped.sum() / max(fraud.sum(), 1)),
        "good_declined": int(good_declined.sum()),
        "good_declined_rate": float(good_declined.sum() / max((~fraud).sum(), 1)),
        "reviews": int((dec == REVIEW).sum()),
        "review_rate": float((dec == REVIEW).mean()),
        "decline_rate": float((dec == DECLINE).mean()),
        "cost_fraud_loss": float(cost[fraud & (dec == APPROVE)].sum()),
        "cost_friction": float(cost[good_declined].sum()),
        "cost_review": float(cost[dec == REVIEW].sum()),
        "total_cost": float(cost.sum()),
    }


# ------------------------------------------------------------------------- policies


def approve_all(n: int) -> np.ndarray:
    return np.full(n, APPROVE)


def decline_above(p, threshold: float) -> np.ndarray:
    return np.where(np.asarray(p) > threshold, DECLINE, APPROVE)


def amount_aware(p, amt, c: Costs) -> np.ndarray:
    """Decline exactly when the expected cost of approving exceeds that of declining."""
    e_a, e_d, _ = expected_costs(p, amt, c)
    return np.where(e_a > e_d, DECLINE, APPROVE)


def review_benefit(p, amt, c: Costs) -> np.ndarray:
    """Expected saving from sending a case to review instead of the best automatic action."""
    e_a, e_d, e_r = expected_costs(p, amt, c)
    return np.minimum(e_a, e_d) - e_r


def three_tier(p, amt, c: Costs, review_threshold: float) -> np.ndarray:
    """Review where it saves the most (above a capacity-derived bar), else approve/decline."""
    e_a, e_d, _ = expected_costs(p, amt, c)
    dec = np.where(e_a > e_d, DECLINE, APPROVE)
    benefit = review_benefit(p, amt, c)
    dec[benefit > review_threshold] = REVIEW
    return dec


def fit_review_threshold(p_calib, amt_calib, c: Costs) -> float:
    """Smallest benefit bar that keeps reviews within capacity on the calibration window.

    Only cases where review beats both automatic actions qualify (benefit > 0). If more of
    them exist than the reviewers can handle, the bar rises until the volume fits the cap.
    """
    benefit = review_benefit(p_calib, amt_calib, c)
    cap_n = int(np.floor(c.review_cap * len(benefit)))
    positive = np.sort(benefit[benefit > 0])[::-1]
    if cap_n == 0:
        return float("inf")
    if len(positive) <= cap_n:
        return 0.0
    return float(positive[cap_n])  # strictly-greater-than this keeps exactly cap_n


def fit_global_threshold(p_calib, share: float) -> float:
    """Threshold that flags the top `share` of calibration traffic."""
    return float(np.quantile(np.asarray(p_calib), 1 - share))


def cost_curve(p, y, amt, c: Costs, thresholds=None) -> pd.DataFrame:
    """Total cost of the flat policy 'decline if p > t' across thresholds."""
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(p, np.linspace(0.80, 0.9995, 300)))
    rows = []
    for t in thresholds:
        s = summarize(decline_above(p, t), y, amt, c)
        rows.append({"threshold": float(t), "flag_rate": s["decline_rate"], "total_cost": s["total_cost"],
                     "fraud_dollar_recall": s["fraud_dollar_recall"],
                     "good_declined_rate": s["good_declined_rate"]})  # fmt: skip
    return pd.DataFrame(rows)


def fit_cost_optimal_threshold(p_calib, y_calib, amt_calib, c: Costs) -> float:
    curve = cost_curve(p_calib, y_calib, amt_calib, c)
    return float(curve.loc[curve.total_cost.idxmin(), "threshold"])


def rules_policy(n_rules, amt, amount_bar: float) -> np.ndarray:
    """Rules as an ops team would run them: 2+ rules -> decline, 1 rule and big -> review."""
    n_rules, amt = np.asarray(n_rules), np.asarray(amt, dtype=float)
    dec = np.full(len(n_rules), APPROVE)
    dec[(n_rules == 1) & (amt >= amount_bar)] = REVIEW
    dec[n_rules >= 2] = DECLINE
    return dec


def fit_rules_amount_bar(n_rules_calib, amt_calib, cap: float) -> float:
    one = np.asarray(amt_calib, dtype=float)[np.asarray(n_rules_calib) == 1]
    cap_n = int(np.floor(cap * len(n_rules_calib)))
    if len(one) <= cap_n:
        return 0.0
    if cap_n == 0:
        return float("inf")
    return float(np.sort(one)[::-1][cap_n - 1])


# ----------------------------------------------------------------- uncertainty / ops


def daily_cost_table(decisions: dict[str, np.ndarray], y, amt, day, c: Costs) -> pd.DataFrame:
    d = pd.DataFrame({name: realized_cost(dec, y, amt, c) for name, dec in decisions.items()})
    d["day"] = np.asarray(day)
    return d.groupby("day").sum()


def bootstrap_savings(
    daily: pd.DataFrame, policy: str, baseline: str, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of 1 - cost(policy)/cost(baseline), resampling whole days."""
    rng = np.random.default_rng(seed)
    a, b = daily[policy].to_numpy(), daily[baseline].to_numpy()
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    ratio = 1 - a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return float(lo), float(hi)


def review_queue(dec, day, cap: float) -> pd.DataFrame:
    """Daily review volume against the capacity it was sized for."""
    d = pd.DataFrame({"day": np.asarray(day), "review": np.asarray(dec) == REVIEW})
    g = d.groupby("day").agg(volume=("review", "size"), reviews=("review", "sum"))
    g["capacity"] = np.floor(cap * g.volume)
    g["over_capacity"] = np.maximum(g.reviews - g.capacity, 0)
    return g.reset_index()
