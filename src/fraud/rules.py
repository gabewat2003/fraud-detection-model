"""A hand-written rules engine: the "what a fraud team might run today" baseline.

Each rule is something an analyst could explain to an auditor in one sentence. Tunable
thresholds are picked on the validation split (highest precision that still catches at
least 5% of fraud), never on test. The rule score is the number of rules a transaction
trips, which the policy layer can rank and calibrate like any model score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_RECALL = 0.05

RULES = {
    "velocity_1h": {
        "text": "card used {n}+ times in the previous hour",
        "fn": lambda d, n: d.card_tx_1h.fillna(0) >= n,
        "grid": [2, 3, 4, 5, 8],
    },
    "amount_spike": {
        "text": "amount is {n}x+ the account's average",
        "fn": lambda d, n: d.amt_vs_uid_mean.fillna(0) >= n,
        "grid": [3, 5, 10, 20],
    },
    "new_account_high_amount": {
        "text": "first transaction on the account and amount >= ${n}",
        "fn": lambda d, n: (d.is_first_txn_uid == 1) & (d.TransactionAmt >= n),
        "grid": [100, 200, 500, 1000],
    },
    "many_regions_7d": {
        "text": "card number seen with {n}+ billing regions in 7 days",
        "fn": lambda d, n: d.card1_n_addr_7d.fillna(0) >= n,
        "grid": [3, 5, 10, 20],
    },
    "email_mismatch_no_identity": {
        "text": "purchaser/recipient email mismatch and no identity record",
        "fn": lambda d, n: (d.email_match == 0) & (d.has_identity == 0),
        "grid": [None],
    },
}


def fit(valid: pd.DataFrame) -> dict:
    y = valid.isFraud.to_numpy()
    params = {}
    for name, rule in RULES.items():
        best = None
        for n in rule["grid"]:
            hit = rule["fn"](valid, n).to_numpy()
            if hit.sum() == 0:
                continue
            precision, recall = y[hit].mean(), y[hit].sum() / max(y.sum(), 1)
            if recall >= MIN_RECALL and (best is None or precision > best[1]):
                best = (n, precision, recall)
        if best is not None:
            params[name] = {"threshold": best[0], "valid_precision": best[1], "valid_recall": best[2],
                            "text": rule["text"].format(n=best[0])}  # fmt: skip
    return params


def hits(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pd.DataFrame({name: RULES[name]["fn"](df, p["threshold"]).to_numpy() for name, p in params.items()},
                        index=df.index)  # fmt: skip


def score(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Number of rules tripped, with amount as a tiny tie-breaker for ranking."""
    h = hits(df, params)
    n = h.sum(axis=1).to_numpy().astype(float)
    return n + 1e-6 * np.log1p(df.TransactionAmt.to_numpy())
