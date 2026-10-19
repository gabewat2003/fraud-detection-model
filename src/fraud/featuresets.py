"""Feature groups, named feature sets for experiments, and frame preparation.

`prepare_frame` is the single place that turns a feature table into model input
(column order, float32, category levels fixed from training). Training, evaluation and
the scoring API all call it, which is what keeps offline and online scores identical.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# never model inputs: identifiers, label, raw time, account keys, bookkeeping
EXCLUDE = {"TransactionID", "isFraud", "TransactionDT", "tx_day", "card_key", "uid", "split",
           "is_new_uid"}  # fmt: skip

TEXT_RAW = ["DeviceInfo", "id_30", "id_31", "P_emaildomain", "R_emaildomain"]
TEXT_NORM = [
    "device_brand", "device_family", "os_family", "os_version", "browser_family",
    "browser_major", "browser_is_mobile", "p_email_provider", "p_email_tld", "p_is_free_mail",
    "p_email_region", "r_email_provider", "r_email_tld", "r_is_free_mail",
    "email_provider_match", "browser_versions_behind", "freq_p_email", "freq_device",
]  # fmt: skip
KNOWN_RAW = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "DeviceType", "has_identity", "hour", "dow",
]  # fmt: skip
SIMPLE = ["amt_log", "amt_cents", "email_match"]
VELOCITY = [
    "card_tx_1h", "card_tx_24h", "card_tx_7d", "card_amt_24h", "uid_tx_1h", "uid_tx_24h",
    "uid_tx_7d", "card1_n_addr_7d", "card1_n_email_7d", "card1_n_device_7d",
    "card_secs_since_prev", "uid_secs_since_prev",
]  # fmt: skip
HISTORY = [
    "uid_prior_n", "uid_prior_mean_amt", "uid_prior_std_amt", "uid_prior_max_amt",
    "card_prior_n", "card_prior_mean_amt", "is_first_txn_uid", "amt_vs_uid_mean",
    "amt_zscore_uid", "amt_vs_uid_max", "amt_vs_card_mean",
]  # fmt: skip
FREQ = ["freq_card1", "freq_card2", "freq_addr1"]
MASKED_RX = re.compile(r"^(V\d+|C\d+|D\d+|M\d+|id_\d+)$")

DESCRIPTIONS = {
    "TransactionAmt": "transaction amount (USD)",
    "amt_log": "log(1 + amount)",
    "amt_cents": "cents part of the amount (round vs odd amounts)",
    "email_match": "purchaser and recipient email domains are identical",
    "email_provider_match": "purchaser and recipient email providers match after normalization",
    "has_identity": "transaction has an identity/device record",
    "hour": "hour of day (relative to unknown reference)",
    "dow": "day-of-week cohort label (not a calendar weekday)",
    "card_tx_1h": "earlier transactions on this card in the past hour",
    "card_tx_24h": "earlier transactions on this card in the past 24h",
    "card_tx_7d": "earlier transactions on this card in the past 7 days",
    "card_amt_24h": "amount spent on this card in the past 24h",
    "uid_tx_1h": "earlier transactions on this account in the past hour",
    "uid_tx_24h": "earlier transactions on this account in the past 24h",
    "uid_tx_7d": "earlier transactions on this account in the past 7 days",
    "card1_n_addr_7d": "distinct billing regions used with this card number in 7 days",
    "card1_n_email_7d": "distinct email providers used with this card number in 7 days",
    "card1_n_device_7d": "distinct device families used with this card number in 7 days",
    "card_secs_since_prev": "time since this card's previous transaction",
    "uid_secs_since_prev": "time since this account's previous transaction",
    "uid_prior_n": "number of earlier transactions on this account",
    "uid_prior_mean_amt": "account's average earlier amount",
    "uid_prior_std_amt": "spread of the account's earlier amounts",
    "uid_prior_max_amt": "account's largest earlier amount",
    "card_prior_n": "number of earlier transactions on this card key",
    "card_prior_mean_amt": "card key's average earlier amount",
    "is_first_txn_uid": "first transaction seen for this account",
    "amt_vs_uid_mean": "amount relative to the account's average",
    "amt_zscore_uid": "amount z-score against the account's history",
    "amt_vs_uid_max": "amount relative to the account's largest earlier amount",
    "amt_vs_card_mean": "amount relative to the card key's average",
    "freq_card1": "how common this card1 value was in training",
    "freq_card2": "how common this card2 value was in training",
    "freq_addr1": "how common this billing region was in training",
    "freq_p_email": "how common this email provider was in training",
    "freq_device": "how common this device family was in training",
    "device_brand": "device brand (normalized from DeviceInfo)",
    "device_family": "device family (normalized from DeviceInfo)",
    "os_family": "operating system family (from id_30)",
    "os_version": "operating system version (from id_30)",
    "browser_family": "browser family (from id_31)",
    "browser_major": "browser major version (from id_31)",
    "browser_is_mobile": "mobile browser",
    "browser_versions_behind": "major versions behind the newest seen for this browser",
    "p_email_provider": "purchaser email provider (normalized)",
    "p_email_tld": "purchaser email top-level domain",
    "p_is_free_mail": "purchaser uses a free webmail provider",
    "p_email_region": "purchaser email country (from TLD)",
    "r_email_provider": "recipient email provider (normalized)",
    "r_email_tld": "recipient email top-level domain",
    "r_is_free_mail": "recipient uses a free webmail provider",
    "ProductCD": "product code",
    "card4": "card network",
    "card6": "card type (debit/credit)",
    "addr1": "billing region (masked code)",
    "dist1": "distance between addresses (masked)",
    "DeviceType": "device type (mobile/desktop)",
}


def group_of(col: str) -> str:
    if col in VELOCITY:
        return "velocity"
    if col in HISTORY:
        return "history"
    if col in FREQ:
        return "frequency"
    if col in TEXT_NORM:
        return "text_norm"
    if col in TEXT_RAW:
        return "text_raw"
    if col in SIMPLE:
        return "simple"
    if col in KNOWN_RAW:
        return "known_raw"
    if MASKED_RX.match(col):
        return "masked"
    return "other"


INTERPRETABLE_GROUPS = {"velocity", "history", "frequency", "text_norm", "simple", "known_raw"}


def manifest(columns: list[str]) -> list[dict]:
    return [
        {
            "name": c,
            "group": group_of(c),
            "interpretable": group_of(c) in INTERPRETABLE_GROUPS,
            "description": DESCRIPTIONS.get(
                c, "masked vendor feature" if MASKED_RX.match(c) else ""
            ),
        }
        for c in columns
        if c not in EXCLUDE
    ]


FEATURE_SETS = {
    # everything, with text fields normalized (the default)
    "full": {"known_raw", "masked", "text_norm", "velocity", "history", "frequency", "simple"},
    # ablation: raw text strings instead of normalized fields
    "raw_text": {"known_raw", "masked", "text_raw", "velocity", "history", "frequency", "simple"},
    # ablation: no engineered behavioral features
    "no_behavioral": {"known_raw", "masked", "text_norm", "simple"},
    # ablation: only features a human can explain (drops V*/C*/D*/M*/id_*)
    "interpretable": {"known_raw", "text_norm", "velocity", "history", "frequency", "simple"},
}

# Logistic regression baseline: small, numeric, explainable.
LR_FEATURES = [
    "amt_log", "amt_cents", "has_identity", "email_match", "hour", "freq_card1", "freq_addr1",
    "freq_p_email", "card_tx_1h", "card_tx_24h", "uid_tx_24h", "uid_prior_n",
    "is_first_txn_uid", "amt_vs_uid_mean", "card1_n_addr_7d", "p_is_free_mail", "dist1",
]  # fmt: skip


def feature_columns(columns: list[str], feature_set: str) -> list[str]:
    groups = FEATURE_SETS[feature_set]
    return [c for c in columns if c not in EXCLUDE and group_of(c) in groups]


def is_categorical(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))


def fit_categories(train: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    """Category levels from the training window only (unseen levels become NaN later)."""
    return {
        c: sorted(train[c].dropna().astype(str).unique().tolist())
        for c in features
        if is_categorical(train[c])
    }


def prepare_frame(
    df: pd.DataFrame, features: list[str], categories: dict[str, list[str]]
) -> pd.DataFrame:
    cols = {}
    for c in features:
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)
        if c in categories:
            vals = s.astype(object)
            vals = vals.where(vals.isin(categories[c]), None)  # unseen levels -> missing
            cols[c] = pd.Categorical(vals, categories=categories[c])
        else:
            cols[c] = pd.to_numeric(s, errors="coerce").astype(np.float32).to_numpy()
    return pd.DataFrame(cols, index=df.index)
