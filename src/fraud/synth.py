"""Generate a small IEEE-CIS-shaped dataset for tests and CI.

The real competition data cannot be redistributed, so the test suite and CI run the whole
pipeline on this instead. It copies the real schema (column names, types, missingness
pattern, messy device/browser/email strings) and plants a few learnable fraud patterns:
card-testing bursts, out-of-pattern amounts on established cards, and identity-less
transactions with mismatched emails. Numbers produced on it mean nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86_400
START_DT = DAY  # the real data also starts one day after its unknown reference point

DEVICE_STRINGS = [
    "SM-G930V Build/NRD90M", "SM-G950U Build/R16NW", "samsung SM-G935F", "SM-J700M",
    "Moto G (5) Plus Build/NPNS25.137-35-5", "moto e5 play", "XT1635-02 Build/NPNS26.118",
    "LG-K500 Build/MXB48T", "LG-H870", "HUAWEI ALE-L23 Build/HuaweiALE-L23",
    "ALE-L21 Build/HuaweiALE-L21", "Pixel 2 Build/OPM1.171019.011", "Redmi Note 4",
    "iOS Device", "Windows", "MacOS", "Trident/7.0", "rv:11.0", "Linux x86_64", "ZTE Z983",
]  # fmt: skip
OS_STRINGS = [
    "Windows 10", "Windows 7", "iOS 11.2.1", "iOS 10.3.3", "Android 7.0", "Android 8.0.0",
    "Mac OS X 10_13_2", "Mac OS X 10.12", "Linux", "Android", "other",
]  # fmt: skip
BROWSER_STRINGS = [
    "chrome 63.0", "chrome 62.0 for android", "mobile safari 11.0", "mobile safari 10.0",
    "ie 11.0 for desktop", "firefox 57.0", "samsung browser 6.2", "safari generic",
    "edge 16.0", "chrome generic", "opera 49.0", "google search application 48.0",
]  # fmt: skip
EMAILS = [
    "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "aol.com",
    "icloud.com", "comcast.net", "yahoo.com.mx", "hotmail.es", "gmail", "live.com",
    "protonmail.com", "att.net", "me.com",
]  # fmt: skip
EMAIL_P = np.array([30, 15, 10, 8, 6, 5, 4, 3, 3, 2, 2, 3, 1, 2, 2], dtype=float)
PRODUCTS = ["W", "C", "R", "H", "S"]


def generate(
    n_cards: int = 900, days: int = 182, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    cards = pd.DataFrame(
        {
            "card1": rng.choice(np.arange(1000, 18000), n_cards, replace=False),
            "card2": np.where(rng.random(n_cards) < 0.02, np.nan, rng.integers(100, 600, n_cards)),
            "card3": rng.choice([150, 185, 106], n_cards, p=[0.9, 0.07, 0.03]).astype(float),
            "card4": rng.choice(
                ["visa", "mastercard", "american express", "discover"],
                n_cards,
                p=[0.65, 0.3, 0.03, 0.02],
            ),
            "card5": rng.choice([226, 224, 166, 102, 117], n_cards).astype(float),
            "card6": rng.choice(["debit", "credit"], n_cards, p=[0.75, 0.25]),
            "addr1": np.where(rng.random(n_cards) < 0.1, np.nan, rng.integers(100, 540, n_cards)),
            "email": rng.choice(EMAILS, n_cards, p=EMAIL_P / EMAIL_P.sum()),
            "device": rng.choice(DEVICE_STRINGS, n_cards),
            "os": rng.choice(OS_STRINGS, n_cards),
            "browser": rng.choice(BROWSER_STRINGS, n_cards),
            "start_day": rng.integers(-400, days, n_cards),
            "rate": rng.gamma(1.5, 0.12, n_cards),  # transactions per day
            "typical_amt": rng.lognormal(3.8, 0.8, n_cards),
            "identity_p": rng.choice([0.05, 0.25, 0.9], n_cards, p=[0.5, 0.3, 0.2]),
        }
    )
    compromised = rng.random(n_cards) < 0.12

    rows = []
    for i, c in cards.iterrows():
        first = max(c.start_day, 0)
        n = rng.poisson(c.rate * (days - first)) + 1
        t = np.sort(rng.uniform(first * DAY, days * DAY, n))
        amt = np.round(c.typical_amt * rng.lognormal(0, 0.35, n), 2)
        for dt, a in zip(t, amt, strict=True):
            rows.append((i, dt, a, 0))
        if compromised[i]:
            # card-testing burst: several small charges inside an hour, then large ones
            t0 = rng.uniform(first * DAY, days * DAY - 4 * 3600)
            for k in range(rng.integers(3, 8)):
                rows.append((i, t0 + k * rng.uniform(60, 500), round(rng.uniform(1, 10), 2), 1))
            for k in range(rng.integers(1, 4)):
                rows.append(
                    (i, t0 + 3600 + k * 900, round(c.typical_amt * rng.uniform(4, 12), 2), 1)
                )

    tx = pd.DataFrame(rows, columns=["card_idx", "TransactionDT", "TransactionAmt", "isFraud"])
    tx = tx[tx.TransactionDT < days * DAY].copy()
    n = len(tx)

    # sporadic fraud unrelated to bursts: identity-less, mismatched email, odd hours
    tx["isFraud"] = np.where((tx.isFraud == 0) & (rng.random(n) < 0.012), 1, tx.isFraud)

    tx["TransactionDT"] = (START_DT + tx.TransactionDT).astype(np.int64)
    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    fraud = tx.isFraud.to_numpy() == 1
    tx["TransactionID"] = 2_987_000 + np.arange(n)

    c = cards.loc[tx.card_idx].reset_index(drop=True)
    for col in ("card1", "card2", "card3", "card4", "card5", "card6", "addr1"):
        tx[col] = c[col].to_numpy()
    tx["addr2"] = np.where(tx.addr1.isna(), np.nan, 87.0)
    tx["dist1"] = np.where(rng.random(n) < 0.6, np.nan, rng.exponential(30, n).round())
    tx["dist2"] = np.where(rng.random(n) < 0.93, np.nan, rng.exponential(200, n).round())
    tx["ProductCD"] = rng.choice(PRODUCTS, n, p=[0.74, 0.12, 0.06, 0.05, 0.03])
    tx.loc[fraud, "ProductCD"] = rng.choice(PRODUCTS, fraud.sum(), p=[0.45, 0.35, 0.08, 0.07, 0.05])
    tx["P_emaildomain"] = np.where(rng.random(n) < 0.15, None, c.email.to_numpy())
    r_email = np.where(rng.random(n) < 0.75, None, c.email.to_numpy())
    swap = fraud & (rng.random(n) < 0.6)
    r_email[swap] = rng.choice(EMAILS, swap.sum())
    tx["R_emaildomain"] = r_email

    tx_day = (tx.TransactionDT // DAY).to_numpy()
    tx["D1"] = np.maximum(tx_day - c.start_day.clip(lower=-400).to_numpy() - 1, 0).astype(float)
    tx.loc[rng.random(n) < 0.002, "D1"] = np.nan
    for k in range(2, 16):
        tx[f"D{k}"] = np.where(rng.random(n) < 0.5, np.nan, rng.integers(0, 500, n).astype(float))
    for k in range(1, 15):
        lam = 1.5 + 2.5 * fraud if k in (1, 2) else 1.0
        tx[f"C{k}"] = rng.poisson(np.broadcast_to(lam, n)).astype(float)
    for k in range(1, 10):
        tx[f"M{k}"] = rng.choice(["T", "F", None], n, p=[0.5, 0.2, 0.3])
    tx.loc[swap, "M4"] = "M2"
    for k in range(1, 21):
        shift = 0.8 * fraud if k in (1, 2, 3) else 0
        v = rng.normal(0, 1, n) + shift
        tx[f"V{k}"] = np.where(rng.random(n) < 0.3, np.nan, v.round(4))

    # identity rows exist for a minority of transactions, more often for fraud
    p_id = c.identity_p.to_numpy() + 0.3 * fraud
    has_id = rng.random(n) < np.clip(p_id, 0, 1)
    idn = pd.DataFrame({"TransactionID": tx.TransactionID[has_id].to_numpy()})
    m = len(idn)
    ci = c[has_id].reset_index(drop=True)
    for k in range(1, 12):
        idn[f"id_{k:02d}"] = np.where(rng.random(m) < 0.3, np.nan, rng.normal(0, 10, m).round(1))
    for k in range(12, 30):
        idn[f"id_{k}"] = rng.choice(["Found", "NotFound", None], m, p=[0.5, 0.4, 0.1])
    idn["id_30"] = np.where(rng.random(m) < 0.5, None, ci.os.to_numpy())
    idn["id_31"] = np.where(rng.random(m) < 0.05, None, ci.browser.to_numpy())
    idn["id_32"] = rng.choice([24.0, 32.0, np.nan], m)
    idn["id_33"] = rng.choice(["1920x1080", "1366x768", "2208x1242", None], m)
    for k in range(34, 39):
        idn[f"id_{k}"] = rng.choice(["T", "F", None], m)
    mobile = ci.device.str.contains("SM-|Moto|moto|LG|HUAWEI|ALE|Pixel|Redmi|iOS|XT|ZTE").to_numpy()
    idn["DeviceType"] = np.where(mobile, "mobile", "desktop")
    idn["DeviceInfo"] = np.where(rng.random(m) < 0.15, None, ci.device.to_numpy())

    lead = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    rest = [col for col in tx.columns if col not in lead and col not in ("card_idx",)]
    return tx[lead + rest], idn


def write(out_dir: Path, seed: int = 0, n_cards: int = 900) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tx, idn = generate(n_cards=n_cards, seed=seed)
    tx.to_csv(out_dir / "train_transaction.csv", index=False)
    idn.to_csv(out_dir / "train_identity.csv", index=False)
    return {"transactions": len(tx), "identity": len(idn), "fraud_rate": float(tx.isFraud.mean())}
