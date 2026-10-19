"""Online scoring that reuses the batch feature SQL, so offline and online features match.

For one incoming transaction the scorer:
  1. writes it into empty `transactions` / `identity` tables shaped like the training CSVs
  2. runs sql/features/01 (join) and 02 (keys + normalized text) unchanged
  3. builds `window_input` from the stored history of the same card1 plus the new row,
     then runs 03 (velocity), 04 (history) and 06 (assemble) unchanged
  4. prepares the row with the training-time feature list and category levels, scores it,
     calibrates, applies the three-tier policy and attaches reason codes

Every card_key and uid starts with card1, so the card1 history is a superset of every
window partition the new row belongs to. tests/test_api.py replays test transactions
through this path and checks the scores equal the batch scores.

This is a demo service: history is a read-only snapshot and nothing is written back.
"""

from __future__ import annotations

import contextlib
import pickle
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud import db, explain, policy
from fraud import featuresets as fs
from fraud.config import Costs, costs
from fraud.features import FREQ_TABLES, WINDOW_COLS, distinct_text_values, register_maps
from fraud.io import load_json


class Scorer:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.meta = load_json(self.models_dir / "metadata.json")
        self.booster = lgb.Booster(model_file=str(self.models_dir / self.meta["model_file"]))
        with open(self.models_dir / self.meta["calibrator_file"], "rb") as f:
            self.calibrator = pickle.load(f)["calibrator"]
        pol_path = self.models_dir / "policy.json"
        self.policy = load_json(pol_path) if pol_path.exists() else {}
        self.features, self.categories = self.meta["features"], self.meta["categories"]

        self.conn = db.connect()
        self.conn.execute(
            f"ATTACH '{(self.models_dir / 'serving.duckdb').as_posix()}' AS srv (READ_ONLY)"
        )
        for t in FREQ_TABLES:
            self.conn.execute(f"CREATE TABLE {t} AS SELECT * FROM srv.{t}")
        self.tx_cols = [r[0] for r in self.conn.execute("DESCRIBE srv.tx_template").fetchall()]
        self.id_cols = [r[0] for r in self.conn.execute("DESCRIBE srv.id_template").fetchall()]
        self.next_id = self.conn.execute(
            "SELECT max(TransactionID) + 1 FROM srv.window_store"
        ).fetchone()[0]
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return f"{self.meta['model_name']}@{self.meta['trained_at']}"

    def build_features(self, txn: dict) -> pd.DataFrame:
        txn = dict(txn)
        txn.setdefault("TransactionID", self.next_id)
        tx_row = {k: v for k, v in txn.items() if k in self.tx_cols}
        id_row = {k: v for k, v in txn.items() if k in self.id_cols and k != "TransactionID"}
        c = self.conn
        with self._lock:
            c.execute("CREATE OR REPLACE TEMP TABLE transactions AS SELECT * FROM srv.tx_template")
            c.execute("CREATE OR REPLACE TEMP TABLE identity AS SELECT * FROM srv.id_template")
            c.register("_tx", pd.DataFrame([tx_row]))
            c.execute("INSERT INTO transactions BY NAME SELECT * FROM _tx")
            if id_row:
                c.register("_id", pd.DataFrame([{"TransactionID": txn["TransactionID"], **id_row}]))
                c.execute("INSERT INTO identity BY NAME SELECT * FROM _id")
            db.run_sql(c, "features/01_base_join.sql")
            register_maps(c, distinct_text_values(c))
            db.run_sql(c, "features/02_normalize_keys.sql")
            cols = ", ".join(WINDOW_COLS)
            c.execute(
                f"CREATE OR REPLACE TEMP TABLE window_input AS "
                f"SELECT {cols} FROM srv.window_store WHERE card1 = ? AND TransactionID <> ? "
                f"UNION ALL SELECT {cols} FROM keyed",
                [txn["card1"], txn["TransactionID"]],
            )
            db.run_sql(c, "features/03_velocity.sql")
            db.run_sql(c, "features/04_card_history.sql")
            db.run_sql(c, "features/06_assemble.sql")
            row = c.execute("SELECT * FROM features").df()
            for t in ("_tx", "_id"):
                with contextlib.suppress(Exception):
                    c.unregister(t)
        return row

    def score(self, txn: dict, cost_override: dict | None = None) -> dict:
        c: Costs = costs()
        if cost_override:
            c = c.model_copy(update={k: v for k, v in cost_override.items() if v is not None})
        row = self.build_features(txn)
        X = fs.prepare_frame(row, self.features, self.categories)
        p_raw = float(self.booster.predict(X)[0])
        p_cal = float(self.calibrator.predict([p_raw])[0])
        amt = float(row.TransactionAmt.iloc[0])
        e_a, e_d, e_r = (float(v[0]) for v in policy.expected_costs([p_cal], [amt], c))
        threshold = self.policy.get("review_benefit_threshold", 0.0)
        decision = policy.LABELS[
            int(policy.three_tier(np.array([p_cal]), np.array([amt]), c, threshold)[0])
        ]
        contrib = explain.contributions(self.booster, X).iloc[0]
        reasons = explain.reason_codes(contrib, row.iloc[0])
        return {
            "transaction_id": int(row.TransactionID.iloc[0]),
            "p_raw": p_raw,
            "p_calibrated": p_cal,
            "decision": decision,
            "expected_cost": {"approve": e_a, "decline": e_d, "review": e_r},
            "break_even_probability": float(policy.break_even_probability([amt], c)[0]),
            "reason_codes": reasons,
            "model_version": self.version,
        }


def example_payload(db_path: Path, transaction_id: int) -> dict:
    """Raw CSV fields of one stored transaction (+ identity), as an API request body."""
    conn = db.connect(db_path, read_only=True)
    tx = conn.execute("SELECT * FROM transactions WHERE TransactionID = ?", [transaction_id]).df()
    idn = conn.execute("SELECT * FROM identity WHERE TransactionID = ?", [transaction_id]).df()
    conn.close()
    row = tx.drop(columns=["isFraud"]).iloc[0].to_dict()
    if len(idn):
        row.update(idn.drop(columns=["TransactionID"]).iloc[0].to_dict())
    return {k: _plain(v) for k, v in row.items() if not _missing(v)}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA


def _plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v
