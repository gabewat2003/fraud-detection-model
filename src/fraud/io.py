"""Raw-data checks, CSV ingest into DuckDB, and loading the feature table for modeling."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud import db
from fraud.config import Paths

TX_FILE = "train_transaction.csv"
ID_FILE = "train_identity.csv"

# Columns the pipeline relies on by name. Everything else (V*, C*, D*, M*, id_*) is used
# generically, so the synthetic sample can carry fewer of them than the real data.
REQUIRED_TX = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "D1",
]  # fmt: skip
REQUIRED_ID = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31"]
EXPECTED_ROWS = {TX_FILE: 590_540, ID_FILE: 144_233}

# Monetary amount and time stay float64; everything else is downcast to float32 on load.
KEEP_FLOAT64 = {"TransactionAmt", "TransactionDT"}


class DataError(RuntimeError):
    pass


def _csv_header(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.readline().strip().split(",")


def data_check(paths: Paths) -> dict:
    """Confirm both CSVs exist and carry the columns the pipeline needs."""
    report = {}
    for fname, required in ((TX_FILE, REQUIRED_TX), (ID_FILE, REQUIRED_ID)):
        f = paths.raw / fname
        if not f.exists():
            raise DataError(
                f"missing {f}. Download it from the Kaggle IEEE-CIS competition page "
                "into data/raw/ (see data/README.md)."
            )
        cols = _csv_header(f)
        missing = [c for c in required if c not in cols]
        if missing:
            raise DataError(f"{fname} is missing columns: {missing}")
        with open(f, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        report[fname] = {"rows": n, "columns": len(cols)}
        if paths.profile == "full" and n != EXPECTED_ROWS[fname]:
            report[fname]["warning"] = f"expected {EXPECTED_ROWS[fname]:,} rows"
    return report


def ingest(paths: Paths) -> dict:
    """Load both CSVs into DuckDB tables `transactions` and `identity`, then build `base`."""
    data_check(paths)
    paths.ensure()
    conn = db.connect(paths.db)
    for table, fname in (("transactions", TX_FILE), ("identity", ID_FILE)):
        conn.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT * FROM read_csv(?, header=true, sample_size=-1)",
            [str(paths.raw / fname)],
        )
    db.run_sql(conn, "features/01_base_join.sql")
    summary = (
        conn.execute(
            """
        SELECT count(*) AS rows,
               avg(isFraud) AS fraud_rate,
               avg(has_identity) AS identity_share,
               avg(isFraud) FILTER (WHERE has_identity = 1) AS fraud_rate_with_identity,
               avg(isFraud) FILTER (WHERE has_identity = 0) AS fraud_rate_without_identity
        FROM base
        """
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    summary["rows_lost_by_inner_join"] = int(summary["rows"] * (1 - summary["identity_share"]))
    conn.close()
    counts = {"rows", "rows_lost_by_inner_join"}
    return {k: int(v) if k in counts else round(float(v), 5) for k, v in summary.items()}


def write_parquet_downcast(conn, query: str, out) -> None:
    """Export a DuckDB query to parquet with floats stored as float32 (halves memory)."""
    table = conn.execute(query).to_arrow_table()
    fields = []
    for field in table.schema:
        if pa.types.is_float64(field.type) and field.name not in KEEP_FLOAT64:
            fields.append(pa.field(field.name, pa.float32()))
        else:
            fields.append(field)
    pq.write_table(table.cast(pa.schema(fields)), out)


def load_features(paths: Paths, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(paths.features, columns=columns)


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
