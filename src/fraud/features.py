"""Feature pipeline: runs sql/features/*.sql in DuckDB and exports the modeling table.

Order:
  01 base join (done at ingest)  ->  text-normalization lookup tables (Python)
  02 keys + normalized fields    ->  03 velocity  ->  04 spend history
  05 train-window frequency tables  ->  06 assemble  ->  features.parquet

It also writes the serving store (models/serving.duckdb): the narrow history table plus
the frequency lookups, so the scoring API can recompute identical features online.
"""

from __future__ import annotations

import time

import pandas as pd

from fraud import db, split, text_norm
from fraud.config import Paths
from fraud.featuresets import manifest
from fraud.io import save_json, write_parquet_downcast

# Columns the window features need. The API keeps only these for its history store.
WINDOW_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "card_key", "uid", "card1",
    "addr1", "p_email_provider", "device_family",
]  # fmt: skip

TEXT_SOURCES = {
    "device": ["DeviceInfo"],
    "os": ["id_30"],
    "browser": ["id_31"],
    "email": ["P_emaildomain", "R_emaildomain"],
}
FREQ_TABLES = [
    "freq_card1",
    "freq_card2",
    "freq_addr1",
    "freq_p_email",
    "freq_device",
    "browser_latest",
]


def register_maps(conn, values: dict[str, list]) -> None:
    """Create map_<kind> lookup tables from the distinct raw strings of each kind."""
    for kind, raw in values.items():
        df = text_norm.lookup_table(kind, raw)
        conn.register("_map_df", df)
        conn.execute(f"CREATE OR REPLACE TABLE map_{kind} AS SELECT * FROM _map_df")
        conn.unregister("_map_df")


def distinct_text_values(conn) -> dict[str, list]:
    out = {}
    for kind, cols in TEXT_SOURCES.items():
        vals = set()
        for c in cols:
            vals.update(v for (v,) in conn.execute(f"SELECT DISTINCT {c} FROM base").fetchall())
        out[kind] = sorted(v for v in vals if v is not None)
    return out


def run_feature_sql(conn, train_end: float) -> None:
    """Steps 02-06 against whatever `base` and map tables the connection holds."""
    db.run_sql(conn, "features/02_normalize_keys.sql")
    conn.execute(
        f"CREATE OR REPLACE VIEW window_input AS SELECT {', '.join(WINDOW_COLS)} FROM keyed"
    )
    db.run_sql(conn, "features/03_velocity.sql")
    db.run_sql(conn, "features/04_card_history.sql")
    db.run_sql(conn, "features/05_freq_tables.sql", train_end=train_end)
    db.run_sql(conn, "features/06_assemble.sql")


def text_norm_coverage(conn) -> dict:
    base = conn.execute(
        "SELECT DeviceInfo, id_30, id_31, P_emaildomain, R_emaildomain FROM base"
    ).df()
    email = pd.concat([base.P_emaildomain, base.R_emaildomain], ignore_index=True)
    return {
        "DeviceInfo -> device_family": text_norm.coverage_report(
            base.DeviceInfo, "device", "device_family"
        ),
        "DeviceInfo -> device_brand": text_norm.coverage_report(
            base.DeviceInfo, "device", "device_brand"
        ),
        "id_30 -> os_family": text_norm.coverage_report(base.id_30, "os", "os_family"),
        "id_31 -> browser_family": text_norm.coverage_report(
            base.id_31, "browser", "browser_family"
        ),
        "emaildomain -> email_provider": text_norm.coverage_report(
            email, "email", "email_provider"
        ),
    }


def build(paths: Paths) -> dict:
    t0 = time.time()
    paths.ensure()
    conn = db.connect(paths.db)

    cutoffs = split.compute_cutoffs(conn)
    save_json(cutoffs, paths.processed / "split_cutoffs.json")

    register_maps(conn, distinct_text_values(conn))
    run_feature_sql(conn, cutoffs["train_end"])

    query = (
        f"SELECT *, {split.split_case_sql(cutoffs)} AS split "
        "FROM features ORDER BY TransactionDT, TransactionID"
    )
    write_parquet_downcast(conn, query, paths.features)

    cols = [r[0] for r in conn.execute("DESCRIBE features").fetchall()]
    save_json(manifest(cols), paths.processed / "feature_manifest.json")
    save_json(text_norm_coverage(conn), paths.reports / "text_norm_coverage.json")

    _write_serving_store(conn, paths)
    n = conn.execute("SELECT count(*) FROM features").fetchone()[0]
    conn.close()
    return {"rows": n, "columns": len(cols), "seconds": round(time.time() - t0, 1), **cutoffs}


def _write_serving_store(conn, paths: Paths) -> None:
    """Narrow history + lookups + empty raw-table templates for the scoring API."""
    paths.serving_db.unlink(missing_ok=True)
    conn.execute(f"ATTACH '{paths.serving_db.as_posix()}' AS srv")
    conn.execute("CREATE TABLE srv.window_store AS SELECT * FROM window_input")
    for t in FREQ_TABLES:
        conn.execute(f"CREATE TABLE srv.{t} AS SELECT * FROM {t}")
    conn.execute("CREATE TABLE srv.tx_template AS SELECT * FROM transactions LIMIT 0")
    conn.execute("CREATE TABLE srv.id_template AS SELECT * FROM identity LIMIT 0")
    conn.execute("DETACH srv")
