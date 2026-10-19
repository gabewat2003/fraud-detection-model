"""Thin DuckDB helpers: open a connection and run the versioned SQL files in sql/."""

from __future__ import annotations

from pathlib import Path
from string import Template

import duckdb

from fraud.config import SQL_DIR


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if path is None:
        return duckdb.connect()
    return duckdb.connect(str(path), read_only=read_only)


def read_sql(name: str, **params) -> str:
    """Load sql/<name>, substituting $placeholders (string.Template syntax)."""
    text = (SQL_DIR / name).read_text(encoding="utf-8")
    return Template(text).safe_substitute(params) if params else text


def run_sql(conn: duckdb.DuckDBPyConnection, name: str, **params) -> None:
    conn.execute(read_sql(name, **params))


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
    return conn.execute(q, [name]).fetchone()[0] > 0
