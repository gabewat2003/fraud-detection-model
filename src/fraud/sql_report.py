"""Run the analyst queries in sql/analysis/ and write each result as Markdown + CSV."""

from __future__ import annotations

from fraud import db
from fraud.config import SQL_DIR, Paths
from fraud.report import md_table

MAX_MD_ROWS = 20


def run(paths: Paths) -> dict:
    out_dir = paths.reports / "sql"
    out_dir.mkdir(parents=True, exist_ok=True)
    min_support = 500 if paths.profile == "full" else 30
    conn = db.connect(paths.db, read_only=True)
    if paths.scored.exists():
        conn.execute(
            f"CREATE TEMP VIEW scored AS SELECT * FROM read_parquet('{paths.scored.as_posix()}')"
        )
    decisions = paths.processed / "decisions.parquet"
    if decisions.exists():
        conn.execute(
            f"CREATE TEMP VIEW decisions AS SELECT * FROM read_parquet('{decisions.as_posix()}')"
        )

    summary = {}
    for f in sorted((SQL_DIR / "analysis").glob("*.sql")):
        sql = db.read_sql(f"analysis/{f.name}", min_support=min_support)
        try:
            df = conn.execute(sql).df()
        except Exception as e:  # e.g. policy query before `fraud policy` has run
            summary[f.stem] = f"skipped: {e.__class__.__name__}"
            continue
        df.to_csv(out_dir / f"{f.stem}.csv", index=False)
        question = " ".join(
            line.removeprefix("--").strip() for line in sql.splitlines() if line.startswith("--")
        )
        shown = df.head(MAX_MD_ROWS)
        note = (
            f"\n\n_{len(df)} rows; first {MAX_MD_ROWS} shown, full result in `{f.stem}.csv`._"
            if len(df) > MAX_MD_ROWS
            else ""
        )
        body = (
            f"### {f.stem}\n\n{question}\n\n```sql\n{sql.strip()}\n```\n\n{md_table(shown)}{note}\n"
        )
        (out_dir / f"{f.stem}.md").write_text(body, encoding="utf-8")
        summary[f.stem] = len(df)
    conn.close()
    return summary
