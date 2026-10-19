"""Command-line entry point: `python -m fraud.cli <stage> [--profile full|sample]`."""

from __future__ import annotations

import json
import os
import sys
import time

import typer

from fraud.config import get_paths

app = typer.Typer(add_completion=False, help="IEEE-CIS fraud detection pipeline.")
Profile = typer.Option(
    "full", "--profile", "-p", help="full = Kaggle data, sample = synthetic data"
)


def _echo(title: str, obj) -> None:
    typer.echo(f"\n== {title}")
    typer.echo(json.dumps(obj, indent=2, default=str)[:4000])


def _stage(name: str, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    _echo(f"{name} ({time.time() - t0:.1f}s)", out)
    return out


@app.command()
def synth(seed: int = 0, n_cards: int = 900):
    """Generate the synthetic IEEE-shaped sample into data/sample/."""
    from fraud import synth as s

    _stage("synth", s.write, get_paths("sample").raw, seed=seed, n_cards=n_cards)


@app.command("data-check")
def data_check(profile: str = Profile):
    """Check the raw CSVs exist and have the expected columns."""
    from fraud import io

    _stage("data-check", io.data_check, get_paths(profile))


@app.command()
def ingest(profile: str = Profile):
    """Load CSVs into DuckDB and build the LEFT-joined base table."""
    from fraud import io

    _stage("ingest", io.ingest, get_paths(profile))


@app.command()
def features(profile: str = Profile):
    """Run the SQL feature pipeline and export features.parquet."""
    from fraud import features as f

    _stage("features", f.build, get_paths(profile))


@app.command()
def tune(
    profile: str = Profile,
    n_trials: int = typer.Option(None),
    timeout_minutes: float = typer.Option(None),
):
    """Optuna search over LightGBM parameters (validation PR-AUC)."""
    from fraud import tune as t

    _stage("tune", t.run, get_paths(profile), n_trials=n_trials, timeout_minutes=timeout_minutes)


@app.command()
def train(profile: str = Profile, no_ablations: bool = False):
    """Train rules / LR / LightGBM variants, calibrate, and score every split."""
    from fraud import train as t

    out = _stage("train", t.run, get_paths(profile), include_ablations=not no_ablations)
    for name, info in out["experiments"].items():
        typer.echo(
            f"  {name:20s} valid PR-AUC {info['valid']['pr_auc']:.4f}  test PR-AUC {info['test']['pr_auc']:.4f}"
        )
    typer.echo(f"  champion: {out['champion']}")


@app.command()
def evaluate(profile: str = Profile, n_boot: int = 300):
    """Metrics table, bootstrap CIs, PR and calibration charts."""
    from fraud import report

    _stage("evaluate", report.evaluate, get_paths(profile), n_boot=n_boot)


@app.command()
def policy(profile: str = Profile):
    """Approve / review / decline policies, sensitivity grid, review queue."""
    from fraud import report

    out = _stage("policy", report.policy_report, get_paths(profile))
    typer.echo((get_paths(profile).reports / "policy.md").read_text(encoding="utf-8")[:3000])
    return out


@app.command()
def explain(profile: str = Profile):
    """TreeSHAP attributions and example reason codes."""
    from fraud import explain as e

    _stage("explain", e.run, get_paths(profile))


@app.command()
def backtest(profile: str = Profile):
    """Rolling-origin backtest over 30-day windows."""
    from fraud import backtest as b

    _stage("backtest", b.run, get_paths(profile))


@app.command()
def drift(profile: str = Profile):
    """PSI drift of top features and the score vs the training window."""
    from fraud import drift as d

    _stage("drift", d.run, get_paths(profile))


@app.command()
def sql(profile: str = Profile):
    """Run the analyst queries in sql/analysis/."""
    from fraud import sql_report

    _stage("sql", sql_report.run, get_paths(profile))


@app.command()
def writeup(profile: str = Profile, force: bool = False):
    """Refresh the generated blocks in README.md, reports/memo.md, docs/model_card.md."""
    from fraud import writeup as w

    if profile != "full" and not force:
        raise typer.BadParameter("writeup only uses real-data results (pass --force to override)")
    _stage("writeup", w.run, get_paths(profile))


@app.command("all")
def run_all(profile: str = Profile, skip_tune: bool = False, n_trials: int = typer.Option(None)):
    """Every stage in order."""
    paths = get_paths(profile)
    from fraud import backtest as b
    from fraud import drift as d
    from fraud import explain as e
    from fraud import features as f
    from fraud import io, report, sql_report
    from fraud import train as t
    from fraud import tune as tu

    _stage("ingest", io.ingest, paths)
    _stage("features", f.build, paths)
    if not skip_tune:
        _stage("tune", tu.run, paths, n_trials=n_trials)
    _stage("train", t.run, paths)
    _stage("evaluate", report.evaluate, paths)
    _stage("policy", report.policy_report, paths)
    _stage("explain", e.run, paths)
    _stage("backtest", b.run, paths)
    _stage("drift", d.run, paths)
    _stage("sql", sql_report.run, paths)
    if profile == "full":
        from fraud import writeup as w

        _stage("writeup", w.run, paths)


def main():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
