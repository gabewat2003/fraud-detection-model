"""End-to-end smoke test on synthetic data: every stage runs and produces its outputs."""

from fraud import backtest, drift, explain, report, sql_report, tune
from fraud.io import load_json


def test_model_beats_baselines(trained_paths):
    res = load_json(trained_paths.reports / "train_results.json")
    ap = {k: v["test"]["pr_auc"] for k, v in res["experiments"].items()}
    base = res["experiments"]["lgbm"]["test"]["base_rate"]
    assert ap["lgbm"] > ap["lr"] > base
    assert ap["rules"] > base
    assert res["champion"].startswith("lgbm")


def test_remaining_stages_run(trained_paths):
    p = trained_paths
    report.evaluate(p, n_boot=20)
    explain.run(p)
    backtest.run(p)
    drift.run(p)
    sql = sql_report.run(p)
    assert all(isinstance(v, int) for v in sql.values()), sql
    for f in ("pr_curves", "calibration", "cost_curve", "sensitivity_three_tier", "review_queue",
              "shap_top_features", "backtest", "drift_psi"):  # fmt: skip
        assert (p.figures / f"{f}.png").exists(), f
    for f in ("metrics.json", "policy_results.json", "explain.json", "backtest.json", "drift.json"):
        assert (p.reports / f).exists(), f


def test_tune_writes_params(trained_paths):
    out = tune.run(trained_paths, n_trials=2, timeout_minutes=2)
    assert out["n_trials"] == 2
    assert (trained_paths.models / "model_tuned.yaml").exists()


def test_writeup_fills_marked_blocks(trained_paths, tmp_path):
    from fraud import writeup

    report.evaluate(trained_paths, n_boot=20)
    doc = tmp_path / "doc.md"
    doc.write_text("intro\n<!-- BEGIN:headline -->\nold\n<!-- END:headline -->\nkeep me\n"
                   "<!-- BEGIN:policy -->\n<!-- END:policy -->\n", encoding="utf-8")  # fmt: skip
    filled = writeup.apply(doc, writeup.blocks(trained_paths))
    text = doc.read_text(encoding="utf-8")
    assert filled == ["headline", "policy"]
    assert "STALE_SENTINEL" not in text and "keep me" in text and "intro" in text
    assert "lower fraud + friction cost" in text
    assert "Approve everything" in text
