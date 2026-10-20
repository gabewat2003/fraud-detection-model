"""Fill the generated blocks in README.md, reports/memo.md and docs/model_card.md.

Documents contain marker pairs:

    <!-- BEGIN:policy -->
    ...replaced on every run...
    <!-- END:policy -->

Everything between markers is regenerated from the JSON/Markdown under reports/, so every
number in the write-up traces back to a pipeline output. Prose outside the markers is
never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fraud.config import ROOT, Paths
from fraud.io import load_json
from fraud.report import MODEL_LABELS, md_table

DOCS = [ROOT / "README.md", ROOT / "reports" / "memo.md", ROOT / "docs" / "model_card.md"]
MISSING = "_Not generated yet: run `make all` on the Kaggle data, then `make writeup`._"


def _read(p: Path):
    if not p.exists():
        return None
    return load_json(p) if p.suffix == ".json" else p.read_text(encoding="utf-8")


def _pct(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}%}"


def _usd(v: float) -> str:
    return f"${v:,.0f}"


def blocks(paths: Paths) -> dict[str, str]:
    r = paths.reports
    train = _read(r / "train_results.json")
    pol = _read(r / "policy_results.json")
    met = _read(r / "metrics.json")
    out: dict[str, str] = {}

    if train and pol and met:
        champ = train["champion"]
        t = train["experiments"][champ]["test"]
        by = {p["policy"]: p for p in pol["policies"]}
        three = by["Model: three-tier with review cap"]
        rules = by["Rules engine (2+ rules decline, 1 rule review)"]
        top1 = by["Model: decline top 1%"]
        lo, hi = three["savings_ci"]
        ci = met["pr_auc_ci"].get(champ, [float("nan")] * 2)
        out["headline"] = "\n".join([
            f"- **{_pct(three['savings_vs_approve_all'])} lower fraud + friction cost** than approving everything on the "
            f"held-out test window (95% CI {_pct(lo)} to {_pct(hi)}, day-block bootstrap), and "
            f"{_pct(1 - three['total_cost'] / rules['total_cost'])} lower than a hand-written rules engine.",
            f"- Stops **{_pct(three['fraud_dollar_recall'])} of fraud dollars** while declining "
            f"**{_pct(three['good_declined_rate'], 2)} of legitimate customers** and sending "
            f"{_pct(three['review_rate'], 2)} of traffic to manual review (cap {_pct(pol['costs']['review_cap'])}).",
            f"- {_pct(three['savings_vs_top_1pct'])} cheaper than the common 'decline the top 1% of scores' rule: "
            f"declines follow expected cost, and the most uncertain high-value cases go to manual review.",
            f"- Model: {MODEL_LABELS.get(champ, champ)}, test PR-AUC **{t['pr_auc']:.3f}** (95% CI {ci[0]:.3f}-{ci[1]:.3f}) "
            f"against a {t['base_rate']:.3f} base rate.",
        ])  # fmt: skip
        out["policy"] = _read(r / "policy.md") or MISSING

        rows = []
        for k in ("lgbm", "abl_raw_text", "abl_no_behavior", "abl_interpretable"):
            if k in train["experiments"]:
                cost = next(
                    (m["total_cost"] for m in pol["by_model"] if m["model"] == MODEL_LABELS[k]),
                    None,
                )
                rows.append({"variant": MODEL_LABELS[k], "features": train["experiments"][k].get("n_features"),
                             "test PR-AUC": train["experiments"][k]["test"]["pr_auc"], "three-tier cost": cost})  # fmt: skip
        out["ablation"] = md_table(
            pd.DataFrame(rows), {"test PR-AUC": ".4f", "three-tier cost": "$,.0f"}
        )
        out["memo_numbers"] = "\n".join([
            "| | Approve all | Rules engine | Recommended policy |",
            "|---|---|---|---|",
            f"| Total cost (test window) | {_usd(by['Approve everything']['total_cost'])} | {_usd(rules['total_cost'])} | {_usd(three['total_cost'])} |",
            f"| Fraud $ stopped | $0 | {_usd(rules['fraud_usd_caught'])} | {_usd(three['fraud_usd_caught'])} |",
            f"| Good customers declined | 0 | {rules['good_declined']:,} | {three['good_declined']:,} ({_pct(three['good_declined_rate'], 2)}) |",
            f"| Manual reviews | 0 | {rules['reviews']:,} | {three['reviews']:,} |",
            f"| Top-1%-decline rule, for reference | | | {_usd(top1['total_cost'])} |",
        ])  # fmt: skip

    out["metrics"] = _read(r / "metrics.md") or MISSING
    out["splits"] = _read(r / "splits.md") or MISSING

    cov = _read(r / "text_norm_coverage.json")
    if cov:
        df = pd.DataFrame([{"mapping": k, "raw levels": v["raw_levels"], "normalized levels": v["normalized_levels"],
                            "rows mapped": v["row_coverage"]} for k, v in cov.items()])  # fmt: skip
        out["textnorm"] = md_table(df, {"rows mapped": ".1%"})

    for name, f in (("sql_identity", "02_identity_missingness"), ("sql_cohorts", "01_fraud_rate_by_cohort"),
                    ("sql_card_testing", "05_card_testing"), ("sql_segments", "04_top_risk_segments")):  # fmt: skip
        md = _read(r / "sql" / f"{f}.md")
        out[name] = md.split("```\n\n", 1)[-1].strip() if md else MISSING  # drop the echoed query

    ex = _read(r / "explain.json")
    if ex:
        shares = ", ".join(
            f"{g} {_pct(s, 0)}" for g, s in sorted(ex["group_share"].items(), key=lambda t: -t[1])
        )
        top = pd.DataFrame(ex["top_features"])
        out["explain"] = (f"Share of total |SHAP| by feature group: {shares}. Interpretable features carry "
                          f"{_pct(ex['interpretable_share'], 0)} of attribution.\n\n"
                          + md_table(top, {"mean_abs_shap": ".4f"}))  # fmt: skip

    bt = _read(r / "backtest.json")
    if bt:
        out["backtest"] = md_table(pd.DataFrame(bt["folds"])[["test_window", "train_rows", "test_rows", "fraud_rate",
                                                              "pr_auc", "recall_at_1pct", "dollar_recall_at_1pct"]],
                                   {"fraud_rate": ".4f", "pr_auc": ".3f", "recall_at_1pct": ".3f",
                                    "dollar_recall_at_1pct": ".3f"})  # fmt: skip
    dr = _read(r / "drift.json")
    if dr:
        top = pd.DataFrame(dr["last_window_top"])[["feature", "psi"]]
        out["drift"] = (f"Last scored window vs training: features with PSI > 0.2: "
                        f"{', '.join(dr['last_window_large_shift']) or 'none'}.\n\n" + md_table(top, {"psi": ".3f"}))  # fmt: skip
    return out


def apply(doc: Path, generated: dict[str, str]) -> list[str]:
    text = doc.read_text(encoding="utf-8")
    filled = []

    def sub(m: re.Match) -> str:
        name = m.group(1)
        filled.append(name)
        body = generated.get(name, MISSING)
        return f"<!-- BEGIN:{name} -->\n{body.strip()}\n<!-- END:{name} -->"

    new = re.sub(r"<!-- BEGIN:(\w+) -->.*?<!-- END:\1 -->", sub, text, flags=re.S)
    doc.write_text(new, encoding="utf-8")
    return filled


def run(paths: Paths, docs: list[Path] | None = None) -> dict:
    generated = blocks(paths)
    return {str(d.relative_to(ROOT)): apply(d, generated) for d in (docs or DOCS) if d.exists()}
