"""Evaluation and policy reports: tables (JSON + Markdown) and charts under reports/."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud import charts, policy
from fraud.config import Costs, Paths, costs
from fraud.evaluate import bootstrap_by_day, pr_curve_points, reliability_table
from fraud.io import load_json, save_json

MODEL_LABELS = {
    "rules": "Rules engine", "lr": "Logistic regression", "lgbm": "LightGBM",
    "lgbm_spw": "LightGBM (scale_pos_weight)", "lgbm_cost": "LightGBM (cost-weighted)",
    "lgbm_tuned": "LightGBM (Optuna-tuned)", "abl_raw_text": "Ablation: raw text strings",
    "abl_no_behavior": "Ablation: no velocity/history/freq", "abl_interpretable": "Ablation: interpretable features only",
}  # fmt: skip


def md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in zip(*(df[c].tolist() for c in cols), strict=True):  # keeps per-column dtypes
        cells = []
        for c, v in zip(cols, row, strict=True):
            fmt = formats.get(c)
            if fmt and isinstance(v, (int, float, np.number)) and not pd.isna(v):
                if fmt.startswith("$"):  # currency: "$,.0f" -> "$1,234" / "-$1,234"
                    cells.append(("-" if v < 0 else "") + "$" + format(abs(v), fmt[1:]))
                elif fmt.endswith("d"):
                    cells.append(format(int(v), fmt))
                else:
                    cells.append(format(v, fmt))
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load(paths: Paths):
    return load_json(paths.reports / "train_results.json"), pd.read_parquet(paths.scored)


# ------------------------------------------------------------------------ evaluate


def evaluate(paths: Paths, n_boot: int = 300) -> dict:
    res, scored = _load(paths)
    champion = res["champion"]
    test = scored[scored.split == "test"].reset_index(drop=True)
    y = test.isFraud.to_numpy()

    rows = []
    for name, info in res["experiments"].items():
        t, v = info["test"], info["valid"]
        rows.append({
            "model": MODEL_LABELS.get(name, name), "key": name,
            "valid PR-AUC": v["pr_auc"], "test PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"],
            "precision @1%": t["precision_at_1pct"], "recall @1%": t["recall_at_1pct"],
            "$ recall @1%": t["dollar_recall_at_1pct"], "recall @1% FPR": t["recall_at_fpr_0.01"],
            "Brier (calibrated)": t.get("brier"),
            "new-account PR-AUC": info.get("test_new_uid", {}).get("pr_auc"),
        })  # fmt: skip
    table = pd.DataFrame(rows)

    # day-block bootstrap CI for test PR-AUC of the headline models
    ci = {}
    for name in dict.fromkeys(["rules", "lr", champion]):
        col = f"p_{name}"
        ci[name] = bootstrap_by_day(test, lambda d, c=col: average_precision_score(d.isFraud, d[c])
                                    if d.isFraud.any() else np.nan, n_boot=n_boot)  # fmt: skip
    table["test PR-AUC 95% CI"] = table.key.map(
        lambda k: f"{ci[k][0]:.3f}-{ci[k][1]:.3f}" if k in ci else ""
    )

    base = {"model": "Random (base rate)", "key": "base", "test PR-AUC": float(y.mean()), "ROC-AUC": 0.5,
            "precision @1%": float(y.mean()), "recall @1%": 0.01, "$ recall @1%": 0.01}  # fmt: skip
    table = pd.concat([pd.DataFrame([base]), table], ignore_index=True)

    fmt = {c: ".3f" for c in table.columns if c not in ("model", "key", "test PR-AUC 95% CI")}
    md = md_table(table.drop(columns="key"), fmt)
    (paths.reports / "metrics.md").write_text(md + "\n", encoding="utf-8")

    split_md = md_table(pd.DataFrame(res["splits"]), {"fraud_rate": ".4f", "fraud_usd": ",.0f"})
    (paths.reports / "splits.md").write_text(split_md + "\n", encoding="utf-8")

    # charts
    order = [MODEL_LABELS[n] for n in dict.fromkeys([champion, "lr", "rules"])]
    pts = pd.concat([pr_curve_points(y, test[f"p_{n}"]).assign(model=MODEL_LABELS[n])
                     for n in dict.fromkeys([champion, "lr", "rules"])])  # fmt: skip
    charts.save(charts.pr_curves(pts, float(y.mean()), order), paths.figures, "pr_curves")

    rel = []
    for n in dict.fromkeys([champion, "lgbm_spw", "lgbm_cost"]):
        if f"p_{n}" not in test:
            continue
        for kind, col in (("raw", f"p_{n}"), ("calibrated", f"pc_{n}")):
            rel.append(reliability_table(y, test[col]).assign(model=MODEL_LABELS[n], kind=kind))
    charts.save(charts.reliability(pd.concat(rel)), paths.figures, "calibration")

    out = {
        "champion": champion,
        "table": table.drop(columns="key").to_dict(orient="records"),
        "pr_auc_ci": ci,
    }
    save_json(out, paths.reports / "metrics.json")
    return out


# -------------------------------------------------------------------------- policy


def _decisions(
    test: pd.DataFrame, calib: pd.DataFrame, champion: str, c: Costs
) -> tuple[dict, dict]:
    pc, pt = calib[f"pc_{champion}"].to_numpy(), test[f"pc_{champion}"].to_numpy()
    # flat thresholds are ranking rules, so they use raw scores: isotonic output is a step
    # function whose ties would make "the top 1%" (or a fine threshold grid) ill-defined
    raw_c, raw_t = calib[f"p_{champion}"].to_numpy(), test[f"p_{champion}"].to_numpy()
    amt_c, amt_t = calib.TransactionAmt.to_numpy(), test.TransactionAmt.to_numpy()
    th = {
        "top_1pct_threshold": policy.fit_global_threshold(raw_c, 0.01),
        "cost_optimal_flat_threshold": policy.fit_cost_optimal_threshold(
            raw_c, calib.isFraud.to_numpy(), amt_c, c
        ),
        "review_benefit_threshold": policy.fit_review_threshold(pc, amt_c, c),
        "rules_review_amount_bar": policy.fit_rules_amount_bar(
            np.floor(calib.p_rules), amt_c, c.review_cap
        ),
    }
    dec = {
        "Approve everything": policy.approve_all(len(test)),
        "Rules engine (2+ rules decline, 1 rule review)": policy.rules_policy(np.floor(test.p_rules), amt_t, th["rules_review_amount_bar"]),
        "Model: decline top 1%": policy.decline_above(raw_t, th["top_1pct_threshold"]),
        "Model: best flat threshold": policy.decline_above(raw_t, th["cost_optimal_flat_threshold"]),
        "Model: amount-aware decline": policy.amount_aware(pt, amt_t, c),
        "Model: three-tier with review cap": policy.three_tier(pt, amt_t, c, th["review_benefit_threshold"]),
    }  # fmt: skip
    return dec, th


def policy_report(paths: Paths) -> dict:
    res, scored = _load(paths)
    champion = res["champion"]
    c = costs()
    test = scored[scored.split == "test"].reset_index(drop=True)
    calib = scored[scored.split == "calib"].reset_index(drop=True)
    y, amt, day = test.isFraud.to_numpy(), test.TransactionAmt.to_numpy(), test.tx_day.to_numpy()

    dec, th = _decisions(test, calib, champion, c)
    recommended = dec["Model: three-tier with review cap"]
    pd.DataFrame({"TransactionID": test.TransactionID,
                  "decision": pd.Series(recommended).map(policy.LABELS)}).to_parquet(
        paths.processed / "decisions.parquet", index=False)  # fmt: skip
    daily = policy.daily_cost_table(dec, y, amt, day, c)
    base_cost = policy.summarize(dec["Approve everything"], y, amt, c)["total_cost"]
    rows = []
    for name, d in dec.items():
        s = policy.summarize(d, y, amt, c)
        s["policy"] = name
        s["savings_vs_approve_all"] = 1 - s["total_cost"] / base_cost
        if name != "Approve everything":
            s["savings_ci"] = policy.bootstrap_savings(daily, name, "Approve everything")
        rows.append(s)
    table = pd.DataFrame(rows)
    flat = table.set_index("policy").loc["Model: decline top 1%", "total_cost"]
    table["savings_vs_top_1pct"] = 1 - table.total_cost / flat

    # same three-tier policy, but driven by each model's calibrated scores
    by_model = []
    for name in res["experiments"]:
        if f"pc_{name}" not in test:
            continue
        rt = policy.fit_review_threshold(
            calib[f"pc_{name}"].to_numpy(), calib.TransactionAmt.to_numpy(), c
        )
        d = policy.three_tier(test[f"pc_{name}"].to_numpy(), amt, c, rt)
        s = policy.summarize(d, y, amt, c)
        by_model.append({"model": MODEL_LABELS.get(name, name), "total_cost": s["total_cost"],
                         "savings_vs_approve_all": 1 - s["total_cost"] / base_cost,
                         "fraud_dollar_recall": s["fraud_dollar_recall"],
                         "good_declined_rate": s["good_declined_rate"], "review_rate": s["review_rate"]})  # fmt: skip
    by_model = pd.DataFrame(by_model).sort_values("total_cost")

    # cost curve: flat thresholds on test, with the fitted policies as references
    curve = policy.cost_curve(test[f"p_{champion}"].to_numpy(), y, amt, c)
    markers = table[table.policy.isin(["Model: amount-aware decline", "Model: three-tier with review cap",
                                       "Model: decline top 1%", "Model: best flat threshold"])
                    ][["policy", "decline_rate", "total_cost"]]  # fmt: skip
    charts.save(charts.cost_curve(curve, markers), paths.figures, "cost_curve")
    charts.save(charts.policy_bars(table.rename(columns={"savings_vs_approve_all": "savings"})),
                paths.figures, "policy_costs")  # fmt: skip

    grid = sensitivity(test, calib, champion)
    charts.save(
        charts.sensitivity_heatmap(grid, "three-tier"), paths.figures, "sensitivity_three_tier"
    )
    charts.save(
        charts.sensitivity_heatmap(grid, "amount-aware"), paths.figures, "sensitivity_amount_aware"
    )

    q = policy.review_queue(dec["Model: three-tier with review cap"], day, c.review_cap)
    charts.save(charts.review_queue(q), paths.figures, "review_queue")
    queue_stats = {
        "mean_daily_review_rate": float((q.reviews / q.volume).mean()),
        "max_daily_review_rate": float((q.reviews / q.volume).max()),
        "days_over_capacity": int((q.over_capacity > 0).sum()),
        "days": int(len(q)),
        "reviews_over_capacity": int(q.over_capacity.sum()),
    }

    break_even = pd.DataFrame({"amount": [5, 25, 100, 500, 2000]})
    break_even["decline if p >"] = policy.break_even_probability(break_even.amount, c)

    out = {
        "champion": champion, "costs": c.model_dump(exclude={"sensitivity"}), "thresholds": th,
        "policies": table.to_dict(orient="records"), "by_model": by_model.to_dict(orient="records"),
        "review_queue": queue_stats, "break_even": break_even.to_dict(orient="records"),
        "cost_curve_min": {"flag_rate": float(curve.loc[curve.total_cost.idxmin(), "flag_rate"]),
                           "total_cost": float(curve.total_cost.min())},
    }  # fmt: skip
    save_json(out, paths.reports / "policy_results.json")
    save_json({"champion": champion, "costs": out["costs"], **th}, paths.models / "policy.json")
    grid.to_csv(paths.reports / "sensitivity.csv", index=False)
    _write_policy_md(paths, table, by_model, break_even, queue_stats, th, c)
    _write_policy_html(paths, out)
    return out


def sensitivity(test: pd.DataFrame, calib: pd.DataFrame, champion: str) -> pd.DataFrame:
    base = costs()
    s = base.sensitivity
    y, amt = test.isFraud.to_numpy(), test.TransactionAmt.to_numpy()
    pt, pc = test[f"pc_{champion}"].to_numpy(), calib[f"pc_{champion}"].to_numpy()
    rows = []
    for f in s.friction_cost:
        for m in s.lost_margin_rate:
            for cap in s.review_cap:
                c = base.model_copy(
                    update={"friction_cost": f, "lost_margin_rate": m, "review_cap": cap}
                )
                approve = policy.summarize(policy.approve_all(len(y)), y, amt, c)["total_cost"]
                rt = policy.fit_review_threshold(pc, calib.TransactionAmt.to_numpy(), c)
                for name, d in (("three-tier", policy.three_tier(pt, amt, c, rt)),
                                ("amount-aware", policy.amount_aware(pt, amt, c))):  # fmt: skip
                    sm = policy.summarize(d, y, amt, c)
                    rows.append({"friction_cost": f, "lost_margin_rate": m, "review_cap": cap, "policy": name,
                                 "total_cost": sm["total_cost"], "savings": 1 - sm["total_cost"] / approve,
                                 "good_declined_rate": sm["good_declined_rate"],
                                 "fraud_dollar_recall": sm["fraud_dollar_recall"]})  # fmt: skip
    return pd.DataFrame(rows)


def _write_policy_md(paths, table, by_model, break_even, queue, th, c) -> None:
    t = table.copy()
    t["savings 95% CI"] = t.get("savings_ci").map(
        lambda v: f"{v[0]:.1%} to {v[1]:.1%}" if isinstance(v, (list, tuple)) else ""
    )
    cols = {"policy": "Policy", "fraud_usd_caught": "Fraud $ stopped", "fraud_usd_missed": "Fraud $ missed",
            "good_declined": "Good customers declined", "good_declined_rate": "False-decline rate",
            "reviews": "Reviews", "total_cost": "Total cost", "savings_vs_approve_all": "Saving vs approve-all",
            "savings 95% CI": "Saving 95% CI", "savings_vs_top_1pct": "Saving vs top-1%"}  # fmt: skip
    t = t[list(cols)].rename(columns=cols)
    fmt = {"Fraud $ stopped": "$,.0f", "Fraud $ missed": "$,.0f", "Good customers declined": ",d",
           "False-decline rate": ".2%", "Reviews": ",d", "Total cost": "$,.0f",
           "Saving vs approve-all": ".1%", "Saving vs top-1%": ".1%"}  # fmt: skip
    t["Fraud $ stopped"] = t["Fraud $ stopped"].astype(float)
    parts = [
        f"Costs: chargeback fee ${c.chargeback_fee:g}, friction ${c.friction_cost:g} + {c.lost_margin_rate:.0%} of the amount per false decline, review ${c.review_cost:g}, "
        f"review capacity {c.review_cap:.1%} of daily volume.\n",
        md_table(t, fmt),
        "\n\n**Three-tier policy driven by each model**\n",
        md_table(by_model.rename(columns={"model": "Model", "total_cost": "Total cost",
                                          "savings_vs_approve_all": "Saving vs approve-all",
                                          "fraud_dollar_recall": "Fraud $ recall",
                                          "good_declined_rate": "False-decline rate", "review_rate": "Review rate"}),
                 {"Total cost": "$,.0f", "Saving vs approve-all": ".1%", "Fraud $ recall": ".1%",
                  "False-decline rate": ".2%", "Review rate": ".2%"}),
        "\n\n**Break-even decline probability by amount**\n",
        md_table(break_even, {"decline if p >": ".3f", "amount": "$,.0f"}),
        f"\n\n**Review queue (three-tier, test window):** mean {queue['mean_daily_review_rate']:.2%} of daily volume, "
        f"max {queue['max_daily_review_rate']:.2%}; {queue['days_over_capacity']} of {queue['days']} days above capacity "
        f"({queue['reviews_over_capacity']:,} overflow cases).\n",
        f"\nThresholds fit on calib: {json.dumps({k: round(v, 5) for k, v in th.items()})}\n",
    ]  # fmt: skip
    (paths.reports / "policy.md").write_text("".join(parts), encoding="utf-8")


def _write_policy_html(paths: Paths, out: dict) -> None:
    md = (paths.reports / "policy.md").read_text(encoding="utf-8")
    figs = ["policy_costs", "cost_curve", "sensitivity_three_tier", "review_queue"]
    body = "".join(
        f'<img src="figures/{f}.png" alt="{f}" style="max-width:100%;margin:16px 0">' for f in figs
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Policy report</title>
<style>body{{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;
color:#0b0b0b;background:#fcfcfb}} pre{{white-space:pre-wrap;font-size:13px;background:#f3f2ef;padding:12px;
border-radius:6px}}</style></head><body><h1>Approve / review / decline policy</h1>
<p>Champion model: <b>{out["champion"]}</b>. All thresholds fit on the calibration window, results on test.</p>
<pre>{md}</pre>{body}</body></html>"""
    (paths.reports / "policy.html").write_text(html, encoding="utf-8")
