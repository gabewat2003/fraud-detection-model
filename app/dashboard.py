"""Streamlit policy simulator.

    streamlit run app/dashboard.py                  (real-data outputs)
    FRAUD_PROFILE=sample streamlit run app/dashboard.py

Move the cost assumptions and review capacity; thresholds are refit on the calibration
window and the policies are re-scored on the test window live.
"""

from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from fraud import charts, explain, policy
from fraud import featuresets as fs
from fraud.config import costs, get_paths
from fraud.io import load_json
from fraud.report import MODEL_LABELS
from fraud.train import load_champion

st.set_page_config(page_title="Fraud policy simulator", layout="wide")
paths = get_paths(os.environ.get("FRAUD_PROFILE", "full"))


@st.cache_data
def load():
    scored = pd.read_parquet(paths.scored)
    res = load_json(paths.reports / "train_results.json")
    return scored, res


if not paths.scored.exists():
    st.error(f"No scored data at {paths.scored}. Run `make all` (or `make sample`) first.")
    st.stop()

scored, res = load()
champion = res["champion"]
calib = scored[scored.split == "calib"].reset_index(drop=True)
test = scored[scored.split == "test"].reset_index(drop=True)
y, amt, day = test.isFraud.to_numpy(), test.TransactionAmt.to_numpy(), test.tx_day.to_numpy()
base = costs()

st.title("Approve / review / decline policy simulator")
st.caption(f"Profile: **{paths.profile}** · champion model: **{MODEL_LABELS.get(champion, champion)}** · "
           f"test window: {len(test):,} transactions, {y.mean():.2%} fraud. Costs are assumptions — move them.")  # fmt: skip

with st.sidebar:
    st.header("Cost assumptions")
    fee = st.slider("Chargeback fee ($)", 0.0, 50.0, base.chargeback_fee, 1.0)
    friction = st.slider("Friction per false decline ($)", 0.0, 30.0, base.friction_cost, 0.5)
    review_cost = st.slider("Cost per manual review ($)", 0.5, 20.0, base.review_cost, 0.5)
    cap = (
        st.slider("Review capacity (% of daily volume)", 0.0, 5.0, base.review_cap * 100, 0.25)
        / 100
    )
    models = [m for m in res["experiments"] if f"pc_{m}" in scored]
    model = st.selectbox("Model driving the policy", models, index=models.index(champion),
                         format_func=lambda m: MODEL_LABELS.get(m, m))  # fmt: skip

c = base.model_copy(update={"chargeback_fee": fee, "friction_cost": friction, "review_cost": review_cost,
                            "review_cap": cap})  # fmt: skip
pc, pt = calib[f"pc_{model}"].to_numpy(), test[f"pc_{model}"].to_numpy()
raw_c, raw_t = (
    calib[f"p_{model}"].to_numpy(),
    test[f"p_{model}"].to_numpy(),
)  # flat thresholds rank raw scores
rt = policy.fit_review_threshold(pc, calib.TransactionAmt.to_numpy(), c)
flat = policy.fit_cost_optimal_threshold(
    raw_c, calib.isFraud.to_numpy(), calib.TransactionAmt.to_numpy(), c
)
decisions = {
    "Approve everything": policy.approve_all(len(test)),
    "Best flat threshold": policy.decline_above(raw_t, flat),
    "Amount-aware decline": policy.amount_aware(pt, amt, c),
    "Three-tier with review cap": policy.three_tier(pt, amt, c, rt),
}
rows = []
for name, d in decisions.items():
    s = policy.summarize(d, y, amt, c)
    rows.append({"policy": name, **s})
table = pd.DataFrame(rows)
approve_cost = table.total_cost.iloc[0]
table["savings"] = 1 - table.total_cost / approve_cost
best = table.iloc[table.total_cost.idxmin()]

tab_policy, tab_model, tab_drift, tab_explain = st.tabs(
    ["Policy", "Model quality", "Drift & backtest", "Explain a transaction"]
)

with tab_policy:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cheapest policy", best.policy)
    k2.metric(
        "Total cost",
        f"${best.total_cost:,.0f}",
        f"{-best.savings:.1%} vs approve-all",
        delta_color="inverse",
    )
    k3.metric("Fraud $ stopped", f"{best.fraud_dollar_recall:.1%}")
    k4.metric("Good customers declined", f"{best.good_declined:,} ({best.good_declined_rate:.2%})")
    show = table[["policy", "total_cost", "savings", "fraud_usd_caught", "fraud_usd_missed", "good_declined",
                  "good_declined_rate", "reviews", "review_rate"]]  # fmt: skip
    st.dataframe(show.style.format({"total_cost": "${:,.0f}", "savings": "{:.1%}", "fraud_usd_caught": "${:,.0f}",
                                    "fraud_usd_missed": "${:,.0f}", "good_declined_rate": "{:.2%}",
                                    "review_rate": "{:.2%}"}), hide_index=True, use_container_width=True)  # fmt: skip
    left, right = st.columns(2)
    with left:
        curve = policy.cost_curve(raw_t, y, amt, c)
        markers = table[table.policy != "Approve everything"][
            ["policy", "decline_rate", "total_cost"]
        ]
        st.altair_chart(
            charts.cost_curve(curve, markers).properties(width="container"),
            use_container_width=True,
        )
    with right:
        q = policy.review_queue(decisions["Three-tier with review cap"], day, cap)
        st.altair_chart(
            charts.review_queue(q).properties(width="container"), use_container_width=True
        )
        st.caption(f"Review bar (fit on calib): expected saving > ${rt:,.2f}. "
                   f"{int((q.over_capacity > 0).sum())} of {len(q)} test days exceeded capacity.")  # fmt: skip
    be = pd.DataFrame({"amount": [5, 25, 100, 500, 2000]})
    be["decline if p >"] = policy.break_even_probability(be.amount, c)
    st.markdown("**Break-even decline probability** — bigger transactions get a stricter cutoff:")
    st.dataframe(
        be.style.format({"amount": "${:,.0f}", "decline if p >": "{:.3f}"}), hide_index=True
    )

with tab_model:
    if (paths.reports / "metrics.json").exists():
        mt = pd.DataFrame(load_json(paths.reports / "metrics.json")["table"])
        st.dataframe(mt.style.format(precision=3), hide_index=True, use_container_width=True)
    for f in ("pr_curves", "calibration", "shap_by_group", "shap_interpretable_features"):
        if (paths.figures / f"{f}.png").exists():
            st.image(str(paths.figures / f"{f}.png"))

with tab_drift:
    for f in ("backtest", "drift_psi"):
        if (paths.figures / f"{f}.png").exists():
            st.image(str(paths.figures / f"{f}.png"))

with tab_explain:
    booster, iso, meta = load_champion(paths)
    pick = st.radio("Show", ["Highest-scored fraud", "Highest-scored legitimate (false alarms)", "Missed fraud"],
                    horizontal=True)  # fmt: skip
    col = f"p_{champion}"
    pool = {"Highest-scored fraud": test[test.isFraud == 1].nlargest(25, col),
            "Highest-scored legitimate (false alarms)": test[test.isFraud == 0].nlargest(25, col),
            "Missed fraud": test[test.isFraud == 1].nsmallest(25, col)}[pick]  # fmt: skip
    tid = st.selectbox("TransactionID", pool.TransactionID.tolist())
    feats = pd.read_parquet(paths.features, columns=list(dict.fromkeys(["TransactionID", *meta["features"]])),
                            filters=[("TransactionID", "==", int(tid))])  # fmt: skip
    X = fs.prepare_frame(feats, meta["features"], meta["categories"])
    contrib = explain.contributions(booster, X).iloc[0]
    p_raw = float(booster.predict(X)[0])
    p_cal = float(iso.predict([p_raw])[0])
    a = (
        float(feats.TransactionAmt.iloc[0])
        if "TransactionAmt" in feats
        else float(pool.set_index("TransactionID").loc[tid, "TransactionAmt"])
    )
    d = policy.LABELS[int(policy.three_tier(np.array([p_cal]), np.array([a]), c, rt)[0])]
    m1, m2, m3 = st.columns(3)
    m1.metric("Calibrated fraud probability", f"{p_cal:.1%}")
    m2.metric("Amount", f"${a:,.2f}")
    m3.metric("Decision (three-tier)", d.upper())
    st.markdown("**Top reasons pushing toward fraud**")
    for r in explain.reason_codes(contrib, feats.iloc[0]):
        st.write(f"- {r['text']}  _(+{r['contribution']:.2f} log-odds)_")
    top = (
        contrib.drop("_bias")
        .sort_values(key=np.abs, ascending=False)
        .head(12)
        .rename("shap")
        .reset_index()
    )
    top.columns = ["feature", "shap"]
    top["direction"] = np.where(top.shap > 0, "toward fraud", "toward legit")
    st.altair_chart(
        alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("shap:Q", title="SHAP (log-odds)"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["toward fraud", "toward legit"],
                                                           range=[charts.SERIES[1], charts.SERIES[0]]), title=None),
            tooltip=["feature", alt.Tooltip("shap:Q", format=".3f")],
        ).properties(height=320),
        use_container_width=True,
    )  # fmt: skip
