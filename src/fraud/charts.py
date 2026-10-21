"""Altair charts for the report. Each function returns a chart; `save` writes HTML + PNG.

Palette: categorical slots in fixed order (never cycled), one-hue blue ramp for
magnitudes, recessive grey grid/axes, text in ink colors rather than series colors.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
WIDTH, HEIGHT = 520, 300


@alt.theme.register("fraud", enable=True)
def _theme():
    return {
        "config": {
            "background": SURFACE,
            "view": {"stroke": None},
            "font": "Inter, Segoe UI, Helvetica, Arial, sans-serif",
            "title": {"color": INK, "fontSize": 15, "fontWeight": 600, "anchor": "start",
                      "subtitleColor": INK_2, "subtitleFontSize": 12, "offset": 12},
            "axis": {"labelColor": INK_2, "titleColor": INK_2, "gridColor": GRID, "domainColor": GRID,
                     "tickColor": GRID, "labelFontSize": 11, "titleFontSize": 12, "titleFontWeight": 500},
            "legend": {"labelColor": INK_2, "titleColor": INK_2, "labelFontSize": 11, "orient": "top",
                       "symbolStrokeWidth": 2},
            "header": {"labelColor": INK, "titleColor": INK_2, "labelFontSize": 12},
            "line": {"strokeWidth": 2},
            "range": {"category": SERIES},
        }
    }  # fmt: skip


def save(chart: alt.Chart, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chart.save(str(out_dir / f"{name}.html"))
    png = out_dir / f"{name}.png"
    chart.save(str(png), scale_factor=2)
    return png


def _color(field: str, domain: list[str], title: str | None = None) -> alt.Color:
    return alt.Color(f"{field}:N", scale=alt.Scale(domain=domain, range=SERIES[: len(domain)]),
                     title=title, sort=domain)  # fmt: skip


def pr_curves(points: pd.DataFrame, base_rate: float, order: list[str]) -> alt.Chart:
    """points: model, recall, precision."""
    lines = (
        alt.Chart(points)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "recall:Q", title="Recall (share of fraud caught)", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            color=_color("model", order, "Model"),
            tooltip=[
                "model:N",
                alt.Tooltip("recall:Q", format=".2f"),
                alt.Tooltip("precision:Q", format=".2f"),
            ],
        )
    )
    base = (
        alt.Chart(pd.DataFrame({"y": [base_rate]}))
        .mark_rule(strokeDash=[4, 4], color=MUTED)
        .encode(y="y:Q")
    )
    label = alt.Chart(pd.DataFrame({"y": [base_rate], "x": [1.0], "t": [f"random model ({base_rate:.1%})"]})) \
        .mark_text(align="right", dy=-6, color=INK_2, fontSize=11).encode(x="x:Q", y="y:Q", text="t:N")  # fmt: skip
    return (base + lines + label).properties(
        width=WIDTH, height=HEIGHT,
        title=alt.Title("Precision-recall on the test window", subtitle="Dashed line = what random scores achieve"),
    )  # fmt: skip


def reliability(table: pd.DataFrame) -> alt.Chart:
    """table: model, kind (raw/calibrated), mean_pred, observed."""
    diag = (
        alt.Chart()
        .mark_rule(strokeDash=[4, 4], color=MUTED, strokeWidth=1)
        .encode(x=alt.datum(0), y=alt.datum(0), x2=alt.datum(1), y2=alt.datum(1))
    )
    pts = (
        alt.Chart()
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True, stroke=SURFACE, strokeWidth=2))
        .encode(
            x=alt.X(
                "mean_pred:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("observed:Q", title="Observed fraud rate", scale=alt.Scale(domain=[0, 1])),
            color=_color("kind", ["raw", "calibrated"], "Scores"),
            tooltip=[
                "model:N",
                "kind:N",
                alt.Tooltip("mean_pred:Q", format=".3f"),
                alt.Tooltip("observed:Q", format=".3f"),
            ],
        )
    )
    return alt.layer(diag, pts, data=table).properties(width=230, height=230).facet(
        column=alt.Column("model:N", title=None)
    ).properties(title=alt.Title("Calibration on the test window (10 probability bins, n >= 20)",
                                 subtitle="On the dashed diagonal, a predicted 30% means 30% are fraud"))  # fmt: skip


def cost_curve(curve: pd.DataFrame, markers: pd.DataFrame) -> alt.Chart:
    """curve: flag_rate, total_cost. markers: policy, decline_rate, total_cost."""
    line = alt.Chart(curve).mark_line(color=SERIES[0]).encode(
        x=alt.X("flag_rate:Q", title="Share of transactions declined", axis=alt.Axis(format="%")),
        y=alt.Y("total_cost:Q", title="Total cost on test (USD)", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("flag_rate:Q", format=".2%"), alt.Tooltip("total_cost:Q", format="$,.0f"),
                 alt.Tooltip("fraud_dollar_recall:Q", format=".1%", title="fraud $ caught")],
    )  # fmt: skip
    m = markers.reset_index(drop=True).copy()
    m["dy"] = [-14 if i % 2 == 0 else 16 for i in range(len(m))]  # alternate label side
    m["label"] = m.policy.str.replace("Model: ", "", regex=False)
    pts = (
        alt.Chart(m)
        .mark_point(size=90, filled=True, color=SERIES[1], stroke=SURFACE, strokeWidth=2)
        .encode(
            x="decline_rate:Q",
            y="total_cost:Q",
            tooltip=[
                "policy:N",
                alt.Tooltip("decline_rate:Q", format=".2%"),
                alt.Tooltip("total_cost:Q", format="$,.0f"),
            ],
        )
    )
    labels = [
        alt.Chart(m.iloc[[i]])
        .mark_text(align="left", dx=8, dy=int(m.dy[i]), color=INK_2, fontSize=11)
        .encode(x="decline_rate:Q", y="total_cost:Q", text="label:N")
        for i in range(len(m))
    ]
    return alt.layer(line, pts, *labels).properties(
        width=WIDTH, height=HEIGHT,
        title=alt.Title("Cost of a flat decline threshold, and where each fitted policy lands",
                        subtitle="Line = decline everything above a score cutoff; dots = policies fit on calib"),
    )  # fmt: skip


def policy_bars(table: pd.DataFrame) -> alt.Chart:
    """table: policy, total_cost, savings."""
    order = table.sort_values("total_cost", ascending=False).policy.tolist()
    bars = (
        alt.Chart(table)
        .mark_bar(color=SERIES[0], cornerRadiusEnd=4, height=18)
        .encode(
            y=alt.Y("policy:N", sort=order, title=None, axis=alt.Axis(labelLimit=400)),
            x=alt.X(
                "total_cost:Q", title="Total cost on test (USD)", axis=alt.Axis(format="$,.0f")
            ),
            tooltip=[
                "policy:N",
                alt.Tooltip("total_cost:Q", format="$,.0f"),
                alt.Tooltip("savings:Q", format=".1%"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=4, color=INK_2, fontSize=11).encode(
        text=alt.Text("savings:Q", format=".0%")
    )
    return (bars + text).properties(width=WIDTH, height=36 * len(table),
                                    title=alt.Title("Total cost by policy", subtitle="Label = saving vs approving everything"))  # fmt: skip


def sensitivity_heatmap(grid: pd.DataFrame, policy: str) -> alt.Chart:
    """grid: friction_cost, lost_margin_rate, review_cap, policy, savings."""
    d = grid[grid.policy == policy].copy()
    d["cap"] = (d.review_cap * 100).map(lambda v: f"review cap {v:g}%")
    d["margin"] = d.lost_margin_rate.map(lambda v: f"{v:.0%}")
    base = alt.Chart().encode(
        x=alt.X("friction_cost:O", title="Fixed friction per false decline ($)"),
        y=alt.Y("margin:O", title="Lost margin (share of amount)", sort=["50%", "25%", "10%"]),
    )
    rect = base.mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("savings:Q", scale=alt.Scale(range=SEQ_BLUE[1:]), title="Saving vs approve-all",
                        legend=alt.Legend(format=".0%")),
        tooltip=["friction_cost:O", "margin:N", "cap:N", alt.Tooltip("savings:Q", format=".1%")],
    )  # fmt: skip
    text = base.mark_text(fontSize=12, fontWeight=500).encode(
        text=alt.Text("savings:Q", format=".0%"),
        color=alt.condition("datum.savings > 0.5", alt.value("#ffffff"), alt.value(INK)),
    )
    return alt.layer(rect, text, data=d).properties(width=170, height=150).facet(column=alt.Column("cap:N", title=None)).properties(
        title=alt.Title(f"Sensitivity of the {policy} policy to cost assumptions",
                        subtitle="Thresholds refit on calib for every cell; savings measured on test"))  # fmt: skip


def review_sensitivity_heatmap(grid: pd.DataFrame) -> alt.Chart:
    """grid: reviewer_accuracy, review_cost, review_cap, savings, no_review_savings."""
    d = grid.copy()
    d["cap"] = (d.review_cap * 100).map(lambda v: f"review cap {v:g}%")
    d["accuracy"] = d.reviewer_accuracy.map(lambda v: f"{v:.0%}")
    floor = float(d.no_review_savings.iloc[0])
    acc_order = [f"{v:.0%}" for v in sorted(d.reviewer_accuracy.unique(), reverse=True)]
    base = alt.Chart().encode(
        x=alt.X("review_cost:O", title="Cost per manual review ($)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:O", title="Reviewer accuracy", sort=acc_order),
    )
    rect = base.mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("savings:Q", scale=alt.Scale(range=SEQ_BLUE[1:]), title="Saving vs approve-all",
                        legend=alt.Legend(format=".0%")),
        tooltip=["review_cost:O", "accuracy:N", "cap:N", alt.Tooltip("savings:Q", format=".1%"),
                 alt.Tooltip("review_rate:Q", format=".2%", title="review rate")],
    )  # fmt: skip
    text = base.mark_text(fontSize=12, fontWeight=500).encode(
        text=alt.Text("savings:Q", format=".0%"),
        color=alt.condition("datum.savings > 0.4", alt.value("#ffffff"), alt.value(INK)),
    )
    return alt.layer(rect, text, data=d).properties(width=170, height=150).facet(column=alt.Column("cap:N", title=None)).properties(
        title=alt.Title("Sensitivity of the three-tier policy to manual-review assumptions",
                        subtitle=f"Decline costs at base values; without a review tier the best policy saves {floor:.0%}"))  # fmt: skip


def review_queue(q: pd.DataFrame) -> alt.Chart:
    long = q.melt(
        id_vars="day", value_vars=["reviews", "capacity"], var_name="series", value_name="count"
    )
    return alt.Chart(long).mark_line().encode(
        x=alt.X("day:Q", title="Day (relative)"),
        y=alt.Y("count:Q", title="Cases per day"),
        color=_color("series", ["reviews", "capacity"], None),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["reviews", "capacity"], range=[[1, 0], [4, 4]]), legend=None),
        tooltip=["day:Q", "series:N", "count:Q"],
    ).properties(width=WIDTH, height=220, title=alt.Title("Daily manual-review queue vs capacity",
                                                         subtitle="Threshold fit on calib; test days shown"))  # fmt: skip


def importance(table: pd.DataFrame, groups: list[str]) -> alt.Chart:
    """table: feature, mean_abs_shap, group."""
    order = table.sort_values("mean_abs_shap", ascending=False).feature.tolist()
    return alt.Chart(table).mark_bar(cornerRadiusEnd=4, height=14).encode(
        y=alt.Y("feature:N", sort=order, title=None),
        x=alt.X("mean_abs_shap:Q", title="Mean |SHAP| (log-odds)"),
        color=_color("group", groups, "Feature group"),
        tooltip=["feature:N", "group:N", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
    ).properties(width=WIDTH - 120, height=18 * len(table),
                 title=alt.Title("What drives the champion model", subtitle="TreeSHAP on a test sample"))  # fmt: skip


def group_importance(table: pd.DataFrame) -> alt.Chart:
    order = table.sort_values("share", ascending=False).group.tolist()
    bars = (
        alt.Chart(table)
        .mark_bar(color=SERIES[0], cornerRadiusEnd=4, height=18)
        .encode(
            y=alt.Y("group:N", sort=order, title=None),
            x=alt.X("share:Q", title="Share of total |SHAP|", axis=alt.Axis(format="%")),
            tooltip=["group:N", alt.Tooltip("share:Q", format=".1%")],
        )
    )
    text = bars.mark_text(align="left", dx=4, color=INK_2).encode(
        text=alt.Text("share:Q", format=".0%")
    )
    return (bars + text).properties(width=WIDTH - 120, height=30 * len(table),
                                    title=alt.Title("Attribution by feature group", subtitle="Masked = vendor V/C/D/M/id columns"))  # fmt: skip


def backtest(table: pd.DataFrame) -> alt.Chart:
    """table: test_window, metric, value."""
    order = table.metric.unique().tolist()
    return alt.Chart(table).mark_line(point=alt.OverlayMarkDef(size=60, filled=True, stroke=SURFACE, strokeWidth=2)).encode(
        x=alt.X("test_window:O", title="Scored window (30-day blocks)"),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=_color("metric", order, None),
        tooltip=["test_window:O", "metric:N", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=WIDTH, height=240, title=alt.Title("Rolling-origin backtest",
                                                         subtitle="Train on all earlier windows, score the next one"))  # fmt: skip


def psi_heatmap(table: pd.DataFrame) -> alt.Chart:
    """table: feature, window, psi."""
    order = table.groupby("feature").psi.max().sort_values(ascending=False).index.tolist()
    base = alt.Chart(table).encode(x=alt.X("window:O", title="Window (30-day blocks)"),
                                   y=alt.Y("feature:N", sort=order, title=None))  # fmt: skip
    rect = base.mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("psi:Q", scale=alt.Scale(range=SEQ_BLUE, domainMin=0), title="PSI"),
        tooltip=["feature:N", "window:O", alt.Tooltip("psi:Q", format=".3f")],
    )
    text = base.mark_text(fontSize=10).encode(
        text=alt.Text("psi:Q", format=".2f"),
        color=alt.condition("datum.psi > 0.2", alt.value("#ffffff"), alt.value(INK_2)),
    )
    return (rect + text).properties(width=60 * table.window.nunique(), height=16 * len(order),
                                    title=alt.Title("Feature drift vs the training window (PSI)",
                                                    subtitle="> 0.1 moderate shift, > 0.2 large shift"))  # fmt: skip
