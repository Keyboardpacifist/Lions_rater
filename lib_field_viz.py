"""
Field-shaped visualizations — the "wow" charts.

Two builders:

  build_route_tree(targeted_plays, metric, ...) -> plotly Figure
    Reception-Perception-style: all routes branch off a single receiver
    stem, each colored by chosen performance metric on the
    red→green heatmap. Filterable by coverage / man-zone / formation
    / personnel upstream — caller passes the already-filtered slice.

  build_gap_diagram(rusher_plays, metric, ...) -> plotly Figure
    Line-of-scrimmage diagram with the offensive line drawn as boxes
    and the 7 gap zones colored by chosen rushing metric. Filterable
    by box / formation / personnel upstream.

Both helpers ASSUME the caller has already filtered the dataframe.
This module is pure presentation — it doesn't load data, doesn't
own filter state, just turns a slice into a beautiful figure.

Geometry note: routes are drawn from the receiver's perspective at
(0, 0) facing upfield (+Y). Inside breaks go +X, outside breaks go
−X. This is a stylized canonical layout — NOT real per-play tracking.
The real tracking data isn't in nflverse. This visualization is
about *which* routes get run and *how well*, not where on the field.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


# ──────────────────────────────────────────────────────────────────
# Route geometry library
#
# Each route is a polyline of (x, y) waypoints. Receiver stem at
# (0, 0), upfield is +Y. Inside breaks go +X (toward the middle of
# the field), outside breaks go −X. Stylized for visual clarity —
# the real thing has more curve and variance.
# ──────────────────────────────────────────────────────────────────
ROUTE_PATHS: dict[str, list[tuple[float, float]]] = {
    # --- Vertical / deep ---
    "GO":           [(0, 0), (0, 22)],
    "POST":         [(0, 0), (0, 12), (4.5, 20)],     # 45° in at 12y
    "CORNER":       [(0, 0), (0, 12), (-4.5, 20)],    # 45° out at 12y

    # --- Intermediate in-breaking ---
    "IN":           [(0, 0), (0, 9), (5, 9)],
    "IN/DIG":       [(0, 0), (0, 13), (6, 13)],

    # --- Intermediate out-breaking ---
    "OUT":          [(0, 0), (0, 9), (-5, 9)],
    "DEEP OUT":     [(0, 0), (0, 14), (-6, 14)],

    # --- Comeback / settle ---
    "HITCH":        [(0, 0), (0, 6)],
    "HITCH/CURL":   [(0, 0), (0, 13), (-1, 12)],

    # --- Quick / short ---
    "QUICK OUT":    [(0, 0), (0, 4), (-3.5, 4)],
    "SLANT":        [(0, 0), (0, 2), (5, 6)],
    "CROSS":        [(0, 0), (0, 3), (8, 4)],
    "SHALLOW CROSS/DRAG": [(0, 0), (0, 2), (7, 3)],

    # --- Behind / at LOS ---
    "FLAT":         [(0, 0), (-2, 0.5), (-7, 1)],
    "SCREEN":       [(0, 0), (0, -1.5), (-3, -1.5)],
    "SWING":        [(0, 0), (-2, -0.5), (-7, 0)],

    # --- RB-specific ---
    "WHEEL":        [(0, 0), (-1.5, 1), (-3, 3), (-3, 14)],
    "ANGLE":        [(0, 0), (-2, 2), (-3, 4), (4, 8)],
    "TEXAS/ANGLE":  [(0, 0), (-2, 2), (-3, 5), (-2, 12)],
}

# Routes we don't have geometry for (unknown/legacy values) get a
# small radial offset so they still appear without overlapping the
# stem. Sorted in canonical order so the chart stays stable across
# sessions.
_ROUTE_DISPLAY_ORDER = list(ROUTE_PATHS.keys())


def _format_metric(metric_key: str, value: float) -> str:
    """Format a metric value for display in hover/labels."""
    if pd.isna(value):
        return "—"
    if metric_key == "epa_per_target":
        return f"{value:+.2f}"
    if metric_key.endswith("_rate") or metric_key in ("catch_rate", "success_rate"):
        return f"{value * 100:.0f}%"
    if metric_key in ("targets", "completions", "tds"):
        return f"{int(value)}"
    if metric_key in ("yards_per_target", "adot"):
        return f"{value:.1f}"
    return f"{value:.2f}"


# Heatmap range per metric — used to map a metric value to the
# red→green gradient via lib_shared.heatmap_color.
_METRIC_RANGES: dict[str, tuple[float, float]] = {
    "epa_per_target":  (-0.30, 0.50),    # NFL passing EPA range
    "catch_rate":      (0.40, 0.85),
    "success_rate":    (0.30, 0.65),
    "yards_per_target": (5.0, 15.0),
    "adot":            (4.0, 14.0),
    "targets":         (1, 30),          # purely volume — green = lots of targets
    "td_rate":         (0.0, 0.20),
}


def build_route_tree(plays: pd.DataFrame, *, metric: str = "epa_per_target",
                      min_targets: int = 1,
                      title: str | None = None) -> go.Figure:
    """Build a route-tree figure from a slice of targeted plays.

    `plays` must have at minimum a `route` column. Per-route metrics
    are computed inline.

    `metric` ∈ {epa_per_target, catch_rate, success_rate,
    yards_per_target, adot, targets, td_rate}. Routes with fewer than
    `min_targets` targets are drawn faded (low opacity) to honor
    sample size.
    """
    from lib_shared import heatmap_color

    if plays.empty or "route" not in plays.columns:
        return _empty_fig("No targeted plays in this slice.")

    pool = plays[plays["route"].notna() & (plays["route"] != "")].copy()
    if pool.empty:
        return _empty_fig("No labeled routes in this slice.")

    # Per-route aggregates. nfl_targeted_plays uses `pass_touchdown`
    # not `touchdown`; success isn't in the schema, so derive it from
    # epa>0 (a fair public-data proxy for success rate on pass plays).
    td_col = "pass_touchdown" if "pass_touchdown" in pool.columns else "touchdown"
    grp = pool.groupby("route")
    route_stats: dict[str, dict] = {}
    for route, sub in grp:
        n = len(sub)
        if n < 1:
            continue
        targets = n
        completions = int(sub.get("complete_pass", pd.Series([0]*n)).fillna(0).sum())
        yards = float(sub.get("yards_gained", pd.Series([0]*n)).fillna(0).sum())
        epa = float(sub.get("epa", pd.Series([0]*n)).fillna(0).mean())
        air = float(sub.get("air_yards", pd.Series([0]*n)).fillna(0).mean())
        tds = int(sub.get(td_col, pd.Series([0]*n)).fillna(0).sum())
        # Prefer fo_success (PFF/PFR-aligned) when available — added by
        # the loaders in lib_splits. Fallback to nflverse 'success'
        # (EPA-based) only if neither column exists.
        if "fo_success" in sub.columns:
            success = float(sub["fo_success"].fillna(0).mean())
        elif "success" in sub.columns:
            success = float(sub["success"].fillna(0).mean())
        else:
            success = float((sub.get("epa", pd.Series([0]*n)).fillna(0) > 0).mean())
        catch_rate = completions / targets if targets else 0.0
        ypt = yards / targets if targets else 0.0
        td_rate = tds / targets if targets else 0.0
        route_stats[route] = {
            "targets": targets,
            "completions": completions,
            "yards": yards,
            "epa_per_target": epa,
            "adot": air,
            "td_rate": td_rate,
            "catch_rate": catch_rate,
            "success_rate": success,
            "yards_per_target": ypt,
            "tds": tds,
        }

    # Build the figure
    fig = go.Figure()
    lo, hi = _METRIC_RANGES.get(metric, (0.0, 1.0))

    # Light field grid for context (10-yd line at +10, end of frame)
    for y_yd in (5, 10, 15, 20):
        fig.add_shape(type="line",
                      x0=-12, y0=y_yd, x1=12, y1=y_yd,
                      line=dict(color="rgba(180,180,180,0.30)",
                                width=1, dash="dot"),
                      layer="below")

    # Line of scrimmage
    fig.add_shape(type="line",
                  x0=-12, y0=0, x1=12, y1=0,
                  line=dict(color="#333", width=2),
                  layer="below")

    # Stem dot — the receiver
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=14, color="#1a1a2e",
                    line=dict(color="white", width=2)),
        showlegend=False, hoverinfo="skip",
    ))

    # Draw each route the receiver actually ran
    for route in _ROUTE_DISPLAY_ORDER:
        if route not in route_stats:
            continue
        if route not in ROUTE_PATHS:
            continue
        stats = route_stats[route]
        n = stats["targets"]
        if n < 1:
            continue

        path = ROUTE_PATHS[route]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        val = stats.get(metric)
        color = heatmap_color(val if val is not None else 0.0,
                              lo=lo, hi=hi)
        # Faded if below sample threshold
        opacity = 1.0 if n >= min_targets else 0.35
        # Line thickness scales with target volume (capped)
        width = max(2.5, min(8.0, 2.0 + n * 0.20))

        m_disp = _format_metric(metric, val)
        hover = (
            f"<b>{route}</b><br>"
            f"Targets: {n}<br>"
            f"Catch%: {stats['catch_rate']*100:.0f}%<br>"
            f"YPT: {stats['yards_per_target']:.1f} · "
            f"aDOT: {stats['adot']:.1f}<br>"
            f"EPA/target: {stats['epa_per_target']:+.2f}<br>"
            f"Success: {stats['success_rate']*100:.0f}%<br>"
            f"TDs: {stats['tds']}"
        )

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=width),
            marker=dict(size=[0] * (len(xs) - 1) + [10],
                         color=color,
                         line=dict(color="white", width=1.5),
                         symbol="circle"),
            opacity=opacity,
            name=f"{route} ({n})",
            hovertext=[hover] * len(xs),
            hoverinfo="text",
            showlegend=False,
        ))

        # Label at the route endpoint — small, color-matched
        end_x, end_y = xs[-1], ys[-1]
        fig.add_annotation(
            x=end_x + (1.0 if end_x >= 0 else -1.0),
            y=end_y + 0.5,
            text=route,
            showarrow=False,
            xanchor="left" if end_x >= 0 else "right",
            font=dict(size=10, color="#2a3a4d", family="Arial Black"),
            opacity=opacity,
        )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        title=(dict(text=title, font=dict(size=15)) if title else None),
        xaxis=dict(range=[-13, 13], visible=False, fixedrange=True),
        yaxis=dict(range=[-3, 24], visible=False, fixedrange=True,
                    scaleanchor="x", scaleratio=1.0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,247,250,0.6)",
        showlegend=False,
        hovermode="closest",
    )
    return fig


def build_gap_diagram(plays: pd.DataFrame, *, metric: str = "epa_per_carry",
                      title: str | None = None) -> go.Figure:
    """Build a line-of-scrimmage gap diagram from rushing plays.

    Draws 5 OL boxes (LT/LG/C/RG/RT) plus optional TE flares; the 7
    gap zones (D-L, C-L, B-L, A, B-R, C-R, D-R) are colored by the
    chosen metric. Caller pre-filters `plays` for the slice they want
    to visualize.

    `plays` must have `run_location`, `run_gap`, `epa`, `yards_gained`,
    `success` columns (matches the schema of nfl_rusher_plays.parquet).
    """
    from lib_shared import heatmap_color

    if plays.empty:
        return _empty_fig("No carries in this slice.")

    # Same gap classifier as the panel uses — keep them in sync
    def _classify(row):
        loc = row.get("run_location")
        gap = row.get("run_gap")
        if loc is None or pd.isna(loc) or loc == "":
            return None
        if loc == "middle":
            return "A"
        if pd.isna(gap) or gap is None or gap == "":
            return None
        side = "L" if loc == "left" else "R" if loc == "right" else None
        if side is None:
            return None
        letter = {"guard": "B", "tackle": "C", "end": "D"}.get(gap)
        return f"{letter}-{side}" if letter else None

    pool = plays.copy()
    pool["gap"] = pool.apply(_classify, axis=1)
    pool = pool[pool["gap"].notna()]
    if pool.empty:
        return _empty_fig("No labeled gap locations in this slice.")

    # Per-gap aggregates — prefer fo_success (PFF/PFR aligned)
    success_col = "fo_success" if "fo_success" in pool.columns else "success"
    agg = pool.groupby("gap").agg(
        carries=("yards_gained", "size"),
        yards=("yards_gained", "sum"),
        epa=("epa", "mean"),
        success=(success_col, "mean"),
        tds=("touchdown", lambda s: int(s.fillna(0).sum())),
        stuffs=("yards_gained", lambda s: int((s.fillna(0) <= 0).sum())),
        chunks=("yards_gained", lambda s: int((s.fillna(0) >= 10).sum())),
    ).reset_index()
    agg["ypc"] = agg["yards"] / agg["carries"]
    metric_value = {
        "epa_per_carry":  dict(zip(agg["gap"], agg["epa"])),
        "ypc":            dict(zip(agg["gap"], agg["ypc"])),
        "success_rate":   dict(zip(agg["gap"], agg["success"])),
        "stuff_rate":     dict(zip(agg["gap"], agg["stuffs"] / agg["carries"])),
        "chunk_rate":     dict(zip(agg["gap"], agg["chunks"] / agg["carries"])),
        "carries":        dict(zip(agg["gap"], agg["carries"])),
    }
    if metric not in metric_value:
        metric = "epa_per_carry"
    vals = metric_value[metric]

    # Per-gap counts for hover
    by_gap = {row["gap"]: row.to_dict() for _, row in agg.iterrows()}

    # Heatmap range
    ranges = {
        "epa_per_carry":  (-0.30, 0.30),
        "ypc":            (2.0, 7.0),
        "success_rate":   (0.30, 0.60),
        "stuff_rate":     (0.30, 0.10),  # reversed — lower is better
        "chunk_rate":     (0.0, 0.20),
        "carries":        (0, 50),
    }
    lo, hi = ranges.get(metric, (0.0, 1.0))

    # Geometry — symmetric, scaled wide for full-page display.
    # Linemen are slim rectangles; gap zones are tall vertical bars
    # (the "vertical bars in formation" Brett asked for) so they
    # read as the dominant visual element.
    OL_X = {"LT": -6.0, "LG": -3.0, "C": 0.0, "RG": 3.0, "RT": 6.0}
    OL_W = 0.8         # slim — the LOS bar isn't the focal point
    HALF = OL_W / 2
    OL_Y0, OL_Y1 = -0.85, -0.05   # short OL boxes hugging the LOS

    # Gap zones — vertical bars between the linemen. Generous height
    # so the heatmap colors carry the panel.
    GAP_ZONES = {
        "D-L": (-11.0, -6.4),                        # outside LT
        "C-L": (-5.6, -3.4),                         # LT ↔ LG
        "B-L": (-2.6, -2.0),                         # LG ↔ A
        "A":   (-2.0, 2.0),                          # both A-gaps over the C
        "B-R": (2.0, 2.6),                           # A ↔ RG
        "C-R": (3.4, 5.6),                           # RG ↔ RT
        "D-R": (6.4, 11.0),                          # outside RT
    }
    GAP_Y0, GAP_Y1 = 0.15, 6.5    # tall vertical bars

    fig = go.Figure()

    # LOS line — thin and crisp
    fig.add_shape(type="line", x0=-11.5, y0=0, x1=11.5, y1=0,
                  line=dict(color="#1a1a2e", width=1.8), layer="below")

    # Gap zone rectangles
    for gap_code, (x0, x1) in GAP_ZONES.items():
        if gap_code in vals:
            v = vals[gap_code]
            color = heatmap_color(v, lo=lo, hi=hi)
        else:
            color = "rgba(220,220,220,0.35)"

        b = by_gap.get(gap_code, {})
        car = int(b.get("carries", 0))
        ypc = b.get("ypc", float("nan"))
        epa = b.get("epa", float("nan"))
        succ = b.get("success", float("nan"))
        chk = int(b.get("chunks", 0))
        stf = int(b.get("stuffs", 0))
        td  = int(b.get("tds", 0))
        ypc_s = f"{ypc:.2f}" if not pd.isna(ypc) else "—"
        epa_s = f"{epa:+.2f}" if not pd.isna(epa) else "—"
        succ_s = f"{succ*100:.0f}%" if not pd.isna(succ) else "—"
        hover = (
            f"<b>{gap_code} gap</b><br>"
            f"{car} carries<br>"
            f"YPC: {ypc_s} · EPA: {epa_s}<br>"
            f"Success: {succ_s}<br>"
            f"Chunks: {chk} · Stuffed: {stf} · TDs: {td}"
        )

        fig.add_shape(type="rect",
                      x0=x0, y0=GAP_Y0,
                      x1=x1, y1=GAP_Y1,
                      fillcolor=color, opacity=0.90,
                      line=dict(color="rgba(0,0,0,0.22)", width=0.6),
                      layer="below")

        cx = (x0 + x1) / 2
        # Gap letter label — top of zone, bold.
        fig.add_trace(go.Scatter(
            x=[cx], y=[GAP_Y1 - 0.5],
            mode="markers+text",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            text=[f"<b>{gap_code}</b>"],
            textfont=dict(size=15, color="#0a3d62", family="Arial"),
            textposition="middle center",
            hovertext=[hover], hoverinfo="text",
            showlegend=False,
        ))
        # Carry count near the bottom of the zone.
        if car > 0:
            fig.add_annotation(
                x=cx, y=GAP_Y0 + 0.4,
                text=f"{car}",
                showarrow=False,
                font=dict(size=11, color="#2a3a4d"),
            )

    # OL boxes — slim bars, drawn ABOVE the A-gap zone so center poking
    # through doesn't distort the data layer.
    for label, x in OL_X.items():
        fig.add_shape(type="rect",
                      x0=x - HALF, y0=OL_Y0,
                      x1=x + HALF, y1=OL_Y1,
                      fillcolor="#1a1a2e", opacity=0.95,
                      line=dict(color="#000", width=0.7),
                      layer="above")
        fig.add_annotation(
            x=x, y=(OL_Y0 + OL_Y1) / 2,
            text=label, showarrow=False,
            font=dict(size=8, color="white", family="Arial Black"),
        )

    # Backfield dot — RB
    fig.add_trace(go.Scatter(
        x=[0], y=[-2.4], mode="markers+text",
        marker=dict(size=11, color="#1a1a2e",
                    line=dict(color="white", width=1.5)),
        text=["RB"], textposition="bottom center",
        textfont=dict(size=9, color="#2a3a4d"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=30, r=30, t=40 if title else 12, b=12),
        title=(dict(text=title, font=dict(size=14)) if title else None),
        xaxis=dict(range=[-12, 12], visible=False, fixedrange=True),
        yaxis=dict(range=[-3.4, 7.2], visible=False, fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.7)",
        showlegend=False,
        hovermode="closest",
    )
    return fig


def _empty_fig(message: str) -> go.Figure:
    """Placeholder figure for empty slices — keeps layout stable."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=12, color="#888"),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(
        height=300,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,247,250,0.5)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
