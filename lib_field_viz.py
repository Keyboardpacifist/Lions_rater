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


def build_rb_narrative(player_carries: pd.DataFrame,
                        full_rusher_plays: pd.DataFrame | None = None,
                        peer_pools: dict | None = None,
                        min_player_carries_per_gap: int = 8,
                        min_peer_carries_per_gap: int = 50) -> str | None:
    """Generate 2-3 sentences describing the RB's gap profile.

    Pattern:
        "Gibbs's signature is the A-gap — 3rd of 47 RBs (50+ carries)
         at +0.19 EPA per carry. His weakness is the right-B-gap —
         stuffed on 19% of attempts, ranks 38 of 41 in EPA there."

    Args:
        player_carries: this player's full per-play rusher dataframe
            (already classified into gap_code via _classify_gap).
            Must have columns: gap_code, epa, yards_gained, fo_success.
        full_rusher_plays: the complete league rusher_plays dataframe
            for peer pool construction. Must include gap_code column
            (or we'll classify on the fly).
        min_player_carries_per_gap: don't pick a "signature" or "weakness"
            from a gap with fewer than this many carries — too noisy.
        min_peer_carries_per_gap: minimum carries through a gap for an
            RB to count as a peer for ranking purposes.

    Returns None if the player has too little data for a meaningful
    narrative (< 50 career carries or < 2 gaps with sample).
    """
    if player_carries is None or player_carries.empty:
        return None
    if len(player_carries) < 50:
        return None  # journeyman with too little data — skip blurb
    if "gap_code" not in player_carries.columns:
        return None

    # Player's per-gap aggregate
    player_pool = player_carries[player_carries["gap_code"].notna()]
    if player_pool.empty:
        return None
    player_agg = (player_pool.groupby("gap_code")
                              .agg(carries=("yards_gained", "size"),
                                    epa=("epa", "mean"),
                                    yards=("yards_gained", "sum"),
                                    stuffs=("yards_gained",
                                             lambda s: int((s.fillna(0) <= 0).sum())))
                              .reset_index())
    player_agg["ypc"] = player_agg["yards"] / player_agg["carries"]
    player_agg["stuff_rate"] = player_agg["stuffs"] / player_agg["carries"]

    # Drop gaps with too few carries — protects against noise.
    qualifying = player_agg[player_agg["carries"] >= min_player_carries_per_gap]
    if len(qualifying) < 2:
        return None

    # Build peer pools per gap — RBs with ≥ min_peer carries through
    # that gap. Use the caller's pre-built pools when available
    # (typically from a Streamlit cached loader); otherwise build
    # from `full_rusher_plays`.
    if peer_pools is None:
        if full_rusher_plays is None:
            return None
        peer_pools = _build_peer_gap_pools(full_rusher_plays,
                                             min_carries=min_peer_carries_per_gap)

    # Signature gap: highest EPA among qualifying gaps.
    sig = qualifying.loc[qualifying["epa"].idxmax()]
    weak = qualifying.loc[qualifying["epa"].idxmin()]

    sig_label = _GAP_PROSE.get(sig["gap_code"], sig["gap_code"])
    weak_label = _GAP_PROSE.get(weak["gap_code"], weak["gap_code"])

    # Look up rank for each
    from lib_shared import compute_rank_in_pool

    sig_rank, sig_total = (None, 0)
    if sig["gap_code"] in peer_pools:
        sig_rank, sig_total = compute_rank_in_pool(
            sig["epa"], peer_pools[sig["gap_code"]]["epa"], ascending=False
        )

    weak_rank, weak_total = (None, 0)
    if weak["gap_code"] in peer_pools:
        weak_rank, weak_total = compute_rank_in_pool(
            weak["epa"], peer_pools[weak["gap_code"]]["epa"], ascending=False
        )

    # Format the sentences.
    sentences = []

    # Signature sentence
    sig_epa_str = f"{sig['epa']:+.2f}"
    if sig_rank and sig_total:
        sentences.append(
            f"**Signature: the {sig_label}** — "
            f"{sig_rank} of {sig_total} RBs (50+ carries) at "
            f"{sig_epa_str} EPA per carry on {int(sig['carries'])} attempts."
        )
    else:
        sentences.append(
            f"**Signature: the {sig_label}** — "
            f"{sig_epa_str} EPA per carry on {int(sig['carries'])} attempts."
        )

    # Weakness sentence — only show if it's meaningfully negative or
    # ranks poorly. If everything is positive, frame as "second-best."
    if sig["gap_code"] != weak["gap_code"]:
        weak_epa_str = f"{weak['epa']:+.2f}"
        if weak["epa"] < 0 or (weak_rank and weak_total
                                and weak_rank > weak_total * 0.6):
            stuff_pct = f"{weak['stuff_rate']*100:.0f}%"
            if weak_rank and weak_total:
                sentences.append(
                    f"**Weakness: the {weak_label}** — "
                    f"{weak_epa_str} EPA, stuffed on {stuff_pct} of attempts, "
                    f"{weak_rank} of {weak_total} in EPA there."
                )
            else:
                sentences.append(
                    f"**Weakness: the {weak_label}** — "
                    f"{weak_epa_str} EPA, stuffed on {stuff_pct} of attempts."
                )
        else:
            # Player is productive everywhere — soften the framing
            sentences.append(
                f"Productive in every direction — even his "
                f"least-effective gap ({weak_label}) is at "
                f"{weak_epa_str} EPA per carry."
            )

    return " ".join(sentences)


def build_wr_narrative(player_targets: pd.DataFrame,
                        full_targeted_plays: pd.DataFrame | None = None,
                        peer_pools: dict | None = None,
                        min_player_targets_per_route: int = 8,
                        min_peer_targets_per_route: int = 30) -> str | None:
    """Generate 2-3 sentences describing the receiver's route-tree
    profile.

    Pattern:
        "Signature: the SLANT — 8th of 92 receivers (30+ targets) at
         +0.45 EPA per target on 50 attempts. Weakness: the GO route —
         -0.30 EPA, only catches 22% of deep targets."

    Returns None when the player has too little data (< 30 career
    targets or < 2 routes with sample).
    """
    if player_targets is None or player_targets.empty:
        return None
    if len(player_targets) < 30:
        return None
    if "route" not in player_targets.columns:
        return None

    # Player's per-route aggregate
    pool = player_targets[player_targets["route"].notna()
                           & (player_targets["route"] != "")]
    if pool.empty:
        return None

    td_col = "pass_touchdown" if "pass_touchdown" in pool.columns else "touchdown"

    agg = (pool.groupby("route")
                .agg(targets=("epa", "size"),
                      epa=("epa", "mean"),
                      yards=("yards_gained", "sum"),
                      catches=("complete_pass",
                                lambda s: int(s.fillna(0).sum())),
                      tds=(td_col,
                            lambda s: int(s.fillna(0).sum())))
                .reset_index())
    agg["catch_rate"] = agg["catches"] / agg["targets"]
    agg["ypt"] = agg["yards"] / agg["targets"]

    qualifying = agg[agg["targets"] >= min_player_targets_per_route]
    if len(qualifying) < 2:
        return None

    # Peer pools: per-route, all targeted players with ≥ min targets.
    if peer_pools is None:
        if full_targeted_plays is None:
            return None
        peer_pools = _build_route_peer_pools(
            full_targeted_plays, min_targets=min_peer_targets_per_route
        )

    sig = qualifying.loc[qualifying["epa"].idxmax()]
    weak = qualifying.loc[qualifying["epa"].idxmin()]

    from lib_shared import compute_rank_in_pool

    sig_rank, sig_total = (None, 0)
    if sig["route"] in peer_pools:
        sig_rank, sig_total = compute_rank_in_pool(
            sig["epa"], peer_pools[sig["route"]]["epa"], ascending=False
        )

    weak_rank, weak_total = (None, 0)
    if weak["route"] in peer_pools:
        weak_rank, weak_total = compute_rank_in_pool(
            weak["epa"], peer_pools[weak["route"]]["epa"], ascending=False
        )

    # Format
    sig_label = _route_label(sig["route"])
    weak_label = _route_label(weak["route"])

    sentences = []
    sig_epa_str = f"{sig['epa']:+.2f}"
    if sig_rank and sig_total:
        sentences.append(
            f"**Signature route: the {sig_label}** — "
            f"{sig_rank} of {sig_total} receivers (30+ targets) at "
            f"{sig_epa_str} EPA per target on {int(sig['targets'])} attempts."
        )
    else:
        sentences.append(
            f"**Signature route: the {sig_label}** — "
            f"{sig_epa_str} EPA per target on {int(sig['targets'])} attempts."
        )

    if sig["route"] != weak["route"]:
        weak_epa_str = f"{weak['epa']:+.2f}"
        if weak["epa"] < 0 or (weak_rank and weak_total
                                and weak_rank > weak_total * 0.6):
            catch_pct = f"{weak['catch_rate']*100:.0f}%"
            if weak_rank and weak_total:
                sentences.append(
                    f"**Weakness: the {weak_label}** — "
                    f"{weak_epa_str} EPA, {catch_pct} catch rate, "
                    f"{weak_rank} of {weak_total} on this route."
                )
            else:
                sentences.append(
                    f"**Weakness: the {weak_label}** — "
                    f"{weak_epa_str} EPA, {catch_pct} catch rate."
                )
        else:
            sentences.append(
                f"Productive across the route tree — even his "
                f"least-effective route ({weak_label}) is at "
                f"{weak_epa_str} EPA per target."
            )

    return " ".join(sentences)


def build_position_narrative(player_row, peer_pool: pd.DataFrame,
                               stat_labels: dict[str, str],
                               position_label: str,
                               max_stats_to_consider: int = 12) -> str | None:
    """Generic 'signature stat + weakness stat' narrative engine for
    positions that don't have a custom panel (defense / OL / QB / K / P).

    Picks the player's highest-z-score stat as their signature and
    lowest-z-score stat as their weakness, then ranks them against
    the position pool by z-score (which is monotonic with raw stat,
    and accounts for pipeline-level inversions on "lower is better"
    metrics like penalty rate).

    Args:
        player_row: a pandas Series indexed by column name (one
            row from the position's league parquet).
        peer_pool: full DataFrame for the position (the league
            parquet, all rows).
        stat_labels: mapping from z-column name → human label,
            from {position}_stat_metadata.json.
        position_label: short string for the rank context, e.g.
            "EDGE rushers" or "interior linemen" or "linebackers".
        max_stats_to_consider: cap on how many z-cols to weigh —
            defends against late-tier inferred stats from skewing
            the signature pick.

    Returns formatted text or None if data insufficient.
    """
    if player_row is None:
        return None

    # Player's z-score columns
    z_cols = {c: player_row[c] for c in player_row.index
                if c.endswith("_z")
                and c in peer_pool.columns
                and pd.notna(player_row[c])}
    if len(z_cols) < 2:
        return None

    # Cap on stats considered (already-sorted-by-importance order
    # would be ideal, but z-cols dict order from the parquet is
    # close enough — just clamp to the first N).
    z_cols_list = list(z_cols.items())[:max_stats_to_consider]
    z_dict = dict(z_cols_list)

    sig_col = max(z_dict, key=z_dict.get)
    weak_col = min(z_dict, key=z_dict.get)
    if sig_col == weak_col:
        return None

    from lib_shared import compute_rank_in_pool, format_rank

    sig_rank, sig_total = compute_rank_in_pool(
        player_row[sig_col], peer_pool[sig_col].dropna(), ascending=False
    )
    weak_rank, weak_total = compute_rank_in_pool(
        player_row[weak_col], peer_pool[weak_col].dropna(), ascending=False
    )

    sig_label = stat_labels.get(sig_col, sig_col.replace("_z", "").replace("_", " "))
    weak_label = stat_labels.get(weak_col, weak_col.replace("_z", "").replace("_", " "))

    sig_z = float(player_row[sig_col])
    weak_z = float(player_row[weak_col])

    sentences = []
    sig_rank_str = format_rank(sig_rank, sig_total)
    if sig_rank_str != "—":
        # If the player's "best" stat is still below average, frame
        # the narrative honestly instead of pretending it's a strength.
        if sig_z >= 0:
            sentences.append(
                f"**Signature: {sig_label}** — {sig_rank_str} {position_label}."
            )
        else:
            sentences.append(
                f"Best area: {sig_label} — {sig_rank_str} {position_label}, "
                f"but still below the position average."
            )

    weak_rank_str = format_rank(weak_rank, weak_total)
    if weak_rank_str != "—":
        if weak_z >= 0 and weak_rank is not None and weak_total > 0 and weak_rank <= weak_total // 2:
            sentences.append(
                f"Weakest area still above-average: **{weak_label}** "
                f"({weak_rank_str})."
            )
        else:
            sentences.append(
                f"**Weakness: {weak_label}** — {weak_rank_str}."
            )

    return " ".join(sentences) if sentences else None


def _route_label(route_code: str) -> str:
    """Render route names as Title-Cased prose for the narrative."""
    if not route_code:
        return route_code
    if route_code.upper() == "GO":
        return "GO route"
    if route_code.upper() == "SLANT":
        return "slant"
    if route_code.upper() == "POST":
        return "post"
    if route_code.upper() == "CORNER":
        return "corner"
    if route_code.upper() == "HITCH":
        return "hitch"
    if route_code.upper() in ("IN", "IN/DIG"):
        return route_code.lower().replace("_", " ")
    if route_code.upper() == "OUT":
        return "out"
    if route_code.upper() == "DEEP OUT":
        return "deep out"
    if route_code.upper() == "QUICK OUT":
        return "quick out"
    if route_code.upper() == "WHEEL":
        return "wheel"
    if route_code.upper() == "FLAT":
        return "flat"
    if route_code.upper() == "SCREEN":
        return "screen"
    if route_code.upper() == "ANGLE":
        return "angle"
    if route_code.upper() == "TEXAS/ANGLE":
        return "Texas/angle"
    if "CROSS" in route_code.upper():
        return route_code.lower()
    return route_code.lower()


def _build_route_peer_pools(targeted_plays: pd.DataFrame,
                             min_targets: int = 30) -> dict[str, pd.DataFrame]:
    """For each route, return a per-player aggregate of receivers
    with ≥ min_targets on that route. Used to rank a player's
    per-route EPA against peers."""
    if targeted_plays is None or targeted_plays.empty:
        return {}
    df = targeted_plays[targeted_plays["route"].notna()
                          & (targeted_plays["route"] != "")]
    if df.empty:
        return {}
    pools: dict[str, pd.DataFrame] = {}
    for route, sub in df.groupby("route"):
        per_player = (sub.groupby("player_id")
                          .agg(targets=("epa", "size"),
                                epa=("epa", "mean"))
                          .reset_index())
        qualified = per_player[per_player["targets"] >= min_targets]
        pools[str(route)] = qualified
    return pools


# Long-form prose names for each gap, used in narrative sentences.
_GAP_PROSE = {
    "A":   "A-gap (up the middle)",
    "B-L": "left B-gap",
    "B-R": "right B-gap",
    "C-L": "left C-gap (off-tackle)",
    "C-R": "right C-gap (off-tackle)",
    "D-L": "left D-gap (outside)",
    "D-R": "right D-gap (outside)",
}


def _build_peer_gap_pools(full_rusher_plays: pd.DataFrame,
                            min_carries: int = 50) -> dict[str, pd.DataFrame]:
    """For each gap, return a per-player aggregate DataFrame containing
    only RBs with ≥ `min_carries` through that gap. Used as the peer
    pool for ranking calculations.

    The result is keyed by gap_code; each value is a DataFrame with
    columns: player_id, carries, epa.
    """
    if full_rusher_plays is None or full_rusher_plays.empty:
        return {}

    df = full_rusher_plays.copy()
    # Ensure gap_code is present (caller may have already classified)
    if "gap_code" not in df.columns:
        from lib_splits import _classify_gap   # avoid circular at module load
        df["gap_code"] = df.apply(_classify_gap, axis=1)
    df = df[df["gap_code"].notna()]

    pools: dict[str, pd.DataFrame] = {}
    for gap, sub in df.groupby("gap_code"):
        per_player = (sub.groupby("player_id")
                          .agg(carries=("yards_gained", "size"),
                                epa=("epa", "mean"))
                          .reset_index())
        qualified = per_player[per_player["carries"] >= min_carries]
        pools[str(gap)] = qualified
    return pools


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
