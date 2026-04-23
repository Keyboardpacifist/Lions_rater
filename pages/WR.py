"""
WR Rater
Wide receivers only. Z-scored within the WR pool — all WRs league-wide with 100+ offensive snaps.
"""
import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from team_selector import get_team_and_season, filter_by_team_and_season, NFL_TEAMS, display_abbr
from career_arc import career_arc_section
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, score_players

st.set_page_config(page_title="WR Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "wr"
PAGE_URL = "https://lions-rater.streamlit.app/WR"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_wr_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wr_te_stat_metadata.json"

@st.cache_data
def load_wr_data():
    df = pl.read_parquet(DATA_PATH).to_pandas()
    return df[df["position"] == "WR"].copy()
@st.cache_data
def load_wr_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "rec_yards_z": "rec_yards", "receptions_z": "receptions",
    "rec_tds_z": "rec_tds", "targets_z": "targets",
    "yards_per_target_z": "yards_per_target", "epa_per_target_z": "epa_per_target",
    "success_rate_z": "success_rate", "catch_rate_z": "catch_rate",
    "first_down_rate_z": "first_down_rate",
    "yac_per_reception_z": "yac_per_reception", "yac_above_exp_z": "yac_above_exp",
    "targets_per_snap_z": "targets_per_snap", "yards_per_snap_z": "yards_per_snap",
    "avg_separation_z": "avg_separation",
}

BUNDLES = {
    "reliability": {
        "label": "🎯 Reliability",
        "description": "Does he catch everything thrown his way? Catch rate, success rate, and first downs.",
        "why": "Think sure hands and moving the chains is what separates great WRs? Crank this up.",
        "stats": {"catch_rate_z": 0.35, "success_rate_z": 0.35, "first_down_rate_z": 0.30},
    },
    "explosive": {
        "label": "💥 Explosive plays",
        "description": "Does he turn targets into chunk plays? Yards per target and YAC above expected.",
        "why": "Want receivers who create big plays, not just catch dump-offs? Slide right.",
        "stats": {"yards_per_target_z": 0.50, "yac_above_exp_z": 0.30, "yards_per_snap_z": 0.20},
    },
    "deep_threat": {
        "label": "🔥 Field stretcher",
        "description": "Can he take the top off the defense? Separation and downfield production.",
        "why": "Value receivers who win deep and create space? Slide right.",
        "stats": {"yards_per_target_z": 0.40, "avg_separation_z": 0.30, "yards_per_snap_z": 0.30},
    },
    "volume": {
        "label": "📊 Volume & usage",
        "description": "How much of the offense runs through him? Targets and yards per snap.",
        "why": "Think the best WR is the one the offense depends on most? Slide right.",
        "stats": {"targets_per_snap_z": 0.50, "yards_per_snap_z": 0.50},
    },
    "after_catch": {
        "label": "🏃 After the catch",
        "description": "What happens once the ball is in his hands? YAC and YAC over expected.",
        "why": "Want receivers who create yards the QB didn't throw? Slide right.",
        "stats": {"yac_per_reception_z": 0.50, "yac_above_exp_z": 0.50},
    },
}
DEFAULT_BUNDLE_WEIGHTS = {"reliability": 60, "explosive": 50, "deep_threat": 30, "volume": 60, "after_catch": 30}

RADAR_STATS = ["yards_per_target_z", "catch_rate_z", "first_down_rate_z", "yac_per_reception_z", "yards_per_snap_z", "epa_per_target_z", "yac_above_exp_z", "avg_separation_z"]
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"yards_per_target_z": "Yds/target", "catch_rate_z": "Catch rate", "first_down_rate_z": "First downs", "yac_per_reception_z": "YAC", "yards_per_snap_z": "Yds/snap", "epa_per_target_z": "EPA/target", "yac_above_exp_z": "YAC over exp", "avg_separation_z": "Separation"}

# ── Score formatting ──────────────────────────────────────────
def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)

def format_percentile(pct):
    if pct is None or pd.isna(pct): return "—"
    if pct >= 99: return "top 1%"
    if pct >= 50: return f"top {100 - int(pct)}%"
    return f"bottom {int(pct)}%"

def format_score(score):
    if pd.isna(score): return "—"
    sign = "+" if score >= 0 else ""
    pct = zscore_to_percentile(score)
    return f"{sign}{score:.2f} ({format_percentile(pct)})"

def sample_size_warning(snaps):
    if pd.isna(snaps): return ""
    if snaps < 300: return f"⚠️ Only {int(snaps)} snaps — small sample, treat with caution"
    if snaps < 500: return f"⚠️ {int(snaps)} snaps — moderate sample"
    return ""

# ── Tier system ───────────────────────────────────────────────
TIER_LABELS = {1: "Counting stats", 2: "Rate stats", 3: "Modeled stats", 4: "Estimated stats"}
TIER_DESCRIPTIONS = {1: "Yards, receptions, TDs — raw totals.", 2: "Per-target and per-snap rates that adjust for opportunity.", 3: "Stats from NFL tracking data — separation, YAC over expected.", 4: "Inferred from limited data — least reliable."}
def tier_badge(tier): return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")

def filter_bundles_by_tier(bundles, stat_tiers, enabled_tiers):
    filtered = {}
    for bk, bdef in bundles.items():
        kept = {z: w for z, w in bdef["stats"].items() if stat_tiers.get(z, 2) in enabled_tiers}
        if kept:
            filtered[bk] = {k: v for k, v in bdef.items()}
            filtered[bk]["stats"] = kept
    return filtered

def bundle_tier_summary(bundle_stats, stat_tiers):
    counts = {}
    for z in bundle_stats: t = stat_tiers.get(z, 2); counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))

# Per-stat raw value formatting for the radar benchmark hover.
# Keys are z_col names; values are (label, format_fn).
_RADAR_RAW_FORMATTERS = {
    "yards_per_target_z": ("yds/target", lambda v: f"{v:.1f}"),
    "catch_rate_z": ("catch rate", lambda v: f"{v*100:.1f}%"),
    "first_down_rate_z": ("1D rate", lambda v: f"{v*100:.1f}%"),
    "yac_per_reception_z": ("YAC/rec", lambda v: f"{v:.1f}"),
    "yards_per_snap_z": ("yds/snap", lambda v: f"{v:.2f}"),
    "epa_per_target_z": ("EPA/tgt", lambda v: f"{v:+.2f}"),
    "yac_above_exp_z": ("YAC/exp", lambda v: f"{v:+.2f}"),
    "avg_separation_z": ("separation", lambda v: f"{v:.2f}yd"),
}

def _format_radar_raw(z_col, raw_value):
    """Format a raw stat value for the radar benchmark hover.
    Returns '' when no formatter or value is missing."""
    if raw_value is None or pd.isna(raw_value):
        return ""
    spec = _RADAR_RAW_FORMATTERS.get(z_col)
    if spec is None:
        return f"{raw_value:.2f}"
    label, fmt = spec
    return f"{label}: {fmt(raw_value)}"


def build_radar_figure(player, stat_labels, stat_methodology,
                        benchmark=None, benchmark_raw=None,
                        benchmark_label="Top 32 starter avg"):
    """Build the player's percentile radar.

    Args:
        benchmark: dict {z_col: mean_z} for the benchmark polygon
        benchmark_raw: dict {z_col: mean_raw_value} — appears in hover so
            users see the typical starter's actual numbers, not just percentile.
    """
    axes, values, descriptions, bench_values, bench_raw_strs = [], [], [], [], []
    for z_col in RADAR_STATS:
        if z_col not in player.index: continue
        z = player.get(z_col)
        if pd.isna(z): continue
        pct = zscore_to_percentile(z)
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        desc = stat_methodology.get(z_col, {}).get("what", "")
        axes.append(label); values.append(pct); descriptions.append(desc)
        if benchmark is not None:
            bz = benchmark.get(z_col)
            bench_values.append(zscore_to_percentile(bz) if bz is not None and pd.notna(bz) else None)
            raw_v = benchmark_raw.get(z_col) if benchmark_raw else None
            bench_raw_strs.append(_format_radar_raw(z_col, raw_v))
    if not axes: return None
    fig = go.Figure()

    # Player polygon FIRST so the benchmark sits on top — its diamond markers
    # need to be hoverable even where the player polygon overlaps them.
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=axes + [axes[0]],
        customdata=descriptions + [descriptions[0]],
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        name="This player",
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))

    if benchmark is not None and any(v is not None for v in bench_values):
        bv_clean = [v if v is not None else 50 for v in bench_values]
        # Use hovertext (full strings) instead of customdata templates — more
        # reliable than %{customdata[0]} indexing across plotly versions.
        bench_hover = []
        for ax, raw_str, pct in zip(axes, bench_raw_strs, bv_clean):
            extra = f"{raw_str} · " if raw_str else ""
            bench_hover.append(
                f"<b>{ax}</b><br>{benchmark_label}<br>{extra}{pct:.0f}th percentile"
            )
        bench_hover.append(bench_hover[0])  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=bv_clean + [bv_clean[0]],
            theta=axes + [axes[0]],
            mode="lines+markers",
            line=dict(color="rgba(102, 102, 102, 0.9)", width=2, dash="dot"),
            marker=dict(size=10, color="rgba(102, 102, 102, 0.95)",
                        symbol="diamond", line=dict(width=2, color="white")),
            # No fill so it doesn't intercept hover over the player polygon
            name=benchmark_label,
            hovertext=bench_hover,
            hoverinfo="text",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[25, 50, 75, 100],
                            ticktext=["25th", "50th", "75th", "100th"],
                            tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=(benchmark is not None),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=60, r=60, t=20, b=20),
        height=380, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Session state ─────────────────────────────────────────────
if "wr_loaded_algo" not in st.session_state: st.session_state.wr_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "wr_tiers_enabled" not in st.session_state: st.session_state.wr_tiers_enabled = [1, 2]

try: df = load_wr_data()
except FileNotFoundError: st.error(f"Couldn't find WR data at {DATA_PATH}."); st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="recent_team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} wide receivers found for {selected_season}.")
    st.stop()

meta = load_wr_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.wr_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP: apply_algo_weights(linked, BUNDLES); st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════
st.subheader(f"{team_name} wide receivers")
st.markdown("What makes a great WR? **You decide.** Use the sliders on the left to tell us what you value most, and the rankings update instantly.")
st.caption(f"{selected_season} regular season · Compared to all WRs league-wide with 100+ offensive snaps")

st.sidebar.header("What matters to you?")
st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize it, or all the way left to ignore it.")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if st.session_state.wr_loaded_algo:
    la = st.session_state.wr_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"): st.session_state.wr_loaded_algo = None

st.markdown("### Which stats should count?")
st.markdown("Check more boxes to include more types of stats. More boxes = more data, but less certainty.")
available_tiers = set(stat_tiers.values()) if stat_tiers else {1, 2}
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        has_stats = tier in available_tiers
        if has_stats:
            checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.wr_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"wr_tier_checkbox_{tier}")
            if checked: new_enabled.append(tier)
        else:
            st.markdown(f"<span style='opacity:0.35'>{tier_badge(tier)} {TIER_LABELS[tier]}</span>", unsafe_allow_html=True)
            st.caption("No stats available")
st.session_state.wr_tiers_enabled = new_enabled
if not new_enabled: st.warning("Check at least one box."); st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

advanced_mode = False
bundle_weights = {}; effective_weights = {}
if not active_bundles: st.info("No stat bundles available."); st.stop()

for bk, bundle in active_bundles.items():
    st.sidebar.markdown(f"**{bundle['label']}**")
    st.sidebar.markdown(f"{bundle['description']}")
    if f"wr_bundle_{bk}" not in st.session_state: st.session_state[f"wr_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
    bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"wr_bundle_{bk}", label_visibility="collapsed", help=bundle.get("why", ""))
    st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")
for bk in BUNDLES:
    if bk not in bundle_weights: bundle_weights[bk] = 0
effective_weights = compute_effective_weights(active_bundles, bundle_weights)

with st.sidebar.expander("Want more control? Adjust individual stats"):
    advanced_mode = st.checkbox("Enable individual stat control", value=False, key="wr_advanced_toggle")
    if advanced_mode:
        st.caption("Set the weight of each individual stat. This overrides the bundle sliders above.")
        effective_weights = {}
        all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
        for z_col in all_enabled_stats:
            label = stat_labels.get(z_col, z_col); meth = stat_methodology.get(z_col, {})
            help_text = meth.get("what", "")
            if meth.get("limits"): help_text += f"\n\nLimits: {meth['limits']}"
            w = st.slider(f"{tier_badge(stat_tiers.get(z_col, 2))} {label}", 0, 100, 50, 5, key=f"adv_wr_{z_col}", help=help_text if help_text else None)
            if w > 0: effective_weights[z_col] = w
        bundle_weights = {bk: 0 for bk in BUNDLES}

# Filter and score
min_snaps = st.slider("Minimum offensive snaps", 0, 1000, 100, step=25, help="Hide players who barely played.")
players = df[df["off_snaps"].fillna(0) >= min_snaps].copy()
if len(players) == 0: st.warning("No WRs match the current filter."); st.stop()
players = score_players(players, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All sliders are at zero — slide at least one to the right.")
players = players.sort_values("score", ascending=False).reset_index(drop=True)
players.index = players.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ranked = players.copy()

st.markdown("**How to read the score:** 0.00 = league average WR. The percentile shows where this player ranks among all qualifying WRs (100+ snaps).")

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_display_name", "—"); top_score = top["score"]
    top_pct = format_percentile(zscore_to_percentile(top_score))
    sign = "+" if top_score >= 0 else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong> &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({top_pct})</span></div>", unsafe_allow_html=True)
    warn = sample_size_warning(top.get("off_snaps", 0))
    if warn: st.warning(warn)

def _fmt_int(v): return f"{int(v)}" if pd.notna(v) else "—"
def _fmt_pct(v): return f"{v*100:.1f}%" if pd.notna(v) else "—"
def _fmt_signed(v, places=2): return f"{v:+.{places}f}" if pd.notna(v) else "—"

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_display_name"],
    "Snaps": ranked.get("off_snaps", pd.Series([0]*len(ranked))).apply(lambda s: f"{int(s)} ⚠️" if pd.notna(s) and s < 300 else _fmt_int(s)),
    "Rec": ranked.get("receptions", pd.Series([0]*len(ranked))).apply(_fmt_int),
    "Yds": ranked.get("rec_yards", pd.Series([0]*len(ranked))).apply(_fmt_int),
    "TDs": ranked.get("rec_tds", pd.Series([0]*len(ranked))).apply(_fmt_int),
    "Tgt%": ranked.get("target_share", pd.Series([float("nan")]*len(ranked))).apply(_fmt_pct),
    "EPA/tgt": ranked.get("epa_per_target", pd.Series([float("nan")]*len(ranked))).apply(lambda v: _fmt_signed(v, 2)),
    "YAC/exp": ranked.get("yac_above_exp", pd.Series([float("nan")]*len(ranked))).apply(lambda v: _fmt_signed(v, 1)),
    "Your score": ranked["score"].apply(format_score),
})
st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption("⚠️ = under 300 snaps — small sample, treat with caution. **Tgt%** = share of team's targets · **EPA/tgt** = Expected Points Added per target (modern efficiency stat) · **YAC/exp** = yards-after-catch above NGS expectation, per reception.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
selected = st.selectbox("Pick a wide receiver to see their full breakdown", options=ranked["player_display_name"].tolist(), index=0)
player = ranked[ranked["player_display_name"] == selected].iloc[0]
warn = sample_size_warning(player.get("off_snaps", 0))
if warn: st.warning(warn)

# ── Split-season panel: surface other stints if traded mid-season ──
all_wrs_full = load_wr_data()
season_stints = all_wrs_full[
    (all_wrs_full["player_id"] == player.get("player_id"))
    & (all_wrs_full["season_year"] == selected_season)
].copy()
if len(season_stints) > 1:
    n = len(season_stints)
    st.info(f"**Split season** — {selected} played for {n} teams in {selected_season}.")
    season_stints = season_stints.sort_values("off_snaps", ascending=False)
    split_rows = []
    for _, stint in season_stints.iterrows():
        team_disp = display_abbr(stint["recent_team"])
        is_current = stint["recent_team"] == player["recent_team"]
        split_rows.append({
            "Team": f"⮕ {team_disp}" if is_current else team_disp,
            "Games": _fmt_int(stint.get("games")),
            "Snaps": _fmt_int(stint.get("off_snaps")),
            "Rec": _fmt_int(stint.get("receptions")),
            "Yds": _fmt_int(stint.get("rec_yards")),
            "TDs": _fmt_int(stint.get("rec_tds")),
            "Tgt%": _fmt_pct(stint.get("target_share")),
            "EPA/tgt": _fmt_signed(stint.get("epa_per_target"), 2),
            "YAC/exp": _fmt_signed(stint.get("yac_above_exp"), 1),
        })

    # ── Season total row (weighted aggregates, not naive averages) ──
    def _safe_sum(col):
        return season_stints[col].fillna(0).sum() if col in season_stints.columns else float("nan")

    def _weighted_mean(value_col, weight_col):
        """Mean of value_col weighted by weight_col, ignoring stints with NaN value or zero weight."""
        if value_col not in season_stints.columns or weight_col not in season_stints.columns:
            return float("nan")
        v = season_stints[value_col]
        w = season_stints[weight_col]
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return float("nan")
        return (v[mask] * w[mask]).sum() / w[mask].sum()

    total_games = _safe_sum("games")
    total_snaps = _safe_sum("off_snaps")
    total_targets = _safe_sum("targets")
    total_receptions = _safe_sum("receptions")
    total_yards = _safe_sum("rec_yards")
    total_tds = _safe_sum("rec_tds")

    # Tgt%: implied team targets per stint = stint_targets / stint_target_share.
    # Season Tgt% = total player targets / sum of implied team targets.
    if "target_share" in season_stints.columns and "targets" in season_stints.columns:
        ts = season_stints["target_share"]
        tg = season_stints["targets"]
        mask = ts.notna() & tg.notna() & (ts > 0)
        if mask.any():
            implied_team_targets = (tg[mask] / ts[mask]).sum()
            season_tgt_share = total_targets / implied_team_targets if implied_team_targets > 0 else float("nan")
        else:
            season_tgt_share = float("nan")
    else:
        season_tgt_share = float("nan")

    season_epa_per_tgt = _weighted_mean("epa_per_target", "targets")
    season_yac_over_exp = _weighted_mean("yac_above_exp", "receptions")

    split_rows.append({
        "Team": f"**Total ({selected_season})**",
        "Games": _fmt_int(total_games),
        "Snaps": _fmt_int(total_snaps),
        "Rec": _fmt_int(total_receptions),
        "Yds": _fmt_int(total_yards),
        "TDs": _fmt_int(total_tds),
        "Tgt%": _fmt_pct(season_tgt_share),
        "EPA/tgt": _fmt_signed(season_epa_per_tgt, 2),
        "YAC/exp": _fmt_signed(season_yac_over_exp, 1),
    })

    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
    st.caption(f"⮕ marks the stint shown on this page ({display_abbr(player['recent_team'])}). Stints sorted by snaps; total uses weighted aggregates (Tgt% and rate stats are properly recomputed, not averaged).")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"### {selected}")
    st.caption(f"{int(player.get('off_snaps') or 0)} snaps · {int(player.get('targets') or 0)} targets · {int(player.get('rec_yards') or 0)} yards · {int(player.get('rec_tds') or 0)} TDs")
    player_score = player["score"]
    player_pct = format_percentile(zscore_to_percentile(player_score))
    sign = "+" if player_score >= 0 else ""
    st.markdown(f"**Your score: {sign}{player_score:.2f} ({player_pct})**")
    st.markdown("_This score is based on your slider settings. Change the sliders and this number changes._")
    st.markdown("---")
    st.markdown("**Where the score comes from**")
    st.markdown("Each row shows how much one skill contributed to the total, based on your slider weights.")
    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0: continue
            contribution = sum(player.get(z, 0) * (bw * internal / total_weight) for z, internal in bundle["stats"].items() if pd.notna(player.get(z)) and total_weight > 0)
            bundle_rows.append({"Skill": bundle["label"], "Your weight": f"{bw}", "Points added": f"{contribution:+.2f}"})
        if bundle_rows: st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        with st.expander("See the individual stats behind each skill"):
            stat_rows = []; shown = set()
            for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                raw_fmt = f"{raw:.2f}" if pd.notna(raw) else "—"
                stat_rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": raw_fmt, "Percentile": f"{int(pct)}th" if pct is not None else "—"})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0); contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            pct = zscore_to_percentile(z) if pd.notna(z) else None
            rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": f"{raw:.2f}" if pd.notna(raw) else "—", "Percentile": f"{int(pct)}th" if pct is not None else "—", "Weight": f"{w}", "Points added": f"{contrib:+.2f}"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**Percentile profile vs. all league WRs**")
    st.caption("Solid blue = this player. Dashed gray = top-32 starter average.")
    # Build the starter benchmark for this season
    season_pool = all_wrs_full[all_wrs_full["season_year"] == selected_season]
    top32 = season_pool.sort_values("off_snaps", ascending=False).head(32)
    radar_bench = {z: top32[z].mean() for z in RADAR_STATS if z in top32.columns and top32[z].notna().any()}
    # Raw means for the same stats (so hover shows e.g. "yds/target: 8.4")
    radar_bench_raw = {}
    for z in RADAR_STATS:
        raw_col = RAW_COL_MAP.get(z)
        if raw_col and raw_col in top32.columns and top32[raw_col].notna().any():
            radar_bench_raw[z] = top32[raw_col].mean()
    fig = build_radar_figure(player, stat_labels, stat_methodology,
                              benchmark=radar_bench, benchmark_raw=radar_bench_raw)
    if fig: st.plotly_chart(fig, use_container_width=True)

career_arc_section(
    player=player,
    league_parquet_path=DATA_PATH,
    z_score_cols=list(RAW_COL_MAP.keys()),
    stat_labels=stat_labels,
    id_col="player_id",
    name_col="player_display_name",
    position_label="wide receivers",
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) · NGS tracking data · regular season only · Z-scored against league-wide WRs with 100+ offensive snaps · Fan project, not affiliated with the NFL or any team.")
