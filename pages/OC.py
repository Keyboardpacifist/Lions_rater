"""
Lions OC Rater — Offensive Coordinators
Career default with 2024-only toggle. League-wide.
"""
import json
import re
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm


def extract_hc_architect_name(architect_status: str) -> str | None:
    """Pull the HC's name out of an architect_status cell.

    Examples:
      'HC = architect (Mike LaFleur runs offense)' -> 'Mike LaFleur'
      'HC = architect (Andy Reid calls plays in most years)' -> 'Andy Reid'
      'HC = architect (Schottenheimer is offensive HC)' -> 'Schottenheimer'
      'HC = architect (Todd Monken)' -> 'Todd Monken'
      'Coordinator runs offense (Ryans = defensive HC)' -> None
    """
    s = str(architect_status or "")
    if "HC = architect" not in s:
        return None
    m = re.search(r"\(([^)]+)\)", s)
    if not m:
        return None
    inner = m.group(1).strip()
    inner = re.split(r"\s+(?:runs|calls|is)\b", inner, maxsplit=1)[0].strip()
    return inner or None


def find_oc_in_pool(name: str, pool_names: list[str]) -> str | None:
    """Match a curated OC/HC name against the rater data pool.
    Tries exact match first, then case-insensitive, then last-name
    substring (handles 'McCarthy' -> 'Mike McCarthy')."""
    if not name or not pool_names:
        return None
    target = name.strip()
    for n in pool_names:
        if str(n).strip() == target:
            return n
    target_l = target.lower()
    for n in pool_names:
        if str(n).strip().lower() == target_l:
            return n
    if " " not in target:
        for n in pool_names:
            if target_l in str(n).lower().split():
                return n
    return None
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, render_player_card, score_players, team_theme
from lib_top_nav import render_home_button

st.set_page_config(page_title="NFL OC Rater", page_icon="🦁", layout="wide", initial_sidebar_state="expanded")
inject_css()

render_home_button()  # ← back to landing
POSITION_GROUP = "oc"
PAGE_URL = "https://lions-rater.streamlit.app/OC"
DATA_PATH_CAREER = Path(__file__).resolve().parent.parent / "data" / "master_ocs_with_z.parquet"
DATA_PATH_SEASON = Path(__file__).resolve().parent.parent / "data" / "master_ocs_2025_with_z.parquet"
DATA_PATH_PER_SEASON = Path(__file__).resolve().parent.parent / "data" / "master_ocs_seasons_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_stat_metadata.json"
OC_GAS_CAREER_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_gas_career.parquet"
OC_GAS_SEASON_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_gas_2025.parquet"


@st.cache_data
def load_oc_per_season():
    if not DATA_PATH_PER_SEASON.exists():
        return pd.DataFrame()
    return pd.read_parquet(DATA_PATH_PER_SEASON)

GAS_COLS = ["gas_score", "gas_label", "gas_confidence",
            "gas_efficiency_grade", "gas_explosiveness_grade",
            "gas_situational_grade", "gas_clutch_grade"]


def _merge_oc_gas(df: pd.DataFrame, gas_path: Path) -> pd.DataFrame:
    """Merge GAS columns into the OC dataframe by coordinator name."""
    if not gas_path.exists():
        return df
    gas = pd.read_parquet(gas_path)
    keep = ["coordinator"] + [c for c in GAS_COLS if c in gas.columns]
    return df.merge(gas[keep], on="coordinator", how="left")


@st.cache_data
def load_oc_career():
    df = pl.read_parquet(DATA_PATH_CAREER).to_pandas()
    return _merge_oc_gas(df, OC_GAS_CAREER_PATH)


@st.cache_data
def load_oc_2024():
    """Loader name kept for compatibility — actually loads 2025 file
    after the source-of-truth rebuild."""
    df = pl.read_parquet(DATA_PATH_SEASON).to_pandas()
    return _merge_oc_gas(df, OC_GAS_SEASON_PATH)
@st.cache_data
def load_oc_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "epa_per_play_z": "epa_per_play", "pass_epa_per_play_z": "pass_epa_per_play",
    "rush_epa_per_play_z": "rush_epa_per_play", "success_rate_z": "success_rate",
    "explosive_pass_rate_z": "explosive_pass_rate", "explosive_rush_rate_z": "explosive_rush_rate",
    "third_down_rate_z": "third_down_rate", "red_zone_td_rate_z": "red_zone_td_rate",
    "win_pct_z": "win_pct",
    "off_cap_pct_z": "off_cap_pct", "off_draft_capital_z": "off_draft_capital",
}

BUNDLES = {
    "efficiency": {"label": "📊 Offensive efficiency", "description": "Overall EPA per play, passing and rushing EPA. The core measure of offensive production.", "why": "Think raw offensive output is what defines a great OC? Crank this up.", "stats": {"epa_per_play_z": 0.40, "pass_epa_per_play_z": 0.30, "rush_epa_per_play_z": 0.30}},
    "execution": {"label": "🎯 Situational execution", "description": "Third down conversions and red zone TD rate. Measures playcalling in critical moments.", "why": "Value coordinators who convert when it matters most? Slide right.", "stats": {"third_down_rate_z": 0.50, "red_zone_td_rate_z": 0.50}},
    "explosiveness": {"label": "💥 Big play ability", "description": "Explosive pass plays (20+ yds) and explosive rush plays (10+ yds). Creates game-breaking moments.", "why": "Want offenses that hit home runs, not just move the chains? Slide right.", "stats": {"explosive_pass_rate_z": 0.55, "explosive_rush_rate_z": 0.45}},
    "winning": {"label": "🏆 Winning", "description": "Team win percentage during coordinator tenure. The ultimate measure, but the least isolatable.", "why": "Think wins are all that matter, regardless of how? Slide right.", "stats": {"win_pct_z": 1.00}},
}
DEFAULT_BUNDLE_WEIGHTS = {"efficiency": 60, "execution": 50, "explosiveness": 40, "winning": 30}

RADAR_STATS = ["epa_per_play_z", "pass_epa_per_play_z", "rush_epa_per_play_z", "success_rate_z", "explosive_pass_rate_z", "explosive_rush_rate_z", "third_down_rate_z", "red_zone_td_rate_z", "win_pct_z"]
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"epa_per_play_z": "Off EPA", "pass_epa_per_play_z": "Pass EPA", "rush_epa_per_play_z": "Rush EPA", "success_rate_z": "Success rate", "explosive_pass_rate_z": "Explosive pass", "explosive_rush_rate_z": "Explosive rush", "third_down_rate_z": "3rd down", "red_zone_td_rate_z": "Red zone", "win_pct_z": "Win %"}

def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)

def format_percentile(pct):
    if pct is None or pd.isna(pct): return "—"
    if pct >= 99: return "top 1%"
    if pct >= 50: return f"top {100 - int(pct)}%"
    return f"bottom {int(pct)}%"

def build_radar_figure(player, stat_labels, stat_methodology):
    axes, values, descriptions = [], [], []
    for z_col in RADAR_STATS:
        if z_col not in player.index: continue
        z = player.get(z_col)
        if pd.isna(z): continue
        pct = zscore_to_percentile(z)
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        desc = stat_methodology.get(z_col, {}).get("what", "")
        axes.append(label); values.append(pct); descriptions.append(desc)
    if not axes: return None
    fig = go.Figure()
    # League average reference (50th percentile) — dotted black ring with markers
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(axes) + [50],
        theta=axes + [axes[0]],
        mode="lines+markers",
        line=dict(color="rgba(0,0,0,0.7)", dash="dot", width=1.5),
        marker=dict(size=6, color="rgba(0,0,0,0.85)", symbol="circle"),
        name="League avg",
        hovertemplate="<b>%{theta}</b><br>League avg = 50th pct<extra></extra>",
        showlegend=True,
    ))
    fig.add_trace(go.Scatterpolar(
        r=values+[values[0]], theta=axes+[axes[0]],
        customdata=descriptions+[descriptions[0]],
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=7, color="rgba(31, 119, 180, 1)"),
        name="OC",
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                             tickvals=[25, 50, 75, 100],
                             ticktext=["25", "50", "75", "100"],
                             tickfont=dict(size=9, color="#888"),
                             gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=20, b=40),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

TIER_LABELS = {1: "Counting stats", 2: "Rate stats", 3: "Modeled stats", 4: "Estimated stats"}
TIER_DESCRIPTIONS = {1: "Raw totals — sacks, tackles, yards, touchdowns.", 2: "Per-game and per-snap averages that adjust for playing time.", 3: "Stats adjusted for expected performance based on a model.", 4: "Inferred from patterns."}
def tier_badge(tier): return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")
def filter_bundles_by_tier(bundles, stat_tiers, enabled_tiers):
    filtered = {}
    for bk, bdef in bundles.items():
        kept = {z: w for z, w in bdef["stats"].items() if stat_tiers.get(z, 2) in enabled_tiers}
        if kept: filtered[bk] = {"label": bdef["label"], "description": bdef["description"], "stats": kept}
    return filtered
def bundle_tier_summary(bundle_stats, stat_tiers):
    counts = {}
    for z in bundle_stats: t = stat_tiers.get(z, 2); counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))
def score_label(score):
    if pd.isna(score): return "—"
    if score >= 1.0: return "well above group"
    if score >= 0.4: return "above group"
    if score >= -0.4: return "about average"
    if score >= -1.0: return "below group"
    return "well below group"
def format_score(score):
    if pd.isna(score): return "—"
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} ({score_label(score)})"
def sample_size_badge(seasons):
    if pd.isna(seasons): return ""
    if seasons < 2: return "🔴"
    if seasons < 4: return "🟡"
    return ""

SCORE_EXPLAINER = """
**What this number means.** Weighted average of z-scores — 0 is league-average OC, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**Talent caveat.** These stats measure the offense's output, not the coordinator in isolation. A great QB inflates OC numbers. A bad offensive line deflates them. The scores reflect the whole unit's performance under this coordinator's playcalling.

**Career vs 2024.** Career mode averages across all seasons (2016-2024). 2024-only shows single-season performance. Career is more stable; 2024 is more current.
"""

if "oc_loaded_algo" not in st.session_state: st.session_state.oc_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "oc_tiers_enabled" not in st.session_state: st.session_state.oc_tiers_enabled = [1, 2]

# ─────────────────────────────────────────────────────────────
# Curation-driven OC profile (Tier 3 hand-curated)
# ─────────────────────────────────────────────────────────────
CURATION_PATH = Path(__file__).resolve().parent.parent / "data" / "scheme" / "curation" / "oc_curation.csv"
OC_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "scheme" / "oc_career_profile.parquet"
OC_PHILOSOPHY_PATH = Path(__file__).resolve().parent.parent / "data" / "scheme" / "oc_career_philosophy.parquet"
OC_FULCRUM_PATH = Path(__file__).resolve().parent.parent / "data" / "scheme" / "oc_fulcrum_profile.parquet"
OC_PLAYER_LIFT_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_player_lift.parquet"
OC_GAMESCRIPT_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_gamescript.parquet"
OC_QB_ARCHETYPE_LIFT_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_qb_archetype_lift.parquet"


@st.cache_data
def load_oc_player_lift():
    if not OC_PLAYER_LIFT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_PLAYER_LIFT_PATH)


@st.cache_data
def load_oc_gamescript():
    if not OC_GAMESCRIPT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_GAMESCRIPT_PATH)


@st.cache_data
def load_oc_qb_archetype_lift():
    if not OC_QB_ARCHETYPE_LIFT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_QB_ARCHETYPE_LIFT_PATH)


OC_PLAY_DISTRIBUTION_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_play_distribution.parquet"


@st.cache_data
def load_oc_play_distribution():
    if not OC_PLAY_DISTRIBUTION_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_PLAY_DISTRIBUTION_PATH)

@st.cache_data
def load_oc_curation():
    if not CURATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CURATION_PATH).fillna("")

@st.cache_data
def load_oc_career_profile():
    if not OC_PROFILE_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_PROFILE_PATH)

@st.cache_data
def load_oc_career_philosophy():
    if not OC_PHILOSOPHY_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_PHILOSOPHY_PATH)

@st.cache_data
def load_oc_fulcrum_profile():
    if not OC_FULCRUM_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(OC_FULCRUM_PATH)


# Leverage-definition labels for the clutch panel dropdown
LEVERAGE_DEF_LABELS = {
    "wp_volatility":  "WP volatility — plays where |WPA| ≥ 5%",
    "high_stakes":    "High-stakes situations — 4Q close + RZ + 3rd & med+ + 2-min",
    "epa_x_leverage": "EPA × leverage — every play weighted by |WPA|",
    "hybrid":         "Hybrid — |WPA| × situation multiplier",
    "all_plays":      "All plays — baseline (no leverage filter)",
}
LEVERAGE_DEF_BLURB = {
    "wp_volatility":  "Only the plays that actually swung the game by 5%+ WP. Strict, low sample.",
    "high_stakes":    "Rule-based critical situations even if no single play swung WP. Larger sample.",
    "epa_x_leverage": "Every play counts, weighted by how much it mattered. No cliff, all signal.",
    "hybrid":         "|WPA| with a 1.5× boost for situational plays (RZ / 3rd-and-medium+ / 4Q close / 2-min).",
    "all_plays":      "Sanity-check baseline — equally-weighted overall mean, no leverage filter.",
}


# Friendlier labels for D&D buckets (chart x-axis)
_DND_LABEL = {
    "1st_10": "1st & 10",   "1st_short": "1st & <7",
    "2nd_short": "2nd & ≤3", "2nd_med": "2nd & 4-7", "2nd_long": "2nd & 8+",
    "3rd_short": "3rd & ≤3", "3rd_med": "3rd & 4-7", "3rd_long": "3rd & 8+",
    "4th": "4th down",
}
_DND_ORDER = ["1st_10", "1st_short", "2nd_short", "2nd_med", "2nd_long",
              "3rd_short", "3rd_med", "3rd_long", "4th"]
_RUN_GAP_LABEL = {"end": "End (outside)", "tackle": "Tackle", "guard": "Guard (interior)"}
_RUN_GAP_ORDER = ["end", "tackle", "guard"]


def _bar_oc_vs_league(rows, title, label_map=None, order=None,
                      y_pct=True, height=300):
    """Plotly grouped bar: OC value (colored) + league avg (dashed line),
    with z-score badge above each OC bar.
    `rows` = list of dicts with category, value, league_value, value_z_avg.
    """
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if order:
        order_idx = {c: i for i, c in enumerate(order)}
        df = df.sort_values("category", key=lambda s: s.map(lambda c: order_idx.get(c, 99)))
    df["x_label"] = df["category"].map(lambda c: (label_map or {}).get(c, c))
    df["z_text"] = df["value_z_avg"].apply(
        lambda z: f"{'+' if z >= 0 else ''}{z:.1f}σ" if pd.notna(z) else ""
    )
    bar_colors = df["value_z_avg"].apply(
        lambda z: "#1f7a3a" if pd.notna(z) and z >= 1.0
        else "#52a370" if pd.notna(z) and z >= 0.3
        else "#aa3a2a" if pd.notna(z) and z <= -1.0
        else "#cc6651" if pd.notna(z) and z <= -0.3
        else "#0076B6"
    ).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["x_label"], y=df["value"], name="OC",
        marker=dict(color=bar_colors),
        text=df["z_text"], textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x}<br>OC: %{y:.1%}" + ("<br>" if y_pct else "<br>") + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["x_label"], y=df["league_value"], name="League avg",
        mode="lines+markers",
        line=dict(color="#888", dash="dot", width=2),
        marker=dict(size=7, color="#888"),
        hovertemplate="%{x}<br>League: %{y:.1%}<extra></extra>" if y_pct
                      else "%{x}<br>League: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=10, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        yaxis=dict(tickformat=".0%" if y_pct else None),
        xaxis=dict(tickfont=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_oc_coord_detail(curation_row: pd.Series) -> None:
    """Coordinator detail card embedded inside the team profile.
    Year selector (Career / each tenured season) drives the radar +
    accompanying year-by-year stat table."""
    oc_name = str(curation_row.get("oc_name", "") or "").strip()
    arch_name = extract_hc_architect_name(curation_row.get("architect_status", ""))

    career = load_oc_career()
    per_season = load_oc_per_season()

    # Resolve which name to look up (architect first, then OC of record)
    candidates = [n for n in [arch_name, oc_name] if n]
    pool_career = career.get("coordinator", pd.Series()).dropna().astype(str).tolist()
    matched = None
    for n in candidates:
        m = find_oc_in_pool(n, pool_career)
        if m: matched = m; break

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Coordinator detail")
    st.markdown(
        "**What this is:** A snapshot of how this OC's offense has performed across "
        "their career — or any single season. The radar chart shows nine key stats "
        "as percentiles vs the league of OCs.\n\n"
        "**How to read the radar:** The **dotted black ring** is league-average "
        "(50th percentile). Anything **outside** the ring is above average; **inside** "
        "is below. The blue polygon is this OC. Bigger and more outside the ring = "
        "better. Hover any point for the exact percentile and what we're measuring.\n\n"
        "**Why we believe this matters:** Most stat sheets dump numbers and let you "
        "do the math. The radar lets you see strengths and weaknesses at a glance — "
        "so you can spot, say, a great scoring offense with a brutal turnover problem."
    )

    if not matched:
        names_tried = " or ".join(f"_{n}_" for n in candidates) or "this play-caller"
        st.info(
            f"No play-calling history in our data for {names_tried}. "
            "Add their team-seasons to `data/scheme/curation/oc_team_seasons.csv` "
            "and re-run `rebuild_oc_master.py` to populate."
        )
        return

    # Year selector — Career + each season the play-caller has data for
    year_options = ["Career"]
    if not per_season.empty:
        years = sorted(per_season[per_season["coordinator"] == matched]["season"].unique())
        year_options += [str(int(y)) for y in years]
    if len(year_options) <= 1:
        year_options = ["Career"]

    sel_year = st.radio(
        "View:",
        options=year_options,
        horizontal=True,
        key=f"coord_detail_year_{matched}",
    )

    # Pick the row to display
    if sel_year == "Career":
        sub = career[career["coordinator"] == matched]
        if sub.empty:
            st.warning(f"No career row for {matched}."); return
        player = sub.iloc[0].copy()
        scope_label = f"Career ({int(player.get('first_season', 0))}-{int(player.get('last_season', 0))})"
        team_label = str(player.get("teams", "—"))
    else:
        yr = int(sel_year)
        # If the play-caller had multiple teams in a season (rare), aggregate
        ps_sub = per_season[(per_season["coordinator"] == matched)
                            & (per_season["season"] == yr)]
        if ps_sub.empty:
            st.info(f"No data for {matched} in {yr}."); return
        # Multi-team season would be very rare; take the first row
        player = ps_sub.iloc[0].copy()
        scope_label = f"{yr} season"
        team_label = str(player.get("team", "—"))

    # GAS badge for context
    gas_career = (career[career["coordinator"] == matched].iloc[0]
                  if (career["coordinator"] == matched).any() else None)
    gas_score = gas_career.get("gas_score") if gas_career is not None else None
    gas_label = gas_career.get("gas_label") if gas_career is not None else None

    # Header strip
    hdr_left = f"**{matched}** · {scope_label}"
    if team_label and team_label != "—":
        hdr_left += f" · {team_label}"
    hdr_right = ""
    if gas_score is not None and pd.notna(gas_score):
        hdr_right = f"GAS **{gas_score:.1f}** · {gas_label}"
    cc1, cc2 = st.columns([3, 1])
    with cc1:
        st.markdown(hdr_left)
    with cc2:
        if hdr_right:
            st.markdown(f"<div style='text-align:right'>{hdr_right}</div>",
                        unsafe_allow_html=True)

    # Radar + year-by-year table side by side
    rcol, tcol = st.columns([1, 1])
    with rcol:
        st.markdown("**Profile** (percentiles vs league of OCs · dotted black = league avg)")
        meta = load_oc_metadata()
        stat_labels = meta.get("stat_labels", {})
        stat_methodology = meta.get("stat_methodology", {})
        fig = build_radar_figure(player, stat_labels, stat_methodology)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("_No radar data available for this view._")

    with tcol:
        st.markdown("**Year-by-year** (sortable)")
        if per_season.empty:
            st.caption("_No per-season data loaded._")
        else:
            yr_rows = per_season[per_season["coordinator"] == matched].copy()
            if yr_rows.empty:
                st.caption("_No tenure data._")
            else:
                yr_rows = yr_rows.sort_values("season", ascending=False)
                table = pd.DataFrame({
                    "Year": yr_rows["season"].astype(int),
                    "Team": yr_rows.get("team", yr_rows.get("posteam", "—")),
                    "Plays": yr_rows.get("total_plays", 0).astype(int),
                    "EPA/play": yr_rows.get("epa_per_play", float("nan")).round(3),
                    "Succ%": (yr_rows.get("success_rate", float("nan")) * 100).round(1),
                    "3rd%": (yr_rows.get("third_down_rate", float("nan")) * 100).round(1),
                    "RZ TD%": (yr_rows.get("red_zone_td_rate", float("nan")) * 100).round(1),
                    "Win%": (yr_rows.get("win_pct", float("nan")) * 100).round(1),
                })
                st.dataframe(
                    table, use_container_width=True, hide_index=True,
                    column_config={
                        "Year": st.column_config.NumberColumn(format="%d"),
                        "Plays": st.column_config.NumberColumn(format="%d"),
                        "Succ%": st.column_config.NumberColumn(format="%.1f%%"),
                        "3rd%": st.column_config.NumberColumn(format="%.1f%%"),
                        "RZ TD%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Win%": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                    height=min(40 + 35 * len(table), 350),
                )

    # Per-stat raw values for the selected scope (helps explain the radar)
    with st.expander("🔬 Underlying stats (selected view)"):
        for z_col in RADAR_STATS:
            raw_col = RAW_COL_MAP.get(z_col)
            if raw_col is None or raw_col not in player.index: continue
            raw = player.get(raw_col)
            z = player.get(z_col)
            label = RADAR_LABEL_OVERRIDES.get(z_col, raw_col)
            if raw_col in ("success_rate", "third_down_rate", "red_zone_td_rate",
                           "win_pct", "explosive_pass_rate", "explosive_rush_rate"):
                raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
            else:
                raw_fmt = f"{raw:+.4f}" if pd.notna(raw) else "—"
            z_fmt = f"{z:+.2f}" if pd.notna(z) else "—"
            st.markdown(f"- **{label}**: {raw_fmt} (z = {z_fmt})")


def _z_color(z):
    """Color a z-score badge: green good, red bad, neutral grey."""
    if pd.isna(z): return "#888"
    if z >= 1.0: return "#1f7a3a"
    if z >= 0.3: return "#52a370"
    if z >= -0.3: return "#888"
    if z >= -1.0: return "#cc6651"
    return "#aa3a2a"


def _z_text(z):
    if pd.isna(z): return "—"
    sign = "+" if z >= 0 else ""
    return f"{sign}{z:.2f}σ"


def _render_oc_clutch_panel(oc_name: str) -> None:
    """Phase 3: per-OC clutch / fulcrum profile.

    Three lenses, all under a leverage-definition dropdown:
    1. RAW — actual fulcrum metric (with z vs league of OCs)
    2. ROSTER-ADJUSTED — residual z after regressing on roster proxies
    3. ELEVATION — fulcrum metric minus same OC's non-fulcrum baseline
       (partial roster cancellation; the headline "is this guy clutch?" cut)
    """
    fulcrum = load_oc_fulcrum_profile()
    if fulcrum.empty:
        return

    sub_oc = fulcrum[fulcrum["oc_name"] == oc_name]
    if sub_oc.empty:
        return  # already handled by scheme panel empty state

    st.markdown('<div class="section-divider" style="margin-top:8px"></div>',
                unsafe_allow_html=True)
    st.markdown("### 🎯 Clutch profile — fulcrum performance")
    st.markdown(
        "**What this is:** How does this OC's offense perform in **the moments that "
        "decide games** — vs how it performs in low-leverage moments? Pick a "
        "leverage definition below to set what \"clutch\" means.\n\n"
        "**How to read the three columns:**\n"
        "- **Raw fulcrum** = actual EPA / success in clutch plays (color-coded vs the league of OCs)\n"
        "- **Roster-adjusted** = same number, after subtracting how much credit the roster deserves\n"
        "- **🎯 Elevation index (the headline)** = `clutch − non-clutch` for the same OC. "
        "Tells you whether they **step up or fade** when the game's on the line. "
        "Same roster on both sides cancels most confounds — this is the cleanest \"is this guy clutch?\" cut.\n\n"
        "**Why we believe this matters:** Average-output OCs who light it up in clutch "
        "deserve credit. Star OCs who fold in big moments deserve scrutiny. The raw stats "
        "lump both together; the leverage filter separates them."
    )

    # Leverage definition dropdown
    available_defs = [d for d in LEVERAGE_DEF_LABELS.keys()
                      if d in sub_oc["leverage_def"].unique()]
    default_idx = (available_defs.index("wp_volatility")
                   if "wp_volatility" in available_defs else 0)
    leverage_def = st.selectbox(
        "Leverage definition",
        options=available_defs,
        index=default_idx,
        format_func=lambda d: LEVERAGE_DEF_LABELS.get(d, d),
        key=f"clutch_leverage_def_{oc_name}",
    )
    st.caption(f"_{LEVERAGE_DEF_BLURB.get(leverage_def, '')}_")

    sub = sub_oc[sub_oc["leverage_def"] == leverage_def]
    if sub.empty:
        st.info("No fulcrum data for this definition.")
        return

    # Two metrics: epa_per_play, success_rate
    is_baseline = (leverage_def == "all_plays")

    for metric_key, metric_label, value_fmt in [
        ("epa_per_play", "EPA / play", "{:+.3f}"),
        ("success_rate", "Success rate", "{:.1%}"),
    ]:
        row = sub[sub["metric"] == metric_key]
        if row.empty:
            continue
        r = row.iloc[0]
        st.markdown(f"#### {metric_label}")

        c1, c2, c3 = st.columns(3)
        with c1:
            # RAW
            st.markdown(
                f"<div style='font-size:0.85rem;color:#666;text-transform:uppercase;"
                f"letter-spacing:1px;'>Raw fulcrum</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:1.6rem;font-weight:700;color:{_z_color(r['fulcrum_z'])};'>"
                f"{value_fmt.format(r['fulcrum_value'])}</div>",
                unsafe_allow_html=True)
            st.caption(
                f"vs league of OCs: **{_z_text(r['fulcrum_z'])}** · "
                f"n = {int(r['n_fulcrum']):,} fulcrum plays"
            )
        with c2:
            # ROSTER-ADJUSTED
            adj = r.get("fulcrum_adj_z")
            st.markdown(
                f"<div style='font-size:0.85rem;color:#666;text-transform:uppercase;"
                f"letter-spacing:1px;'>Roster-adjusted</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:1.6rem;font-weight:700;color:{_z_color(adj)};'>"
                f"{_z_text(adj)}</div>",
                unsafe_allow_html=True)
            st.caption(
                "Residual after regressing on team-rating + cap-allocation proxies. "
                "OC value-add over expected."
            )
        with c3:
            # ELEVATION (headline metric — the "is this guy clutch?" cut)
            elev = r.get("elevation")
            elev_z = r.get("elevation_z")
            st.markdown(
                f"<div style='font-size:0.85rem;color:#666;text-transform:uppercase;"
                f"letter-spacing:1px;'>🎯 Elevation index</div>", unsafe_allow_html=True)
            if is_baseline or pd.isna(elev):
                st.markdown(
                    f"<div style='font-size:1.6rem;font-weight:700;color:#888;'>—</div>",
                    unsafe_allow_html=True)
                st.caption("_Not applicable for the all-plays baseline._" if is_baseline
                           else "_No non-fulcrum baseline available._")
            else:
                arrow = "↑" if elev >= 0 else "↓"
                verb = "steps up" if elev_z >= 0.5 else ("fades" if elev_z <= -0.5 else "steady")
                st.markdown(
                    f"<div style='font-size:1.6rem;font-weight:700;color:{_z_color(elev_z)};'>"
                    f"{arrow} {value_fmt.format(elev)}</div>",
                    unsafe_allow_html=True)
                st.caption(
                    f"Fulcrum {value_fmt.format(r['fulcrum_value'])} − "
                    f"non-fulcrum {value_fmt.format(r['non_fulcrum_value'])} · "
                    f"{_z_text(elev_z)} vs league · **{verb}** in clutch"
                )

    st.caption(
        "Three lenses, three questions: **Raw** = output in clutch moments. "
        "**Roster-adjusted** = output net of roster quality. "
        "**Elevation** = does this OC step up or fade vs their own baseline (roster confound largely cancels)."
    )


def _render_oc_scheme_panels(curation_row: pd.Series) -> None:
    """Render data fingerprint panels for the OC (or HC architect)
    referenced by this curation row. Falls back to empty state when the
    play-caller has no historical OC tenure in our data."""
    profile = load_oc_career_profile()
    philosophy = load_oc_career_philosophy()

    if profile.empty:
        return  # silent — Phase 1 fingerprint hasn't been built yet

    oc_name = str(curation_row.get("oc_name", "") or "").strip()
    arch_name = extract_hc_architect_name(curation_row.get("architect_status", ""))

    # Prefer the architect (actual play-caller) when applicable; fall back to OC of record.
    # Use fuzzy match so curation last-names like "Schottenheimer" hit the
    # full "Brian Schottenheimer" in the profile pool.
    candidates = [n for n in [arch_name, oc_name] if n]
    pool_names = profile["oc_name"].dropna().astype(str).unique().tolist()
    matched = None
    for n in candidates:
        m = find_oc_in_pool(n, pool_names)
        if m:
            matched = m
            break

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Scheme fingerprint")
    st.markdown(
        "**What this is:** How this OC actually calls plays. Eight different views — "
        "down-and-distance pass rate, run gap split, shotgun usage, coverage-reactive "
        "passing, tempo, pressure faced, field position, and philosophy fit.\n\n"
        "**How to read every chart below:** Each colored bar = **this OC**. The "
        "dashed grey line = **league average**. Above the line = they do it MORE than "
        "league. Below = LESS. The **σ badge** above each bar shows how unusual the "
        "number is. Anything past **±1σ** is meaningfully different from average; "
        "past **±2σ** is extreme.\n\n"
        "**Why we believe this matters:** The fingerprint reveals identity. A "
        "tackle-heavy run gap split + low shotgun rate = classic Shanahan-tree zone "
        "blocking. A high guard-gap share + heavy formations = power-run team. "
        "Two OCs can have the same EPA but get there completely differently."
    )

    if not matched:
        names_tried = " or ".join(f"_{n}_" for n in candidates) or "the play-caller"
        st.info(
            f"No historical fingerprint data for {names_tried} yet. "
            "Either too new an OC tenure, or not in our 15-OC seed mapping. "
            "Will fill in once their team-seasons are added to "
            "`data/scheme/curation/oc_team_seasons.csv`."
        )
        return

    sub = profile[profile["oc_name"] == matched]
    n_seasons = int(sub["n_seasons"].max()) if not sub.empty else 0
    label = matched if matched == oc_name else f"{matched} (HC = architect)"
    st.caption(
        f"Aggregated across **{n_seasons}** OC season(s) for **{label}**. "
        "Bars show OC value vs league average (dashed grey). "
        "Badge above each bar = z-score vs league of OCs."
    )

    # ── 1. Pass rate by D&D ───────────────────────────────────
    dnd = sub[sub["dimension"] == "dnd_pass_rate"]
    if not dnd.empty:
        fig = _bar_oc_vs_league(
            dnd[["category", "value", "league_value", "value_z_avg"]].to_dict("records"),
            "Pass rate by down & distance",
            label_map=_DND_LABEL, order=_DND_ORDER,
        )
        if fig: st.plotly_chart(fig, use_container_width=True)

    # ── 2. Run gap split | 3. Shotgun rate ─────────────────────
    c1, c2 = st.columns(2)
    with c1:
        rg = sub[sub["dimension"] == "run_gap_share"]
        if not rg.empty:
            fig = _bar_oc_vs_league(
                rg[["category", "value", "league_value", "value_z_avg"]].to_dict("records"),
                "Run gap distribution (zone-vs-power proxy)",
                label_map=_RUN_GAP_LABEL, order=_RUN_GAP_ORDER, height=280,
            )
            if fig: st.plotly_chart(fig, use_container_width=True)
    with c2:
        sg = sub[sub["dimension"] == "shotgun_rate"]
        if not sg.empty:
            fig = _bar_oc_vs_league(
                sg[["category", "value", "league_value", "value_z_avg"]].to_dict("records"),
                "Shotgun rate by D&D (under-center signal)",
                label_map=_DND_LABEL, order=_DND_ORDER, height=280,
            )
            if fig: st.plotly_chart(fig, use_container_width=True)

    # ── 4. Coverage-reactive | 5. Tempo + pressure ──────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Coverage-reactive attack**")
        st.caption(
            "**What this is:** how this OC throws differently when the defense "
            "shows **MAN** coverage (one defender on every receiver) vs **ZONE** "
            "(each defender covers an area). **Pass rate** = % of plays they "
            "passed (vs run) when seeing that look. **Avg AY** = average air "
            "yards = how far downfield the throw was. **What we believe it shows:** "
            "scheme philosophy. OCs who throw deep vs man are attacking single-high "
            "with one-on-ones; OCs who dump off vs zone are taking the underneath "
            "concession; pass-happy OCs vs both ignore the look entirely. "
            "_2018+ data only._"
        )
        cov_pr = sub[sub["dimension"] == "vs_coverage_pass_rate"]
        cov_ay = sub[sub["dimension"] == "vs_coverage_avg_ay"]
        for cat, cat_label in [("vs_man", "vs MAN"), ("vs_zone", "vs ZONE")]:
            mc1, mc2 = st.columns(2)
            with mc1:
                pr = cov_pr[cov_pr["category"] == cat]
                if not pr.empty:
                    r = pr.iloc[0]
                    delta_pp = (r["value"] - r["league_value"]) * 100
                    st.metric(
                        f"{cat_label} · pass rate",
                        f"{r['value']:.1%}",
                        delta=f"{delta_pp:+.1f}pp · {r['value_z_avg']:+.1f}σ",
                    )
            with mc2:
                ay = cov_ay[cov_ay["category"] == cat]
                if not ay.empty:
                    r = ay.iloc[0]
                    st.metric(
                        f"{cat_label} · avg AY",
                        f"{r['value']:.1f} yds",
                        delta=f"{r['value']-r['league_value']:+.1f} · {r['value_z_avg']:+.1f}σ",
                    )

    with c2:
        st.markdown("**Tempo & pressure**")
        tempo = sub[sub["dimension"] == "tempo"]
        press = sub[sub["dimension"] == "pressure_faced"]
        rows = [
            ("Overall no-huddle", tempo[tempo["category"] == "overall_no_huddle"]),
            ("2-min no-huddle", tempo[tempo["category"] == "2min_no_huddle"]),
            ("Pressure faced (5+ rushers)", press[press["category"] == "rate_5plus"]),
        ]
        for label, rr in rows:
            if rr.empty: continue
            r = rr.iloc[0]
            delta_pp = (r["value"] - r["league_value"]) * 100
            st.metric(
                label,
                f"{r['value']:.1%}",
                delta=f"{delta_pp:+.1f}pp · {r['value_z_avg']:+.1f}σ",
            )

    # ── 6. Field-position pass rate (RZ / GL / backed-up) ──────
    fpr = sub[sub["dimension"] == "field_pass_rate"]
    if not fpr.empty:
        fpr_label = {"rz": "Red zone (≤20)", "gl": "Goal line (≤5)", "backed_up": "Backed up (own ≤10)"}
        fpr_order = ["backed_up", "rz", "gl"]
        fig = _bar_oc_vs_league(
            fpr[["category", "value", "league_value", "value_z_avg"]].to_dict("records"),
            "Pass rate by field position",
            label_map=fpr_label, order=fpr_order, height=280,
        )
        if fig: st.plotly_chart(fig, use_container_width=True)

    # ── 7. Clutch / fulcrum panel ──────────────────────────────
    _render_oc_clutch_panel(matched)

    # ── 8. Philosophy fit ──────────────────────────────────────
    if not philosophy.empty:
        phil = philosophy[philosophy["oc_name"] == matched].sort_values("rank")
        if not phil.empty:
            st.markdown(
                "**Philosophy archetype fit** "
                "_(within-OC z-score across 5 archetypes — which lineage's route shape pulls hardest)_"
            )
            phi_colors = ["#1f77b4" if z >= 0 else "#aaaaaa" for z in phil["fit_z_avg"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=phil["fit_z_avg"],
                y=phil["philosophy"],
                orientation="h",
                marker=dict(color=phi_colors),
                text=phil["fit_z_avg"].apply(lambda z: f"{z:+.2f}σ"),
                textposition="outside",
                hovertemplate="%{y}<br>Fit z (within-OC): %{x:+.2f}<extra></extra>",
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=10, r=40, t=10, b=20),
                xaxis=dict(title="Within-OC z (across philosophies)",
                           zeroline=True, zerolinecolor="#888"),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── 8.5. Play-call distribution explorer — granular cross-filtered view ─
    _render_oc_play_distribution_panel(matched)

    # ── 9. Player Lift (Feature A) — boost/drag of skill players under this OC ─
    _render_oc_player_lift_panel(matched)

    # ── 10. Game-script splits (Feature B) — leading/tied/trailing identity ───
    _render_oc_gamescript_panel(matched)

    # ── 11. Career drift (Feature C) — year-over-year fingerprint evolution ──
    _render_oc_drift_panel(matched)


# ─────────────────────────────────────────────────────────────
# Play-call distribution explorer (granular cross-filtered view)
# ─────────────────────────────────────────────────────────────

def _render_oc_play_distribution_panel(oc_name: str) -> None:
    """Cross-filterable play-distribution view: pick season + filters →
    see route / gap / field-area distributions for this OC."""
    df = load_oc_play_distribution()
    if df.empty:
        return
    sub_full = df[df["oc_name"] == oc_name].copy()
    if sub_full.empty:
        return

    st.markdown('<div class="section-divider" style="margin-top:8px"></div>',
                unsafe_allow_html=True)
    st.markdown("### 🗺️ Play-call distribution explorer")
    st.markdown(
        "**What this is:** The deep-dive view into this OC's actual play-calling. "
        "Pick a season (or career), apply any combination of filters — down, "
        "distance, pressure, coverage, quarter, personnel, formation — and the "
        "three charts below show **what this OC actually called** in that exact slice.\n\n"
        "**The three charts:**\n"
        "- **🎯 Pass routes** — top routes the receivers ran on pass plays\n"
        "- **🏃 Run gaps** — where the running backs attacked (end / tackle / guard)\n"
        "- **📡 Field area thrown to** — heatmap of where receivers caught balls "
        "(left / middle / right × short / intermediate / deep)\n\n"
        "**Why we believe this matters:** The fingerprint up top tells you the "
        "OC's overall identity. This view lets you ask **specific** scouting "
        "questions like *\"On 3rd-and-7 against zone coverage when trailing, "
        "what does this OC actually call?\"* That's where real coordinator "
        "intelligence lives."
    )

    # ── Season selector ────────────────────────────────────────
    seasons_avail = sorted(sub_full["season"].unique().tolist())
    sel_year = st.selectbox(
        "Season",
        options=["Career"] + [str(int(s)) for s in seasons_avail],
        index=0,
        key=f"play_dist_season_{oc_name}",
    )
    if sel_year != "Career":
        sub = sub_full[sub_full["season"] == int(sel_year)].copy()
    else:
        sub = sub_full.copy()

    if sub.empty:
        st.info("No plays for this season."); return

    # ── Filter row 1 (always visible) ──────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        downs = st.multiselect(
            "Down", options=[1, 2, 3, 4], default=[1, 2, 3, 4],
            key=f"pd_downs_{oc_name}",
            help="Filter to specific downs."
        )
    with f2:
        distance_opts = ["Short (≤3)", "Medium (4-7)", "Long (8+)"]
        distances = st.multiselect(
            "Yards to go", options=distance_opts, default=distance_opts,
            key=f"pd_dist_{oc_name}",
            help="Bucketed distance to first down."
        )
    with f3:
        pressure = st.radio(
            "Pressure", options=["All", "Pressured", "Clean"],
            horizontal=True, index=0,
            key=f"pd_press_{oc_name}",
            help="Pressured = QB faced rush pressure on the play (PFF/NGS-derived). "
                 "Clean = no pressure. 2018+ data only.",
        )
    with f4:
        cov_opts = sorted(sub["coverage_simple"].dropna().unique().tolist())
        covs = st.multiselect(
            "Coverage faced", options=cov_opts, default=[],
            key=f"pd_cov_{oc_name}",
            help="Defensive coverage shown to the offense (man/zone × Cover-1/2/3/4/6). "
                 "Empty = all coverages. 2018+ data only.",
        )

    # ── Filter row 2 (in expander) ─────────────────────────────
    with st.expander("More filters: game state · quarter · personnel · formation",
                       expanded=False):
        f5, f6, f7, f8 = st.columns(4)
        with f5:
            gs_opts = ["Leading by 8+", "Tied / one-score", "Trailing by 8+"]
            gs_pick = st.multiselect(
                "Game state", options=gs_opts, default=[],
                key=f"pd_gs_{oc_name}")
        with f6:
            qtrs = st.multiselect(
                "Quarter", options=[1, 2, 3, 4, 5], default=[],
                key=f"pd_qtr_{oc_name}",
                help="5 = OT.")
        with f7:
            pers_opts = sorted([p for p in sub["personnel_simple"].dropna().unique()
                                if p != "Other"])
            pers = st.multiselect(
                "Personnel", options=pers_opts + ["Other"], default=[],
                key=f"pd_pers_{oc_name}",
                help="Offensive personnel: 11 = 1 RB + 1 TE + 3 WR; "
                     "12 = 1 RB + 2 TE + 2 WR; 21 = 2 RB + 1 TE + 2 WR; etc.")
        with f8:
            form_opts = sorted([f for f in sub["formation_simple"].dropna().unique()])
            forms = st.multiselect(
                "Formation", options=form_opts, default=[],
                key=f"pd_form_{oc_name}")

    # ── Apply filters ───────────────────────────────────────────
    if downs:
        sub = sub[sub["down"].fillna(0).astype(int).isin(downs)]
    if distances:
        sub = sub[sub["distance_bucket"].isin(distances)]
    if pressure == "Pressured":
        sub = sub[sub["pressure_cat"] == "Pressured"]
    elif pressure == "Clean":
        sub = sub[sub["pressure_cat"] == "Clean"]
    if covs:
        sub = sub[sub["coverage_simple"].isin(covs)]
    if gs_pick:
        sub = sub[sub["gamestate"].isin(gs_pick)]
    if qtrs:
        sub = sub[sub["qtr"].fillna(0).astype(int).isin(qtrs)]
    if pers:
        sub = sub[sub["personnel_simple"].isin(pers)]
    if forms:
        sub = sub[sub["formation_simple"].isin(forms)]

    # ── Filter summary ──────────────────────────────────────────
    n_total = len(sub)
    n_pass = int((sub["play_type"] == "pass").sum())
    n_run = int((sub["play_type"] == "run").sum())
    pass_rate = (n_pass / n_total) if n_total > 0 else 0.0
    avg_epa = sub["epa"].mean() if n_total > 0 else float("nan")
    succ_rate = sub["success"].mean() if n_total > 0 else float("nan")

    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Plays in slice", f"{n_total:,}")
    sm2.metric("Pass rate", f"{pass_rate:.0%}",
               help=f"{n_pass:,} pass · {n_run:,} run")
    sm3.metric("EPA / play",
               f"{avg_epa:+.3f}" if pd.notna(avg_epa) else "—",
               help="Average EPA per play in this filter slice.")
    sm4.metric("Success rate",
               f"{succ_rate:.1%}" if pd.notna(succ_rate) else "—",
               help="% of plays that gained enough yards for the situation.")

    if n_total < 10:
        st.warning("Too few plays in this slice (need ≥10) — loosen filters.")
        return

    # ── Three charts ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    # 1) Pass routes
    with c1:
        st.markdown("**🎯 Pass routes** (top 12)")
        pass_only = sub[sub["play_type"] == "pass"]
        routes = pass_only["route"].dropna()
        if len(routes) >= 5:
            counts = routes.value_counts().head(12)
            shares = counts / len(routes) * 100
            fig = go.Figure(go.Bar(
                x=counts.values, y=counts.index,
                orientation="h",
                marker=dict(color="#0076B6"),
                text=[f"{int(c)} ({s:.0f}%)" for c, s in zip(counts.values, shares.values)],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} plays<extra></extra>",
            ))
            fig.update_layout(
                height=440, margin=dict(l=10, r=70, t=10, b=20),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                xaxis=dict(title="# plays"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("_Not enough pass plays with route data in this slice._")

    # 2) Run gaps
    with c2:
        st.markdown("**🏃 Run gaps**")
        run_only = sub[sub["play_type"] == "run"]
        gap_data = run_only.dropna(subset=["run_gap"])
        if len(gap_data) >= 5:
            order = ["end", "tackle", "guard"]
            counts = gap_data["run_gap"].value_counts().reindex(order, fill_value=0)
            shares = counts / counts.sum() * 100
            colors = {"end": "#52a370", "tackle": "#0076B6", "guard": "#aa3a2a"}
            fig = go.Figure(go.Bar(
                x=[f"{c.title()}<br>(outside)" if c == "end" else
                    f"{c.title()}<br>(power)" if c == "guard" else
                    f"{c.title()}<br>(zone)" for c in counts.index],
                y=counts.values,
                marker=dict(color=[colors.get(c, "#888") for c in counts.index]),
                text=[f"{int(v)}<br>({s:.0f}%)" for v, s in zip(counts.values, shares.values)],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>%{y} runs<extra></extra>",
            ))
            fig.update_layout(
                height=440, margin=dict(l=10, r=10, t=10, b=20),
                yaxis=dict(title="# runs"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            # Run location split
            loc_data = run_only.dropna(subset=["run_location"])
            if len(loc_data) >= 5:
                loc_counts = loc_data["run_location"].value_counts()
                loc_shares = loc_counts / loc_counts.sum() * 100
                st.caption(
                    "**Direction:** "
                    + " · ".join(f"{loc.title()} {sh:.0f}%"
                                  for loc, sh in loc_shares.items())
                )
        else:
            st.caption("_Not enough run plays with gap data in this slice._")

    # 3) Field area thrown to (3x4 heatmap: pass_location × pass_depth)
    with c3:
        st.markdown("**📡 Field area thrown to**")
        pass_only = sub[sub["play_type"] == "pass"]
        loc_depth = pass_only.dropna(subset=["pass_location", "pass_depth_bucket"])
        if len(loc_depth) >= 5:
            depth_order = ["Deep (20+)", "Intermediate (10-19)",
                           "Short (0-9)", "Behind LOS"]
            loc_order = ["left", "middle", "right"]
            heat = (loc_depth.groupby(["pass_depth_bucket", "pass_location"])
                            .size().unstack(fill_value=0)
                            .reindex(index=depth_order, columns=loc_order, fill_value=0))
            heat_pct = heat / heat.values.sum() * 100 if heat.values.sum() > 0 else heat
            text_vals = [[f"{int(heat.iloc[i, j])}<br>{heat_pct.iloc[i, j]:.0f}%"
                          for j in range(heat.shape[1])]
                          for i in range(heat.shape[0])]
            fig = go.Figure(go.Heatmap(
                z=heat_pct.values,
                x=[c.title() for c in loc_order],
                y=depth_order,
                colorscale=[
                    [0, "#f4f4f4"], [0.3, "#9ec5db"],
                    [0.6, "#0076B6"], [1, "#003e63"],
                ],
                text=text_vals, texttemplate="%{text}",
                textfont=dict(size=12, color="black"),
                hovertemplate="%{x} · %{y}<br>%{z:.1f}% of throws<extra></extra>",
                showscale=True,
                colorbar=dict(title=dict(text="% of<br>throws"), thickness=10),
            ))
            fig.update_layout(
                height=440, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(side="top", title=""),
                yaxis=dict(title=""),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("_Not enough pass plays with location data in this slice._")

    st.caption(
        "_Tip: layer filters together to scout specific tendencies — e.g. "
        "select **Down: 3rd**, **Pressure: Clean**, **Coverage: Zone · COVER 3** "
        "to see what this OC dials up on a clean 3rd-and-medium against off-zone._"
    )


# ─────────────────────────────────────────────────────────────
# Feature A: Player before/after lift panel
# ─────────────────────────────────────────────────────────────

def _render_oc_player_lift_panel(oc_name: str) -> None:
    lift = load_oc_player_lift()
    if lift.empty:
        return
    sub = lift[lift["oc_name"] == oc_name]
    if sub.empty:
        return

    st.markdown('<div class="section-divider" style="margin-top:8px"></div>',
                unsafe_allow_html=True)
    st.markdown("### 🚀 Player lift — before / after this OC")
    st.markdown(
        "**What this is:** When a skill-position player has played under **this OC AND "
        "elsewhere** in their career, we compare their **opponent-adjusted z-score** in "
        "each chunk. The same player serves as their own control — same skill, "
        "different scheme.\n\n"
        "**How to read the rows:**\n"
        "- **z (with)** = the player's adjusted z-score during their seasons under this OC\n"
        "- **z (without)** = the same player's adjusted z-score during all *other* career seasons\n"
        "- **Δ (shrunk)** = `with − without`, with Bayesian shrinkage for small samples (low samples get pulled toward 0 so we don't over-interpret one good year)\n"
        "- **n_with / n_without** = sample sizes (targets / carries / dropbacks on each side)\n\n"
        "**Per-position lift strip below:** the OC's mean delta across all qualifying "
        "players at QB / WR / TE / RB. Shows where this OC's value-add concentrates.\n\n"
        "**Why we believe this matters:** EPA stats can't tell you if a great offense "
        "comes from the OC or the talent. This metric controls for the player — so a "
        "boost almost has to come from the OC. It's the closest thing to a controlled "
        "experiment we can run with NFL data."
    )

    # Per-position summary metric strip
    pos_order = ["QB", "WR", "TE", "RB"]
    cols = st.columns(len(pos_order))
    for col, pos in zip(cols, pos_order):
        pos_sub = sub[sub["position"] == pos]
        with col:
            if pos_sub.empty:
                st.metric(f"{pos} lift", "—", help="No qualifying players.")
            else:
                pos_sub = pos_sub.copy()
                pos_sub["weight"] = pos_sub[["n_with", "n_without"]].min(axis=1)
                w = pos_sub["weight"].sum()
                lift_score = ((pos_sub["shrunk_delta"] * pos_sub["weight"]).sum() / w
                              if w > 0 else float("nan"))
                arrow = "↑" if lift_score > 0 else ("↓" if lift_score < 0 else "→")
                st.metric(
                    f"{pos} lift",
                    f"{arrow} {lift_score:+.2f}σ",
                    delta=f"n = {len(pos_sub)}",
                    delta_color="off",
                )

    # Top boost / drag tables
    bd_l, bd_r = st.columns(2)
    show_cols = ["position", "player_name", "with_oc_z", "without_oc_z",
                 "shrunk_delta", "n_with", "n_without"]
    pretty = {"position": "Pos", "player_name": "Player",
              "with_oc_z": "z (with)", "without_oc_z": "z (without)",
              "shrunk_delta": "Δ (shrunk)", "n_with": "n_with",
              "n_without": "n_without"}

    with bd_l:
        st.markdown("**🔺 Top boosted players**")
        boosted = sub.nlargest(8, "shrunk_delta")[show_cols].rename(columns=pretty)
        if not boosted.empty:
            st.dataframe(boosted, use_container_width=True, hide_index=True,
                         column_config={
                             "z (with)": st.column_config.NumberColumn(format="%+.2f"),
                             "z (without)": st.column_config.NumberColumn(format="%+.2f"),
                             "Δ (shrunk)": st.column_config.ProgressColumn(
                                 format="%+.2f", min_value=-3.0, max_value=3.0,
                                 help="OC's shrunk lift (z-units) on this player."),
                         })
        else:
            st.caption("_None._")
    with bd_r:
        st.markdown("**🔻 Top dragged players**")
        dragged = sub.nsmallest(8, "shrunk_delta")[show_cols].rename(columns=pretty)
        if not dragged.empty:
            st.dataframe(dragged, use_container_width=True, hide_index=True,
                         column_config={
                             "z (with)": st.column_config.NumberColumn(format="%+.2f"),
                             "z (without)": st.column_config.NumberColumn(format="%+.2f"),
                             "Δ (shrunk)": st.column_config.ProgressColumn(
                                 format="%+.2f", min_value=-3.0, max_value=3.0,
                                 help="OC's shrunk lift (z-units) on this player."),
                         })
        else:
            st.caption("_None._")

    # ── Feature D — OC × QB archetype interaction matrix ──────
    qb_lift = load_oc_qb_archetype_lift()
    if not qb_lift.empty:
        m_sub = qb_lift[qb_lift["oc_name"] == oc_name]
        if not m_sub.empty:
            st.markdown("---")
            st.markdown("**🎮 QB-archetype interaction**")
            st.markdown(
                "**What this is:** Same OC, different QBs. We categorize each QB on "
                "the team by their **carries-per-pass-attempt ratio**:\n"
                "- **Mobile dual-threat (≥18%)** — Lamar, Hurts, Allen, Kyler, Daniels\n"
                "- **Pocket-mobile (10-18%)** — Mahomes, Burrow, Stroud, modern hybrids\n"
                "- **Pocket passer (<10%)** — Goff, Cousins, Brady, Stafford\n\n"
                "**How to read the heatmap:** Rows = QB archetype on the team that "
                "season. Columns = target position (WR / TE / RB). Each cell = the "
                "OC's mean lift on that position when paired with that QB type. "
                "**Green = lift, red = drag, gray = no effect.** Empty cell = the OC "
                "never had that QB type on the team in our data window.\n\n"
                "**Why we believe this matters:** Some OCs need a mobile QB to make "
                "their offense work; others lift WRs regardless. This is the cross-tab "
                "that exposes those dependencies. *\"Does Reid lift WRs more under "
                "Mahomes than Alex Smith?\"* — read it off the chart."
            )

            arch_order = ["Mobile dual-threat", "Pocket-mobile", "Pocket passer"]
            pos_order = ["WR", "TE", "RB"]
            piv = (m_sub.pivot_table(index="qb_archetype", columns="position",
                                       values="mean_lift_z", aggfunc="first")
                          .reindex(index=arch_order, columns=pos_order))
            piv_n = (m_sub.pivot_table(index="qb_archetype", columns="position",
                                        values="n_player_seasons", aggfunc="first")
                          .reindex(index=arch_order, columns=pos_order))

            # Heatmap-as-table
            z_vals = piv.values.astype(float)
            text_vals = []
            for i in range(piv.shape[0]):
                row = []
                for j in range(piv.shape[1]):
                    v = piv.iloc[i, j]
                    n = piv_n.iloc[i, j]
                    if pd.isna(v):
                        row.append("—")
                    else:
                        row.append(f"{v:+.2f}σ<br><span style='font-size:0.7rem;opacity:0.7;'>n={int(n)}</span>")
                text_vals.append(row)

            fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=pos_order,
                y=arch_order,
                colorscale=[
                    [0.0, "#aa3a2a"], [0.25, "#cc6651"], [0.5, "#f0f0f0"],
                    [0.75, "#52a370"], [1.0, "#1f7a3a"],
                ],
                zmid=0, zmin=-1.5, zmax=1.5,
                text=text_vals, texttemplate="%{text}",
                textfont=dict(size=14),
                hovertemplate="<b>%{y}</b> QB<br>Lift on <b>%{x}</b>: %{z:+.2f}σ<extra></extra>",
                showscale=True,
                colorbar=dict(title=dict(text="Mean<br>lift (σ)"), thickness=12),
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(side="top", title=""),
                yaxis=dict(title=""),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "_Cells with `n=` show qualifying player-seasons in that cell. "
                "Empty cells = OC never had that QB archetype on the team in our window._"
            )


# ─────────────────────────────────────────────────────────────
# Feature B: Game-script splits panel
# ─────────────────────────────────────────────────────────────

def _render_oc_gamescript_panel(oc_name: str) -> None:
    gs = load_oc_gamescript()
    if gs.empty:
        return
    sub = gs[gs["oc_name"] == oc_name]
    if sub.empty:
        return

    st.markdown('<div class="section-divider" style="margin-top:8px"></div>',
                unsafe_allow_html=True)
    st.markdown("### ⏱️ Game-script identity — leading / tied / trailing")
    st.markdown(
        "**What this is:** Same OC, three game states — **Leading by 8+**, "
        "**Tied or within 7**, **Trailing by 8+**. We measure their EPA, success "
        "rate, pass rate, and no-huddle tempo within each state, then z-score within "
        "the bucket (vs all OCs in our pool when their team was in that same state).\n\n"
        "**How to read the verbs:** the big colored word at the top of each column "
        "is your at-a-glance answer:\n"
        "- **🔥 grinds** (z ≥ +1.0) = elite in this state\n"
        "- **✅ holds** (z ≥ 0) = above average\n"
        "- **⚠️ slips** (z ≥ −1.0) = below average\n"
        "- **❌ folds** (z < −1.0) = significantly hurt\n\n"
        "**Why we believe this matters:** OCs that grind when leading are sustainable "
        "winners; OCs that fold when trailing have a comeback problem. Identity matters."
    )

    bucket_order = ["leading", "tied", "trailing"]
    bucket_label = {"leading": "🟢 Leading by 8+", "tied": "⚖️ Tied / one-score",
                    "trailing": "🔴 Trailing by 8+"}

    # Three columns, one per bucket
    cols = st.columns(3)
    for col, bucket in zip(cols, bucket_order):
        b_row = sub[sub["gamescript"] == bucket]
        with col:
            st.markdown(f"**{bucket_label[bucket]}**")
            if b_row.empty:
                st.caption("_No data._")
                continue
            r = b_row.iloc[0]
            n = int(r["n_plays"])
            epa = float(r["epa_per_play"])
            epa_z = float(r["epa_per_play_z"])
            pass_rate = float(r["pass_rate"])
            pass_rate_z = float(r["pass_rate_z"])
            no_huddle = float(r["no_huddle_rate"])
            verb = ("🔥 grinds" if epa_z >= 1.0 else
                    "✅ holds" if epa_z >= 0.0 else
                    "⚠️ slips" if epa_z >= -1.0 else
                    "❌ folds")
            st.markdown(
                f"<div style='font-size:1.4rem;font-weight:700;color:"
                f"{'#1f7a3a' if epa_z >= 0.5 else '#aa3a2a' if epa_z <= -0.5 else '#666'};'>"
                f"{verb}</div>",
                unsafe_allow_html=True)
            st.metric("EPA/play", f"{epa:+.3f}", delta=f"{epa_z:+.2f}σ")
            st.caption(
                f"Pass rate: **{pass_rate:.0%}** ({pass_rate_z:+.1f}σ) · "
                f"No-huddle: **{no_huddle:.1%}** · "
                f"n = {n:,} plays"
            )


# ─────────────────────────────────────────────────────────────
# Feature C: Career drift visualization
# ─────────────────────────────────────────────────────────────

def _render_oc_drift_panel(oc_name: str) -> None:
    """Year-by-year fingerprint evolution for the play-caller."""
    per_season = load_oc_per_season()
    if per_season.empty:
        return
    sub = per_season[per_season["coordinator"] == oc_name].copy()
    if len(sub) < 2:
        return  # need 2+ seasons to show drift
    sub = sub.sort_values("season")

    st.markdown('<div class="section-divider" style="margin-top:8px"></div>',
                unsafe_allow_html=True)
    st.markdown("### 📈 Career drift — year-over-year fingerprint")
    st.markdown(
        "**What this is:** Eight key z-scores plotted over this OC's tenured seasons. "
        "Each line is one stat. Each year is a single point. The dotted black line "
        "at z=0 is league-average.\n\n"
        "**How to read it:**\n"
        "- Lines **above the zero line** = above-average that season\n"
        "- Lines **below** = below-average\n"
        "- Lines **trending up over time** = improving relative to peers\n"
        "- Lines **trending down** = regressing\n\n"
        "**Why we believe this matters:** Most OCs don't stand still. They evolve. "
        "Some peak in year 2-3 and decline. Some keep getting better. Some flip styles "
        "after a QB change. The drift chart tells the story; the **Notable drifts** "
        "expander below flags any year-over-year change of |1σ| or more — those are "
        "usually real events (QB swap, scheme overhaul, coordinator shake-up)."
    )

    drift_metrics = [
        ("epa_per_play_z", "EPA/play"),
        ("pass_epa_per_play_z", "Pass EPA"),
        ("rush_epa_per_play_z", "Rush EPA"),
        ("success_rate_z", "Success rate"),
        ("explosive_pass_rate_z", "Explosive pass"),
        ("third_down_rate_z", "3rd down conv"),
        ("red_zone_td_rate_z", "RZ TD%"),
        ("win_pct_z", "Win %"),
    ]
    have = [(c, l) for c, l in drift_metrics if c in sub.columns]
    if not have:
        st.caption("_No drift data available for this OC._"); return

    fig = go.Figure()
    palette = ["#0076B6", "#1f7a3a", "#cc6651", "#9467bd", "#aa3a2a",
               "#ff7f0e", "#52a370", "#888888"]
    for i, (col, lbl) in enumerate(have):
        fig.add_trace(go.Scatter(
            x=sub["season"].astype(int).tolist(),
            y=sub[col].astype(float).tolist(),
            mode="lines+markers",
            name=lbl,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{lbl}</b><br>%{{x}}: %{{y:+.2f}}σ<extra></extra>",
        ))
    # Reference line at z=0
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(title="Season",
                   tickmode="array",
                   tickvals=sub["season"].astype(int).tolist(),
                   tickformat="d"),
        yaxis=dict(title="z-score (within season pool)",
                   zeroline=True, zerolinecolor="rgba(0,0,0,0.4)"),
        legend=dict(orientation="h", y=-0.2, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Notable drift events: |Δz| > 1.0 between adjacent seasons
    drifts = []
    seasons_list = sub["season"].astype(int).tolist()
    for col, lbl in have:
        vals = sub[col].astype(float).tolist()
        for j in range(1, len(vals)):
            if pd.notna(vals[j]) and pd.notna(vals[j-1]):
                d = vals[j] - vals[j-1]
                if abs(d) >= 1.0:
                    drifts.append({
                        "Year": f"{seasons_list[j-1]} → {seasons_list[j]}",
                        "Metric": lbl,
                        "From": f"{vals[j-1]:+.2f}σ",
                        "To": f"{vals[j]:+.2f}σ",
                        "Δ": f"{d:+.2f}σ",
                    })
    if drifts:
        with st.expander(f"📍 Notable year-over-year drifts ({len(drifts)})"):
            st.dataframe(pd.DataFrame(drifts),
                         use_container_width=True, hide_index=True)


NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

st.title("🦁 NFL Offensive Coordinator")

curation_df = load_oc_curation()

if "oc_profile_team" not in st.session_state:
    st.session_state.oc_profile_team = None

# URL query param ↔ session state sync — makes individual OC profiles
# shareable via links like ?team=DET
_qp_team = st.query_params.get("team")
if _qp_team:
    _qp_team_upper = str(_qp_team).upper().strip()
    # Validate against curation; fall back to grid if invalid
    if not curation_df.empty and _qp_team_upper in curation_df["team"].values:
        if st.session_state.oc_profile_team != _qp_team_upper:
            st.session_state.oc_profile_team = _qp_team_upper
    else:
        # Bad team in URL — clear it to avoid loops
        try:
            del st.query_params["team"]
        except Exception:
            pass


def _gas_badge_html(oc_name: str, arch_name: str | None) -> str:
    """Look up the GAS score for an OC (or HC architect) and return an
    HTML badge for the team-color header card."""
    if not OC_GAS_CAREER_PATH.exists() or not oc_name:
        return ""
    try:
        gas = pd.read_parquet(OC_GAS_CAREER_PATH)
    except Exception:
        return ""
    candidates = [n for n in [arch_name, oc_name] if n]
    pool = gas["coordinator"].dropna().astype(str).tolist()
    matched = None
    for n in candidates:
        m = find_oc_in_pool(n, pool)
        if m:
            matched = m
            break
    if not matched:
        return ""
    row = gas[gas["coordinator"] == matched].iloc[0]
    score = row.get("gas_score")
    label = row.get("gas_label", "")
    conf = row.get("gas_confidence", "")
    if pd.isna(score):
        return ""
    return (
        '<div style="background: rgba(255,255,255,0.15); padding: 8px 14px; '
        'border-radius: 10px; min-width: 110px; text-align: center; '
        'margin-left: auto;">'
        '<div style="font-size: 0.7rem; opacity: 0.85; letter-spacing: 1.2px; '
        'text-transform: uppercase;">GAS Score</div>'
        f'<div style="font-size: 1.9rem; font-weight: 700; line-height: 1.0;">{score:.1f}</div>'
        f'<div style="font-size: 0.72rem; opacity: 0.9;">{label}</div>'
        f'<div style="font-size: 0.6rem; opacity: 0.75;">conf: {conf} · for {matched}</div>'
        '</div>'
    )


def _render_oc_detail(row: pd.Series) -> None:
    theme = team_theme(row["team"])
    yrs_html = (f"<strong>{row['years_in_role']}</strong> &nbsp;·&nbsp; "
                if row['years_in_role'] else "")
    arch_html = f"<em>{row['architect_status']}</em>" if row['architect_status'] else ""
    arch_name = extract_hc_architect_name(row.get("architect_status"))
    gas_html = _gas_badge_html(row.get("oc_name", ""), arch_name)
    header_html = (
        f'<div style="background: linear-gradient(135deg, {theme["primary"]} 0%, {theme["secondary"]} 100%);'
        f'padding: 22px 26px; border-radius: 12px; color: white; margin-bottom: 16px;'
        f'box-shadow: 0 2px 8px rgba(0,0,0,0.12);">'
        f'<div style="display: flex; align-items: center; gap: 22px;">'
        f'<img src="{theme["logo"]}" style="height: 80px; background: rgba(255,255,255,0.12); padding: 8px; border-radius: 10px;" />'
        f'<div>'
        f'<div style="font-size: 0.78rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1.4px;">'
        f'{theme["name"]} · Offensive Coordinator</div>'
        f'<div style="font-size: 2rem; font-weight: 700; margin: 4px 0;">{row["oc_name"] or "—"}</div>'
        f'<div style="font-size: 0.95rem; opacity: 0.92;">{yrs_html}{arch_html}</div>'
        f'</div>'
        f'{gas_html}'
        f'</div></div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # GAS Score explainer (only show if a GAS badge is rendered in the header)
    if "GAS Score" in (gas_html or ""):
        st.caption(
            "**GAS Score** (top-right) = our 0-100 grade for the OC, like PFF. "
            "**50 = league average. 80+ = elite. 30-40 = below average.** "
            "It blends four things: Efficiency (45%), Explosiveness (15%), "
            "Situational execution (20%), and Clutch performance (20%). "
            "Confidence (HIGH/MEDIUM/LOW) reflects how many seasons we have on them."
        )

    if row.get("one_liner_identity"):
        st.markdown(f"> _{row['one_liner_identity']}_")

    # Curation grid plain-English intro
    st.markdown(
        "**The card below** is curated football intel — the OC's identity "
        "in plain football terms (system, mentor, what kind of QB they need, etc.). "
        "Then the data layers below back up (or contradict) the human read."
    )

    grid_specs = [
        ("🎯 Passing system", "passing_system"),
        ("🏃 Running system", "running_system"),
        ("👥 Personnel preference", "personnel_preference"),
        ("⏱️ Pace identity", "pace_identity"),
        ("🎮 QB type designed for", "qb_type"),
        ("🏁 Red-zone identity", "redzone_identity"),
        ("🌳 Coaching tree", "coaching_tree"),
        ("👤 Mentor (primary)", "mentor_primary"),
        ("🧱 OL coach (run-game architect)", "ol_coach"),
    ]
    for i in range(0, len(grid_specs), 2):
        c1, c2 = st.columns(2)
        with c1:
            label, key = grid_specs[i]
            st.markdown(f"**{label}**")
            st.markdown(row.get(key, "") or "_—_")
        if i + 1 < len(grid_specs):
            with c2:
                label, key = grid_specs[i + 1]
                st.markdown(f"**{label}**")
                st.markdown(row.get(key, "") or "_—_")

    # Coordinator detail (moved up from the rater section): year selector
    # → radar (vs league avg) + per-year stat table for this play-caller.
    _render_oc_coord_detail(row)

    # Phase 1: data-derived scheme fingerprint (D&D, run gap, coverage,
    # tempo, pressure, philosophy fit). Renders empty state when the
    # play-caller has no historical OC tenure in our seed mapping yet.
    _render_oc_scheme_panels(row)


def _render_oc_stat_glossary_DEAD() -> None:
    """Standalone glossary — replaced by inline explainers at each panel.
    Kept here as dead code for reference; not called anywhere."""
    with st.expander("❓ **Stat glossary — what every number on this page means**", expanded=False):
        st.markdown("""
*Click any section to learn what we're measuring and what we believe it tells us.*

---

### 🎯 The basics (used everywhere)

**EPA / play (Expected Points Added)** — Football has a number for every play that says *"how much did this play help us score?"* Gain 5 yards on 1st-and-10? +0.10 EPA. Lose 3? −0.20. Add up every play and you see whether the offense actually moves the ball or just spins its wheels. Higher is always better.

**Success rate** — What percent of plays were "successful" — meaning they gained enough yards for the situation. (50% on 1st down. 70% of remaining yards on 2nd. 100% on 3rd/4th.) Higher = a more *consistent* offense (lots of small wins). EPA can be inflated by a few huge plays; success rate can't.

**z-score (those σ symbols)** — A z-score tells you how unusual a number is.
- **0** = league-average
- **+1σ** = above average (better than ~84% of teams)
- **+2σ** = top 2%
- **−1σ** = below average
Think of it as a *"how surprising is this number?"* meter.

**Percentile (the radar chart)** — If a player ranks at the **90th percentile**, they're better than 90 out of every 100 OCs. The **dotted black ring on the radar = 50th percentile = league average.** Anywhere outside the ring is good; anywhere inside is bad.

**Adjusted (SOS-adjusted)** — The opponent matters. A QB throwing against bad defenses looks better than the same QB facing great ones. *SOS-adjusted* means we accounted for who they faced. Bigger number on a hard schedule beats a bigger number on a cupcake schedule.

**Roster-adjusted** — Some OCs have Mahomes. Some have a backup. *Roster-adjusted* tries to subtract how much credit goes to the roster vs the OC. (We use team-rating as the proxy — it's imperfect but it's honest.)

---

### 🏈 GAS Score (the headline grade)

**GAS Score (0-100)** — One simple grade for the OC, like PFF gives players. **50 = league-average. 80+ = elite. 30-40 = below average.** It's a weighted blend of four things:

| Bundle | Weight | What it measures |
|---|---|---|
| **Efficiency** | 45% | EPA, success rate, pass/rush EPA — how productive is the offense? |
| **Explosiveness** | 15% | Big-play rates (20+ yard passes, 10+ yard runs) |
| **Situational** | 20% | 3rd-down conversion, red-zone TD%, win% |
| **Clutch** | 20% | How they performed in game-on-the-line moments |

**Confidence (HIGH / MEDIUM / LOW)** — How much sample we have to grade them. 4+ seasons = HIGH. 1 season = LOW.

---

### 📋 Coordinator detail (the radar)

**Year selector** — *"Career"* averages all of their OC seasons together. Click a single year to see just that season — radar redraws, year-by-year table stays.

**Year-by-year table** — Every season they coached, sorted newest-first. **Plays** = total scrimmage plays. **EPA / Succ% / 3rd% / RZ TD% / Win%** = the season's actual numbers (not z-scores).

---

### 📈 Scheme fingerprint (how they actually call plays)

**Pass rate by D&D (down-and-distance)** — On 1st-and-10, do they throw or run? On 3rd-and-7? Different OCs have very different fingerprints — a Shanahan-tree OC like Ben Johnson runs a *lot* on 1st-and-10. The bars show this OC; the **dotted grey line = league average**.

**Run gap distribution** — When a team runs the ball, the back aims for a specific gap.
- **End** = outside (zone runs / sweeps) — the Shanahan signature
- **Tackle** = mid (mid-zone, off-tackle) — outside-zone teams hit this a lot
- **Guard** = interior (power runs between center and guard) — Greg-Roman-style power offenses

A tackle-heavy OC is a zone-blocking team. A guard-heavy OC is a power team.

**Shotgun rate by D&D** — How often the QB lines up in shotgun (vs. under center). **Low shotgun = under-center, play-action-heavy** (classic Shanahan). **High shotgun = spread-style, faster reads** (Reid / Spread-RPO).

**Coverage-reactive attack** — Defenses run **MAN** coverage (one defender per receiver) or **ZONE** (each defender covers an area). When this OC sees man, what do they call? When they see zone? **Avg AY = average air yards** = how far downfield throws go. Some OCs throw deep against man (single-high beats one-on-one); some dump-off (quick game vs press).

**Tempo & pressure**
- **Overall no-huddle** = how often they go fast between plays (no huddle)
- **2-min no-huddle** = same, but only in end-of-half drills
- **Pressure faced (5+ rushers)** = how often defenses bring extra rushers (a blitz). High = defenses don't trust this OC's quick-game

**Pass rate by field position**
- **Backed-up** = own 10-yard line or worse
- **Red zone (RZ)** = inside the opponent's 20
- **Goal line (GL)** = inside the 5

Reveals philosophy: pass-happy in the RZ vs grind-it-out run game.

---

### 🎯 Clutch profile

**Leverage definition** — *Which plays count as "clutch"?* Five options:
1. **WP volatility** — only plays that swung the win-probability by 5%+ (the strict definition)
2. **High-stakes situations** — 4Q close games + RZ + 3rd-and-medium+ + 2-min drill (rule-based)
3. **EPA × leverage** — every play counted, but weighted by how much it mattered (continuous)
4. **Hybrid** — leverage × situation multiplier (a smart blend)
5. **All plays** — baseline (no filter, sanity check)

**Three lenses (per metric):**
- **Raw fulcrum** = actual EPA in clutch plays (color-coded vs league of OCs)
- **Roster-adjusted** = same, but accounting for roster quality
- **🎯 Elevation index (the headline)** = `clutch EPA − non-clutch EPA` for the same OC. Tells you *does this OC step up or fade when the game's on the line?* Roster effect mostly cancels (same roster either way). Plain-English verb tells you the answer.

---

### 🌳 Philosophy archetype fit

We score how much each OC's actual route distribution looks like one of 5 historical schools:
- **WCO (West Coast Offense)** — Bill Walsh's timing-based passing. Modern Shanahan-tree adopted it heavily.
- **Air Coryell** — Don Coryell's vertical attack. Push the ball downfield.
- **Erhardt-Perkins** — Belichick's concept-based system. Same play, multiple looks.
- **Spread / RPO** — College-imported quick-game with run-pass options.
- **Power Run / Vertical** — Greg Roman style. Run-heavy with vertical play-action.

The bars show **within-OC z-score**: which archetype pulls hardest on *this OC's* play-calling. Higher = stronger fit.

---

### 🚀 Player lift (the causal cut)

**The big idea:** When a player plays under one OC AND elsewhere in their career, the *same player* serves as their own control. We measure how their **SOS-adjusted z-score** changed.

- **z (with)** = the player's adjusted z-score during their seasons under this OC
- **z (without)** = the same player's adjusted z-score during their *other* career years
- **Δ (shrunk)** = with − without, with Bayesian shrinkage for small samples (low samples get pulled toward zero so we don't over-interpret)
- **n_with / n_without** = sample sizes (targets / carries / dropbacks)

**What it tells you:** Did this OC make the player *better* (boost), *worse* (drag), or no change? It's the cleanest causal story we can tell with the data — same player, different scheme.

**Per-position lift strip** — The OC's mean Δ across all qualifying players at QB / WR / TE / RB. Tells you where this OC's value-add concentrates (or doesn't).

---

### 🎮 QB-archetype interaction matrix

Same OC, different QBs. We categorize each QB by their **carries-per-pass-attempt ratio**:
- **Mobile dual-threat (≥18%)** — Lamar, Hurts, Allen, Kyler, Daniels
- **Pocket-mobile (10-18%)** — Mahomes, Burrow, Stroud, modern hybrids
- **Pocket passer (<10%)** — Goff, Cousins, Brady, Stafford

Each cell = the OC's mean lift on that *target position* during seasons with that *QB type*. Reveals questions like *"Does Reid's WR lift drop without Mahomes? Does McDaniel's run game collapse without a mobile QB?"*

Empty cell = the OC never had that QB archetype on the team in our window.

---

### ⏱️ Game-script identity

Same OC, three game states:
- **Leading by 8+** = comfortably ahead
- **Tied / one-score** = within 7 points either way
- **Trailing by 8+** = comfortably behind

The plain-English verb at the top of each column tells you the answer:
- **🔥 grinds** (z ≥ +1.0) = elite in this state
- **✅ holds** (z ≥ 0) = above average
- **⚠️ slips** (z ≥ −1.0) = below average
- **❌ folds** (z < −1.0) = significantly hurt

Reveals identity: *Andy Reid grinds when ahead. Some OCs fold when trailing.*

---

### 📈 Career drift

Year-over-year line chart of the OC's z-scores across 8 metrics. **z = 0 line is league average.** Lines going up = improving relative to peers. Going down = regressing. **Notable drifts** = year-over-year changes of |1σ| or more — usually corresponds to a real event (QB change, OC philosophy shift, scheme overhaul).

---

### 🦁 The big picture

Every chart and number on this page is trying to answer one of three questions:

1. **What does this OC's offense LOOK like?** (Scheme fingerprint, philosophy fit)
2. **How GOOD is the offense?** (GAS, leaderboard, EPA, success rate)
3. **How much of that is the OC vs the roster?** (Player lift, roster-adjusted, elevation, QB matrix)

The whole page is built on one principle: **show how the data answers the question, not just the answer.** Every stat is a clue — your eye does the synthesis.
        """)
        st.caption("This glossary is on every OC profile. Bookmark mentally — once it clicks, the rest of the page is a video game.")


def _render_oc_grid(df: pd.DataFrame) -> None:
    """4×8 grid of team tiles. Each tile = logo, team, OC name, architect tag.
    Clicking a tile sets st.session_state.oc_profile_team and reruns."""
    teams_sorted = sorted(df["team"].tolist())
    cols_per_row = 4
    for i in range(0, len(teams_sorted), cols_per_row):
        row_teams = teams_sorted[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, t in zip(cols, row_teams):
            r = df[df["team"] == t].iloc[0]
            theme = team_theme(t)
            with col:
                role_tag = ('HC = architect'
                            if 'HC = architect' in str(r.get('architect_status', ''))
                            else 'OC')
                yrs_tail = f" · {r['years_in_role']}" if r['years_in_role'] else ''
                tile_html = (
                    f'<div style="background: linear-gradient(135deg, {theme["primary"]} 0%, {theme["secondary"]} 100%);'
                    f'border-radius: 10px; padding: 12px; color: white; height: 130px;'
                    f'display: flex; flex-direction: column; justify-content: space-between;'
                    f'box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 4px;">'
                    f'<div style="display: flex; align-items: center; gap: 8px;">'
                    f'<img src="{theme["logo"]}" style="height: 36px; background: rgba(255,255,255,0.15); padding: 3px; border-radius: 6px;" />'
                    f'<div>'
                    f'<div style="font-size: 0.95rem; font-weight: 700; line-height: 1.1;">{t}</div>'
                    f'<div style="font-size: 0.7rem; opacity: 0.85; line-height: 1.1;">{theme["name"]}</div>'
                    f'</div></div>'
                    f'<div>'
                    f'<div style="font-size: 0.95rem; font-weight: 600; line-height: 1.2;">{r["oc_name"] or "—"}</div>'
                    f'<div style="font-size: 0.7rem; opacity: 0.85; line-height: 1.2;">{role_tag}{yrs_tail}</div>'
                    f'</div></div>'
                )
                st.markdown(tile_html, unsafe_allow_html=True)
                if st.button("Open profile", key=f"oc_tile_{t}", use_container_width=True):
                    st.session_state.oc_profile_team = t
                    st.query_params["team"] = t
                    st.rerun()


if not curation_df.empty:
    if st.session_state.oc_profile_team is None:
        st.markdown("### 📔 Offensive coordinators — all 32 teams")
        st.caption(
            "Curated 2026 staff. Click a team to open the full profile. "
            "Profile URLs are shareable — copy the address bar after clicking a team."
        )
        _render_oc_grid(curation_df)
    else:
        sel_team = st.session_state.oc_profile_team
        # Keep query param in sync (handles cases where session state was set
        # programmatically without the param being updated)
        if st.query_params.get("team") != sel_team:
            st.query_params["team"] = sel_team
        bcol, lcol = st.columns([1, 4])
        with bcol:
            if st.button("← Back to all teams", key="oc_back_btn"):
                st.session_state.oc_profile_team = None
                try:
                    del st.query_params["team"]
                except Exception:
                    pass
                st.rerun()
        with lcol:
            st.caption(
                f"📎 _Shareable link: `?team={sel_team}` is appended to the URL — "
                "anyone with the link lands directly on this profile._"
            )
        match = curation_df[curation_df["team"] == sel_team]
        if match.empty:
            st.warning(f"No curation row for {sel_team}.")
        else:
            _render_oc_detail(match.iloc[0])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("### 📊 Stat-based rater")
st.markdown("What makes a great OC? **You decide.** Drag the sliders to weight what you value, and watch NFL offensive coordinators re-rank in real time.")

# Career vs single-season toggle
view_mode = st.radio(
    "View mode",
    ["Career (2016-2025)", "2025 season only"],
    horizontal=True, index=0,
    help=(
        "Career averages all tenured seasons of actual play-callers "
        "(per oc_team_seasons.csv). 2025 is the most recent season."
    ),
)
is_career = view_mode.startswith("Career")

# Current-OCs filter
fcol1, fcol2 = st.columns(2)
with fcol1:
    filter_current = st.toggle(
        "Rank current OCs only (2026 staff)",
        value=False,
        key="oc_filter_current",
        help=(
            "Restrict the ranking pool to coordinators on a 2026 NFL staff "
            "(per the curation file). Their score still reflects their "
            "career or 2024 stats — this just hides historical-only OCs."
        ),
    )
with fcol2:
    filter_architects = st.toggle(
        "HC architects only (actual play-callers)",
        value=False,
        key="oc_filter_architects",
        help=(
            "Replace admin/co-coord OCs with the HC who actually calls "
            "the plays (Sean McVay, Andy Reid, Kyle Shanahan, Mike McDaniel, "
            "Ben Johnson, etc.). Score reflects the HC's pre-HC OC tenure "
            "where data exists. Admin OCs whose HC isn't in the rater data "
            "drop out of the pool."
        ),
    )

adjust_roster = st.toggle(
    "🛠️ Adjust for roster quality (OC value-add over expected)",
    value=True,
    key="oc_adjust_roster",
    help=(
        "Subtract the offense's expected output given roster-quality proxies. "
        "Scores reflect what the OC produced *over and above* what the roster "
        "should have produced. 2024 mode adjusts on cap allocation + draft "
        "capital (all 31 OCs). Career mode uses team-rating proxies aggregated "
        "from team_context (currently 24/106 OCs covered — the rest get raw "
        "z-scores)."
    ),
)

if is_career:
    st.caption("Career averages 2016-2025 • Play-callers only (HC architects + OCs of record who actually call plays) • Z-scores within play-caller pool")
else:
    st.caption("2025 regular season only • Play-callers only • Z-scores within 2025 play-caller pool")

try:
    if is_career:
        df = load_oc_career()
    else:
        df = load_oc_2024()
except FileNotFoundError:
    st.error("Couldn't find OC data."); st.stop()

meta = load_oc_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle("🔬 Advanced mode", value=False)

# HIDDEN 2026-05-03 — tier-checkbox UI; defaults
# applied via session_state read below.
if False:
    st.markdown("### Which stats should count?")
    tier_cols = st.columns(4)
    new_enabled = []
    for i, tier in enumerate([1, 2, 3, 4]):
        with tier_cols[i]:
            checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.oc_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"oc_tier_checkbox_{tier}")
            if checked: new_enabled.append(tier)
new_enabled = list(
    st.session_state.get(
        "oc_tiers_enabled", [1, 2])
) or [1, 2]
st.session_state.oc_tiers_enabled = new_enabled
if not new_enabled: st.warning("Enable at least one tier."); st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

bundle_weights = {}; effective_weights = {}
if not advanced_mode:
    if not active_bundles: st.info("No bundles in enabled tiers."); st.stop()
    st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")
    for bk, bundle in active_bundles.items():
        tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(f"<div class='bundle-desc'>{bundle['description']}<br><small>{tier_summary}</small></div>", unsafe_allow_html=True)
        if f"oc_bundle_{bk}" not in st.session_state: st.session_state[f"oc_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"oc_bundle_{bk}", label_visibility="collapsed", help=bundle.get("why", ""))
        st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")
    for bk in BUNDLES:
        if bk not in bundle_weights: bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption("Direct control over every stat.")
    all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2); label = stat_labels.get(z_col, z_col); meth = stat_methodology.get(z_col, {}); help_parts = []
        if meth.get("what"): help_parts.append(f"What: {meth['what']}")
        if meth.get("limits"): help_parts.append(f"Limits: {meth['limits']}")
        w = st.sidebar.slider(f"{tier_badge(tier)} {label}", 0, 100, 50, 5, key=f"adv_oc_{z_col}", help="\n\n".join(help_parts) if help_parts else None)
        if w > 0: effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}

ocs = df.copy()
if len(ocs) == 0: st.warning("No OCs found."); st.stop()

# Roster adjustment: swap raw *_z columns with *_adj_z residual z-scores
# when the toggle is on. Downstream scoring/display code references *_z and
# stays unchanged. OCs without adj data (career rows missing team-context
# proxies) keep their raw z — quietly degraded behavior.
if adjust_roster:
    for col in list(ocs.columns):
        if col.endswith("_adj_z"):
            base = col.replace("_adj_z", "_z")
            if base in ocs.columns:
                # Only override where the adjustment value is non-NaN; keep
                # raw z otherwise so OCs missing the proxy data still rank.
                mask = ocs[col].notna()
                ocs.loc[mask, base] = ocs.loc[mask, col]

ocs = score_players(ocs, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")

# Apply staff-membership filters (uses curation_df as the whitelist)
name_col = "coordinator" if "coordinator" in ocs.columns else "player_name"
if (filter_current or filter_architects) and not curation_df.empty and name_col in ocs.columns:
    pool_names = ocs[name_col].dropna().astype(str).str.strip().unique().tolist()
    keep = set()
    if filter_current:
        for n in curation_df["oc_name"].dropna().astype(str).str.strip():
            m = find_oc_in_pool(n, pool_names)
            if m: keep.add(m)
    if filter_architects:
        for status in curation_df["architect_status"].dropna().astype(str):
            hc = extract_hc_architect_name(status)
            if not hc: continue
            m = find_oc_in_pool(hc, pool_names)
            if m: keep.add(m)
    ocs = ocs[ocs[name_col].astype(str).str.strip().isin(keep)].copy()
    if len(ocs) == 0:
        st.warning(
            "No matches in the rater data for this filter combination — "
            "try turning a toggle off."
        )

ocs = ocs.sort_values("score", ascending=False).reset_index(drop=True)
ocs.index = ocs.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
adj_tag = "🛠️ **Roster-adjusted** (OC value-add over expected)" if adjust_roster else "**Raw** (offense output, not roster-adjusted)"
if filter_current:
    st.caption(
        f"📌 Pool: **2026 current OCs only** ({len(ocs)} matched). "
        f"Mode: {adj_tag}."
    )
else:
    st.caption(
        f"📌 Pool: **actual play-callers** from `oc_team_seasons.csv` "
        f"(rebuilt from PBP — 32 in career mode, 11 in 2025-only). "
        f"Mode: {adj_tag}. "
        "Career score weights z-scores across all their play-calling seasons; "
        "2025 mode scores only the most-recent season."
    )
ranked = ocs.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("coordinator", "—"); top_score = top["score"]
    top_teams = top.get("teams", top.get("team", ""))
    sign = "+" if top_score >= 0 else ""
    seasons_val = top.get("seasons", 1)
    badge = sample_size_badge(seasons_val) if is_career else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong> ({top_teams}) {badge} &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(top_score))})</span></div>", unsafe_allow_html=True)

def _fmt_pct(x):
    return f"{x:.1%}" if pd.notna(x) else "—"
def _fmt_epa(x):
    return f"{x:+.3f}" if pd.notna(x) else "—"
def _fmt_int_comma(x):
    return f"{int(x):,}" if pd.notna(x) else "—"

def _fmt_gas(g):
    return f"{g:.1f}" if pd.notna(g) else "—"

if is_career:
    rows = []
    for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "": sample_size_badge(r.get("seasons", 1)),
            "Coordinator": r.get("coordinator", r.get("player_name", "—")),
            "Teams": r.get("teams", r.get("team", "—")),
            "Seasons": int(r["seasons"]) if pd.notna(r.get("seasons")) else 1,
            "GAS": _fmt_gas(r.get("gas_score")),
            "W-L": f"{int(r.get('total_wins', 0) or 0)}-{int(r.get('total_losses', 0) or 0)}",
            "EPA/play": _fmt_epa(r.get("epa_per_play")),
            "💰 Cap %": _fmt_pct(r.get("off_cap_pct")),
            "💰 Draft $": _fmt_int_comma(r.get("off_draft_capital")),
            "Your score": format_score(r["score"]),
        })
    display_df = pd.DataFrame(rows)
else:
    rows = []
    for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
        rows.append({
            "Rank": rank,
            "Coordinator": r.get("coordinator", r.get("player_name", "—")),
            "Team": r.get("team", "—"),
            "GAS": _fmt_gas(r.get("gas_score")),
            "EPA/play": _fmt_epa(r.get("epa_per_play")),
            "3rd down": _fmt_pct(r.get("third_down_rate")),
            "Red zone": _fmt_pct(r.get("red_zone_td_rate")),
            "Your score": format_score(r["score"]),
        })
    display_df = pd.DataFrame(rows)

st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is the score calculated?"): st.markdown(SCORE_EXPLAINER)

coord_col = "coordinator" if "coordinator" in ranked.columns else "player_name"
pool_for_lookup = ranked[coord_col].dropna().astype(str).tolist()

# Detail card moved into the team-profile section above. The rater
# section now keeps the leaderboard + a Compare utility for slicing
# the slider-weighted pool.

# ── Compare against other OCs ─────────────────────────────────
with st.expander("➕ Compare OCs (uses current slider weights + filters)", expanded=False):
    if pool_for_lookup:
        cmp_picks = st.multiselect(
            "Pick OCs to compare side-by-side",
            options=pool_for_lookup,
            default=pool_for_lookup[:1],
            key="oc_compare_picks",
            help="Each pick gets a row in the comparison table.",
        )
        if cmp_picks:
            cmp_specs = [
                ("Score", "score", lambda x: format_score(x)),
                ("Seasons", "seasons", lambda x: int(x) if pd.notna(x) else 1),
                ("EPA/play", "epa_per_play", _fmt_epa),
                ("Pass EPA", "pass_epa_per_play", _fmt_epa),
                ("Rush EPA", "rush_epa_per_play", _fmt_epa),
                ("Success%", "success_rate", _fmt_pct),
                ("Explosive pass%", "explosive_pass_rate", _fmt_pct),
                ("3rd down%", "third_down_rate", _fmt_pct),
                ("Red zone TD%", "red_zone_td_rate", _fmt_pct),
                ("Win%", "win_pct", _fmt_pct),
            ]
            cmp_rows = []
            for nm in cmp_picks:
                sub = ranked[ranked[coord_col] == nm]
                if sub.empty: continue
                r = sub.iloc[0]
                row = {"OC": nm, "Team(s)": str(r.get("teams", r.get("team", "")) or "—")}
                for label, col, fmt in cmp_specs:
                    row[label] = fmt(r.get(col))
                cmp_rows.append(row)
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
            st.caption("All numbers from the same view mode + filter set as the leaderboard above.")

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: Scheme Search — cross-sortable matrix of every OC × every dim
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def build_scheme_search_table() -> pd.DataFrame:
    """Wide one-row-per-OC matrix joining every signal we have."""
    career = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "master_ocs_with_z.parquet")
    profile = load_oc_career_profile()
    philosophy = load_oc_career_philosophy()
    fulcrum = load_oc_fulcrum_profile()
    curation = load_oc_curation()

    # Pull GAS career file for the gas_score column
    if OC_GAS_CAREER_PATH.exists():
        gas_career = pd.read_parquet(OC_GAS_CAREER_PATH)
        career = career.merge(
            gas_career[["coordinator", "gas_score", "gas_label",
                         "gas_confidence", "gas_efficiency_grade",
                         "gas_explosiveness_grade", "gas_situational_grade",
                         "gas_clutch_grade"]].drop_duplicates("coordinator"),
            on="coordinator", how="left",
        )

    base_cols_keep = [
        "coordinator", "teams", "seasons", "first_season", "last_season",
        "gas_score", "gas_label", "gas_confidence",
        "gas_efficiency_grade", "gas_explosiveness_grade",
        "gas_situational_grade", "gas_clutch_grade",
        "epa_per_play", "epa_per_play_z", "epa_per_play_adj_z",
        "pass_epa_per_play", "pass_epa_per_play_z", "pass_epa_per_play_adj_z",
        "rush_epa_per_play", "rush_epa_per_play_z", "rush_epa_per_play_adj_z",
        "success_rate", "success_rate_z", "success_rate_adj_z",
        "explosive_pass_rate", "explosive_pass_rate_z",
        "explosive_rush_rate", "explosive_rush_rate_z",
        "third_down_rate", "third_down_rate_z",
        "red_zone_td_rate", "red_zone_td_rate_z",
        "win_pct", "win_pct_z",
    ]
    have = [c for c in base_cols_keep if c in career.columns]
    base = career[have].copy().rename(columns={"coordinator": "oc_name"})

    # Pivot scheme fingerprint key signals
    key_signals = [
        ("dnd_pass_rate",        "1st_10",            "dnd_1st10_pass_z"),
        ("dnd_pass_rate",        "3rd_med",           "dnd_3rdM_pass_z"),
        ("dnd_pass_rate",        "3rd_long",          "dnd_3rdL_pass_z"),
        ("run_gap_share",        "end",               "run_end_z"),
        ("run_gap_share",        "tackle",            "run_tackle_z"),
        ("run_gap_share",        "guard",             "run_guard_z"),
        ("vs_coverage_pass_rate","vs_man",            "vs_man_pass_z"),
        ("vs_coverage_pass_rate","vs_zone",           "vs_zone_pass_z"),
        ("vs_coverage_avg_ay",   "vs_man",            "vs_man_AY_z"),
        ("vs_coverage_avg_ay",   "vs_zone",           "vs_zone_AY_z"),
        ("tempo",                "overall_no_huddle", "no_huddle_z"),
        ("tempo",                "2min_no_huddle",    "2min_no_huddle_z"),
        ("pressure_faced",       "rate_5plus",        "pressure_z"),
        ("field_pass_rate",      "rz",                "rz_pass_z"),
        ("field_pass_rate",      "gl",                "gl_pass_z"),
    ]
    if not profile.empty:
        for dim, cat, colname in key_signals:
            piv = profile[(profile["dimension"] == dim) & (profile["category"] == cat)]
            piv = piv[["oc_name", "value_z_avg"]].rename(columns={"value_z_avg": colname})
            base = base.merge(piv, on="oc_name", how="left")

    # Philosophy fit (within-OC z per archetype)
    if not philosophy.empty:
        phil_map = {"WCO": "phi_WCO_z", "Air Coryell": "phi_Coryell_z",
                    "Erhardt-Perkins": "phi_EP_z", "Spread/RPO": "phi_SpreadRPO_z",
                    "Power Run / Vertical": "phi_PowerRun_z"}
        for phi_name, colname in phil_map.items():
            piv = philosophy[philosophy["philosophy"] == phi_name][["oc_name", "fit_z_avg"]]
            piv = piv.rename(columns={"fit_z_avg": colname})
            base = base.merge(piv, on="oc_name", how="left")

    # Clutch (wp_volatility default — both metrics)
    if not fulcrum.empty:
        for metric, short in [("epa_per_play", "EPA"), ("success_rate", "succ")]:
            sub = fulcrum[(fulcrum["leverage_def"] == "wp_volatility")
                          & (fulcrum["metric"] == metric)]
            sub = sub[["oc_name", "fulcrum_z", "fulcrum_adj_z", "elevation_z"]]
            sub = sub.rename(columns={
                "fulcrum_z": f"clutch_{short}_z",
                "fulcrum_adj_z": f"clutch_{short}_adj_z",
                "elevation_z": f"clutch_{short}_elev_z",
            })
            base = base.merge(sub, on="oc_name", how="left")

    # Curation: current team + tree + mentor (match on OC name OR on HC architect name)
    if not curation.empty:
        # Build OC↔team direct map
        cur_oc = curation[["team", "oc_name", "architect_status",
                           "coaching_tree", "mentor_primary"]].rename(
            columns={"team": "current_team_oc"})
        base = base.merge(cur_oc, on="oc_name", how="left")
        # Build architect↔team map for HC=architect cases
        arch_rows = []
        for _, r in curation.iterrows():
            hc = extract_hc_architect_name(r.get("architect_status"))
            if hc:
                arch_rows.append({"oc_name": hc, "current_team_arch": r["team"],
                                  "coaching_tree_arch": r["coaching_tree"],
                                  "mentor_primary_arch": r["mentor_primary"]})
        if arch_rows:
            arch_df = pd.DataFrame(arch_rows).drop_duplicates(subset=["oc_name"])
            base = base.merge(arch_df, on="oc_name", how="left")
            # If oc_name matches as architect (HC=architect), use that team
            base["current_team"] = base["current_team_oc"].fillna(base.get("current_team_arch"))
            base["coaching_tree"] = base["coaching_tree"].fillna(base.get("coaching_tree_arch"))
            base["mentor_primary"] = base["mentor_primary"].fillna(base.get("mentor_primary_arch"))
            base = base.drop(columns=["current_team_arch", "coaching_tree_arch",
                                      "mentor_primary_arch"], errors="ignore")
        else:
            base["current_team"] = base["current_team_oc"]
        base = base.drop(columns=["current_team_oc"], errors="ignore")

    return base


def _render_scheme_search() -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 🔎 Scheme Search")
    st.caption(
        "Cross-sortable matrix of every OC × every dimension we track — career stats, "
        "scheme fingerprint, clutch performance, philosophy fit. Click any column "
        "header to sort. Filter to focus."
    )

    table = build_scheme_search_table()
    if table.empty:
        st.warning("No data available."); return

    # ── Filters row 1 ──────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        pool_choice = st.radio(
            "Pool",
            ["All historical OCs", "Current 2026 staff", "Current play-callers (HC architects)"],
            horizontal=True, key="ss_pool",
        )
    with f2:
        min_seasons = st.number_input(
            "Min seasons", min_value=1, max_value=15, value=1, step=1, key="ss_min_seasons",
        )
    with f3:
        adj_default = st.toggle(
            "🛠️ Use roster-adjusted z-scores",
            value=True, key="ss_use_adj",
            help="When on, _z columns swap to _adj_z for stats where available.",
        )

    # ── Filters row 2 ──────────────────────────────────────────
    trees = sorted({t for t in table.get("coaching_tree", pd.Series()).dropna().astype(str).unique()
                    if t.strip()})
    f4, f5 = st.columns([2, 2])
    with f4:
        tree_filter = st.multiselect(
            "Coaching tree (any selected)", options=trees, default=[], key="ss_tree")
    with f5:
        groups = st.multiselect(
            "Column groups",
            options=["Stats", "Scheme fingerprint", "Clutch", "Philosophy fit"],
            default=["Stats", "Scheme fingerprint", "Clutch"],
            key="ss_groups",
        )

    # ── Apply filters ───────────────────────────────────────────
    df = table.copy()
    if "seasons" in df.columns:
        df = df[df["seasons"].fillna(0) >= min_seasons]

    if pool_choice == "Current 2026 staff" and not curation_df.empty:
        current_oc_names = set(curation_df["oc_name"].dropna().astype(str).str.strip())
        df = df[df["oc_name"].astype(str).str.strip().isin(current_oc_names)]
    elif pool_choice == "Current play-callers (HC architects)" and not curation_df.empty:
        arch_names = set()
        for _, r in curation_df.iterrows():
            hc = extract_hc_architect_name(r.get("architect_status"))
            if hc: arch_names.add(hc.strip())
        df = df[df["oc_name"].astype(str).str.strip().isin(arch_names)]

    if tree_filter and "coaching_tree" in df.columns:
        df = df[df["coaching_tree"].astype(str).isin(tree_filter)]

    # ── Apply roster-adjusted swap ─────────────────────────────
    if adj_default:
        for c in list(df.columns):
            if c.endswith("_adj_z"):
                base = c.replace("_adj_z", "_z")
                if base in df.columns:
                    mask = df[c].notna()
                    df.loc[mask, base] = df.loc[mask, c]

    # ── Build column list per group selection ──────────────────
    core = ["oc_name", "current_team", "coaching_tree", "mentor_primary",
            "seasons", "gas_score", "gas_label"]

    stats_cols = ["epa_per_play_z", "pass_epa_per_play_z", "rush_epa_per_play_z",
                  "success_rate_z", "explosive_pass_rate_z", "third_down_rate_z",
                  "red_zone_td_rate_z", "win_pct_z"]
    scheme_cols = ["dnd_1st10_pass_z", "dnd_3rdM_pass_z", "dnd_3rdL_pass_z",
                   "run_end_z", "run_tackle_z", "run_guard_z",
                   "vs_man_pass_z", "vs_zone_pass_z", "vs_man_AY_z", "vs_zone_AY_z",
                   "no_huddle_z", "2min_no_huddle_z", "pressure_z",
                   "rz_pass_z", "gl_pass_z"]
    clutch_cols = ["clutch_EPA_z", "clutch_EPA_elev_z",
                   "clutch_succ_z", "clutch_succ_elev_z"]
    phil_cols = ["phi_WCO_z", "phi_Coryell_z", "phi_EP_z",
                 "phi_SpreadRPO_z", "phi_PowerRun_z"]

    selected = list(core)
    if "Stats" in groups: selected += stats_cols
    if "Scheme fingerprint" in groups: selected += scheme_cols
    if "Clutch" in groups: selected += clutch_cols
    if "Philosophy fit" in groups: selected += phil_cols
    selected = [c for c in selected if c in df.columns]
    display = df[selected].copy()

    # ── Friendly column labels + sort default ───────────────────
    rename_map = {
        "oc_name": "OC", "current_team": "Team", "coaching_tree": "Tree",
        "mentor_primary": "Mentor", "seasons": "Yrs",
        "gas_score": "GAS", "gas_label": "Tier",
        "epa_per_play_z": "EPA z", "pass_epa_per_play_z": "Pass EPA z",
        "rush_epa_per_play_z": "Rush EPA z", "success_rate_z": "Succ% z",
        "explosive_pass_rate_z": "Expl pass z", "third_down_rate_z": "3rd dn z",
        "red_zone_td_rate_z": "RZ TD z", "win_pct_z": "Win% z",
        "dnd_1st10_pass_z": "Pass z 1st&10", "dnd_3rdM_pass_z": "Pass z 3rd&med",
        "dnd_3rdL_pass_z": "Pass z 3rd&long",
        "run_end_z": "Run end z", "run_tackle_z": "Run tackle z",
        "run_guard_z": "Run guard z",
        "vs_man_pass_z": "vs Man pass z", "vs_zone_pass_z": "vs Zone pass z",
        "vs_man_AY_z": "vs Man AY z", "vs_zone_AY_z": "vs Zone AY z",
        "no_huddle_z": "No-huddle z", "2min_no_huddle_z": "2-min NH z",
        "pressure_z": "Pressure faced z",
        "rz_pass_z": "RZ pass z", "gl_pass_z": "GL pass z",
        "clutch_EPA_z": "Clutch EPA z", "clutch_EPA_elev_z": "Clutch EPA elev z",
        "clutch_succ_z": "Clutch succ z", "clutch_succ_elev_z": "Clutch succ elev z",
        "phi_WCO_z": "WCO fit z", "phi_Coryell_z": "Coryell fit z",
        "phi_EP_z": "EP fit z", "phi_SpreadRPO_z": "Spread/RPO fit z",
        "phi_PowerRun_z": "Power Run fit z",
    }
    display = display.rename(columns=rename_map)

    if "GAS" in display.columns:
        display = display.sort_values("GAS", ascending=False, na_position="last")
    elif "EPA z" in display.columns:
        display = display.sort_values("EPA z", ascending=False)

    # ── Column config: heatmap z-score columns (-3 to +3 progress bar) ─
    col_config = {}
    for c in display.columns:
        if c.endswith(" z") or "z" in c.split()[-1:]:
            col_config[c] = st.column_config.ProgressColumn(
                label=c, format="%+.2f", min_value=-3.0, max_value=3.0,
                help=f"Z-score column. Above 0 = above league average."
            )
    if "GAS" in display.columns:
        col_config["GAS"] = st.column_config.ProgressColumn(
            label="GAS", format="%.1f", min_value=0.0, max_value=100.0,
            help="OC GAS Score: 0-100 composite. 50 = league average. "
                 "Bundles: Efficiency 45%, Explosiveness 15%, Situational 20%, Clutch 20%.",
        )
    if "Yrs" in display.columns:
        col_config["Yrs"] = st.column_config.NumberColumn("Yrs", format="%d")

    st.markdown(f"**{len(display)} OCs match.**")
    st.dataframe(display, use_container_width=True, hide_index=True,
                 column_config=col_config, height=600)

    # ── Quick-pick example sorts ──────────────────────────────
    st.markdown("##### Quick-pick views")
    with st.expander("💡 Example questions you can answer"):
        st.markdown("""
- **"Most pass-happy on 1st & 10"** → sort by `Pass z 1st&10` desc
- **"Most run-pass-balanced clutch performers"** → sort by `Clutch EPA elev z` desc, filter to "Current play-callers"
- **"Strongest WCO offenses"** → sort by `WCO fit z` desc
- **"Best clutch EPA, lowest cap %"** → sort by `Clutch EPA z` desc, look at Stats columns alongside
- **"Outside-zone teams"** → sort by `Run tackle z` desc
- **"Up-tempo offenses"** → sort by `No-huddle z` desc
- **"Vertical passing offenses"** → sort by `vs Zone AY z` desc and `Coryell fit z` desc
        """)


_render_scheme_search()


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • 2016-2024 regular seasons • Coordinator tenures manually compiled • ⚠️ Stats reflect the entire offensive unit, not the coordinator in isolation • Fan project, not affiliated with the NFL.")
