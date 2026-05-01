"""Movement & Athleticism + Advanced Tracking panels for NFL pages.

Two surfaces, same data file:
  • render_movement_panel — combine + pro-day measurables only.
    Pre-NFL "tested" athleticism (40 time, vert, broad, shuttle, cone).
  • render_advanced_tracking — in-game NextGenStats production
    metrics. Lives alongside the player's passing/rushing/receiving
    production stats, not in the athleticism section.

Both skip silently for positions where the underlying data isn't
published (NGS for OL / defense, combine for older / undrafted).

UI label is "Advanced Tracking" — generic descriptor, not the
NFL-trademarked term.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"


# ── Data loaders ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_workouts() -> pd.DataFrame:
    p = DATA / "college" / "nfl_all_workouts.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_ngs(stat_type: str) -> pd.DataFrame:
    p = DATA / f"ngs_{stat_type}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# Volume thresholds matching the existing rater's "starter cohort"
# (the same population we z-score the master parquets against).
_NGS_COHORT_QUALIFIER = {
    "passing":   ("attempts",      100),
    "rushing":   ("rush_attempts",  75),
    "receiving": ("targets",        50),
}


@st.cache_data(show_spinner=False)
def _ngs_cohort(stat_type: str, season: int) -> pd.DataFrame:
    """Season-aggregate rows for qualifying starters at the
    position. Used to compute league percentiles for tracking metrics."""
    df = _load_ngs(stat_type)
    if df.empty:
        return df
    qual_col, qual_min = _NGS_COHORT_QUALIFIER.get(
        stat_type, ("attempts", 0))
    cohort = df[(df["season"] == season) & (df["week"] == 0)].copy()
    if qual_col in cohort.columns:
        cohort = cohort[cohort[qual_col] >= qual_min]
    return cohort


@st.cache_data(show_spinner=False)
def _combine_cohort(position: str) -> pd.DataFrame:
    """All historical combine entries at the player's position. The
    cohort is multi-year because combine data is one-time per player."""
    df = _load_workouts()
    if df.empty or "pos" not in df.columns:
        return df
    return df[df["pos"] == position.upper()].copy()


def _percentile(value, cohort_values, direction: str) -> int | None:
    """Returns the player's percentile rank within the cohort.
       direction='high' — higher is better (pctl = % below this value)
       direction='low'  — lower is better  (pctl = % above this value)
       direction='neutral' — just shows raw rank by value
    Returns None when value is missing or cohort is too small."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    vals = [v for v in cohort_values
            if v is not None
            and not (isinstance(v, float) and pd.isna(v))]
    if len(vals) < 5:
        return None
    if direction == "low":
        # Lower value → higher percentile (e.g., faster 40 time = elite)
        below = sum(1 for v in vals if v >= value)
    else:
        below = sum(1 for v in vals if v <= value)
    return int(round(100 * below / len(vals)))


def _pctl_label(pctl: int | None, suffix: str = "") -> str:
    if pctl is None:
        return ""
    icon = "🟢" if pctl >= 75 else ("🔴" if pctl <= 25 else "⚪")
    suffix_str = f" {suffix}" if suffix else ""
    return f"{icon} {pctl}th pctl{suffix_str}"


def _combine_row(player_name: str) -> dict | None:
    df = _load_workouts()
    if df.empty:
        return None
    m = df[df["player_name"] == player_name]
    if m.empty:
        return None
    # Prefer combine over pro day when both exist
    if "source" in m.columns and (m["source"] == "combine").any():
        m = m[m["source"] == "combine"]
    return dict(m.iloc[0])


def _ngs_season_row(player_name: str, stat_type: str,
                      season: int) -> dict | None:
    df = _load_ngs(stat_type)
    if df.empty:
        return None
    # week=0 is the season aggregate row in NGS data
    m = df[(df["player_display_name"] == player_name)
           & (df["season"] == season) & (df["week"] == 0)]
    if m.empty:
        return None
    return dict(m.iloc[0])


# ── Tile rendering ──────────────────────────────────────────────

def _tile(label: str, value: str, pctl_str: str = "",
            sub: str = "") -> str:
    """One stat tile. `value` is the headline number (big),
    `pctl_str` is rendered in parens after it (smaller, inline),
    `sub` is the descriptive note explaining what the metric means."""
    pctl_html = (
        f' <span style="font-size:13px; font-weight:600; '
        f'opacity:0.65;">({pctl_str})</span>' if pctl_str else ""
    )
    sub_html = (
        f'<div style="font-size:11px; opacity:0.7; '
        f'margin-top:4px;">{sub}</div>' if sub else ""
    )
    return (
        f'<div style="background:#f6f7fa; border-radius:10px; '
        f'padding:12px 14px; flex:1; min-width:140px;">'
        f'<div style="font-size:10px; font-weight:700; '
        f'letter-spacing:1.5px; opacity:0.55;">{label}</div>'
        f'<div style="font-size:22px; font-weight:800; '
        f'margin-top:4px; line-height:1.1;">'
        f'{value}{pctl_html}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _pctl_text(pctl: int | None) -> str:
    """Plain-English percentile label without an icon."""
    if pctl is None:
        return ""
    suffix = "th"
    if pctl % 100 < 11 or pctl % 100 > 13:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(pctl % 10, "th")
    return f"{pctl}{suffix} pctl"


def _fmt(v, fmt: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def _height_str(height_in) -> str:
    if height_in is None or (isinstance(height_in, float)
                                and pd.isna(height_in)):
        return "—"
    try:
        h = int(height_in)
        return f"{h//12}'{h%12}\""
    except (ValueError, TypeError):
        return str(height_in)


# ── Position-specific NGS tile mappings ─────────────────────────

_NGS_STAT_TYPE_BY_POS = {
    "qb": "passing",
    "wr": "receiving",
    "te": "receiving",
    "rb": "rushing",
}


# Tile specs: (LABEL, field, fmt_str, direction, descriptive_sub).
# direction: "high"=higher is better, "low"=lower, "neutral"=raw rank
_NGS_TILE_SPEC = {
    "qb": [
        ("TIME TO THROW",     "avg_time_to_throw",
         "{:.2f}s",  "low",     "from snap to release"),
        ("INTENDED AIR",      "avg_intended_air_yards",
         "{:.1f}",   "high",    "yds per attempt"),
        ("AIR DIFFERENTIAL",  "avg_air_yards_differential",
         "{:+.1f}",  "neutral", "completed minus intended"),
        ("AGGRESSIVENESS",    "aggressiveness",
         "{:.1f}%",  "neutral", "tight-window throw rate"),
        ("CPOE",              "completion_percentage_above_expectation",
         "{:+.1f}%", "high",    "completion % over expected"),
        ("MAX AIR",           "max_completed_air_distance",
         "{:.0f}",   "high",    "longest completed pass air-distance"),
    ],
    "rb": [
        ("EFFICIENCY",        "efficiency",
         "{:.2f}",   "low",     "yards traveled per yard gained"),
        ("TIME TO LOS",       "avg_time_to_los",
         "{:.2f}s",  "low",     "snap to crossing line of scrimmage"),
        ("RYOE / ATT",        "rush_yards_over_expected_per_att",
         "{:+.2f}",  "high",    "rush yds over expected per attempt"),
        ("RYOE TOTAL",        "rush_yards_over_expected",
         "{:+.0f}",  "high",    "season total over expected"),
        ("8+ DEFENDERS",      "percent_attempts_gte_eight_defenders",
         "{:.1f}%",  "neutral", "% carries vs stacked box"),
        ("RYOE %",            "rush_pct_over_expected",
         "{:+.1%}",  "high",    "% attempts gaining over expected"),
    ],
    "wr": [
        ("AVG SEPARATION",    "avg_separation",
         "{:.2f}",   "high",    "yards from nearest defender at catch"),
        ("AVG CUSHION",       "avg_cushion",
         "{:.2f}",   "neutral", "yards off DB at snap"),
        ("YAC OVER EXPECTED", "avg_yac_above_expectation",
         "{:+.2f}",  "high",    "yards after catch above model"),
        ("AVG YAC",           "avg_yac",
         "{:.2f}",   "high",    "yards after catch per reception"),
        ("INTENDED AIR YDS",  "avg_intended_air_yards",
         "{:.1f}",   "high",    "yds per target"),
        ("TARGET SHARE",      "percent_share_of_intended_air_yards",
         "{:.1f}%",  "high",    "% of team intended air yards"),
    ],
    "te": [
        ("AVG SEPARATION",    "avg_separation",
         "{:.2f}",   "high",    "yards from nearest defender at catch"),
        ("AVG CUSHION",       "avg_cushion",
         "{:.2f}",   "neutral", "yards off DB at snap"),
        ("YAC OVER EXPECTED", "avg_yac_above_expectation",
         "{:+.2f}",  "high",    "yards after catch above model"),
        ("AVG YAC",           "avg_yac",
         "{:.2f}",   "high",    "yards after catch per reception"),
        ("INTENDED AIR YDS",  "avg_intended_air_yards",
         "{:.1f}",   "high",    "yds per target"),
        ("TARGET SHARE",      "percent_share_of_intended_air_yards",
         "{:.1f}%",  "high",    "% of team intended air yards"),
    ],
}


# (LABEL, field, fmt, direction, descriptive_sub)
_COMBINE_TILE_SPEC = [
    ("HEIGHT",       "height_in",  "{}",       "neutral", ""),
    ("40 YARD",      "forty",      "{:.2f}s",  "low",
        "combine 40-yard dash"),
    ("VERTICAL",     "vertical",   "{:.1f}\"", "high",
        "vertical jump"),
    ("BROAD JUMP",   "broad_jump", "{:.0f}\"", "high",
        "standing broad jump"),
    ("3-CONE",       "cone",       "{:.2f}s",  "low",
        "agility 3-cone drill"),
    ("20-YD SHUTTLE","shuttle",    "{:.2f}s",  "low",
        "lateral quickness"),
]


def _ngs_tiles(row: dict, position: str, season: int
                ) -> list[tuple[str, str, str, str]]:
    """Returns (LABEL, value, pctl_str, descriptive_sub) tuples."""
    pos = position.lower()
    spec = _NGS_TILE_SPEC.get(pos, [])
    if not spec:
        return []
    stat_type = _NGS_STAT_TYPE_BY_POS.get(pos)
    cohort = (_ngs_cohort(stat_type, season)
              if stat_type else pd.DataFrame())

    out = []
    for label, field, fmt, direction, sub in spec:
        raw = row.get(field)
        val = _fmt(raw, fmt)
        if cohort.empty or field not in cohort.columns:
            pctl_str = ""
        else:
            pctl = _percentile(raw, cohort[field].tolist(), direction)
            pctl_str = _pctl_text(pctl)
        out.append((label, val, pctl_str, sub))
    return out


def _combine_tiles(row: dict | None, position: str
                     ) -> list[tuple[str, str, str, str]]:
    if not row:
        return []
    cohort = _combine_cohort(position)

    out = []
    for label, field, fmt, direction, sub in _COMBINE_TILE_SPEC:
        if label == "HEIGHT":
            val = _height_str(row.get("height_in") or row.get("ht"))
            wt = row.get("weight")
            sub_render = f"{int(wt)} lbs" if wt and not pd.isna(wt) else ""
            out.append((label, val, "", sub_render))
            continue
        raw = row.get(field)
        val = _fmt(raw, fmt)
        if cohort.empty or field not in cohort.columns:
            pctl_str = ""
        else:
            pctl = _percentile(raw, cohort[field].tolist(), direction)
            pctl_str = _pctl_text(pctl)
        out.append((label, val, pctl_str, sub))
    return out


# ── Public entrypoint ───────────────────────────────────────────

# ── Defensive + OL production specs ─────────────────────────────
# Stats already in our league_*_all_seasons.parquet — built from PFR
# play-by-play + nflfastR. Same (LABEL, field, fmt, direction, sub)
# tuple format as the NGS specs above.

_DEFENSIVE_PRODUCTION_SPEC = {
    "de": [
        ("SACKS / G",       "sacks_per_game",         "{:.2f}",   "high",
            "sacks per game"),
        ("PRESSURES / G",   "pressures_per_game",     "{:.1f}",   "high",
            "pressures per game (sacks + hurries + hits)"),
        ("PRESSURE %",      "pressure_rate",          "{:.1%}",   "high",
            "% of pass plays generating pressure"),
        ("HURRIES / G",     "hurries_per_game",       "{:.1f}",   "high",
            "hurries per game"),
        ("QB HITS / G",     "qb_hits_per_game",       "{:.1f}",   "high",
            "QB hits per game"),
        ("TFL / G",         "tfl_per_game",           "{:.1f}",   "high",
            "tackles for loss per game"),
        ("MISSED TKL %",    "missed_tackle_pct",      "{:.1%}",   "low",
            "% of tackle attempts missed"),
    ],
    "dt": [
        ("SACKS / G",       "sacks_per_game",         "{:.2f}",   "high",
            "sacks per game"),
        ("PRESSURES / G",   "pressures_per_game",     "{:.1f}",   "high",
            "pressures per game"),
        ("PRESSURE %",      "pressure_rate",          "{:.1%}",   "high",
            "% of pass plays generating pressure"),
        ("HURRIES / G",     "hurries_per_game",       "{:.1f}",   "high",
            "hurries per game"),
        ("QB HITS / G",     "qb_hits_per_game",       "{:.1f}",   "high",
            "QB hits per game"),
        ("TFL / G",         "tfl_per_game",           "{:.1f}",   "high",
            "tackles for loss per game"),
        ("MISSED TKL %",    "missed_tackle_pct",      "{:.1%}",   "low",
            "% of tackle attempts missed"),
    ],
    "lb": [
        ("TACKLES / G",     "tackles_per_game",       "{:.1f}",   "high",
            "tackles per game"),
        ("SOLO TKL RATE",   "solo_tackle_rate",       "{:.1%}",   "high",
            "% of tackles unassisted"),
        ("TFL / G",         "tfl_per_game",           "{:.2f}",   "high",
            "tackles for loss per game"),
        ("SACKS / G",       "sacks_per_game",         "{:.2f}",   "high",
            "sacks per game"),
        ("PRESSURES / G",   "pressures_per_game",     "{:.1f}",   "high",
            "pressures per game"),
        ("PD / G",          "passes_defended_per_game", "{:.2f}", "high",
            "passes defended per game"),
        ("INT / G",         "interceptions_per_game", "{:.2f}",   "high",
            "interceptions per game"),
        ("MISSED TKL %",    "missed_tackle_pct",      "{:.1%}",   "low",
            "% of tackle attempts missed"),
        ("CMP % ALLOWED",   "completion_pct_allowed", "{:.1%}",   "low",
            "completions allowed when targeted"),
        ("RTG ALLOWED",     "passer_rating_allowed",  "{:.1f}",   "low",
            "passer rating in coverage"),
    ],
    "cb": [
        ("TARGETS / G",     "targets_per_game",       "{:.1f}",   "neutral",
            "times targeted per game"),
        ("CMP % ALLOWED",   "completion_pct_allowed", "{:.1%}",   "low",
            "completions allowed in coverage"),
        ("YDS / TGT ALLOWED","yards_per_target_allowed","{:.1f}", "low",
            "yards per target allowed"),
        ("RTG ALLOWED",     "passer_rating_allowed",  "{:.1f}",   "low",
            "passer rating into his coverage"),
        ("PD / G",          "passes_defended_per_game", "{:.2f}", "high",
            "passes defended per game"),
        ("INT / G",         "interceptions_per_game", "{:.2f}",   "high",
            "interceptions per game"),
        ("MISSED TKL %",    "missed_tackle_pct",      "{:.1%}",   "low",
            "% of tackle attempts missed"),
        ("AVG DEPTH OF TGT","avg_depth_of_target",    "{:.1f}",   "neutral",
            "average depth of target (deeper = bigger plays in/out)"),
    ],
    "s": [
        ("TACKLES / SNAP",  "tackles_per_snap",       "{:.3f}",   "high",
            "tackle rate per defensive snap"),
        ("TFL / G",         "tfl_per_game",           "{:.2f}",   "high",
            "tackles for loss per game"),
        ("PD / G",          "passes_defended_per_game", "{:.2f}", "high",
            "passes defended per game"),
        ("INT / G",         "interceptions_per_game", "{:.2f}",   "high",
            "interceptions per game"),
        ("FF / G",          "forced_fumbles_per_game", "{:.2f}",  "high",
            "forced fumbles per game"),
        ("CMP % ALLOWED",   "completion_pct_allowed", "{:.1%}",   "low",
            "completions allowed in coverage"),
        ("RTG ALLOWED",     "passer_rating_allowed",  "{:.1f}",   "low",
            "passer rating in coverage"),
        ("MISSED TKL %",    "missed_tackle_pct",      "{:.1%}",   "low",
            "% of tackle attempts missed"),
    ],
}

_OL_PRODUCTION_SPEC = [
    ("RUN-PLAY EPA",    "pos_run_epa",         "{:+.3f}", "high",
        "EPA on rushes this lineman blocked on"),
    ("RUN SUCCESS %",   "pos_run_success",     "{:.1%}",  "high",
        "% of run plays with positive EPA"),
    ("EXPLOSIVE RUN %", "pos_run_explosive",   "{:.1%}",  "high",
        "% of runs gaining 10+ yards"),
    ("TEAM SACK %",     "team_sack_rate",      "{:.1%}",  "low",
        "% of dropbacks ending in sack while this player on field"),
    ("TEAM PRESSURE %", "team_pressure_rate",  "{:.1%}",  "low",
        "% of dropbacks under pressure while this player on field"),
    ("PENALTIES / G",   "penalty_rate",        "{:.2f}",  "low",
        "this player's penalties per game"),
]


def _production_tiles(player_row, position: str, cohort_df,
                         season: int) -> list[tuple[str, str, str, str]]:
    """Build (LABEL, value, pctl_str, sub) tiles for a defender or OL,
    using stats already in the position's league parquet."""
    pos = position.lower()
    spec = (_DEFENSIVE_PRODUCTION_SPEC.get(pos)
            or (_OL_PRODUCTION_SPEC if pos == "ol" else []))
    if not spec or cohort_df is None or cohort_df.empty:
        return []
    # Cohort: same season + minimum-snap qualifier so we're comparing
    # against starters, not bench warmers.
    season_col = "season_year" if "season_year" in cohort_df.columns \
        else "season"
    snap_col = ("def_snaps" if "def_snaps" in cohort_df.columns
                else ("snap_share" if "snap_share" in cohort_df.columns
                       else None))
    cohort = cohort_df[cohort_df[season_col] == season].copy()
    if snap_col == "def_snaps":
        cohort = cohort[cohort["def_snaps"].fillna(0) >= 200]
    elif snap_col == "snap_share":
        cohort = cohort[cohort["snap_share"].fillna(0) >= 0.4]

    out = []
    for label, field, fmt, direction, sub in spec:
        raw = player_row.get(field) if hasattr(player_row, "get") \
            else (player_row[field] if field in player_row.index
                   else None)
        val = _fmt(raw, fmt)
        if cohort.empty or field not in cohort.columns:
            pctl_str = ""
        else:
            pctl = _percentile(raw, cohort[field].tolist(), direction)
            pctl_str = _pctl_text(pctl)
        out.append((label, val, pctl_str, sub))
    return out


def render_advanced_production(player_row, position: str,
                                  cohort_df, season: int = 2025) -> None:
    """Defender / OL production panel using existing PFR-derived
    stats from the position parquet. Skips silently when there's no
    spec for this position or no data."""
    pos = position.lower()
    spec_exists = (pos in _DEFENSIVE_PRODUCTION_SPEC) or pos == "ol"
    if not spec_exists:
        return

    title_by_pos = {
        "de": "📡  Advanced Production — Edge Rushing",
        "dt": "📡  Advanced Production — Interior",
        "lb": "📡  Advanced Production — Linebacker",
        "cb": "📡  Advanced Production — Cornerback",
        "s":  "📡  Advanced Production — Safety",
        "ol": "📡  Team-Context Impact — Offensive Line",
    }
    cohort_label = {
        "de": "qualified edge rushers (200+ defensive snaps)",
        "dt": "qualified interior linemen (200+ defensive snaps)",
        "lb": "qualified linebackers (200+ defensive snaps)",
        "cb": "qualified cornerbacks (200+ defensive snaps)",
        "s":  "qualified safeties (200+ defensive snaps)",
        "ol": "qualified offensive linemen (40%+ snap share)",
    }
    tiles = _production_tiles(player_row, pos, cohort_df, season)
    if not tiles:
        return

    st.markdown(f"### {title_by_pos[pos]}")
    if pos == "ol":
        st.caption(
            "Team-context impact metrics — what happened on plays "
            "this lineman was on the field for. Percentiles compare "
            f"to {cohort_label[pos]} this season."
        )
    else:
        st.caption(
            "Pressure, coverage, and tackling production from "
            "play-by-play charting (sourced through nflverse / PFR). "
            "Percentiles compare to "
            f"{cohort_label[pos]} this season."
        )

    tile_html = (
        '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        + "".join(_tile(*t) for t in tiles)
        + '</div>'
    )
    st.markdown(tile_html, unsafe_allow_html=True)


def render_movement_panel(player_name: str, position: str,
                            season: int = 2025) -> None:
    """Athleticism panel — combine + pro-day measurables only.
    Pre-NFL tested athleticism. Skips silently when no data."""
    combine = _combine_row(player_name)
    if not combine:
        return

    st.markdown("### 🏃  Athleticism — Combine + Pro Day")
    st.caption(
        "Pre-draft tested measurables. The player's physical "
        "ceiling, captured before he ever played an NFL snap. "
        f"Percentiles compare to all {position.upper()} entries "
        "in the combine archive."
    )

    tiles = _combine_tiles(combine, position)
    tile_html = (
        '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        + "".join(_tile(*t) for t in tiles)
        + '</div>'
    )
    st.markdown(tile_html, unsafe_allow_html=True)


def render_advanced_tracking(player_name: str, position: str,
                                season: int = 2025) -> None:
    """Advanced Tracking — in-game tracking metrics for the
    player's primary stat category (passing / rushing / receiving).
    Generic UI label avoids the NFL Next Gen Stats trademark.
    Silently no-ops when no data."""
    pos = position.lower()
    stat_type = _NGS_STAT_TYPE_BY_POS.get(pos)
    if not stat_type:
        return
    ngs = _ngs_season_row(player_name, stat_type, season)
    if not ngs:
        return

    title_by_stat = {
        "passing":   "📡  Advanced Tracking — Passing",
        "rushing":   "📡  Advanced Tracking — Rushing",
        "receiving": "📡  Advanced Tracking — Receiving",
    }
    cohort_label = {
        "passing":   "qualified starting QBs (100+ attempts)",
        "rushing":   "qualified RBs (75+ rush attempts)",
        "receiving": "qualified pass-catchers (50+ targets)",
    }
    st.markdown(f"### {title_by_stat[stat_type]}")
    st.caption(
        "Tracking metrics from radar and optical sensors capturing "
        "every player on every play. These measure things the box "
        "score misses — release quickness, separation at the catch, "
        f"decisiveness through the line. Percentiles compare to "
        f"{cohort_label[stat_type]} this season."
    )

    tiles = _ngs_tiles(ngs, pos, season)
    tile_html = (
        '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        + "".join(_tile(*t) for t in tiles)
        + '</div>'
    )
    st.markdown(tile_html, unsafe_allow_html=True)
    st.caption(
        "_Tracking data via NFL Next Gen Stats, sourced through the "
        "nflverse community project._"
    )
