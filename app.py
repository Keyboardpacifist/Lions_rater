"""
NFL Rater — Landing page with NFL / College toggle
Enriched college profiles with recruiting, usage, adjusted metrics, transfers.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="NFL Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"

# ── Mode toggle ───────────────────────────────────────────────
col_title, col_toggle = st.columns([3, 2])
with col_title:
    st.markdown("<h1 style='margin:0; padding:4px 0;'>🏈 NFL Rater</h1>", unsafe_allow_html=True)
with col_toggle:
    mode = st.radio("Mode", ["NFL", "College"], horizontal=True, key="mode_toggle", label_visibility="collapsed")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# NFL MODE
# ══════════════════════════════════════════════════════════════
if mode == "NFL":
    from team_selector import (
        NFL_TEAMS,
        LEAGUE_WIDE_KEY,
        LEAGUE_WIDE_LABEL,
        display_abbr,
        internal_abbr,
    )

    if "selected_team" not in st.session_state:
        st.session_state.selected_team = LEAGUE_WIDE_KEY
    if "selected_season" not in st.session_state:
        st.session_state.selected_season = 2025

    # League-wide first, then sorted real teams (matches the per-page selector).
    real_teams = sorted(k for k in NFL_TEAMS.keys() if k != LEAGUE_WIDE_KEY)
    team_options = [LEAGUE_WIDE_KEY] + real_teams
    team_labels = [LEAGUE_WIDE_LABEL] + [
        f"{display_abbr(abbr)} — {NFL_TEAMS[abbr]}" for abbr in real_teams
    ]
    current_idx = team_options.index(st.session_state.selected_team) if st.session_state.selected_team in team_options else 0
    AVAILABLE_SEASONS = list(range(2025, 2015, -1))

    col_team, col_season, col_spacer = st.columns([2, 1, 3])
    with col_team:
        selected_label = st.selectbox("Team", options=team_labels, index=current_idx,
                                       key="landing_team_v2", label_visibility="collapsed")
    with col_season:
        selected_season = st.selectbox("Season", options=AVAILABLE_SEASONS,
                                        index=0, key="landing_season", label_visibility="collapsed")

    if selected_label == LEAGUE_WIDE_LABEL:
        selected_team = LEAGUE_WIDE_KEY
    else:
        selected_team = internal_abbr(selected_label.split(" — ")[0])
    st.session_state.selected_team = selected_team
    st.session_state.selected_season = selected_season
    team_name = NFL_TEAMS.get(selected_team, selected_team)

    # ── Position + metric pickers (inline leaderboard preview) ─────
    import polars as pl
    from team_selector import filter_by_team_and_season
    from lib_shared import metric_picker

    NFL_POSITION_CONFIGS = {
        "QB": {
            "file": "league_qb_all_seasons.parquet",
            "filter": None,
            "snap_col": "off_snaps",
            "min_snaps": 100,
            "noun": "quarterbacks",
            "cols": [("Player", "player_display_name"),
                     ("Team", "recent_team"),
                     ("Att", "attempts"),
                     ("Yds", "passing_yards"),
                     ("TDs", "passing_tds"),
                     ("INT", "passing_interceptions"),
                     ("EPA/play", "pass_epa_per_play"),
                     ("CPOE", "passing_cpoe"),
                     ("Success%", "pass_success_rate")],
            "metrics": {
                "Passing yards": ("passing_yards", False),
                "Passing TDs": ("passing_tds", False),
                "EPA per play": ("pass_epa_per_play", False),
                "CPOE": ("passing_cpoe", False),
                "Pass success rate": ("pass_success_rate", False),
                "Yards per attempt": ("yards_per_attempt", False),
                "Completion %": ("completion_pct", False),
                "INT rate (lower better)": ("int_rate", True),
                "Sack rate (lower better)": ("sack_rate", True),
                "Turnover rate (lower better)": ("turnover_rate", True),
            },
        },
        "WR": {
            "file": "league_wr_all_seasons.parquet",
            "filter": ("position", "WR"),
            "snap_col": "off_snaps",
            "min_snaps": 100,
            "noun": "wide receivers",
            "cols": [("Player", "player_display_name"),
                     ("Team", "recent_team"),
                     ("Snaps", "off_snaps"),
                     ("Rec", "receptions"),
                     ("Yds", "rec_yards"),
                     ("TDs", "rec_tds"),
                     ("Tgt%", "target_share"),
                     ("EPA/tgt", "epa_per_target"),
                     ("YAC/exp", "yac_above_exp")],
            "metrics": {
                "Receiving yards": ("rec_yards", False),
                "Receptions": ("receptions", False),
                "TDs": ("rec_tds", False),
                "Target share": ("target_share", False),
                "EPA per target": ("epa_per_target", False),
                "Yards per target": ("yards_per_target", False),
                "YAC over expected": ("yac_above_exp", False),
                "Catch rate": ("catch_rate", False),
                "WOPR (opportunity)": ("wopr", False),
                "Average separation (NGS)": ("avg_separation", False),
            },
        },
        "TE": {
            "file": "league_te_all_seasons.parquet",
            "filter": ("position", "TE"),
            "snap_col": "off_snaps",
            "min_snaps": 100,
            "noun": "tight ends",
            "cols": [("Player", "player_display_name"),
                     ("Team", "recent_team"),
                     ("Snaps", "off_snaps"),
                     ("Rec", "receptions"),
                     ("Yds", "rec_yards"),
                     ("TDs", "rec_tds"),
                     ("Tgt%", "target_share"),
                     ("EPA/tgt", "epa_per_target")],
            "metrics": {
                "Receiving yards": ("rec_yards", False),
                "Receptions": ("receptions", False),
                "TDs": ("rec_tds", False),
                "Target share": ("target_share", False),
                "EPA per target": ("epa_per_target", False),
                "Yards per target": ("yards_per_target", False),
                "YAC over expected": ("yac_above_exp", False),
                "Catch rate": ("catch_rate", False),
            },
        },
        "RB": {
            "file": "league_rb_all_seasons.parquet",
            "filter": None,
            "snap_col": "off_snaps",
            "min_snaps": 100,
            "noun": "running backs",
            "cols": [("Player", "player_display_name"),
                     ("Team", "recent_team"),
                     ("Att", "carries"),
                     ("Yds", "rush_yards"),
                     ("TDs", "rush_tds"),
                     ("Rec", "receptions"),
                     ("YPC", "yards_per_carry"),
                     ("EPA/rush", "epa_per_rush"),
                     ("YACO/att", "yards_after_contact_per_att")],
            "metrics": {
                "Rushing yards": ("rush_yards", False),
                "Rushing TDs": ("rush_tds", False),
                "Yards per carry": ("yards_per_carry", False),
                "EPA per rush": ("epa_per_rush", False),
                "YACO per attempt": ("yards_after_contact_per_att", False),
                "Broken tackles per att": ("broken_tackles_per_att", False),
                "RYOE per attempt (NGS)": ("ryoe_per_att", False),
                "Snap share": ("snap_share", False),
                "Touches per game": ("touches_per_game", False),
            },
        },
        "DE": {
            "file": "league_de_all_seasons.parquet",
            "filter": None,
            "snap_col": "def_snaps",
            "min_snaps": 100,
            "noun": "defensive ends",
            "cols": [("Player", "player_name"),
                     ("Team", "recent_team"),
                     ("Snaps", "def_snaps"),
                     ("Sacks", "def_sacks"),
                     ("QB hits", "def_qb_hits"),
                     ("TFL", "def_tackles_for_loss"),
                     ("Pressures", "pfr_pressures"),
                     ("Press rate", "pressure_rate")],
            "metrics": {
                "Sacks": ("def_sacks", False),
                "QB hits": ("def_qb_hits", False),
                "Tackles for loss": ("def_tackles_for_loss", False),
                "Pressures (PFR)": ("pfr_pressures", False),
                "Pressure rate": ("pressure_rate", False),
                "Sacks per game": ("sacks_per_game", False),
                "Forced fumbles per game": ("forced_fumbles_per_game", False),
            },
        },
        "DT": {
            "file": "league_dt_all_seasons.parquet",
            "filter": None,
            "snap_col": "def_snaps",
            "min_snaps": 100,
            "noun": "defensive tackles",
            "cols": [("Player", "player_name"),
                     ("Team", "recent_team"),
                     ("Snaps", "def_snaps"),
                     ("Sacks", "def_sacks"),
                     ("TFL", "def_tackles_for_loss"),
                     ("QB hits", "def_qb_hits"),
                     ("Pressures", "pfr_pressures"),
                     ("Press rate", "pressure_rate")],
            "metrics": {
                "Sacks": ("def_sacks", False),
                "QB hits": ("def_qb_hits", False),
                "Tackles for loss": ("def_tackles_for_loss", False),
                "Pressures (PFR)": ("pfr_pressures", False),
                "Pressure rate": ("pressure_rate", False),
                "Sacks per game": ("sacks_per_game", False),
            },
        },
        "LB": {
            "file": "league_lb_all_seasons.parquet",
            "filter": None,
            "snap_col": "def_snaps",
            "min_snaps": 100,
            "noun": "linebackers",
            "cols": [("Player", "player_name"),
                     ("Team", "recent_team"),
                     ("Snaps", "def_snaps"),
                     ("Solo Tkl", "def_tackles_solo"),
                     ("TFL", "def_tackles_for_loss"),
                     ("Sacks", "def_sacks"),
                     ("INT", "def_interceptions"),
                     ("PD", "def_pass_defended"),
                     ("Missed Tkl%", "pfr_missed_tackle_pct")],
            "metrics": {
                "Solo tackles": ("def_tackles_solo", False),
                "Tackles for loss": ("def_tackles_for_loss", False),
                "Sacks": ("def_sacks", False),
                "Interceptions": ("def_interceptions", False),
                "Passes defended": ("def_pass_defended", False),
                "Tackles per snap": ("tackles_per_snap", False),
                "Missed tackle % (lower better)": ("pfr_missed_tackle_pct", True),
            },
        },
        "Punter": {
            "file": "league_p_all_seasons.parquet",
            "filter": None,
            "snap_col": "off_snaps",
            "min_snaps": 0,
            "noun": "punters",
            "cols": [("Player", "player_display_name"),
                     ("Team", "recent_team"),
                     ("Punts", "punts"),
                     ("Gross avg", "avg_distance"),
                     ("Net avg", "avg_net"),
                     ("In-20%", "inside_20_rate"),
                     ("TB%", "touchback_rate"),
                     ("Pin%", "pin_rate")],
            "metrics": {
                "Net average": ("avg_net", False),
                "Gross distance": ("avg_distance", False),
                "Inside-20 rate": ("inside_20_rate", False),
                "Pin rate": ("pin_rate", False),
                "Touchback rate (lower better)": ("touchback_rate", True),
                "EPA per punt": ("punt_epa", False),
            },
        },
    }

    if selected_team == LEAGUE_WIDE_KEY:
        st.markdown(f"### {selected_season} league-wide leaderboards")
        st.caption("Pick a position and a metric to see who's #1 in the NFL.")
    else:
        st.markdown(f"### {selected_season} {team_name}")
        st.caption("Pick a position and a metric to see how the roster stacks up.")

    col_pos, col_metric = st.columns([1, 2])
    with col_pos:
        selected_pos = st.selectbox("Position", list(NFL_POSITION_CONFIGS.keys()),
                                     index=0, key="landing_position")
    cfg = NFL_POSITION_CONFIGS[selected_pos]
    with col_metric:
        sort_label, sort_col, sort_asc = metric_picker(
            cfg["metrics"], key=f"landing_metric_{selected_pos}", label="🔍 Sort leaderboard by"
        )

    DATA_DIR = Path(__file__).resolve().parent / "data"
    data_path = DATA_DIR / cfg["file"]
    if not data_path.exists():
        st.warning(f"Data file missing: {cfg['file']}")
    else:
        ldf = pl.read_parquet(str(data_path)).to_pandas()
        if cfg["filter"]:
            fcol, fval = cfg["filter"]
            ldf = ldf[ldf[fcol] == fval]
        ldf = filter_by_team_and_season(ldf, selected_team, selected_season,
                                          team_col="recent_team", season_col="season_year")
        if cfg["snap_col"] in ldf.columns:
            ldf = ldf[ldf[cfg["snap_col"]].fillna(0) >= cfg["min_snaps"]]

        if len(ldf) == 0:
            st.info(f"No {cfg['noun']} found for this team/season.")
        else:
            if sort_col in ldf.columns:
                ldf = ldf.sort_values(sort_col, ascending=sort_asc, na_position="last")
            ldf = ldf.head(25).reset_index(drop=True)
            ldf.index = ldf.index + 1

            def _fmt(col, val):
                if pd.isna(val):
                    return "—"
                if "rate" in col or "share" in col or "_pct" in col or col == "pin_rate":
                    return f"{val*100:.1f}%" if abs(val) < 2 else f"{val:.1f}%"
                if col in ("epa_per_target", "pass_epa_per_play", "epa_per_rush", "punt_epa", "passing_cpoe", "yac_above_exp"):
                    return f"{val:+.2f}"
                if col in ("yards_per_carry", "yards_per_target", "avg_distance", "avg_net", "yards_after_contact_per_att"):
                    return f"{val:.2f}"
                if isinstance(val, float) and val == int(val):
                    return f"{int(val)}"
                if isinstance(val, float):
                    return f"{val:.1f}"
                return str(val)

            display = pd.DataFrame({"#": ldf.index})
            for label, col in cfg["cols"]:
                if col in ldf.columns:
                    display[label] = ldf[col].apply(lambda v, c=col: _fmt(c, v))
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption(f"Showing top {len(ldf)} {cfg['noun']} sorted by **{sort_label}**. Click into the position page from the sidebar for the full feature set.")
    st.divider()
    st.markdown("### Pick a position")
    st.markdown("**Offense:** QB · WR · TE · RB · OL\n\n**Defense:** DE · DT · LB · CB · S\n\n"
                "**Special teams:** Kicker · Punter\n\n**Front office:** Coaches · OC · DC · GM")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Our ethos")
        st.markdown("Every stat on every page has its formula, its data source, and its known weaknesses on display.")
    with col2:
        st.markdown("### Why this exists")
        st.markdown("Free data, open methodology, community-built. No grade is final, no stat is beyond questioning.")

# ══════════════════════════════════════════════════════════════
# COLLEGE MODE
# ══════════════════════════════════════════════════════════════
else:
    # ── Helpers ───────────────────────────────────────────
    def zscore_to_percentile(z):
        if pd.isna(z): return None
        return float(norm.cdf(z) * 100)

    def format_percentile(pct):
        if pct is None or pd.isna(pct): return "—"
        if pct >= 99: return "top 1%"
        if pct >= 50: return f"top {100 - int(pct)}%"
        return f"bottom {int(pct)}%"

    def star_display(stars):
        if pd.isna(stars) or stars is None: return ""
        return "⭐" * int(stars)

    # ── Cached data loaders ───────────────────────────────
    ALL_SCHOOLS_LABEL = "🏫 All schools"
    ALL_CONFERENCES_LABEL = "🏈 All conferences"

    @st.cache_data
    def get_school_list():
        schools = set()
        for fname in ["college_qb_all_seasons.parquet", "college_wr_all_seasons.parquet",
                      "college_te_all_seasons.parquet", "college_rb_all_seasons.parquet",
                      "college_def_all_seasons.parquet"]:
            path = COLLEGE_DATA_DIR / fname
            if path.exists():
                df = pd.read_parquet(path, columns=["team"])
                schools.update(df["team"].dropna().unique())
        return sorted(schools)

    @st.cache_data
    def load_draft_class(year):
        """Return a set of (last_name_lower, school_lower) tuples for the
        given draft year, sourced from nflverse's combine invitation list
        (the closest public proxy for "declared prospects in the class").
        Returns an empty set if the data isn't available.
        """
        try:
            import nflreadpy as nfl
            cb = nfl.load_combine([year]).to_pandas()
            out = set()
            for _, row in cb.iterrows():
                name = str(row.get("player_name", "") or "")
                school = str(row.get("school", "") or "")
                last = name.split()[-1].lower() if name else ""
                out.add((last, school.lower()))
                # Also include last-name-only fallback for fuzzy match
                if last:
                    out.add((last, ""))
            return out
        except Exception:
            return set()

    @st.cache_data
    def get_conference_list():
        confs = set()
        for fname in ["college_qb_all_seasons.parquet", "college_wr_all_seasons.parquet",
                      "college_te_all_seasons.parquet", "college_rb_all_seasons.parquet",
                      "college_def_all_seasons.parquet"]:
            path = COLLEGE_DATA_DIR / fname
            if path.exists():
                df = pd.read_parquet(path, columns=["conference"])
                if "conference" in df.columns:
                    confs.update(df["conference"].dropna().unique())
        return sorted(confs)

    @st.cache_data
    def load_college_position(fname):
        path = COLLEGE_DATA_DIR / fname
        if not path.exists(): return pd.DataFrame()
        df = pd.read_parquet(path)

        # Merge in CFBD-enriched columns (EPA + usage + downs splits) when
        # available. We match on (player_id, season). Existing columns win
        # on collisions so the basic CFBD stats already in the file aren't
        # overwritten by the enrichment file's potentially-different values.
        pos_key = fname.replace("college_", "").replace("_all_seasons.parquet", "").lower()
        adv_path = COLLEGE_DATA_DIR / f"college_{pos_key}_cfbd_advanced.parquet"
        if adv_path.exists():
            adv = pd.read_parquet(adv_path)
            # Cols to bring across — anything advanced not already in the base
            advanced_cols = [
                c for c in adv.columns
                if (c.startswith("epa_") or c.startswith("usage_"))
            ]
            if advanced_cols and "player_id" in adv.columns and "season" in adv.columns:
                merge_keys = ["player_id", "season"]
                # Coerce to string on both sides for a clean join
                df["player_id"] = df["player_id"].astype(str)
                adv["player_id"] = adv["player_id"].astype(str)
                df = df.merge(
                    adv[merge_keys + advanced_cols].drop_duplicates(subset=merge_keys),
                    on=merge_keys, how="left",
                )
        return df

    @st.cache_data
    def load_enrichment(fname):
        path = COLLEGE_DATA_DIR / fname
        if not path.exists(): return pd.DataFrame()
        return pd.read_parquet(path)

    # ── Load enrichment data ──────────────────────────────
    recruiting_df = load_enrichment("college_recruiting.parquet")
    usage_df = load_enrichment("college_usage.parquet")
    adjusted_df = load_enrichment("college_adjusted_metrics.parquet")
    transfers_df = load_enrichment("college_transfers.parquet")
    combine_df = load_enrichment("nfl_combine.parquet")

    # ── School + Conference + Season selector ─────────────
    schools = [ALL_SCHOOLS_LABEL] + get_school_list()
    conferences = [ALL_CONFERENCES_LABEL] + get_conference_list()
    COLLEGE_SEASONS = list(range(2025, 2013, -1))

    # Default school: All schools (was Michigan before).
    if "college_school_v2" not in st.session_state:
        st.session_state.college_school_v2 = ALL_SCHOOLS_LABEL

    COLLEGE_POSITIONS_FOR_TOP = ["QB", "WR", "TE", "RB", "DE", "DT", "LB", "CB", "S"]

    col_school, col_conf, col_season, col_position = st.columns([2, 2, 1, 1])
    with col_school:
        selected_school = st.selectbox("School", options=schools,
                                        index=schools.index(st.session_state.college_school_v2) if st.session_state.college_school_v2 in schools else 0,
                                        key="college_school_v2", label_visibility="collapsed")
    with col_conf:
        selected_conf = st.selectbox("Conference", options=conferences,
                                      index=0, key="college_conference",
                                      label_visibility="collapsed")
    with col_season:
        selected_college_season = st.selectbox("Season", options=COLLEGE_SEASONS,
                                               index=0, key="college_season_landing",
                                               label_visibility="collapsed")
    with col_position:
        selected_position = st.selectbox("Position", options=COLLEGE_POSITIONS_FOR_TOP,
                                          index=0, key="college_position_top",
                                          label_visibility="collapsed")

    # Resolve filter modes:
    #   - Specific school: school filter wins; conference is informational.
    #   - All schools + specific conference: filter by conference.
    #   - All schools + All conferences: show everyone (FBS-wide).
    school_is_all = selected_school == ALL_SCHOOLS_LABEL
    conf_is_all = selected_conf == ALL_CONFERENCES_LABEL

    if not school_is_all:
        header_label = f"{selected_school} — {selected_college_season}"
    elif not conf_is_all:
        header_label = f"{selected_conf} — {selected_college_season}"
    else:
        header_label = f"FBS-wide — {selected_college_season}"
    st.markdown(f"### {header_label}")
    st.caption(f"Every player z-scored against all FBS players at their position that season. Pick a metric per position to re-sort the leaderboard.")

    # ── Volume threshold + 2026 prospects filter ──────────
    # College doesn't track snap counts publicly, so volume is per-position:
    # carries for RB, receptions for WR/TE, pass attempts for QB, games for DEF.
    POS_VOLUME = {
        "QB": ("pass_att",         "Min pass attempts",  100, 0, 700,  25),
        "WR": ("receptions_total", "Min receptions",      15, 0, 150,  5),
        "TE": ("receptions_total", "Min receptions",      10, 0, 100,  2),
        "RB": ("carries_total",    "Min carries",         50, 0, 400,  10),
        "DE": ("games",            "Min games played",     6, 1, 15,   1),
        "DT": ("games",            "Min games played",     6, 1, 15,   1),
        "LB": ("games",            "Min games played",     6, 1, 15,   1),
        "CB": ("games",            "Min games played",     6, 1, 15,   1),
        "S":  ("games",            "Min games played",     6, 1, 15,   1),
    }
    vol_col, vol_label, vol_default, vol_min, vol_max, vol_step = POS_VOLUME.get(
        selected_position, ("games", "Min games played", 6, 1, 15, 1)
    )

    col_g, col_d1, col_d2 = st.columns([2, 2, 3])
    with col_g:
        min_volume = st.slider(
            vol_label,
            min_value=vol_min, max_value=vol_max, value=vol_default, step=vol_step,
            key=f"college_min_volume_{selected_position}",
            help="Filter out low-volume players (college's snap-equivalent — varies by position).",
        )
    with col_d1:
        prospects_only = st.checkbox(
            "🎯 2026 draft prospects only",
            value=False,
            key="college_2026_filter",
            help="Two-layer filter: nflverse combine invites (~319 declared prospects) "
                 "PLUS the heuristic of recruits from 2022-2024 who played in 2025. "
                 "Combine invites are the most reliable signal; recruit-year heuristic "
                 "catches smaller-school prospects who weren't invited.",
        )
    draft_class_set = load_draft_class(2026) if prospects_only else set()
    if prospects_only and recruiting_df is not None and len(recruiting_df) > 0 and "recruit_year" in recruiting_df.columns:
        # Heuristic pool: recruits from 2022-2024 (covers 4-year + redshirt 5th-years)
        heuristic_recruits = recruiting_df[
            recruiting_df["recruit_year"].between(2022, 2024)
        ]
        heuristic_set = set()
        for _, row in heuristic_recruits.iterrows():
            name = str(row.get("name", "") or "")
            school = str(row.get("school", "") or "")
            last = name.split()[-1].lower() if name else ""
            if last:
                heuristic_set.add((last, school.lower()))
                heuristic_set.add((last, ""))
        draft_class_set = draft_class_set | heuristic_set
    if prospects_only:
        with col_d2:
            n_combine = len(load_draft_class(2026))
            st.caption(f"📋 {n_combine} combine invites + recruits from 2022-2024 still playing.")

    # ── Position configs with bundles ─────────────────────
    # Offensive positions also get CFBD-enriched z-cols (EPA per play, usage)
    # via the merge in load_college_position(). We list them here so they
    # appear in the metric picker and the radar / leaderboard hover.
    POSITION_FILES = {
        "QB": ("college_qb_all_seasons.parquet",
               ["completion_pct_z", "td_rate_z", "int_rate_z", "yards_per_attempt_z", "pass_tds_z", "rush_yards_total_z",
                "epa_per_play_avg_z", "epa_per_pass_avg_z", "epa_third_down_avg_z", "epa_passing_downs_avg_z",
                "usage_overall_z", "usage_pass_z", "usage_third_down_z", "usage_passing_downs_z"],
               {"completion_pct_z": "Comp %", "td_rate_z": "TD rate", "int_rate_z": "INT rate",
                "yards_per_attempt_z": "Yds/att", "pass_tds_z": "Pass TDs", "rush_yards_total_z": "Rush yds",
                "epa_per_play_avg_z": "EPA/play", "epa_per_pass_avg_z": "EPA/pass",
                "epa_third_down_avg_z": "EPA on 3rd down", "epa_passing_downs_avg_z": "EPA on passing downs",
                "usage_overall_z": "Snap usage %", "usage_pass_z": "Pass usage %",
                "usage_third_down_z": "3rd-down usage %", "usage_passing_downs_z": "Passing-down usage %"}),
        "WR": ("college_wr_all_seasons.parquet",
               ["rec_yards_total_z", "rec_tds_total_z", "receptions_total_z", "yards_per_rec_z",
                "epa_per_play_avg_z", "epa_per_pass_avg_z", "epa_third_down_avg_z",
                "usage_overall_z", "usage_pass_z", "usage_third_down_z"],
               {"rec_yards_total_z": "Rec yds", "rec_tds_total_z": "Rec TDs",
                "receptions_total_z": "Receptions", "yards_per_rec_z": "Yds/rec",
                "epa_per_play_avg_z": "EPA/play", "epa_per_pass_avg_z": "EPA/target (CFBD)",
                "epa_third_down_avg_z": "EPA on 3rd down",
                "usage_overall_z": "Snap usage %", "usage_pass_z": "Pass tgt %",
                "usage_third_down_z": "3rd-down tgt %"}),
        "TE": ("college_te_all_seasons.parquet",
               ["rec_yards_total_z", "rec_tds_total_z", "receptions_total_z", "yards_per_rec_z",
                "epa_per_play_avg_z", "epa_per_pass_avg_z",
                "usage_overall_z", "usage_pass_z"],
               {"rec_yards_total_z": "Rec yds", "rec_tds_total_z": "Rec TDs",
                "receptions_total_z": "Receptions", "yards_per_rec_z": "Yds/rec",
                "epa_per_play_avg_z": "EPA/play", "epa_per_pass_avg_z": "EPA/target (CFBD)",
                "usage_overall_z": "Snap usage %", "usage_pass_z": "Pass tgt %"}),
        "RB": ("college_rb_all_seasons.parquet",
               ["rush_yards_total_z", "rush_tds_total_z", "yards_per_carry_z", "total_yards_z", "receptions_total_z",
                "epa_per_play_avg_z", "epa_per_rush_avg_z", "epa_per_pass_avg_z",
                "usage_overall_z", "usage_rush_z", "usage_pass_z"],
               {"rush_yards_total_z": "Rush yds", "rush_tds_total_z": "Rush TDs",
                "yards_per_carry_z": "Yds/carry", "total_yards_z": "Total yds", "receptions_total_z": "Receptions",
                "epa_per_play_avg_z": "EPA/play", "epa_per_rush_avg_z": "EPA/rush",
                "epa_per_pass_avg_z": "EPA/target (CFBD)",
                "usage_overall_z": "Snap usage %", "usage_rush_z": "Rush usage %", "usage_pass_z": "Pass tgt %"}),
        "DE": ("college_def_all_seasons.parquet",
                ["sacks_per_game_z", "tfl_per_game_z", "qb_hurries_per_game_z", "tackles_per_game_z",
                 "pressure_rate_z", "splash_plays_per_game_z", "tfl_share_z",
                 "pressure_conversion_rate_z"],
                {"sacks_per_game_z": "Sacks/gm", "tfl_per_game_z": "TFL/gm",
                 "qb_hurries_per_game_z": "QB hurries/gm", "tackles_per_game_z": "Tackles/gm",
                 "pressure_rate_z": "Pressure rate",
                 "splash_plays_per_game_z": "Splash plays/gm",
                 "tfl_share_z": "TFL share (% of tackles)",
                 "pressure_conversion_rate_z": "Pressure conv rate (sacks/pressures)"}),
        "DT": ("college_def_all_seasons.parquet",
                ["sacks_per_game_z", "tfl_per_game_z", "qb_hurries_per_game_z", "tackles_per_game_z",
                 "pressure_rate_z", "splash_plays_per_game_z", "tfl_share_z",
                 "pressure_conversion_rate_z"],
                {"sacks_per_game_z": "Sacks/gm", "tfl_per_game_z": "TFL/gm",
                 "qb_hurries_per_game_z": "QB hurries/gm", "tackles_per_game_z": "Tackles/gm",
                 "pressure_rate_z": "Pressure rate",
                 "splash_plays_per_game_z": "Splash plays/gm",
                 "tfl_share_z": "TFL share (% of tackles)",
                 "pressure_conversion_rate_z": "Pressure conv rate (sacks/pressures)"}),
        "LB": ("college_def_all_seasons.parquet",
                ["tackles_per_game_z", "solo_tackles_per_game_z", "tfl_per_game_z", "sacks_per_game_z",
                 "pd_per_game_z", "int_per_game_z",
                 "splash_plays_per_game_z", "tfl_share_z", "ball_production_per_game_z"],
                {"tackles_per_game_z": "Tackles/gm", "solo_tackles_per_game_z": "Solo tkl/gm",
                 "tfl_per_game_z": "TFL/gm", "sacks_per_game_z": "Sacks/gm",
                 "pd_per_game_z": "PD/gm", "int_per_game_z": "INT/gm",
                 "splash_plays_per_game_z": "Splash plays/gm",
                 "tfl_share_z": "TFL share (penetration)",
                 "ball_production_per_game_z": "Ball production/gm (PD+INT)"}),
        "CB": ("college_def_all_seasons.parquet",
                ["pd_per_game_z", "int_per_game_z", "tackles_per_game_z",
                 "solo_tackles_per_game_z", "tfl_per_game_z",
                 "ball_production_per_game_z", "int_per_pd_ratio_z", "splash_plays_per_game_z"],
                {"pd_per_game_z": "PD/gm", "int_per_game_z": "INT/gm",
                 "tackles_per_game_z": "Tackles/gm", "solo_tackles_per_game_z": "Solo tkl/gm",
                 "tfl_per_game_z": "TFL/gm",
                 "ball_production_per_game_z": "Ball production/gm (PD+INT)",
                 "int_per_pd_ratio_z": "INT-per-PD ratio (instinct)",
                 "splash_plays_per_game_z": "Splash plays/gm"}),
        "S": ("college_def_all_seasons.parquet",
                ["pd_per_game_z", "int_per_game_z", "tackles_per_game_z",
                 "solo_tackles_per_game_z", "tfl_per_game_z",
                 "splash_plays_per_game_z", "ball_production_per_game_z", "int_per_pd_ratio_z"],
                {"pd_per_game_z": "PD/gm", "int_per_game_z": "INT/gm",
                 "tackles_per_game_z": "Tackles/gm", "solo_tackles_per_game_z": "Solo tkl/gm",
                 "tfl_per_game_z": "TFL/gm",
                 "splash_plays_per_game_z": "Splash plays/gm",
                 "ball_production_per_game_z": "Ball production/gm (PD+INT)",
                 "int_per_pd_ratio_z": "INT-per-PD ratio (instinct)"}),
    }

    # NFL position -> college pos_group filter values (matches college_data.py).
    COLLEGE_DEF_POS_FILTER = {
        "DE": ["EDGE"],
        "DT": ["DL"],
        "LB": ["LB"],
        "CB": ["CB"],
        "S":  ["DB"],
    }

    # ── Bundle definitions by position ────────────────────
    COLLEGE_BUNDLES = {
        "QB": {
            "efficiency": {"label": "📊 Passing efficiency", "why": "Pure passing production per attempt",
                          "stats": {"yards_per_attempt_z": 0.5, "td_rate_z": 0.3, "completion_pct_z": 0.2}},
            "ball_security": {"label": "🛡️ Ball security", "why": "How well does he protect the football?",
                             "stats": {"int_rate_z": 1.0}},
            "production": {"label": "🏆 Volume production", "why": "Total output — yards and touchdowns",
                          "stats": {"pass_tds_z": 0.5, "rush_yards_total_z": 0.5}},
        },
        "WR": {
            "production": {"label": "🏆 Production", "why": "Total receiving output",
                          "stats": {"rec_yards_total_z": 0.4, "rec_tds_total_z": 0.3, "receptions_total_z": 0.3}},
            "efficiency": {"label": "📊 Efficiency", "why": "Quality per catch",
                          "stats": {"yards_per_rec_z": 1.0}},
        },
        "TE": {
            "production": {"label": "🏆 Production", "why": "Total receiving output",
                          "stats": {"rec_yards_total_z": 0.4, "rec_tds_total_z": 0.3, "receptions_total_z": 0.3}},
            "efficiency": {"label": "📊 Efficiency", "why": "Quality per catch",
                          "stats": {"yards_per_rec_z": 1.0}},
        },
        "RB": {
            "rushing": {"label": "🏃 Rushing", "why": "Ground game production and efficiency",
                       "stats": {"rush_yards_total_z": 0.3, "rush_tds_total_z": 0.2, "yards_per_carry_z": 0.5}},
            "versatility": {"label": "🎯 Versatility", "why": "Total offensive contribution including receiving",
                           "stats": {"total_yards_z": 0.4, "receptions_total_z": 0.3, "total_tds_z": 0.3}},
        },
        "DE": {
            "pass_rush": {"label": "⚡ Pass rush", "why": "Get to the QB",
                         "stats": {"sacks_per_game_z": 0.4, "qb_hurries_per_game_z": 0.3, "pressure_rate_z": 0.3}},
            "run_defense": {"label": "🛡️ Run defense", "why": "Stop the run at the line",
                           "stats": {"tfl_per_game_z": 0.6, "tackles_per_game_z": 0.4}},
        },
        "DT": {
            "pass_rush": {"label": "⚡ Interior pass rush", "why": "Collapse the pocket",
                         "stats": {"sacks_per_game_z": 0.4, "qb_hurries_per_game_z": 0.3, "pressure_rate_z": 0.3}},
            "run_defense": {"label": "🛡️ Run defense", "why": "Plug the gaps",
                           "stats": {"tfl_per_game_z": 0.5, "tackles_per_game_z": 0.5}},
        },
        "LB": {
            "tackling": {"label": "💪 Tackling", "why": "Sure tackler in space",
                        "stats": {"tackles_per_game_z": 0.5, "solo_tackles_per_game_z": 0.5}},
            "run_disruption": {"label": "💥 Run disruption", "why": "TFLs and stops behind the line",
                              "stats": {"tfl_per_game_z": 1.0}},
            "coverage": {"label": "🛡️ Coverage", "why": "Drops in coverage and disrupts pass game",
                        "stats": {"pd_per_game_z": 0.5, "int_per_game_z": 0.5}},
            "pass_rush": {"label": "⚡ Pass rush", "why": "Blitz LB sack production",
                         "stats": {"sacks_per_game_z": 1.0}},
        },
        "CB": {
            "ball_hawk": {"label": "🦅 Ball-hawking", "why": "Picks and pass breakups",
                         "stats": {"int_per_game_z": 0.5, "pd_per_game_z": 0.5}},
            "tackling": {"label": "💪 Tackling", "why": "Doesn't shy from contact",
                        "stats": {"tackles_per_game_z": 0.7, "tfl_per_game_z": 0.3}},
        },
        "S": {
            "ball_hawk": {"label": "🦅 Ball-hawking", "why": "Range and ball production",
                         "stats": {"int_per_game_z": 0.5, "pd_per_game_z": 0.5}},
            "tackling": {"label": "💪 Run support", "why": "Comes downhill to tackle",
                        "stats": {"tackles_per_game_z": 0.5, "solo_tackles_per_game_z": 0.3, "tfl_per_game_z": 0.2}},
        },
    }

    # ── Sidebar: position selector + sliders ──────────────
    # Position is now driven by the top-bar dropdown so the sliders apply
    # to whatever the user picked up there.
    st.sidebar.header("What matters to you?")
    st.sidebar.caption(f"Adjust what you value for **{selected_position}** — change the position dropdown at the top of the page to switch.")
    active_pos = selected_position if selected_position in COLLEGE_BUNDLES else "QB"

    bundles = COLLEGE_BUNDLES[active_pos]
    bundle_weights = {}
    st.sidebar.markdown("---")
    for bk, bundle in bundles.items():
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.caption(f"_{bundle['why']}_")
        if f"college_bundle_{active_pos}_{bk}" not in st.session_state:
            st.session_state[f"college_bundle_{active_pos}_{bk}"] = 50
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100, step=5,
            key=f"college_bundle_{active_pos}_{bk}",
            label_visibility="collapsed",
        )

    # ── Compute effective weights from bundles ────────────
    effective_weights = {}
    for bk, bundle in bundles.items():
        bw = bundle_weights.get(bk, 0)
        if bw == 0: continue
        for stat, internal_w in bundle["stats"].items():
            effective_weights[stat] = effective_weights.get(stat, 0) + bw * internal_w

    # ── Scoring function ─────────────────────────────────
    def score_college_players(df, weights):
        total_w = sum(weights.values())
        if total_w == 0:
            df["your_score"] = 0.0
            return df
        score = pd.Series(0.0, index=df.index)
        for stat, w in weights.items():
            if stat in df.columns:
                score += df[stat].fillna(0) * (w / total_w)
        df["your_score"] = score
        return df

    # ── Helper: find recruiting info for a player ─────────
    def get_recruiting_info(player_name):
        if len(recruiting_df) == 0: return None
        last = player_name.split()[-1] if player_name else ""
        matches = recruiting_df[recruiting_df["name"].str.contains(last, na=False, case=False)]
        if len(matches) == 1: return matches.iloc[0]
        # Try tighter match
        first = player_name.split()[0] if player_name else ""
        tight = matches[matches["name"].str.contains(first, na=False, case=False)]
        if len(tight) == 1: return tight.iloc[0]
        if len(tight) > 0: return tight.iloc[0]
        if len(matches) > 0: return matches.iloc[0]
        return None

    # ── Helper: find usage info ───────────────────────────
    def get_usage_info(player_name, team, season):
        if len(usage_df) == 0: return None
        last = player_name.split()[-1] if player_name else ""
        matches = usage_df[(usage_df["name"].str.contains(last, na=False, case=False)) &
                           (usage_df["team"] == team) & (usage_df["season"] == season)]
        return matches.iloc[0] if len(matches) > 0 else None

    # ── Helper: find adjusted metrics ─────────────────────
    def get_adjusted_info(player_name, team, season):
        if len(adjusted_df) == 0: return None
        last = player_name.split()[-1] if player_name else ""
        matches = adjusted_df[(adjusted_df["name"].str.contains(last, na=False, case=False)) &
                              (adjusted_df["team"] == team) & (adjusted_df["season"] == season)]
        return matches if len(matches) > 0 else None

    # ── Helper: find transfer info ────────────────────────
    def get_transfer_info(player_name, school=None):
        if len(transfers_df) == 0: return None
        if not player_name: return None
        name_parts = player_name.split()
        if len(name_parts) < 2: return None
        first = name_parts[0]
        last = name_parts[-1]
        # Match on both first and last name
        matches = transfers_df[
            (transfers_df["name"].str.contains(last, na=False, case=False)) &
            (transfers_df["name"].str.contains(first, na=False, case=False))
        ]
        # If school provided, further filter by origin or destination
        if school and len(matches) > 1:
            school_matches = matches[
                (matches["origin"].str.contains(school, na=False, case=False)) |
                (matches["destination"].str.contains(school, na=False, case=False))
            ]
            if len(school_matches) > 0:
                matches = school_matches
        if len(matches) > 0: return matches
        return None

    # ── Helper: find combine/pro day info ───────────────
    all_workouts_df = load_enrichment("nfl_all_workouts.parquet")

    def get_combine_info(player_name, school=None):
        # Use all_workouts as primary — it has the source column (combine vs pro_day)
        if len(all_workouts_df) > 0:
            result = _lookup_workout(all_workouts_df, player_name, school, "player_name", "school")
            if result is not None:
                # Set display source from the source column
                src = result.get("source", "")
                if src == "pro_day":
                    result["_workout_source"] = "Pro Day"
                elif src == "combine+pro_day":
                    result["_workout_source"] = "Combine + Pro Day"
                else:
                    result["_workout_source"] = "NFL Combine"
                return result
        # Fallback to combine parquet for body measurements
        result = _lookup_workout(combine_df, player_name, school, "player_name", "school")
        if result is not None:
            result["_workout_source"] = "NFL Combine"
        return result

    def _lookup_workout(df, player_name, school, name_col, school_col):
        if len(df) == 0: return None
        last = player_name.split()[-1] if player_name else ""
        first = player_name.split()[0] if player_name else ""
        matches = df[df[name_col].str.contains(last, na=False, case=False)]
        if school:
            school_first = school.split()[0] if school else ""
            school_matches = matches[matches[school_col].str.contains(school_first, na=False, case=False)]
            if len(school_matches) > 0:
                matches = school_matches
        tight = matches[matches[name_col].str.contains(first, na=False, case=False)]
        if len(tight) > 0: return tight.iloc[0]
        if len(matches) == 1: return matches.iloc[0]
        return None

    def format_combine_display(comb):
        """Format combine/pro day data as a readable string."""
        if comb is None: return None
        parts = []
        # Handle both combine format (ht as "6-2") and scraped format (height_in as 74.0)
        if pd.notna(comb.get("ht")):
            parts.append(f"Ht: {comb['ht']}")
        elif pd.notna(comb.get("height_in")):
            inches = int(comb["height_in"])
            feet = inches // 12
            remaining = inches % 12
            parts.append(f"Ht: {feet}-{remaining}")
        if pd.notna(comb.get("wt")):
            parts.append(f"Wt: {int(comb['wt'])}")
        elif pd.notna(comb.get("weight")):
            parts.append(f"Wt: {int(comb['weight'])}")
        if pd.notna(comb.get("forty")): parts.append(f"40: {comb['forty']:.2f}s")
        if pd.notna(comb.get("bench")): parts.append(f"Bench: {int(comb['bench'])}")
        if pd.notna(comb.get("vertical")): parts.append(f"Vert: {comb['vertical']}\"")
        if pd.notna(comb.get("broad_jump")): parts.append(f"Broad: {int(comb['broad_jump'])}\"")
        if pd.notna(comb.get("cone")): parts.append(f"3-cone: {comb['cone']:.2f}s")
        if pd.notna(comb.get("shuttle")): parts.append(f"Shuttle: {comb['shuttle']:.2f}s")
        return " · ".join(parts) if parts else None

    # ── Helper: build career line chart ───────────────────
    def build_career_chart(player_name, df, season_col, z_cols, labels, unique_key=""):
        """Build career line chart with metric dropdown."""
        all_player = df[df["player"] == player_name].sort_values(season_col)
        if len(all_player) < 2: return

        # Compute composite for each season
        available_z = [c for c in z_cols if c in all_player.columns]
        all_player["composite_z"] = all_player[available_z].mean(axis=1)

        seasons = [int(s) for s in all_player[season_col].tolist()]
        teams = all_player["team"].tolist()

        # Build metric options
        metric_options = {"Composite score": all_player["composite_z"].tolist()}
        for z_col in z_cols:
            if z_col in all_player.columns and all_player[z_col].notna().any():
                label = labels.get(z_col, z_col.replace("_z", ""))
                metric_options[label] = all_player[z_col].tolist()

        selected_metric = st.selectbox("Metric", options=list(metric_options.keys()),
                                        index=0, key=f"college_metric_{player_name}_{unique_key}",
                                        label_visibility="collapsed")

        values = metric_options[selected_metric]
        percentiles = [zscore_to_percentile(v) if pd.notna(v) else None for v in values]

        colors = []
        for v in values:
            if pd.isna(v): colors.append("#ccc")
            elif v >= 1.0: colors.append("#B8860B")
            elif v >= 0.0: colors.append("#DAA520")
            elif v >= -1.0: colors.append("#CD853F")
            else: colors.append("#A0522D")

        hover_text = []
        for s, v, p, t in zip(seasons, values, percentiles, teams):
            if pd.notna(v):
                pct_str = f"top {100 - int(p)}%" if p and p >= 50 else f"bottom {int(p)}%" if p else "—"
                hover_text.append(f"<b>{s}</b> ({t})<br>{selected_metric}: {v:+.2f}<br>{pct_str} of FBS")
            else:
                hover_text.append(f"<b>{s}</b> ({t})<br>No data")

        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                      annotation_text="FBS avg", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color="#888")

        fig.add_trace(go.Scatter(
            x=seasons, y=values, mode="lines+markers",
            line=dict(color="#B8860B", width=3),
            marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
            hovertext=hover_text, hoverinfo="text",
        ))

        all_valid = [v for v in values if pd.notna(v)]
        if all_valid:
            y_max = max(max(all_valid) + 0.5, 2.0)
            y_min = min(min(all_valid) - 0.5, -2.0)
            fig.add_hrect(y0=1.0, y1=y_max, fillcolor="rgba(184,134,11,0.05)", line_width=0)
            fig.add_hrect(y0=-1.0, y1=y_min, fillcolor="rgba(160,82,45,0.05)", line_width=0)

        fig.update_layout(
            xaxis=dict(title="Season", tickmode="array", tickvals=seasons,
                       ticktext=[str(s) for s in seasons], gridcolor="#eee"),
            yaxis=dict(title=selected_metric, gridcolor="#eee",
                       zeroline=True, zerolinecolor="#888", zerolinewidth=1),
            height=300, margin=dict(l=50, r=20, t=20, b=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"career_chart_{unique_key}")

    # ══════════════════════════════════════════════════════
    # RENDER EACH POSITION GROUP
    # ══════════════════════════════════════════════════════
    from lib_shared import metric_picker as _metric_picker

    # Only render the position picked from the top dropdown
    positions_to_render = [(p, POSITION_FILES[p]) for p in [selected_position] if p in POSITION_FILES]

    for pos_name, (fname, z_cols, labels) in positions_to_render:
        df = load_college_position(fname)
        if len(df) == 0: continue

        season_col = "season" if "season" in df.columns else "season_year"

        # Apply selectors: school > conference > FBS-wide.
        mask = df[season_col] == selected_college_season
        if not school_is_all:
            mask = mask & (df["team"] == selected_school)
        elif not conf_is_all and "conference" in df.columns:
            mask = mask & (df["conference"] == selected_conf)
        # For defensive positions, narrow to the specific pos_group bucket
        if pos_name in COLLEGE_DEF_POS_FILTER and "pos_group" in df.columns:
            allowed = COLLEGE_DEF_POS_FILTER[pos_name] + ["UNKNOWN"]
            mask = mask & (df["pos_group"].isin(allowed))
        filtered = df[mask].copy()

        # Volume filter (college's snap-equivalent — column varies by position)
        if vol_col in filtered.columns and min_volume > 0:
            filtered = filtered[filtered[vol_col].fillna(0) >= min_volume]

        # 2026 draft prospect filter — keep only rows whose (last name, team)
        # appears in the prospect set (combine invites + heuristic).
        if prospects_only and len(draft_class_set) > 0:
            def _is_prospect(row):
                name = str(row.get("player", "") or "")
                team = str(row.get("team", "") or "")
                last = name.split()[-1].lower() if name else ""
                if not last:
                    return False
                return (last, team.lower()) in draft_class_set or (last, "") in draft_class_set
            filtered = filtered[filtered.apply(_is_prospect, axis=1)]

        if len(filtered) == 0: continue

        available_z = [c for c in z_cols if c in filtered.columns]
        if available_z:
            filtered["composite_z"] = filtered[available_z].mean(axis=1)

        # Apply slider weights if this is the active position
        is_active = (pos_name == active_pos)
        if is_active and effective_weights:
            filtered = score_college_players(filtered, effective_weights)
            score_col = "your_score"
            score_label_text = "Your score"
        else:
            score_col = "composite_z"
            score_label_text = "Score"

        pos_display = {"QB": "Quarterbacks", "WR": "Wide Receivers", "TE": "Tight Ends",
                       "RB": "Running Backs",
                       "DE": "Defensive Ends", "DT": "Defensive Tackles",
                       "LB": "Linebackers", "CB": "Cornerbacks", "S": "Safeties"}[pos_name]
        pos_col = "pos_group" if pos_name in COLLEGE_DEF_POS_FILTER and "pos_group" in filtered.columns else None

        active_marker = " ← your sliders" if is_active else ""
        st.markdown(f"#### {pos_display}{active_marker}")

        # Per-position metric picker — sort the leaderboard by any nerd stat
        position_metrics = {}
        for z_col, label in labels.items():
            raw_col = z_col.replace("_z", "")
            if z_col in filtered.columns:
                position_metrics[label + " (z)"] = (z_col, False)
            if raw_col in filtered.columns:
                position_metrics[label] = (raw_col, False)
        # Default to composite ("Your score" if active, "Score" otherwise)
        col_metric_label, col_sort, col_asc = _metric_picker(
            position_metrics,
            default_label=score_label_text if score_col != "composite_z" else "Your score",
            key=f"college_metric_{pos_name}",
            label=f"🔍 Sort {pos_display.lower()} by",
        )
        # If user picked "Your score" but there's no slider score, fall back to composite_z
        if col_sort == "score":
            col_sort = score_col
        if col_sort in filtered.columns:
            filtered = filtered.sort_values(col_sort, ascending=col_asc, na_position="last")
        else:
            filtered = filtered.sort_values(score_col, ascending=False)

        # Cap the leaderboard for non-school views so the page stays usable.
        # 25 expanders × multiple sub-charts is the upper bound of "renders
        # in a reasonable time"; bigger pools should drill into a school.
        if school_is_all:
            filtered = filtered.head(25)

        # ── Summary table ─────────────────────────────────
        display_rows = []
        for _, row in filtered.iterrows():
            name = row.get("player", "—")
            comp = row.get(score_col, np.nan)
            pct = zscore_to_percentile(comp)

            entry = {"Player": name}
            if pos_col and pd.notna(row.get(pos_col)):
                entry["Pos"] = row[pos_col]

            # Add recruiting stars
            rec = get_recruiting_info(name)
            if rec is not None and pd.notna(rec.get("stars")):
                entry["Stars"] = star_display(rec["stars"])

            # Add draft eligibility
            if rec is not None and pd.notna(rec.get("recruit_year")):
                elig_year = int(rec["recruit_year"]) + 3
                if elig_year <= 2025:
                    entry["Elig"] = f"✅ {elig_year}"
                elif elig_year == 2026:
                    entry["Elig"] = f"🟡 {elig_year}"
                else:
                    entry["Elig"] = f"🔒 {elig_year}"

            # Add combine highlights
            comb = get_combine_info(name, selected_school)
            if comb is not None:
                if pd.notna(comb.get("ht")): entry["Ht"] = comb["ht"]
                if pd.notna(comb.get("wt")): entry["Wt"] = int(comb["wt"])
                if pd.notna(comb.get("forty")): entry["40"] = f"{comb['forty']:.2f}"
            
            if pd.notna(comp):
                entry[score_label_text] = f"{'+' if comp >= 0 else ''}{comp:.2f}"
                entry["Pctl"] = format_percentile(pct)
            else:
                entry[score_label_text] = "—"
                entry["Pctl"] = "—"

            for z_col, label in labels.items():
                raw_col = z_col.replace("_z", "")
                raw = row.get(raw_col)
                if pd.notna(raw):
                    if "rate" in raw_col or "pct" in raw_col:
                        entry[label] = f"{raw:.1%}" if raw < 1 else f"{raw:.1f}%"
                    else:
                        entry[label] = f"{raw:.1f}" if isinstance(raw, float) else str(int(raw))
                else:
                    entry[label] = "—"
            display_rows.append(entry)

        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        # ── Player expanders ──────────────────────────────
        # Top-25 cap above keeps these usable in FBS-wide / conference views.
        if school_is_all:
            st.caption(f"💡 Showing top {len(filtered)} expanders below. Pick a specific school for the full roster.")
        for _, row in filtered.iterrows():
            name = row.get("player", "—")
            comp = row.get("composite_z", np.nan)
            pos_tag = f" ({row[pos_col]})" if pos_col and pd.notna(row.get(pos_col)) else ""
            score_tag = f" · {comp:+.2f}" if pd.notna(comp) else ""

            # Recruiting badge
            rec = get_recruiting_info(name)
            stars_tag = f" {star_display(rec['stars'])}" if rec is not None and pd.notna(rec.get("stars")) else ""

            with st.expander(f"**{name}**{pos_tag}{score_tag}{stars_tag}"):

                # ── Recruiting header ─────────────────────
                if rec is not None:
                    rec_parts = []
                    if pd.notna(rec.get("stars")): rec_parts.append(f"{star_display(rec['stars'])} ({int(rec['stars'])}-star)")
                    if pd.notna(rec.get("ranking")): rec_parts.append(f"#{int(rec['ranking'])} nationally")
                    if pd.notna(rec.get("rating")): rec_parts.append(f"rating: {rec['rating']:.4f}")
                    if pd.notna(rec.get("city")) and pd.notna(rec.get("state")): rec_parts.append(f"{rec['city']}, {rec['state']}")
                    # Draft eligibility
                    if pd.notna(rec.get("recruit_year")):
                        elig_year = int(rec["recruit_year"]) + 3
                        if elig_year <= 2025:
                            rec_parts.append(f"✅ Draft eligible ({elig_year})")
                        elif elig_year == 2026:
                            rec_parts.append(f"🟡 Eligible 2026")
                        else:
                            rec_parts.append(f"🔒 Eligible {elig_year}")
                    if rec_parts:
                        st.caption(f"Recruiting: {' · '.join(rec_parts)}")

                # ── Workout measurables ───────────────────
                comb = get_combine_info(name, selected_school)
                comb_display = format_combine_display(comb)
                if comb_display:
                    draft_info_parts = []
                    if comb is not None and pd.notna(comb.get("draft_round")):
                        draft_info_parts.append(f"Rd {int(comb['draft_round'])}")
                    if comb is not None and pd.notna(comb.get("draft_ovr")):
                        draft_info_parts.append(f"Pick #{int(comb['draft_ovr'])}")
                    if comb is not None and pd.notna(comb.get("draft_team")):
                        draft_info_parts.append(f"→ {comb['draft_team']}")
                    draft_str = f"<div style='margin-top:4px;font-size:0.85rem;color:#888;'>{' '.join(draft_info_parts)}</div>" if draft_info_parts else ""

                    # Build individual metric badges
                    metric_badges = []
                    badge_data = []
                    if pd.notna(comb.get("ht")):
                        badge_data.append(("Height", str(comb["ht"])))
                    elif pd.notna(comb.get("height_in")):
                        inches = int(comb["height_in"])
                        badge_data.append(("Height", f"{inches // 12}-{inches % 12}"))
                    if pd.notna(comb.get("wt")):
                        badge_data.append(("Weight", f"{int(comb['wt'])} lbs"))
                    elif pd.notna(comb.get("weight")):
                        badge_data.append(("Weight", f"{int(comb['weight'])} lbs"))
                    if pd.notna(comb.get("forty")):
                        badge_data.append(("40-yard", f"{comb['forty']:.2f}s"))
                    if pd.notna(comb.get("bench")):
                        badge_data.append(("Bench", f"{int(comb['bench'])} reps"))
                    if pd.notna(comb.get("vertical")):
                        badge_data.append(("Vertical", f"{comb['vertical']}\""))
                    if pd.notna(comb.get("broad_jump")):
                        badge_data.append(("Broad", f"{int(comb['broad_jump'])}\""))
                    if pd.notna(comb.get("cone")):
                        badge_data.append(("3-cone", f"{comb['cone']:.2f}s"))
                    if pd.notna(comb.get("shuttle")):
                        badge_data.append(("Shuttle", f"{comb['shuttle']:.2f}s"))

                    for label, val in badge_data:
                        metric_badges.append(
                            f"<div style='display:inline-block;background:#f0f2f6;border-radius:8px;"
                            f"padding:6px 12px;margin:3px;text-align:center;'>"
                            f"<div style='font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;'>{label}</div>"
                            f"<div style='font-size:1.1rem;font-weight:bold;color:#1a1a2e;'>{val}</div>"
                            f"</div>"
                        )

                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);"
                        f"border-radius:10px;padding:12px 16px;margin:8px 0;'>"
                        f"<div style='color:#e0e0e0;font-size:0.85rem;margin-bottom:6px;'>🏋️ <b>MEASURABLES</b></div>"
                        f"<div style='display:flex;flex-wrap:wrap;gap:2px;'>{''.join(metric_badges)}</div>"
                        f"{draft_str}</div>",
                        unsafe_allow_html=True,
                    )

                # ── Pedigree score ────────────────────────
                try:
                    from pedigree import compute_pedigree, render_pedigree
                    # Build college seasons data
                    all_player_for_pedigree = df[df["player"] == name].sort_values(season_col)
                    college_seasons_for_ped = []
                    for _, prow in all_player_for_pedigree.iterrows():
                        pz = [prow.get(c) for c in available_z if c in prow.index and pd.notna(prow.get(c))]
                        college_seasons_for_ped.append({
                            "season": int(prow[season_col]),
                            "team": prow.get("team", ""),
                            "composite_z": np.mean(pz) if pz else np.nan,
                            "conference": prow.get("conference", ""),
                        })

                    rec_for_ped = None
                    if rec is not None:
                        rec_for_ped = {"stars": rec.get("stars"), "rating": rec.get("rating"), "ranking": rec.get("ranking")}

                    # Check if player was drafted
                    draft_for_ped = None
                    try:
                        draft_linkage = load_enrichment("college_to_nfl_draft_linkage.parquet")
                        if len(draft_linkage) > 0:
                            last_name = name.split()[-1] if name else ""
                            draft_match = draft_linkage[draft_linkage["college_player"].str.contains(last_name, na=False, case=False)]
                            if len(draft_match) > 0:
                                dm = draft_match.iloc[0]
                                draft_for_ped = {"round": dm.get("round"), "overall": dm.get("overall"), "nfl_team": dm.get("nfl_team")}
                    except:
                        pass

                    ped_result = compute_pedigree(
                        player_name=name,
                        college_seasons_data=college_seasons_for_ped,
                        recruiting_info=rec_for_ped,
                        draft_info=draft_for_ped,
                    )
                    render_pedigree(ped_result, name)
                except (ImportError, Exception) as e:
                    st.caption(f"_Pedigree error: {e}_")

                pc1, pc2 = st.columns([1, 1])

                with pc1:
                    # ── Stat table ────────────────────────
                    stat_rows = []
                    for z_col, label in labels.items():
                        if z_col not in row.index: continue
                        z = row.get(z_col)
                        if pd.isna(z): continue
                        raw_col = z_col.replace("_z", "")
                        raw = row.get(raw_col)
                        p = zscore_to_percentile(z)
                        stat_rows.append({
                            "Stat": label,
                            "Value": f"{raw:.2f}" if pd.notna(raw) else "—",
                            "Z-score": f"{z:+.2f}",
                            "Pctl": f"{int(p)}th" if p else "—",
                        })
                    if stat_rows:
                        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

                    # ── Usage data ────────────────────────
                    usage = get_usage_info(name, selected_school, selected_college_season)
                    if usage is not None:
                        st.markdown("**Usage rates**")
                        usage_items = []
                        if pd.notna(usage.get("usage_overall")): usage_items.append(f"Overall: {usage['usage_overall']:.1%}")
                        if pd.notna(usage.get("usage_pass")): usage_items.append(f"Pass: {usage['usage_pass']:.1%}")
                        if pd.notna(usage.get("usage_rush")): usage_items.append(f"Rush: {usage['usage_rush']:.1%}")
                        if pd.notna(usage.get("usage_third_down")): usage_items.append(f"3rd down: {usage['usage_third_down']:.1%}")
                        if usage_items:
                            st.caption(" · ".join(usage_items))

                    # ── Adjusted metrics ──────────────────
                    adj = get_adjusted_info(name, selected_school, selected_college_season)
                    if adj is not None and len(adj) > 0:
                        st.markdown("**Opponent-adjusted**")
                        for _, arow in adj.iterrows():
                            stype = arow.get("stat_type", "")
                            wepa = arow.get("wepa")
                            plays = arow.get("plays")
                            if pd.notna(wepa):
                                st.caption(f"{stype.title()} WEPA: {wepa:+.2f} ({int(plays)} plays)" if pd.notna(plays) else f"{stype.title()} WEPA: {wepa:+.2f}")

                    # ── Composite score ───────────────────
                    if pd.notna(comp):
                        pct = zscore_to_percentile(comp)
                        st.markdown(f"**Composite: {comp:+.2f}** ({format_percentile(pct)} of FBS {pos_display.lower()})")

                    # ── Transfer history ──────────────────
                    all_player_data = df[df["player"] == name].sort_values(season_col)
                    unique_teams = all_player_data["team"].unique()
                    if len(unique_teams) > 1:
                        st.markdown("**Transfer history**")
                        t_rows = []
                        for _, trow in all_player_data.iterrows():
                            tz = [trow.get(c) for c in available_z if c in trow.index and pd.notna(trow.get(c))]
                            tcomp = np.mean(tz) if tz else np.nan
                            t_rows.append({
                                "Season": int(trow[season_col]),
                                "Team": trow.get("team", ""),
                                "Conf": trow.get("conference", ""),
                                "Score": f"{tcomp:+.2f}" if pd.notna(tcomp) else "—",
                            })
                        st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

                    # Portal info
                    portal = get_transfer_info(name, selected_school)
                    if portal is not None and len(portal) > 0:
                        portal_lines = []
                        for _, prow in portal.iterrows():
                            origin = prow.get("origin", "")
                            dest = prow.get("destination", "")
                            season = prow.get("season", "")
                            if pd.notna(origin) and pd.notna(dest) and dest:
                                portal_lines.append(f"{int(season)}: {origin} → {dest}" if pd.notna(season) else f"{origin} → {dest}")
                            elif pd.notna(origin):
                                portal_lines.append(f"{int(season)}: Entered portal from {origin}" if pd.notna(season) else f"Entered portal from {origin}")
                        if portal_lines:
                            st.caption(f"Portal: {' · '.join(portal_lines)}")

                with pc2:
                    # ── Radar chart with year picker ─────
                    college_career_df = df[df["player"] == name]
                    radar_year_options = sorted(
                        set(int(s) for s in college_career_df[season_col].dropna().unique()),
                        reverse=True,
                    )
                    radar_year_options_full = (
                        radar_year_options + (["All-career mean"] if len(radar_year_options) > 1 else [])
                    )
                    try:
                        radar_default_idx = radar_year_options_full.index(int(selected_college_season))
                    except (ValueError, TypeError):
                        radar_default_idx = 0
                    radar_year_choice = st.selectbox(
                        "Radar season",
                        options=radar_year_options_full,
                        index=radar_default_idx,
                        key=f"college_radar_year_{pos_name}_{name}",
                        format_func=lambda v: f"Season {v}" if isinstance(v, int) else v,
                    )
                    if radar_year_choice == "All-career mean":
                        radar_source = college_career_df.select_dtypes(include="number").mean()
                    else:
                        season_rows = college_career_df[college_career_df[season_col] == radar_year_choice]
                        if len(season_rows) == 1:
                            radar_source = season_rows.iloc[0]
                        elif len(season_rows) > 1:
                            radar_source = season_rows.select_dtypes(include="number").mean()
                        else:
                            radar_source = row

                    radar_axes, radar_values = [], []
                    for z_col, label in labels.items():
                        if z_col not in radar_source.index: continue
                        z = radar_source.get(z_col)
                        if pd.isna(z): continue
                        p = zscore_to_percentile(z)
                        radar_axes.append(label)
                        radar_values.append(p)

                    if len(radar_axes) >= 3:
                        st.markdown(f"**Percentile profile** vs. FBS {pos_display.lower()}")
                        st.caption("50th = FBS average")
                        rfig = go.Figure()
                        rfig.add_trace(go.Scatterpolar(
                            r=radar_values + [radar_values[0]],
                            theta=radar_axes + [radar_axes[0]],
                            fill="toself",
                            fillcolor="rgba(218, 165, 32, 0.25)",
                            line=dict(color="rgba(184, 134, 11, 0.9)", width=2),
                            marker=dict(size=6, color="rgba(184, 134, 11, 1)"),
                            hovertemplate="<b>%{theta}</b><br>%{r:.0f}th pctl<extra></extra>",
                        ))
                        rfig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100],
                                                tickvals=[25, 50, 75, 100],
                                                ticktext=["25th", "50th", "75th", "100th"],
                                                tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
                                angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
                                bgcolor="rgba(0,0,0,0)",
                            ),
                            showlegend=False, margin=dict(l=60, r=60, t=20, b=20),
                            height=320, paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(rfig, use_container_width=True, key=f"radar_{pos_name}_{name}")

                # ── Career line chart with metric dropdown ─
                all_player_career = df[df["player"] == name].sort_values(season_col)
                if len(all_player_career) >= 2:
                    st.markdown("**College career arc**")
                    build_career_chart(name, df, season_col, z_cols, labels, unique_key=f"{pos_name}_{name}")
                    st.caption("Each point = one season vs. all FBS players at this position. 0.00 = FBS average.")

                # ── Statistical comps + college-to-pro prediction ─
                try:
                    from comps import render_college_comps, render_college_to_pro
                    pos_map_comps = {"QB": "qb", "WR": "wr", "TE": "te", "RB": "rb"}
                    comp_pos = pos_map_comps.get(pos_name)
                    if comp_pos:
                        render_college_comps(name, selected_school, selected_college_season, comp_pos, pos_display.lower())
                        render_college_to_pro(name, selected_school, selected_college_season, comp_pos, pos_display.lower())
                except (ImportError, Exception):
                    pass

    st.divider()
    st.caption("College data via CollegeFootballData.com · Recruiting, usage, and adjusted metrics via CFBD API · Z-scored against all FBS players per position per season")

st.divider()
st.caption("Built with Streamlit · NFL data from nflverse · College data from CFBD · Open source on GitHub")
