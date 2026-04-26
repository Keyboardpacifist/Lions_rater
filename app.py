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

# Fresh-session reset — clear detail-mode markers from any prior browser
# session so the page always loads into the leaderboard view, never into
# a stale player profile. Filter widgets (school, season, position) and
# slider settings are intentionally NOT cleared.
if "_app_session_initialized_v3" not in st.session_state:
    _STALE_DETAIL_KEYS = ("expand_college_player", "nfl_search_target",
                          "_college_filter_ctx", "_last_mode")
    # Detail-marker prefixes used across pages:
    #   lb_selected_*           — College mode landing
    #   wr_selected_player_*    — pages/WR.py master/detail
    #   <pos>_selected_player_* — same pattern as we port to other pages
    #   <pos>_lb_expanded_*     — "show all" toggles
    _STALE_PREFIXES = ("lb_selected_", "_selected_player_", "_lb_expanded_")
    for _k in list(st.session_state.keys()):
        if _k in _STALE_DETAIL_KEYS or any(p in _k for p in _STALE_PREFIXES):
            st.session_state.pop(_k, None)
    st.session_state._app_session_initialized_v3 = True

COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"

# ── Mode toggle ───────────────────────────────────────────────
col_title, col_toggle = st.columns([3, 2])
with col_title:
    st.markdown("<h1 style='margin:0; padding:4px 0;'>🏈 NFL Rater</h1>", unsafe_allow_html=True)
with col_toggle:
    mode = st.radio("Mode", ["NFL", "College"], horizontal=True, key="mode_toggle", label_visibility="collapsed")

# Toggling NFL ↔ College should drop any sticky detail markers from
# the prior mode so the new mode lands on its leaderboard.
_prev_mode = st.session_state.get("_last_mode")
if _prev_mode is not None and _prev_mode != mode:
    for _k in list(st.session_state.keys()):
        if (_k.startswith("lb_selected_") or _k == "expand_college_player"
                or _k == "nfl_search_target"):
            st.session_state.pop(_k, None)
st.session_state._last_mode = mode

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

    # ── Player index (cached) for quick search ────────────
    # Hardcoded (position, file, optional row filter) so the search box
    # can render BEFORE NFL_POSITION_CONFIGS is defined further down.
    # Only positions surfaced on the landing leaderboard; jumping to a
    # position the landing doesn't render would silently fail.
    NFL_PLAYER_INDEX_FILES = [
        ("QB", "league_qb_all_seasons.parquet", None),
        ("WR", "league_wr_all_seasons.parquet", ("position", "WR")),
        ("TE", "league_te_all_seasons.parquet", ("position", "TE")),
        ("RB", "league_rb_all_seasons.parquet", None),
        ("DE", "league_de_all_seasons.parquet", None),
        ("DT", "league_dt_all_seasons.parquet", None),
        ("LB", "league_lb_all_seasons.parquet", None),
        ("Punter", "league_p_all_seasons.parquet", None),
    ]

    @st.cache_data
    def _build_nfl_player_index():
        """Returns sorted list of (player_name, recent_team, recent_season,
        position_key). One row per player, using the most recent season
        for the jump target so the leaderboard contains them."""
        DATA_DIR_LOCAL = Path(__file__).resolve().parent / "data"
        rows = []
        for pos_key, fname, filt in NFL_PLAYER_INDEX_FILES:
            path = DATA_DIR_LOCAL / fname
            if not path.exists(): continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if filt:
                fcol, fval = filt
                if fcol in df.columns:
                    df = df[df[fcol] == fval]
            name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
            team_col = "recent_team" if "recent_team" in df.columns else "team"
            season_col_l = "season_year" if "season_year" in df.columns else "season"
            if name_col not in df.columns or season_col_l not in df.columns: continue
            latest = df.sort_values(season_col_l, ascending=False).drop_duplicates(name_col)
            for _, r in latest.iterrows():
                n = r.get(name_col)
                if pd.isna(n) or not str(n).strip(): continue
                t = r.get(team_col, "")
                s = r.get(season_col_l)
                rows.append((str(n).strip(),
                              str(t) if pd.notna(t) else "",
                              int(s) if pd.notna(s) else None,
                              pos_key))
        rows.sort(key=lambda x: x[0].lower())
        return rows

    nfl_index = _build_nfl_player_index()
    nfl_search_options = [
        f"{n} — {display_abbr(t) if t else '?'} · {p}" + (f" ({s})" if s else "")
        for n, t, s, p in nfl_index
    ]

    # Nonce makes the search box's `key` change after every successful
    # pick, which forces Streamlit to re-instantiate the widget with its
    # `index=0` default. Setting `st.session_state.<same_key>` from inside
    # the widget's own callback is unreliable in Streamlit — the nonce
    # trick sidesteps that by giving the next render a fresh key.
    if "nfl_search_nonce" not in st.session_state:
        st.session_state.nfl_search_nonce = 0
    _nfl_search_key = f"nfl_player_search_{st.session_state.nfl_search_nonce}"

    def _on_nfl_search_change():
        choice = st.session_state.get(_nfl_search_key)
        if not choice or choice not in nfl_search_options:
            return
        idx = nfl_search_options.index(choice)
        if idx < 0 or idx >= len(nfl_index): return
        sel_name, sel_team, sel_season, sel_pos = nfl_index[idx]
        target_team_label = (
            f"{display_abbr(sel_team)} — {NFL_TEAMS.get(sel_team, sel_team)}"
            if sel_team and sel_team in NFL_TEAMS else None
        )
        if target_team_label and target_team_label in team_labels:
            st.session_state.landing_team_v2 = target_team_label
        if sel_season:
            st.session_state.landing_season = sel_season
        st.session_state.landing_position = sel_pos
        st.session_state.nfl_search_target = sel_name
        # Bump nonce → next render uses a NEW key → widget shows blank.
        st.session_state.nfl_search_nonce += 1

    col_search, col_team, col_season = st.columns([3, 2, 1])
    with col_search:
        st.selectbox(
            "🔎 Search any NFL player",
            options=nfl_search_options,
            index=None,
            placeholder="🔎 Type a player name...",
            key=_nfl_search_key,
            on_change=_on_nfl_search_change,
            label_visibility="collapsed",
            help="Type a name and pick a player. The box clears automatically after each pick — no backspacing needed.",
        )
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

    ALL_POSITIONS_LABEL_NFL = "🏈 All positions"
    position_options_nfl = [ALL_POSITIONS_LABEL_NFL] + list(NFL_POSITION_CONFIGS.keys())

    col_pos, col_metric = st.columns([1, 2])
    with col_pos:
        selected_pos = st.selectbox("Position", position_options_nfl,
                                     index=0, key="landing_position")

    if selected_pos == ALL_POSITIONS_LABEL_NFL:
        # Multi-position mode: no global metric picker — each position's
        # leaderboard uses the first metric in its own config.
        positions_to_iter = list(NFL_POSITION_CONFIGS.items())
        sort_label_global = None
    else:
        cfg_sel = NFL_POSITION_CONFIGS[selected_pos]
        with col_metric:
            sort_label_global, sort_col_global, sort_asc_global = metric_picker(
                cfg_sel["metrics"], key=f"landing_metric_{selected_pos}", label="🔍 Sort leaderboard by"
            )
        positions_to_iter = [(selected_pos, cfg_sel)]

    DATA_DIR = Path(__file__).resolve().parent / "data"

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

    all_pos_mode = (selected_pos == ALL_POSITIONS_LABEL_NFL)
    if all_pos_mode:
        st.caption("Showing the top of every position. Pick a single position above to unlock the metric picker and a deeper leaderboard.")

    # ── Active player-search filter (with clear button) ──
    nfl_search_target = st.session_state.get("nfl_search_target")
    if nfl_search_target:
        b_msg, b_btn = st.columns([5, 1])
        with b_msg:
            st.info(f"🔍 Filtered to **{nfl_search_target}** — leaderboard shows only this player.")
        with b_btn:
            if st.button("❌ Clear filter", key="clear_nfl_search_filter"):
                st.session_state.pop("nfl_search_target", None)
                st.rerun()

    for pos_iter_name, cfg_iter in positions_to_iter:
        if all_pos_mode:
            # First metric in the position's config = that leaderboard's
            # default sort when no global metric picker is shown.
            default_metric_label = list(cfg_iter["metrics"].keys())[0]
            sort_col_iter, sort_asc_iter = cfg_iter["metrics"][default_metric_label]
            sort_label_iter = default_metric_label
            st.markdown(f"#### {cfg_iter['noun'].title()}")
        else:
            sort_col_iter, sort_asc_iter = sort_col_global, sort_asc_global
            sort_label_iter = sort_label_global

        data_path = DATA_DIR / cfg_iter["file"]
        if not data_path.exists():
            st.warning(f"Data file missing: {cfg_iter['file']}")
            continue
        ldf = pl.read_parquet(str(data_path)).to_pandas()
        if cfg_iter["filter"]:
            fcol, fval = cfg_iter["filter"]
            ldf = ldf[ldf[fcol] == fval]
        ldf = filter_by_team_and_season(ldf, selected_team, selected_season,
                                          team_col="recent_team", season_col="season_year")
        if cfg_iter["snap_col"] in ldf.columns:
            ldf = ldf[ldf[cfg_iter["snap_col"]].fillna(0) >= cfg_iter["min_snaps"]]

        # Player-search filter — overrides the snap floor and other
        # filters so the searched player always shows. Match on either
        # `player_display_name` or `player_name`, whichever the parquet has.
        if nfl_search_target:
            name_col_l = "player_display_name" if "player_display_name" in ldf.columns else "player_name"
            if name_col_l in ldf.columns:
                ldf = ldf[ldf[name_col_l] == nfl_search_target]

        if len(ldf) == 0:
            st.info(f"No {cfg_iter['noun']} found for this team/season.")
            continue

        if sort_col_iter in ldf.columns:
            ldf = ldf.sort_values(sort_col_iter, ascending=sort_asc_iter, na_position="last")
        # All-positions: top 10 per group so the stacked page stays
        # skimmable; single-position: top 25 for depth.
        head_size = 10 if all_pos_mode else 25
        ldf = ldf.head(head_size).reset_index(drop=True)
        ldf.index = ldf.index + 1

        display = pd.DataFrame({"#": ldf.index})
        for label, col in cfg_iter["cols"]:
            if col in ldf.columns:
                display[label] = ldf[col].apply(lambda v, c=col: _fmt(c, v))
        st.dataframe(display, use_container_width=True, hide_index=True)
        if not all_pos_mode:
            st.caption(f"Showing top {len(ldf)} {cfg_iter['noun']} sorted by **{sort_label_iter}**. Click into the position page from the sidebar for the full feature set.")
        else:
            st.caption(f"Sorted by default metric: **{sort_label_iter}**.")
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
        # Compact form: "5⭐" instead of "⭐⭐⭐⭐⭐". The 5-emoji string
        # was eating ~80px of column width on mobile.
        if pd.isna(stars) or stars is None: return ""
        return f"{int(stars)}⭐"

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
    def _load_pos_group_overrides(mtime):
        # mtime keys the cache so edits to the JSON file refresh.
        path = COLLEGE_DATA_DIR / "pos_group_overrides.json"
        if not path.exists():
            return {}
        import json
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}
        # Drop comment/metadata keys (anything starting with _).
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def _pos_group_overrides():
        path = COLLEGE_DATA_DIR / "pos_group_overrides.json"
        m = path.stat().st_mtime if path.exists() else 0
        return _load_pos_group_overrides(m)

    @st.cache_data
    def _load_college_position_cached(fname, mtime):
        # `mtime` is part of the cache key — when the parquet is
        # regenerated (e.g., after a fresh data pull), the cache busts
        # automatically without needing to restart Streamlit.
        path = COLLEGE_DATA_DIR / fname
        if not path.exists(): return pd.DataFrame()
        df = pd.read_parquet(path)
        return df

    def load_college_position(fname):
        path = COLLEGE_DATA_DIR / fname
        mtime = path.stat().st_mtime if path.exists() else 0
        df = _load_college_position_cached(fname, mtime)
        if df.empty: return df

        # Apply manual pos_group overrides for the defensive parquet —
        # CFBD occasionally miscategorizes edge rushers as interior DL
        # (e.g., Eric O'Neill). The overrides JSON lets us correct
        # these without re-running the data pipeline.
        if fname == "college_def_all_seasons.parquet" and "pos_group" in df.columns:
            overrides = _pos_group_overrides()
            if overrides:
                df = df.copy()
                df["pos_group"] = df.apply(
                    lambda r: overrides.get(r.get("player"), r.get("pos_group")),
                    axis=1,
                )

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

    ALL_POSITIONS_LABEL_COLLEGE = "🏈 All positions"
    COLLEGE_POSITIONS_FOR_TOP = [ALL_POSITIONS_LABEL_COLLEGE, "QB", "WR", "TE", "RB", "OL", "DE", "DT", "LB", "CB", "S"]

    # ── Player index (cached) for quick search ────────────
    # Hardcoded so the search can render before POSITION_FILES is
    # defined further down. Defensive parquet is shared across DE/DT/
    # LB/CB/S — we read it once and dispatch via pos_group.
    COLLEGE_PLAYER_INDEX_FILES = [
        ("QB", "college_qb_all_seasons.parquet", None),
        ("WR", "college_wr_all_seasons.parquet", None),
        ("TE", "college_te_all_seasons.parquet", None),
        ("RB", "college_rb_all_seasons.parquet", None),
        # OL — identity-only roster from CFBD; no per-player stats. The
        # search dropdown labels them with their granular position
        # (OT/G/C/OL) but routes the click to the umbrella OL page.
        ("OL", "college_ol_roster.parquet", None),
    ]
    COLLEGE_DEF_INDEX_MAP = {
        "EDGE": "DE", "DL": "DT", "LB": "LB", "CB": "CB", "DB": "S",
    }

    @st.cache_data
    def _build_college_player_index(_overrides_mtime=0):
        """Returns sorted list of (player_name, recent_team, recent_season,
        position_key, position_label_for_dropdown) for every FBS player
        in our position parquets. The 5th tuple field exists so defensive
        UNKNOWN-pos_group players can show as "Def" in the dropdown while
        still routing the click to a real defensive position page."""
        rows = []
        for pos_key, fname, _filt in COLLEGE_PLAYER_INDEX_FILES:
            path = COLLEGE_DATA_DIR / fname
            if not path.exists(): continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            season_col_l = "season" if "season" in df.columns else "season_year"
            if "player" not in df.columns or season_col_l not in df.columns: continue
            latest = df.sort_values(season_col_l, ascending=False).drop_duplicates("player")
            for _, r in latest.iterrows():
                n = r.get("player")
                if pd.isna(n) or not str(n).strip(): continue
                t = r.get("team", "")
                s = r.get(season_col_l)
                # OL roster has a granular `position` column (OT/G/C/OL) —
                # show it in the search dropdown so the user gets more info,
                # but route the click to the umbrella OL leaderboard.
                gran_pos = r.get("position") if pos_key == "OL" else None
                pos_label = (str(gran_pos).strip() if (gran_pos and not pd.isna(gran_pos))
                              else pos_key)
                rows.append((str(n).strip(),
                              str(t) if pd.notna(t) else "",
                              int(s) if pd.notna(s) else None,
                              pos_key,
                              pos_label))
        # Defense: one shared parquet, dispatch per pos_group. CFBD's
        # defensive table contains any player with a defensive stat, so
        # OL/QB/TE who recorded a fumble-recovery tackle slip in too.
        # For UNKNOWN-pos_group rows (~65% of the file), require a
        # minimum defensive footprint before surfacing them as "Def" —
        # otherwise Blake Miller (Clemson OL, 2 tackles ever) shows up
        # in the safety/LB search results.
        def _has_defensive_footprint(r):
            def _v(k):
                v = r.get(k)
                return v if v is not None and not pd.isna(v) else 0
            return (_v("tackles_total") >= 5
                    or _v("sacks") >= 1
                    or _v("interceptions") >= 1
                    or _v("passes_deflected") >= 1)

        def_path = COLLEGE_DATA_DIR / "college_def_all_seasons.parquet"
        if def_path.exists():
            try:
                ddf = pd.read_parquet(def_path)
                season_col_l = "season" if "season" in ddf.columns else "season_year"
                if "player" in ddf.columns and "pos_group" in ddf.columns and season_col_l in ddf.columns:
                    # Apply pos_group overrides BEFORE drop_duplicates
                    # so a corrected EDGE doesn't get deduped against a
                    # stale DL row for the same player.
                    _pg_overrides = _pos_group_overrides()
                    if _pg_overrides:
                        ddf = ddf.copy()
                        ddf["pos_group"] = ddf.apply(
                            lambda r: _pg_overrides.get(r.get("player"), r.get("pos_group")),
                            axis=1,
                        )
                    ddf_sorted = ddf.sort_values(season_col_l, ascending=False).drop_duplicates(["player", "pos_group"])
                    for _, r in ddf_sorted.iterrows():
                        n = r.get("player")
                        pg = r.get("pos_group")
                        if pd.isna(n): continue
                        if pd.notna(pg) and pg in COLLEGE_DEF_INDEX_MAP:
                            pos_key = COLLEGE_DEF_INDEX_MAP[pg]
                            pos_label_for_option = pos_key
                        else:
                            # UNKNOWN: only include if the player has a
                            # real defensive footprint, otherwise it's
                            # an offensive/ST player who fluked a stat.
                            if not _has_defensive_footprint(r):
                                continue
                            pos_key = "DE"
                            pos_label_for_option = "Def"
                        t = r.get("team", "")
                        s = r.get(season_col_l)
                        rows.append((str(n).strip(),
                                      str(t) if pd.notna(t) else "",
                                      int(s) if pd.notna(s) else None,
                                      pos_key,
                                      pos_label_for_option))
            except Exception:
                pass
        rows.sort(key=lambda x: x[0].lower())
        return rows

    # Pass the overrides JSON mtime as the cache key so editing
    # pos_group_overrides.json forces a fresh search-index rebuild
    # (otherwise stale cached results keep mislabeling players).
    _overrides_path = COLLEGE_DATA_DIR / "pos_group_overrides.json"
    _overrides_mtime = _overrides_path.stat().st_mtime if _overrides_path.exists() else 0
    college_index = _build_college_player_index(_overrides_mtime)
    college_search_options = [
        f"{n} — {t or '?'} · {disp}" + (f" ({s})" if s else "")
        for n, t, s, _p, disp in college_index
    ]

    # Nonce-keyed widget: bumping the nonce changes the widget's key,
    # which forces Streamlit to re-instantiate it with its `index=0`
    # default. This is the reliable way to "clear" a selectbox — setting
    # st.session_state for the same key from inside the widget's own
    # on_change callback is unreliable.
    if "college_search_nonce" not in st.session_state:
        st.session_state.college_search_nonce = 0
    _college_search_key = f"college_player_search_{st.session_state.college_search_nonce}"

    def _on_college_search_change():
        choice = st.session_state.get(_college_search_key)
        if not choice or choice not in college_search_options:
            return
        idx = college_search_options.index(choice)
        if idx < 0 or idx >= len(college_index): return
        sel_name, sel_team, sel_season, sel_pos, _disp = college_index[idx]
        if sel_team and sel_team in schools:
            st.session_state.college_school_v2 = sel_team
        if sel_season and sel_season in COLLEGE_SEASONS:
            st.session_state.college_season_landing = sel_season
        if sel_pos in COLLEGE_POSITIONS_FOR_TOP:
            st.session_state.college_position_top = sel_pos
        st.session_state.expand_college_player = sel_name
        # Bump nonce → next render uses a NEW key → widget shows blank.
        st.session_state.college_search_nonce += 1

    col_search_c, col_school, col_conf, col_season, col_position = st.columns([3, 2, 2, 1, 1])
    with col_search_c:
        st.selectbox(
            "🔎 Search any college player",
            options=college_search_options,
            index=None,
            placeholder="🔎 Type a player name...",
            key=_college_search_key,
            on_change=_on_college_search_change,
            label_visibility="collapsed",
            help="Type a name and pick a player. The box clears automatically after each pick — no backspacing needed.",
        )
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
        # OL has no game stats — filter by weight so the user can scope
        # to draftable size (NFL OL median ~310 lbs, smallest ~280).
        "OL": ("weight",           "Min weight (lbs)",   270, 200, 400, 5),
        "DE": ("games",            "Min games played",     6, 1, 15,   1),
        "DT": ("games",            "Min games played",     6, 1, 15,   1),
        "LB": ("games",            "Min games played",     6, 1, 15,   1),
        "CB": ("games",            "Min games played",     6, 1, 15,   1),
        "S":  ("games",            "Min games played",     6, 1, 15,   1),
    }
    all_pos_mode_college = (selected_position == ALL_POSITIONS_LABEL_COLLEGE)

    if all_pos_mode_college:
        # All-positions mode: no single volume slider fits — the loop
        # applies each position's own default floor inline below.
        vol_col, min_volume = None, 0
        col_g, col_d1, col_d2 = st.columns([2, 2, 3])
        with col_g:
            st.caption("Volume filter: per-position defaults (pick a single position to customize).")
        with col_d1:
            prospects_only = st.checkbox(
                "🎯 2026 draft prospects only",
                value=False,
                key="college_2026_filter",
                help="Two-layer filter: nflverse combine invites (~319 declared prospects) "
                     "PLUS the heuristic of recruits from 2022-2024 who played in 2025.",
            )
    else:
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
        # OL — identity from CFBD roster + team-level proxy stats from
        # CFBD /stats/season/advanced. There are no per-player OL grades
        # in the free API, so each metric is the OL UNIT'S team-level
        # rank assigned to every OL on that team-season. Labels carry
        # the "(Team)" tag everywhere so users never mistake them for
        # individual grades. Stuff-rate is inverted so positive z = better.
        "OL": ("college_ol_roster.parquet",
               ["line_yards_z", "stuff_rate_avoid_z", "power_success_z",
                "std_downs_success_z", "rushing_ppa_z", "passing_ppa_z"],
               {"line_yards_z":      "(Team) Run-block line yds/rush",
                "stuff_rate_avoid_z":"(Team) Stuff-rate avoidance",
                "power_success_z":   "(Team) Short-yardage convert %",
                "std_downs_success_z":"(Team) Standard-downs success %",
                "rushing_ppa_z":     "(Team) Rushing PPA",
                "passing_ppa_z":     "(Team) Passing PPA (pass-pro proxy)"}),
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

    # ── Typical-starter benchmark helpers ─────────────────
    # Mirrors career_arc.COLLEGE_VOLUME_COL / COLLEGE_STARTER_TOP_N. The
    # "typical starter" pool for a (season, position) is the top-N players
    # by the position's natural volume column, after pos_group + games
    # filters for defense.
    COLLEGE_RADAR_VOLUME = {
        "QB": "pass_att",
        "WR": "receptions_total",
        "TE": "receptions_total",
        "RB": "carries_total",
        "DE": "tackles_total",
        "DT": "tackles_total",
        "LB": "tackles_total",
        "CB": "tackles_total",
        "S":  "tackles_total",
    }
    COLLEGE_RADAR_TOP_N = 130
    COLLEGE_RADAR_DEF_MIN_GAMES = 4  # matches derived_defense.py eligibility

    def _typical_starter_pool(full_df, season, pos_name, season_col):
        """Slice `full_df` down to the 'typical starting <position>' pool
        for one season. Defense filters by pos_group + games >= 4 floor;
        then top-N by the position's volume column (tackles for defense,
        receptions for WR/TE, carries for RB, pass attempts for QB)."""
        if season_col not in full_df.columns:
            return full_df.iloc[0:0]
        pool = full_df[full_df[season_col] == season]
        if pos_name in COLLEGE_DEF_POS_FILTER and "pos_group" in pool.columns:
            pool = pool[pool["pos_group"].isin(COLLEGE_DEF_POS_FILTER[pos_name])]
            if "games" in pool.columns:
                pool = pool[pool["games"].fillna(0) >= COLLEGE_RADAR_DEF_MIN_GAMES]
        sort_col = COLLEGE_RADAR_VOLUME.get(pos_name)
        if sort_col and sort_col in pool.columns:
            pool = pool.sort_values(sort_col, ascending=False).head(COLLEGE_RADAR_TOP_N)
        return pool

    def _starter_label(pos_display):
        """'Wide Receivers' -> 'Typical starting wide receiver'."""
        base = pos_display.lower()
        if base.endswith("s"):
            base = base[:-1]
        return f"Typical starting {base}"

    def _render_player_radar(player_name, key_prefix, full_df, labels_dict,
                              season_col, pos_name, pos_display, default_season,
                              header_prefix=None):
        """Render a season picker + radar polygon for one player, with the
        typical-starter benchmark layered on top. Used both for the main
        player and for the comparison player."""
        career = full_df[full_df["player"] == player_name]
        if len(career) == 0:
            st.caption(f"No data found for {player_name}.")
            return

        year_options = sorted(
            set(int(s) for s in career[season_col].dropna().unique()),
            reverse=True,
        )
        year_options_full = (
            year_options + (["All-career mean"] if len(year_options) > 1 else [])
        )
        try:
            default_idx = year_options_full.index(int(default_season))
        except (ValueError, TypeError):
            default_idx = 0
        year_choice = st.selectbox(
            "Radar season",
            options=year_options_full,
            index=default_idx,
            key=f"college_radar_year_{key_prefix}",
            format_func=lambda v: f"Season {v}" if isinstance(v, int) else v,
        )
        if year_choice == "All-career mean":
            radar_source = career.select_dtypes(include="number").mean()
        else:
            srows = career[career[season_col] == year_choice]
            if len(srows) == 1:
                radar_source = srows.iloc[0]
            elif len(srows) > 1:
                radar_source = srows.select_dtypes(include="number").mean()
            else:
                radar_source = career.iloc[0]

        axes, vals, z_used = [], [], []
        for z_col, label in labels_dict.items():
            if z_col not in radar_source.index: continue
            z = radar_source.get(z_col)
            if pd.isna(z): continue
            axes.append(label)
            vals.append(zscore_to_percentile(z))
            z_used.append(z_col)

        if year_choice == "All-career mean":
            bench_seasons = [int(s) for s in year_options]
        else:
            bench_seasons = [int(year_choice)]
        bench_pcts, bench_raws = [], []
        for z_col in z_used:
            per_z, per_raw = [], []
            raw_col = z_col.replace("_z", "")
            for s in bench_seasons:
                pool = _typical_starter_pool(full_df, s, pos_name, season_col)
                if z_col in pool.columns:
                    med_z = pool[z_col].dropna().median()
                    if pd.notna(med_z): per_z.append(float(med_z))
                if raw_col in pool.columns:
                    med_raw = pool[raw_col].dropna().median()
                    if pd.notna(med_raw): per_raw.append(float(med_raw))
            bench_pcts.append(zscore_to_percentile(np.mean(per_z)) if per_z else None)
            bench_raws.append(np.mean(per_raw) if per_raw else None)

        if len(axes) < 3:
            st.caption(f"Not enough metrics to render a radar for {player_name}.")
            return

        # Header is symmetric so the comparison stack reads the same on
        # top and bottom — player name in the same spot for both rows.
        suffix = f" ({header_prefix.lower()})" if header_prefix else ""
        st.markdown(
            f"**{player_name}** — Percentile profile vs. FBS {pos_display.lower()}{suffix}"
        )
        st.caption("50th = FBS average")

        rfig = go.Figure()
        rfig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=axes + [axes[0]],
            fill="toself",
            fillcolor="rgba(218, 165, 32, 0.25)",
            line=dict(color="rgba(184, 134, 11, 0.9)", width=2),
            marker=dict(size=6, color="rgba(184, 134, 11, 1)"),
            name=player_name,
            hovertemplate="<b>%{theta}</b><br>%{r:.0f}th pctl<extra></extra>",
        ))
        bench_label = _starter_label(pos_display)
        has_bench = any(p is not None for p in bench_pcts)
        if has_bench:
            bv_clean = [p if p is not None else 50 for p in bench_pcts]
            bench_hover = []
            for ax, pct, raw in zip(axes, bv_clean, bench_raws):
                raw_str = f"median: {raw:.2f} · " if raw is not None else ""
                bench_hover.append(
                    f"<b>{ax}</b><br>{bench_label}<br>{raw_str}{pct:.0f}th percentile"
                )
            bench_hover.append(bench_hover[0])
            rfig.add_trace(go.Scatterpolar(
                r=bv_clean + [bv_clean[0]],
                theta=axes + [axes[0]],
                mode="lines+markers",
                line=dict(color="rgba(102, 102, 102, 0.9)", width=2, dash="dot"),
                marker=dict(size=10, color="rgba(102, 102, 102, 0.95)",
                            symbol="diamond", line=dict(width=2, color="white")),
                name=bench_label,
                hovertext=bench_hover, hoverinfo="text",
            ))
        # Legend goes BELOW the chart (horizontal) instead of overlaying the
        # top-left corner — the prior placement covered angular axis labels,
        # especially bad on phone widths where the legend occupied the
        # whole top row.
        rfig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                tickvals=[25, 50, 75, 100],
                                ticktext=["25th", "50th", "75th", "100th"],
                                tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
                angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=has_bench,
            legend=dict(orientation="h", yanchor="top", y=-0.05,
                        xanchor="center", x=0.5,
                        bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)",
                        font=dict(size=10)),
            margin=dict(l=60, r=60, t=20, b=70),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(rfig, use_container_width=True, key=f"radar_{key_prefix}")

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
    # Position is driven by the top-bar dropdown so the sliders apply
    # to whatever the user picked up there. In all-positions mode, the
    # sliders are hidden (they only make sense for one position at a time).
    st.sidebar.header("What matters to you?")
    active_pos = selected_position if selected_position in COLLEGE_BUNDLES else None
    bundle_weights = {}
    if active_pos is None:
        st.sidebar.caption(
            "Pick a single position at the top of the page to unlock the "
            "bundle sliders. In **All positions** mode, leaderboards sort "
            "by each position's composite z-score."
        )
    else:
        st.sidebar.caption(f"Adjust what you value for **{active_pos}** — change the position dropdown at the top of the page to switch.")
        bundles = COLLEGE_BUNDLES[active_pos]
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
    # In all-positions mode there are no bundles/sliders, so this is a
    # no-op and `effective_weights` stays empty (no per-player score).
    effective_weights = {}
    if active_pos is not None:
        for bk, bundle in COLLEGE_BUNDLES[active_pos].items():
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
    # Color palette for multi-metric mode (first slot is the gold we use
    # in single-metric mode, so single-then-multi visually carries over).
    MULTI_METRIC_COLORS = [
        "#B8860B", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
        "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f",
    ]

    def build_career_chart(player_name, df, season_col, z_cols, labels, unique_key="",
                            pos_name=None, pos_display=None):
        """Build career line chart with single- or multi-metric mode.

        Single metric (default): selectbox + one gold line + dashed-gray
        typical-starter benchmark.
        Multi-metric: multiselect + one colored line per metric, no
        benchmark (each metric has its own scale, so a single dotted line
        wouldn't make sense), legend below the chart.
        """
        all_player = df[df["player"] == player_name].sort_values(season_col)
        if len(all_player) < 2: return

        available_z = [c for c in z_cols if c in all_player.columns]
        all_player["composite_z"] = all_player[available_z].mean(axis=1)

        seasons = [int(s) for s in all_player[season_col].tolist()]
        teams = all_player["team"].tolist()

        metric_options = {"Composite score": all_player["composite_z"].tolist()}
        metric_to_col = {"Composite score": "composite_z"}
        for z_col in z_cols:
            if z_col in all_player.columns and all_player[z_col].notna().any():
                label = labels.get(z_col, z_col.replace("_z", ""))
                metric_options[label] = all_player[z_col].tolist()
                metric_to_col[label] = z_col
        all_metric_labels = list(metric_options.keys())

        # ── Multi-metric toggle ───────────────────────────
        multi_metric = st.checkbox(
            "📊 Show multiple metrics",
            key=f"multi_metric_{player_name}_{unique_key}",
            help=("Overlay multiple metrics on the same chart. "
                  "Each metric draws as its own colored line; the typical "
                  "starter benchmark is hidden in this mode."),
        )

        if multi_metric:
            # `st.pills` renders every option as a clickable button/pill
            # so the user can see all available metrics at once and
            # toggle them on/off individually — closer to the "click
            # buttons to add lines" behaviour than a multiselect caret.
            # Default: just the composite so the starting state is one
            # line and the user adds from there.
            selected_metrics = st.pills(
                "Metrics to overlay — click to add or remove lines",
                options=all_metric_labels,
                default=["Composite score"] if "Composite score" in all_metric_labels else all_metric_labels[:1],
                selection_mode="multi",
                key=f"multi_metric_pills_{player_name}_{unique_key}",
            )
            if not selected_metrics:
                st.caption("Click at least one metric button above.")
                return
        else:
            selected_metric = st.selectbox(
                "Metric", options=all_metric_labels, index=0,
                key=f"college_metric_{player_name}_{unique_key}",
                label_visibility="collapsed",
            )
            selected_metrics = [selected_metric]

        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                      annotation_text="FBS avg", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color="#888")

        # Plot each selected metric
        all_y_for_axis = []
        for i, metric in enumerate(selected_metrics):
            values = metric_options[metric]
            color = MULTI_METRIC_COLORS[i % len(MULTI_METRIC_COLORS)]
            percentiles = [zscore_to_percentile(v) if pd.notna(v) else None for v in values]

            if multi_metric:
                # Uniform color across the line — varying-marker-color in
                # single mode is a player-by-player heat cue that doesn't
                # translate when several metrics share the chart.
                marker_colors = color
                line_w = 2.5
                marker_sz = 9
            else:
                marker_colors = [
                    "#ccc" if pd.isna(v)
                    else "#B8860B" if v >= 1.0
                    else "#DAA520" if v >= 0.0
                    else "#CD853F" if v >= -1.0
                    else "#A0522D"
                    for v in values
                ]
                line_w = 3
                marker_sz = 12

            hover_text = []
            for s, v, p, t in zip(seasons, values, percentiles, teams):
                if pd.notna(v):
                    pct_str = f"top {100 - int(p)}%" if p and p >= 50 else f"bottom {int(p)}%" if p else "—"
                    hover_text.append(f"<b>{s}</b> ({t})<br>{metric}: {v:+.2f}<br>{pct_str} of FBS")
                else:
                    hover_text.append(f"<b>{s}</b> ({t})<br>No data")

            fig.add_trace(go.Scatter(
                x=seasons, y=values, mode="lines+markers",
                name=metric,
                line=dict(color=color, width=line_w),
                marker=dict(size=marker_sz, color=marker_colors,
                            line=dict(width=2, color="white")),
                hovertext=hover_text, hoverinfo="text",
                showlegend=multi_metric,
            ))
            all_y_for_axis.extend(v for v in values if pd.notna(v))

        # ── Typical-starter benchmark (single-metric mode only) ──
        bench_xs = []
        if not multi_metric and pos_name:
            bench_col = metric_to_col.get(selected_metrics[0])
            bench_ys, bench_hover = [], []
            for s in seasons:
                pool = _typical_starter_pool(df, s, pos_name, season_col)
                if len(pool) == 0:
                    continue
                if bench_col == "composite_z":
                    pool_avail = [c for c in z_cols if c in pool.columns]
                    if not pool_avail: continue
                    series = pool[pool_avail].mean(axis=1)
                elif bench_col in pool.columns:
                    series = pool[bench_col]
                else:
                    continue
                med = series.dropna().median()
                if pd.isna(med): continue
                bench_xs.append(s)
                bench_ys.append(float(med))
                bench_hover.append(
                    f"<b>{s} typical starter</b><br>{selected_metrics[0]}: {med:+.2f}"
                )
            if bench_xs:
                starter_label = _starter_label(pos_display) if pos_display else "Typical starter"
                fig.add_trace(go.Scatter(
                    x=bench_xs, y=bench_ys, mode="lines+markers",
                    line=dict(color="#666", width=2, dash="dot"),
                    marker=dict(size=10, color="#666", symbol="diamond",
                                line=dict(width=2, color="white")),
                    hovertext=bench_hover, hoverinfo="text",
                    name=starter_label, showlegend=True,
                ))
                all_y_for_axis.extend(bench_ys)

        if all_y_for_axis:
            y_max = max(max(all_y_for_axis) + 0.5, 2.0)
            y_min = min(min(all_y_for_axis) - 0.5, -2.0)
            fig.add_hrect(y0=1.0, y1=y_max, fillcolor="rgba(184,134,11,0.05)", line_width=0)
            fig.add_hrect(y0=-1.0, y1=y_min, fillcolor="rgba(160,82,45,0.05)", line_width=0)

        # Layout: legend at bottom in multi-metric mode, top-left otherwise.
        if multi_metric:
            legend_kwargs = dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="center", x=0.5,
                                  bgcolor="rgba(255,255,255,0.7)",
                                  bordercolor="#ccc", borderwidth=1,
                                  font=dict(size=10))
            chart_height, bottom_margin = 360, 80
            y_title = "Z-score (vs. FBS avg)"
        else:
            legend_kwargs = dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                  bgcolor="rgba(255,255,255,0.7)",
                                  bordercolor="#ccc", borderwidth=1,
                                  font=dict(size=10))
            chart_height, bottom_margin = 300, 50
            y_title = selected_metrics[0]

        fig.update_layout(
            xaxis=dict(title="Season", tickmode="array", tickvals=seasons,
                       ticktext=[str(s) for s in seasons], gridcolor="#eee"),
            yaxis=dict(title=y_title, gridcolor="#eee",
                       zeroline=True, zerolinecolor="#888", zerolinewidth=1),
            height=chart_height, margin=dict(l=50, r=20, t=20, b=bottom_margin),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=multi_metric or bool(bench_xs),
            legend=legend_kwargs,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"career_chart_{unique_key}")

    # ══════════════════════════════════════════════════════
    # RENDER EACH POSITION GROUP
    # ══════════════════════════════════════════════════════
    from lib_shared import metric_picker as _metric_picker

    # All-positions mode: render every position's leaderboard/expanders.
    # Single-position mode: render only the picked position.
    if all_pos_mode_college:
        positions_to_render = [(p, POSITION_FILES[p]) for p in POSITION_FILES]
    else:
        positions_to_render = [(p, POSITION_FILES[p]) for p in [selected_position] if p in POSITION_FILES]

    # ── Detail mode resolution ────────────────────────────
    # Detail mode is triggered ONLY by an explicit name click. The
    # search bar narrows the leaderboard but never auto-enters detail —
    # otherwise stale `expand_college_player` from a previous search
    # would dump the user into a player profile on reload.
    #
    # ALSO: when the user changes the high-level filter context
    # (school / conference / season / position), clear any sticky
    # click-detail markers so navigating around always lands in the
    # leaderboard view for the new context, not in a stale player.
    _filter_ctx = (selected_school, selected_conf,
                    int(selected_college_season) if selected_college_season else None,
                    selected_position)
    _prev_ctx = st.session_state.get("_college_filter_ctx")
    if _prev_ctx is not None and _prev_ctx != _filter_ctx:
        for _p in POSITION_FILES:
            st.session_state.pop(f"lb_selected_{_p}", None)
    st.session_state._college_filter_ctx = _filter_ctx

    college_search_target = st.session_state.get("expand_college_player")
    detail_pos, detail_player = None, None
    for _p, _ in positions_to_render:
        _sel = st.session_state.get(f"lb_selected_{_p}")
        if _sel:
            detail_pos, detail_player = _p, _sel
            break

    if detail_pos:
        if st.button("← Back to leaderboards", key="back_to_lb_master", type="primary"):
            for _p, _ in positions_to_render:
                st.session_state.pop(f"lb_selected_{_p}", None)
            st.rerun()
    elif college_search_target:
        # Search-filter active but no detail click — show a clear banner
        # so the user knows the leaderboard is narrowed and can drop the filter.
        b_msg, b_btn = st.columns([5, 1])
        with b_msg:
            st.info(
                f"🔍 Filtered to **{college_search_target}** — click their name "
                f"in the leaderboard to view their profile."
            )
        with b_btn:
            if st.button("❌ Clear filter", key="clear_college_search_filter"):
                st.session_state.pop("expand_college_player", None)
                st.rerun()

    for pos_name, (fname, z_cols, labels) in positions_to_render:
        # In detail mode, only the position that owns the selected player
        # renders anything. Other positions skip entirely so the page is
        # just header + back button + that one player's profile.
        if detail_pos and pos_name != detail_pos:
            continue

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

        # Player-search overrides the volume + prospects filters.
        # Searching for someone by name should always show that player,
        # otherwise a default volume floor (e.g., TE >= 10 receptions)
        # silently strips low-snap players from the search result and
        # leaves the user staring at a "Filtered to X" banner with no
        # row to click.
        if college_search_target:
            filtered = filtered[filtered["player"] == college_search_target]
        else:
            # Volume filter. In single-position mode the user sets
            # `min_volume` above; in all-positions mode we fall back to
            # each position's default floor from POS_VOLUME.
            if all_pos_mode_college:
                pos_vol = POS_VOLUME.get(pos_name)
                if pos_vol:
                    _vc, _, _vd, _, _, _ = pos_vol
                    if _vc in filtered.columns and _vd > 0:
                        filtered = filtered[filtered[_vc].fillna(0) >= _vd]
            else:
                if vol_col and vol_col in filtered.columns and min_volume > 0:
                    filtered = filtered[filtered[vol_col].fillna(0) >= min_volume]

            # 2026 draft prospect filter — keep only rows whose (last
            # name, team) appears in the prospect set. Also bypassed
            # when searching by name.
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
                       "RB": "Running Backs", "OL": "Offensive Linemen",
                       "DE": "Defensive Ends", "DT": "Defensive Tackles",
                       "LB": "Linebackers", "CB": "Cornerbacks", "S": "Safeties"}[pos_name]
        pos_col = "pos_group" if pos_name in COLLEGE_DEF_POS_FILTER and "pos_group" in filtered.columns else None

        active_marker = " ← your sliders" if is_active else ""
        in_detail_mode_here = (detail_pos == pos_name)
        # Header style: in detail mode, lead with the player name; in
        # browse mode, the position group name.
        if in_detail_mode_here:
            st.markdown(f"#### {detail_player} — {pos_display}")
        else:
            st.markdown(f"#### {pos_display}{active_marker}")

        if not in_detail_mode_here:
            # Per-position metric picker — sort the leaderboard by any nerd stat.
            # Skipped for OL since there are no per-player stats; it sorts by
            # weight (largest first) so the leaderboard surfaces draftable size.
            position_metrics = {}
            for z_col, label in labels.items():
                raw_col = z_col.replace("_z", "")
                if z_col in filtered.columns:
                    position_metrics[label + " (z)"] = (z_col, False)
                if raw_col in filtered.columns:
                    position_metrics[label] = (raw_col, False)
            col_metric_label, col_sort, col_asc = _metric_picker(
                position_metrics,
                default_label=score_label_text if score_col != "composite_z" else "Your score",
                key=f"college_metric_{pos_name}",
                label=f"🔍 Sort {pos_display.lower()} by",
            )
            # OL: prominent caption that every metric below is team-level.
            if pos_name == "OL":
                st.caption(
                    "ℹ️ **OL metrics are team-level** — every OL on a team gets the same z-score. "
                    "Use these to find players on quality OL units, then combine with measurables "
                    "(height/weight/40) and recruiting stars to evaluate the individual player."
                )
            # If user picked "Your score" but there's no slider score, fall back to composite_z
            if col_sort == "score":
                col_sort = score_col
            if col_sort in filtered.columns:
                filtered = filtered.sort_values(col_sort, ascending=col_asc, na_position="last")
            else:
                filtered = filtered.sort_values(score_col, ascending=False)

            # Cap the leaderboard to top 6 — keeps the page short on
            # mobile and matches Brett's "half a dozen options" request.
            # Specific-school views still show the full school roster.
            if school_is_all:
                filtered = filtered.head(6)

        if in_detail_mode_here:
            # Detail-only view: skip the leaderboard build/render entirely
            # and render just this one player's profile below.
            if detail_player in filtered["player"].values:
                detail_rows = filtered[filtered["player"] == detail_player]
            else:
                # Player not in current filter (probably a stale click
                # from before the user changed school/conference/etc.).
                # Silently clear the marker and rerun in browse mode —
                # better UX than showing a warning the user has to dismiss.
                st.session_state.pop(f"lb_selected_{pos_name}", None)
                st.rerun()
        else:
            # ── Browse view: build the leaderboard rows ──
            display_rows = []
            for _, row in filtered.iterrows():
                name = row.get("player", "—")
                comp = row.get(score_col, np.nan)
                pct = zscore_to_percentile(comp)

                entry = {"Player": name}
                if pos_col and pd.notna(row.get(pos_col)):
                    entry["Pos"] = row[pos_col]

                rec = get_recruiting_info(name)
                if rec is not None and pd.notna(rec.get("stars")):
                    entry["Stars"] = star_display(rec["stars"])

                if rec is not None and pd.notna(rec.get("recruit_year")):
                    elig_year = int(rec["recruit_year"]) + 3
                    if elig_year <= 2025:
                        entry["Elig"] = f"✅ {elig_year}"
                    elif elig_year == 2026:
                        entry["Elig"] = f"🟡 {elig_year}"
                    else:
                        entry["Elig"] = f"🔒 {elig_year}"

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

            # ── Click-to-detail leaderboard ────────────
            _lb_sel_key = f"lb_selected_{pos_name}"
            if display_rows:
                _PREFERRED = ["Player", "Pos", "Stars", "Elig", "Ht", "Wt", "40",
                              score_label_text, "Pctl"]
                _all_keys = set().union(*(d.keys() for d in display_rows))
                visible_cols = [c for c in _PREFERRED if c in _all_keys]
                visible_cols += [c for c in display_rows[0].keys() if c not in visible_cols and c in _all_keys]

                def _w(c):
                    if c == "Player": return 2.4
                    if c in ("Pos", "Stars", "Elig"): return 0.7
                    if c in ("Ht", "Wt", "40"): return 0.6
                    if c in (score_label_text, "Pctl"): return 0.8
                    return 0.9
                weights = [0.4] + [_w(c) for c in visible_cols]

                _hdrs = st.columns(weights)
                _hdrs[0].markdown("**#**")
                for _i_h, _c_h in enumerate(visible_cols):
                    _hdrs[_i_h + 1].markdown(f"**{_c_h}**")

                for _i_r, _row_data in enumerate(display_rows):
                    _row_cols = st.columns(weights)
                    _row_cols[0].markdown(f"_{_i_r + 1}_")
                    for _j, _c in enumerate(visible_cols):
                        _val = _row_data.get(_c, "—")
                        _val_str = "—" if _val is None or (isinstance(_val, float) and pd.isna(_val)) else str(_val)
                        if _c == "Player":
                            if _row_cols[_j + 1].button(
                                _val_str,
                                key=f"lb_btn_{pos_name}_{_i_r}_{_val_str}",
                                type="tertiary",
                                use_container_width=True,
                            ):
                                st.session_state[_lb_sel_key] = _val_str
                                st.rerun()
                        else:
                            _row_cols[_j + 1].markdown(_val_str)

            # In browse mode, no detail card — user has to click a name.
            detail_rows = filtered.iloc[0:0]
            st.caption(f"💡 Click any **player name** above to see their full profile.")

        for _, row in detail_rows.iterrows():
            name = row.get("player", "—")
            comp = row.get("composite_z", np.nan)
            pos_tag = f" ({row[pos_col]})" if pos_col and pd.notna(row.get(pos_col)) else ""
            score_tag = f" · {comp:+.2f}" if pd.notna(comp) else ""

            # Recruiting badge
            rec = get_recruiting_info(name)
            stars_tag = f" {star_display(rec['stars'])}" if rec is not None and pd.notna(rec.get("stars")) else ""

            # Header used to be the expander label; now it's an h3 above the card.
            st.markdown(f"### {name}{pos_tag}{score_tag}{stars_tag}")

            # ── Counting-stats line (right under the name) ──
            # Per-position raw season totals so users get an at-a-glance
            # read on volume + production before the radar/stats below.
            _COUNTING_BY_POS = {
                "QB": [("pass_att", "{:.0f}", "att"),
                       ("pass_yards", "{:.0f}", "yds"),
                       ("pass_tds", "{:.0f}", "TD"),
                       ("pass_ints", "{:.0f}", "INT"),
                       ("rush_yards_total", "{:.0f}", "rush yds")],
                "WR": [("receptions_total", "{:.0f}", "rec"),
                       ("rec_yards_total", "{:.0f}", "yds"),
                       ("rec_tds_total", "{:.0f}", "TD")],
                "TE": [("receptions_total", "{:.0f}", "rec"),
                       ("rec_yards_total", "{:.0f}", "yds"),
                       ("rec_tds_total", "{:.0f}", "TD")],
                "RB": [("carries_total", "{:.0f}", "car"),
                       ("rush_yards_total", "{:.0f}", "rush yds"),
                       ("rush_tds_total", "{:.0f}", "rush TD"),
                       ("receptions_total", "{:.0f}", "rec")],
                "DE": [("games", "{:.0f}", "G"),
                       ("tackles_total", "{:.0f}", "tkl"),
                       ("sacks", "{:.1f}", "sk"),
                       ("tfl", "{:.1f}", "TFL"),
                       ("qb_hurries", "{:.0f}", "QH")],
                "DT": [("games", "{:.0f}", "G"),
                       ("tackles_total", "{:.0f}", "tkl"),
                       ("sacks", "{:.1f}", "sk"),
                       ("tfl", "{:.1f}", "TFL"),
                       ("qb_hurries", "{:.0f}", "QH")],
                "LB": [("games", "{:.0f}", "G"),
                       ("tackles_total", "{:.0f}", "tkl"),
                       ("tfl", "{:.1f}", "TFL"),
                       ("sacks", "{:.1f}", "sk"),
                       ("interceptions", "{:.0f}", "INT"),
                       ("passes_deflected", "{:.0f}", "PD")],
                "CB": [("games", "{:.0f}", "G"),
                       ("tackles_total", "{:.0f}", "tkl"),
                       ("interceptions", "{:.0f}", "INT"),
                       ("passes_deflected", "{:.0f}", "PD"),
                       ("tfl", "{:.1f}", "TFL")],
                "S":  [("games", "{:.0f}", "G"),
                       ("tackles_total", "{:.0f}", "tkl"),
                       ("interceptions", "{:.0f}", "INT"),
                       ("passes_deflected", "{:.0f}", "PD"),
                       ("tfl", "{:.1f}", "TFL")],
                "OL": [("class_year", "{:.0f}", "yr"),
                       ("height", "{:.0f}", "in"),
                       ("weight", "{:.0f}", "lbs")],
            }
            _counting_parts = []
            for _col, _fmt, _lbl in _COUNTING_BY_POS.get(pos_name, []):
                _v = row.get(_col)
                if pd.notna(_v):
                    try:
                        _counting_parts.append(f"{_fmt.format(_v)} {_lbl}")
                    except (ValueError, TypeError):
                        pass
            if _counting_parts:
                _season = int(row.get(season_col)) if pd.notna(row.get(season_col)) else ""
                _team = row.get("team", "")
                _ctx = " · ".join(x for x in [str(_season) if _season else "", _team] if x)
                _ctx_str = f"**{_ctx}** · " if _ctx else ""
                st.caption(f"{_ctx_str}{' · '.join(_counting_parts)}")

            with st.container(border=True):

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
                    # ── Radar chart (factored helper) ─────
                    if z_cols:
                        if pos_name == "OL":
                            st.caption(
                                "📋 _Radar shows **team-level** OL unit metrics "
                                "(every OL on this team-season has the same z-scores). "
                                "Use it to gauge the quality of the unit, not the individual._"
                            )
                        _render_player_radar(
                            player_name=name,
                            key_prefix=f"{pos_name}_{name}",
                            full_df=df, labels_dict=labels, season_col=season_col,
                            pos_name=pos_name, pos_display=pos_display,
                            default_season=selected_college_season,
                        )

                # ── Compare to another player (radar + line chart) ──
                compare_active = st.checkbox(
                    "🔍 Compare to another player",
                    key=f"compare_active_{pos_name}_{name}",
                    help=("Stack a second player's radar and line chart "
                          "below the originals. Pick the same player to "
                          "compare different seasons on the radar."),
                )
                compare_name = None
                if compare_active:
                    # Same-position pool only — radar axes are position-specific.
                    pos_players = sorted(set(df["player"].dropna().unique()))
                    if pos_players:
                        # Default: first non-current player so the UI is
                        # useful immediately; user can also pick the same
                        # player to compare different seasons.
                        default_compare = next(
                            (p for p in pos_players if p != name), pos_players[0]
                        )
                        compare_name = st.selectbox(
                            f"Comparison {pos_display[:-1].lower() if pos_display.endswith('s') else pos_display.lower()}",
                            options=pos_players,
                            index=pos_players.index(default_compare),
                            key=f"compare_select_{pos_name}_{name}",
                        )
                    else:
                        st.caption("No other players available to compare.")

                # ── Compare radar (stacked directly below radar A) ──
                if compare_name:
                    cmp_pc1, cmp_pc2 = st.columns([1, 1])
                    with cmp_pc1:
                        # Compact profile strip for player B so the
                        # stats-vs-radar layout matches player A's row.
                        cmp_career = df[df["player"] == compare_name]
                        if len(cmp_career) > 0:
                            latest_season = int(cmp_career[season_col].max())
                            latest_row = cmp_career[cmp_career[season_col] == latest_season].iloc[0]
                            cmp_school = latest_row.get("team", "")
                            cmp_conf = latest_row.get("conference", "")
                            st.markdown(
                                f"**{compare_name}** — {latest_season} {cmp_school}"
                                + (f"  \n_{cmp_conf}_" if cmp_conf else "")
                            )
                            cmp_avail_z = [c for c in z_cols if c in cmp_career.columns]
                            if cmp_avail_z:
                                cmp_comp = latest_row[cmp_avail_z].mean()
                                if pd.notna(cmp_comp):
                                    cmp_pct = zscore_to_percentile(cmp_comp)
                                    st.caption(
                                        f"{latest_season} composite: {cmp_comp:+.2f} "
                                        f"({format_percentile(cmp_pct)} of FBS {pos_display.lower()})"
                                    )
                    with cmp_pc2:
                        if z_cols:
                            _render_player_radar(
                                player_name=compare_name,
                                key_prefix=f"{pos_name}_{compare_name}_cmp_for_{name}",
                                full_df=df, labels_dict=labels, season_col=season_col,
                                pos_name=pos_name, pos_display=pos_display,
                                default_season=selected_college_season,
                                header_prefix="Comparison:",
                            )

                # ── Career line chart with metric dropdown ─
                # Skipped for OL (no z_cols → would draw an empty NaN line).
                all_player_career = df[df["player"] == name].sort_values(season_col)
                if z_cols and len(all_player_career) >= 2:
                    st.markdown(f"**College career arc — {name}**")
                    build_career_chart(name, df, season_col, z_cols, labels,
                                        unique_key=f"{pos_name}_{name}",
                                        pos_name=pos_name, pos_display=pos_display)
                    st.caption(
                        f"Each point = one season vs. all FBS players at this position. 0.00 = FBS average. "
                        f"Dashed gray = {_starter_label(pos_display).lower()} (median of top {COLLEGE_RADAR_TOP_N} starters per season)."
                    )

                # ── Compare line chart (stacked below line chart A) ──
                if compare_name and z_cols:
                    cmp_career = df[df["player"] == compare_name].sort_values(season_col)
                    if len(cmp_career) >= 2:
                        st.markdown(f"**College career arc — {compare_name}** (comparison)")
                        build_career_chart(
                            compare_name, df, season_col, z_cols, labels,
                            unique_key=f"{pos_name}_{compare_name}_cmp_for_{name}",
                            pos_name=pos_name, pos_display=pos_display,
                        )
                    elif len(cmp_career) == 1:
                        st.caption(f"{compare_name} has only one season of data — no career arc to plot.")

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
