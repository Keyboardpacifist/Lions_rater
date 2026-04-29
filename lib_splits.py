"""
Game-by-game splits explorer for player detail pages.

Renders an expandable section that:
  • surfaces three "headline" tiles — recent form, schedule strength,
    consistency — schedule-adjusted using the precomputed baselines
  • lets the user slice the player's games by opponent defense tier,
    roof, surface, weather, home/away, and game outcome
  • shows the filtered summary + game-by-game table

Reads from the precomputed parquets in data/games/. If those files are
missing the helper renders nothing (so the rest of the page survives).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lib_data_remote import get_parquet_path

_DATA_GAMES = Path(__file__).resolve().parent / "data" / "games"
_ADJUSTED = _DATA_GAMES / "nfl_weekly_adjusted.parquet"
_DEF_PLAYER_ADJ = _DATA_GAMES / "nfl_defensive_player_adjusted.parquet"
_BASELINES = _DATA_GAMES / "nfl_defense_baselines.parquet"
_OFF_BASELINES = _DATA_GAMES / "nfl_offense_baselines.parquet"
_SCHEDULES = _DATA_GAMES / "nfl_schedules.parquet"
_DEF_SCHEME_GAME = _DATA_GAMES / "nfl_defense_game_scheme.parquet"
_EXPL_PLAYER_GAMES = _DATA_GAMES / "nfl_explosive_player_games.parquet"
_EXPL_DEF_BASELINES = _DATA_GAMES / "nfl_explosive_def_baselines.parquet"
_ADV_PLAYER_GAMES = _DATA_GAMES / "nfl_advanced_player_games.parquet"
_ADV_DEF_BASELINES = _DATA_GAMES / "nfl_advanced_def_baselines.parquet"
_OFF_SCHEME_GAME = _DATA_GAMES / "nfl_offense_game_scheme.parquet"
_ROUTE_PLAYER_GAMES = _DATA_GAMES / "nfl_route_distribution_player_games.parquet"
_TARGETED_PLAYS = _DATA_GAMES / "nfl_targeted_plays.parquet"
_RUSHER_PLAYS = _DATA_GAMES / "nfl_rusher_plays.parquet"


# ──────────────────────────────────────────────────────────────
# Per-position config — what stats matter for the headline tiles
# and the filtered summary.
# ──────────────────────────────────────────────────────────────
POSITION_CONFIG = {
    "RB": {
        "headline_actual": "rushing_yards",
        "headline_label": "Rush yds",
        "headline_delta": "rushing_yards_delta",
        "headline_expected": "rushing_yards_expected",
        "headline_unit": "yds",
        "summary_stats": [
            ("carries", "Car", "{:.1f}"),
            ("rushing_yards", "Rush yds", "{:.1f}"),
            ("yards_per_carry", "YPC", "{:.2f}"),
            ("rushing_tds", "Rush TD", "{:.2f}"),
            ("rushing_epa", "Rush EPA/g", "{:+.2f}"),
            ("explosive_runs", "Chunk runs", "{:.1f}"),
            ("targets", "Tgt", "{:.1f}"),
            ("receptions", "Rec", "{:.1f}"),
            ("receiving_yards", "Rec yds", "{:.1f}"),
            ("receiving_tds", "Rec TD", "{:.2f}"),
            ("yards_per_target", "Y/Tgt", "{:.2f}"),
            ("receiving_epa", "Rec EPA/g", "{:+.2f}"),
            ("explosive_receptions", "Chunk recs", "{:.1f}"),
        ],
        "tier_metric": "yards_per_carry",  # lower = tougher D (col on baselines parquet)
    },
    "QB": {
        "headline_actual": "passing_yards",
        "headline_label": "Pass yds",
        "headline_delta": "passing_yards_delta",
        "headline_expected": "passing_yards_expected",
        "headline_unit": "yds",
        "summary_stats": [
            ("attempts", "Att", "{:.1f}"),
            ("completions", "Cmp", "{:.1f}"),
            ("passing_yards", "Pass yds", "{:.1f}"),
            ("passing_tds", "Pass TD", "{:.2f}"),
            ("passing_interceptions", "INT", "{:.2f}"),
            ("passing_epa", "Pass EPA/g", "{:+.2f}"),
            ("yards_per_attempt", "Y/Att", "{:.2f}"),
            ("completion_pct", "Cmp%", "{:.1%}"),
            ("chunk_completions", "Chunk cmp", "{:.1f}"),
            ("deep_attempts", "Deep att", "{:.1f}"),
            ("td_long_passes", "20+ TD", "{:.2f}"),
            ("scramble_first_downs", "Scram 1D", "{:.2f}"),
            ("carries", "Carries", "{:.1f}"),
            ("rushing_yards", "Rush yds", "{:.1f}"),
        ],
        "tier_metric": "passing_epa",
    },
    "WR": {
        "headline_actual": "receiving_yards",
        "headline_label": "Rec yds",
        "headline_delta": "receiving_yards_delta",
        "headline_expected": "receiving_yards_expected",
        "headline_unit": "yds",
        "summary_stats": [
            ("targets", "Tgt", "{:.1f}"),
            ("receptions", "Rec", "{:.1f}"),
            ("receiving_yards", "Rec yds", "{:.1f}"),
            ("receiving_tds", "Rec TD", "{:.2f}"),
            ("yards_per_target", "Y/Tgt", "{:.2f}"),
            ("catch_rate", "Catch%", "{:.1%}"),
            ("receiving_epa", "EPA/g", "{:+.2f}"),
            ("explosive_receptions", "Chunk rec", "{:.1f}"),
            ("yac_chunks", "YAC chunk", "{:.1f}"),
            ("deep_targets", "Deep tgt", "{:.1f}"),
            ("rz_targets", "RZ tgt", "{:.1f}"),
            ("first_down_recs", "1D rec", "{:.1f}"),
            ("target_share", "Tgt%", "{:.1%}"),
        ],
        "tier_metric": "yards_per_target",
    },
    "TE": {
        "headline_actual": "receiving_yards",
        "headline_label": "Rec yds",
        "headline_delta": "receiving_yards_delta",
        "headline_expected": "receiving_yards_expected",
        "headline_unit": "yds",
        "summary_stats": [
            ("targets", "Tgt", "{:.1f}"),
            ("receptions", "Rec", "{:.1f}"),
            ("receiving_yards", "Rec yds", "{:.1f}"),
            ("receiving_tds", "Rec TD", "{:.2f}"),
            ("yards_per_target", "Y/Tgt", "{:.2f}"),
            ("catch_rate", "Catch%", "{:.1%}"),
            ("receiving_epa", "EPA/g", "{:+.2f}"),
            ("explosive_receptions", "Chunk rec", "{:.1f}"),
            ("yac_chunks", "YAC chunk", "{:.1f}"),
            ("deep_targets", "Deep tgt", "{:.1f}"),
            ("rz_targets", "RZ tgt", "{:.1f}"),
            ("first_down_recs", "1D rec", "{:.1f}"),
            ("target_share", "Tgt%", "{:.1%}"),
        ],
        "tier_metric": "yards_per_target",
    },
    # ── Defensive positions: schedule-strength flips. Tier metric
    # is the OFFENSE the defender faced (high = tougher offense).
    # `side: defense` tells render_splits_section to load the
    # defensive_player_adjusted parquet instead of the offensive one.
    "DE": {
        "side": "defense",
        "headline_actual": "def_sacks",
        "headline_label": "Sacks",
        "headline_delta": "def_sacks_delta",
        "headline_expected": "def_sacks_expected",
        "headline_unit": "sacks",
        "summary_stats": [
            ("def_tackles", "Tkl", "{:.1f}"),
            ("def_sacks", "Sacks", "{:.2f}"),
            ("def_qb_hits", "QB hits", "{:.1f}"),
            ("def_tackles_for_loss", "TFL", "{:.1f}"),
            ("def_pass_defended", "PD", "{:.1f}"),
            ("def_fumbles_forced", "FF", "{:.2f}"),
            ("def_interceptions", "INT", "{:.2f}"),
        ],
        "tier_metric": "def_sacks",  # easier offenses give up more sacks
    },
    "DT": {
        "side": "defense",
        "headline_actual": "def_tackles",
        "headline_label": "Tackles",
        "headline_delta": "def_tackles_delta",
        "headline_expected": "def_tackles_expected",
        "headline_unit": "tkl",
        "summary_stats": [
            ("def_tackles", "Tkl", "{:.1f}"),
            ("def_tackles_for_loss", "TFL", "{:.1f}"),
            ("def_sacks", "Sacks", "{:.2f}"),
            ("def_qb_hits", "QB hits", "{:.1f}"),
            ("def_pass_defended", "PD", "{:.1f}"),
            ("def_fumbles_forced", "FF", "{:.2f}"),
        ],
        "tier_metric": "def_sacks",
    },
    "LB": {
        "side": "defense",
        "headline_actual": "def_tackles",
        "headline_label": "Tackles",
        "headline_delta": "def_tackles_delta",
        "headline_expected": "def_tackles_expected",
        "headline_unit": "tkl",
        "summary_stats": [
            ("def_tackles", "Tkl", "{:.1f}"),
            ("def_tackles_for_loss", "TFL", "{:.2f}"),
            ("def_sacks", "Sacks", "{:.2f}"),
            ("def_qb_hits", "QB hits", "{:.1f}"),
            ("def_pass_defended", "PD", "{:.1f}"),
            ("def_interceptions", "INT", "{:.2f}"),
            ("def_fumbles_forced", "FF", "{:.2f}"),
        ],
        "tier_metric": "def_tackles",
    },
    "CB": {
        "side": "defense",
        "headline_actual": "def_pass_defended",
        "headline_label": "Pass def",
        "headline_delta": "def_pass_defended_delta",
        "headline_expected": "def_pass_defended_expected",
        "headline_unit": "PD",
        "summary_stats": [
            ("def_tackles", "Tkl", "{:.1f}"),
            ("def_pass_defended", "PD", "{:.2f}"),
            ("def_interceptions", "INT", "{:.2f}"),
            ("def_fumbles_forced", "FF", "{:.2f}"),
            ("def_tackles_for_loss", "TFL", "{:.2f}"),
        ],
        "tier_metric": "def_pass_defended",
    },
    "S": {
        "side": "defense",
        "headline_actual": "def_tackles",
        "headline_label": "Tackles",
        "headline_delta": "def_tackles_delta",
        "headline_expected": "def_tackles_expected",
        "headline_unit": "tkl",
        "summary_stats": [
            ("def_tackles", "Tkl", "{:.1f}"),
            ("def_pass_defended", "PD", "{:.2f}"),
            ("def_interceptions", "INT", "{:.2f}"),
            ("def_fumbles_forced", "FF", "{:.2f}"),
            ("def_tackles_for_loss", "TFL", "{:.2f}"),
            ("def_sacks", "Sacks", "{:.2f}"),
        ],
        "tier_metric": "def_tackles",
    },
}


# All loaders go through the remote helper: locally they pick up the
# file from data/games/; on Streamlit Cloud they download once from
# Supabase Storage and cache. Returning None in either path keeps every
# downstream renderer's silent-skip branch working.
def _read_remote(filename: str):
    p = get_parquet_path(filename)
    if p is None:
        return None
    return pd.read_parquet(p)


@st.cache_data
def _load_adjusted():
    return _read_remote("nfl_weekly_adjusted.parquet")


@st.cache_data
def _load_defensive_player_adjusted():
    return _read_remote("nfl_defensive_player_adjusted.parquet")


@st.cache_data
def _load_offense_baselines():
    return _read_remote("nfl_offense_baselines.parquet")


@st.cache_data
def _load_baselines():
    return _read_remote("nfl_defense_baselines.parquet")


@st.cache_data
def _load_schedules():
    return _read_remote("nfl_schedules.parquet")


@st.cache_data
def _load_def_scheme_game():
    """Per-game defensive scheme profile (box, blitz, man/zone, shells,
    personnel). Optional — older seasons (2016-17) have a subset of
    these fields; 2018+ has the full set."""
    return _read_remote("nfl_defense_game_scheme.parquet")


@st.cache_data
def _load_explosive_player_games():
    """Per (player_id, season, week, team) explosive run + reception
    counts. Industry standard cuts: rush ≥10 yds, rec ≥20 yds."""
    return _read_remote("nfl_explosive_player_games.parquet")


@st.cache_data
def _load_explosive_def_baselines():
    """Per (defense_team, season, position) avg explosive plays
    allowed per qualifying player-game."""
    return _read_remote("nfl_explosive_def_baselines.parquet")


@st.cache_data
def _load_advanced_player_games():
    """Per-(player, game) advanced offensive metrics: chunk completions,
    deep attempts, YAC chunks, RZ targets, etc."""
    return _read_remote("nfl_advanced_player_games.parquet")


@st.cache_data
def _load_advanced_def_baselines():
    """Per (defense_team, season, position) avg advanced metrics
    allowed per qualifying player-game."""
    return _read_remote("nfl_advanced_def_baselines.parquet")


@st.cache_data
def _load_offense_game_scheme():
    """Per-game OFFENSIVE scheme profile: shotgun rate, personnel
    breakdown, pass rate, deep attempt rate, etc. Used as 'own
    offense' context on offensive player pages."""
    return _read_remote("nfl_offense_game_scheme.parquet")


@st.cache_data
def _load_route_distribution():
    """Per (player_id, season, week, team) targeted-route counts —
    rt_go / rt_hitch / rt_slant / rt_out / etc."""
    return _read_remote("nfl_route_distribution_player_games.parquet")


def _add_fo_success_column(df):
    """Add an `fo_success` column to a per-play dataframe — Football
    Outsiders / PFR convention so panel-side aggregations align with
    the league parquets and with PFF/PFR.

        1st: yards_gained ≥ 40% of yards-to-go
        2nd: yards_gained ≥ 60%
        3rd / 4th: full conversion (yards_gained ≥ ydstogo)

    No-op if down/ydstogo/yards_gained aren't all present."""
    if df is None or df.empty:
        return df
    needed = {"down", "ydstogo", "yards_gained"}
    if not needed.issubset(df.columns):
        return df
    down = df["down"]
    ytg = df["ydstogo"]
    yg = df["yards_gained"]
    threshold = pd.Series(float("nan"), index=df.index)
    threshold = threshold.where(~down.eq(1), 0.4 * ytg)
    threshold = threshold.where(~down.eq(2), 0.6 * ytg)
    threshold = threshold.where(~down.isin([3, 4]), ytg)
    df = df.copy()
    df["fo_success"] = (yg >= threshold).astype(float)
    return df


@st.cache_data
def _load_targeted_plays():
    """Per-targeted-play feed: route × man/zone × coverage shell ×
    result. Used to build the coverage matchup profile per receiver.

    Adds `fo_success` (PFF/PFR convention) so panel aggregations
    align with the league parquets. The original `success` column
    (nflverse EPA-based) stays available for any consumer that
    wants the rigorous version."""
    return _add_fo_success_column(_read_remote("nfl_targeted_plays.parquet"))


@st.cache_data
def _load_rusher_plays():
    """Per-run-play feed: gap / direction / box / formation / personnel
    + result. Used to build the run scheme profile per RB.

    Adds `fo_success` for PFF/PFR alignment (see _load_targeted_plays
    for rationale)."""
    return _add_fo_success_column(_read_remote("nfl_rusher_plays.parquet"))


def _data_ready() -> bool:
    """True if the core parquets are obtainable (locally OR remotely).
    On production, get_parquet_path triggers a download which is then
    cached, so this becomes True once Supabase has the files."""
    return all(get_parquet_path(f) is not None for f in (
        "nfl_weekly_adjusted.parquet",
        "nfl_defense_baselines.parquet",
        "nfl_schedules.parquet",
    ))


# ──────────────────────────────────────────────────────────────
# Defense tiers — rank each defense within (season, position) by
# the position-appropriate "expected" metric. Higher tier_pct = tougher.
# ──────────────────────────────────────────────────────────────
@st.cache_data
def _build_tier_lookup(season: int, position_group: str, metric: str,
                        side: str = "offense"):
    """{team: tier_percentile} for one (season, position).

    For offensive positions (side='offense'): tier the DEFENSE that
    season — lower expected stat allowed = tougher D = higher tier_pct.

    For defensive positions (side='defense'): tier the OFFENSE that
    season — higher expected stat (= more sacks/PDs/etc generated by
    opposing defenders) means a more vulnerable offense. So we still
    rank ascending and flip, so HIGHER tier_pct = TOUGHER offense (one
    that gives up fewer of these defensive events).
    """
    if side == "defense":
        bases = _load_offense_baselines()
        team_col = "offense_team"
    else:
        bases = _load_baselines()
        team_col = "defense_team"
    if bases is None:
        return {}
    pool = bases[(bases["season"] == season)
                 & (bases["position"] == position_group)
                 & bases[metric].notna()].copy()
    if pool.empty:
        return {}
    pool = pool.sort_values(metric, ascending=True).reset_index(drop=True)
    n = len(pool)
    pool["tier_pct"] = ((n - 1 - pool.index) / max(n - 1, 1)) * 100
    return dict(zip(pool[team_col], pool["tier_pct"]))


def _tier_label(pct: float) -> str:
    if pd.isna(pct):
        return "—"
    if pct >= 90:
        return "🟥 Top 10%"
    if pct >= 75:
        return "🟧 Top 25%"
    if pct >= 50:
        return "🟨 Top half"
    if pct >= 25:
        return "🟩 Bottom half"
    return "🟦 Bottom 25%"


# ──────────────────────────────────────────────────────────────
# Filter helpers
# ──────────────────────────────────────────────────────────────
def _classify_weather(temp, wind):
    """Bucket each game into a weather category."""
    if pd.isna(temp) and pd.isna(wind):
        return "—"  # indoor / no weather captured
    if not pd.isna(temp) and temp < 40:
        return "Cold (<40°)"
    if not pd.isna(wind) and wind >= 15:
        return "Windy (>15mph)"
    return "Mild / clear"


def _classify_roof(roof):
    if pd.isna(roof):
        return "—"
    return "Indoor" if roof in ("dome", "closed") else "Outdoor"


def _classify_surface(surface):
    if pd.isna(surface) or surface == "":
        return "—"
    s = str(surface).lower()
    if "grass" in s:
        return "Grass"
    if any(x in s for x in ("turf", "fieldturf", "sportsturf", "matrixturf",
                              "astroplay", "a_turf", "astro")):
        return "Turf"
    return surface.title()


# ──────────────────────────────────────────────────────────────
# Headline tiles
# ──────────────────────────────────────────────────────────────
def _style_epa_table(df: pd.DataFrame, epa_col: str):
    """Apply green/red color to a signed-string EPA column (e.g. '+0.74'
    or '-0.18'). Returns a pandas Styler ready for st.dataframe."""
    def _color(v):
        try:
            f = float(str(v).replace("+", ""))
        except (ValueError, TypeError):
            return ""
        if f > 0.05:
            return "color:#0a7a23;font-weight:600"
        if f < -0.05:
            return "color:#b3261e;font-weight:600"
        return ""
    if epa_col not in df.columns:
        return df
    return df.style.map(_color, subset=[epa_col])


def render_coverage_matchup_section(*, player_name: str, season,
                                     position_group: str, key_prefix: str,
                                     is_career_view: bool = False) -> None:
    """Public entry point — renders the coverage matchup panel as an
    exposed top-level section on a player page (not inside an
    expander). The panel has its own season/career dropdown so a user
    can override the page-level pick without leaving the section.
    """
    if not _data_ready():
        return
    if position_group not in ("WR", "TE", "RB"):
        return  # only receivers / pass-catching backs

    adj = _load_adjusted()
    tp = _load_targeted_plays()
    if adj is None or tp is None:
        return

    # Resolve the player_id from the offensive adjusted parquet
    base_mask = ((adj["player_display_name"] == player_name)
                 & (adj["position_group"] == position_group))
    full_career = adj[base_mask]
    if full_career.empty:
        return
    pid_series = full_career["player_id"].dropna()
    if pid_series.empty:
        return
    pid = pid_series.iloc[0]

    # Pull this player's full targeted-play history once — the panel's
    # dropdown filters it down without re-loading.
    pf_all = tp[tp["player_id"] == pid].copy()
    if pf_all.empty:
        return
    pf_all["season"] = pf_all["season"].astype(int)
    pf_all["week"] = pf_all["week"].astype(int)

    # Resolve the player's most recent team for theme accenting on
    # the narrative blurb. Uses the latest season's most-frequent team.
    team_abbr = None
    if "team" in full_career.columns:
        latest_season = full_career["season"].max()
        latest_rows = full_career[full_career["season"] == latest_season]
        team_counts = latest_rows["team"].dropna().value_counts()
        if not team_counts.empty:
            team_abbr = str(team_counts.index[0])
    from lib_shared import team_theme
    theme = team_theme(team_abbr)

    # Render the header FIRST, so the dropdown appears below the
    # section title (cleaner UX than dropdown-above-heading).
    _coverage_matchup_header()

    seasons_present = sorted(pf_all["season"].unique().tolist(), reverse=True)
    default_label = ("All career" if is_career_view
                     else f"Season {int(season)}"
                          if season in seasons_present
                          else f"Season {seasons_present[0]}")
    options = [f"Season {s}" for s in seasons_present] + ["All career"]

    pf = _render_panel_view_picker(
        options, default_label, key=f"{key_prefix}_view_pick", pf_all=pf_all
    )
    if pf is None or len(pf) < 5:
        return  # too few targets — would be noise

    _render_coverage_matchup_panel(pf, key_prefix, wrap_in_expander=False,
                                     render_header=False, theme=theme)


def _render_panel_view_picker(options, default_label, *, key, pf_all):
    """Render the in-panel 'Season N / All career' dropdown and return
    the filtered DataFrame to use. Lives outside the panel rendering
    so both reception + rushing panels share it."""
    try:
        idx = options.index(default_label)
    except ValueError:
        idx = 0
    pick = st.selectbox("View", options, index=idx, key=key,
                         help="Switch this panel between a single season "
                              "and the player's full career — independent "
                              "of the page-level year picker above.")
    if pick == "All career":
        return pf_all
    try:
        season_int = int(pick.replace("Season ", ""))
    except ValueError:
        return pf_all
    return pf_all[pf_all["season"] == season_int]


def _coverage_matchup_header():
    """Section header markdown — pulled out so the public section
    can show it BEFORE the in-panel season/career dropdown."""
    st.markdown(
        "<div style='margin:18px 0 6px 0;padding-left:12px;"
        "border-left:5px solid #0076B6;'>"
        "<div style='font-size:1.25rem;font-weight:800;color:#0a3d62;"
        "letter-spacing:0.3px;'>🎯 Coverage matchup profile</div>"
        "<div style='font-size:0.78rem;color:#5b6b7e;margin-top:2px;'>"
        "Targeted plays only · 2018+ has full coverage labeling · "
        "non-targeted routes aren't tracked publicly (PFF territory)."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _apply_route_filters(pf: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Top-row filter pills for the coverage matchup panel — Coverage /
    Man-Zone / Pass rush / Down. Returns the filtered slice. All
    filters cascade through the panel.

    (Defense personnel filter dropped — ~30% of plays have null
    defense_personnel labels, which would mislead users by silently
    excluding data. Pass rush is the more reliable defensive filter.)"""
    f1, f2, f4, f5 = st.columns(4)

    # Coverage shell
    canonical_cov = ["Cover-0", "Cover-1", "Cover-2", "Cover-3",
                      "Cover-4", "Cover-6", "Cover-9", "2-Man"]
    if "coverage_shell" in pf.columns:
        present = set(pf["coverage_shell"].dropna().unique())
        cov_options = ["All"] + [c for c in canonical_cov if c in present]
    else:
        cov_options = ["All"]
    with f1:
        cov_pick = st.selectbox(
            "Coverage", cov_options,
            key=f"{key_prefix}_filt_cov",
            help="Coverage shell faced. NGS coverage labels (2018+).",
        )

    # Man / Zone
    mz_options = ["All", "Man", "Zone"]
    with f2:
        mz_pick = st.selectbox(
            "Man / Zone", mz_options,
            key=f"{key_prefix}_filt_mz",
        )

    # Blitz
    canonical_blitz = ["Drop (3 or fewer)", "Standard rush (4)",
                        "Blitz (5)", "Heavy blitz (6+)"]
    if "blitz_bucket" in pf.columns:
        present = set(pf["blitz_bucket"].dropna().unique())
        blitz_options = ["All"] + [b for b in canonical_blitz if b in present]
    else:
        blitz_options = ["All"]
    with f4:
        blitz_pick = st.selectbox(
            "Pass rush", blitz_options,
            key=f"{key_prefix}_filt_blitz",
            help="Number of pass rushers — 4 is standard, 5+ is a blitz.",
        )

    # Down
    down_options = ["All", "1st", "2nd", "3rd", "4th"]
    with f5:
        down_pick = st.selectbox(
            "Down", down_options,
            key=f"{key_prefix}_filt_down",
        )

    # Apply filters
    if cov_pick != "All" and "coverage_shell" in pf.columns:
        pf = pf[pf["coverage_shell"] == cov_pick]
    if mz_pick != "All":
        pf = pf[pf["man_zone"] == mz_pick]
    if blitz_pick != "All" and "blitz_bucket" in pf.columns:
        pf = pf[pf["blitz_bucket"] == blitz_pick]
    if down_pick != "All" and "down" in pf.columns:
        d_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        pf = pf[pf["down"] == d_map[down_pick]]

    return pf


def _render_coverage_matchup_panel(pf: pd.DataFrame, key_prefix: str,
                                     wrap_in_expander: bool = True,
                                     render_header: bool = True,
                                     theme: dict | None = None) -> None:
    """Coverage matchup panel — how this receiver fares vs different
    coverage looks. Two side-by-side views:
      • Route × man/zone target counts (grouped horizontal bars)
      • Performance per coverage shell (table)

    If `wrap_in_expander=False`, renders exposed without a collapsible
    wrapper (so it can be used as a top-level page section). Set
    `render_header=False` if the caller already drew the section
    header (lets the public section render dropdowns between header
    and content).
    """
    import plotly.graph_objects as go

    if wrap_in_expander:
        container = st.expander("🎯 Coverage matchup profile", expanded=False)
    else:
        if render_header:
            _coverage_matchup_header()
        container = st.container()
    with container:
        # Capture the unfiltered career data BEFORE filters narrow pf.
        # The narrative blurb + career-volume rank reflect the player's
        # whole story regardless of what the user filters below.
        pf_unfiltered = pf.copy()
        career_targets = len(pf_unfiltered)
        career_rank_label = ""
        if "player_id" in pf_unfiltered.columns and not pf_unfiltered.empty:
            pid_series = pf_unfiltered["player_id"].dropna()
            if not pid_series.empty:
                volume_pool = _load_wr_volume_pool()
                if volume_pool is not None and not volume_pool.empty:
                    from lib_shared import compute_rank_in_pool, format_rank
                    rank, total = compute_rank_in_pool(
                        career_targets, volume_pool["career_targets"],
                        ascending=False
                    )
                    career_rank_label = format_rank(rank, total)

        # Render the narrative blurb (theme-accented).
        _render_wr_narrative_blurb(pf_unfiltered, theme=theme)

        # Top-row filter pills (Coverage / Man-Zone / Defense / Blitz / Down)
        pf = _apply_route_filters(pf, key_prefix)

        n = len(pf)
        n_man = int((pf["man_zone"] == "Man").sum())
        n_zone = int((pf["man_zone"] == "Zone").sum())
        n_labeled = n_man + n_zone
        if n_labeled == 0:
            st.info("No coverage labels for this slice. "
                    "Coverage data is 2018+ only — try loosening the filters.")
            return

        # ── Top: 3 stat tiles (Targets / Man / Zone) ──
        man_pct = (n_man / n_labeled * 100) if n_labeled else 0
        zone_pct = (n_zone / n_labeled * 100) if n_labeled else 0
        targets_sub = f"{n_labeled} with coverage labeled"
        if career_rank_label and career_rank_label != "—":
            targets_sub += f" · career: {career_rank_label}"
        t1, t2, t3 = st.columns(3)
        for col, label, big, sub, accent in [
            (t1, "Targets", f"{n}", targets_sub, "#0076B6"),
            (t2, "Vs Man", f"{n_man}", f"{man_pct:.0f}% of labeled targets",
             "#d62728"),
            (t3, "Vs Zone", f"{n_zone}", f"{zone_pct:.0f}% of labeled targets",
             "#1f77b4"),
        ]:
            col.markdown(
                f"<div style='background:#f3f6fa;border:1px solid #d6dde6;"
                f"border-left:4px solid {accent};border-radius:8px;"
                f"padding:10px 14px;'>"
                f"<div style='font-size:0.65rem;color:#5b6b7e;letter-spacing:1.2px;"
                f"text-transform:uppercase;font-weight:700;'>{label}</div>"
                f"<div style='font-size:1.7rem;font-weight:900;color:#0a3d62;"
                f"line-height:1.0;margin-top:4px;'>{big}</div>"
                f"<div style='font-size:0.75rem;color:#5b6b7e;margin-top:3px;'>"
                f"{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

        c_left, c_right = st.columns([1, 1])

        # ── LEFT: route tree — every route radiates from the receiver ──
        with c_left:
            st.markdown("**Route tree** _(every route the receiver runs · color = EPA per target)_")
            rt_pool = pf[pf["route"].notna() & (pf["route"] != "")]
            if rt_pool.empty:
                st.caption("_No labeled routes in this slice._")
            else:
                from lib_field_viz import build_route_tree
                fig = build_route_tree(rt_pool, metric="epa_per_target")
                st.plotly_chart(fig, use_container_width=True,
                                  key=f"{key_prefix}_route_tree")
                st.caption(
                    "_Each route line is colored by EPA per target on a heatmap "
                    "and weighted by target volume. Hover for full breakdown. "
                    "Targeted routes only — non-targeted routes (the receiver "
                    "running but not getting the ball) and option-route "
                    "designations require PFF charting._",
                    unsafe_allow_html=True,
                )

        # ── RIGHT: per-route performance table — responds to filters ──
        with c_right:
            st.markdown("**Performance by route** _(this slice)_")
            rt_pool = pf[pf["route"].notna() & (pf["route"] != "")]
            if rt_pool.empty:
                st.caption("_No labeled routes in this slice._")
            else:
                route_rows = []
                for route, grp in rt_pool.groupby("route"):
                    targets = len(grp)
                    if targets < 1:
                        continue
                    catches = int(grp["complete_pass"].fillna(0).sum())
                    yards = float(grp["yards_gained"].fillna(0).sum())
                    tds = int(grp["pass_touchdown"].fillna(0).sum())
                    epa_per_tgt = float(grp["epa"].fillna(0).mean())
                    catch_pct = (catches / targets) if targets else 0
                    ypt = (yards / targets) if targets else 0
                    route_rows.append({
                        "Route": route,
                        "Tgts": targets,
                        "Rec": catches,
                        "Catch%": f"{catch_pct*100:.0f}%",
                        "Y/Tgt": f"{ypt:.1f}",
                        "TD": tds,
                        "EPA/Tgt": f"{epa_per_tgt:+.2f}",
                    })
                if route_rows:
                    rtable = (pd.DataFrame(route_rows)
                                .sort_values("Tgts", ascending=False)
                                .reset_index(drop=True))
                    st.dataframe(_style_epa_table(rtable, "EPA/Tgt"),
                                  use_container_width=True, hide_index=True)
                    st.caption(
                        "_Per-route stats for the current filter slice — "
                        "table updates live when you change Coverage / "
                        "Defense / Pass rush / Down._"
                    )


def render_run_scheme_section(*, player_name: str, season,
                                key_prefix: str,
                                is_career_view: bool = False) -> None:
    """Run scheme profile for an RB. Has its own season/career
    dropdown so a user can override the page-level pick."""
    if not _data_ready():
        return
    adj = _load_adjusted()
    rp = _load_rusher_plays()
    if adj is None or rp is None:
        return

    base_mask = ((adj["player_display_name"] == player_name)
                 & (adj["position_group"] == "RB"))
    full_career = adj[base_mask]
    if full_career.empty:
        return
    pid_series = full_career["player_id"].dropna()
    if pid_series.empty:
        return
    pid = pid_series.iloc[0]

    pf_all = rp[rp["player_id"] == pid].copy()
    if pf_all.empty:
        return
    pf_all["season"] = pf_all["season"].astype(int)
    pf_all["week"] = pf_all["week"].astype(int)

    # Pull this player's most recent team for theming. The adjusted
    # parquet has weekly rows with `team` per row — take the latest
    # season's most-frequent team in case of mid-season trade.
    team_abbr = None
    if "team" in full_career.columns:
        latest_season = full_career["season"].max()
        latest_rows = full_career[full_career["season"] == latest_season]
        team_counts = latest_rows["team"].dropna().value_counts()
        if not team_counts.empty:
            team_abbr = str(team_counts.index[0])
    from lib_shared import team_theme
    theme = team_theme(team_abbr)

    # Render the header BEFORE the dropdown so the dropdown sits
    # under the section title.
    _run_scheme_header(theme=theme)

    seasons_present = sorted(pf_all["season"].unique().tolist(), reverse=True)
    default_label = ("All career" if is_career_view
                     else f"Season {int(season)}"
                          if season in seasons_present
                          else f"Season {seasons_present[0]}")
    options = [f"Season {s}" for s in seasons_present] + ["All career"]

    pf = _render_panel_view_picker(
        options, default_label, key=f"{key_prefix}_view_pick", pf_all=pf_all
    )
    if pf is None or len(pf) < 5:
        return

    _render_run_scheme_panel(pf, key_prefix, render_header=False,
                              theme=theme)


@st.cache_data
def _load_rb_peer_pools():
    """League-wide peer pools per gap — cached once per session.
    Used by the narrative engine to compute "Nth of 47 RBs at this
    gap" rank context. Returns a dict {gap_code: pd.DataFrame}.
    """
    rp = _load_rusher_plays()
    if rp is None:
        return {}
    rp = rp.copy()
    rp["gap_code"] = rp.apply(_classify_gap, axis=1)
    from lib_field_viz import _build_peer_gap_pools
    return _build_peer_gap_pools(rp, min_carries=50)


@st.cache_data
def _load_wr_route_peer_pools(min_targets: int = 30):
    """League-wide per-route peer pools — receivers with ≥ min_targets
    on each route, used to rank a receiver's route-level EPA."""
    tp = _load_targeted_plays()
    if tp is None:
        return {}
    from lib_field_viz import _build_route_peer_pools
    return _build_route_peer_pools(tp, min_targets=min_targets)


@st.cache_data
def _load_wr_volume_pool(min_targets: int = 50):
    """Per-receiver career target totals across the targeted_plays
    parquet, filtered to players with ≥ min_targets total. Used for
    the volume rank tag on the Targets stat tile."""
    tp = _load_targeted_plays()
    if tp is None:
        return None
    totals = (tp.groupby("player_id")
                 .size()
                 .reset_index(name="career_targets"))
    return totals[totals["career_targets"] >= min_targets]


def _render_wr_narrative_blurb(pf_unfiltered: pd.DataFrame,
                                  theme: dict | None) -> None:
    """Render the auto-generated 'signature route + weakness' blurb
    above the WR coverage panel's filter row. Theme primary color
    drives the accent border."""
    if pf_unfiltered is None or pf_unfiltered.empty:
        return
    from lib_field_viz import build_wr_narrative
    peer_pools = _load_wr_route_peer_pools()
    narrative = build_wr_narrative(pf_unfiltered, peer_pools=peer_pools)
    if not narrative:
        return
    accent = (theme or {}).get("primary", "#0076B6")
    st.markdown(
        f"<div style='background:#fff;border:1px solid #e6e9ee;"
        f"border-left:4px solid {accent};border-radius:6px;"
        f"padding:12px 16px;margin:8px 0 14px 0;'>"
        f"<div style='font-size:0.62rem;color:#5b6b7e;letter-spacing:1.4px;"
        f"text-transform:uppercase;font-weight:700;margin-bottom:5px;'>"
        f"📖 Story</div>"
        f"<div style='font-size:0.95rem;color:#2a3a4d;line-height:1.5;'>"
        f"{narrative}</div></div>",
        unsafe_allow_html=True,
    )


@st.cache_data
def _load_rb_volume_pool(min_carries: int = 100):
    """Per-player career carry totals across the rusher_plays parquet,
    filtered to RBs with at least `min_carries` total. Used to rank
    a player by career volume in the Carries stat tile."""
    rp = _load_rusher_plays()
    if rp is None:
        return None
    totals = (rp.groupby("player_id")
                .size()
                .reset_index(name="career_carries"))
    totals = totals[totals["career_carries"] >= min_carries]
    return totals


# Mapping: position_group → (league parquet filename, metadata
# filename, label for narrative). Used to drive the generic
# narrative engine for positions without a dedicated panel.
_POSITION_NARRATIVE_CONFIG = {
    "QB":  ("league_qb_all_seasons.parquet",     "qb_stat_metadata.json",     "starting QBs"),
    "DE":  ("league_de_all_seasons.parquet",     "de_stat_metadata.json",     "EDGE rushers"),
    "DT":  ("league_dt_all_seasons.parquet",     "dt_stat_metadata.json",     "interior DLs"),
    "LB":  ("league_lb_all_seasons.parquet",     "lb_stat_metadata.json",     "linebackers"),
    "CB":  ("league_cb_all_seasons.parquet",     "cb_stat_metadata.json",     "cornerbacks"),
    "S":   ("league_s_all_seasons.parquet",      "safety_stat_metadata.json", "safeties"),
    "OL":  ("league_ol_all_seasons.parquet",     "ol_stat_metadata.json",     "offensive linemen"),
    "K":   ("league_k_all_seasons.parquet",      "kicker_stat_metadata.json", "kickers"),
    "P":   ("league_p_all_seasons.parquet",      "punter_stat_metadata.json", "punters"),
}


@st.cache_data
def _load_position_pool(position_group: str):
    """Load the league parquet for a position group. Returns None
    if the position isn't in the narrative config (RB/WR/TE have
    their own panels with richer narratives)."""
    cfg = _POSITION_NARRATIVE_CONFIG.get(position_group)
    if cfg is None:
        return None
    parquet_name, _, _ = cfg
    from pathlib import Path
    p = Path(__file__).resolve().parent / "data" / parquet_name
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def _load_position_labels(position_group: str) -> dict:
    """Load the stat_labels dict from a position's metadata JSON."""
    cfg = _POSITION_NARRATIVE_CONFIG.get(position_group)
    if cfg is None:
        return {}
    _, meta_name, _ = cfg
    from pathlib import Path
    import json
    p = Path(__file__).resolve().parent / "data" / meta_name
    if not p.exists():
        return {}
    with open(p) as f:
        meta = json.load(f)
    return meta.get("stat_labels", {})


def _render_generic_position_narrative(player_name: str,
                                          position_group: str,
                                          theme: dict | None) -> None:
    """For positions without a dedicated panel (DE/DT/LB/CB/S/OL/QB/K/P),
    render a 'signature stat + weakness' blurb at the top of the
    splits section. Reads from the position's league parquet."""
    cfg = _POSITION_NARRATIVE_CONFIG.get(position_group)
    if cfg is None:
        return  # RB/WR/TE handled by their own panels
    _, _, position_label = cfg

    pool = _load_position_pool(position_group)
    if pool is None or pool.empty:
        return
    labels = _load_position_labels(position_group)

    # Find the player's most-recent row. League parquets vary in
    # name column — try common alternatives.
    name_col = None
    for cand in ("player_display_name", "full_name", "player", "player_name"):
        if cand in pool.columns:
            name_col = cand
            break
    if name_col is None:
        return
    rows = pool[pool[name_col] == player_name]
    if rows.empty:
        return
    if "season_year" in rows.columns:
        rows = rows.sort_values("season_year", ascending=False)
    player_row = rows.iloc[0]

    from lib_field_viz import build_position_narrative
    narrative = build_position_narrative(
        player_row, peer_pool=pool, stat_labels=labels,
        position_label=position_label,
    )
    if not narrative:
        return

    accent = (theme or {}).get("primary", "#0076B6")
    st.markdown(
        f"<div style='background:#fff;border:1px solid #e6e9ee;"
        f"border-left:4px solid {accent};border-radius:6px;"
        f"padding:12px 16px;margin:8px 0 14px 0;'>"
        f"<div style='font-size:0.62rem;color:#5b6b7e;letter-spacing:1.4px;"
        f"text-transform:uppercase;font-weight:700;margin-bottom:5px;'>"
        f"📖 Story</div>"
        f"<div style='font-size:0.95rem;color:#2a3a4d;line-height:1.5;'>"
        f"{narrative}</div></div>",
        unsafe_allow_html=True,
    )


def _render_rb_narrative_blurb(pf: pd.DataFrame, theme: dict | None) -> None:
    """Render the auto-generated 'signature + weakness' blurb above
    the filter row. Theme primary color drives the accent border so
    the blurb feels team-native."""
    if pf is None or pf.empty:
        return
    from lib_field_viz import build_rb_narrative
    # Reuse the cached peer pools instead of rebuilding them per render.
    peer_pools = _load_rb_peer_pools()
    narrative = build_rb_narrative(pf, peer_pools=peer_pools,
                                    min_player_carries_per_gap=8,
                                    min_peer_carries_per_gap=50)
    if not narrative:
        return
    accent = (theme or {}).get("primary", "#0076B6")
    st.markdown(
        f"<div style='background:#fff;border:1px solid #e6e9ee;"
        f"border-left:4px solid {accent};border-radius:6px;"
        f"padding:12px 16px;margin:8px 0 14px 0;'>"
        f"<div style='font-size:0.62rem;color:#5b6b7e;letter-spacing:1.4px;"
        f"text-transform:uppercase;font-weight:700;margin-bottom:5px;'>"
        f"📖 Story</div>"
        f"<div style='font-size:0.95rem;color:#2a3a4d;line-height:1.5;'>"
        f"{narrative}</div></div>",
        unsafe_allow_html=True,
    )


def _bucket_box(v):
    if v is None or pd.isna(v):
        return None
    if v <= 6:
        return "Light (≤6)"
    if v <= 7:
        return "Neutral (7)"
    return "Stacked (8+)"


# ─────────────────────────────────────────────────────────────
# Gap classification — translate nflverse run_location/run_gap
# into football's A/B/C/D-gap convention. The selector below
# uses these short codes so the user picks "B-L" instead of
# "Left guard." Mapping:
#     middle             → A   (PBP doesn't tag a side for A-gap)
#     left  + guard      → B-L
#     left  + tackle     → C-L
#     left  + end        → D-L
#     right + guard      → B-R
#     right + tackle     → C-R
#     right + end        → D-R
# ─────────────────────────────────────────────────────────────
def _classify_gap(row) -> str | None:
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
    if letter is None:
        return None
    return f"{letter}-{side}"


# Display label for the bar chart (longer/clearer than the pill code).
_GAP_DISPLAY = {
    "D-L": "Left end (D)",
    "C-L": "Left tackle (C)",
    "B-L": "Left guard (B)",
    "A":   "Middle (A)",
    "B-R": "Right guard (B)",
    "C-R": "Right tackle (C)",
    "D-R": "Right end (D)",
}
# Render order: outside-left → middle → outside-right
_GAP_ORDER = ["D-L", "C-L", "B-L", "A", "B-R", "C-R", "D-R"]


def _epa_bar_color(epa: float) -> str:
    """Green-to-red diverging palette tied to EPA per carry. Used
    for the gap chart bars so coloring tracks an objective benchmark
    rather than the player's own average."""
    if pd.isna(epa):
        return "#9aa6b3"
    if epa >= 0.10:
        return "#1a8c3d"   # strong green
    if epa >= 0.0:
        return "#7ab87a"   # light green
    if epa >= -0.10:
        return "#e08a8a"   # light red
    return "#b3261e"       # strong red


def _run_scheme_header(theme: dict | None = None):
    """Section header — pulled out so the public section can render
    the in-panel season/career dropdown between header and body.

    `theme` is the team theme from lib_shared.team_theme(). When
    provided, the accent border uses the team's primary color and a
    small logo sits inline with the title. Falls back to Lions blue.
    """
    accent = (theme or {}).get("primary", "#0076B6")
    logo = (theme or {}).get("logo", "")
    title_inline = ""
    if logo:
        title_inline = (
            f"<img src='{logo}' style='height:24px;width:auto;"
            f"vertical-align:middle;margin-right:8px;"
            f"object-fit:contain;'/>")
    st.markdown(
        f"<div style='margin:18px 0 6px 0;padding-left:12px;"
        f"border-left:5px solid {accent};'>"
        f"<div style='font-size:1.25rem;font-weight:800;color:#0a3d62;"
        f"letter-spacing:0.3px;display:flex;align-items:center;'>"
        f"{title_inline}<span>🏃 Run scheme profile</span></div>"
        f"<div style='font-size:0.78rem;color:#5b6b7e;margin-top:2px;'>"
        f"Where he runs, how he handles different boxes, and how the "
        f"formation affects production. Free PBP + NGS data only — "
        f"block grades and yards-after-contact need PFF."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _apply_run_scheme_filters(pf: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Top-row filter pills for the run-scheme panel — Formation /
    Personnel / Box / Down. Returns the filtered slice. All filters
    cascade through the whole panel.

    (Defense personnel filter dropped — ~30% of plays have null
    defense_personnel labels in nflverse, which would silently
    exclude data and mislead the user.)"""
    f1, f2, f4, f5 = st.columns(4)

    # Formation
    formations_present = ["All"] + sorted(
        f.title() for f in pf["offense_formation"].dropna().unique()
        if f and str(f).strip()
    )
    with f1:
        form_pick = st.selectbox(
            "Formation", formations_present,
            key=f"{key_prefix}_filt_form",
            help="Restrict to runs out of this offensive formation.",
        )

    # Personnel — derive bucket from offense_personnel
    def _pers_bucket(d):
        if d is None or pd.isna(d) or not d:
            return None
        rb = te = 0
        for chunk in str(d).split(","):
            chunk = chunk.strip()
            parts = chunk.split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                n = int(parts[0])
            except ValueError:
                continue
            pos = parts[1].strip()
            if pos == "RB":
                rb = n
            elif pos == "TE":
                te = n
        if 0 <= rb <= 9 and 0 <= te <= 9:
            return f"{rb}{te}"
        return None

    pf = pf.copy()
    pf["pers_bucket"] = pf["offense_personnel"].apply(_pers_bucket)
    pers_present = ["All"] + sorted(
        p for p in pf["pers_bucket"].dropna().unique() if p
    )
    with f2:
        pers_pick = st.selectbox(
            "Personnel", pers_present,
            key=f"{key_prefix}_filt_pers",
            help="11 = 1 RB · 1 TE · 3 WR. 12 = 1 RB · 2 TE · 2 WR. "
                 "21 = 2 RB · 1 TE · 2 WR. 13 = 1 RB · 3 TE · 1 WR.",
        )

    # Box count bucket
    pf["box_bucket"] = pf["defenders_in_box"].apply(_bucket_box)
    box_options = ["All", "Light (≤6)", "Neutral (7)", "Stacked (8+)"]
    with f4:
        box_pick = st.selectbox(
            "Box", box_options,
            key=f"{key_prefix}_filt_box",
            help="Defenders in the box — number the offense had to block.",
        )

    # Down
    down_options = ["All", "1st", "2nd", "3rd", "4th"]
    with f5:
        down_pick = st.selectbox(
            "Down", down_options,
            key=f"{key_prefix}_filt_down",
        )

    # Apply filters
    if form_pick != "All":
        pf = pf[pf["offense_formation"].str.title() == form_pick]
    if pers_pick != "All":
        pf = pf[pf["pers_bucket"] == pers_pick]
    if box_pick != "All":
        pf = pf[pf["box_bucket"] == box_pick]
    if down_pick != "All":
        d_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        pf = pf[pf["down"] == d_map[down_pick]]

    return pf


def _render_run_scheme_panel(pf: pd.DataFrame, key_prefix: str,
                              render_header: bool = True,
                              theme: dict | None = None) -> None:
    """The actual run scheme rendering. Layout:

        ┌ Auto-narrative blurb (career signature + weakness) ─┐
        │ Filter pills (Formation · Personnel · Box · Down)   │
        │ Stat tiles (filter-aware)                            │
        │ Gap pill (drill into one gap)                        │
        ├ Gap diagram ────── + detail table ──────────────────┤
        │ Performance by box count (filtered)                  │
        │ Performance by formation × personnel (filtered)      │
        └──────────────────────────────────────────────────────┘
    """
    import plotly.graph_objects as go

    if render_header:
        _run_scheme_header()

    pf = pf.copy()
    pf["gap_code"] = pf.apply(_classify_gap, axis=1)

    # Capture the unfiltered player carries for "career rank" lookup
    # before filters narrow pf below. Look up player_id and total
    # career carries against the league pool.
    career_carries = len(pf)
    career_rank_label = ""
    if "player_id" in pf.columns and not pf.empty:
        pid = pf["player_id"].dropna().iloc[0] if not pf["player_id"].dropna().empty else None
        if pid:
            volume_pool = _load_rb_volume_pool()
            if volume_pool is not None and not volume_pool.empty:
                from lib_shared import compute_rank_in_pool, format_rank
                rank, total = compute_rank_in_pool(
                    career_carries, volume_pool["career_carries"], ascending=False
                )
                career_rank_label = format_rank(rank, total)

    # ── Career-signature narrative blurb ──
    # Computed from the UNFILTERED player career so the headline
    # story stays stable when the user changes filters below.
    _render_rb_narrative_blurb(pf, theme)

    # ── TOP FILTER ROW: alignment + down + box ──
    # All filters narrow the entire panel — stat tiles, gap chart,
    # detail table, box count, formation table — so the user can ask
    # "how does this RB do on Shotgun 11 personnel runs against a
    # heavy box on first down" with four clicks.
    pf = _apply_run_scheme_filters(pf, key_prefix)

    # ── Gap selector — narrows the box-count + formation tables to
    # one specific gap so you can drill: "On B-Right, how do
    # different formations look?"
    gap_options = ["All", "D-L", "C-L", "B-L", "A", "B-R", "C-R", "D-R"]
    gap_choice = st.pills(
        "Drill by gap",
        gap_options,
        default="All",
        key=f"{key_prefix}_gap_pick",
        help="Filter the right-side box-count table and bottom formation "
             "table to runs through this gap. The gap-distribution chart "
             "always shows the full split so you keep your bearings.",
    )
    if gap_choice is None:
        gap_choice = "All"

    if gap_choice == "All":
        pf_filtered = pf
    else:
        pf_filtered = pf[pf["gap_code"] == gap_choice]

    # ── Top stat tiles (filter-aware) ──
    n_carries = len(pf_filtered)
    total_yards = float(pf_filtered["yards_gained"].fillna(0).sum())
    ypc = total_yards / n_carries if n_carries else 0
    avg_box = (float(pf_filtered["defenders_in_box"].dropna().mean())
                if pf_filtered["defenders_in_box"].notna().any()
                else float("nan"))
    stacked_n = int((pf_filtered["defenders_in_box"].fillna(0) >= 8).sum())
    stacked_pct = (stacked_n / n_carries * 100) if n_carries else 0
    light_pct = ((pf_filtered["defenders_in_box"].fillna(0) <= 6).sum()
                  / n_carries * 100) if n_carries else 0

    suffix = "" if gap_choice == "All" else f" · {gap_choice}"
    # Build the Carries tile sub-text. Includes career-volume rank
    # context (always vs the full unfiltered career) when available.
    carries_sub = f"{total_yards:.0f} total yds · {ypc:.2f} YPC"
    if career_rank_label and career_rank_label != "—":
        carries_sub += f" · career: {career_rank_label}"
    t1, t2, t3, t4 = st.columns(4)
    for col, label, big, sub, accent in [
        (t1, f"Carries{suffix}", f"{n_carries}",
         carries_sub,
         "#0076B6"),
        (t2, "Avg box", f"{avg_box:.2f}" if not pd.isna(avg_box) else "—",
         "defenders in the box per run",
         "#7f7f7f"),
        (t3, "Stacked rate", f"{stacked_pct:.0f}%",
         f"{stacked_n} of {n_carries} runs vs 8+ box",
         "#d62728"),
        (t4, "Light box rate", f"{light_pct:.0f}%",
         "vs 6-or-fewer in the box",
         "#2ca02c"),
    ]:
        col.markdown(
            f"<div style='background:#f3f6fa;border:1px solid #d6dde6;"
            f"border-left:4px solid {accent};border-radius:8px;"
            f"padding:10px 14px;'>"
            f"<div style='font-size:0.65rem;color:#5b6b7e;letter-spacing:1.2px;"
            f"text-transform:uppercase;font-weight:700;'>{label}</div>"
            f"<div style='font-size:1.7rem;font-weight:900;color:#0a3d62;"
            f"line-height:1.0;margin-top:4px;'>{big}</div>"
            f"<div style='font-size:0.75rem;color:#5b6b7e;margin-top:3px;'>"
            f"{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("")

    # ── FULL-WIDTH: line-of-scrimmage gap diagram ──
    st.markdown("**Line of scrimmage** _(vertical bars = gap zones · "
                "color = EPA per carry)_")
    gap_pool = pf[pf["gap_code"].notna()]
    if gap_pool.empty:
        st.caption("_No labeled run locations in this slice. "
                   "Loosen the filters above._")
    else:
        from lib_field_viz import build_gap_diagram
        fig = build_gap_diagram(gap_pool, metric="epa_per_carry")
        st.plotly_chart(fig, use_container_width=True,
                          key=f"{key_prefix}_gap_diagram")
        st.caption(
            "_Each gap colored by EPA per carry — "
            "<span style='color:#14b428'>vivid green</span> = elite, "
            "<span style='color:#c8141c'>vivid red</span> = struggles. "
            "Hover any zone for full breakdown. Where the back **ended up**, "
            "not where the play was designed — design intent requires PFF._",
            unsafe_allow_html=True,
        )

        # Build the agg dataframe for the detail table below.
        agg = (gap_pool.groupby("gap_code")
                        .agg(carries=("yards_gained", "size"),
                              yards=("yards_gained", "sum"),
                              epa=("epa", "mean"),
                              success=("fo_success", "mean"),
                              stuffs=("yards_gained",
                                       lambda s: int((s.fillna(0) <= 0).sum())),
                              chunks=("yards_gained",
                                       lambda s: int((s.fillna(0) >= 10).sum())),
                              tds=("touchdown",
                                    lambda s: int(s.fillna(0).sum())))
                        .reset_index())
        agg["ypc"] = agg["yards"] / agg["carries"]
        agg["display"] = agg["gap_code"].map(_GAP_DISPLAY).fillna(agg["gap_code"])
        agg["sort"] = agg["gap_code"].apply(
            lambda x: _GAP_ORDER.index(x) if x in _GAP_ORDER else 99)
        agg = agg.sort_values("sort").reset_index(drop=True)

        # Compact detail table — full width below the diagram.
        detail = pd.DataFrame({
            "Gap": agg["display"],
            "Car": agg["carries"].astype(int),
            "YPC": [f"{y:.2f}" for y in agg["ypc"]],
            "EPA/Car": [f"{e:+.2f}" for e in agg["epa"]],
            "Success%": [f"{s*100:.0f}%" for s in agg["success"]],
            "Stuff%": [f"{st_/c*100:.0f}%"
                       for st_, c in zip(agg["stuffs"], agg["carries"])],
            "Chunk%": [f"{ch/c*100:.0f}%"
                       for ch, c in zip(agg["chunks"], agg["carries"])],
            "TD": agg["tds"].astype(int),
        })
        st.dataframe(_style_epa_table(detail, "EPA/Car"),
                      use_container_width=True, hide_index=True)

    # ── Performance by box count (full-width, filtered to gap pill) ──
    title_suffix = "" if gap_choice == "All" else f"  ·  {gap_choice} only"
    st.markdown(f"**Performance by box count**{title_suffix}")
    pf2 = pf_filtered.copy()
    pf2["box_bucket"] = pf2["defenders_in_box"].apply(_bucket_box)
    box_pool = pf2[pf2["box_bucket"].notna()]
    if box_pool.empty:
        st.caption("_No labeled box counts in this slice._")
    else:
        min_n = 5 if gap_choice != "All" else 1
        rows = []
        for bucket in ["Light (≤6)", "Neutral (7)", "Stacked (8+)"]:
            grp = box_pool[box_pool["box_bucket"] == bucket]
            if len(grp) < min_n:
                continue
            car = len(grp)
            yds = float(grp["yards_gained"].fillna(0).sum())
            tds = int(grp["touchdown"].fillna(0).sum())
            success = float(grp["fo_success"].fillna(0).mean())
            epa = float(grp["epa"].fillna(0).mean())
            chunks = int((grp["yards_gained"].fillna(0) >= 10).sum())
            rows.append({
                "Box": bucket,
                "Car": car,
                "YPC": f"{yds/car:.2f}",
                "Success%": f"{success*100:.0f}%",
                "EPA/Car": f"{epa:+.2f}",
                "Chunks": chunks,
                "TD": tds,
            })
        if rows:
            box_table = pd.DataFrame(rows)
            st.dataframe(_style_epa_table(box_table, "EPA/Car"),
                          use_container_width=True, hide_index=True)
            if gap_choice != "All":
                st.caption(
                    f"_Buckets with fewer than {min_n} carries through "
                    f"the {gap_choice} gap are hidden — sample too thin._")
        else:
            st.caption(
                f"_All box-count buckets have fewer than {min_n} "
                f"carries through the {gap_choice} gap. "
                "Try 'All career' in the dropdown above for a bigger sample._")

    # ── Below: performance by formation × personnel (filtered) ──
    if pf_filtered["offense_formation"].notna().any():
        title_suffix = "" if gap_choice == "All" else f"  ·  {gap_choice} only"
        st.markdown(f"**Performance by formation × personnel**{title_suffix}")

        def _pers_bucket(d):
            """'1 RB, 1 TE, 3 WR' → '11', '1 RB, 2 TE, 2 WR' → '12'."""
            if d is None or pd.isna(d) or not d:
                return None
            rb = te = 0
            for chunk in str(d).split(","):
                chunk = chunk.strip()
                parts = chunk.split(" ", 1)
                if len(parts) != 2:
                    continue
                try:
                    n = int(parts[0])
                except ValueError:
                    continue
                pos = parts[1].strip()
                if pos == "RB":
                    rb = n
                elif pos == "TE":
                    te = n
            if 0 <= rb <= 9 and 0 <= te <= 9:
                return f"{rb}{te}"
            return None

        pf2 = pf_filtered.copy()
        pf2["pers_bucket"] = pf2["offense_personnel"].apply(_pers_bucket)
        form_pool = pf2[pf2["offense_formation"].notna()]
        # Tighter threshold when filtered — fewer carries per cell.
        min_combo = 5 if gap_choice != "All" else 3
        rows = []
        for (form, pers), grp in form_pool.groupby(
                ["offense_formation", "pers_bucket"], dropna=False):
            if len(grp) < min_combo:
                continue
            car = len(grp)
            yds = float(grp["yards_gained"].fillna(0).sum())
            tds = int(grp["touchdown"].fillna(0).sum())
            success = float(grp["fo_success"].fillna(0).mean())
            epa = float(grp["epa"].fillna(0).mean())
            chunks = int((grp["yards_gained"].fillna(0) >= 10).sum())
            stuffs = int((grp["yards_gained"].fillna(0) <= 0).sum())
            avg_box = (float(grp["defenders_in_box"].dropna().mean())
                        if grp["defenders_in_box"].notna().any()
                        else float("nan"))
            pers_disp = (f"{pers} pers"
                          if pers is not None and not pd.isna(pers)
                          else "—")
            rows.append({
                "Formation": form.title(),
                "Personnel": pers_disp,
                "Car": car,
                "YPC": f"{yds/car:.2f}",
                "Avg box": f"{avg_box:.2f}" if not pd.isna(avg_box) else "—",
                "Success%": f"{success*100:.0f}%",
                "EPA/Car": f"{epa:+.2f}",
                "Chunks": chunks,
                "Stuffed": stuffs,
                "TD": tds,
            })
        if rows:
            form_table = (pd.DataFrame(rows)
                            .sort_values("Car", ascending=False)
                            .reset_index(drop=True))
            st.dataframe(_style_epa_table(form_table, "EPA/Car"),
                          use_container_width=True, hide_index=True)
            cap_extra = (f" Combos with <{min_combo} carries through the "
                          f"{gap_choice} gap are hidden — sample too thin."
                          if gap_choice != "All"
                          else f" Combos with fewer than {min_combo} carries hidden.")
            st.caption(
                "_Personnel: '11' = 1 RB + 1 TE + 3 WR · '12' = 1 RB + 2 TE + 2 WR · "
                "'21' = 2 RB + 1 TE + 2 WR · '13' = 1 RB + 3 TE + 1 WR. "
                "Stuffed = run for ≤0 yards · Chunks = ≥10 yards." + cap_extra + "_"
            )
        elif gap_choice != "All":
            st.caption(
                f"_No formation × personnel combo has ≥{min_combo} carries "
                f"through the {gap_choice} gap. Try 'All career' above._")


# Canonical order of cohort labels for each "color games by" choice.
# The ordering matters — palette colors get assigned in this sequence,
# so for ordinal axes (defense tier, box weight, blitz rate) the user
# sees the gradient walk in the right direction.
_COHORT_ORDERS = {
    "Defense tier": ["🟥 Top 10%", "🟧 Top 25%", "🟨 Top half",
                      "🟩 Bottom half", "🟦 Bottom 25%"],
    "Roof":        ["Outdoor", "Indoor"],
    "Surface":     ["Grass", "Turf"],
    "Weather":     ["Cold (<40°)", "Windy (>15mph)", "Mild / clear"],
    "Location":    ["Home", "Away"],
    "Result":      ["Win", "Loss", "Tie"],
    "Box defenders": ["Light box", "Balanced box", "Heavy box"],
    "Blitz rate":  ["Low blitz (<20%)", "Avg blitz (20–35%)",
                     "High blitz (>35%)"],
    "Man / zone":  ["Zone-heavy (<40% man)", "Mixed coverage",
                     "Man-heavy (>55% man)"],
    "Top coverage shell": ["Cover-0", "Cover-1", "Cover-2", "Cover-3",
                            "Cover-4", "Cover-6", "Cover-9", "2-Man"],
    "Own personnel": ["11-heavy", "12-heavy", "21-heavy", "Mixed"],
    "Own pace":    ["Run-heavy (<50%)", "Balanced (50–60%)",
                     "Pass-heavy (>60%)"],
    "Own formation profile": ["Under-center heavy", "Shotgun-heavy"],
}


# Coloring strategy per "color by" choice.
#   "heatmap_forward":  first cohort = unfavorable for the player → red,
#                       last = favorable → green. Use when the canonical
#                       order goes hard→easy (e.g., Defense tier where
#                       "Top 10%" is the toughest defense).
#   "heatmap_reverse":  first cohort = favorable → green, last = unfavorable
#                       → red. Use when canonical order goes easy→hard
#                       (e.g., Box defenders: Light → Heavy).
#   anything else / missing: categorical — use team palette shades.
_COHORT_DIRECTIONS = {
    "Defense tier":  "heatmap_forward",
    "Box defenders": "heatmap_reverse",
    "Blitz rate":    "heatmap_reverse",
    "Man / zone":    "heatmap_reverse",
}


def _render_cohort_chart(games: pd.DataFrame, cfg: dict,
                          key_prefix: str,
                          theme: dict | None = None) -> None:
    """Game-by-game chart of the chosen stat. A 'Metric' picker switches
    which stat is plotted; a 'Color by' picker splits games into cohorts
    (legend doubles as a show/hide toggle). A dotted line shows the
    season's expected baseline for each game's opponent.

    `theme` is the team theme from lib_shared.team_theme(). When given,
    cohort colors are generated as shades of the team's primary +
    secondary so every chart on the player's page stays in their team's
    color family. Falls back to a generic palette if no theme.
    """
    import plotly.graph_objects as go
    from lib_shared import team_palette

    # Build the metric options from the position's summary_stats.
    # Each entry maps a friendly label → (actual_col, fmt).
    metric_options: dict[str, tuple[str, str]] = {}
    for col, lbl, fmt in cfg["summary_stats"]:
        if col in games.columns:
            metric_options[lbl] = (col, fmt)
    if not metric_options:
        return  # nothing to chart

    headline_label = cfg["headline_label"]
    default_idx = (list(metric_options.keys()).index(headline_label)
                   if headline_label in metric_options else 0)

    pc1, pc2 = st.columns(2)
    with pc1:
        metric_label = st.selectbox(
            "Metric",
            list(metric_options.keys()),
            index=default_idx,
            key=f"{key_prefix}_chart_metric",
            help="Switch the stat being charted.",
        )
    with pc2:
        # Build the option list dynamically — only show scheme cohort
        # options if those columns exist on the games frame.
        color_options = ["Defense tier", "Roof", "Surface", "Weather",
                          "Location", "Result"]
        if "box_bucket" in games.columns:
            color_options.append("Box defenders")
        if "blitz_bucket" in games.columns:
            color_options.append("Blitz rate")
        if "man_bucket" in games.columns:
            color_options.append("Man / zone")
        if "top_coverage" in games.columns:
            color_options.append("Top coverage shell")
        if "off_pers_bucket" in games.columns:
            color_options.append("Own personnel")
        if "off_pace_bucket" in games.columns:
            color_options.append("Own pace")
        if "off_shotgun_bucket" in games.columns:
            color_options.append("Own formation profile")
        color_by = st.selectbox(
            "Color games by",
            color_options,
            key=f"{key_prefix}_chart_colorby",
            help=("Each cohort gets its own color. **Click the legend** "
                  "to show or hide individual cohorts."),
        )

    actual_col, _fmt = metric_options[metric_label]
    expected_col = f"{actual_col}_expected"
    delta_col = f"{actual_col}_delta"
    label = metric_label
    color_col_map = {
        "Defense tier": "opp_tier_label",
        "Roof": "roof_bucket",
        "Surface": "surface_bucket",
        "Weather": "weather_bucket",
        "Location": "loc_bucket",
        "Result": "result_bucket",
        "Box defenders": "box_bucket",
        "Blitz rate": "blitz_bucket",
        "Man / zone": "man_bucket",
        "Top coverage shell": "top_coverage",
        "Own personnel": "off_pers_bucket",
        "Own pace": "off_pace_bucket",
        "Own formation profile": "off_shotgun_bucket",
    }
    color_col = color_col_map[color_by]

    # Build the cohort → color map. Two strategies:
    #   1. Ordinal (Defense tier, Box defenders, Blitz rate, Man/zone):
    #      use a smooth red→yellow→green heatmap so color carries
    #      data signal — red = unfavorable context for the player,
    #      green = favorable. Direction is configured per color_by in
    #      _COHORT_DIRECTIONS.
    #   2. Categorical (everything else — Surface, Roof, Weather,
    #      coverage shells, etc.): the categories don't have inherent
    #      better/worse, so use shades of the team's primary/secondary
    #      so identity stays consistent.
    canonical = _COHORT_ORDERS.get(color_by, [])
    cohorts_present = [c for c in games[color_col].dropna().unique()]
    ordered = [c for c in canonical if c in cohorts_present]
    extras = [c for c in cohorts_present if c not in canonical and c != "—"]
    ordered = ordered + extras

    direction = _COHORT_DIRECTIONS.get(color_by, "categorical")
    if direction in ("heatmap_forward", "heatmap_reverse") and len(ordered) > 1:
        from lib_shared import heatmap_color
        n = len(ordered)
        if direction == "heatmap_forward":
            # First cohort = unfavorable → red (t=0); last = favorable → green (t=1)
            colors_list = [heatmap_color(i / (n - 1), lo=0.0, hi=1.0)
                           for i in range(n)]
        else:  # heatmap_reverse
            # First cohort = favorable → green; last = unfavorable → red
            colors_list = [heatmap_color(i / (n - 1), lo=0.0, hi=1.0,
                                          reverse=True)
                           for i in range(n)]
    else:
        colors_list = team_palette(theme or {}, len(ordered))

    cohort_colors = dict(zip(ordered, colors_list))
    cohort_colors["—"] = "#cccccc"  # missing-data sentinel stays neutral

    # Career view = games span multiple seasons. Use date X-axis so
    # weeks from different years don't collide; single-season uses
    # week numbers (more intuitive).
    is_career = games["season"].nunique() > 1
    if is_career and "gameday" in games.columns:
        games = games.copy()
        games["_x"] = pd.to_datetime(games["gameday"], errors="coerce")
        x_title = "Date"
        x_axis_kwargs = dict(title=x_title, gridcolor="#eee")
        sort_key = "_x"
    else:
        games = games.copy()
        games["_x"] = games["week"]
        x_title = "Week"
        x_axis_kwargs = dict(title=x_title, tickmode="linear", dtick=1,
                             gridcolor="#eee")
        sort_key = "_x"

    sorted_games = games.sort_values(sort_key).reset_index(drop=True)

    fig = go.Figure()

    # Reference: per-game expected baseline (the schedule).
    if expected_col in sorted_games.columns and sorted_games[expected_col].notna().any():
        fig.add_trace(go.Scatter(
            x=sorted_games["_x"],
            y=sorted_games[expected_col],
            mode="lines",
            line=dict(color="#888", dash="dot", width=2),
            name="Expected (schedule)",
            hovertemplate=(f"Expected {label.lower()}: "
                           "%{y:.1f}<extra></extra>"),
        ))

    # Per-game scheme summary string for the chart hover. Two halves:
    #   1. opp scheme — what the defense did to him
    #   2. own scheme — what his own team called that game
    # Each falls back to "—" gracefully for older seasons / missing data.
    def _opp_scheme_line(row):
        bits = []
        b = row.get("avg_box_run")
        if b is not None and pd.notna(b):
            bits.append(f"box {b:.1f}")
        bz = row.get("pct_blitz")
        if bz is not None and pd.notna(bz):
            bits.append(f"blitz {bz*100:.0f}%")
        m = row.get("pct_man")
        if m is not None and pd.notna(m):
            bits.append(f"man {m*100:.0f}%")
        tc = row.get("top_coverage")
        if tc:
            bits.append(str(tc))
        return " · ".join(bits) if bits else "—"

    def _own_scheme_line(row):
        bits = []
        sg = row.get("shotgun_rate")
        if sg is not None and pd.notna(sg):
            bits.append(f"shotgun {sg*100:.0f}%")
        nh = row.get("no_huddle_rate")
        if nh is not None and pd.notna(nh) and nh > 0:
            bits.append(f"no-huddle {nh*100:.0f}%")
        for pers_col, pers_label in [
            ("pct_11_personnel", "11"),
            ("pct_12_personnel", "12"),
            ("pct_21_personnel", "21"),
        ]:
            v = row.get(pers_col)
            if v is not None and pd.notna(v) and v >= 0.30:
                bits.append(f"{pers_label} pers {v*100:.0f}%")
                break  # show the dominant grouping only
        pr = row.get("pass_rate")
        if pr is not None and pd.notna(pr):
            bits.append(f"pass {pr*100:.0f}%")
        return " · ".join(bits) if bits else "—"

    sorted_games = sorted_games.copy()
    sorted_games["_scheme_line"] = sorted_games.apply(_opp_scheme_line, axis=1)
    sorted_games["_own_scheme_line"] = sorted_games.apply(_own_scheme_line, axis=1)

    # One trace per cohort — Plotly's legend then doubles as the toggle.
    # Hover shows season + week + opponent + opp-scheme + own-scheme.
    for cohort in sorted_games[color_col].dropna().unique():
        sub = sorted_games[sorted_games[color_col] == cohort]
        if sub.empty:
            continue
        custom = sub[["season", "week", "opponent_team",
                       expected_col, delta_col,
                       "_scheme_line", "_own_scheme_line"]].values
        color = cohort_colors.get(cohort, None)
        fig.add_trace(go.Scatter(
            x=sub["_x"],
            y=sub[actual_col],
            mode="markers",
            marker=dict(size=12, color=color,
                         line=dict(color="white", width=1.5)),
            name=str(cohort),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]} Wk %{customdata[1]} vs "
                "%{customdata[2]}</b><br>"
                f"{label}: " "%{y:.1f}<br>"
                "Expected: %{customdata[3]:.1f}<br>"
                "Δ vs expected: %{customdata[4]:+.1f}<br>"
                "<i>Opp scheme:</i> %{customdata[5]}<br>"
                "<i>Own scheme:</i> %{customdata[6]}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=x_axis_kwargs,
        yaxis=dict(title=label, gridcolor="#eee", zeroline=False),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.6)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True,
                     key=f"{key_prefix}_cohort_chart")
    st.caption("_Click any cohort in the legend to hide/show it. "
               "Dotted gray line = what a typical opponent's defense "
               "allowed on average for each week._")


def _render_headline_tiles(games: pd.DataFrame, cfg: dict) -> None:
    """Three tiles: recent form, schedule strength faced, consistency."""
    actual_col = cfg["headline_actual"]
    delta_col = cfg["headline_delta"]
    label = cfg["headline_label"]

    # Recent form: last 5 chronological games
    last5 = games.sort_values(["season", "week"]).tail(5)
    if len(last5) > 0 and last5[delta_col].notna().any():
        rf_delta = last5[delta_col].mean()
        rf_actual = last5[actual_col].mean()
        rf_n = len(last5)
        rf_text = f"{'+' if rf_delta >= 0 else ''}{rf_delta:.1f}"
        rf_sub = f"{rf_actual:.1f} {label.lower()}/g · last {rf_n}"
    else:
        rf_text = "—"
        rf_sub = "no recent games"

    # Schedule strength faced — average opponent's expected output
    exp_col = cfg["headline_expected"]
    if games[exp_col].notna().any():
        avg_exp = games[exp_col].mean()
        ss_text = f"{avg_exp:.1f}"
        ss_sub = f"avg {label.lower()} a typical opp allowed"
    else:
        ss_text = "—"
        ss_sub = "no schedule data"

    # Consistency: stdev of headline stat + boom/bust counts (>1σ above/below)
    if games[actual_col].notna().sum() >= 3:
        mean_v = games[actual_col].mean()
        std_v = games[actual_col].std()
        boom = int((games[actual_col] >= mean_v + std_v).sum())
        bust = int((games[actual_col] <= mean_v - std_v).sum())
        n = int(games[actual_col].notna().sum())
        cons_text = f"σ {std_v:.1f}"
        cons_sub = f"{boom} boom / {bust} bust ({n} games)"
    else:
        cons_text = "—"
        cons_sub = "not enough games"

    c1, c2, c3 = st.columns(3)
    for col, title, value, sub in [
        (c1, "Recent form", rf_text, rf_sub),
        (c2, "Schedule strength", ss_text, ss_sub),
        (c3, "Consistency", cons_text, cons_sub),
    ]:
        col.markdown(
            f"<div style='background:#f3f6fa;border:1px solid #d6dde6;"
            f"border-radius:8px;padding:10px 12px;'>"
            f"<div style='font-size:0.65rem;color:#5b6b7e;letter-spacing:1px;"
            f"text-transform:uppercase;font-weight:700;'>{title}</div>"
            f"<div style='font-size:1.6rem;font-weight:900;color:#0a3d62;"
            f"line-height:1.1;margin-top:4px;'>{value}</div>"
            f"<div style='font-size:0.75rem;color:#5b6b7e;margin-top:2px;'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# Main entry — call this from the player detail page.
# ──────────────────────────────────────────────────────────────
def render_splits_section(*, player_name: str, season,
                           position_group: str, key_prefix: str,
                           is_career_view: bool = False) -> None:
    """Renders the expandable splits explorer. Silent no-op if the
    pre-computed parquets are missing or the player has no games."""
    if not _data_ready():
        return  # data layer not built — fail silent

    if position_group not in POSITION_CONFIG:
        return  # position not yet supported in splits

    cfg = POSITION_CONFIG[position_group]
    side = cfg.get("side", "offense")

    if side == "defense":
        adj = _load_defensive_player_adjusted()
    else:
        adj = _load_adjusted()
    sched = _load_schedules()

    if adj is None:
        return  # data layer for this side not built — fail silent

    base_mask = ((adj["player_display_name"] == player_name)
                 & (adj["position_group"] == position_group))
    if is_career_view:
        games = adj[base_mask].copy()
    else:
        games = adj[base_mask & (adj["season"] == season)].copy()
    if games.empty:
        return

    # Resolve theme once at the top — every chart in this section
    # inherits team primary/secondary so the player's page reads as a
    # unified visual identity. Use the most-recent-season team in case
    # of mid-career trade.
    team_abbr = None
    if "team" in games.columns:
        latest_season = games["season"].max()
        latest_rows = games[games["season"] == latest_season]
        team_counts = latest_rows["team"].dropna().value_counts()
        if not team_counts.empty:
            team_abbr = str(team_counts.index[0])
    from lib_shared import team_theme as _team_theme
    section_theme = _team_theme(team_abbr)

    # Auto-narrative blurb at the top — shows for non-RB/WR/TE
    # positions. RB and WR have richer narratives in their own panels;
    # this generic version covers DE/DT/LB/CB/S/OL/QB/K/P.
    _render_generic_position_narrative(player_name, position_group,
                                        theme=section_theme)

    # Join schedule data on (season, week, team). Each game has one team
    # entry per side, so we match the player's `team` column.
    sched_slim = sched[["season", "week", "home_team", "away_team",
                        "home_score", "away_score", "gameday",
                        "roof", "surface", "temp", "wind", "div_game"]].copy()
    sched_slim["season"] = sched_slim["season"].astype(int)
    sched_slim["week"] = sched_slim["week"].astype(int)
    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)

    # Build a long-form schedule keyed by (season, week, team)
    home_view = sched_slim.assign(team=sched_slim["home_team"],
                                  is_home=True,
                                  team_score=sched_slim["home_score"],
                                  opp_score=sched_slim["away_score"])
    away_view = sched_slim.assign(team=sched_slim["away_team"],
                                  is_home=False,
                                  team_score=sched_slim["away_score"],
                                  opp_score=sched_slim["home_score"])
    sched_long = pd.concat([home_view, away_view], ignore_index=True)[[
        "season", "week", "team", "is_home", "gameday",
        "roof", "surface",
        "temp", "wind", "div_game", "team_score", "opp_score"
    ]]

    games = games.merge(sched_long, on=["season", "week", "team"], how="left")

    # ── Explosive plays layer (chunk runs / catches) ──
    # Player-game counts get joined in directly; the matching defense
    # baseline (avg allowed for the player's position group, that
    # season) becomes our `_expected` column. `_delta` is computed
    # the same way the main adjusted parquet does it. Offensive only.
    expl_pg = _load_explosive_player_games() if side == "offense" else None
    expl_def = _load_explosive_def_baselines() if side == "offense" else None
    if expl_pg is not None:
        expl_pg = expl_pg.copy()
        expl_pg["season"] = expl_pg["season"].astype(int)
        expl_pg["week"] = expl_pg["week"].astype(int)
        games = games.merge(
            expl_pg[["player_id", "season", "week", "team",
                     "explosive_runs", "explosive_receptions"]],
            on=["player_id", "season", "week", "team"], how="left",
        )
        # Players who never carried/caught simply have 0 explosives.
        games["explosive_runs"] = games["explosive_runs"].fillna(0)
        games["explosive_receptions"] = games["explosive_receptions"].fillna(0)

    if expl_def is not None and "explosive_runs" in games.columns:
        # Build the per-(opponent_team, season, position_group) lookup.
        b = expl_def.rename(columns={
            "defense_team": "opponent_team",
            "position": "position_group",
            "avg_explosive_runs": "explosive_runs_expected",
            "avg_explosive_receptions": "explosive_receptions_expected",
        })[["opponent_team", "season", "position_group",
             "explosive_runs_expected", "explosive_receptions_expected"]]
        b["season"] = b["season"].astype(int)
        games = games.merge(
            b, on=["opponent_team", "season", "position_group"], how="left"
        )
        games["explosive_runs_delta"] = (
            games["explosive_runs"] - games["explosive_runs_expected"]
        )
        games["explosive_receptions_delta"] = (
            games["explosive_receptions"] - games["explosive_receptions_expected"]
        )

    # ── Advanced offensive metrics (chunk completions, deep attempts,
    #    YAC chunks, RZ targets, etc.) ── offensive only.
    adv_pg = _load_advanced_player_games() if side == "offense" else None
    adv_def = _load_advanced_def_baselines() if side == "offense" else None
    ADV_STAT_NAMES = [
        "chunk_completions", "deep_attempts", "td_long_passes",
        "scramble_first_downs",
        "yac_chunks", "deep_targets", "rz_targets", "first_down_recs",
    ]
    if adv_pg is not None:
        adv_pg = adv_pg.copy()
        adv_pg["season"] = adv_pg["season"].astype(int)
        adv_pg["week"] = adv_pg["week"].astype(int)
        join_cols = ["player_id", "season", "week", "team"] + [
            c for c in ADV_STAT_NAMES if c in adv_pg.columns
        ]
        games = games.merge(adv_pg[join_cols],
                             on=["player_id", "season", "week", "team"],
                             how="left")
        for c in ADV_STAT_NAMES:
            if c in games.columns:
                games[c] = games[c].fillna(0)

    if adv_def is not None:
        rename_map = {"defense_team": "opponent_team",
                      "position": "position_group"}
        for c in ADV_STAT_NAMES:
            rename_map[f"avg_{c}"] = f"{c}_expected"
        ad = adv_def.rename(columns=rename_map)
        keep = ["opponent_team", "season", "position_group"] + [
            f"{c}_expected" for c in ADV_STAT_NAMES
            if f"{c}_expected" in ad.columns
        ]
        ad = ad[[c for c in keep if c in ad.columns]]
        ad["season"] = ad["season"].astype(int)
        games = games.merge(
            ad, on=["opponent_team", "season", "position_group"], how="left"
        )
        for c in ADV_STAT_NAMES:
            if c in games.columns and f"{c}_expected" in games.columns:
                games[f"{c}_delta"] = games[c] - games[f"{c}_expected"]

    # ── OWN-OFFENSE scheme join (offensive players only) ──
    # Tells us what the player's TEAM was running each game — shotgun
    # rate, personnel mix, pace, etc. — so we can both surface it in
    # the chart hover AND let users filter by their own scheme.
    own_scheme = _load_offense_game_scheme() if side == "offense" else None
    if own_scheme is not None:
        cols = [c for c in [
            "offense_team", "season", "week",
            "shotgun_rate", "no_huddle_rate", "pass_rate",
            "deep_attempt_rate", "avg_air_yards", "avg_time_to_throw",
            "pct_11_personnel", "pct_12_personnel", "pct_21_personnel",
            "pct_10_personnel", "pct_13_personnel",
            "form_shotgun", "form_under_center", "form_singleback",
            "form_empty", "form_pistol", "form_i",
        ] if c in own_scheme.columns]
        os = own_scheme[cols].rename(columns={"offense_team": "team"})
        os["season"] = os["season"].astype(int)
        os["week"] = os["week"].astype(int)
        games = games.merge(os, on=["team", "season", "week"], how="left")

        # Bucket scheme stats so they can be cohort filters / colors.
        def _bucket_pers(row):
            v11 = row.get("pct_11_personnel") or 0
            v12 = row.get("pct_12_personnel") or 0
            v21 = row.get("pct_21_personnel") or 0
            if v11 >= 0.65: return "11-heavy"
            if v12 >= 0.30: return "12-heavy"
            if v21 >= 0.10: return "21-heavy"
            return "Mixed"

        def _bucket_pace(v):
            if v is None or pd.isna(v): return None
            if v < 0.50: return "Run-heavy (<50%)"
            if v < 0.60: return "Balanced (50–60%)"
            return "Pass-heavy (>60%)"

        def _bucket_shotgun(v):
            if v is None or pd.isna(v): return None
            if v < 0.50: return "Under-center heavy"
            if v < 0.75: return "Mixed"
            return "Shotgun-heavy"

        def _bucket_pa(v):
            # we lost play_action; deep_attempt_rate is a vague proxy
            # for "vertical scheme" but skip explicit PA bucket
            return None

        games["off_pers_bucket"] = games.apply(_bucket_pers, axis=1)
        if "pass_rate" in games.columns:
            games["off_pace_bucket"] = games["pass_rate"].apply(_bucket_pace)
        if "shotgun_rate" in games.columns:
            games["off_shotgun_bucket"] = games["shotgun_rate"].apply(_bucket_shotgun)

    # ── Route distribution per receiver (offensive only) ──
    routes = _load_route_distribution() if side == "offense" else None
    if routes is not None:
        rt_cols = [c for c in routes.columns if c.startswith("rt_")
                    or c == "total_routes"]
        join_cols = ["player_id", "season", "week", "team"] + rt_cols
        rdf = routes[join_cols].copy()
        rdf["season"] = rdf["season"].astype(int)
        rdf["week"] = rdf["week"].astype(int)
        games = games.merge(rdf, on=["player_id", "season", "week", "team"],
                              how="left")
        for c in rt_cols:
            if c in games.columns:
                games[c] = games[c].fillna(0)

    # Join opposing defense's per-game scheme profile so we know HOW
    # they played the player that day (box, blitz, man/zone, coverage,
    # personnel). Available 2016+ for box/blitz, 2018+ for coverage.
    # Offensive players only — for defenders the relevant scheme would
    # be THEIR OWN team's, which is a different conversation.
    scheme_game = _load_def_scheme_game() if side == "offense" else None
    if scheme_game is not None:
        scheme_slim_cols = [c for c in [
            "defense_team", "season", "week",
            "avg_box_run", "pct_stacked_box", "pct_light_box",
            "pct_blitz", "pressure_rate",
            "pct_man", "pct_zone",
            "cover_0", "cover_1", "cover_2", "cover_3",
            "cover_4", "cover_6", "cover_9", "two_man",
            "pct_base", "pct_nickel", "pct_dime",
        ] if c in scheme_game.columns]
        scheme_slim = scheme_game[scheme_slim_cols].rename(
            columns={"defense_team": "opponent_team"}
        )
        scheme_slim["season"] = scheme_slim["season"].astype(int)
        scheme_slim["week"] = scheme_slim["week"].astype(int)
        games = games.merge(
            scheme_slim, on=["opponent_team", "season", "week"], how="left"
        )

        # Pick the dominant coverage shell each game (highest rate).
        shell_cols = [c for c in ["cover_0", "cover_1", "cover_2", "cover_3",
                                    "cover_4", "cover_6", "cover_9", "two_man"]
                       if c in games.columns]
        if shell_cols:
            shell_label = {"cover_0": "Cover-0", "cover_1": "Cover-1",
                           "cover_2": "Cover-2", "cover_3": "Cover-3",
                           "cover_4": "Cover-4", "cover_6": "Cover-6",
                           "cover_9": "Cover-9", "two_man": "2-Man"}

            def _top_shell(row):
                best, best_v = None, -1.0
                for c in shell_cols:
                    v = row.get(c)
                    if v is not None and pd.notna(v) and v > best_v:
                        best_v = v
                        best = shell_label.get(c, c)
                return best

            games["top_coverage"] = games.apply(_top_shell, axis=1)

        # Bucket continuous scheme stats so they can be used as cohorts
        # in the chart's "Color by" picker. Cuts loosely tracked to
        # league-wide distribution for readability.
        def _bucket_box(v):
            if pd.isna(v):
                return None
            if v <= 6.5:
                return "Light box"
            if v <= 7.0:
                return "Balanced box"
            return "Heavy box"

        def _bucket_blitz(v):
            if pd.isna(v):
                return None
            if v < 0.20:
                return "Low blitz (<20%)"
            if v < 0.35:
                return "Avg blitz (20–35%)"
            return "High blitz (>35%)"

        def _bucket_man(v):
            if pd.isna(v):
                return None
            if v < 0.40:
                return "Zone-heavy (<40% man)"
            if v < 0.55:
                return "Mixed coverage"
            return "Man-heavy (>55% man)"

        if "avg_box_run" in games.columns:
            games["box_bucket"] = games["avg_box_run"].apply(_bucket_box)
        if "pct_blitz" in games.columns:
            games["blitz_bucket"] = games["pct_blitz"].apply(_bucket_blitz)
        if "pct_man" in games.columns:
            games["man_bucket"] = games["pct_man"].apply(_bucket_man)

    # Derived classifier columns
    games["roof_bucket"] = games["roof"].apply(_classify_roof)
    games["surface_bucket"] = games["surface"].apply(_classify_surface)
    games["weather_bucket"] = games.apply(
        lambda r: _classify_weather(r["temp"], r["wind"]), axis=1)
    games["result_bucket"] = games.apply(
        lambda r: ("Win" if pd.notna(r["team_score"]) and pd.notna(r["opp_score"])
                            and r["team_score"] > r["opp_score"]
                   else "Loss" if pd.notna(r["team_score"]) and pd.notna(r["opp_score"])
                            and r["team_score"] < r["opp_score"]
                   else "Tie"),
        axis=1,
    )
    games["loc_bucket"] = games["is_home"].map(
        {True: "Home", False: "Away"}).fillna("—")
    games["div_bucket"] = games["div_game"].map(
        {True: "Division", False: "Non-division"}).fillna("—")

    # Defense tier per opponent — one lookup per (season, position),
    # then look up each game's tier using THAT game's season.
    seasons_present = sorted(set(int(s) for s in games["season"].unique()))
    tier_lookups = {
        s: _build_tier_lookup(s, position_group, cfg["tier_metric"],
                                side=side)
        for s in seasons_present
    }
    games["opp_tier_pct"] = games.apply(
        lambda r: tier_lookups.get(int(r["season"]), {}).get(
            r["opponent_team"]),
        axis=1,
    )
    games["opp_tier_label"] = games["opp_tier_pct"].apply(_tier_label)

    # ── EXPANDER UI ──────────────────────────────────────────────
    with st.expander("📊 Game-by-game splits", expanded=False):
        st.caption(
            "Slice this player's games by opponent strength, surface, "
            "roof, weather, location, or outcome. Tiles up top track "
            "form, schedule strength, and consistency across **all** "
            "games this season; the table below reflects your filters."
        )

        _render_headline_tiles(games, cfg)
        st.markdown("")

        # ─── FILTER UI: chips + popover ─────────────────────────
        # All filter dropdowns live inside one popover. Above the
        # popover we render a chip row showing only the *active*
        # filters; clicking a chip's ✕ clears that one filter.
        # Default state = no clutter.

        # Filter spec: (suffix, label, options_or_None, column_required_or_None)
        # If options is None, we derive from data at render time.
        _ALL = "All"
        _filter_specs = [
            ("tier",    "Opp defense",
             [_ALL, "Top 10% (toughest)", "Top 25%", "Top half",
              "Bottom half", "Bottom 25% (easiest)"], None),
            ("roof",    "Roof",
             [_ALL, "Outdoor", "Indoor"], None),
            ("surface", "Surface",
             [_ALL, "Grass", "Turf"], None),
            ("weather", "Weather",
             [_ALL, "Cold (<40°)", "Windy (>15mph)", "Mild / clear"], None),
            ("loc",     "Location",
             [_ALL, "Home", "Away"], None),
            ("result",  "Result",
             [_ALL, "Win", "Loss"], None),
            ("box",     "Box defenders",
             [_ALL, "Light box", "Balanced box", "Heavy box"], "box_bucket"),
            ("blitz",   "Blitz rate",
             [_ALL, "Low blitz (<20%)", "Avg blitz (20–35%)",
              "High blitz (>35%)"], "blitz_bucket"),
            ("man",     "Man / zone",
             [_ALL, "Zone-heavy (<40% man)", "Mixed coverage",
              "Man-heavy (>55% man)"], "man_bucket"),
            ("cov",     "Coverage shell", None, "top_coverage"),
            ("pers",    "Personnel",      None, "off_pers_bucket"),
            ("pace",    "Game pace",
             [_ALL, "Run-heavy (<50%)", "Balanced (50–60%)",
              "Pass-heavy (>60%)"], "off_pace_bucket"),
            ("sg",      "Formation",
             [_ALL, "Under-center heavy", "Mixed", "Shotgun-heavy"],
             "off_shotgun_bucket"),
        ]

        def _opts_for(suffix: str, fixed_opts, col_req: str | None):
            if col_req and col_req not in games.columns:
                return None
            if fixed_opts is not None:
                return fixed_opts
            # Data-derived options
            if suffix == "cov":
                present = set(games["top_coverage"].dropna().unique())
                return [_ALL] + [c for c in
                                  ["Cover-0", "Cover-1", "Cover-2", "Cover-3",
                                   "Cover-4", "Cover-6", "Cover-9", "2-Man"]
                                  if c in present]
            if suffix == "pers":
                return [_ALL] + sorted(
                    set(games["off_pers_bucket"].dropna().unique())
                )
            return [_ALL]

        # Active chips: read session_state for any non-default filter
        active_filters = []
        for suffix, label, fixed_opts, col_req in _filter_specs:
            opts = _opts_for(suffix, fixed_opts, col_req)
            if opts is None:
                continue
            state_key = f"{key_prefix}_split_{suffix}"
            val = st.session_state.get(state_key, _ALL)
            if val != _ALL:
                active_filters.append((state_key, label, val))

        # Render chip row + popover trigger
        cap_col, btn_col = st.columns([5, 1])
        with cap_col:
            if active_filters:
                chip_cols = st.columns(min(len(active_filters), 6))
                for i, (state_key, label, val) in enumerate(active_filters):
                    with chip_cols[i % len(chip_cols)]:
                        if st.button(f"✕ {label}: {val}",
                                       key=f"{state_key}_chip",
                                       help="Click to clear this filter"):
                            st.session_state[state_key] = _ALL
                            st.rerun()
            else:
                st.caption("Showing **all games** — open the filter "
                            "panel to slice by opponent, scheme, weather, etc.")
        with btn_col:
            with st.popover(f"🔧 Filters{f' ({len(active_filters)})' if active_filters else ''}",
                              use_container_width=True):
                # ── Game-context filters ──
                st.markdown("**Game context**")
                f1, f2, f3 = st.columns(3)
                f4, f5, f6 = st.columns(3)
                ctx_pairs = [(f1, "tier"), (f2, "roof"), (f3, "surface"),
                              (f4, "weather"), (f5, "loc"), (f6, "result")]
                for col, suffix in ctx_pairs:
                    spec = next(s for s in _filter_specs if s[0] == suffix)
                    opts = _opts_for(suffix, spec[2], spec[3])
                    if opts is None:
                        continue
                    with col:
                        st.selectbox(spec[1], opts,
                                       key=f"{key_prefix}_split_{suffix}")

                # ── Opp scheme filters (only if data present) ──
                scheme_specs = [s for s in _filter_specs
                                  if s[0] in ("box", "blitz", "man", "cov")
                                  and _opts_for(s[0], s[2], s[3]) is not None]
                if scheme_specs:
                    st.markdown("**Opponent defensive scheme**")
                    s_cols = st.columns(min(len(scheme_specs), 4))
                    for i, spec in enumerate(scheme_specs):
                        opts = _opts_for(spec[0], spec[2], spec[3])
                        with s_cols[i % len(s_cols)]:
                            st.selectbox(spec[1], opts,
                                           key=f"{key_prefix}_split_{spec[0]}")

                # ── Own-offense scheme filters ──
                own_specs = [s for s in _filter_specs
                              if s[0] in ("pers", "pace", "sg")
                              and _opts_for(s[0], s[2], s[3]) is not None]
                if own_specs:
                    st.markdown("**Own-offense scheme**")
                    o_cols = st.columns(min(len(own_specs), 3))
                    for i, spec in enumerate(own_specs):
                        opts = _opts_for(spec[0], spec[2], spec[3])
                        with o_cols[i % len(o_cols)]:
                            st.selectbox(spec[1], opts,
                                           key=f"{key_prefix}_split_{spec[0]}")

                # ── Reset all ──
                if active_filters:
                    if st.button("Reset all filters",
                                   key=f"{key_prefix}_split_reset",
                                   use_container_width=True):
                        for state_key, _, _ in active_filters:
                            st.session_state[state_key] = _ALL
                        st.rerun()

        # Read final values from session_state (set by the popover widgets)
        def _val(suffix):
            return st.session_state.get(f"{key_prefix}_split_{suffix}", _ALL)
        tier_pick  = _val("tier")
        roof_pick  = _val("roof")
        surf_pick  = _val("surface")
        wx_pick    = _val("weather")
        loc_pick   = _val("loc")
        res_pick   = _val("result")
        box_pick   = _val("box")
        blitz_pick = _val("blitz")
        man_pick   = _val("man")
        cov_pick   = _val("cov")
        pers_pick  = _val("pers")
        pace_pick  = _val("pace")
        sg_pick    = _val("sg")

        # ── Apply filters ──
        filt = games.copy()
        if tier_pick == "Top 10% (toughest)":
            filt = filt[filt["opp_tier_pct"] >= 90]
        elif tier_pick == "Top 25%":
            filt = filt[filt["opp_tier_pct"] >= 75]
        elif tier_pick == "Top half":
            filt = filt[filt["opp_tier_pct"] >= 50]
        elif tier_pick == "Bottom half":
            filt = filt[filt["opp_tier_pct"] < 50]
        elif tier_pick == "Bottom 25% (easiest)":
            filt = filt[filt["opp_tier_pct"] < 25]
        if roof_pick != "All":
            filt = filt[filt["roof_bucket"] == roof_pick]
        if surf_pick != "All":
            filt = filt[filt["surface_bucket"] == surf_pick]
        if wx_pick != "All":
            filt = filt[filt["weather_bucket"] == wx_pick]
        if loc_pick != "All":
            filt = filt[filt["loc_bucket"] == loc_pick]
        if res_pick != "All":
            filt = filt[filt["result_bucket"] == res_pick]
        if box_pick != "All" and "box_bucket" in filt.columns:
            filt = filt[filt["box_bucket"] == box_pick]
        if blitz_pick != "All" and "blitz_bucket" in filt.columns:
            filt = filt[filt["blitz_bucket"] == blitz_pick]
        if man_pick != "All" and "man_bucket" in filt.columns:
            filt = filt[filt["man_bucket"] == man_pick]
        if cov_pick != "All" and "top_coverage" in filt.columns:
            filt = filt[filt["top_coverage"] == cov_pick]
        if pers_pick != "All" and "off_pers_bucket" in filt.columns:
            filt = filt[filt["off_pers_bucket"] == pers_pick]
        if pace_pick != "All" and "off_pace_bucket" in filt.columns:
            filt = filt[filt["off_pace_bucket"] == pace_pick]
        if sg_pick != "All" and "off_shotgun_bucket" in filt.columns:
            filt = filt[filt["off_shotgun_bucket"] == sg_pick]

        n_filt = len(filt)
        n_total = len(games)
        if n_filt == 0:
            st.info("No games match the current filters.")
            return

        # ── Filtered summary ──
        summary_parts = []
        for col, lbl, fmt in cfg["summary_stats"]:
            if col in filt.columns and filt[col].notna().any():
                summary_parts.append(f"**{lbl}**: {fmt.format(filt[col].mean())}")
        delta_avg = filt[cfg["headline_delta"]].mean() if cfg["headline_delta"] in filt.columns else None
        delta_text = ""
        if delta_avg is not None and pd.notna(delta_avg):
            sign = "+" if delta_avg >= 0 else ""
            delta_text = f" · vs expected: **{sign}{delta_avg:.1f}** {cfg['headline_unit']}/g"
        st.markdown(
            f"**{n_filt} of {n_total} games match** — per game: "
            + " · ".join(summary_parts) + delta_text
        )

        # ── Route distribution summary (receivers only) ──
        # Shows the top 5 routes the player was targeted on across the
        # filtered games. Quick scan of how the offense uses him.
        rt_cols = [c for c in filt.columns
                    if c.startswith("rt_") and c != "rt_other"]
        if rt_cols:
            totals = {c: float(filt[c].sum()) for c in rt_cols}
            totals = {k: v for k, v in totals.items() if v > 0}
            if totals:
                top_routes = sorted(totals.items(), key=lambda kv: -kv[1])[:5]
                grand = sum(totals.values())
                pretty = []
                for col, n in top_routes:
                    label = (col.replace("rt_", "")
                                .replace("_", " ").upper())
                    pct = (n / grand * 100) if grand else 0
                    pretty.append(f"**{label}** {int(n)} ({pct:.0f}%)")
                st.markdown(
                    "🎯 _Routes targeted in this filter:_ " + " · ".join(pretty)
                )

        st.markdown("")

        # ── Cohort chart now uses the filtered set ──
        st.markdown("**Game-by-game performance**")
        _render_cohort_chart(filt, cfg, key_prefix, theme=section_theme)
        st.markdown("")

        # ── Game-by-game table ──
        # Show "Season" column in career view (multiple seasons in pool)
        is_career_table = filt["season"].nunique() > 1
        cols_show = (["season"] if is_career_table else []) + [
            "week", "opponent_team", "opp_tier_label",
            "loc_bucket", "roof_bucket", "surface_bucket",
            "weather_bucket", "result_bucket"]
        cols_show += [c for c, _, _ in cfg["summary_stats"] if c in filt.columns]
        cols_show += [cfg["headline_delta"]]
        # Opponent's scheme on this game — only the most decision-relevant
        # fields, formatted for readability.
        scheme_show_cols = [c for c in
                             ["avg_box_run", "pct_blitz", "pct_man", "top_coverage"]
                             if c in filt.columns]
        cols_show += scheme_show_cols

        rename = {"season": "Yr", "week": "Wk", "opponent_team": "Opp",
                  "opp_tier_label": "D tier", "loc_bucket": "Loc",
                  "roof_bucket": "Roof", "surface_bucket": "Surface",
                  "weather_bucket": "Weather", "result_bucket": "Result",
                  cfg["headline_delta"]: f"{cfg['headline_label']} Δ",
                  "avg_box_run": "Box",
                  "pct_blitz": "Blitz%",
                  "pct_man": "Man%",
                  "top_coverage": "Top cov"}
        for c, lbl, _ in cfg["summary_stats"]:
            rename[c] = lbl

        sort_cols = (["season", "week"] if is_career_table else ["week"])
        table = (filt[cols_show]
                 .rename(columns=rename)
                 .sort_values([rename.get(s, s) for s in sort_cols])
                 .reset_index(drop=True))

        # Format the rate columns as percentages for readability
        for pct_col in ("Blitz%", "Man%"):
            if pct_col in table.columns:
                table[pct_col] = table[pct_col].apply(
                    lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—"
                )
        if "Box" in table.columns:
            table["Box"] = table["Box"].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—"
            )

        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "_**Box** = avg defenders in the box on run plays · "
            "**Blitz%** = % of pass plays with 5+ pass rushers · "
            "**Man%** = man-coverage rate (NGS) · **Top cov** = the "
            "coverage shell the opp ran most often that game. "
            "Coverage data is 2018+ only._"
        )
