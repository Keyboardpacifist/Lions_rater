"""Reusable Streamlit panels for rendering GAS Scores on position
pages. Drop-in module — every position page imports the helpers it
needs.

Public API
----------
    load_gas_data(position)
        Cached loader for the position's GAS parquet.

    render_gas_score_card(player_row, position)
        Top-of-page hero card: composite score + label + confidence
        + bundle breakdown.

    render_show_math_panel(player_row, position)
        Expandable panel with raw stats, z-scores, bundle weights,
        and SOS / adjustment annotations.

    render_hot_take(player_row, league_df, position)
        Data-driven contrarian take. Picks the most surprising story
        the bundle grades + z-scores tell.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"


# ── Position metadata ─────────────────────────────────────────────

POSITION_PARQUETS = {
    "qb": DATA_DIR / "qb_gas_seasons.parquet",
    "rb": DATA_DIR / "rb_gas_seasons.parquet",
    "wr": DATA_DIR / "wr_gas_seasons.parquet",
    "te": DATA_DIR / "te_gas_seasons.parquet",
    "ol": DATA_DIR / "ol_gas_seasons.parquet",
    "cb": DATA_DIR / "cb_gas_seasons.parquet",
    "safety": DATA_DIR / "safety_gas_seasons.parquet",
    "lb": DATA_DIR / "lb_gas_seasons.parquet",
    "de": DATA_DIR / "de_gas_seasons.parquet",
    "dt": DATA_DIR / "dt_gas_seasons.parquet",
    "k": DATA_DIR / "k_gas_seasons.parquet",
    "p": DATA_DIR / "p_gas_seasons.parquet",
}

BUNDLE_LABELS = {
    "qb": {
        "efficiency":    "Passing efficiency",
        "volume":        "Volume",
        "ball_security": "Ball security",
        "pressure":      "Under pressure",
        "mobility":      "Mobility",
        "clutch":        "Clutch / situation",
    },
    "rb": {
        "rushing_efficiency": "Rushing efficiency",
        "receiving":          "Receiving",
        "volume_durability":  "Volume / durability",
        "explosiveness":      "Explosiveness",
        "red_zone":           "Red zone",
        "short_yardage":      "Short yardage",
    },
    "wr": {
        "per_target_efficiency": "Per-target efficiency",
        "volume_role":           "Volume / role",
        "coverage_beating":      "Coverage-beating",
        "yac":                   "YAC",
        "scoring_chains":        "Scoring + chains",
        "catch_quality":         "Catch quality",
    },
    "te": {
        "per_target_efficiency": "Per-target efficiency",
        "volume_role":           "Volume / role",
        "yac":                   "YAC",
        "coverage_beating":      "Coverage-beating",
        "scoring_chains":        "Scoring + chains",
    },
    "ol": {
        "run_blocking":  "Run blocking",
        "pass_blocking": "Pass blocking",
        "discipline":    "Discipline",
        "durability":    "Durability / role",
    },
    "cb": {
        "coverage":        "Coverage",
        "ball_production": "Ball production",
        "tackling":        "Tackling",
    },
    "safety": {
        "coverage":        "Coverage",
        "run_support":     "Run support",
        "ball_production": "Ball production",
        "playmaking":      "Playmaking / blitz",
    },
    "lb": {
        "run_defense":     "Run defense",
        "pass_rush":       "Pass rush",
        "coverage":        "Coverage",
        "ball_production": "Ball production",
        "volume_role":     "Volume / role",
    },
    "de": {
        "pass_rush":       "Pass rush",
        "run_defense":     "Run defense",
        "disruption":      "Disruption",
        "ball_production": "Ball production",
    },
    "dt": {
        "pass_rush":       "Pass rush",
        "run_defense":     "Run defense",
        "disruption":      "Disruption",
        "ball_production": "Ball production",
    },
    "k": {
        "accuracy": "Accuracy",
        "value":    "Value (EPA)",
        "xp":       "Extra points",
    },
    "p": {
        "distance": "Distance",
        "pinning":  "Pinning",
        "coverage": "Coverage",
        "value":    "Value (EPA)",
    },
}

# Per-position adjustment annotations for the "Show Math" panel.
ADJUSTMENT_NOTES = {
    "qb": [
        "**Defense LOO** — opposing defense's pressure / sack rate "
        "subtracted (excluding this QB's own contribution).",
        "**Weapons-availability** — credit when the team's top "
        "receivers were missing in a given game (75% strength).",
        "**Mobility 21%** — modern dual-threat valuation; pure "
        "pocket QBs penalized accordingly.",
    ],
    "rb": [
        "**Defense LOO** — opposing run defense subtracted.",
        "**OL-context adjustment** — credit removed for running "
        "behind elite OL grades (top-5 starter avg).",
    ],
    "wr": [
        "**Defense LOO (game-level)** — opposing pass coverage "
        "subtracted per game.",
        "**QB LOO (game-level, 75%)** — QB's contribution to "
        "completions / EPA / yards backed out per game.",
    ],
    "te": [
        "**Defense LOO (game-level)** — opposing pass coverage "
        "subtracted.",
        "**QB LOO (75%)** — QB contribution backed out.",
        "**v1 caveat** — receiving only; blocking grade not in v1.",
    ],
    "ol": [
        "**Schedule-adjusted pressure / sack rate** — opposing "
        "rush quality subtracted.",
        "**QB sack-avoidance** — credit removed for mobile QBs "
        "who escape sacks (Allen, Lamar).",
        "**RB-talent** — credit removed for elite RB ryoe/att "
        "(Saquon, Henry).",
        "**v1 caveat** — pass-block signal is team-level (PFF "
        "would be needed to grade individual blockers).",
    ],
    "cb": [
        "**v1 caveat** — no WR-matchup adjustment (a CB shadowing "
        "Chase isn't graded easier than one shadowing a #4 WR).",
    ],
    "safety": [
        "**v1 caveat** — same as CB on matchup; SS vs FS roles "
        "graded by same spec.",
    ],
    "lb": [
        "**v1 caveat** — off-ball ILB and pass-rushing 3-4 OLB "
        "graded by same bundle spec; future v1.1 will split.",
    ],
    "de": [
        "**v1 caveat** — no PFF pass-rush win rate; based on "
        "play-result stats.",
    ],
    "dt": [
        "**v1 caveat** — same as DE; no PFF charting.",
    ],
    "k": [
        "**v1 caveat** — kicker stats are inherently volatile "
        "year-to-year; YoY is low.",
    ],
    "p": [],
}


@st.cache_data(show_spinner=False)
def load_gas_data(position: str) -> Optional[pd.DataFrame]:
    """Load a position's GAS parquet. Returns None if missing."""
    path = POSITION_PARQUETS.get(position)
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path)


def lookup_player_gas(position: str, player_id: Optional[str],
                         season_year: Optional[int]
                         ) -> Optional[pd.Series]:
    """Find a (player_id, season_year) row in this position's GAS
    table. Returns None if missing."""
    if not player_id or season_year is None:
        return None
    df = load_gas_data(position)
    if df is None:
        return None
    matches = df[(df["player_id"] == player_id)
                 & (df["season_year"] == int(season_year))]
    if not len(matches):
        return None
    return matches.iloc[0]


def gas_league_percentile(position: str, season_year: Optional[int],
                            score: Optional[float]) -> Optional[float]:
    """Pct of league players (this position, this season) with a
    GAS score lower than `score`. None if data missing."""
    if score is None or season_year is None:
        return None
    df = load_gas_data(position)
    if df is None:
        return None
    season_pop = df[df["season_year"] == int(season_year)]
    if not len(season_pop):
        return None
    return float((season_pop["gas_score"] < score).mean() * 100)


def _label_color(label: str) -> str:
    if "Elite" in label:
        return "#16a34a"
    if "High-end" in label:
        return "#22c55e"
    if "Above" in label:
        return "#84cc16"
    if "Solid" in label:
        return "#eab308"
    if "Average" in label or "replaceable" in label:
        return "#f59e0b"
    if "Below" in label:
        return "#ef4444"
    return "#dc2626"


def render_gas_score_card(player_row: pd.Series, position: str) -> None:
    """Big hero card at the top of a player detail view."""
    score = float(player_row.get("gas_score", 50.0))
    label = str(player_row.get("gas_label", ""))
    confidence = str(player_row.get("gas_confidence", ""))
    color = _label_color(label)

    name = player_row.get("player_display_name") or player_row.get(
        "full_name", "Unknown")
    season = player_row.get("season_year", "")
    team = player_row.get("recent_team") or player_row.get("team", "")

    st.markdown(
        f"""
<div style="border:2px solid {color};border-radius:12px;
            padding:18px 22px;margin:8px 0 16px 0;
            background-color:rgba(255,255,255,0.03);">
  <div style="display:flex;align-items:center;gap:24px;
              justify-content:space-between;">
    <div>
      <div style="font-size:13px;color:#9ca3af;
                  letter-spacing:1px;text-transform:uppercase;">
        GAS Score &middot; {team} {season}
      </div>
      <div style="font-size:28px;font-weight:700;color:#f3f4f6;">
        {name}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:48px;font-weight:800;color:{color};
                  line-height:1;">{score:.1f}</div>
      <div style="font-size:13px;color:#d1d5db;
                  font-weight:600;">{label}</div>
      <div style="font-size:11px;color:#9ca3af;margin-top:4px;">
        Confidence: {confidence}
      </div>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # Bundle breakdown bars
    bundle_labels = BUNDLE_LABELS.get(position, {})
    bundle_rows = []
    for bundle_key, label_text in bundle_labels.items():
        col = f"gas_{bundle_key}_grade"
        if col in player_row.index:
            grade = float(player_row[col])
            bundle_rows.append((label_text, grade))
    if bundle_rows:
        bdf = pd.DataFrame(bundle_rows, columns=["Bundle", "Grade"])
        st.bar_chart(bdf.set_index("Bundle"), height=240)


def render_show_math_panel(player_row: pd.Series, position: str) -> None:
    """Expandable panel — bundle grades, raw stats, adjustment notes."""
    with st.expander("📐 Show the math", expanded=False):
        st.markdown("**Bundle composition**")
        bundle_labels = BUNDLE_LABELS.get(position, {})
        rows = []
        for bundle_key, label_text in bundle_labels.items():
            col = f"gas_{bundle_key}_grade"
            if col in player_row.index:
                rows.append({
                    "Bundle": label_text,
                    "Grade": f"{float(player_row[col]):.1f}",
                })
        if rows:
            st.table(pd.DataFrame(rows))

        notes = ADJUSTMENT_NOTES.get(position, [])
        if notes:
            st.markdown("**Adjustments applied**")
            for n in notes:
                st.markdown(f"- {n}")
        else:
            st.markdown("*No SOS / contextual adjustments in this v1 spec.*")

        # Composite formula reminder
        st.caption(
            "Composite GAS Score = Σ(bundle_grade × bundle_weight). "
            "Bundle grade = Σ(stat_grade × stat_weight). "
            "Stat grade = z_to_grade(shrunk_z(z_value, games)). "
            "z_to_grade maps standard normal z to a 0-100 scale "
            "where 50 = league average, 84 = +1σ, 95 = +1.7σ."
        )


def render_hot_take(player_row: pd.Series, league_df: pd.DataFrame,
                      position: str) -> None:
    """Auto-generate a contrarian / surprising take from the data."""
    score = float(player_row.get("gas_score", 50.0))
    name = (player_row.get("player_display_name")
              or player_row.get("full_name", "this player"))

    bundle_labels = BUNDLE_LABELS.get(position, {})
    grades = {}
    for bundle_key, label_text in bundle_labels.items():
        col = f"gas_{bundle_key}_grade"
        if col in player_row.index:
            grades[(bundle_key, label_text)] = float(player_row[col])
    if not grades:
        return

    top_bundle = max(grades.items(), key=lambda x: x[1])
    bot_bundle = min(grades.items(), key=lambda x: x[1])
    spread = top_bundle[1] - bot_bundle[1]

    # Build a take based on observed pattern
    take = None
    if spread > 35 and top_bundle[1] > 75:
        take = (
            f"**One-trick or genuinely elite?** "
            f"{name}'s {top_bundle[0][1].lower()} grade "
            f"({top_bundle[1]:.0f}) is a strength — but his "
            f"{bot_bundle[0][1].lower()} ({bot_bundle[1]:.0f}) is a "
            f"real weakness. Specialist profile."
        )
    elif spread < 12 and score > 65:
        take = (
            f"**Quietly complete.** {name} doesn't lead any single "
            f"category but every bundle is above {bot_bundle[1]:.0f}. "
            f"This is the bundle profile of a player whose floor is "
            f"high even when his ceiling looks ordinary."
        )
    elif score > 75 and bot_bundle[1] < 45:
        take = (
            f"**Top-tier with a hole.** {name} grades elite overall "
            f"but his {bot_bundle[0][1].lower()} ({bot_bundle[1]:.0f}) "
            f"is below average — a real weakness teams could exploit "
            f"if they game-plan for it."
        )
    elif score < 50 and top_bundle[1] > 70:
        take = (
            f"**Skill-set without a role.** {name}'s overall grade "
            f"is below average ({score:.1f}), but his "
            f"{top_bundle[0][1].lower()} ({top_bundle[1]:.0f}) is "
            f"genuinely good. A misused or wrong-scheme fit?"
        )

    if take is None:
        # Generic fallback — highlight the standout bundle
        if top_bundle[1] > 80:
            take = (
                f"**{top_bundle[0][1]} is the calling card.** "
                f"{name}'s {top_bundle[0][1].lower()} grade is "
                f"{top_bundle[1]:.0f} — top-tier in the league at "
                f"this skill, regardless of what the composite says."
            )
        else:
            take = (
                f"**Solid mid-tier across the board.** No single "
                f"bundle stands out for {name}, no single bundle "
                f"drags him down. A reliable starter."
            )

    st.markdown(
        f"""<div style="background:rgba(234,179,8,0.10);
            border-left:4px solid #eab308;padding:12px 16px;
            border-radius:6px;margin:12px 0;">
            <div style="font-size:11px;letter-spacing:1px;
                        color:#eab308;text-transform:uppercase;
                        font-weight:600;">GAS says...</div>
            <div style="font-size:14px;color:#f3f4f6;
                        margin-top:6px;">{take}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Post-card extras for the player detail view ───────────────────

def render_player_gas_extras(player_row: pd.Series,
                                position: str) -> None:
    """Render the bundle breakdown + Show Math + Hot Take below
    a player card. Each in its own expander so the card stays clean."""
    bundle_labels = BUNDLE_LABELS.get(position, {})
    bundle_rows = []
    for bundle_key, label_text in bundle_labels.items():
        col = f"gas_{bundle_key}_grade"
        if col in player_row.index:
            bundle_rows.append(
                (label_text, float(player_row[col])))

    with st.expander("📊 GAS bundle breakdown",
                       expanded=False):
        if bundle_rows:
            bdf = pd.DataFrame(bundle_rows,
                                  columns=["Bundle", "Grade"])
            st.bar_chart(bdf.set_index("Bundle"), height=240)
            st.caption(
                "Each bundle grades 0-100. The composite GAS Score is "
                "a weighted average across these. Click 'Show the math' "
                "for the formula and adjustments applied."
            )
        else:
            st.caption("No bundle data available.")

    # Hot take inline (not in expander — should be visible)
    league_df = load_gas_data(position)
    if league_df is not None:
        season_year = int(player_row.get("season_year", 0) or 0)
        season_df = league_df[league_df["season_year"] == season_year]
        render_hot_take(player_row, season_df, position)

    # Show math collapsed
    render_show_math_panel(player_row, position)


# ── One-call team renderer for position pages ─────────────────────

# Some position pages use `recent_team` / `team_abbr` / `team`. Try them.
TEAM_COLS = ["recent_team", "team", "team_abbr", "posteam", "recent_team_abbr"]


def _filter_team_season(df: pd.DataFrame, team: str,
                          season: int) -> pd.DataFrame:
    """Return rows for (team, season). Tries each known team-col."""
    for col in TEAM_COLS:
        if col in df.columns:
            sub = df[(df[col] == team) & (df["season_year"] == season)]
            if len(sub):
                return sub
    return df.iloc[:0]


def render_team_gas_section(position: str, team: str, season: int,
                                title: str = "GAS Score") -> None:
    """Drop-in: renders the position's full GAS section for a team.

    Usage in a position page:
        import lib_gas_panels as gp
        gp.render_team_gas_section("qb", selected_team, selected_season)
    """
    df = load_gas_data(position)
    if df is None:
        st.info(
            f"GAS data for {position.upper()} not found — "
            "build it via tools/build_{position}_gas.py"
        )
        return
    sub = _filter_team_season(df, team, season)
    if sub.empty:
        st.caption(
            f"No GAS-graded {position.upper()} for {team} {season} — "
            "either no qualifying snaps or data not yet rebuilt for "
            "this season."
        )
        return

    sub = sub.sort_values("gas_score", ascending=False)

    st.markdown(f"### 🏆 {title}")
    st.caption(
        "**GAS Score** — Game-Adjusted Skill: 0-100 proprietary "
        "Lions Rater grade. Opponent + context + role-adjusted. "
        "Higher is better. 50 = league average. 70+ = above-average "
        "starter. 80+ = high-end starter. 90+ = elite."
    )

    # Quick league-rank context
    season_df = df[df["season_year"] == season].copy()

    for _, row in sub.iterrows():
        name = (row.get("player_display_name")
                  or row.get("full_name", "Unknown"))
        score = float(row.get("gas_score", 50.0))
        label = str(row.get("gas_label", ""))
        rank = (season_df["gas_score"] > score).sum() + 1
        total = len(season_df)
        with st.expander(
            f"**{name}** — GAS {score:.1f}  ·  {label}  ·  "
            f"#{rank}/{total} at position",
            expanded=False,
        ):
            render_gas_score_card(row, position)
            render_hot_take(row, season_df, position)
            render_show_math_panel(row, position)
