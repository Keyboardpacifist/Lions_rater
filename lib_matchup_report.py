"""Matchup Report — auto-generated game-bet analysis.

Pick a Team A vs Team B + season + week and the engine returns a
structured multi-section report pulling from every game-bet engine
in the lab. This is the "lazy bettor showcase" — retail users get
a sharp friend's full take on a game without touching any data tab.

Sections (in order):
  • headline           — teams, line, total, moneyline, kickoff
  • game_environment   — weather, surface, roof, rest, divisional
  • injuries           — both teams, ranked by severity, with cohort
                         play probability for each affected starter
  • scheme_matchup     — top 3 distinctive scheme metrics for each
                         team (offense + defense), with deltas
  • coaching_angles    — top 3 coaching tendency outliers per team
  • books_vs_model     — historical book-behavior signal IF a key
                         starter is on the report
  • bottom_line        — composite bet-actionable summary

The output is a structured `MatchupReport` object the UI can render
as cards / sections / collapsibles.

Public entry point
------------------
    generate_matchup_report(home_team, away_team, season, week)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib_injury_cohort import (
    body_part_normalize,
    practice_status_code,
    predict as cohort_predict,
    report_status_code,
)
from lib_scheme_deltas import (
    DEFENSE_METRICS,
    METRIC_LABELS,
    OFFENSE_METRICS,
    load_scheme_deltas,
)


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
SCHEDULES = DATA / "nfl_schedules.parquet"
INJURIES = DATA / "nfl_injuries_historical.parquet"
PLAYER_STATS = DATA / "nfl_player_stats_weekly.parquet"
COACHING = DATA / "coaching_tendencies.parquet"
BOOKS = DATA / "books_vs_model.parquet"
GAME_SCRIPT = DATA / "game_script_deltas.parquet"


@st.cache_data(show_spinner=False)
def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# ── Report dataclass ─────────────────────────────────────────────

@dataclass
class InjuryNote:
    player_name: str
    position: str
    body_part: str
    status: str
    practice: str
    p_play: float       # cohort probability
    snap_retention: float
    role: str           # QB1 / RB1 / WR1 / TE1 / OTHER
    cohort_n: int


@dataclass
class SchemeNote:
    team: str
    metric: str
    metric_label: str
    value: float
    delta: float
    side: str           # offense / defense


@dataclass
class CoachingNote:
    team: str
    metric: str
    metric_label: str
    value: float
    delta: float


@dataclass
class BooksNote:
    team: str
    role_lost: str
    body_part: str
    status: str
    n_games: int
    mean_line_miss: float
    cover_rate: float
    verdict: str


@dataclass
class MatchupNarrative:
    """The plain-English take that goes ABOVE the data sections."""
    one_liner: str           # single-sentence summary of our take
    primary_lean: str        # "HOU -4.5" / "OVER 49" / "PASS"
    primary_confidence: int  # 1–5 stars
    primary_confidence_label: str  # HIGH / MEDIUM-HIGH / MEDIUM / LOW / PASS
    secondary_lean: str | None     # secondary bet (over/under often)
    secondary_confidence: int | None
    why_bullets: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    inefficiencies: list[str] = field(default_factory=list)


@dataclass
class MatchupReport:
    home_team: str
    away_team: str
    season: int
    week: int
    headline: dict          # spread/total/moneyline + meta
    game_environment: dict  # weather + surface + rest
    home_injuries: list[InjuryNote] = field(default_factory=list)
    away_injuries: list[InjuryNote] = field(default_factory=list)
    home_scheme: list[SchemeNote] = field(default_factory=list)
    away_scheme: list[SchemeNote] = field(default_factory=list)
    home_coaching: list[CoachingNote] = field(default_factory=list)
    away_coaching: list[CoachingNote] = field(default_factory=list)
    books_signals: list[BooksNote] = field(default_factory=list)
    bottom_line_bullets: list[str] = field(default_factory=list)
    narrative: MatchupNarrative | None = None


# ── Helpers ──────────────────────────────────────────────────────

KEY_ROLES = {"QB", "RB", "WR", "TE"}

ROLE_PRIORITY = {"QB1": 5, "RB1": 4, "WR1": 4, "WR2": 3, "TE1": 3,
                 "OTHER": 1}
STATUS_SEVERITY = {"OUT": 5, "DOUBTFUL": 4, "QUESTIONABLE": 3,
                   "PROBABLE": 2, "NONE": 0}


def _identify_team_starters(team: str, season: int) -> dict[str, str]:
    """Map (gsis_id) → role for this team-season's starters."""
    df = _load(PLAYER_STATS)
    if df.empty:
        return {}
    sub = df[(df["team"] == team) & (df["season"] == int(season))]
    out = {}
    qb = (sub[sub["position"] == "QB"]
          .groupby("player_id")["attempts"].sum()
          .sort_values(ascending=False))
    if not qb.empty and qb.iloc[0] > 50:
        out[qb.index[0]] = "QB1"
    rb = (sub[sub["position"] == "RB"]
          .groupby("player_id")["carries"].sum()
          .sort_values(ascending=False))
    for i, slot in enumerate(["RB1", "RB2"]):
        if len(rb) > i and rb.iloc[i] > 30:
            out[rb.index[i]] = slot
    wr = (sub[sub["position"] == "WR"]
          .groupby("player_id")["targets"].sum()
          .sort_values(ascending=False))
    for i, slot in enumerate(["WR1", "WR2", "WR3"]):
        if len(wr) > i and wr.iloc[i] > 20:
            out[wr.index[i]] = slot
    te = (sub[sub["position"] == "TE"]
          .groupby("player_id")["targets"].sum()
          .sort_values(ascending=False))
    if not te.empty and te.iloc[0] > 20:
        out[te.index[0]] = "TE1"
    return out


def _team_injuries(team: str, season: int, week: int) -> list[InjuryNote]:
    """Pull all injury-report rows for this team in (season, week) and
    annotate with cohort prediction + role tag."""
    inj = _load(INJURIES)
    if inj.empty:
        return []
    sub = inj[(inj["team"] == team)
              & (inj["season"] == int(season))
              & (inj["week"] == int(week))
              & inj["report_status"].notna()]
    if sub.empty:
        return []
    starter_map = _identify_team_starters(team, season)

    notes: list[InjuryNote] = []
    for _, r in sub.iterrows():
        status = report_status_code(r["report_status"])
        practice = practice_status_code(r["practice_status"])
        body_part = body_part_normalize(r["report_primary_injury"])
        position = str(r["position"])

        result = cohort_predict(
            position=position, body_part=body_part,
            report_status=status, practice_status=practice,
        )
        role = starter_map.get(str(r["gsis_id"]), "OTHER")
        notes.append(InjuryNote(
            player_name=str(r["full_name"]),
            position=position,
            body_part=body_part,
            status=status,
            practice=practice,
            p_play=float(result.p_played),
            snap_retention=float(result.snap_retention_if_played),
            role=role,
            cohort_n=int(result.n),
        ))

    # Sort by combined severity: starter-role × status-severity
    notes.sort(
        key=lambda x: (ROLE_PRIORITY.get(x.role, 1)
                       * STATUS_SEVERITY.get(x.status, 0)),
        reverse=True,
    )
    return notes


def _team_scheme_outliers(team: str, season: int,
                            top_n: int = 3) -> list[SchemeNote]:
    """Return the N most-distinctive offensive + defensive scheme
    metrics for this team-season."""
    sd = load_scheme_deltas()
    if sd.empty:
        return []
    out: list[SchemeNote] = []
    for side, metrics in (("offense", OFFENSE_METRICS),
                          ("defense", DEFENSE_METRICS)):
        row = sd[(sd["team"] == team)
                 & (sd["season"] == int(season))
                 & (sd["side"] == side)]
        if row.empty:
            continue
        r = row.iloc[0]
        ranked = []
        for m in metrics:
            delta_col = f"{m}_delta"
            if delta_col in r.index and pd.notna(r[delta_col]):
                ranked.append((m, float(r[m]) if pd.notna(r[m]) else 0,
                                float(r[delta_col])))
        ranked.sort(key=lambda x: abs(x[2]), reverse=True)
        for m, val, delta in ranked[:top_n]:
            out.append(SchemeNote(
                team=team, metric=m,
                metric_label=METRIC_LABELS.get(m, m),
                value=val, delta=delta, side=side,
            ))
    return out


def _team_coaching_outliers(team: str, season: int,
                              top_n: int = 3) -> list[CoachingNote]:
    """Top distinctive coaching-tendency metrics."""
    df = _load(COACHING)
    if df.empty:
        return []
    row = df[(df["team"] == team) & (df["season"] == int(season))]
    if row.empty:
        return []
    r = row.iloc[0]
    metric_labels = {
        "fourth_short_go_rate": "4th-and-short go rate",
        "fourth_long_go_rate": "4th-and-long go rate",
        "two_pt_attempt_rate": "2-point attempt rate",
        "pass_rate_leading_7p": "Pass rate when leading 7+",
        "pass_rate_trailing_7p": "Pass rate when trailing 7+",
        "pass_rate_q4_trailing": "Q4 pass rate when trailing",
        "run_rate_q4_leading": "Q4 run rate when leading",
        "rz_run_rate": "Red-zone run rate",
        "two_min_drill_plays_pg": "2-min drill plays/g",
    }
    ranked: list[tuple[str, float, float]] = []
    for m in metric_labels:
        delta_col = f"{m}_delta"
        if delta_col in r.index and pd.notna(r[delta_col]):
            ranked.append((m, float(r[m]) if pd.notna(r[m]) else 0,
                           float(r[delta_col])))
    ranked.sort(key=lambda x: abs(x[2]), reverse=True)
    return [
        CoachingNote(team=team, metric=m,
                      metric_label=metric_labels[m],
                      value=val, delta=delta)
        for m, val, delta in ranked[:top_n]
    ]


def _books_signals_for_team(team: str,
                              injuries: list[InjuryNote]) -> list[BooksNote]:
    """For each starter on the injury report, find the matching
    historical book-behavior cohort and surface the signal."""
    bv = _load(BOOKS)
    if bv.empty:
        return []
    out: list[BooksNote] = []
    for inj in injuries:
        if inj.role not in ("QB1", "RB1", "WR1") or inj.status == "NONE":
            continue
        role_key = inj.role.replace("1", "")
        # Try (role, status, body_part) → loosen if too thin
        sub = bv[(bv["position_lost"] == role_key)
                 & (bv["status"] == inj.status)
                 & (bv["body_part"] == inj.body_part)]
        if sub.empty or sub.iloc[0]["n_games"] < 10:
            sub = bv[(bv["position_lost"] == role_key)
                     & (bv["status"] == inj.status)]
        if sub.empty or sub.iloc[0]["n_games"] < 20:
            continue
        row = sub.iloc[0]
        miss = float(row["mean_line_miss"])
        if miss > 1.5:
            verdict = ("Books historically OVER-react — contrarian "
                       f"bet ON {team}")
        elif miss < -1.5:
            verdict = f"Books historically UNDER-react — fade {team}"
        else:
            verdict = "Books price this fairly"
        out.append(BooksNote(
            team=team,
            role_lost=role_key,
            body_part=inj.body_part,
            status=inj.status,
            n_games=int(row["n_games"]),
            mean_line_miss=miss,
            cover_rate=float(row["cover_rate"]),
            verdict=verdict,
        ))
    return out


def _bottom_line(report: MatchupReport) -> list[str]:
    """Synthesize 3-5 bet-actionable bullets from all sections."""
    bullets: list[str] = []
    headline = report.headline

    # 1. Spread + total framing
    # nflverse spread_line = home team's expected margin: positive = home
    # favored. Verified empirically (corr with home_margin = +0.43).
    if headline.get("spread_line") is not None:
        sl = headline["spread_line"]
        favored = report.home_team if sl > 0 else report.away_team
        bullets.append(
            f"**Line:** {favored} -{abs(sl):.1f} · Total {headline.get('total_line', '?'):.1f}"
        )

    # 2. Material injuries
    home_starters_out = [i for i in report.home_injuries
                          if i.role in ("QB1", "RB1", "WR1", "TE1")
                          and i.status in ("OUT", "DOUBTFUL")]
    away_starters_out = [i for i in report.away_injuries
                          if i.role in ("QB1", "RB1", "WR1", "TE1")
                          and i.status in ("OUT", "DOUBTFUL")]
    if home_starters_out:
        names = ", ".join(f"{i.role} {i.player_name}" for i in home_starters_out[:3])
        bullets.append(f"**{report.home_team} key absences:** {names}")
    if away_starters_out:
        names = ", ".join(f"{i.role} {i.player_name}" for i in away_starters_out[:3])
        bullets.append(f"**{report.away_team} key absences:** {names}")

    # 3. Book-behavior verdicts
    for s in report.books_signals:
        if abs(s.mean_line_miss) >= 1.5:
            bullets.append(
                f"**Book signal:** {s.team} {s.role_lost} {s.status}/"
                f"{s.body_part} → "
                f"{s.mean_line_miss:+.1f} pts vs. close in "
                f"{s.n_games} historical games. {s.verdict}"
            )

    # 4. Scheme matchup take
    home_blitz = next((s for s in report.home_scheme
                        if s.metric == "blitz_rate" and s.side == "defense"),
                       None)
    if home_blitz and abs(home_blitz.delta) >= 0.05:
        direction = "elite" if home_blitz.delta > 0 else "soft"
        bullets.append(
            f"**Scheme:** {report.home_team} D blitz rate "
            f"{home_blitz.value:.0%} ({home_blitz.delta:+.0%} vs "
            f"league) — {direction} pressure tilt"
        )

    # 5. Weather warning
    env = report.game_environment
    if env.get("temp") is not None and env["temp"] < 35:
        bullets.append(
            f"**Weather:** {env['temp']:.0f}°F, "
            f"{env.get('wind', 0):.0f} mph wind — fade passing volume"
        )
    elif env.get("wind") and env["wind"] >= 15:
        bullets.append(
            f"**Weather:** {env['wind']:.0f} mph wind — fade deep "
            "balls + kicker props"
        )

    if not bullets:
        bullets.append("No flag-level signals on this matchup. "
                       "Default to baseline rate analysis.")
    return bullets


def generate_matchup_report(home_team: str, away_team: str,
                              season: int, week: int) -> MatchupReport:
    """Top-level entry point. Returns a fully-populated MatchupReport."""
    sch = _load(SCHEDULES)
    g = sch[(sch["season"] == int(season))
            & (sch["week"] == int(week))
            & (sch["home_team"] == home_team)
            & (sch["away_team"] == away_team)]
    if g.empty:
        # Allow swap (sometimes user enters away/home reversed)
        g = sch[(sch["season"] == int(season))
                & (sch["week"] == int(week))
                & (sch["home_team"] == away_team)
                & (sch["away_team"] == home_team)]
        if not g.empty:
            home_team, away_team = away_team, home_team
    if g.empty:
        # Empty report when game not found
        return MatchupReport(
            home_team=home_team, away_team=away_team,
            season=int(season), week=int(week),
            headline={"error": "Matchup not found in schedule."},
            game_environment={},
        )

    row = g.iloc[0]
    headline = {
        "kickoff": str(row.get("gameday", "?")),
        "spread_line": (float(row["spread_line"])
                        if pd.notna(row.get("spread_line")) else None),
        "total_line": (float(row["total_line"])
                       if pd.notna(row.get("total_line")) else None),
        "home_moneyline": (int(row["home_moneyline"])
                            if pd.notna(row.get("home_moneyline")) else None),
        "away_moneyline": (int(row["away_moneyline"])
                            if pd.notna(row.get("away_moneyline")) else None),
        "home_qb": str(row.get("home_qb_name", "?")),
        "away_qb": str(row.get("away_qb_name", "?")),
        "home_coach": str(row.get("home_coach", "?")),
        "away_coach": str(row.get("away_coach", "?")),
        "referee": str(row.get("referee", "?")),
    }
    env = {
        "stadium": str(row.get("stadium", "?")),
        "roof": str(row.get("roof", "?")),
        "surface": str(row.get("surface", "?")),
        "temp": (float(row["temp"]) if pd.notna(row.get("temp")) else None),
        "wind": (float(row["wind"]) if pd.notna(row.get("wind")) else None),
        "home_rest": (int(row["home_rest"])
                      if pd.notna(row.get("home_rest")) else None),
        "away_rest": (int(row["away_rest"])
                      if pd.notna(row.get("away_rest")) else None),
        "div_game": (int(row["div_game"])
                     if pd.notna(row.get("div_game")) else 0),
    }

    report = MatchupReport(
        home_team=home_team, away_team=away_team,
        season=int(season), week=int(week),
        headline=headline, game_environment=env,
    )
    report.home_injuries = _team_injuries(home_team, season, week)
    report.away_injuries = _team_injuries(away_team, season, week)
    report.home_scheme = _team_scheme_outliers(home_team, season)
    report.away_scheme = _team_scheme_outliers(away_team, season)
    report.home_coaching = _team_coaching_outliers(home_team, season)
    report.away_coaching = _team_coaching_outliers(away_team, season)
    report.books_signals = (_books_signals_for_team(home_team,
                                                      report.home_injuries)
                              + _books_signals_for_team(away_team,
                                                         report.away_injuries))
    report.bottom_line_bullets = _bottom_line(report)
    report.narrative = _build_narrative(report)
    return report


# ════════════════════════════════════════════════════════════════
#                   NARRATIVE GENERATOR
# ════════════════════════════════════════════════════════════════
# Translates the structured report into a plain-English "take" with
# directional lean, confidence rating, why-bullets, and risk flags.
# This is what retail gamblers actually read — the data sections
# below it are the receipts.

CONFIDENCE_LABELS = {
    5: "HIGH",
    4: "MEDIUM-HIGH",
    3: "MEDIUM",
    2: "LOW",
    1: "VERY LOW",
    0: "PASS",
}


def _confidence_label(score: int) -> str:
    return CONFIDENCE_LABELS.get(max(0, min(5, score)), "MEDIUM")


def _key_starters_out(injuries: list[InjuryNote]) -> list[InjuryNote]:
    """Return only the QB1/RB1/WR1/TE1 injuries that materially
    affect the game."""
    return [i for i in injuries
            if i.role in ("QB1", "RB1", "WR1", "TE1")
            and i.status in ("OUT", "DOUBTFUL")]


def _build_narrative(r: MatchupReport) -> MatchupNarrative:
    """Synthesize structured report into plain-English take.
    Confidence scoring:
        +2 strong book-behavior signal (n≥30, |miss|≥3)
        +1 medium book-behavior signal (n≥20, |miss|≥1.5)
        +1 a key starter (QB1) OUT
        +0.5 a non-QB key starter (RB1/WR1/TE1) OUT
        +1 extreme weather affecting passing
        +0.5 elite defensive scheme outlier
        +0.5 multiple corroborating signals
    """
    why: list[str] = []
    risk: list[str] = []
    inefficiencies: list[str] = []
    spread = r.headline.get("spread_line")
    total = r.headline.get("total_line")
    env = r.game_environment

    # Score primary spread direction
    spread_score = 0  # positive = lean home; negative = lean away
    confidence = 0

    home_starters_out = _key_starters_out(r.home_injuries)
    away_starters_out = _key_starters_out(r.away_injuries)

    # ── Books-vs-model signals (strongest direct evidence) ──
    for s in r.books_signals:
        if s.n_games < 20:
            continue
        sign = +1 if s.team == r.away_team else -1
        # line_miss < 0 = team underperformed → fade them
        # line_miss > 0 = team overperformed → bet ON them
        weight = 2 if abs(s.mean_line_miss) >= 3 else 1
        if abs(s.mean_line_miss) >= 1.5:
            confidence += weight
            if s.mean_line_miss < -1.5:
                # fade the affected team → push spread away from them
                spread_score -= sign  # if home, pushes negative (toward away)
                inefficiencies.append(
                    f"Books historically UNDER-react when "
                    f"{s.role_lost} is {s.status} with {s.body_part} — "
                    f"{s.team} underperforms close by "
                    f"{s.mean_line_miss:+.1f} pts in {s.n_games} cases"
                )
                why.append(
                    f"**Fade {s.team}** — historically lose by "
                    f"{abs(s.mean_line_miss):.1f}+ more than the "
                    f"close implies in this exact injury scenario "
                    f"({s.n_games} games, {s.cover_rate:.0%} cover rate)"
                )
            else:
                spread_score += sign
                inefficiencies.append(
                    f"Books historically OVER-react when "
                    f"{s.role_lost} is {s.status} with {s.body_part} — "
                    f"{s.team} actually beats close by "
                    f"{s.mean_line_miss:+.1f} pts in {s.n_games} cases"
                )
                why.append(
                    f"**Contrarian on {s.team}** — historically beat "
                    f"close by {s.mean_line_miss:+.1f} pts in this "
                    f"exact injury scenario ({s.n_games} games)"
                )

    # ── QB1 out (4 pt scoring drop league-wide) ──
    for i in home_starters_out:
        if i.role == "QB1":
            confidence += 1
            spread_score -= 1
            why.append(
                f"**{r.home_team} QB1 {i.player_name} {i.status}** "
                f"({i.body_part}) — league-wide QB1-out games average "
                f"−4 pts/game"
            )
    for i in away_starters_out:
        if i.role == "QB1":
            confidence += 1
            spread_score += 1
            why.append(
                f"**{r.away_team} QB1 {i.player_name} {i.status}** "
                f"({i.body_part}) — league-wide QB1-out games average "
                f"−4 pts/game"
            )

    # ── Other key starters (lighter weight) ──
    for team_label, side_starters, sign in (
        (r.home_team, home_starters_out, -1),
        (r.away_team, away_starters_out, +1),
    ):
        non_qb = [i for i in side_starters if i.role != "QB1"]
        if non_qb:
            names = ", ".join(f"{i.role} {i.player_name}"
                              for i in non_qb[:2])
            why.append(f"**{team_label} absences:** {names}")
            # Modest weight — RB1/WR1/TE1 individually ≈ 0 pts league-wide,
            # but coaches/lines DO move on these names. Half-point.
            spread_score += 0  # neutral on lean, but show the info

    # ── Extreme weather ──
    temp = env.get("temp")
    wind = env.get("wind")
    if temp is not None and temp <= 32:
        confidence += 1
        why.append(f"**Cold ({temp:.0f}°F)** — passing volume drops, "
                   f"defense covers easier")
        # Cold games tend toward unders
        inefficiencies.append(
            f"Cold weather ({temp:.0f}°F) historically reduces "
            "passing volume; total-line bias favors the under"
        )
    if wind is not None and wind >= 15:
        confidence += 1
        why.append(f"**Wind {wind:.0f} mph** — fade deep balls + "
                   f"kicker props")
        inefficiencies.append(
            f"Wind ({wind:.0f} mph) — books often under-price the "
            "downward effect on passing distance + kicking accuracy"
        )

    # ── Elite defensive EPA ──
    for team_label, scheme in ((r.home_team, r.home_scheme),
                                (r.away_team, r.away_scheme)):
        epa_def = next((s for s in scheme
                         if s.metric == "epa_per_play_def"), None)
        if epa_def and epa_def.delta <= -0.06:
            confidence += 0.5
            why.append(
                f"**{team_label} defense elite** — EPA/play allowed "
                f"{epa_def.value:+.3f} ({epa_def.delta:+.3f} vs league)"
            )
            inefficiencies.append(
                f"{team_label} defensive EPA ranks elite — markets "
                "under-price stingy defenses on team totals"
            )

    # ── Risk flags ──
    if env.get("div_game"):
        risk.append(
            "Divisional matchup — games trend tighter than expected; "
            "narrow your edge"
        )
    if total and total >= 50:
        risk.append(
            f"High implied total ({total:.1f}) — score variance is "
            "wide; small spread edges can flip"
        )
    if not r.books_signals:
        risk.append(
            "No book-behavior cohort matched — relying on softer "
            "signals (scheme + injuries + weather)"
        )

    # ── Translate score → primary lean ──
    # Use half-up rounding so 0.5 → 1 (not Python's banker's 0)
    primary_conf = max(0, min(5, int(confidence + 0.5)))
    if spread is None or primary_conf == 0:
        primary_lean = "PASS — no clear edge"
        primary_label = _confidence_label(0)
    else:
        # nflverse: positive spread = home favored, negative = home dog.
        favored = r.home_team if spread > 0 else r.away_team
        underdog = r.away_team if spread > 0 else r.home_team
        if abs(spread_score) < 0.5:
            primary_lean = "PASS — signals balanced"
            primary_label = _confidence_label(0)
        elif spread_score > 0:
            primary_lean = (f"Take {r.home_team} "
                            f"{'-' if spread > 0 else '+'}"
                            f"{abs(spread):.1f}")
            primary_label = _confidence_label(primary_conf)
        else:
            primary_lean = (f"Take {r.away_team} "
                            f"{'-' if spread < 0 else '+'}"
                            f"{abs(spread):.1f}")
            primary_label = _confidence_label(primary_conf)

    # ── Secondary lean — total ──
    secondary_lean: str | None = None
    secondary_conf: int | None = None
    total_score = 0  # negative = under, positive = over
    if temp is not None and temp <= 32:
        total_score -= 1
    if wind is not None and wind >= 15:
        total_score -= 1
    if any(s.metric == "epa_per_play_def" and s.delta <= -0.06
           for s in r.home_scheme + r.away_scheme):
        total_score -= 0.5
    if env.get("div_game"):
        total_score -= 0.3  # divisional games trend under
    if total is not None and abs(total_score) >= 1:
        secondary_conf = min(5, max(1, int(abs(total_score) + 1.5)))
        secondary_lean = (f"UNDER {total:.1f}" if total_score < 0
                           else f"OVER {total:.1f}")

    # ── One-liner ──
    if primary_conf == 0:
        one_liner = ("No exploitable inefficiency detected on this "
                      "matchup — pass.")
    else:
        stars = "★" * primary_conf + "☆" * (5 - primary_conf)
        one_liner = f"{primary_lean}  {stars} ({primary_label} confidence)"

    if not why:
        why = ["No high-conviction signals — this matchup is mostly "
               "noise."]
    if not inefficiencies:
        inefficiencies = ["No structural mispricing detected on this "
                           "matchup."]

    return MatchupNarrative(
        one_liner=one_liner,
        primary_lean=primary_lean,
        primary_confidence=primary_conf,
        primary_confidence_label=primary_label,
        secondary_lean=secondary_lean,
        secondary_confidence=secondary_conf,
        why_bullets=why,
        risk_flags=risk,
        inefficiencies=inefficiencies,
    )
