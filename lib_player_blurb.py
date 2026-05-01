"""Auto-generate plain-English player blurbs for the player cards.

Three-beat format Brett wants for fans:

  Part A  →  What he's best at vs. league average + the skill set
             those exemplify.
  Part B  →  Situational setup he's most dangerous in.
             Fallback ladder: situational pbp → draft slot value →
             FA contract value → skip.
  Part C  →  Main drawbacks.

Style rules (from Brett):
  • Plain conversational sentences. Periods do the work.
  • No em-dashes, no colons, no comma-heavy run-ons.
  • Numbers only when a number is the point ("most explosive"),
    not a stat dump.

Public entrypoint:
  generate_blurb(player_row, cohort_df, position) -> str
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Variant rotation ────────────────────────────────────────────
# Each "sentence type" has a list of phrasings. We pick one
# deterministically per (player, section) so the same player gets the
# same blurb on every page load, but two players with similar profiles
# don't read identically. Hash → mod into the variant list.

def _pick(player_name: str, section: str, variants: list[str]) -> str:
    if not variants:
        return ""
    seed = f"{player_name}|{section}"
    h = hashlib.md5(seed.encode()).hexdigest()
    idx = int(h[:8], 16) % len(variants)
    return variants[idx]

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"

PBP_PATH = DATA / "game_pbp.parquet"
DRAFT_PATH = DATA / "nfl_draft_picks.parquet"
CONTRACT_PATH = DATA / "nfl_contracts.parquet"


# ── Skill-group definitions ─────────────────────────────────────
# Each "skill group" bundles 1-N related z-cols into a single
# fan-readable concept. The blurb engine ranks groups by mean z
# and picks the strongest for Part A and weakest for Part C.

@dataclass
class SkillGroup:
    name: str
    stats: list           # list of *_z column names
    # All three phrases follow "He ..." cleanly so the engine can
    # chain them with "and" / "but" without grammar acrobatics.
    high: str             # used in Part A when z is high (strength)
    low: str              # used in Part C (NFL) — current drawback
    improve: str = ""     # used in Part C (prospect) — "He could
                          # take his game to the next level by
                          # {improve}." Phrase is a gerund clause:
                          # "improving his short-yardage running",
                          # "developing into a touchdown threat", etc.


PROSPECT_SKILLS: dict[str, list[SkillGroup]] = {
    "QB": [
        SkillGroup("accuracy", ["completion_pct_z"],
            "is one of the most accurate passers in college football",
            "struggles with accuracy",
            "tightening up his accuracy"),
        SkillGroup("big_play_passing", ["yards_per_attempt_z"],
            "throws the deep ball as well as anyone in the country",
            "rarely pushes the ball downfield",
            "pushing the ball downfield more"),
        SkillGroup("td_production", ["td_rate_z", "pass_tds_z"],
            "puts up huge touchdown numbers",
            "doesn't put up big touchdown numbers",
            "becoming a bigger touchdown producer"),
        SkillGroup("ball_security", ["int_rate_z"],
            "is extremely careful with the football",
            "is turnover-prone",
            "cleaning up the turnovers"),
        SkillGroup("mobility", ["rush_yards_total_z"],
            "is a real threat to extend plays with his legs",
            "is a pure pocket passer who rarely runs",
            "adding a bit more mobility to his game"),
    ],
    "WR": [
        SkillGroup("volume", ["receptions_total_z", "rec_yards_total_z"],
            "is the #1 target absorber for his offense",
            "is lightly used in his offense",
            "earning a bigger role in the passing game"),
        SkillGroup("td_production", ["rec_tds_total_z"],
            "is a touchdown machine in the red zone",
            "rarely finds the end zone",
            "developing into a real red-zone threat"),
        SkillGroup("big_play", ["yards_per_rec_z"],
            "turns short catches into big gains",
            "is more of a possession receiver than a deep threat",
            "adding a deep-threat element to his game"),
    ],
    "TE": [
        SkillGroup("volume", ["receptions_total_z", "rec_yards_total_z"],
            "is a featured pass-catcher in his offense",
            "is underused in the passing game",
            "earning more targets in the passing game"),
        SkillGroup("td_production", ["rec_tds_total_z"],
            "is a real touchdown threat near the goal line",
            "rarely finds the end zone",
            "developing into a red-zone weapon"),
        SkillGroup("big_play", ["yards_per_rec_z"],
            "is an explosive playmaker after the catch",
            "is more of a chain-mover than a vertical threat",
            "adding more explosive plays to his game"),
    ],
    "RB": [
        SkillGroup("workload", ["carries_total_z", "total_yards_z"],
            "carries a true workhorse load",
            "is stuck in a committee role",
            "earning a true workhorse role"),
        SkillGroup("td_finishing", ["rush_tds_total_z"],
            "is a finisher near the goal line",
            "rarely gets the ball at the goal line",
            "becoming a bigger finisher near the goal line"),
        SkillGroup("big_play", ["yards_per_carry_z"],
            "is an explosive home-run hitter on every carry",
            "rarely breaks long runs",
            "creating more explosive runs"),
        SkillGroup("production", ["rush_yards_total_z"],
            "puts up massive rushing yardage totals",
            "has low rushing volume",
            "carrying a heavier rushing load"),
    ],
    "CB": [
        SkillGroup("ball_hawk", ["int_per_game_z"],
            "is a ball-hawk who picks off passes at an elite rate",
            "rarely takes the ball away",
            "creating more turnovers"),
        SkillGroup("coverage", ["pd_per_game_z"],
            "is constantly breaking up passes in coverage",
            "rarely makes plays on the ball",
            "making more plays on the ball in coverage"),
        SkillGroup("tackling", ["tackles_per_game_z", "solo_tackles_per_game_z"],
            "is a willing and reliable tackler in run support",
            "is a liability as a tackler",
            "becoming a more reliable tackler"),
    ],
    "S": [
        SkillGroup("ball_hawk", ["int_per_game_z"],
            "is a ball-hawk safety who creates turnovers",
            "rarely takes the ball away",
            "creating more turnovers"),
        SkillGroup("coverage", ["pd_per_game_z"],
            "is constantly breaking up passes in the deep half",
            "rarely impacts the passing game",
            "making more plays on the ball"),
        SkillGroup("tackling", ["tackles_per_game_z", "solo_tackles_per_game_z"],
            "is a thumper in run support",
            "shies away from contact",
            "becoming a more reliable tackler"),
    ],
    "LB": [
        SkillGroup("backfield_disruption", ["tfl_per_game_z", "sacks_per_game_z"],
            "lives in the offensive backfield wrecking plays",
            "rarely makes plays behind the line of scrimmage",
            "making more plays behind the line of scrimmage"),
        SkillGroup("tackling", ["tackles_per_game_z", "solo_tackles_per_game_z"],
            "is a heat-seeking tackling machine",
            "doesn't rack up tackle volume",
            "becoming a higher-volume tackler"),
    ],
    "DE": [
        SkillGroup("pass_rush", ["sacks_per_game_z", "qb_hurries_per_game_z"],
            "is an elite pass-rusher who lives in the quarterback's lap",
            "doesn't generate consistent pressure",
            "becoming a more consistent pass-rusher"),
        SkillGroup("backfield_disruption", ["tfl_per_game_z"],
            "is a wrecking ball behind the line of scrimmage",
            "rarely makes plays behind the line",
            "creating more disruption behind the line of scrimmage"),
        SkillGroup("run_defense", ["tackles_per_game_z"],
            "is stout against the run",
            "is limited as a run defender",
            "becoming a more reliable run defender"),
    ],
    "DT": [
        SkillGroup("pass_rush", ["sacks_per_game_z"],
            "is a rare interior pass-rushing threat",
            "doesn't generate interior pressure",
            "becoming a real interior pass-rushing threat"),
        SkillGroup("backfield_disruption", ["tfl_per_game_z"],
            "lives in the backfield disrupting plays",
            "rarely makes plays behind the line",
            "creating more disruption behind the line"),
        SkillGroup("run_defense", ["tackles_per_game_z"],
            "is an anchor against the run",
            "gets washed out against the run",
            "anchoring better against the run"),
    ],
}


# Shared receiver skill set (WR + TE share the same parquet schema).
_RECEIVER_GROUPS = [
    SkillGroup("target_volume",
        ["targets_z", "target_share_z", "receptions_z", "wopr_z"],
        "soaks up a starter's share of the targets",
        "is barely involved in his offense",
        "earning a bigger share of the targets"),
    SkillGroup("production",
        ["rec_yards_z", "rec_tds_z"],
        "puts up huge receiving production",
        "doesn't put up big receiving numbers",
        "putting up bigger receiving numbers"),
    SkillGroup("big_play",
        ["yards_per_target_z", "air_yards_share_z"],
        "is a big-play threat down the field",
        "rarely makes big plays down the field",
        "becoming more of a downfield threat"),
    SkillGroup("yac",
        ["yac_per_reception_z", "yac_above_exp_z"],
        "creates yards after the catch better than almost anyone",
        "rarely creates yards after the catch",
        "creating more yards after the catch"),
    SkillGroup("separation",
        ["avg_separation_z"],
        "consistently gets open against tight coverage",
        "rarely creates separation",
        "creating more consistent separation"),
    SkillGroup("efficiency",
        ["catch_rate_z", "success_rate_z", "first_down_rate_z",
         "epa_per_target_z", "racr_z"],
        "is one of the most efficient receivers in the league",
        "is inefficient on a per-target basis",
        "improving his efficiency on a per-target basis"),
]


# Shared edge / interior pass-rusher skill set.
_PASS_RUSHER_GROUPS = [
    SkillGroup("pass_rush",
        ["sacks_per_game_z", "qb_hits_per_game_z",
         "hurries_per_game_z", "pressures_per_game_z",
         "qb_knockdowns_per_game_z", "pressure_rate_z"],
        "is one of the league's elite pass-rushers",
        "doesn't generate consistent pressure",
        "becoming a more consistent pass-rusher"),
    SkillGroup("backfield_disruption",
        ["tfl_per_game_z"],
        "lives in the offensive backfield",
        "rarely makes plays behind the line of scrimmage",
        "making more plays behind the line of scrimmage"),
    SkillGroup("run_defense",
        ["tackles_per_snap_z", "solo_tackle_rate_z"],
        "anchors against the run as well as anyone at his position",
        "is a liability against the run",
        "becoming a more reliable run defender"),
    SkillGroup("ball_disruption",
        ["forced_fumbles_per_game_z", "passes_defended_per_game_z",
         "interceptions_per_game_z"],
        "creates turnovers at a high rate",
        "rarely creates turnovers",
        "creating more turnovers"),
    SkillGroup("tackling",
        ["missed_tackle_pct_z"],
        "is one of the most reliable tacklers at his position",
        "misses too many tackles",
        "tightening up his tackling"),
]


POSITION_SKILLS: dict[str, list[SkillGroup]] = {
    "qb": [
        SkillGroup("volume_passing",
            ["passing_yards_per_game_z", "passing_tds_per_game_z",
             "td_rate_z"],
            "is one of the league's top volume passers",
            "doesn't put up huge passing numbers",
            "putting up bigger passing numbers"),
        SkillGroup("accuracy",
            ["completion_pct_z", "passing_cpoe_z"],
            "is one of the most accurate passers in the league",
            "struggles with accuracy",
            "tightening up his accuracy"),
        SkillGroup("big_play_passing",
            ["yards_per_attempt_z", "air_yards_per_attempt_z"],
            "throws the deep ball as well as anyone",
            "rarely pushes the ball downfield",
            "pushing the ball downfield more often"),
        SkillGroup("ball_security",
            ["int_rate_z", "turnover_rate_z"],
            "is extremely careful with the football",
            "is turnover-prone",
            "cleaning up the turnovers"),
        SkillGroup("pocket_management",
            ["sack_rate_z"],
            "rarely takes sacks",
            "takes too many sacks",
            "getting the ball out faster to avoid sacks"),
        SkillGroup("mobility",
            ["rush_yards_per_game_z", "rush_epa_per_carry_z"],
            "is a real threat to extend plays with his legs",
            "is a true pocket passer who rarely runs",
            "adding a bit more mobility to his game"),
        SkillGroup("efficiency",
            ["pass_epa_per_play_z", "pass_success_rate_z",
             "first_down_rate_z"],
            "is one of the most efficient quarterbacks in the league",
            "is inefficient on a per-play basis",
            "improving his per-play efficiency"),
    ],
    "wr": _RECEIVER_GROUPS,
    "te": _RECEIVER_GROUPS,
    "rb": [
        SkillGroup(
            "big_play_speed",
            ["explosive_run_rate_z", "explosive_15_rate_z"],
            "is one of the league's most explosive big-play runners",
            "rarely breaks long runs",
            "creating more explosive runs"),
        SkillGroup(
            "efficiency",
            ["yards_per_carry_z", "rush_success_rate_z"],
            "racks up elite yards per carry",
            "struggles with per-carry efficiency",
            "improving his per-carry efficiency"),
        SkillGroup(
            "workload",
            ["snap_share_z", "touches_per_game_z", "carries_z"],
            "carries a true starter's workload",
            "is stuck in a committee role",
            "earning a bigger share of the backfield touches"),
        SkillGroup(
            "creating_yards",
            ["ryoe_per_att_z", "yards_after_contact_per_att_z",
             "broken_tackles_per_att_z"],
            "creates yards on his own through balance and elusiveness",
            "rarely creates yards beyond what his blockers give him",
            "creating more yards on his own after contact"),
        SkillGroup(
            "receiving",
            ["receptions_z", "targets_per_game_z",
             "rec_yards_per_target_z", "rec_tds_z"],
            "is a real threat as a receiver out of the backfield",
            "is rarely used in the passing game",
            "becoming a real threat in the passing game"),
        SkillGroup(
            "short_yardage",
            ["short_yardage_conv_rate_z", "goal_line_td_rate_z",
             "rz_carry_share_z"],
            "is trusted in short-yardage and goal-line spots",
            "is not the guy in short-yardage and goal-line situations",
            "earning trust in short-yardage and goal-line spots"),
    ],
    "de": _PASS_RUSHER_GROUPS,
    "dt": _PASS_RUSHER_GROUPS,
    "lb": [
        SkillGroup("tackle_volume",
            ["tackles_per_game_z", "solo_tackle_rate_z",
             "tackles_per_snap_z"],
            "is a tackling machine in the middle of the defense",
            "doesn't pile up the tackles",
            "becoming a higher-volume tackler"),
        SkillGroup("backfield_disruption",
            ["tfl_per_game_z", "sacks_per_game_z",
             "qb_hits_per_game_z"],
            "lives in the offensive backfield wrecking plays",
            "rarely makes plays behind the line of scrimmage",
            "making more plays behind the line of scrimmage"),
        SkillGroup("coverage",
            ["completion_pct_allowed_z", "yards_per_target_allowed_z",
             "passer_rating_allowed_z"],
            "shuts down passes thrown his way in coverage",
            "is a liability in coverage",
            "tightening up his coverage"),
        SkillGroup("ball_disruption",
            ["forced_fumbles_per_game_z", "passes_defended_per_game_z",
             "interceptions_per_game_z"],
            "creates turnovers at a high rate",
            "rarely creates turnovers",
            "creating more turnovers"),
        SkillGroup("pass_rush",
            ["pressures_per_game_z", "hurries_per_game_z",
             "qb_knockdowns_per_game_z"],
            "is a real pass-rushing threat from the second level",
            "doesn't get home as a pass-rusher",
            "becoming a bigger pass-rushing threat"),
        SkillGroup("tackling",
            ["missed_tackle_pct_z"],
            "is one of the most reliable tacklers at his position",
            "misses too many tackles",
            "tightening up his tackling"),
    ],
    "cb": [
        SkillGroup("coverage",
            ["completion_pct_allowed_z", "yards_per_target_allowed_z",
             "passer_rating_allowed_z"],
            "is a true shutdown corner against the throws his way",
            "gives up too much in coverage",
            "tightening up his coverage"),
        SkillGroup("ball_hawk",
            ["interceptions_per_game_z", "passes_defended_per_game_z"],
            "is a ball-hawk who picks off passes at an elite rate",
            "rarely makes plays on the ball",
            "creating more plays on the ball"),
        SkillGroup("run_support",
            ["tackles_per_snap_z", "solo_tackle_rate_z",
             "tfl_per_game_z"],
            "is a willing and reliable tackler in run support",
            "is a liability in run support",
            "becoming a more willing run defender"),
        SkillGroup("tackling",
            ["missed_tackle_pct_z"],
            "rarely misses a tackle in space",
            "misses too many tackles in space",
            "tightening up his tackling in space"),
        SkillGroup("forced_pressure",
            ["forced_fumbles_per_game_z"],
            "punches the ball out at a high rate",
            "rarely creates fumbles",
            "creating more fumbles on contact"),
    ],
    "s": [
        SkillGroup("coverage",
            ["completion_pct_allowed_z", "yards_per_target_allowed_z",
             "passer_rating_allowed_z"],
            "shuts down the deep half of the field",
            "gives up too much in coverage",
            "tightening up his coverage"),
        SkillGroup("ball_hawk",
            ["interceptions_per_game_z", "passes_defended_per_game_z"],
            "is a ball-hawk safety who creates turnovers",
            "rarely makes plays on the ball",
            "creating more plays on the ball"),
        SkillGroup("run_support",
            ["tackles_per_snap_z", "solo_tackle_rate_z",
             "tfl_per_game_z"],
            "is a thumper in run support",
            "shies away from contact",
            "becoming a more reliable thumper"),
        SkillGroup("blitz",
            ["sacks_per_game_z"],
            "is a real factor as a blitzer",
            "doesn't impact the game as a blitzer",
            "becoming a real blitz threat"),
        SkillGroup("tackling",
            ["missed_tackle_pct_z"],
            "is a sure tackler from the back end",
            "misses too many tackles from the back end",
            "tightening up his tackling"),
    ],
}


# ── Scoring helpers ─────────────────────────────────────────────

def _group_z(row: pd.Series, group: SkillGroup) -> float | None:
    """Mean z across the group's stats for one player-season row."""
    vals = [row[c] for c in group.stats if c in row.index
            and pd.notna(row[c])]
    return sum(vals) / len(vals) if vals else None


def _rank_groups(row: pd.Series, position: str, mode: str
                  ) -> list[tuple[SkillGroup, float]]:
    if mode == "prospect":
        groups = PROSPECT_SKILLS.get(position.upper(), [])
    else:
        groups = POSITION_SKILLS.get(position.lower(), [])
    scored = []
    for g in groups:
        z = _group_z(row, g)
        if z is not None:
            scored.append((g, z))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


# ── Part A: strengths ───────────────────────────────────────────

_STRENGTH_THRESHOLD = 0.5  # avg z must beat this to count as "best"


_BALANCED_PROSPECT = [
    "{name} is a well-rounded prospect without a single elite separator.",
    "{name} grades out solid across the board without one elite trait.",
    "{name} is a balanced prospect with no glaring hole and no clear separator.",
    "{name} doesn't have one elite skill but doesn't have a glaring hole either.",
]
_BALANCED_NFL = [
    "{name} is a jack-of-all-trades without one truly elite trait.",
    "{name} grades out solid across the board but lacks a single elite skill.",
    "{name} is a steady all-arounder rather than an elite specialist.",
    "{name} produces in every area without standing out in any one.",
]


def _part_a(name: str, ranked: list[tuple[SkillGroup, float]],
             mode: str) -> str:
    strong = [g for g, z in ranked if z >= _STRENGTH_THRESHOLD]
    if not strong:
        pool = _BALANCED_PROSPECT if mode == "prospect" else _BALANCED_NFL
        return _pick(name, "balanced", pool).format(name=name)
    if len(strong) == 1:
        return f"{name} {strong[0].high}."
    return f"{name} {strong[0].high} and {strong[1].high}."


# ── Part C: drawbacks (NFL) / improvement areas (prospect) ──────

_WEAKNESS_THRESHOLD = -0.4
# Prospects: surface a development area even if no skill group is
# strongly negative, since no card should end without a "next-level"
# beat. Use lowest-ranked group when nothing falls below the strict
# threshold.
_PROSPECT_WEAK_FLOOR = 0.3


_NEXT_LEVEL_VARIANTS = [
    "He could take his game to the next level by {improve}.",
    "The next step in his game is {improve}.",
    "His ceiling rises with {improve}.",
    "Watch for his next jump to come from {improve}.",
    "The clearest path to a higher ceiling is {improve}.",
]


def _part_c(name: str, ranked: list[tuple[SkillGroup, float]],
             mode: str) -> str:
    """Both NFL and prospect blurbs always end with a 'next level'
    sentence. Pick the player's lowest-ranked skill group with a
    non-empty improve phrase."""
    for g, _ in reversed(ranked):
        if g.improve:
            return _pick(name, "next_level", _NEXT_LEVEL_VARIANTS
                          ).format(improve=g.improve)
    return ""


# ── Part B (prospect): recruiting/transfer narrative ────────────
# This is a "how is he matching expectations?" beat, framed against
# his recruiting profile or transfer history — not against the
# consensus board (that comparison felt too meta in v1).

_RECRUITING_PATH = DATA / "college" / "college_recruiting.parquet"

# Cached recruiting lookup keyed by normalized name + position.
_REC_INDEX: dict | None = None


def _norm_name(s) -> str:
    if not isinstance(s, str):
        return ""
    return (s.lower().replace(".", "").replace("'", "")
              .replace("-", " ").strip())


def _norm_school(s) -> str:
    if not isinstance(s, str):
        return ""
    return (s.lower().replace("state university", "state")
              .replace("university of ", "").replace("the ", "")
              .replace(" ", "").strip())


def _recruiting_index() -> dict:
    """Build (and cache) two indices in one pass:
       'by_name_pos' — keyed by (norm_name, position)
       'by_name'     — keyed by norm_name; value is the unambiguous
                        single record, or None if multiple recruits
                        share that name (collision risk)."""
    global _REC_INDEX
    if _REC_INDEX is not None:
        return _REC_INDEX
    if not _RECRUITING_PATH.exists():
        _REC_INDEX = {"by_name_pos": {}, "by_name": {}}
        return _REC_INDEX
    rec = pd.read_parquet(_RECRUITING_PATH)
    by_pos: dict = {}
    by_name: dict[str, list] = {}
    for _, r in rec.iterrows():
        nn = _norm_name(r.get("name"))
        pos = str(r.get("position") or "").upper()
        key = (nn, pos)
        prev = by_pos.get(key)
        if prev is None or float(r.get("rating") or 0) > float(
                prev.get("rating") or 0):
            by_pos[key] = dict(r)
        by_name.setdefault(nn, []).append(dict(r))
    # Collapse name index — only keep names with a single recruit
    # (multiple recruits sharing a name = ambiguous, skip).
    by_name_unique: dict = {}
    for nn, rows in by_name.items():
        if len(rows) == 1:
            by_name_unique[nn] = rows[0]
    _REC_INDEX = {"by_name_pos": by_pos, "by_name": by_name_unique}
    return _REC_INDEX


# CFBD recruiting positions don't always match our consensus
# positions. Map a few before lookup.
_REC_POS_ALIASES = {
    "DE": ["DE", "EDGE", "DL"],
    "DT": ["DT", "DL"],
    "OL": ["OL", "OT", "OG", "C", "IOL"],
    "S":  ["S", "SAF", "DB"],
    "CB": ["CB", "DB"],
    "LB": ["LB", "ILB", "OLB"],
    "WR": ["WR"], "RB": ["RB", "APB"],
    "QB": ["QB"], "TE": ["TE"],
}


def _lookup_recruiting(name: str, position: str) -> dict | None:
    idx = _recruiting_index()
    by_pos = idx.get("by_name_pos", {})
    by_name = idx.get("by_name", {})
    if not by_pos:
        return None
    nn = _norm_name(name)
    for pos in _REC_POS_ALIASES.get(position.upper(), [position.upper()]):
        hit = by_pos.get((nn, pos))
        if hit:
            return hit
    # Fallback: position changed since recruiting (e.g. ATH → TE).
    # Only trust unambiguous name match.
    return by_name.get(nn)


def _stars_phrase(stars: int) -> str:
    return {5: "five-star", 4: "four-star",
            3: "three-star", 2: "two-star"}.get(int(stars), "")


_TRANSFER_EXCEL = [
    "He has excelled since transferring from {school}.",
    "He has thrived since transferring from {school}.",
    "The transfer from {school} has paid off in a big way.",
    "He has been a revelation since transferring from {school}.",
]
_TRANSFER_SETTLED = [
    "He has settled in nicely after transferring from {school}.",
    "The transfer from {school} has gone smoothly.",
    "He has held his own since transferring from {school}.",
    "He has fit right in after transferring from {school}.",
]
_TRANSFER_FINDING = [
    "He is still finding his footing after transferring from {school}.",
    "The transfer from {school} has been bumpy.",
    "He has not yet found his stride since transferring from {school}.",
    "Production has lagged since transferring from {school}.",
]


def _transfer_sentence(name: str, prev_school: str,
                         perf_z: float) -> str:
    if perf_z >= 0.7:
        pool = _TRANSFER_EXCEL
    elif perf_z >= -0.3:
        pool = _TRANSFER_SETTLED
    else:
        pool = _TRANSFER_FINDING
    return _pick(name, "transfer", pool).format(school=prev_school)


_FIVE_STAR_LIVED = [
    "As a five-star recruit, he has lived up to the hype.",
    "The five-star billing has been justified.",
    "He has played like the five-star prospect he was billed as.",
    "His five-star recruiting profile looks accurate.",
]
_FIVE_STAR_FLASHES = [
    "He has shown flashes of the five-star ceiling that brought him in.",
    "Glimpses of the five-star ceiling have shown up.",
    "The five-star ceiling is there but not yet consistent.",
    "He has flashed the five-star ceiling in spots.",
]
_FIVE_STAR_NOT_YET = [
    "He is still working to live up to his five-star billing.",
    "His five-star billing remains ahead of his production.",
    "The five-star hype has not yet translated to the field.",
    "He has not yet looked the part of a five-star recruit.",
]
_FOUR_STAR_ELEVATED = [
    "He has elevated his game well beyond his four-star recruiting profile.",
    "He has produced like more than the four-star prospect he was billed as.",
    "He has outpaced his four-star recruiting profile in a big way.",
]
_FOUR_STAR_LIVED = [
    "He has lived up to his four-star billing.",
    "His four-star recruiting profile looks accurate.",
    "He has played like the four-star prospect he was.",
]
_FOUR_STAR_NOT_YET = [
    "He has not yet matched his four-star recruiting profile.",
    "His four-star billing remains ahead of his production.",
    "He has not yet looked the part of a four-star recruit.",
]
_THREE_STAR_WILD = [
    "He has wildly outperformed his three-star recruiting profile.",
    "He has been a massive recruiting steal at three stars.",
    "He has produced like a top-end recruit despite his three-star billing.",
]
_THREE_STAR_OUT = [
    "He has outperformed his three-star recruiting profile.",
    "He has produced above his three-star recruiting profile.",
    "His production has outpaced his three-star billing.",
]
_TWO_STAR_EMERGE = [
    "He has emerged as a real prospect despite a quiet recruiting profile.",
    "He has played his way onto the radar despite a low-key recruiting profile.",
    "He has carved out a real prospect profile despite minimal recruiting buzz.",
]


def _stars_sentence(name: str, stars: int, ranking,
                      perf_z: float) -> str:
    sp = _stars_phrase(stars)
    if not sp:
        return ""
    if stars >= 5:
        pool = (_FIVE_STAR_LIVED if perf_z >= 0.7
                else _FIVE_STAR_FLASHES if perf_z >= -0.3
                else _FIVE_STAR_NOT_YET)
        return _pick(name, "stars5", pool)
    if stars == 4:
        if perf_z >= 1.2:
            return _pick(name, "stars4_elev", _FOUR_STAR_ELEVATED)
        if perf_z >= 0.3:
            return _pick(name, "stars4_lived", _FOUR_STAR_LIVED)
        if perf_z <= -0.5:
            return _pick(name, "stars4_not", _FOUR_STAR_NOT_YET)
        return ""
    if stars == 3:
        if perf_z >= 1.2:
            return _pick(name, "stars3_wild", _THREE_STAR_WILD)
        if perf_z >= 0.3:
            return _pick(name, "stars3_out", _THREE_STAR_OUT)
        return ""
    if stars <= 2 and perf_z >= 0.7:
        return _pick(name, "stars2_emerge", _TWO_STAR_EMERGE)
    return ""


def _part_b_prospect(name: str, current_school: str, position: str,
                       composite_z: float | None) -> str:
    if composite_z is None or pd.isna(composite_z):
        return ""
    rec = _lookup_recruiting(name, position)
    if rec is None:
        # No recruiting data — likely walk-on / FCS. Skip.
        return ""
    committed_school = rec.get("school")
    if (committed_school and current_school
            and _norm_school(committed_school) !=
                _norm_school(current_school)):
        return _transfer_sentence(name, committed_school,
                                    float(composite_z))
    stars = rec.get("stars")
    if stars and not pd.isna(stars):
        return _stars_sentence(name, int(stars), rec.get("ranking"),
                                 float(composite_z))
    return ""


# ── Part B (NFL): fallback ladder ───────────────────────────────

def _part_b(player_id: str | None, player_name: str, position: str,
             season: int, score: float | None,
             cohort: pd.DataFrame) -> str:
    # Ladder: draft-slot value → FA contract value → situational pbp.
    # Draft-slot is the strongest fan hook ("third-round billing")
    # so we lead with it whenever the player was drafted.
    slot = _draft_slot_sentence(player_id, player_name, position,
                                  score, cohort)
    if slot:
        return slot
    fa = _fa_contract_sentence(player_name, position, score, cohort)
    if fa:
        return fa
    sit = _situational_sentence(player_name, position, season)
    if sit:
        return sit
    return ""


# ── Situational pbp sentence ────────────────────────────────────

# Cache the pbp slice per process — pbp is big.
_PBP_RB: pd.DataFrame | None = None


def _load_rb_pbp() -> pd.DataFrame:
    global _PBP_RB
    if _PBP_RB is None:
        if not PBP_PATH.exists():
            _PBP_RB = pd.DataFrame()
            return _PBP_RB
        cols = ["season", "rusher_player_name", "rush_attempt",
                "rushing_yards", "epa", "success", "shotgun",
                "offense_formation", "offense_personnel"]
        df = pd.read_parquet(PBP_PATH, columns=cols)
        _PBP_RB = df[df["rush_attempt"] == 1].copy()
    return _PBP_RB


def _pbp_name(full_name: str) -> str:
    """nflverse pbp uses 'F.Lastname' format."""
    parts = full_name.split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0]}.{' '.join(parts[1:])}"


def _situational_sentence(player_name: str, position: str,
                           season: int) -> str | None:
    if position.lower() != "rb":
        return None
    pbp = _load_rb_pbp()
    if pbp.empty:
        return None
    short = _pbp_name(player_name)
    rows = pbp[pbp["rusher_player_name"] == short]
    if len(rows) < 60:
        return None
    # Best formation by EPA (min 25 carries to be meaningful)
    f = rows.groupby("offense_formation").agg(
        carries=("rush_attempt", "sum"),
        ypc=("rushing_yards", "mean"),
        epa=("epa", "mean"),
        explosive=("rushing_yards", lambda s: (s >= 10).mean()),
    ).query("carries >= 25")
    if f.empty:
        return None
    # Pick the formation where he's clearly best by EPA
    best = f.sort_values("epa", ascending=False).iloc[0]
    best_formation = best.name.title() if isinstance(best.name, str) \
                                          else "Shotgun"
    # Note: pbp 'offense_formation' uppercases. Lowercase for prose.
    formation_label = {
        "Shotgun": "shotgun",
        "Under Center": "under center",
        "Pistol": "pistol",
        "I_Form": "I-formation",
    }.get(best_formation, best_formation.lower())

    # Optional: append personnel context if 11-personnel pops
    p = rows.copy()
    p["is_11"] = p["offense_personnel"].fillna("").str.contains(
        r"1 RB.*1 TE.*3 WR", regex=True, na=False)
    eleven = p[p["is_11"]]
    eleven_phrase = ""
    if len(eleven) >= 25:
        eleven_epa = eleven["epa"].mean()
        rest_epa = p[~p["is_11"]]["epa"].mean()
        if eleven_epa > rest_epa + 0.05:
            eleven_phrase = " out of 11 personnel"

    return _pick(player_name, "situational", _SITUATIONAL_VARIANTS
                   ).format(formation=formation_label,
                              personnel=eleven_phrase)


_SITUATIONAL_VARIANTS = [
    "He does his best work in {formation}{personnel} "
    "where defenses are forced to spread out.",
    "He thrives in {formation}{personnel} against spread-out defenses.",
    "He is at his most dangerous in {formation}{personnel}.",
    "{formation}{personnel} brings out the best in his game.",
]


# ── Draft slot value sentence ───────────────────────────────────

_ROUND_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth",
                  5: "fifth", 6: "sixth", 7: "seventh"}

_DRAFT_OVER = [
    "He has overperformed his {rd}-round billing.",
    "He has played well above his {rd}-round draft slot.",
    "He has been a steal as a {rd}-round pick.",
    "His {rd}-round selection looks like a heist in hindsight.",
]
_DRAFT_MET = [
    "He has lived up to his {rd}-round billing.",
    "He has been exactly what a {rd}-round pick should be.",
    "His {rd}-round draft slot looks accurate.",
    "He has produced like a typical {rd}-round pick.",
]
_DRAFT_UNDER = [
    "He has not yet lived up to his {rd}-round billing.",
    "He has underperformed his {rd}-round draft slot.",
    "His {rd}-round selection has not yet panned out.",
    "His production trails his {rd}-round draft slot.",
]
_DRAFT_UNDER_R1 = [
    "He has not yet lived up to his first-round selection.",
    "His first-round selection has not yet paid off.",
    "He still needs to grow into his first-round billing.",
    "First-round expectations remain ahead of his production.",
]


def _draft_slot_sentence(player_id: str | None, player_name: str,
                          position: str, score: float | None,
                          cohort: pd.DataFrame) -> str | None:
    if score is None or not DRAFT_PATH.exists():
        return None
    draft = pd.read_parquet(DRAFT_PATH)
    pos = position.upper()
    m = draft[(draft["pfr_player_name"].str.lower()
                == player_name.lower())
              & (draft["position"] == pos)]
    if m.empty:
        return None
    rd = int(m.iloc[0]["round"])
    rd_word = _ROUND_WORDS.get(rd)
    if not rd_word:
        return None
    # Compare player's score to the average score for his draft round
    # at his position. Threshold: ±0.4 σ.
    if "score" not in cohort.columns:
        return None
    pos_picks = draft[draft["position"] == pos]
    merged = cohort.merge(
        pos_picks[["pfr_player_name", "round"]],
        left_on="player_display_name",
        right_on="pfr_player_name", how="inner",
    )
    if merged.empty or "round" not in merged.columns:
        return None
    round_avg = merged.groupby("round")["score"].mean().to_dict()
    target = round_avg.get(rd)
    if target is None:
        return None
    diff = float(score) - float(target)
    if diff >= 0.4:
        return _pick(player_name, "draft_over", _DRAFT_OVER
                       ).format(rd=rd_word)
    if diff <= -0.4:
        pool = _DRAFT_UNDER_R1 if rd == 1 else _DRAFT_UNDER
        return _pick(player_name, "draft_under", pool
                       ).format(rd=rd_word)
    return _pick(player_name, "draft_met", _DRAFT_MET
                   ).format(rd=rd_word)


# ── FA contract value sentence ──────────────────────────────────

def _fa_contract_sentence(player_name: str, position: str,
                           score: float | None,
                           cohort: pd.DataFrame) -> str | None:
    if score is None or not CONTRACT_PATH.exists():
        return None
    contracts = pd.read_parquet(CONTRACT_PATH)
    pos = position.upper()
    name_l = player_name.lower()
    # Contract position values can be lower-case or different (e.g.
    # 'rb' vs 'RB'); be tolerant.
    pos_match = contracts["position"].astype(str).str.upper() == pos
    m = contracts[(contracts["player"].str.lower() == name_l)
                   & pos_match]
    if m.empty:
        return None
    # Active contract: latest year_signed
    m = m.sort_values("year_signed", ascending=False).iloc[0]
    apy_m = float(m.get("apy", 0) or 0)  # contracts parquet stores
    if apy_m <= 0:                          # APY in $millions already.
        return None
    pos_contracts = contracts[contracts["position"] == pos]
    median_apy = float(pos_contracts["apy"].median() or 0)
    if score is None:
        return ""
    if apy_m <= median_apy * 0.6 and score >= 0.3:
        return _pick(player_name, "contract_bargain", _CONTRACT_BARGAIN
                      ).format(apy=apy_m)
    if apy_m >= median_apy * 1.5 and score <= -0.2:
        return _pick(player_name, "contract_underwater", _CONTRACT_UNDERWATER
                      ).format(apy=apy_m)
    if apy_m >= median_apy * 1.5 and score >= 0.5:
        return _pick(player_name, "contract_earned", _CONTRACT_EARNED
                      ).format(apy=apy_m)
    return ""


_CONTRACT_BARGAIN = [
    "He has been an absolute bargain at ${apy:.1f}M per year.",
    "His ${apy:.1f}M-per-year deal looks like a steal.",
    "He has produced well above his ${apy:.1f}M-per-year contract.",
    "Few contracts in the league offer better value than his ${apy:.1f}M per year.",
]
_CONTRACT_UNDERWATER = [
    "He has not lived up to his ${apy:.1f}M-per-year contract.",
    "His ${apy:.1f}M-per-year deal has been an albatross.",
    "Production hasn't matched his ${apy:.1f}M-per-year price tag.",
    "He has been a tough watch on a ${apy:.1f}M-per-year contract.",
]
_CONTRACT_EARNED = [
    "He has earned every dollar of his ${apy:.1f}M-per-year contract.",
    "His ${apy:.1f}M-per-year deal has been money well spent.",
    "He has played up to his ${apy:.1f}M-per-year contract.",
    "Production has matched his ${apy:.1f}M-per-year price tag.",
]


# ── Public entrypoint ───────────────────────────────────────────

def make_card_narrative(player_row: pd.Series,
                          cohort: pd.DataFrame, position: str,
                          mode: str = "nfl") -> str | None:
    """Returns the blurb string for use as a trading-card narrative.
    Computes career-avg z internally and adds a per-row 'score' column
    on the cohort so the draft-slot sentence can find a round average.
    Returns None on any failure."""
    try:
        z_cols = [c for c in cohort.columns if c.endswith("_z")]
        score: float | None = None
        cohort_with_score = cohort
        if z_cols and "score" not in cohort.columns:
            cohort_with_score = cohort.copy()
            cohort_with_score["score"] = (
                cohort_with_score[z_cols].mean(axis=1))
        if ("player_id" in cohort.columns
                and "player_id" in player_row.index
                and z_cols):
            career = cohort[cohort["player_id"]
                              == player_row.get("player_id")]
            if len(career):
                score = float(career[z_cols].mean(axis=1).mean())
        elif z_cols:
            vals = [player_row[c] for c in z_cols
                     if c in player_row.index
                     and pd.notna(player_row[c])]
            score = sum(vals) / len(vals) if vals else None
        blurb = generate_blurb(player_row, cohort_with_score, position,
                                 score=score, mode=mode)
        return blurb or None
    except Exception:
        return None


def render_blurb(st_module, player_row: pd.Series,
                  cohort: pd.DataFrame, position: str,
                  mode: str = "nfl") -> None:
    """Streamlit page wrapper that renders the blurb as italic
    markdown. (Currently unused — trading-card narrative is the
    preferred surface.)"""
    blurb = make_card_narrative(player_row, cohort, position, mode)
    if blurb:
        st_module.markdown(f"_{blurb}_")


def generate_blurb(row: pd.Series, cohort: pd.DataFrame,
                    position: str, season: int | None = None,
                    score: float | None = None,
                    mode: str = "nfl") -> str:
    """Return a 2-3 sentence player blurb in Brett's plain-fan style.

    Args:
      row: a single player-season row from the cohort dataframe.
      cohort: the full position dataframe (used for FA / draft slot
              relativity).
      position: position string ('rb', 'wr', etc. for NFL; 'QB',
                'WR', etc. for prospects).
      season: optional season; defaults to row['season_year'].
      score: optional composite score; if not given we skip the
             draft-slot / FA fallbacks.
      mode: 'nfl' (uses POSITION_SKILLS + situational pbp / draft slot
            / contract fallbacks) or 'prospect' (uses PROSPECT_SKILLS,
            no Part B).
    """
    name = (row.get("player_display_name")
            or row.get("player") or "He")
    season = int(season or row.get("season_year") or 0)
    pid = row.get("player_id")

    ranked = _rank_groups(row, position, mode=mode)
    if not ranked:
        return ""

    a = _part_a(name, ranked, mode=mode)
    if mode == "prospect":
        b = _part_b_prospect(
            name, row.get("school", ""), position,
            row.get("composite_z"),
        )
    else:
        b = _part_b(pid, name, position, season, score, cohort)
    c = _part_c(name, ranked, mode=mode)

    parts = [a]
    if b:
        parts.append(b)
    if c:
        parts.append(c)
    return " ".join(parts)
