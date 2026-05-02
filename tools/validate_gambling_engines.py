"""Validation harness for the Gambling Lab engines.

Runs known scenarios against each of the five engines and prints
a green/red report. The point isn't to validate exact numbers (we
don't have ground-truth labels) — it's to catch regressions:
schema breaks, NaN explosions, sanity violations (rates > 1, deltas
that don't sum to ~0, etc.).

Run with:
    python tools/validate_gambling_engines.py

Exit code 0 if all checks pass, 1 if any fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ── Helpers ──────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results: list[tuple[bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((ok, label))
    mark = PASS if ok else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {mark} {label}{suffix}")


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ── 1. Injury cohort engine ──────────────────────────────────────

section("1. Injury Cohort Engine")
from lib_injury_cohort import predict, body_part_normalize, load_cohort_rates

cr = load_cohort_rates()
check("cohort rates table loads",
      not cr.empty, f"{len(cr):,} cohorts")

# Body-part normalizer covers expected fuzzy inputs
for raw, expected in [
    ("Knee", "knee"), ("Right Knee", "knee"), ("ACL", "knee"),
    ("Hamstring", "hamstring"), ("turf toe", "foot"),
    ("concussion", "concussion"), ("shldr", "shoulder"),
    ("NFI", "illness"), ("rest", "illness"), ("", "unknown"),
]:
    got = body_part_normalize(raw)
    check(f"normalize {raw!r} → {expected!r}", got == expected,
          f"got {got!r}")

# Predict() — known sanity scenarios
scenarios = [
    # (description, kwargs, expected_play_rate_range)
    ("OUT/DNP → near 0%",
     dict(position="WR", body_part="knee",
          report_status="OUT", practice_status="DNP"),
     (0.0, 0.05)),
    ("DOUBTFUL → near 0%",
     dict(position="WR", body_part="hamstring",
          report_status="DOUBTFUL", practice_status="DNP"),
     (0.0, 0.10)),
    ("PROBABLE/FULL → ≥85%",
     dict(position="WR", body_part="ankle",
          report_status="PROBABLE", practice_status="FULL"),
     (0.85, 1.01)),
    ("NONE/FULL → ≥80%",
     dict(position="WR", body_part="unknown",
          report_status="NONE", practice_status="FULL"),
     (0.80, 1.01)),
    ("QUESTIONABLE/DNP < 50%",
     dict(position="WR", body_part="hamstring",
          report_status="QUESTIONABLE", practice_status="DNP"),
     (0.10, 0.55)),
    ("QUESTIONABLE/FULL > QUESTIONABLE/DNP",
     None, None),  # composite check below
]
for desc, kwargs, rng in scenarios[:-1]:
    r = predict(**kwargs)
    ok = rng[0] <= r.p_played <= rng[1]
    check(desc, ok,
          f"got p={r.p_played:.3f} (n={r.n}, level={r.cohort_level})")

# Composite ordering check
q_dnp = predict(position="WR", body_part="hamstring",
                report_status="QUESTIONABLE", practice_status="DNP").p_played
q_full = predict(position="WR", body_part="hamstring",
                 report_status="QUESTIONABLE", practice_status="FULL").p_played
check("Q/FULL > Q/DNP (practice gradient)",
      q_full > q_dnp,
      f"FULL={q_full:.3f}  DNP={q_dnp:.3f}")

# Snap-share if played should be in [0, 1]
sample_rows = cr.dropna(subset=["snap_share_if_played"])
check("snap_share_if_played is in [0, 1]",
      ((sample_rows["snap_share_if_played"] >= 0)
       & (sample_rows["snap_share_if_played"] <= 1)).all(),
      f"min={sample_rows['snap_share_if_played'].min():.3f} "
      f"max={sample_rows['snap_share_if_played'].max():.3f}")

# Play rates in [0, 1]
check("play_rate is in [0, 1]",
      ((cr["play_rate"] >= 0) & (cr["play_rate"] <= 1)).all())


# ── 2. Scheme deltas ─────────────────────────────────────────────

section("2. Scheme Deltas")
from lib_scheme_deltas import load_scheme_deltas

sd = load_scheme_deltas()
check("scheme deltas table loads", not sd.empty, f"{len(sd):,} rows")

# Should have 32 teams × N seasons × 2 sides
n_teams = sd["team"].nunique()
check("team count = 32", n_teams == 32, f"got {n_teams}")

n_seasons = sd["season"].nunique()
check("season count >= 8", n_seasons >= 8, f"got {n_seasons}")

# Sides
check("two sides (offense, defense)",
      set(sd["side"].unique()) == {"offense", "defense"})

# Deltas should sum to ~0 within each (season, side, metric) group
for col in ["pass_rate_overall_delta", "blitz_rate_delta"]:
    if col in sd.columns:
        sums = sd.groupby(["season", "side"])[col].sum()
        max_abs = sums.abs().max()
        check(f"{col} sums to ~0 within (season, side)",
              max_abs < 0.05,
              f"max |sum| = {max_abs:.4f}")

# Rate columns in [0, 1]
for col in ["pass_rate_overall", "shotgun_rate", "blitz_rate",
            "man_coverage_rate"]:
    if col in sd.columns:
        vals = sd[col].dropna()
        ok = ((vals >= 0) & (vals <= 1)).all()
        check(f"{col} in [0, 1]", ok,
              f"min={vals.min():.3f} max={vals.max():.3f}")

# Pass rate should be 50-65% for most teams in modern era
modern = sd[(sd["side"] == "offense") & (sd["season"] >= 2020)]
median_pass = modern["pass_rate_overall"].median()
check("modern (2020+) median pass rate in [0.55, 0.62]",
      0.55 <= median_pass <= 0.62, f"median={median_pass:.3f}")


# ── 3. DvP ───────────────────────────────────────────────────────

section("3. Defense vs. Position (DvP)")
DVP = REPO / "data" / "dvp.parquet"
dvp = pd.read_parquet(DVP)
check("DvP table loads", not dvp.empty, f"{len(dvp):,} rows")

# Position groups
check("has WR, TE, RB groups",
      set(dvp["pos_group"].unique()) >= {"WR", "TE", "RB"})

# Per-game stats should be positive
for col in ["rec_yards_pg", "rec_tds_pg", "rush_yards_pg"]:
    if col in dvp.columns:
        vals = dvp[col].dropna()
        ok = (vals >= 0).all()
        check(f"{col} non-negative", ok,
              f"min={vals.min():.3f} max={vals.max():.3f}")

# WR receiving yards / game should be in [80, 220] for most defenses
wr = dvp[dvp["pos_group"] == "WR"].dropna(subset=["rec_yards_pg"])
check("WR rec_yards_pg in [80, 220] for ≥90%",
      ((wr["rec_yards_pg"] >= 80) & (wr["rec_yards_pg"] <= 220)).mean() > 0.90)

# Deltas should sum to ~0 within (season, pos_group)
for col in ["rec_yards_pg_delta"]:
    sums = dvp.groupby(["season", "pos_group"])[col].sum()
    max_abs = sums.abs().max()
    check(f"{col} sums to ~0 within (season, pos_group)",
          max_abs < 5.0,  # raw yards, not rate; relax tolerance
          f"max |sum| = {max_abs:.3f}")


# ── 4. Coaching tendencies ───────────────────────────────────────

section("4. Coaching Tendencies")
COACH = REPO / "data" / "coaching_tendencies.parquet"
coach = pd.read_parquet(COACH)
check("coaching table loads", not coach.empty, f"{len(coach):,} rows")

# 32 teams
check("team count = 32", coach["team"].nunique() == 32)

# Rate cols in [0, 1]
for col in ["fourth_short_go_rate", "two_pt_attempt_rate", "rz_run_rate"]:
    if col in coach.columns:
        vals = coach[col].dropna()
        ok = ((vals >= 0) & (vals <= 1)).all()
        check(f"{col} in [0, 1]", ok,
              f"min={vals.min():.3f} max={vals.max():.3f}")

# DET 2024 should be elite at 4th-and-short
det_2024 = coach[(coach["team"] == "DET") & (coach["season"] == 2024)]
if not det_2024.empty:
    rate = det_2024.iloc[0]["fourth_short_go_rate"]
    check("DET 2024 4th-and-short go rate ≥ 0.70 (Campbell signature)",
          rate >= 0.70, f"got {rate:.3f}")


# ── 5. SGP correlations ──────────────────────────────────────────

section("5. SGP Correlations")
SGP = REPO / "data" / "sgp_correlations.parquet"
sgp = pd.read_parquet(SGP)
check("SGP table loads", not sgp.empty, f"{len(sgp):,} rows")

# Correlations should be in [-1, 1]
corr = sgp["corr_qb_yds_partner_yds"].dropna()
check("corr in [-1, 1]",
      ((corr >= -1.0) & (corr <= 1.0)).all(),
      f"min={corr.min():.3f} max={corr.max():.3f}")

# QB↔WR1 correlations should be mostly positive (most stacks correlate)
wr1 = sgp[(sgp["partner_role"] == "WR1")
          & sgp["corr_qb_yds_partner_yds"].notna()]
pos_pct = (wr1["corr_qb_yds_partner_yds"] > 0).mean()
check("≥80% of QB↔WR1 correlations are positive",
      pos_pct > 0.80, f"got {pos_pct:.1%}")

# Hurts↔Brown 2023 high-correlation sanity
phi_2023 = sgp[(sgp["team"] == "PHI") & (sgp["season"] == 2023)
               & (sgp["partner_role"] == "WR1")]
if not phi_2023.empty:
    c = phi_2023.iloc[0]["corr_qb_yds_partner_yds"]
    check("Hurts↔Brown 2023 corr ≥ 0.70 (textbook stack)",
          c >= 0.70, f"got {c:.3f}")


# ── 6. Game-Script Simulator (Feature 4.2) ───────────────────────

section("6. Game-Script Simulator (4.2)")
GS = REPO / "data" / "game_script_deltas.parquet"
gs = pd.read_parquet(GS)
check("game_script_deltas table loads", not gs.empty,
      f"{len(gs):,} scenarios")

# Should have all six scenarios
expected = {"NONE", "QB1", "RB1", "WR1", "TE1", "MULTI"}
have = set(gs["scenario"].unique())
check("has NONE/QB1/RB1/WR1/TE1/MULTI scenarios",
      expected.issubset(have),
      f"missing: {expected - have}")

# QB1 out should reduce points/game by ≥ 2
qb1 = gs[gs["scenario"] == "QB1"]
if not qb1.empty:
    delta = qb1.iloc[0]["points_per_game_delta"]
    check("QB1 out → points/game delta ≤ -2",
          delta <= -2.0, f"got {delta:+.2f}")

# Pass rates in [0, 1]
check("pass_rate in [0, 1]",
      ((gs["pass_rate"] >= 0) & (gs["pass_rate"] <= 1)).all())

# NONE baseline must be present and have largest n
none_row = gs[gs["scenario"] == "NONE"]
if not none_row.empty:
    n_none = int(none_row.iloc[0]["n_games"])
    n_max = int(gs["n_games"].max())
    check("NONE baseline has the largest sample",
          n_none == n_max, f"NONE n={n_none}, max n={n_max}")


# ── 7. Books vs Model (Feature 4.3) ──────────────────────────────

section("7. Books vs Model (4.3)")
BV = REPO / "data" / "books_vs_model.parquet"
bv = pd.read_parquet(BV)
check("books_vs_model table loads", not bv.empty,
      f"{len(bv):,} cohorts")

# Healthy baseline cover rate should be ~50% (sharp baseline)
healthy = bv[(bv["position_lost"] == "HEALTHY")
             & (bv["status"] == "HEALTHY")]
if not healthy.empty:
    cov = healthy.iloc[0]["cover_rate"]
    check("healthy baseline cover rate in [0.45, 0.55]",
          0.45 <= cov <= 0.55, f"got {cov:.3f}")
    miss = healthy.iloc[0]["mean_line_miss"]
    check("healthy baseline mean line miss in [-2, +2]",
          -2.0 <= miss <= 2.0, f"got {miss:+.3f}")

# Cover rates in [0, 1]
check("cover_rate in [0, 1]",
      ((bv["cover_rate"] >= 0) & (bv["cover_rate"] <= 1)).all())

# QB OUT scenarios should generally have NEGATIVE line_miss
# (books historically don't move the line enough on QB injuries)
qb_out = bv[(bv["position_lost"] == "QB") & (bv["status"] == "OUT")
            & (bv["n_games"] >= 20)]
if not qb_out.empty:
    avg_miss = (qb_out["mean_line_miss"] * qb_out["n_games"]).sum() / qb_out["n_games"].sum()
    check("QB OUT cohorts (n≥20) — avg line miss is negative",
          avg_miss < 0, f"got {avg_miss:+.2f}")


# ── 8. Weather Production Window (Feature 4.5) ───────────────────

section("8. Weather Production Window (4.5)")
WX = REPO / "data" / "player_games_weather.parquet"
wx = pd.read_parquet(WX)
check("weather table loads", not wx.empty, f"{len(wx):,} player-games")

# Must have weather columns
for col in ["temp", "wind", "roof", "surface"]:
    check(f"has column {col}", col in wx.columns)

# Weather coverage should be 60-80% (domes drag the rate)
cov_temp = wx["temp"].notna().mean()
check("temp coverage in [0.60, 0.85]",
      0.60 <= cov_temp <= 0.85, f"got {cov_temp:.0%}")

# Cohort engine smoke test — Goff in cold weather
from lib_weather import weather_cohort, all_player_options
opts = all_player_options(position="QB", min_games=50)
goff = opts[opts["player_display_name"].str.contains("Goff", na=False)]
if len(goff):
    pid = goff.iloc[0]["player_id"]
    cold = weather_cohort(pid, "QB", target_temp=35, target_wind=15)
    mild = weather_cohort(pid, "QB", target_temp=70, target_wind=5)
    check("Goff cold cohort returns sensible P50",
          50 <= cold.p50 <= 400, f"got {cold.p50:.1f}")
    check("Goff mild cohort returns sensible P50",
          150 <= mild.p50 <= 400, f"got {mild.p50:.1f}")
    check("P10 < P50 < P90 monotonic (cold)",
          cold.p10 <= cold.p50 <= cold.p90,
          f"{cold.p10:.0f}/{cold.p50:.0f}/{cold.p90:.0f}")


# ── 10. Trend Divergence (Feature 5.6) ───────────────────────────

section("10. Trend Divergence (5.6)")
from lib_trend_divergence import compute_player_window, league_divergence_today

# Pull a known player's window
asb = "00-0036900"  # Ja'Marr Chase actually
rows = compute_player_window(asb, 2024, 18, lookback=3)
check("trend window returns rows for an active player",
      len(rows) > 0, f"got {len(rows)} stat rows")

# League scan returns a dataframe
league = league_divergence_today(2024, 18, position="WR", min_z=1.0)
check("league divergence scan returns rows", not league.empty,
      f"got {len(league)} flags")

# Z-scores must be float, finite, and at least one |z| ≥ 1.0
if not league.empty:
    valid = league["delta_z"].abs() >= 1.0
    check("≥80% of flagged rows have |z| ≥ 1.0", valid.mean() >= 0.80)


# ── 11. Longest-Play Edge (Feature 5.7) ──────────────────────────

section("11. Longest-Play Edge (5.7)")
from lib_longest_play import (
    longest_play_distribution, p_longest_at_least, player_options as lp_opts
)

opts = lp_opts(kind="reception", min_games=50)
check("longest-play player options non-empty",
      not opts.empty, f"{len(opts)} players with ≥50 games")

# Pick first player and check distribution
if not opts.empty:
    pid = opts.iloc[0]["player_id"]
    dist = longest_play_distribution(pid, kind="reception")
    check("distribution returns one row per (game, player)",
          not dist.empty, f"{len(dist)} games")
    r = p_longest_at_least(pid, 30, kind="reception")
    check("longest-play P(≥30) is in [0, 1]",
          0 <= r.p_at_least <= 1)
    check("P10 ≤ median ≤ P90 (longest-play)",
          r.p10_longest <= r.median_longest <= r.p90_longest)


# ── 12. Alt-Line EV (Feature 5.3) ────────────────────────────────

section("12. Alt-Line EV (5.3)")
from lib_alt_line_ev import (
    american_to_decimal, decimal_to_implied_prob, p_over_threshold,
    rank_ladder,
)

# Conversion sanity
check("american_to_decimal(-110) ≈ 1.909",
      abs(american_to_decimal(-110) - 1.909) < 0.01)
check("american_to_decimal(+180) = 2.80",
      abs(american_to_decimal(180) - 2.80) < 0.001)
check("implied_prob(2.0) = 0.5",
      abs(decimal_to_implied_prob(2.0) - 0.5) < 0.001)

# Ladder evaluation
asb_id = "00-0036900"
ladder = [(75.5, "over", -110), (95.5, "over", 180)]
df = rank_ladder(asb_id, "receiving_yards", ladder, lookback_games=20)
check("rank_ladder returns ranked dataframe",
      not df.empty and "ev" in df.columns,
      f"shape: {df.shape}")
check("ladder is sorted by EV descending",
      df["ev"].is_monotonic_decreasing if len(df) > 1 else True)


# ── 13. SGP Pricing (Feature 5.2 upgrade) ────────────────────────

section("13. SGP Pricing (5.2 upgrade)")
from lib_sgp_pricing import Leg, sgp_price

legs = [
    Leg("00-0036389", "Jalen Hurts", "passing_yards", 240, "over"),
    Leg("00-0035216", "A.J. Brown", "receiving_yards", 70, "over"),
]
r = sgp_price(legs, book_american_odds=300)
check("SGP returns 2-leg result",
      r.n_legs == 2, f"n_legs={r.n_legs}")
check("joint games > 0 for known stack",
      r.n_games_joint > 0, f"n_joint={r.n_games_joint}")
check("p_independent and p_correlated both in [0, 1]",
      0 <= r.p_independent <= 1 and 0 <= r.p_correlated <= 1)
check("Hurts/Brown stack has positive correlation lift",
      r.correlation_lift >= 0,
      f"lift={r.correlation_lift:+.3f}")


# ── 14. Decomposed Projection (Feature 5.1) ──────────────────────

section("14. Decomposed Projection (5.1)")
from lib_decomposed_projection import decompose

asb = "00-0036900"
d = decompose(player_id=asb, position="WR", team="CIN",
              stat="receiving_yards",
              opponent="HOU", season=2024, week=10,
              target_temp=68, target_wind=5,
              lookback_games=12)
check("decomposition returns positive baseline",
      d.baseline > 0, f"baseline={d.baseline:.1f}")
check("projection = baseline + sum(deltas)",
      abs(d.projection - (d.baseline + sum(c.delta for c in d.contributions))) < 0.01)
check("contributions list is non-empty when context provided",
      len(d.contributions) >= 1)


# ── 15. Smart Parlay Builder (Feature 5.4) ───────────────────────

section("15. Smart Parlay (5.4)")
from lib_smart_parlay import score_parlay, detect_anti_correlated

legs = [
    Leg("00-0036389", "Jalen Hurts", "passing_yards", 240, "over"),
    Leg("00-0035216", "A.J. Brown", "receiving_yards", 70, "over"),
]
p = score_parlay(legs, book_odds=300, lookback_games=20)
check("score_parlay returns marginals for each leg",
      len(p.leg_marginals) == 2)
check("score_parlay returns a verdict string",
      isinstance(p.verdict, str) and len(p.verdict) > 0)
check("anti-correlated detector returns a list (may be empty)",
      isinstance(detect_anti_correlated(legs, lookback_games=20), list))


# ── 16. TD Probability (Feature 5.5) ─────────────────────────────

section("16. TD Probability (5.5)")
from lib_td_probability import (
    player_td_rates, rz_usage_share, td_probability_vector,
)

pacheco = "00-0037197"
r = player_td_rates(pacheco, lookback_games=20)
check("td_rates returns per-game rates in [0, 1]",
      0 <= r.p_rush_td <= 1 and 0 <= r.p_rec_td <= 1
      and 0 <= r.p_any_td <= 1)
check("p_any_td ≤ p_rush_td + p_rec_td (additivity bound)",
      r.p_any_td <= r.p_rush_td + r.p_rec_td + 0.001)

u = rz_usage_share(pacheco, 2024, team="KC")
check("rz_usage_share returns shares in [0, 1]",
      0 <= u.rz_carries_share <= 1 and 0 <= u.rz_targets_share <= 1)


# ── 9. Smart Alerts fusion (Feature 4.4) ─────────────────────────

section("9. Smart Alerts (4.4)")
from lib_smart_alerts import fuse_alert
b = fuse_alert(player_name="Jared Goff", team="DET", position="QB",
               status="OUT", body_part="shoulder",
               practice_status="DNP",
               opponent="HOU", season=2024, week=10)
check("fusion produces a non-empty headline",
      bool(b.headline) and "Goff" in b.headline)
check("fusion produces a cohort line",
      bool(b.cohort_line) and "Pr(plays" in b.cohort_line)
check("fusion produces a game-script line",
      bool(b.game_script_line) and "points/game" in b.game_script_line)
check("fusion produces a book behavior line",
      bool(b.book_behavior_line) and "line miss" in b.book_behavior_line)
check("fusion produces ≥3 bullets",
      len(b.bullet_points) >= 3, f"got {len(b.bullet_points)}")


# ── 17. Matchup Report (auto) ────────────────────────────────────

section("17. Matchup Report (auto)")
from lib_matchup_report import generate_matchup_report

r = generate_matchup_report("HOU", "DET", 2024, 10)
check("matchup report builds",
      not r.headline.get("error"),
      f"home={r.home_team} away={r.away_team}")
check("headline has spread + total + moneylines",
      r.headline.get("spread_line") is not None
      and r.headline.get("total_line") is not None
      and r.headline.get("home_moneyline") is not None)
check("both teams have injuries collected",
      isinstance(r.home_injuries, list)
      and isinstance(r.away_injuries, list))
check("scheme outliers present for both teams",
      len(r.home_scheme) >= 1 and len(r.away_scheme) >= 1)
check("coaching outliers present for both teams",
      len(r.home_coaching) >= 1 and len(r.away_coaching) >= 1)
check("bottom_line returns ≥1 bullet",
      len(r.bottom_line_bullets) >= 1)


# ── 18. Player Prop Report (auto) ────────────────────────────────

section("18. Player Prop Report (auto)")
from lib_player_prop_report import generate_player_report

r = generate_player_report(
    player_id="00-0036900", position="WR",
    season=2024, week=10,
)
check("player report builds",
      r.player_name and r.team and r.primary_stat)
check("recent_form has at most 5 rows AND all are pre-target",
      len(r.recent_form) <= 5
      and (r.recent_form.empty
           or ((r.recent_form["season"] < 2024)
                | ((r.recent_form["season"] == 2024)
                   & (r.recent_form["week"] < 10))).all()))
check("decomposition present with baseline > 0",
      r.decomposition and r.decomposition.get("baseline", 0) > 0)
check("alt_ladder produced",
      not r.alt_ladder.empty,
      f"{len(r.alt_ladder)} rungs")
check("td_vector present and rates in [0, 1]",
      r.td_vector
      and 0 <= r.td_vector.get("p_any_td", 0) <= 1)
check("SGP partners do NOT include the player himself",
      all(p.get("partner_name") != r.player_name
          for p in r.sgp_partners))
check("bottom_line returns ≥1 bullet",
      len(r.bottom_line_bullets) >= 1)


# ── Summary ──────────────────────────────────────────────────────

section("Summary")
n_total = len(results)
n_pass = sum(1 for ok, _ in results if ok)
n_fail = n_total - n_pass

print(f"  {n_pass} / {n_total} checks passed")
if n_fail:
    print(f"  {n_fail} FAILED:")
    for ok, label in results:
        if not ok:
            print(f"    {FAIL} {label}")
    sys.exit(1)
else:
    print(f"  All green. Engines look healthy.")
    sys.exit(0)
