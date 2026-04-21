# Data Pipeline Review — What We Found

> Written for Brett to review. This is thorough on purpose.
>
> Tyler built a unified data pipeline so you can regenerate all your
> parquet files locally with one command. While testing it, we ran the
> pipeline for 2024 and compared the output against your existing data
> file, row by row, player by player, stat by stat.
>
> What follows is everything we found — the good, the bad, and the
> honest accounting of where the data diverges and why.

---

## Executive Summary

Your existing `league_wr_all_seasons.parquet` and your `wr_data_pull.py`
script were built by **two completely different processes**. The script in
your repo was never used to generate the live data. We know this because:

1. The raw stats don't match (e.g., Amon-Ra St. Brown: 1263 yards in
   existing vs 1400 in the script's output)
2. The existing data has a **bug** where `success_rate` is actually just a
   copy of `epa_per_target` — they're the same number for all 221 players
3. The population is different (234 WRs vs 96 WR+TE)
4. The column set is different (149 vs 42 columns)

The correlation between the two methods is high (r = 0.96+) for most stats
— **player rankings are similar**. But the numbers are meaningfully different,
and there are real bugs in the existing data that should be fixed.

---

## What We Found — The Evidence

### Finding 1: The existing data was built from `load_player_stats`, not PBP

Your existing parquet has 149 columns including defensive stats, kicking
stats, and passing stats — for wide receivers. These come from nflverse's
`load_player_stats()` function, which dumps everything. The `wr_data_pull.py`
script aggregates from play-by-play data and produces only receiving stats.

**How we know:** The existing parquet has columns like `def_sacks`,
`fg_made`, `passing_yards` — stats that have nothing to do with WRs.
These only appear when you dump the full `load_player_stats()` table.

### Finding 2: Mid-season trades break the top-N pipeline

The `wr_data_pull.py` approach groups snap counts by team. When a player is
traded mid-season, his snaps are split:

| Player | Team 1 | Team 2 | Total | In top 64? |
|---|---|---|---|---|
| **Davante Adams** | LV: 178 | NYJ: 614 | **792** | No — neither half makes it |
| **Amari Cooper** | CLE: 340 | BUF: 327 | **667** | No |
| **DeAndre Hopkins** | TEN: 267 | KC: 315 | **582** | No |
| **Mike Williams** | NYJ: 298 | PIT: 183 | **481** | No |

**16 traded WRs** in 2024 are affected. Davante Adams had 792 snaps and
1,063 receiving yards — a clear starter — but he's invisible in the
pipeline output because neither his LV nor NYJ snap totals alone crack
the top 64.

Your existing data doesn't have this problem because `load_player_stats()`
aggregates across teams automatically.

**Fix:** The pipeline needs to aggregate snaps across teams before selecting
the top-N. This is a code fix, not a design decision.

### Finding 3: The pipeline included postseason data

The initial pipeline run included playoff games, inflating stats:

| Player | Existing (REG only) | Pipeline (REG + POST) | Difference |
|---|---|---|---|
| Amon-Ra St. Brown targets | 141 | 151 | +10 (playoff targets) |
| Amon-Ra St. Brown yards | 1,263 | 1,400 | +137 |

Once filtered to regular season, the PBP numbers match the existing data
for basic counting stats (targets, receptions, yards).

**Fix:** Filter PBP to `season_type == 'REG'`. One-line change.

### Finding 4: PBP overcounts TDs on lateral plays

Amon-Ra St. Brown: 14 TDs per PBP, 12 per nflverse. The difference:

- **Week 3 vs ARI:** St. Brown caught a 1-yard pass, lateraled to Jahmyr
  Gibbs who ran 20 yards for the TD. PBP marks `pass_touchdown = 1`
  because the play resulted in a passing TD. But St. Brown didn't score.

- **Week 17 vs SF:** Same thing — St. Brown caught a 1-yard pass, lateraled
  to Jameson Williams for a 41-yard TD.

PBP's `pass_touchdown` flag marks the **play** as a passing TD, not the
**player** who scored. nflverse's `receiving_tds` correctly attributes
the TD to the actual scorer (Gibbs, Williams).

Only Amon-Ra was affected among the top-10 TD leaders in 2024, but this
will hit other players in other seasons.

**Fix:** Use `td_player_id == receiver_player_id` to attribute TDs, or use
nflverse pre-computed `receiving_tds` for counting stats.

### Finding 5: `success_rate` is a bug in the existing data

This is the big one. In your existing parquet:

```
Garrett Wilson:
  success_rate:     0.1488076632942177
  epa_per_target:   0.1488076632942177   ← identical
```

We checked all 221 players with non-null values: **`success_rate` equals
`epa_per_target` for every single player.** They are literally the same
number.

This means your app's "Success rate" z-score is actually EPA per target
in disguise. The WR page has a bundle called "Reliability" that weights
`success_rate_z` at 35% — but it's been showing EPA, not success rate.

**What success_rate should be:** The percentage of a receiver's targets
that produced a "successful" play (gained enough yards to keep the offense
on schedule). It's a binary yes/no per play, averaged across targets.
Typical values: 0.45-0.55. Garrett Wilson's real success rate from PBP
is **0.53**, not 0.15.

**How it happened:** The Colab notebook that generated the existing data
likely used nflverse's `load_player_stats()` which doesn't include a
`success_rate` column. Someone probably computed it incorrectly or
accidentally assigned EPA per target to the success_rate column.

**Impact:** Every WR z-score that uses success_rate in your app is
actually showing EPA. Since EPA and success_rate are correlated (both
measure "did this play help?"), it partially works — but it's not what
the labels say. The "Reliability" bundle is double-counting EPA.

### Finding 6: Z-score distributions are not proper z-scores

In the existing data:

| Z-score column | Mean | Std Dev | Expected |
|---|---|---|---|
| `rec_yards_z` | -0.60 | 1.06 | 0.0, 1.0 |
| `targets_z` | -0.65 | 1.10 | 0.0, 1.0 |
| `catch_rate_z` | -0.11 | 2.06 | 0.0, 1.0 |
| `yards_per_target_z` | -0.19 | 2.80 | 0.0, 1.0 |
| `epa_per_target_z` | -0.25 | 2.65 | 0.0, 1.0 |

A proper z-score has mean = 0 and std = 1 by definition. The existing
z-scores don't because:

1. Z-scores were computed against a **reference population of ~139 WRs**
   (those with 200+ snaps), but then applied to **all 234 WRs** including
   low-snap players.

2. The low-snap players have worse stats than the reference population
   (obviously), so they pull the mean negative.

3. Rate stats (catch_rate, yards_per_target) have extreme std because
   small-sample players have wild rates (a guy with 3 targets who caught
   all 3 has a 100% catch rate).

**What this means for the app:** A z-score of +1.0 doesn't mean "one
standard deviation above average." The scale varies by stat. The relative
rankings are still mostly correct, but the composite scoring math
(`score = sum(z × weight) / total_weight`) is mixing scales — a
yards_per_target_z of +1.0 means something very different from a
rec_yards_z of +1.0.

The new pipeline computes z-scores only within the reference population,
so they're properly scaled.

---

## Side-by-Side: Actual Player Data

### Detroit Lions WRs (2024)

| Player | Snaps | Yards | rec_yards_z (existing) | In new pipeline? |
|---|---|---|---|---|
| Amon-Ra St. Brown | 1,076 | 1,263 | +1.97 | Yes |
| Jameson Williams | 887 | 1,001 | +1.24 | Yes |
| Tim Patrick | 671 | 394 | -0.47 | Yes |
| Kalif Raymond | 240 | 215 | -0.97 | **No** (below top 64) |
| Allen Robinson | 91 | 30 | -1.49 | **No** |
| Tom Kennedy | 16 | 0 | -1.57 | **No** |
| Maurice Alexander | 0 | 0 | -1.57 | **No** |

With a **100-snap floor** instead of top-64, Raymond (240 snaps) would
be included. Robinson (91) would not.

### Team Coverage by Snap Threshold

| Threshold | Avg WRs/team | Min | Max |
|---|---|---|---|
| 0 snaps (all) | 7.2 | 5 | 11 |
| 100+ snaps | 5.3 | 4 | 7 |
| 200+ snaps | 4.3 | 3 | 7 |
| 400+ snaps | 3.4 | 1 | 7 |

---

## Decisions for Brett

### Decision 1: Who's in the output?

**The question:** How many WRs per team should appear in the data?

This is really two sub-questions:
- **Reference population** — who do we z-score against?
- **Output population** — who shows up on the page?

You can z-score against a tight reference group (starters only) but still
include depth players in the output with their z-scores. That's actually
what the existing data does — z-scores against ~139, output all ~234.

| Option | Reference pool | Output | Per-team coverage |
|---|---|---|---|
| **A) Current behavior** | 200+ snap WRs (~139) | All WRs with stats (~234) | 5-11 per team |
| **B) wr_data_pull.py** | Top 64 WR + top 32 TE | Same 96 players | 1-3 per team |
| **C) Snap floor (recommended)** | 100+ snap WRs (~169) | All WRs ≥ 100 snaps | 4-7 per team |

Option C gives you proper z-scores (computed against a clean population)
while keeping enough depth that every team's WR3/WR4 is visible. Players
with fewer than 100 snaps (3-4 games of playing time) are excluded —
their stats are too noisy to z-score meaningfully anyway.

### Decision 2: WR and TE — together or separate?

| Option | What it means |
|---|---|
| **Separate (current)** | WR z-scores compare WR to WR. TE compare TE to TE. A blocking TE with 200 receiving yards gets a neutral z-score against other TEs, but would look terrible against WRs. |
| **Combined (wr_data_pull.py)** | WR and TE z-scored together. Honestly reflects receiving production. Most TEs will have negative z-scores for receiving stats because they catch fewer passes. |
| **Separate reference, combined output** | Each position z-scored within its own group, but stored in the same file. Best of both worlds for the app. |

### Decision 3: What's the data source for counting stats?

Both methods pull from nflverse. The question is which layer.

| | Pre-aggregated (`load_player_stats`) | Play-by-play aggregation |
|---|---|---|
| **Accuracy** | Handles trades, laterals, edge cases correctly | Overcounts TDs on laterals, needs trade handling |
| **Custom stats** | Limited to what nflverse provides | Can compute anything (red zone, 3rd down, explosive, etc.) |
| **Speed** | Fast (~1 sec/season) | Slow (~30 sec/season, 250MB per season) |
| **Transparency** | Black box — nflverse does the math | Every calculation visible and auditable |

**My recommendation:** Use pre-aggregated stats as the **base** for counting
stats (targets, receptions, yards, TDs) — they handle edge cases correctly.
Add PBP-derived stats **on top** for the advanced/custom metrics that
nflverse doesn't pre-compute (success_rate, EPA per target, first-down
rate, red zone target share, etc.). This is the hybrid approach.

### Decision 4: What columns in the output?

| Option | Columns | File size |
|---|---|---|
| **Current (149 cols)** | Everything including defensive/kicking stats for WRs | ~3MB |
| **Lean (42 cols)** | Only what WR page uses today | ~0.5MB |
| **Position-relevant (~70 cols)** | All receiving + derived + z-scores, drop irrelevant | ~1MB |

The lean option means adding a new stat to the page also requires adding
it to the pipeline config. The 149-column option carries a lot of dead
weight. I'd lean toward ~70 columns — everything receiving-related.

---

## Bugs to Fix Regardless of Decisions

These aren't design choices — they're errors that should be corrected:

1. **`success_rate` = `epa_per_target`**: This affects every WR z-score
   in the app. The "Reliability" bundle is double-counting EPA.

2. **Traded players missing from top-N**: Davante Adams (792 snaps,
   1,063 yards) is invisible. Fix: aggregate snaps across teams before
   selecting population.

3. **Postseason data leaking into regular-season stats**: Fix: filter
   PBP to `season_type == 'REG'`.

4. **Lateral TD overcounting**: Fix: check `td_player_id` instead of
   `pass_touchdown` flag, or use nflverse pre-computed TDs.

---

## What's Already Built

The pipeline framework is done and tested (18 unit tests, all passing).
Once the decisions above are made, implementing them is straightforward
config changes:

```
tools/
  data_pull.py                    # CLI: --position, --seasons, --dry-run
  pipeline/
    base.py                       # PositionConfig dataclass
    sources.py                    # Cached nflverse data pulls
    population.py                 # Snap filtering, top-N selection
    zscore.py                     # Shared z-score engine (tested)
    output.py                     # Validation + parquet/metadata writer
    runner.py                     # Season-loop orchestrator
    positions/
      wr.py                       # WR+TE config (working, uses Method B)
      rb.py                       # RB config (stubbed)
      _template.py                # Template for adding new positions
```

To run it: `python tools/data_pull.py --position wr --seasons 2024`

## Next Steps

1. **Brett reviews this doc** and makes the four decisions
2. **Fix the bugs** (success_rate, trades, postseason, laterals)
3. **Update the pipeline** to match Brett's choices
4. **Run for 2016-2025**, verify the WR page works with new data
5. **Scale out** to remaining positions (QB, OL, DE, DT, LB, CB, S, K, P)
6. **Add position configs** — Brett's Claude uses `_template.py` to add
   each position one at a time
