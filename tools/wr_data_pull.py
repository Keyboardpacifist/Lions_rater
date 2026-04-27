# ─────────────────────────────────────────────────────────────────────────────
# Lions Rater — Receivers (WR + TE) Data Pull (2024)
# ─────────────────────────────────────────────────────────────────────────────
# Builds master_lions_with_z.parquet and wr_stat_metadata.json.
#
# Key architectural decisions (flagged for future maintainers):
#
# 1. POPULATION: top 64 WRs + top 32 TEs by offensive snaps, min 6 games.
#    All 96 players are z-scored together as one combined "pass catcher" pool.
#    This means z-scores are league-wide (across 32 teams), not team-scoped.
#
# 2. FILTER METRIC IS SNAPS, NOT PRODUCTION. We rank players for inclusion by
#    offensive snaps so the filter doesn't pre-select on the same metrics we
#    rate with. Filtering by targets or yards would bias the z-score
#    distribution by construction.
#
# 3. NO MINIMUM TARGETS. The snaps + games floor is the only filter. Some
#    low-target blocking TEs will be in the pool with near-zero target-based
#    stats and deeply negative z-scores for efficiency. That's honest — they
#    were on the field, they weren't producing through the passing game, the
#    stat reflects it.
#
# 4. TIER 1 RAW-COUNT Z-SCORES are added here (z_rec_yards, z_receptions,
#    z_rec_tds, z_targets). These are what users check when they select
#    "Tier 1 — Counted" in the app. Pure volume, honestly measured.
#
# Ethos: strive for the greatest accuracy while being transparent about
# limitations and how we addressed them.
# ─────────────────────────────────────────────────────────────────────────────

!pip install nflreadpy pandas numpy pyarrow --quiet

import nflreadpy as nfl
import pandas as pd
import numpy as np
from pathlib import Path
import json

SEASON = 2024
MIN_GAMES = 6
TOP_N_WR = 64
TOP_N_TE = 32

OUTPUT_PARQUET = Path('master_lions_with_z.parquet')
OUTPUT_METADATA = Path('wr_stat_metadata.json')


# ─── Pull raw data ───────────────────────────────────────────────────────────
print('Pulling play-by-play...')
pbp = nfl.load_pbp([SEASON]).to_pandas()
print(f'  {len(pbp):,} plays loaded')

print('Pulling snap counts...')
snaps = nfl.load_snap_counts([SEASON]).to_pandas()
print(f'  {len(snaps):,} player-game rows loaded')

print('Pulling NGS receiving stats (for separation, cushion, YAC over expected, CPOE)...')
try:
    ngs = nfl.load_nextgen_stats([SEASON], stat_type='receiving').to_pandas()
    ngs_season = ngs[ngs['week'] == 0].copy() if 'week' in ngs.columns else ngs.copy()
    print(f'  {len(ngs_season):,} NGS rows (season aggregates)')
except Exception as e:
    print(f'  NGS pull failed ({e}) — will proceed without NGS-derived stats')
    ngs_season = pd.DataFrame()

print('Pulling rosters...')
rosters = nfl.load_rosters([SEASON]).to_pandas()
print(f'  {len(rosters):,} player rows loaded')


# ─── Restrict PBP to pass plays with a receiver ──────────────────────────────
pass_plays = pbp[
    (pbp['play_type'] == 'pass') &
    (pbp['receiver_player_id'].notna())
].copy()
print(f'\nPass plays with a receiver: {len(pass_plays):,}')

# FO/PFR success rate per play — replaces nflverse's EPA-based
# `success` column so our success_rate aligns with PFF / PFR.
# 1st: ≥40% of yards-to-go · 2nd: ≥60% · 3rd/4th: full conversion.
def _fo_success_play(row):
    d, ytg, yg = row.get('down'), row.get('ydstogo'), row.get('yards_gained')
    if pd.isna(d) or pd.isna(ytg) or pd.isna(yg):
        return np.nan
    if d == 1: return 1.0 if yg >= 0.4 * ytg else 0.0
    if d == 2: return 1.0 if yg >= 0.6 * ytg else 0.0
    return 1.0 if yg >= ytg else 0.0
pass_plays['fo_success'] = pass_plays.apply(_fo_success_play, axis=1)


# ─── Per-player offensive snap totals ────────────────────────────────────────
receiver_positions = {'WR', 'TE'}
all_snaps = snaps[snaps['position'].isin(receiver_positions)].copy()

snap_totals = (
    all_snaps
    .groupby(['player', 'pfr_player_id', 'position', 'team'], as_index=False)
    .agg(
        off_snaps=('offense_snaps', 'sum'),
        games_played=('game_id', 'nunique'),
    )
)
print(f'\nTotal WR/TE player-seasons: {len(snap_totals)}')

eligible = snap_totals[snap_totals['games_played'] >= MIN_GAMES].copy()
print(f'After {MIN_GAMES}-game floor: {len(eligible)}')


# ─── Select top-N by position ────────────────────────────────────────────────
top_wrs = eligible[eligible['position'] == 'WR'].nlargest(TOP_N_WR, 'off_snaps')
top_tes = eligible[eligible['position'] == 'TE'].nlargest(TOP_N_TE, 'off_snaps')
population = pd.concat([top_wrs, top_tes], ignore_index=True)
print(f'\nFinal population: {len(top_wrs)} WRs + {len(top_tes)} TEs = {len(population)} players')


# ─── Match snap-total names to PBP player_ids ────────────────────────────────
# nflverse snap counts use PFR IDs; PBP uses gsis IDs. Join through rosters.
roster_slim = rosters[['gsis_id', 'pfr_id', 'full_name', 'position']].dropna(subset=['gsis_id'])
population = population.merge(
    roster_slim.rename(columns={'pfr_id': 'pfr_player_id'})[['pfr_player_id', 'gsis_id', 'full_name']],
    on='pfr_player_id',
    how='left',
)

# Some players don't join by pfr_id (practice squad / short-stint guys). Drop them.
missing = population['gsis_id'].isna().sum()
if missing:
    print(f'  Dropping {missing} players who couldn\'t be matched to PBP gsis_id')
    population = population.dropna(subset=['gsis_id']).copy()


# ─── Per-player receiving stats from PBP ─────────────────────────────────────
print('\nAggregating PBP receiving stats per player...')

def agg_receiver(group):
    targets = len(group)
    receptions = group['complete_pass'].sum()
    rec_yards = group['receiving_yards'].fillna(0).sum()
    rec_tds = group['touchdown'].fillna(0).sum() if 'pass_touchdown' not in group.columns else group['pass_touchdown'].fillna(0).sum()
    rec_first_downs = group['first_down'].fillna(0).sum() if 'first_down' in group.columns else np.nan
    air_yards = group['air_yards'].fillna(0).sum()
    epa_sum = group['epa'].fillna(0).sum()
    epa_per_target = group['epa'].mean()
    success_rate = group['fo_success'].mean()
    yac_sum = group['yards_after_catch'].fillna(0).sum()
    cpoe_mean = group['cpoe'].mean() if 'cpoe' in group.columns else np.nan
    return pd.Series({
        'targets': targets,
        'receptions': receptions,
        'rec_yards': rec_yards,
        'rec_tds': rec_tds,
        'rec_first_downs': rec_first_downs,
        'air_yards': air_yards,
        'epa_per_target': epa_per_target,
        'success_rate': success_rate,
        'yac': yac_sum,
        'avg_cpoe': cpoe_mean,
    })

receiver_stats = (
    pass_plays
    .groupby('receiver_player_id')
    .apply(agg_receiver, include_groups=False)
    .reset_index()
    .rename(columns={'receiver_player_id': 'gsis_id'})
)

population = population.merge(receiver_stats, on='gsis_id', how='left')
print(f'  {len(population)} players now have PBP receiving stats merged')


# ─── NGS-derived stats (separation, cushion, YAC over expected) ──────────────
if len(ngs_season) > 0:
    print('\nMerging NGS receiving stats...')
    ngs_cols_map = {
        'player_gsis_id': 'gsis_id',
        'avg_separation': 'avg_separation',
        'avg_cushion': 'avg_cushion',
        'avg_intended_air_yards': 'avg_target_depth',
        'avg_yac_above_expectation': 'yac_above_exp',
    }
    available_ngs_cols = {k: v for k, v in ngs_cols_map.items() if k in ngs_season.columns}
    if 'player_gsis_id' in ngs_season.columns:
        ngs_slim = ngs_season[list(available_ngs_cols.keys())].rename(columns=available_ngs_cols)
        population = population.merge(ngs_slim, on='gsis_id', how='left')
        print(f'  merged columns: {list(available_ngs_cols.values())}')
    else:
        print('  NGS table does not have player_gsis_id — skipping NGS merge')
        for col in ['avg_separation', 'avg_cushion', 'avg_target_depth', 'yac_above_exp']:
            population[col] = np.nan
else:
    print('\nNo NGS data — skipping NGS-derived stats')
    for col in ['avg_separation', 'avg_cushion', 'avg_target_depth', 'yac_above_exp']:
        population[col] = np.nan


# ─── Derived rate stats ──────────────────────────────────────────────────────
print('\nComputing derived stats...')
population['yards_per_target'] = population['rec_yards'] / population['targets'].replace(0, np.nan)
population['yards_per_snap'] = population['rec_yards'] / population['off_snaps'].replace(0, np.nan)
population['catch_rate'] = population['receptions'] / population['targets'].replace(0, np.nan)
population['targets_per_snap'] = population['targets'] / population['off_snaps'].replace(0, np.nan)
population['first_down_rate'] = population['rec_first_downs'] / population['targets'].replace(0, np.nan)
population['yac_per_reception'] = population['yac'] / population['receptions'].replace(0, np.nan)
population['pbp_targets'] = population['targets']


# ─── Z-score against the population ─────────────────────────────────────────
print('\nZ-scoring stats within the league population...')

STATS_TO_ZSCORE = [
    # Tier 1 — raw counts
    'rec_yards',
    'receptions',
    'rec_tds',
    'targets',
    # Tier 2 — rates
    'catch_rate',
    'success_rate',
    'first_down_rate',
    'yards_per_target',
    'yac_per_reception',
    'targets_per_snap',
    'yards_per_snap',
    # Tier 3 — adjusted / modeled
    'epa_per_target',
    'avg_cpoe',
    'yac_above_exp',
    'avg_separation',
]

INVERT_STATS = set()  # none for WR — all stats are higher-is-better

for stat in STATS_TO_ZSCORE:
    if stat not in population.columns:
        print(f'  skipping {stat}: column missing')
        population[f'{stat}_z'] = 0.0
        continue
    vals = population[stat].astype(float)
    # Drop NaNs for mean/std; NaN inputs keep NaN z-scores
    clean = vals.dropna()
    if len(clean) < 3:
        print(f'  skipping {stat}: only {len(clean)} non-null values')
        population[f'{stat}_z'] = np.nan
        continue
    mean = clean.mean()
    std = clean.std()
    if std and std > 0:
        z = (vals - mean) / std
        if stat in INVERT_STATS:
            z = -z
        population[f'{stat}_z'] = z
    else:
        population[f'{stat}_z'] = 0.0


# ─── Normalize column names for the Streamlit page ───────────────────────────
# The existing Receivers page expects `player_display_name` and `player_id`.
population['player_display_name'] = population['full_name']
population['player_id'] = population['gsis_id']
population['games'] = population['games_played']


# ─── Write outputs ───────────────────────────────────────────────────────────
# Final column order: keep raw stats + z-scores + identifiers
out_cols = [
    'player_id', 'player_display_name', 'position', 'team', 'games', 'off_snaps',
    # Raw counts
    'pbp_targets', 'targets', 'receptions', 'rec_yards', 'rec_tds', 'rec_first_downs',
    'air_yards', 'yac',
    # NGS (may be NaN)
    'avg_cushion', 'avg_separation', 'avg_target_depth',
    # Derived rates
    'catch_rate', 'success_rate', 'first_down_rate',
    'yards_per_target', 'yards_per_snap', 'targets_per_snap',
    'yac_per_reception', 'yac_above_exp',
    'epa_per_target', 'avg_cpoe',
    # Z-scores (the ones the page actually uses)
    'rec_yards_z', 'receptions_z', 'rec_tds_z', 'targets_z',
    'catch_rate_z', 'success_rate_z', 'first_down_rate_z',
    'yards_per_target_z', 'yards_per_snap_z', 'targets_per_snap_z',
    'yac_per_reception_z', 'yac_above_exp_z',
    'epa_per_target_z', 'avg_cpoe_z', 'avg_separation_z',
]
# Only include columns that exist
out_cols = [c for c in out_cols if c in population.columns]
out = population[out_cols].copy()

out.to_parquet(OUTPUT_PARQUET, index=False)
print(f'\nWrote {OUTPUT_PARQUET} ({len(out)} rows, {len(out.columns)} columns)')


# ─── Metadata: tier assignments, labels, methodology ─────────────────────────
STAT_TIERS = {
    'rec_yards_z': 1,
    'receptions_z': 1,
    'rec_tds_z': 1,
    'targets_z': 1,
    'catch_rate_z': 2,
    'success_rate_z': 2,
    'first_down_rate_z': 2,
    'yards_per_target_z': 2,
    'yards_per_snap_z': 2,
    'targets_per_snap_z': 2,
    'yac_per_reception_z': 2,
    'epa_per_target_z': 3,
    'avg_cpoe_z': 3,
    'yac_above_exp_z': 3,
    'avg_separation_z': 3,
}

STAT_LABELS = {
    'rec_yards_z': 'Receiving yards (raw)',
    'receptions_z': 'Receptions (raw)',
    'rec_tds_z': 'Receiving TDs (raw)',
    'targets_z': 'Targets (raw)',
    'catch_rate_z': 'Catch rate',
    'success_rate_z': 'Success rate',
    'first_down_rate_z': 'First-down rate',
    'yards_per_target_z': 'Yards per target',
    'yards_per_snap_z': 'Yards per snap',
    'targets_per_snap_z': 'Targets per snap',
    'yac_per_reception_z': 'YAC per reception',
    'epa_per_target_z': 'EPA per target',
    'avg_cpoe_z': 'CPOE',
    'yac_above_exp_z': 'YAC over expected',
    'avg_separation_z': 'Average separation',
}

STAT_METHODOLOGY = {
    'rec_yards_z': {
        'what': 'Total raw receiving yards.',
        'how': 'Sum of PBP receiving_yards, z-scored against the league population (top 64 WR + top 32 TE by snaps).',
        'limits': 'Raw volume stat — rewards opportunity as much as skill. High-volume WR1s will always outrank efficient role players here.',
    },
    'receptions_z': {
        'what': 'Total raw receptions.',
        'how': 'Count of complete passes where this player was the receiver, z-scored against the population.',
        'limits': 'Volume stat. A possession receiver with 110 catches outranks a deep threat with 50 catches even if the deep threat averaged more yards.',
    },
    'rec_tds_z': {
        'what': 'Total raw receiving touchdowns.',
        'how': 'Count of TDs on pass plays where this player was the receiver.',
        'limits': 'Small integer samples are noisy. Four TDs vs. six TDs is a 50% difference but could easily be luck over 17 games.',
    },
    'targets_z': {
        'what': 'Total raw targets.',
        'how': 'Count of pass plays where this player was the intended receiver.',
        'limits': 'Pure opportunity — this is "how much did the QB look your way," not a skill measure.',
    },
    'catch_rate_z': {
        'what': 'Percentage of targets caught.',
        'how': 'receptions / targets.',
        'limits': 'Drops and defended passes both count as incomplete. Doesn\'t account for target difficulty.',
    },
    'success_rate_z': {
        'what': 'Percentage of targets that produced a "successful" play by EPA standards.',
        'how': 'nflverse tags each play with a binary success flag; we take the mean across this player\'s targets.',
        'limits': 'Success is defined by an EPA threshold that varies by down/distance. The binary cutoff hides near-misses and runaway successes.',
    },
    'first_down_rate_z': {
        'what': 'Percentage of targets that gained a first down.',
        'how': 'first_downs / targets.',
        'limits': 'Chain-moving is valuable but depends on how the offense uses you — slot receivers on 3rd-and-short will post big numbers here.',
    },
    'yards_per_target_z': {
        'what': 'Average yards per target (not per reception).',
        'how': 'total receiving yards / total targets.',
        'limits': 'Penalizes drops as zeros. Rewards big plays disproportionately.',
    },
    'yards_per_snap_z': {
        'what': 'Receiving yards per offensive snap on the field.',
        'how': 'total receiving yards / offensive snaps.',
        'limits': 'Best efficiency-of-role metric available from free data, but blocking TEs who rarely get targeted will look bad.',
    },
    'targets_per_snap_z': {
        'what': 'How often the QB looks your way per snap on the field.',
        'how': 'targets / offensive snaps.',
        'limits': 'Measures role, not skill. Schemed targets count the same as earned targets.',
    },
    'yac_per_reception_z': {
        'what': 'Average yards gained after the catch, per reception.',
        'how': 'total yards_after_catch / receptions.',
        'limits': 'Credit for YAC is shared between the receiver (did you break tackles / run well) and the scheme / blockers. Not purely a receiver stat.',
    },
    'epa_per_target_z': {
        'what': 'Expected Points Added per target.',
        'how': 'mean of nflverse epa on this player\'s targets.',
        'limits': 'EPA is a modeled value built from historical down/distance/field-position outcomes. Your score depends on trusting the EPA model.',
    },
    'avg_cpoe_z': {
        'what': 'Completion Percentage Over Expected.',
        'how': 'nflverse computes expected completion probability based on throw difficulty, then this stat is actual_completion - expected. We average across the player\'s targets.',
        'limits': 'Measures catching catches you\'re supposed to catch. A model decides what "supposed to" means, using throw distance, separation, etc.',
    },
    'yac_above_exp_z': {
        'what': 'Yards After Catch vs. what a league-average receiver would produce in the same situations.',
        'how': 'NFL Next Gen Stats computes expected YAC from tracking data (defender proximity, angle, etc.); this is actual - expected.',
        'limits': 'Requires NGS tracking data. Small-sample receivers may have missing or unstable values.',
    },
    'avg_separation_z': {
        'what': 'Average yards of separation from nearest defender at the moment of the catch.',
        'how': 'NFL Next Gen Stats tracking data, season average.',
        'limits': 'Depth-blind — a 2-yard separation on a deep route is more impressive than 2 yards of separation on a hitch. Doesn\'t tell you if the separation came from route running or scheme.',
    },
}

metadata = {
    'position_group': 'receiver',
    'season': SEASON,
    'population': f'Top {TOP_N_WR} WR + top {TOP_N_TE} TE by offensive snaps, min {MIN_GAMES} games',
    'n_players': int(len(out)),
    'stat_tiers': STAT_TIERS,
    'stat_labels': STAT_LABELS,
    'stat_methodology': STAT_METHODOLOGY,
    'invert_stats': sorted(list(INVERT_STATS)),
}

with open(OUTPUT_METADATA, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Wrote {OUTPUT_METADATA}')


# ─── Summary ────────────────────────────────────────────────────────────────
print('\n─────────────────────────────────────────')
print('SUMMARY')
print('─────────────────────────────────────────')
print(f'Players in population: {len(out)}')
print(f'  WRs: {(out["position"] == "WR").sum()}')
print(f'  TEs: {(out["position"] == "TE").sum()}')
print()
print('Lions players in the pool:')
lions_in_pool = out[out['team'] == 'DET'].sort_values('off_snaps', ascending=False)
print(lions_in_pool[['player_display_name', 'position', 'off_snaps', 'targets', 'rec_yards']].to_string(index=False))
print()
print('Top 10 by receiving yards:')
top10 = out.nlargest(10, 'rec_yards')
print(top10[['player_display_name', 'position', 'team', 'rec_yards', 'receptions', 'rec_tds']].to_string(index=False))
print()
print('Done. Download both files from Colab and upload to your GitHub repo at:')
print(f'  data/{OUTPUT_PARQUET}')
print(f'  data/{OUTPUT_METADATA}')
