# ─────────────────────────────────────────────────────────────────────────────
# Lions Rater — Running Backs Data Pull (2024)
# ─────────────────────────────────────────────────────────────────────────────
# Builds master_lions_rbs_with_z.parquet and rb_stat_metadata.json.
#
# Key architectural decisions:
#
# 1. POPULATION: top 32 RBs by offensive snaps, min 6 games. Roughly one
#    starter per team. Z-scored within this league-wide population.
#
# 2. FILTER METRIC IS SNAPS, NOT CARRIES. If we filtered by carries or yards
#    we'd be selecting on the same metrics we rate with, biasing the
#    z-score distribution by construction. Snap count is the cleanest
#    opportunity-neutral filter.
#
# 3. TIER 1 RAW-COUNT Z-SCORES: z_rush_yards, z_rush_tds, z_carries,
#    z_receptions (for RBs), z_rec_yards (for RBs), z_rec_tds (for RBs).
#    These fill the "Tier 1 — Counted" checkbox on the app.
#
# 4. FTN CHARTING STATS (broken tackles, YBC, YAC): These come from
#    FTN via nflverse. They're painstakingly-counted facts, not modeled
#    values — so we classify them as Tier 2 even though the counting took
#    human effort. If the FTN data pull fails, those stats will be NaN.
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
TOP_N_RB = 32

OUTPUT_PARQUET = Path('master_lions_rbs_with_z.parquet')
OUTPUT_METADATA = Path('rb_stat_metadata.json')


# ─── Pull raw data ───────────────────────────────────────────────────────────
print('Pulling play-by-play...')
pbp = nfl.load_pbp([SEASON]).to_pandas()
print(f'  {len(pbp):,} plays loaded')

print('Pulling snap counts...')
snaps = nfl.load_snap_counts([SEASON]).to_pandas()
print(f'  {len(snaps):,} player-game rows loaded')

print('Pulling PFR rushing advanced stats (YBC, YAC, broken tackles)...')
try:
    rush_pfr = nfl.load_pfr_advstats([SEASON], stat_type='rush', summary_level='season').to_pandas()
    print(f'  {len(rush_pfr):,} PFR rush rows')
except Exception as e:
    print(f'  PFR rush pull failed ({e}) — FTN-style stats will be NaN')
    rush_pfr = pd.DataFrame()

print('Pulling NGS rushing stats (for RYOE, time to LOS)...')
try:
    ngs = nfl.load_nextgen_stats([SEASON], stat_type='rushing').to_pandas()
    ngs_season = ngs[ngs['week'] == 0].copy() if 'week' in ngs.columns else ngs.copy()
    print(f'  {len(ngs_season):,} NGS rows (season aggregates)')
except Exception as e:
    print(f'  NGS pull failed ({e}) — RYOE will be NaN')
    ngs_season = pd.DataFrame()

print('Pulling rosters...')
rosters = nfl.load_rosters([SEASON]).to_pandas()
print(f'  {len(rosters):,} player rows loaded')


# ─── Per-player offensive snap totals ────────────────────────────────────────
rb_snaps = snaps[snaps['position'] == 'RB'].copy()

snap_totals = (
    rb_snaps
    .groupby(['player', 'pfr_player_id', 'team'], as_index=False)
    .agg(
        off_snaps=('offense_snaps', 'sum'),
        games_played=('game_id', 'nunique'),
    )
)
print(f'\nTotal RB player-seasons: {len(snap_totals)}')

eligible = snap_totals[snap_totals['games_played'] >= MIN_GAMES].copy()
print(f'After {MIN_GAMES}-game floor: {len(eligible)}')


# ─── Select top-32 by snaps ──────────────────────────────────────────────────
population = eligible.nlargest(TOP_N_RB, 'off_snaps').copy()
print(f'\nFinal population: {len(population)} RBs')


# ─── Match to gsis_id via rosters ────────────────────────────────────────────
roster_slim = rosters[['gsis_id', 'pfr_id', 'full_name', 'position']].dropna(subset=['gsis_id'])
population = population.merge(
    roster_slim.rename(columns={'pfr_id': 'pfr_player_id'})[['pfr_player_id', 'gsis_id', 'full_name']],
    on='pfr_player_id',
    how='left',
)

missing = population['gsis_id'].isna().sum()
if missing:
    print(f'  Dropping {missing} RBs unmatched to PBP gsis_id')
    population = population.dropna(subset=['gsis_id']).copy()


# ─── Rushing stats from PBP ──────────────────────────────────────────────────
print('\nAggregating PBP rushing stats per player...')

rush_plays = pbp[
    (pbp['play_type'] == 'run') &
    (pbp['rusher_player_id'].notna())
].copy()

# Explosive thresholds
rush_plays['is_explosive_10'] = (rush_plays['yards_gained'] >= 10).astype(int)
rush_plays['is_explosive_15'] = (rush_plays['yards_gained'] >= 15).astype(int)

# Goal line / short yardage flags
rush_plays['is_gl'] = (rush_plays['yardline_100'] <= 5).astype(int) if 'yardline_100' in rush_plays.columns else 0
rush_plays['is_sy'] = (
    (rush_plays['ydstogo'] <= 2) &
    (rush_plays['down'].isin([3, 4]))
).astype(int) if 'ydstogo' in rush_plays.columns else 0
rush_plays['is_sy_converted'] = (rush_plays['is_sy'] == 1) & (rush_plays['first_down'].fillna(0) == 1) if 'first_down' in rush_plays.columns else 0
rush_plays['is_gl_td'] = (rush_plays['is_gl'] == 1) & (rush_plays['rush_touchdown'].fillna(0) == 1) if 'rush_touchdown' in rush_plays.columns else 0
rush_plays['is_rz'] = (rush_plays['yardline_100'] <= 20).astype(int) if 'yardline_100' in rush_plays.columns else 0


def agg_rusher(group):
    carries = len(group)
    rush_yards = group['yards_gained'].fillna(0).sum()
    rush_tds = group['rush_touchdown'].fillna(0).sum() if 'rush_touchdown' in group.columns else 0
    rush_first_downs = group['first_down'].fillna(0).sum() if 'first_down' in group.columns else 0
    epa_mean = group['epa'].mean()
    success_mean = group['success'].mean()

    explosive_10 = group['is_explosive_10'].sum()
    explosive_15 = group['is_explosive_15'].sum()

    rz_carries = group['is_rz'].sum()
    gl_attempts = group['is_gl'].sum()
    gl_tds = group['is_gl_td'].sum() if 'is_gl_td' in group.columns else 0
    sy_attempts = group['is_sy'].sum()
    sy_conversions = group['is_sy_converted'].sum() if 'is_sy_converted' in group.columns else 0

    return pd.Series({
        'pbp_carries': carries,
        'rush_yards': rush_yards,
        'rush_tds': rush_tds,
        'rush_first_downs': rush_first_downs,
        'epa_per_rush': epa_mean,
        'rush_success_rate': success_mean,
        'explosive_10_count': explosive_10,
        'explosive_15_count': explosive_15,
        'rz_carries': rz_carries,
        'gl_attempts': gl_attempts,
        'gl_tds': gl_tds,
        'sy_attempts': sy_attempts,
        'sy_conversions': sy_conversions,
    })


rushing_stats = (
    rush_plays
    .groupby('rusher_player_id')
    .apply(agg_rusher, include_groups=False)
    .reset_index()
    .rename(columns={'rusher_player_id': 'gsis_id'})
)

population = population.merge(rushing_stats, on='gsis_id', how='left')


# ─── Receiving stats (RBs also catch passes) ────────────────────────────────
print('Aggregating RB receiving stats...')

pass_plays = pbp[
    (pbp['play_type'] == 'pass') &
    (pbp['receiver_player_id'].notna())
].copy()

def agg_rb_receiver(group):
    targets = len(group)
    receptions = group['complete_pass'].sum()
    rec_yards = group['receiving_yards'].fillna(0).sum()
    rec_tds = group['pass_touchdown'].fillna(0).sum() if 'pass_touchdown' in group.columns else 0
    epa_mean = group['epa'].mean()
    success_mean = group['success'].mean()
    yac_sum = group['yards_after_catch'].fillna(0).sum()
    return pd.Series({
        'targets': targets,
        'receptions': receptions,
        'rec_yards': rec_yards,
        'rec_tds': rec_tds,
        'rec_epa_per_target': epa_mean,
        'rec_success_rate': success_mean,
        'yac': yac_sum,
    })

rb_receiving = (
    pass_plays
    .groupby('receiver_player_id')
    .apply(agg_rb_receiver, include_groups=False)
    .reset_index()
    .rename(columns={'receiver_player_id': 'gsis_id'})
)

population = population.merge(rb_receiving, on='gsis_id', how='left')


# ─── PFR advanced rushing stats (YBC, YAC, broken tackles) ───────────────────
if len(rush_pfr) > 0:
    print('\nMerging PFR advanced rushing stats...')
    pfr_cols = ['pfr_player_id', 'att', 'ybc', 'yac', 'brk_tkl']
    pfr_available = [c for c in pfr_cols if c in rush_pfr.columns]
    # PFR uses 'pfr_id' column name sometimes
    if 'pfr_player_id' not in rush_pfr.columns and 'pfr_id' in rush_pfr.columns:
        rush_pfr = rush_pfr.rename(columns={'pfr_id': 'pfr_player_id'})
        pfr_available = [c if c != 'pfr_id' else 'pfr_player_id' for c in pfr_available]

    if 'pfr_player_id' in rush_pfr.columns:
        pfr_slim = rush_pfr[[c for c in ['pfr_player_id', 'att', 'ybc', 'yac', 'brk_tkl'] if c in rush_pfr.columns]].copy()
        # Rename to make downstream derived stats clearer
        pfr_slim = pfr_slim.rename(columns={
            'att': 'pfr_carries',
            'ybc': 'yards_before_contact_total',
            'yac': 'yards_after_contact_total',
            'brk_tkl': 'broken_tackles_total',
        })
        population = population.merge(pfr_slim, on='pfr_player_id', how='left')
        print(f'  merged: {list(pfr_slim.columns)}')
    else:
        print('  PFR data missing pfr_player_id — skipping')
        for col in ['yards_before_contact_total', 'yards_after_contact_total', 'broken_tackles_total']:
            population[col] = np.nan
else:
    print('\nNo PFR advanced rushing data — YBC/YAC/broken tackles will be NaN')
    for col in ['yards_before_contact_total', 'yards_after_contact_total', 'broken_tackles_total']:
        population[col] = np.nan


# ─── NGS rushing stats (RYOE, time to LOS) ──────────────────────────────────
if len(ngs_season) > 0:
    print('\nMerging NGS rushing stats...')
    ngs_cols_map = {
        'player_gsis_id': 'gsis_id',
        'rush_yards_over_expected_per_att': 'ryoe_per_att',
        'avg_time_to_los': 'avg_time_to_los',
        'efficiency': 'efficiency',
        'percent_attempts_gte_eight_defenders': 'stacked_box_rate',
    }
    available_ngs = {k: v for k, v in ngs_cols_map.items() if k in ngs_season.columns}
    if 'player_gsis_id' in ngs_season.columns:
        ngs_slim = ngs_season[list(available_ngs.keys())].rename(columns=available_ngs)
        population = population.merge(ngs_slim, on='gsis_id', how='left')
        print(f'  merged: {list(available_ngs.values())}')
    else:
        print('  NGS has no player_gsis_id — skipping')
        for col in ['ryoe_per_att', 'avg_time_to_los', 'efficiency', 'stacked_box_rate']:
            population[col] = np.nan
else:
    print('\nNo NGS rushing data')
    for col in ['ryoe_per_att', 'avg_time_to_los', 'efficiency', 'stacked_box_rate']:
        population[col] = np.nan


# ─── Derived rate stats ──────────────────────────────────────────────────────
print('\nComputing derived stats...')

population['carries'] = population['pbp_carries']
population['games'] = population['games_played']

safe_div = lambda a, b: a / b.replace(0, np.nan) if hasattr(b, 'replace') else np.where(b > 0, a / b, np.nan)

population['yards_per_carry'] = population['rush_yards'] / population['carries'].replace(0, np.nan)
population['carries_per_game'] = population['carries'] / population['games'].replace(0, np.nan)
population['snap_share'] = population['off_snaps'] / (population['games'] * 65).replace(0, np.nan)
population['touches_per_game'] = (population['carries'].fillna(0) + population['receptions'].fillna(0)) / population['games'].replace(0, np.nan)
population['targets_per_game'] = population['targets'].fillna(0) / population['games'].replace(0, np.nan)

population['explosive_run_rate'] = population['explosive_10_count'] / population['carries'].replace(0, np.nan)
population['explosive_15_rate'] = population['explosive_15_count'] / population['carries'].replace(0, np.nan)

population['rz_carry_share'] = population['rz_carries'] / population['carries'].replace(0, np.nan)
population['goal_line_td_rate'] = population['gl_tds'] / population['gl_attempts'].replace(0, np.nan)
population['short_yardage_conv_rate'] = population['sy_conversions'] / population['sy_attempts'].replace(0, np.nan)

population['rec_yards_per_target'] = population['rec_yards'] / population['targets'].replace(0, np.nan)
population['yac_per_reception'] = population['yac'] / population['receptions'].replace(0, np.nan)

# FTN-derived rates
if 'yards_before_contact_total' in population.columns:
    pfr_carries = population.get('pfr_carries', population['carries']).replace(0, np.nan)
    population['yards_before_contact_per_att'] = population['yards_before_contact_total'] / pfr_carries
    population['yards_after_contact_per_att'] = population['yards_after_contact_total'] / pfr_carries
    population['broken_tackles_per_att'] = population['broken_tackles_total'] / pfr_carries


# ─── Z-scores against the population ────────────────────────────────────────
print('\nZ-scoring stats...')

STATS_TO_ZSCORE = [
    # Tier 1 — raw counts
    'rush_yards',
    'rush_tds',
    'carries',
    'receptions',
    'rec_yards',
    'rec_tds',
    # Tier 2 — rates
    'yards_per_carry',
    'rush_success_rate',
    'carries_per_game',
    'snap_share',
    'touches_per_game',
    'targets_per_game',
    'explosive_run_rate',
    'explosive_15_rate',
    'rz_carry_share',
    'goal_line_td_rate',
    'short_yardage_conv_rate',
    'rec_yards_per_target',
    'yac_per_reception',
    'broken_tackles_per_att',
    'yards_before_contact_per_att',
    'yards_after_contact_per_att',
    # Tier 3 — modeled / NGS
    'epa_per_rush',
    'rec_epa_per_target',
    'ryoe_per_att',
]

INVERT_STATS = set()  # none for RB

for stat in STATS_TO_ZSCORE:
    if stat not in population.columns:
        print(f'  skipping {stat}: column missing')
        population[f'{stat}_z'] = np.nan
        continue
    vals = population[stat].astype(float)
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
population['player_display_name'] = population['full_name']
population['player_id'] = population['gsis_id']
population['position'] = 'RB'


# ─── Write outputs ───────────────────────────────────────────────────────────
out_cols = [
    'player_id', 'player_display_name', 'position', 'team', 'games', 'off_snaps',
    'carries', 'pbp_carries', 'rush_yards', 'rush_tds', 'rush_first_downs',
    'receptions', 'targets', 'rec_yards', 'rec_tds', 'yac',
    # Rate / derived
    'yards_per_carry', 'rush_success_rate', 'carries_per_game', 'snap_share',
    'touches_per_game', 'targets_per_game',
    'explosive_run_rate', 'explosive_15_rate',
    'rz_carry_share', 'goal_line_td_rate', 'short_yardage_conv_rate',
    'rec_yards_per_target', 'yac_per_reception',
    'rz_carries', 'gl_attempts', 'gl_tds', 'sy_attempts', 'sy_conversions',
    'explosive_10_count', 'explosive_15_count',
    'broken_tackles_per_att', 'yards_before_contact_per_att', 'yards_after_contact_per_att',
    # Modeled / NGS
    'epa_per_rush', 'rec_epa_per_target', 'rec_success_rate',
    'ryoe_per_att', 'avg_time_to_los', 'efficiency', 'stacked_box_rate',
    # Z-scores
    'rush_yards_z', 'rush_tds_z', 'carries_z',
    'receptions_z', 'rec_yards_z', 'rec_tds_z',
    'yards_per_carry_z', 'rush_success_rate_z',
    'carries_per_game_z', 'snap_share_z', 'touches_per_game_z', 'targets_per_game_z',
    'explosive_run_rate_z', 'explosive_15_rate_z',
    'rz_carry_share_z', 'goal_line_td_rate_z', 'short_yardage_conv_rate_z',
    'rec_yards_per_target_z', 'yac_per_reception_z',
    'broken_tackles_per_att_z', 'yards_before_contact_per_att_z', 'yards_after_contact_per_att_z',
    'epa_per_rush_z', 'rec_epa_per_target_z', 'ryoe_per_att_z',
]
out_cols = [c for c in out_cols if c in population.columns]
out = population[out_cols].copy()

out.to_parquet(OUTPUT_PARQUET, index=False)
print(f'\nWrote {OUTPUT_PARQUET} ({len(out)} rows, {len(out.columns)} columns)')


# ─── Metadata: tier assignments, labels, methodology ─────────────────────────
STAT_TIERS = {
    # Tier 1 — raw counts
    'rush_yards_z': 1,
    'rush_tds_z': 1,
    'carries_z': 1,
    'receptions_z': 1,
    'rec_yards_z': 1,
    'rec_tds_z': 1,
    # Tier 2 — rates and FTN-counted facts
    'yards_per_carry_z': 2,
    'rush_success_rate_z': 2,
    'carries_per_game_z': 2,
    'snap_share_z': 2,
    'touches_per_game_z': 2,
    'targets_per_game_z': 2,
    'explosive_run_rate_z': 2,
    'explosive_15_rate_z': 2,
    'rz_carry_share_z': 2,
    'goal_line_td_rate_z': 2,
    'short_yardage_conv_rate_z': 2,
    'rec_yards_per_target_z': 2,
    'yac_per_reception_z': 2,
    'broken_tackles_per_att_z': 2,
    'yards_before_contact_per_att_z': 2,
    'yards_after_contact_per_att_z': 2,
    # Tier 3 — modeled / NGS
    'epa_per_rush_z': 3,
    'rec_epa_per_target_z': 3,
    'ryoe_per_att_z': 3,
}

STAT_LABELS = {
    'rush_yards_z': 'Rushing yards (raw)',
    'rush_tds_z': 'Rushing TDs (raw)',
    'carries_z': 'Carries (raw)',
    'receptions_z': 'Receptions (raw)',
    'rec_yards_z': 'Receiving yards (raw)',
    'rec_tds_z': 'Receiving TDs (raw)',
    'yards_per_carry_z': 'Yards per carry',
    'rush_success_rate_z': 'Rush success rate',
    'carries_per_game_z': 'Carries per game',
    'snap_share_z': 'Snap share',
    'touches_per_game_z': 'Touches per game',
    'targets_per_game_z': 'Targets per game',
    'explosive_run_rate_z': 'Explosive run rate (10+)',
    'explosive_15_rate_z': '15+ yard run rate',
    'rz_carry_share_z': 'Red zone carry share',
    'goal_line_td_rate_z': 'Goal line TD rate',
    'short_yardage_conv_rate_z': 'Short yardage conversion rate',
    'rec_yards_per_target_z': 'Receiving yards per target',
    'yac_per_reception_z': 'YAC per reception',
    'broken_tackles_per_att_z': 'Broken tackles per attempt',
    'yards_before_contact_per_att_z': 'Yards before contact per attempt',
    'yards_after_contact_per_att_z': 'Yards after contact per attempt',
    'epa_per_rush_z': 'EPA per rush',
    'rec_epa_per_target_z': 'Receiving EPA per target',
    'ryoe_per_att_z': 'Rush yards over expected per attempt',
}

STAT_METHODOLOGY = {
    'rush_yards_z': {
        'what': 'Total raw rushing yards.',
        'how': 'Sum of PBP yards_gained on run plays where this player was the rusher.',
        'limits': 'Pure volume. Rewards workhorse usage more than efficiency.',
    },
    'rush_tds_z': {
        'what': 'Total rushing touchdowns.',
        'how': 'Count of rushes that resulted in a touchdown.',
        'limits': 'Small-integer stat. A goal-line back can post big numbers without being a strong overall runner.',
    },
    'carries_z': {
        'what': 'Total raw carries.',
        'how': 'Count of rush plays where this player was the rusher.',
        'limits': 'Opportunity stat. Tells you who got the ball, not who did best with it.',
    },
    'receptions_z': {
        'what': 'Total raw receptions (as a pass catcher, not a rusher).',
        'how': 'Count of complete passes where this RB was the receiver.',
        'limits': 'Volume — favors dual-threat backs regardless of efficiency.',
    },
    'rec_yards_z': {
        'what': 'Total receiving yards for this RB.',
        'how': 'Sum of receiving_yards on this player\'s pass targets.',
        'limits': 'Depends on how much the offense uses the RB as a receiver. Reggie Bush-style usage vs. pure grinder.',
    },
    'rec_tds_z': {
        'what': 'Receiving touchdowns.',
        'how': 'Count of TDs on this player\'s pass targets.',
        'limits': 'Small integer samples are noisy for TDs.',
    },
    'yards_per_carry_z': {
        'what': 'Average yards gained per carry.',
        'how': 'total rush yards / total carries.',
        'limits': 'Doesn\'t distinguish 4 yards on first down (good) from 4 yards on 3rd-and-8 (bad). Heavily influenced by blocking.',
    },
    'rush_success_rate_z': {
        'what': 'Percentage of carries that were "successful" by EPA standards.',
        'how': 'nflverse defines success as meeting an EPA threshold that varies by down/distance; we take the mean across this player\'s carries.',
        'limits': 'The binary threshold smooths over near-misses. A 3-yard gain on 3rd-and-4 is a failure; a 3-yard gain on 3rd-and-2 is a success.',
    },
    'carries_per_game_z': {
        'what': 'Average carries per game played.',
        'how': 'total carries / games played.',
        'limits': 'A workhorse metric. Committee backs get penalized; it\'s not their fault the team uses two backs.',
    },
    'snap_share_z': {
        'what': 'Share of team offensive snaps played.',
        'how': 'offensive snaps / (games × 65 snap estimate).',
        'limits': 'Team snap counts vary — 65/game is an estimate. A back on a pass-happy team plays more snaps per carry.',
    },
    'touches_per_game_z': {
        'what': 'Combined carries + receptions per game.',
        'how': '(carries + receptions) / games.',
        'limits': 'Conflates rushing and receiving workloads as if they\'re equivalent. A catch and a carry both count as one touch.',
    },
    'targets_per_game_z': {
        'what': 'Average passing targets per game.',
        'how': 'targets / games.',
        'limits': 'Measures how often the offense looks your way as a receiver, not how well you catch.',
    },
    'explosive_run_rate_z': {
        'what': 'Percentage of carries that gained 10+ yards.',
        'how': '10+ yard runs / total carries.',
        'limits': 'Requires both the line AND the back. Hard to attribute a breakaway to one or the other.',
    },
    'explosive_15_rate_z': {
        'what': 'Percentage of carries that gained 15+ yards.',
        'how': '15+ yard runs / total carries.',
        'limits': 'Higher threshold = noisier for small samples. A back with 80 carries needs 2-3 hits to look "explosive."',
    },
    'rz_carry_share_z': {
        'what': 'Share of this player\'s carries that came inside the 20.',
        'how': 'red zone carries / total carries.',
        'limits': 'Measures trust in scoring position, but volume backs may have lower rates even though they get more red zone work overall.',
    },
    'goal_line_td_rate_z': {
        'what': 'Touchdown rate on inside-the-5 carries.',
        'how': 'goal-line TDs / goal-line attempts.',
        'limits': 'Tiny samples. Most backs have fewer than 10 goal-line carries in a season.',
    },
    'short_yardage_conv_rate_z': {
        'what': 'Conversion rate on 3rd/4th-and-1 or 3rd/4th-and-2.',
        'how': 'short-yardage first-downs / short-yardage attempts.',
        'limits': 'Small samples plus a lot depends on blocking. Short-yardage success is often a line stat as much as a back stat.',
    },
    'rec_yards_per_target_z': {
        'what': 'Average yards per target when targeted.',
        'how': 'total receiving yards / total targets.',
        'limits': 'Drops count as zeros. Screen passes and outlet dumps count the same as downfield targets.',
    },
    'yac_per_reception_z': {
        'what': 'Average yards after catch on receptions.',
        'how': 'total YAC / receptions.',
        'limits': 'Influenced by scheme (screens get big YAC) and blocking, not pure RB skill.',
    },
    'broken_tackles_per_att_z': {
        'what': 'Average broken tackles per rushing attempt, from FTN charting.',
        'how': 'FTN charts each play and flags whether a tackle was broken; we take total broken tackles / attempts.',
        'limits': 'Counting broken tackles is subjective — different charters draw the line differently. FTN is one of the most reliable sources but it\'s still human judgment.',
    },
    'yards_before_contact_per_att_z': {
        'what': 'Average yards gained before first defender contact.',
        'how': 'From FTN/PFR charting: yards before contact / attempts.',
        'limits': 'This is mostly an offensive line stat disguised as a runner stat. A great line gives a mediocre back a great YBC average.',
    },
    'yards_after_contact_per_att_z': {
        'what': 'Average yards gained after first defender contact.',
        'how': 'From FTN/PFR charting: yards after contact / attempts.',
        'limits': 'Pure RB skill metric — what happens when someone gets their hands on you. Still dependent on which defenders are making first contact.',
    },
    'epa_per_rush_z': {
        'what': 'Expected Points Added per rushing attempt.',
        'how': 'mean of nflverse EPA across this player\'s carries.',
        'limits': 'EPA is modeled from historical down/distance/field-position outcomes. Trusts that model.',
    },
    'rec_epa_per_target_z': {
        'what': 'Expected Points Added per receiving target.',
        'how': 'mean of nflverse EPA across this player\'s targets as a receiver.',
        'limits': 'Same EPA-model trust issue as above.',
    },
    'ryoe_per_att_z': {
        'what': 'Rush Yards Over Expected per attempt.',
        'how': 'NFL Next Gen Stats computes expected yards from tracking data (blocking, box count, space around the runner). This stat is actual_yards - expected_yards, averaged per attempt.',
        'limits': 'NGS model is proprietary — we can\'t audit it. Small-sample backs can have unstable RYOE. But it\'s the best available "adjust for the line" metric from free data.',
    },
}

metadata = {
    'position_group': 'rb',
    'season': SEASON,
    'population': f'Top {TOP_N_RB} RBs by offensive snaps, min {MIN_GAMES} games',
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
print()
print('Lions players in the pool:')
lions_in_pool = out[out['team'] == 'DET'].sort_values('off_snaps', ascending=False)
if len(lions_in_pool) > 0:
    print(lions_in_pool[['player_display_name', 'off_snaps', 'carries', 'rush_yards']].to_string(index=False))
else:
    print('  (none — no Lions RBs in top 32 by snaps)')
print()
print('Top 10 RBs by rushing yards:')
top10 = out.nlargest(10, 'rush_yards')
print(top10[['player_display_name', 'team', 'carries', 'rush_yards', 'rush_tds']].to_string(index=False))
print()
print('Done. Download both files from Colab and upload to your GitHub repo at:')
print(f'  data/{OUTPUT_PARQUET}')
print(f'  data/{OUTPUT_METADATA}')
