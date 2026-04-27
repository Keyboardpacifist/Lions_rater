# NFL Rater

> Community-built, transparent alternative to PFF. Fans define their own rating methodologies using z-scored NFL stats with adjustable slider weights.

**Live:** https://lions-rater.streamlit.app
**Stack:** Streamlit + Pandas + Supabase + Plotly
**Data:** nflverse play-by-play, snap counts, PFR advanced stats, NFL Next Gen Stats, CFBD

## Quick Start

```bash
# Install dependencies
make install

# Run locally (requires .streamlit/secrets.toml — see Secrets section)
make run

# Run tests
make test

# Lint
make lint
```

## Secrets

The app needs Supabase credentials. Create `.streamlit/secrets.toml` (NEVER commit this):

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

On Streamlit Cloud, these are configured in the app dashboard under Settings > Secrets.

## Architecture

```
app.py                    # Landing page: NFL mode + College mode
lib_shared.py             # Shared logic: Supabase CRUD, scoring engine, community UI
team_selector.py          # Team/season dropdown used by all NFL pages
career_arc.py             # Career trajectory chart component
comps.py                  # Statistical comparables engine (college-to-pro)
pedigree.py               # Player pedigree/validation scoring
college_data.py           # College data loaders and matching

pages/                    # Streamlit auto-discovers these as navigation pages
  QB.py, WR.py, TE.py    # Offensive skill positions
  2_Running_backs.py      # RB (legacy naming with number prefix)
  3_Offensive_Line.py     # OL (has the most advanced tier system)
  DE.py, DT.py, LB.py    # Defensive front
  CB.py, Safety..py       # Secondary (NOTE: Safety has double-dot typo)
  Kicker.py, Punter.py    # Special teams
  4 coaches.py            # Coaches (NOTE: space in filename)
  OC.py, DC_coord.py      # Coordinators
  GM.py                   # General managers

tools/                    # Data pull scripts (designed for Google Colab)
  wr_data_pull.py         # Generates: master_lions_with_z.parquet
  rb_data_pull.py         # Generates: master_lions_rbs_with_z.parquet

data/                     # Pre-computed parquet files + stat metadata JSON
  league_*_all_seasons.parquet    # League-wide reference data (for z-scores)
  master_*_with_z.parquet         # Position-specific z-scored data
  *_stat_metadata.json            # Stat definitions, tiers, methodology
  college/                        # College player stats, recruiting, comps
```

### Data Flow

```
nflverse APIs ──→ tools/*_data_pull.py ──→ data/*.parquet + *_metadata.json
                                                │
app.py / pages/*.py ◄── lib_shared.py ◄─────────┘
        │                    │
        │                    ├── score_players() — weighted z-score average
        │                    ├── community_section() — save/load/fork/upvote
        │                    └── get_supabase() — cached Supabase client
        │
        └──→ Streamlit Cloud (auto-deploys on push to main)
```

### How Scoring Works

1. Each stat is z-scored against a league reference population (e.g., top 64 WRs)
2. Users adjust bundle sliders (e.g., "Reliability: 60, Explosiveness: 50")
3. `compute_effective_weights()` converts bundles to per-stat weights
4. `score_players()` computes: `score = Σ(z_stat × weight) / total_weight`
5. Result: a composite z-score where 0.00 = league average

### Page Pattern

Every position page follows the same structure:

1. **Config** — `POSITION_GROUP`, `DATA_PATH`, `METADATA_PATH`, `BUNDLES`, `RAW_COL_MAP`, `RADAR_STATS`, `DEFAULT_BUNDLE_WEIGHTS`
2. **Helpers** — `zscore_to_percentile`, `format_percentile`, `format_score`, `sample_size_warning`, tier functions, `build_radar_figure` (these are copy-pasted in every file — see Roadmap Phase 2)
3. **Sidebar** — bundle sliders + optional advanced per-stat mode
4. **Scoring** — filter by snaps, call `score_players()`, sort
5. **Display** — leaderboard table, player detail with radar chart, career arc
6. **Community** — save/browse/fork/upvote algorithms via `community_section()`

## Key Conventions

- **Z-scores everywhere.** Every stat is z-scored against league reference populations. Never show raw stats without z-score context.
- **Bundles over raw weights.** Users interact with "bundles" (plain-English skill groups). Per-stat weights are computed from bundles. Advanced mode allows per-stat override.
- **Tier system.** Stats are classified by epistemic confidence: Tier 1 (counted) → Tier 4 (inferred). Users can filter tiers. Currently implemented on OL, WR, QB pages. Rolling out to others.
- **Reference vs. output populations.** Reference pool (e.g., top 64 WR) sets the z-score baseline. Output includes ALL players on the selected team, even low-snap players.
- **`@st.cache_data` for data loading.** All parquet reads should be cached.
- **Position scoping.** Community algorithms have a `position_group` column. Each page only shows its own position's algorithms.

## Known Issues

- `pages/Safety..py` — double period in filename (cosmetic, works but should be `Safety.py`)
- `pages/4 coaches.py` — space in filename (should be `4_Coaches.py`)
- `gm_exploration_output.csv` — orphaned CSV in repo root (should be in data/ or removed)
- First-load scores sometimes show as zero until a slider is touched (Streamlit session state quirk)
- README.md references `*_data_pull_v2.py` scripts that don't exist (README is outdated)
- Only 2 of ~48 parquet files have generation scripts (see Data Gap below)

## Data Gap — CRITICAL

Only these parquets can be regenerated from scripts in this repo:

| Script | Output |
|--------|--------|
| `tools/wr_data_pull.py` | `master_lions_with_z.parquet` |
| `tools/rb_data_pull.py` | `master_lions_rbs_with_z.parquet` |

All other parquets (14 league files, 19 college files, coaches/GM/coordinator files, workout data) were generated externally — likely in Google Colab notebooks or Claude sessions. **If these files are lost, they cannot be regenerated from this repo.**

**Action needed:** Save/recreate data pull scripts for every position before moving data out of git.

---

## Production Roadmap

### Phase 0: Safety & Hygiene (immediate)

- [x] Remove sensitive PDF from git history
- [x] Add `.gitignore`
- [ ] Fix filenames: `Safety..py` → `Safety.py`, `4 coaches.py` → `4_Coaches.py`
- [ ] Pin dependency versions in `requirements.txt` (`>=` → `==`)
- [ ] Remove `gm_exploration_output.csv` from repo root
- [ ] Update README.md to match current repo structure
- [ ] Verify community algorithms still work on live after the Supabase URL change — saved slider presets load, save/fork/upvote round-trip, position scoping correct. Local + Cloud now point at the same Supabase project, so behavior should match across both.

### Phase 1: Local Development (week 1)

- [ ] Set up local Python venv + Streamlit dev server (use `make install && make run`)
- [ ] Create `.streamlit/secrets.toml` locally for Supabase credentials
- [ ] Verify app runs locally before pushing changes
- [ ] **Every change gets tested locally before it hits main**

### Phase 2: DRY the Architecture (weeks 2-3)

The 16 page files contain ~7,100 lines with massive duplication. Each page repeats:
- Helper functions (`zscore_to_percentile`, `format_percentile`, etc.) — identical across all
- Tier system functions — identical across all
- Radar chart builder — identical across all
- Rendering logic (sidebar, scoring, table, detail view) — same pattern, different config

**Target architecture:**

```
lib_shared.py             # Already exists — scoring + Supabase
lib_page_helpers.py       # NEW — extract: zscore_to_percentile, format_percentile,
                          #   format_score, sample_size_warning, tier functions,
                          #   build_radar_figure
lib_page_renderer.py      # NEW — generic render_position_page(config) function
                          #   that takes a config dict and renders the full page

pages/QB.py               # AFTER: ~80 lines of config + render_position_page(config)
pages/WR.py               # AFTER: ~80 lines of config + render_position_page(config)
...                       # Same for all 16 pages
```

Each page becomes a config dict:
```python
CONFIG = {
    "position_group": "qb",
    "page_title": "QB Rater",
    "page_url": "https://lions-rater.streamlit.app/QB",
    "data_path": "league_qb_all_seasons.parquet",
    "metadata_path": "qb_stat_metadata.json",
    "raw_col_map": { ... },
    "bundles": { ... },
    "default_bundle_weights": { ... },
    "radar_stats": [ ... ],
}
```

**How to do this safely:**
1. Extract helpers into `lib_page_helpers.py` first (zero behavior change)
2. Write tests for the helpers (Phase 4 scaffold exists)
3. Build `lib_page_renderer.py` against ONE position (e.g., WR)
4. Verify it matches the existing WR page output
5. Convert remaining pages one at a time

### Phase 3: Data Pipeline (weeks 3-4)

- [x] Unified pipeline framework: `python tools/data_pull.py --position wr --seasons 2016-2025`
- [x] WR position config (fully working, translated from `tools/wr_data_pull.py`)
- [x] RB position config (stubbed from `tools/rb_data_pull.py`, needs testing)
- [x] Disk-cached nflverse pulls (`.data_cache/`, 7-day TTL)
- [x] Output validation: schema, z-score distributions, required columns
- [x] Template for adding new positions (`tools/pipeline/positions/_template.py`)
- [ ] Write position configs for remaining positions (QB, OL, DE, DT, LB, CB, S, K, P)
- [ ] Move parquets to external storage (Git LFS, S3, or Supabase Storage)
- [ ] Clean git history of duplicate parquet blobs (reduces repo from 30MB+ to ~5MB)

**Pipeline architecture:**
```
tools/
  data_pull.py                    # CLI: --position, --seasons, --dry-run
  pipeline/
    base.py                       # PositionConfig dataclass
    sources.py                    # Cached nflverse data pulls
    population.py                 # Snap filtering, top-N selection
    zscore.py                     # Shared z-score engine
    output.py                     # Validation + parquet/metadata writer
    runner.py                     # Season-loop orchestrator
    positions/
      __init__.py                 # Registry: POSITIONS = {"wr": WR_CONFIG, ...}
      wr.py                       # WR+TE config (working)
      rb.py                       # RB config (stubbed)
      _template.py                # Annotated template for new positions
```

### Phase 4: Testing & CI (week 4+)

Test scaffold is in `tests/`. Run with `make test`.

**Priority tests:**
1. **Scoring math** — `score_players()` produces correct weighted averages
2. **Weight computation** — `compute_effective_weights()` converts bundles correctly
3. **Data integrity** — all parquets load, have expected columns, no NaN-only columns
4. **Metadata consistency** — every z-col in RAW_COL_MAP has a matching raw column in the data
5. **Page smoke tests** — each page module imports without error

**CI pipeline (GitHub Actions):**
```yaml
on: push
jobs:
  test:
    - pip install -r requirements.txt -r requirements-dev.txt
    - make lint
    - make test
```

### Phase 5: Deploy Pipeline (week 5+)

- [ ] Create `staging` branch — PRs merge to staging, staging deploys to a preview URL
- [ ] `main` branch = production (Streamlit Cloud auto-deploys)
- [ ] Add branch protection: require passing CI before merge to main
- [ ] Document the deploy process: PR → staging → test → merge to main → auto-deploy

---

## Commands Reference

| Command | What it does |
|---------|-------------|
| `make install` | Create venv, install dependencies |
| `make run` | Run Streamlit locally on port 8501 |
| `make test` | Run pytest test suite |
| `make lint` | Run ruff linter |
| `make data-refresh` | Pull fresh data (requires nflreadpy) |
| `make clean` | Remove venv, caches, temp files |
