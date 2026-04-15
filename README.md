# Lions Rater

A transparent, fan-built alternative to proprietary football rating services like PFF. The goal is a **Wikipedia of athletic performance**: a place where fans build their own player rating methodologies, share them, and argue about them in the open — with every stat, formula, and known weakness on display.

**Live app:** [lions-rater.streamlit.app](https://lions-rater.streamlit.app)

> **Ethos:** Strive for the greatest accuracy while being transparent about limitations and how we addressed them.

## What it is

Lions Rater is a Streamlit app focused on the 2024 Detroit Lions. For each position it offers a rating page where users:

- Weight what they care about with plain-English sliders (or drop into Advanced mode for per-stat control)
- See Lions players re-rank in real time as they adjust weights
- Save their rating methodology as a named "algorithm" and share a link to it
- Browse, fork, and upvote algorithms other users have built

Three position pages are currently live:

- **Receivers** — WR and TE, 11 efficiency and volume stats
- **Running Backs** — 19 stats covering efficiency, tackle-breaking, explosiveness, volume, receiving, and short-yardage
- **Offensive Line** — the newest page, featuring a four-tier "epistemological" system that labels each stat by how much trust it asks from the user

## The tier system

Every stat on the Offensive Line page is classified into one of four tiers:

- **🟢 Tier 1 — Counted.** Pure recorded facts. No modeling. Example: total snaps played.
- **🔵 Tier 2 — Contextualized.** Counts divided by opportunity. Still no modeling. Example: penalty rate.
- **🟡 Tier 3 — Adjusted.** Compared against a modeled baseline. Example: Gap-Adjusted Run Success Rate.
- **🟠 Tier 4 — Inferred.** Speculative stats the data can't directly see. Example: Mobility Index (a proxy for guard pulling effectiveness).

Users filter by tier with checkboxes at the top of the page. Tier 4 is off by default. Every stat exposes a methodology popover in Advanced mode showing **what** it measures, **how** it's computed, and its **known limits**. Philosophy in a checkbox.

The tier system will eventually roll out to the Receivers and Running Backs pages as well.

## How it works

- **Data:** Pulled from [nflverse](https://github.com/nflverse) play-by-play, snap counts, PFR advanced stats, and NFL Next Gen Stats. FTN charting data is used via nflverse (CC-BY-SA 4.0).
- **Scoring:** Every stat is z-scored against a league-wide reference population (top 64 WR + top 32 TE for receivers; top 32 RBs for running backs; min 6 games played for both). The final score for a player is a weighted average of their z-scores, where the weights come from the user's slider positions.
- **Reference vs. output populations:** The reference pool is used only to compute each stat's league mean and standard deviation. The output parquet then contains **every Lions player with at least one offensive snap**, each scored against that league baseline. A backup TE with six targets shows up on the page with honestly noisy (but correctly computed) league-wide z-scores.
- **Community:** Saved algorithms live in a Supabase table with a `position_group` column for scoping. Each page pulls only its own position's algorithms.
- **Architecture:** `app.py` is a minimal landing page. `pages/1_Receivers.py`, `pages/2_Running_backs.py`, and `pages/3_Offensive_Line.py` are the real rating pages, auto-discovered by Streamlit's multi-page navigation. `lib_shared.py` contains the scoring logic and community-features UI used by all of the position pages.

## Repo layout

```
Lions_rater/
├── app.py                       # Landing page
├── lib_shared.py                # Shared Supabase / scoring / UI helpers
├── requirements.txt
├── pages/
│   ├── 1_Receivers.py
│   ├── 2_Running_backs.py
│   └── 3_Offensive_Line.py
├── data/
│   ├── master_lions_with_z.parquet         # Receivers
│   ├── master_lions_rbs_with_z.parquet     # Running backs
│   ├── master_lions_ol_with_z.parquet      # Offensive line
│   ├── wr_stat_metadata.json               # Tier classifications + methodology
│   ├── rb_stat_metadata.json               # Tier classifications + methodology
│   └── ol_stat_metadata.json               # Tier classifications + methodology
├── ol_data_pull_v2.py           # Data pull script for OL (Colab-ready)
├── wr_data_pull_v2.py           # Data pull script for WR/TE (Colab-ready)
└── rb_data_pull_v2.py           # Data pull script for RB (Colab-ready)
```

## Running your own data refresh

The data pull scripts are designed to run in Google Colab. Each script pulls fresh nflverse data, computes z-scores against a league reference population, and writes a parquet + metadata JSON. To refresh any position's data:

1. Open Colab and create a new notebook
2. Paste the contents of the relevant `*_data_pull_v2.py` into a cell
3. Run it (takes 1–3 minutes)
4. Download the generated parquet and JSON from Colab's file browser
5. Upload them to the `data/` folder on this repo, overwriting the existing files
6. Streamlit Cloud auto-redeploys in ~60 seconds

The scripts print clear summaries at the end showing the reference population size, output population size, and a leaderboard preview.

## Status

**Working well:**
- All three position pages render, score, and rank correctly
- League-wide z-scores for WR and RB
- Community save/browse/fork/upvote works across all three positions (scoped by `position_group`)
- Offensive Line page has the full tier system, methodology popovers, team context banner, and score explainer

**Known rough edges (being addressed):**
- Offensive Line z-scores are still computed within the Lions starting five, not league-wide. This means a Lions lineman's "+1.0" on a stat currently means "one SD above the other four Lions starters," not "one SD above the NFL starting-OL population." Fixing this requires rewriting the OL pull to include league-wide reference data.
- Receivers and Running Backs pages don't yet have the tier system, methodology popovers, or score labels that OL has. Migration is planned.
- Players with very small samples (a few targets / carries) show honestly noisy scores. Pages now warn users about this above the leaderboard.
- First-load scores sometimes show as zero until the user touches a slider — a cosmetic Streamlit session-state quirk that resolves on any interaction.

**On deck (rough order):**
1. Tier migration for Receivers and Running Backs
2. "Show me the math" feature — click any player's score and see exact per-stat contributions
3. League-wide z-scores for Offensive Line
4. Additional positions (QB is the natural next)

## Data credits

- Play-by-play, snap counts, NGS data, and roster data via [nflverse](https://github.com/nflverse)
- FTN charting data via [FTN Data](https://ftndata.com) / nflverse (CC-BY-SA 4.0)
- This is a fan project, not affiliated with the NFL or the Detroit Lions

## License

No license declared yet. Treat as "source-visible, please don't redistribute commercially" until something more formal is decided.
