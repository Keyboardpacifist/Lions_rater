"""
comps.py — Statistical comparison engine.
Place in repo root alongside lib_shared.py.

Three modes:
1. College comps: Find historically similar college profiles
2. Pro comps: Find similar NFL profiles
3. College-to-pro prediction: How did similar college profiles perform in the NFL?
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from scipy.stats import norm
from scipy.spatial.distance import cosine

COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"
NFL_DATA_DIR = Path(__file__).resolve().parent / "data"


def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)


# ============================================================
# SIMILARITY ENGINE
# ============================================================
def compute_similarity(player_z_vector, candidate_z_vector):
    """Cosine similarity between two z-score vectors. Returns 0-100 (100 = identical)."""
    valid_mask = np.isfinite(player_z_vector) & np.isfinite(candidate_z_vector)
    if valid_mask.sum() < 3:
        return None
    p = player_z_vector[valid_mask]
    c = candidate_z_vector[valid_mask]
    try:
        sim = 1 - cosine(p, c)
        return max(0, min(100, sim * 100))
    except:
        return None


def find_comps(target_row, pool_df, z_cols, n=10, exclude_name=None, exclude_team=None, exclude_season=None):
    """Find the n most similar players from a pool based on z-score profile.
    
    Args:
        target_row: Series with z-score columns for the target player
        pool_df: DataFrame of candidates
        z_cols: list of z-score column names to compare
        n: number of comps to return
        exclude_name: player name to exclude (self)
        exclude_team: team to exclude (optional)
        exclude_season: season to exclude (optional)
    
    Returns: DataFrame of top comps with similarity scores
    """
    available_z = [c for c in z_cols if c in target_row.index and c in pool_df.columns]
    if len(available_z) < 3:
        return pd.DataFrame()

    target_vec = np.array([target_row.get(c, np.nan) for c in available_z], dtype=float)
    if np.isfinite(target_vec).sum() < 3:
        return pd.DataFrame()

    results = []
    for idx, cand_row in pool_df.iterrows():
        # Skip self
        cand_name = cand_row.get("player", cand_row.get("player_display_name", ""))
        if exclude_name and cand_name == exclude_name:
            cand_team = cand_row.get("team", cand_row.get("recent_team", ""))
            cand_season = cand_row.get("season", cand_row.get("season_year", ""))
            if exclude_team and cand_team == exclude_team:
                if exclude_season and cand_season == exclude_season:
                    continue

        cand_vec = np.array([cand_row.get(c, np.nan) for c in available_z], dtype=float)
        sim = compute_similarity(target_vec, cand_vec)
        if sim is not None:
            results.append({
                "player": cand_name,
                "team": cand_row.get("team", cand_row.get("recent_team", "")),
                "season": cand_row.get("season", cand_row.get("season_year", "")),
                "conference": cand_row.get("conference", ""),
                "similarity": sim,
                "composite_z": cand_row.get("composite_z", np.nan),
                "_idx": idx,
            })

    if not results:
        return pd.DataFrame()

    comp_df = pd.DataFrame(results).sort_values("similarity", ascending=False).head(n)
    return comp_df


# ============================================================
# COLLEGE-TO-PRO PREDICTION
# ============================================================
@st.cache_data
def load_linkage():
    path = COLLEGE_DATA_DIR / "college_to_nfl_linked.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def college_to_pro_prediction(target_row, college_pool_df, z_cols, position_group, n_comps=20):
    """Find similar college profiles and show how they performed in the NFL.
    
    Returns dict with:
        - comps: list of similar college players with their NFL outcomes
        - hit_rate: % who were above average in NFL
        - avg_nfl_z: average NFL composite z-score
        - best_comp: highest NFL performer
        - worst_comp: lowest NFL performer
    """
    linkage = load_linkage()
    if len(linkage) == 0:
        return None

    # Find comps in college pool (use all historical data, not just current team)
    comps = find_comps(target_row, college_pool_df, z_cols, n=n_comps * 3,
                       exclude_name=target_row.get("player"),
                       exclude_team=target_row.get("team"),
                       exclude_season=target_row.get("season", target_row.get("season_year")))

    if len(comps) == 0:
        return None

    # Match comps to NFL data via linkage
    nfl_outcomes = []
    for _, comp in comps.iterrows():
        comp_name = comp["player"]
        comp_team = comp["team"]

        # Find in linkage by name + team
        last = comp_name.split()[-1] if comp_name else ""
        first = comp_name.split()[0] if comp_name else ""
        link_matches = linkage[
            (linkage["player"].str.contains(last, na=False, case=False)) &
            (linkage["player"].str.contains(first, na=False, case=False))
        ]

        if len(link_matches) == 0:
            continue

        # Get NFL info from linkage
        link_row = link_matches.iloc[0]
        draft_round = link_row.get("draft_round")
        draft_overall = link_row.get("draft_overall")
        nfl_team = link_row.get("nfl_team", "")

        # Get NFL z-scores from the linkage data
        nfl_z_cols = [c for c in link_matches.columns if c.endswith("_z")]
        # Use the college composite from the comp, and try to get NFL performance
        # The linkage file has college stats — we need NFL stats from league parquets
        # For now, use the draft info + any NFL z-scores in linkage
        
        nfl_outcomes.append({
            "player": comp_name,
            "college_team": comp_team,
            "college_season": int(comp["season"]) if pd.notna(comp["season"]) else None,
            "college_similarity": comp["similarity"],
            "college_composite_z": comp["composite_z"],
            "draft_round": draft_round,
            "draft_overall": draft_overall,
            "nfl_team": nfl_team,
        })

        if len(nfl_outcomes) >= n_comps:
            break

    if not nfl_outcomes:
        return None

    outcomes_df = pd.DataFrame(nfl_outcomes)

    # Try to get NFL career data for each comp
    nfl_career_scores = []
    for pg_file_map in [("qb", "league_qb_all_seasons.parquet"),
                         ("wr", "league_wr_all_seasons.parquet"),
                         ("te", "league_te_all_seasons.parquet"),
                         ("rb", "league_rb_all_seasons.parquet"),
                         ("de", "league_de_all_seasons.parquet"),
                         ("dt", "league_dt_all_seasons.parquet"),
                         ("lb", "league_lb_all_seasons.parquet"),
                         ("cb", "league_cb_all_seasons.parquet"),
                         ("s", "league_s_all_seasons.parquet")]:
        pg_code, nfl_file = pg_file_map
        if pg_code != position_group:
            continue
        nfl_path = NFL_DATA_DIR / nfl_file
        if not nfl_path.exists():
            continue
        try:
            nfl_df = pd.read_parquet(nfl_path)
            name_col = "player_display_name" if "player_display_name" in nfl_df.columns else "player_name"
            season_col = "season_year" if "season_year" in nfl_df.columns else "season"
            nfl_z = [c for c in nfl_df.columns if c.endswith("_z")]

            for i, outcome in enumerate(nfl_outcomes):
                comp_name = outcome["player"]
                last = comp_name.split()[-1] if comp_name else ""
                first = comp_name.split()[0] if comp_name else ""
                player_nfl = nfl_df[
                    (nfl_df[name_col].str.contains(last, na=False, case=False)) &
                    (nfl_df[name_col].str.contains(first, na=False, case=False))
                ]
                if len(player_nfl) > 0:
                    # Compute average NFL composite across all seasons
                    avg_z_values = []
                    for _, nrow in player_nfl.iterrows():
                        vals = [nrow.get(c) for c in nfl_z if pd.notna(nrow.get(c))]
                        if vals:
                            avg_z_values.append(np.mean(vals))
                    if avg_z_values:
                        nfl_outcomes[i]["nfl_avg_z"] = np.mean(avg_z_values)
                        nfl_outcomes[i]["nfl_peak_z"] = max(avg_z_values)
                        nfl_outcomes[i]["nfl_seasons"] = len(avg_z_values)
        except:
            pass

    outcomes_df = pd.DataFrame(nfl_outcomes)

    # Compute summary stats
    has_nfl = outcomes_df["nfl_avg_z"].notna() if "nfl_avg_z" in outcomes_df.columns else pd.Series([False] * len(outcomes_df))
    nfl_players = outcomes_df[has_nfl]

    if len(nfl_players) == 0:
        return {
            "comps": outcomes_df,
            "n_comps": len(outcomes_df),
            "n_with_nfl": 0,
            "hit_rate": None,
            "avg_nfl_z": None,
            "best_comp": None,
            "worst_comp": None,
        }

    avg_nfl_z = nfl_players["nfl_avg_z"].mean()
    hit_rate = (nfl_players["nfl_avg_z"] > 0).mean() * 100
    best_idx = nfl_players["nfl_avg_z"].idxmax()
    worst_idx = nfl_players["nfl_avg_z"].idxmin()

    return {
        "comps": outcomes_df,
        "n_comps": len(outcomes_df),
        "n_with_nfl": len(nfl_players),
        "hit_rate": hit_rate,
        "avg_nfl_z": avg_nfl_z,
        "best_comp": nfl_players.loc[best_idx].to_dict() if pd.notna(best_idx) else None,
        "worst_comp": nfl_players.loc[worst_idx].to_dict() if pd.notna(worst_idx) else None,
    }


# ============================================================
# STREAMLIT DISPLAY — COLLEGE COMPS
# ============================================================
def render_college_comps(target_row, pool_df, z_cols, pos_label, player_name):
    """Show college statistical comps in an expander."""
    comps = find_comps(target_row, pool_df, z_cols, n=5,
                       exclude_name=player_name,
                       exclude_team=target_row.get("team"),
                       exclude_season=target_row.get("season", target_row.get("season_year")))

    if len(comps) == 0:
        return

    with st.expander(f"📊 Statistical comps (college {pos_label})"):
        st.caption("Players with the most similar college z-score profile across all FBS seasons")
        comp_rows = []
        for _, c in comps.iterrows():
            pct = zscore_to_percentile(c["composite_z"]) if pd.notna(c["composite_z"]) else None
            comp_rows.append({
                "Player": c["player"],
                "Team": c["team"],
                "Season": int(c["season"]) if pd.notna(c["season"]) else "—",
                "Match": f"{c['similarity']:.0f}%",
                "Score": f"{c['composite_z']:+.2f}" if pd.notna(c["composite_z"]) else "—",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ============================================================
# STREAMLIT DISPLAY — COLLEGE-TO-PRO PREDICTION
# ============================================================
def render_college_to_pro(target_row, pool_df, z_cols, position_group, pos_label, player_name):
    """Show college-to-pro prediction in an expander."""
    result = college_to_pro_prediction(target_row, pool_df, z_cols, position_group, n_comps=15)

    if result is None:
        return

    with st.expander(f"🔮 Draft crystal ball — how did similar college profiles do in the NFL?"):
        if result["n_with_nfl"] == 0:
            st.caption(f"Found {result['n_comps']} similar college profiles, but couldn't match them to NFL career data.")
            if len(result["comps"]) > 0:
                st.dataframe(result["comps"][["player", "college_team", "college_season", "college_similarity", "draft_round", "nfl_team"]].rename(
                    columns={"college_similarity": "Match %", "draft_round": "Rd", "college_team": "College", "college_season": "Year", "nfl_team": "NFL Team"}
                ), use_container_width=True, hide_index=True)
            return

        # Summary banner
        hit_rate = result["hit_rate"]
        avg_z = result["avg_nfl_z"]
        n_nfl = result["n_with_nfl"]

        if hit_rate >= 70:
            banner_color = "#0076B6"
            banner_label = "Strong hit rate"
        elif hit_rate >= 50:
            banner_color = "#4CAF50"
            banner_label = "Above average hit rate"
        elif hit_rate >= 30:
            banner_color = "#FF9800"
            banner_label = "Mixed results"
        else:
            banner_color = "#F44336"
            banner_label = "Low hit rate"

        st.markdown(
            f"<div style='background:{banner_color};color:white;padding:10px 16px;border-radius:8px;"
            f"margin:8px 0;font-size:1rem;'>"
            f"<strong>{hit_rate:.0f}% hit rate</strong> — {banner_label}"
            f" <span style='opacity:0.7;font-size:0.85rem;'>"
            f"({n_nfl} similar college profiles tracked into NFL · avg NFL z: {avg_z:+.2f})</span></div>",
            unsafe_allow_html=True,
        )

        # ── NFL OUTCOME TIERS ─────────────────────────────
        NFL_TIERS = [
            ("All-Pro",          1.5,  None, "🏆", "#0076B6"),
            ("High-end starter", 0.75, 1.5,  "⭐", "#2196F3"),
            ("Solid starter",    0.25, 0.75, "✅", "#4CAF50"),
            ("Average starter",  0.0,  0.25, "📊", "#8BC34A"),
            ("Role player",     -0.5,  0.0,  "🔄", "#FF9800"),
            ("Bust",             None, -0.5, "❌", "#F44336"),
        ]

        nfl_comps = result["comps"]
        if "nfl_avg_z" in nfl_comps.columns:
            nfl_with_data = nfl_comps[nfl_comps["nfl_avg_z"].notna()]
            if len(nfl_with_data) > 0:
                st.markdown("**NFL outcome probability based on similar college profiles:**")
                tier_rows = []
                for tier_name, z_min, z_max, icon, color in NFL_TIERS:
                    if z_min is not None and z_max is not None:
                        count = ((nfl_with_data["nfl_avg_z"] >= z_min) & (nfl_with_data["nfl_avg_z"] < z_max)).sum()
                        z_label = f"z: {z_min:+.2f} to {z_max:+.2f}"
                    elif z_min is not None:
                        count = (nfl_with_data["nfl_avg_z"] >= z_min).sum()
                        z_label = f"z: {z_min:+.2f}+"
                    else:
                        count = (nfl_with_data["nfl_avg_z"] < z_max).sum()
                        z_label = f"z: below {z_max:+.2f}"

                    pct = (count / len(nfl_with_data)) * 100
                    # Find example player in this tier
                    if z_min is not None and z_max is not None:
                        tier_players = nfl_with_data[(nfl_with_data["nfl_avg_z"] >= z_min) & (nfl_with_data["nfl_avg_z"] < z_max)]
                    elif z_min is not None:
                        tier_players = nfl_with_data[nfl_with_data["nfl_avg_z"] >= z_min]
                    else:
                        tier_players = nfl_with_data[nfl_with_data["nfl_avg_z"] < z_max]

                    example = tier_players.iloc[0]["player"] if len(tier_players) > 0 else "—"

                    bar_width = max(2, int(pct))
                    tier_rows.append(
                        f"<div style='margin:3px 0;display:flex;align-items:center;'>"
                        f"<span style='width:20px;'>{icon}</span>"
                        f"<span style='width:130px;font-size:0.9rem;'>{tier_name}</span>"
                        f"<span style='width:180px;height:18px;background:#eee;border-radius:9px;overflow:hidden;'>"
                        f"<span style='display:block;width:{bar_width}%;height:100%;background:{color};border-radius:9px;'></span>"
                        f"</span>"
                        f"<span style='width:50px;font-size:0.9rem;margin-left:8px;font-weight:bold;'>{pct:.0f}%</span>"
                        f"<span style='font-size:0.8rem;color:#888;'>{z_label} · {count}/{len(nfl_with_data)}"
                        f"{f' · e.g. {example}' if example != '—' else ''}</span>"
                        f"</div>"
                    )

                st.markdown("".join(tier_rows), unsafe_allow_html=True)
                st.caption("Tiers based on NFL career average composite z-score of statistically similar college players.")
                st.markdown("---")

        st.caption(
            f"Of the {n_nfl} players with the most similar college z-score profiles who made it to the NFL, "
            f"{hit_rate:.0f}% performed above average (z > 0). "
            f"Average NFL composite: {avg_z:+.2f}."
        )

        # Best and worst
        best = result["best_comp"]
        worst = result["worst_comp"]
        if best and worst and best["player"] != worst["player"]:
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown(f"**Best comp:** {best['player']}")
                st.caption(
                    f"{best.get('college_team', '')} → {best.get('nfl_team', '')} "
                    f"(Rd {int(best['draft_round']) if pd.notna(best.get('draft_round')) else '?'}) · "
                    f"NFL avg: {best['nfl_avg_z']:+.2f} over {int(best.get('nfl_seasons', 0))} seasons"
                )
            with bc2:
                st.markdown(f"**Worst comp:** {worst['player']}")
                st.caption(
                    f"{worst.get('college_team', '')} → {worst.get('nfl_team', '')} "
                    f"(Rd {int(worst['draft_round']) if pd.notna(worst.get('draft_round')) else '?'}) · "
                    f"NFL avg: {worst['nfl_avg_z']:+.2f} over {int(worst.get('nfl_seasons', 0))} seasons"
                )

        # Full comp table
        display_cols = ["player", "college_team", "college_season", "college_similarity",
                        "draft_round", "nfl_team"]
        if "nfl_avg_z" in result["comps"].columns:
            display_cols.extend(["nfl_avg_z", "nfl_seasons"])

        display_df = result["comps"][
            [c for c in display_cols if c in result["comps"].columns]
        ].copy()
        rename_map = {
            "player": "Player", "college_team": "College", "college_season": "Year",
            "college_similarity": "Match %", "draft_round": "Rd",
            "nfl_team": "NFL Team", "nfl_avg_z": "NFL Avg Z", "nfl_seasons": "Seasons",
        }
        display_df = display_df.rename(columns=rename_map)
        if "Match %" in display_df.columns:
            display_df["Match %"] = display_df["Match %"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "—")
        if "NFL Avg Z" in display_df.columns:
            display_df["NFL Avg Z"] = display_df["NFL Avg Z"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        if "Rd" in display_df.columns:
            display_df["Rd"] = display_df["Rd"].apply(lambda x: int(x) if pd.notna(x) else "—")
        if "Year" in display_df.columns:
            display_df["Year"] = display_df["Year"].apply(lambda x: int(x) if pd.notna(x) else "—")
        if "Seasons" in display_df.columns:
            display_df["Seasons"] = display_df["Seasons"].apply(lambda x: int(x) if pd.notna(x) else "—")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
