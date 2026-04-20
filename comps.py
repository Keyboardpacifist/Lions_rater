"""
comps.py — Statistical comparison engine (pre-computed lookup).
Uses pre-computed similarity parquets for instant results.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from scipy.stats import norm

COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"


def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)


# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_data
def _load_comps(pos):
    path = COLLEGE_DATA_DIR / f"college_{pos}_comps.parquet"
    if not path.exists(): return pd.DataFrame()
    return pd.read_parquet(path)


# ============================================================
# LOOKUP FUNCTIONS
# ============================================================
def find_comps(player_name, team, season, pos):
    """Look up pre-computed comps for a player-season."""
    comps_df = _load_comps(pos)
    if len(comps_df) == 0: return pd.DataFrame()

    matches = comps_df[
        (comps_df["player"] == player_name) &
        (comps_df["team"] == team) &
        (comps_df["season"] == season)
    ]
    return matches.sort_values("comp_rank")


def find_comps_fuzzy(player_name, team, pos):
    """Fuzzy lookup — try exact, then name+team, then name only."""
    comps_df = _load_comps(pos)
    if len(comps_df) == 0: return pd.DataFrame()

    # Exact name + team
    matches = comps_df[
        (comps_df["player"] == player_name) &
        (comps_df["team"] == team)
    ]
    if len(matches) > 0:
        # Return most recent season's comps
        latest = matches["season"].max()
        return matches[matches["season"] == latest].sort_values("comp_rank")

    # Name only
    matches = comps_df[comps_df["player"] == player_name]
    if len(matches) > 0:
        latest = matches["season"].max()
        return matches[matches["season"] == latest].sort_values("comp_rank")

    return pd.DataFrame()


# ============================================================
# NFL OUTCOME TIERS
# ============================================================
NFL_TIERS = [
    ("All-Pro",          1.5,  None, "🏆", "#0076B6"),
    ("High-end starter", 0.75, 1.5,  "⭐", "#2196F3"),
    ("Solid starter",    0.25, 0.75, "✅", "#4CAF50"),
    ("Average starter",  0.0,  0.25, "📊", "#8BC34A"),
    ("Role player",     -0.5,  0.0,  "🔄", "#FF9800"),
    ("Bust",             None, -0.5, "❌", "#F44336"),
]


def compute_tier_probs(comps_df):
    """Compute probability of each NFL outcome tier from comps."""
    if "comp_nfl_avg_z" not in comps_df.columns:
        return None
    nfl_comps = comps_df[comps_df["comp_nfl_avg_z"].notna()]
    if len(nfl_comps) == 0:
        return None

    tiers = []
    for tier_name, z_min, z_max, icon, color in NFL_TIERS:
        if z_min is not None and z_max is not None:
            mask = (nfl_comps["comp_nfl_avg_z"] >= z_min) & (nfl_comps["comp_nfl_avg_z"] < z_max)
        elif z_min is not None:
            mask = nfl_comps["comp_nfl_avg_z"] >= z_min
        else:
            mask = nfl_comps["comp_nfl_avg_z"] < z_max

        count = mask.sum()
        pct = (count / len(nfl_comps)) * 100
        examples = nfl_comps[mask]["comp_player"].tolist()

        z_label = f"z: {z_min:+.2f}+" if z_max is None else (f"z: below {z_max:+.2f}" if z_min is None else f"z: {z_min:+.2f} to {z_max:+.2f}")

        tiers.append({
            "tier": tier_name, "icon": icon, "color": color,
            "count": count, "total": len(nfl_comps), "pct": pct,
            "z_label": z_label, "examples": examples[:2],
        })
    return tiers


# ============================================================
# STREAMLIT DISPLAY — COLLEGE COMPS
# ============================================================
def render_college_comps(player_name, team, season, pos, pos_label):
    """Show college statistical comps from pre-computed data."""
    comps = find_comps(player_name, team, season, pos)
    if len(comps) == 0:
        comps = find_comps_fuzzy(player_name, team, pos)
    if len(comps) == 0:
        return

    top5 = comps.head(5)
    with st.expander(f"📊 Statistical comps (college {pos_label})"):
        st.caption("Players with the most similar college z-score profile (conference-adjusted)")
        rows = []
        for _, c in top5.iterrows():
            rows.append({
                "Player": c["comp_player"],
                "Team": c["comp_team"],
                "Season": int(c["comp_season"]) if pd.notna(c["comp_season"]) else "—",
                "Conf": c.get("comp_conference", ""),
                "Match": f"{c['comp_similarity']:.0f}%",
                "Score": f"{c['comp_composite_z']:+.2f}" if pd.notna(c.get("comp_composite_z")) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# STREAMLIT DISPLAY — COLLEGE-TO-PRO PREDICTION
# ============================================================
def render_college_to_pro(player_name, team, season, pos, pos_label):
    """Show college-to-pro prediction from pre-computed comps."""
    comps = find_comps(player_name, team, season, pos)
    if len(comps) == 0:
        comps = find_comps_fuzzy(player_name, team, pos)
    if len(comps) == 0:
        return

    has_nfl_col = "comp_nfl_avg_z" in comps.columns
    if not has_nfl_col:
        return

    nfl_comps = comps[comps["comp_nfl_avg_z"].notna()]
    if len(nfl_comps) == 0:
        return

    with st.expander(f"🔮 Draft crystal ball — how did similar college profiles do in the NFL?"):
        n_nfl = len(nfl_comps)
        avg_z = nfl_comps["comp_nfl_avg_z"].mean()
        hit_rate = (nfl_comps["comp_nfl_avg_z"] > 0).mean() * 100

        if hit_rate >= 70: banner_color, banner_label = "#0076B6", "Strong hit rate"
        elif hit_rate >= 50: banner_color, banner_label = "#4CAF50", "Above average hit rate"
        elif hit_rate >= 30: banner_color, banner_label = "#FF9800", "Mixed results"
        else: banner_color, banner_label = "#F44336", "Low hit rate"

        st.markdown(
            f"<div style='background:{banner_color};color:white;padding:10px 16px;border-radius:8px;"
            f"margin:8px 0;font-size:1rem;'>"
            f"<strong>{hit_rate:.0f}% hit rate</strong> — {banner_label}"
            f" <span style='opacity:0.7;font-size:0.85rem;'>"
            f"({n_nfl} similar profiles tracked into NFL · avg z: {avg_z:+.2f})</span></div>",
            unsafe_allow_html=True,
        )

        # Tier probabilities
        tiers = compute_tier_probs(comps)
        if tiers:
            st.markdown("**NFL outcome probability based on similar college profiles:**")
            tier_html = []
            for t in tiers:
                bar_w = max(2, int(t["pct"]))
                ex_str = f" · e.g. {', '.join(t['examples'])}" if t["examples"] else ""
                tier_html.append(
                    f"<div style='margin:3px 0;display:flex;align-items:center;'>"
                    f"<span style='width:20px;'>{t['icon']}</span>"
                    f"<span style='width:130px;font-size:0.9rem;'>{t['tier']}</span>"
                    f"<span style='width:180px;height:18px;background:#eee;border-radius:9px;overflow:hidden;'>"
                    f"<span style='display:block;width:{bar_w}%;height:100%;background:{t['color']};border-radius:9px;'></span>"
                    f"</span>"
                    f"<span style='width:50px;font-size:0.9rem;margin-left:8px;font-weight:bold;'>{t['pct']:.0f}%</span>"
                    f"<span style='font-size:0.8rem;color:#888;'>{t['z_label']} · {t['count']}/{t['total']}{ex_str}</span>"
                    f"</div>"
                )
            st.markdown("".join(tier_html), unsafe_allow_html=True)
            st.caption("Tiers based on NFL career average composite z-score of statistically similar college players (conference-adjusted).")
            st.markdown("---")

        # Best and worst
        best_idx = nfl_comps["comp_nfl_avg_z"].idxmax()
        worst_idx = nfl_comps["comp_nfl_avg_z"].idxmin()
        best = nfl_comps.loc[best_idx]
        worst = nfl_comps.loc[worst_idx]

        if best["comp_player"] != worst["comp_player"]:
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown(f"**Best comp:** {best['comp_player']}")
                rd = int(best["comp_draft_round"]) if pd.notna(best.get("comp_draft_round")) else "?"
                st.caption(f"{best['comp_team']} → {best.get('comp_nfl_team','')} (Rd {rd}) · NFL avg: {best['comp_nfl_avg_z']:+.2f} over {int(best.get('comp_nfl_seasons',0))} seasons")
            with bc2:
                st.markdown(f"**Worst comp:** {worst['comp_player']}")
                rd = int(worst["comp_draft_round"]) if pd.notna(worst.get("comp_draft_round")) else "?"
                st.caption(f"{worst['comp_team']} → {worst.get('comp_nfl_team','')} (Rd {rd}) · NFL avg: {worst['comp_nfl_avg_z']:+.2f} over {int(worst.get('comp_nfl_seasons',0))} seasons")

        # Full table
        display = nfl_comps[["comp_player","comp_team","comp_season","comp_similarity","comp_nfl_avg_z","comp_nfl_seasons"]].copy()
        display.columns = ["Player","College","Year","Match %","NFL Avg Z","Seasons"]
        display["Match %"] = display["Match %"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "—")
        display["NFL Avg Z"] = display["NFL Avg Z"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        display["Year"] = display["Year"].apply(lambda x: int(x) if pd.notna(x) else "—")
        display["Seasons"] = display["Seasons"].apply(lambda x: int(x) if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)
