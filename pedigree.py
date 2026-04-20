"""
pedigree.py — Player pedigree / validation chain scoring.
Place in repo root alongside lib_shared.py.

Computes a "pedigree score" that measures how consistently a player's
quality has been validated across independent checkpoints:

College chain:
  1. HS recruiting (stars, rating, national ranking)
  2. College production (z-score at position)
  3. Transfer destination quality (SP+ of destination school)
  4. Post-transfer production (did they produce at new school?)

NFL chain:
  5. Draft capital (round, overall pick)
  6. Rookie production (first-year z-score)
  7. Career trajectory (improving or declining?)

Each checkpoint scores 0-100. The pedigree score is a weighted composite.
"Confirmation" bonuses are awarded when consecutive checkpoints agree.
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
# CHECKPOINT SCORERS (each returns 0-100)
# ============================================================

def score_recruiting(stars, rating, ranking):
    """Score HS recruiting pedigree (0-100)."""
    scores = []
    if pd.notna(stars):
        star_map = {5: 95, 4: 75, 3: 50, 2: 25, 1: 10}
        scores.append(star_map.get(int(stars), 30))
    if pd.notna(rating) and rating > 0:
        # Rating is 0-1 scale, ~0.9997 = #1 overall
        scores.append(min(100, rating * 105))
    if pd.notna(ranking) and ranking > 0:
        # National ranking: #1 = 100, #100 = 60, #300 = 40, #1000+ = 20
        if ranking <= 10: scores.append(98)
        elif ranking <= 50: scores.append(90)
        elif ranking <= 100: scores.append(80)
        elif ranking <= 200: scores.append(65)
        elif ranking <= 500: scores.append(50)
        elif ranking <= 1000: scores.append(35)
        else: scores.append(20)
    return np.mean(scores) if scores else None


def score_college_production(composite_z):
    """Score college production from z-score (0-100)."""
    if pd.isna(composite_z): return None
    pct = zscore_to_percentile(composite_z)
    return pct


def score_draft_capital(draft_round, overall_pick):
    """Score NFL draft position (0-100)."""
    if pd.isna(draft_round) and pd.isna(overall_pick): return None
    if pd.notna(overall_pick):
        pick = int(overall_pick)
        if pick <= 5: return 98
        if pick <= 15: return 92
        if pick <= 32: return 85
        if pick <= 64: return 75
        if pick <= 100: return 65
        if pick <= 150: return 50
        if pick <= 200: return 35
        return 25
    if pd.notna(draft_round):
        rd = int(draft_round)
        round_map = {1: 90, 2: 75, 3: 65, 4: 55, 5: 45, 6: 35, 7: 25}
        return round_map.get(rd, 20)
    return None


def score_nfl_production(composite_z):
    """Score NFL production from z-score (0-100)."""
    if pd.isna(composite_z): return None
    return zscore_to_percentile(composite_z)


def score_trajectory(z_scores_by_year):
    """Score career trajectory — are they improving? (0-100).
    z_scores_by_year: list of (season, composite_z) tuples, sorted by season."""
    valid = [(s, z) for s, z in z_scores_by_year if pd.notna(z)]
    if len(valid) < 2: return None
    # Simple: compare last season to first season
    first_z = valid[0][1]
    last_z = valid[-1][1]
    delta = last_z - first_z
    # Also compute trend via simple linear regression
    years = [s for s, _ in valid]
    zs = [z for _, z in valid]
    if len(set(years)) < 2: return None
    slope = np.polyfit(years, zs, 1)[0]
    # Slope of +0.5 per year = strong improvement = 90
    # Slope of 0 = flat = 50
    # Slope of -0.5 = declining = 10
    trajectory_score = 50 + (slope * 80)
    return max(0, min(100, trajectory_score))


# ============================================================
# CONFIRMATION BONUS
# ============================================================
def compute_confirmation_bonus(checkpoints):
    """Award bonus when consecutive checkpoints agree.
    checkpoints: list of (name, score) where score is 0-100 or None.
    Returns bonus points (0-20)."""
    valid = [(name, score) for name, score in checkpoints if score is not None]
    if len(valid) < 2: return 0

    bonus = 0
    for i in range(len(valid) - 1):
        curr_name, curr_score = valid[i]
        next_name, next_score = valid[i + 1]
        # Both above average (>50) = confirmation
        if curr_score >= 50 and next_score >= 50:
            # Stronger confirmation when both are high
            avg = (curr_score + next_score) / 2
            if avg >= 80: bonus += 5  # strong confirmation
            elif avg >= 60: bonus += 3  # moderate confirmation
            else: bonus += 1  # mild confirmation
        # Both below average = consistent but negative
        elif curr_score < 40 and next_score < 40:
            bonus += 0  # no bonus for consistent underperformance
        # Contradiction (one high, one low) = no bonus
        # Big jump up = "breakout" bonus
        if next_score is not None and curr_score is not None:
            if next_score - curr_score > 30:
                bonus += 2  # breakout bonus

    return min(20, bonus)


# ============================================================
# PEDIGREE LABELS
# ============================================================
def pedigree_label(score):
    """Convert pedigree score to human-readable label."""
    if score is None: return "—"
    if score >= 90: return "Elite pedigree"
    if score >= 75: return "Strong pedigree"
    if score >= 60: return "Above average"
    if score >= 45: return "Average"
    if score >= 30: return "Below average"
    return "Unproven"


def pedigree_color(score):
    if score is None: return "#ccc"
    if score >= 75: return "#0076B6"
    if score >= 60: return "#4CAF50"
    if score >= 45: return "#FF9800"
    return "#F44336"


# ============================================================
# MAIN PEDIGREE CALCULATOR
# ============================================================
def compute_pedigree(player_name, college_seasons_data=None, recruiting_info=None,
                     draft_info=None, nfl_seasons_data=None, college_z_cols=None,
                     nfl_z_cols=None):
    """Compute full pedigree score for a player.
    
    Args:
        player_name: str
        college_seasons_data: list of dicts with {season, team, composite_z, conference}
        recruiting_info: dict with {stars, rating, ranking}
        draft_info: dict with {round, overall, nfl_team}
        nfl_seasons_data: list of dicts with {season, team, composite_z}
        college_z_cols: list of z-score column names (for reference)
        nfl_z_cols: list of z-score column names (for reference)
    
    Returns: dict with checkpoint scores, confirmation bonus, and total pedigree score
    """
    checkpoints = []

    # 1. Recruiting
    rec_score = None
    if recruiting_info:
        rec_score = score_recruiting(
            recruiting_info.get("stars"),
            recruiting_info.get("rating"),
            recruiting_info.get("ranking"),
        )
    checkpoints.append(("HS Recruiting", rec_score))

    # 2. Best college production
    best_college_z = None
    if college_seasons_data:
        college_zs = [s.get("composite_z") for s in college_seasons_data if pd.notna(s.get("composite_z"))]
        if college_zs:
            best_college_z = max(college_zs)  # peak season
    college_prod_score = score_college_production(best_college_z)
    checkpoints.append(("College peak", college_prod_score))

    # 3. College trajectory
    college_traj_score = None
    if college_seasons_data and len(college_seasons_data) >= 2:
        college_traj_data = [(s["season"], s.get("composite_z")) for s in college_seasons_data]
        college_traj_score = score_trajectory(college_traj_data)
    checkpoints.append(("College trajectory", college_traj_score))

    # 4. Draft capital
    draft_score = None
    if draft_info:
        draft_score = score_draft_capital(
            draft_info.get("round"),
            draft_info.get("overall"),
        )
    checkpoints.append(("Draft capital", draft_score))

    # 5. NFL rookie production
    rookie_score = None
    if nfl_seasons_data and len(nfl_seasons_data) > 0:
        rookie_z = nfl_seasons_data[0].get("composite_z")
        rookie_score = score_nfl_production(rookie_z)
    checkpoints.append(("NFL rookie", rookie_score))

    # 6. NFL peak
    nfl_peak_score = None
    if nfl_seasons_data:
        nfl_zs = [s.get("composite_z") for s in nfl_seasons_data if pd.notna(s.get("composite_z"))]
        if nfl_zs:
            nfl_peak_score = score_nfl_production(max(nfl_zs))
    checkpoints.append(("NFL peak", nfl_peak_score))

    # 7. NFL trajectory
    nfl_traj_score = None
    if nfl_seasons_data and len(nfl_seasons_data) >= 2:
        nfl_traj_data = [(s["season"], s.get("composite_z")) for s in nfl_seasons_data]
        nfl_traj_score = score_trajectory(nfl_traj_data)
    checkpoints.append(("NFL trajectory", nfl_traj_score))

    # Confirmation bonus
    bonus = compute_confirmation_bonus(checkpoints)

    # Compute weighted average
    weights = {
        "HS Recruiting": 10,
        "College peak": 20,
        "College trajectory": 10,
        "Draft capital": 15,
        "NFL rookie": 15,
        "NFL peak": 20,
        "NFL trajectory": 10,
    }

    weighted_sum = 0
    weight_total = 0
    for name, score in checkpoints:
        if score is not None:
            w = weights.get(name, 10)
            weighted_sum += score * w
            weight_total += w

    base_score = (weighted_sum / weight_total) if weight_total > 0 else None

    # ── VALIDATION DEPTH PENALTY ──────────────────────────
    # A player with only recruiting data hasn't proven anything yet.
    # Cap the pedigree score based on how many checkpoints are filled.
    # 1 checkpoint: max 35 (promising but unproven)
    # 2 checkpoints: max 55 (some validation)
    # 3 checkpoints: max 70 (solid validation)
    # 4 checkpoints: max 82 (well validated)
    # 5+ checkpoints: max 100 (fully validated)
    available = sum(1 for _, s in checkpoints if s is not None)
    depth_caps = {1: 35, 2: 55, 3: 70, 4: 82, 5: 92, 6: 97, 7: 100}
    depth_cap = depth_caps.get(available, 100)

    if base_score is not None:
        capped_score = min(base_score, depth_cap)
        total_score = min(100, capped_score + bonus)
    else:
        total_score = None

    # ── DEPTH LABEL ───────────────────────────────────────
    depth_labels = {
        1: "Unproven — recruiting only",
        2: "Early validation",
        3: "Building case",
        4: "Well validated",
        5: "Strongly validated",
        6: "Fully validated",
        7: "Complete pedigree",
    }
    depth_label = depth_labels.get(available, "")

    return {
        "checkpoints": checkpoints,
        "confirmation_bonus": bonus,
        "base_score": base_score,
        "total_score": total_score,
        "label": pedigree_label(total_score),
        "depth_label": depth_label,
        "depth_cap": depth_cap,
        "color": pedigree_color(total_score),
        "available_checkpoints": available,
        "total_checkpoints": len(checkpoints),
    }


# ============================================================
# STREAMLIT DISPLAY
# ============================================================
def render_pedigree(pedigree_result, player_name):
    """Render pedigree score as a visual component in Streamlit."""
    if pedigree_result is None or pedigree_result.get("total_score") is None:
        return

    score = pedigree_result["total_score"]
    label = pedigree_result["label"]
    color = pedigree_result["color"]
    bonus = pedigree_result["confirmation_bonus"]
    available = pedigree_result["available_checkpoints"]
    total = pedigree_result["total_checkpoints"]

    # Header
    st.markdown(
        f"<div style='background:{color};color:white;padding:10px 16px;border-radius:8px;"
        f"margin:8px 0;font-size:1rem;'>"
        f"<strong>Pedigree: {score:.0f}/100</strong> — {label}"
        f" <span style='opacity:0.7;font-size:0.85rem;'>({available}/{total} checkpoints"
        f"{f' · +{bonus} confirmation bonus' if bonus > 0 else ''})</span></div>",
        unsafe_allow_html=True,
    )
    depth_label = pedigree_result.get("depth_label", "")
    depth_cap = pedigree_result.get("depth_cap", 100)
    if depth_label:
        cap_note = f" Score capped at {depth_cap} until more checkpoints are filled." if depth_cap < 100 else ""
        st.caption(f"_{depth_label}.{cap_note} Pedigree measures how consistently independent evaluators have validated this player's quality._")

    # Checkpoint breakdown
    with st.expander("Pedigree breakdown"):
        st.markdown(
            "Each bar shows how this player scored at that checkpoint (0-100). "
            "When consecutive checkpoints both score high, a **confirmation bonus** is awarded — "
            "it means independent evaluators keep agreeing this player is the real deal."
        )
        st.markdown("---")
        for name, checkpoint_score in pedigree_result["checkpoints"]:
            if checkpoint_score is not None:
                bar_width = int(checkpoint_score)
                if checkpoint_score >= 75:
                    bar_color = "#0076B6"
                elif checkpoint_score >= 50:
                    bar_color = "#4CAF50"
                elif checkpoint_score >= 35:
                    bar_color = "#FF9800"
                else:
                    bar_color = "#F44336"

                st.markdown(
                    f"<div style='margin:4px 0;'>"
                    f"<span style='display:inline-block;width:140px;font-size:0.9rem;'>{name}</span>"
                    f"<span style='display:inline-block;width:200px;height:16px;background:#eee;border-radius:8px;overflow:hidden;vertical-align:middle;'>"
                    f"<span style='display:block;width:{bar_width}%;height:100%;background:{bar_color};border-radius:8px;'></span>"
                    f"</span>"
                    f" <span style='font-size:0.85rem;margin-left:8px;'>{checkpoint_score:.0f}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='margin:4px 0;'>"
                    f"<span style='display:inline-block;width:140px;font-size:0.9rem;color:#999;'>{name}</span>"
                    f"<span style='font-size:0.85rem;color:#999;'>No data</span></div>",
                    unsafe_allow_html=True,
                )

        if bonus > 0:
            st.caption(f"Confirmation bonus: +{bonus} — consecutive checkpoints validated this player's trajectory")
