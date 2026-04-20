"""
NFL Rater — Landing page with NFL / College toggle
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

st.set_page_config(page_title="NFL Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"

# ── Mode toggle ───────────────────────────────────────────────
col_title, col_toggle = st.columns([3, 2])
with col_title:
    st.markdown("<h1 style='margin:0; padding:4px 0;'>🏈 NFL Rater</h1>", unsafe_allow_html=True)
with col_toggle:
    mode = st.radio("Mode", ["NFL", "College"], horizontal=True, key="mode_toggle", label_visibility="collapsed")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# NFL MODE
# ══════════════════════════════════════════════════════════════
if mode == "NFL":
    from team_selector import get_team_and_season, NFL_TEAMS
    
    # Hide the title we already showed and use team selector for dropdowns only
    if "selected_team" not in st.session_state:
        st.session_state.selected_team = "DET"
    if "selected_season" not in st.session_state:
        st.session_state.selected_season = 2025

    team_options = sorted(NFL_TEAMS.keys())
    team_labels = [f"{abbr} — {NFL_TEAMS[abbr]}" for abbr in team_options]
    current_idx = team_options.index(st.session_state.selected_team) if st.session_state.selected_team in team_options else 0

    AVAILABLE_SEASONS = list(range(2025, 2015, -1))

    col_team, col_season, col_spacer = st.columns([2, 1, 3])
    with col_team:
        selected_label = st.selectbox("Team", options=team_labels, index=current_idx,
                                       key="landing_team", label_visibility="collapsed")
    with col_season:
        selected_season = st.selectbox("Season", options=AVAILABLE_SEASONS,
                                        index=0, key="landing_season", label_visibility="collapsed")

    selected_team = selected_label.split(" — ")[0]
    st.session_state.selected_team = selected_team
    st.session_state.selected_season = selected_season
    team_name = NFL_TEAMS.get(selected_team, selected_team)

    st.markdown(
        f"Pick a position from the sidebar to see how the **{selected_season} {team_name}** "
        f"stack up against every player in the league."
    )

    st.divider()

    st.markdown("### Pick a position")
    st.markdown(
        """
**Offense:** QB · WR · TE · RB · OL

**Defense:** DE · DT · LB · CB · S

**Special teams:** Kicker · Punter

**Front office:** Coaches · OC · DC · GM
"""
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Our ethos")
        st.markdown("Every stat on every page has its formula, its data source, and its known weaknesses on display. If something can't be measured honestly from the data we have, we say so.")
    with col2:
        st.markdown("### Why this exists")
        st.markdown("Free data, open methodology, community-built. No grade is final, no stat is beyond questioning.")

# ══════════════════════════════════════════════════════════════
# COLLEGE MODE
# ══════════════════════════════════════════════════════════════
else:
    def zscore_to_percentile(z):
        if pd.isna(z): return None
        return float(norm.cdf(z) * 100)

    def format_percentile(pct):
        if pct is None or pd.isna(pct): return "—"
        if pct >= 99: return "top 1%"
        if pct >= 50: return f"top {100 - int(pct)}%"
        return f"bottom {int(pct)}%"

    # Load all college data to get school list
    @st.cache_data
    def get_school_list():
        schools = set()
        for fname in ["college_qb_all_seasons.parquet", "college_wr_all_seasons.parquet",
                      "college_te_all_seasons.parquet", "college_rb_all_seasons.parquet",
                      "college_def_all_seasons.parquet"]:
            path = COLLEGE_DATA_DIR / fname
            if path.exists():
                df = pd.read_parquet(path, columns=["team"])
                schools.update(df["team"].dropna().unique())
        return sorted(schools)

    @st.cache_data
    def load_college_position(fname):
        path = COLLEGE_DATA_DIR / fname
        if not path.exists(): return pd.DataFrame()
        return pd.read_parquet(path)

    schools = get_school_list()
    COLLEGE_SEASONS = list(range(2025, 2013, -1))

    col_school, col_season, col_spacer = st.columns([2, 1, 3])
    with col_school:
        selected_school = st.selectbox("School", options=schools,
                                        index=schools.index("Michigan") if "Michigan" in schools else 0,
                                        key="college_school", label_visibility="collapsed")
    with col_season:
        selected_college_season = st.selectbox("Season", options=COLLEGE_SEASONS,
                                               index=0, key="college_season_landing",
                                               label_visibility="collapsed")

    st.markdown(f"### {selected_school} — {selected_college_season}")
    st.caption(f"Every player z-scored against all FBS players at their position that season")

    # ── Load and filter data ──────────────────────────────
    POSITION_FILES = {
        "QB": ("college_qb_all_seasons.parquet", ["completion_pct_z", "td_rate_z", "int_rate_z", "yards_per_attempt_z", "pass_tds_z", "rush_yards_total_z"],
               {"completion_pct_z": "Comp %", "td_rate_z": "TD rate", "int_rate_z": "INT rate", "yards_per_attempt_z": "Yds/att", "pass_tds_z": "Pass TDs", "rush_yards_total_z": "Rush yds"},
               {"int_rate_z"}),
        "WR": ("college_wr_all_seasons.parquet", ["rec_yards_total_z", "rec_tds_total_z", "receptions_total_z", "yards_per_rec_z"],
               {"rec_yards_total_z": "Rec yds", "rec_tds_total_z": "Rec TDs", "receptions_total_z": "Receptions", "yards_per_rec_z": "Yds/rec"},
               set()),
        "TE": ("college_te_all_seasons.parquet", ["rec_yards_total_z", "rec_tds_total_z", "receptions_total_z", "yards_per_rec_z"],
               {"rec_yards_total_z": "Rec yds", "rec_tds_total_z": "Rec TDs", "receptions_total_z": "Receptions", "yards_per_rec_z": "Yds/rec"},
               set()),
        "RB": ("college_rb_all_seasons.parquet", ["rush_yards_total_z", "rush_tds_total_z", "yards_per_carry_z", "total_yards_z", "receptions_total_z"],
               {"rush_yards_total_z": "Rush yds", "rush_tds_total_z": "Rush TDs", "yards_per_carry_z": "Yds/carry", "total_yards_z": "Total yds", "receptions_total_z": "Receptions"},
               set()),
        "DEF": ("college_def_all_seasons.parquet", ["tackles_per_game_z", "sacks_per_game_z", "tfl_per_game_z", "pd_per_game_z", "int_per_game_z", "pressure_rate_z"],
                {"tackles_per_game_z": "Tackles/gm", "sacks_per_game_z": "Sacks/gm", "tfl_per_game_z": "TFL/gm", "pd_per_game_z": "PD/gm", "int_per_game_z": "INT/gm", "pressure_rate_z": "Pressure rate"},
                set()),
    }

    for pos_name, (fname, z_cols, labels, invert) in POSITION_FILES.items():
        df = load_college_position(fname)
        if len(df) == 0:
            continue

        season_col = "season" if "season" in df.columns else "season_year"
        filtered = df[(df["team"] == selected_school) & (df[season_col] == selected_college_season)].copy()

        if len(filtered) == 0:
            continue

        # Compute composite score
        available_z = [c for c in z_cols if c in filtered.columns]
        if available_z:
            filtered["composite_z"] = filtered[available_z].mean(axis=1)
            filtered = filtered.sort_values("composite_z", ascending=False)

        # Position header
        if pos_name == "DEF":
            pos_display = "Defense"
            pos_col = "pos_group" if "pos_group" in filtered.columns else None
        else:
            pos_display = {"QB": "Quarterbacks", "WR": "Wide Receivers", "TE": "Tight Ends", "RB": "Running Backs"}[pos_name]
            pos_col = None

        st.markdown(f"#### {pos_display}")

        # Build display table
        display_rows = []
        for _, row in filtered.iterrows():
            name = row.get("player", "—")
            comp = row.get("composite_z", np.nan)
            pct = zscore_to_percentile(comp)

            entry = {"Player": name}

            if pos_col and pd.notna(row.get(pos_col)):
                entry["Pos"] = row[pos_col]

            if pd.notna(comp):
                sign = "+" if comp >= 0 else ""
                entry["Score"] = f"{sign}{comp:.2f}"
                entry["Percentile"] = format_percentile(pct)
            else:
                entry["Score"] = "—"
                entry["Percentile"] = "—"

            # Add key raw stats
            for z_col, label in labels.items():
                raw_col = z_col.replace("_z", "")
                raw = row.get(raw_col)
                if pd.notna(raw):
                    if "rate" in raw_col or "pct" in raw_col:
                        entry[label] = f"{raw:.1%}" if raw < 1 else f"{raw:.1f}%"
                    else:
                        entry[label] = f"{raw:.1f}" if isinstance(raw, float) else str(int(raw))
                else:
                    entry[label] = "—"

            display_rows.append(entry)

        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        # Clickable player expanders with full profile
        st.caption("Click a player below for their full z-score profile and radar chart")
        for _, row in filtered.iterrows():
            name = row.get("player", "—")
            comp = row.get("composite_z", np.nan)
            pos_tag = f" ({row[pos_col]})" if pos_col and pd.notna(row.get(pos_col)) else ""
            score_tag = f" · {comp:+.2f}" if pd.notna(comp) else ""

            with st.expander(f"**{name}**{pos_tag}{score_tag}"):
                pc1, pc2 = st.columns([1, 1])

                with pc1:
                    # Stat breakdown table
                    stat_rows = []
                    for z_col, label in labels.items():
                        if z_col not in row.index: continue
                        z = row.get(z_col)
                        if pd.isna(z): continue
                        raw_col = z_col.replace("_z", "")
                        raw = row.get(raw_col)
                        p = zscore_to_percentile(z)
                        stat_rows.append({
                            "Stat": label,
                            "Value": f"{raw:.2f}" if pd.notna(raw) else "—",
                            "Z-score": f"{z:+.2f}",
                            "Percentile": f"{int(p)}th" if p else "—",
                        })
                    if stat_rows:
                        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

                    if pd.notna(comp):
                        pct = zscore_to_percentile(comp)
                        pct_label = format_percentile(pct)
                        st.markdown(f"**Composite: {comp:+.2f}** ({pct_label} of FBS {pos_display.lower()})")

                    # Multi-season history
                    full_history = df[(df["player"] == name) & (df["team"] == selected_school)].sort_values(season_col)
                    if len(full_history) > 1:
                        st.markdown("**Season history**")
                        hist_rows = []
                        for _, hrow in full_history.iterrows():
                            hz = [hrow.get(c) for c in available_z if c in hrow.index and pd.notna(hrow.get(c))]
                            hcomp = np.mean(hz) if hz else np.nan
                            hist_rows.append({
                                "Season": int(hrow[season_col]),
                                "Team": hrow.get("team", ""),
                                "Score": f"{hcomp:+.2f}" if pd.notna(hcomp) else "—",
                            })
                        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

                    # Check for transfer history (different teams)
                    all_player = df[df["player"] == name].sort_values(season_col)
                    unique_teams = all_player["team"].unique()
                    if len(unique_teams) > 1:
                        st.markdown("**Transfer history**")
                        transfer_rows = []
                        for _, trow in all_player.iterrows():
                            tz = [trow.get(c) for c in available_z if c in trow.index and pd.notna(trow.get(c))]
                            tcomp = np.mean(tz) if tz else np.nan
                            transfer_rows.append({
                                "Season": int(trow[season_col]),
                                "Team": trow.get("team", ""),
                                "Conference": trow.get("conference", ""),
                                "Score": f"{tcomp:+.2f}" if pd.notna(tcomp) else "—",
                            })
                        st.dataframe(pd.DataFrame(transfer_rows), use_container_width=True, hide_index=True)

                with pc2:
                    # Radar chart
                    import plotly.graph_objects as go
                    radar_axes, radar_values = [], []
                    for z_col, label in labels.items():
                        if z_col not in row.index: continue
                        z = row.get(z_col)
                        if pd.isna(z): continue
                        p = zscore_to_percentile(z)
                        radar_axes.append(label)
                        radar_values.append(p)

                    if len(radar_axes) >= 3:
                        st.markdown(f"**Percentile profile** vs. FBS {pos_display.lower()}")
                        st.caption("50th = FBS average")
                        rfig = go.Figure()
                        rfig.add_trace(go.Scatterpolar(
                            r=radar_values + [radar_values[0]],
                            theta=radar_axes + [radar_axes[0]],
                            fill="toself",
                            fillcolor="rgba(218, 165, 32, 0.25)",
                            line=dict(color="rgba(184, 134, 11, 0.9)", width=2),
                            marker=dict(size=6, color="rgba(184, 134, 11, 1)"),
                            hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<extra></extra>",
                        ))
                        rfig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100],
                                                tickvals=[25, 50, 75, 100],
                                                ticktext=["25th", "50th", "75th", "100th"],
                                                tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
                                angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
                                bgcolor="rgba(0,0,0,0)",
                            ),
                            showlegend=False, margin=dict(l=60, r=60, t=20, b=20),
                            height=320, paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(rfig, use_container_width=True, key=f"radar_{pos_name}_{name}")
                    else:
                        st.caption("Not enough stats for radar chart.")

    st.divider()
    st.caption("College data via CollegeFootballData.com (CC-BY-SA) · Z-scored against all FBS players at each position per season")

st.divider()
st.caption("Built with Streamlit · NFL data from nflverse · College data from CFBD · Open source on GitHub")
