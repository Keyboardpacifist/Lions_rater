"""
College team page — destination from the CFB landing grid.

Approximates the NFL Team page (`pages/Team.py`) for college:
hero band with helmet + conference + rank, four tabs covering
team identity (position-group strengths + year-over-year +
recruiting class), roster cards, NFL pipeline history, and
historical comps. Click-throughs deep-link directly into the
player's College-mode profile via the force-detail token.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from lib_shared import inject_css, college_theme, render_player_card
from lib_top_nav import render_home_button
from lib_college_team_comps import (
    find_college_team_comps,
    load_college_team_seasons,
)
from lib_college_grid import _TEAM_COLORS, _DEFAULT_COLORS, _readable_text

st.set_page_config(
    page_title="College Team Profile",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

render_home_button()  # ← back to landing
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "college"
DATA_ROOT = REPO_ROOT / "data"


# ── Pick team / season ──────────────────────────────────────────
qp = st.query_params
qp_team = qp.get("team")
qp_season = qp.get("season")

team_df = load_college_team_seasons()
if team_df.empty:
    st.error(
        "College team data not loaded. Run "
        "`python tools/build_college_team_seasons.py`."
    )
    st.stop()

teams_avail = sorted(team_df["team"].unique().tolist())
seasons_avail = sorted(team_df["season"].unique().tolist(), reverse=True)

# Apply pending nav intent (set by the "Open …" comp buttons) before
# the selectbox widgets render. We can't write to widget keys after
# they're instantiated, so the click handler stashes the target in
# `college_nav_intent` and we resolve it here, on the rerun.
_nav = st.session_state.pop("college_nav_intent", None)
if _nav:
    nav_team, nav_season = _nav
    if nav_team in teams_avail:
        st.session_state["college_team_pick"] = nav_team
    if nav_season in seasons_avail:
        st.session_state["college_season_pick"] = int(nav_season)

if "college_team_pick" not in st.session_state:
    st.session_state["college_team_pick"] = (
        qp_team if (qp_team and qp_team in teams_avail) else "Michigan"
    )
if "college_season_pick" not in st.session_state:
    _s_int = None
    if qp_season:
        try:
            _s_int = int(qp_season)
        except (ValueError, TypeError):
            pass
    st.session_state["college_season_pick"] = (
        _s_int if _s_int in seasons_avail else seasons_avail[0]
    )

c1, c2 = st.columns([2, 1])
with c1:
    team = st.selectbox(
        "School",
        options=teams_avail,
        key="college_team_pick",
    )
with c2:
    season = st.selectbox(
        "Season",
        options=seasons_avail,
        key="college_season_pick",
    )

st.query_params.update({"team": team, "season": str(season)})

row = team_df[(team_df["team"] == team) & (team_df["season"] == season)]
if row.empty:
    st.warning(f"No data for {team} in {season}.")
    st.stop()
row = row.iloc[0]


# ── Helpers ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_team_meta() -> pd.DataFrame:
    """CFBD team metadata — conference + helmet logo per school."""
    p = DATA_ROOT / "cfbd_team_logos.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def _load_draft_linkage() -> pd.DataFrame:
    p = DATA / "college_to_nfl_draft_linkage.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def _load_recruiting() -> pd.DataFrame:
    p = DATA / "college_recruiting.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


_meta = _load_team_meta()
_meta_row = (_meta[_meta["school"] == team].iloc[0]
             if not _meta.empty and (_meta["school"] == team).any()
             else None)
conference = (_meta_row["conference"]
              if _meta_row is not None
              and isinstance(_meta_row["conference"], str)
              else None)
classification = (
    _meta_row["classification"] if _meta_row is not None
    and isinstance(_meta_row.get("classification"), str)
    else None
)
_CLASS_LABEL = {"fbs": "FBS", "fcs": "FCS",
                  "ii": "Division II", "iii": "Division III"}
class_label = _CLASS_LABEL.get(classification, "College Football")

theme = college_theme(team)
primary = theme.get("primary") or _DEFAULT_COLORS[0]
secondary = theme.get("secondary") or _DEFAULT_COLORS[1]
logo_url = theme.get("logo") or ""
text_color = _readable_text(primary)


# ── Rank helpers ────────────────────────────────────────────────

def _rank_in(df: pd.DataFrame, season_year: int, stat: str,
              team_value: str) -> tuple[int | None, int]:
    """Returns (rank, total) of `team_value` within season `season_year`
    by descending `stat`. Rank is 1-indexed, None if team missing."""
    pool = df[(df["season"] == season_year)].copy()
    pool = pool.dropna(subset=[stat])
    if pool.empty:
        return None, 0
    pool = pool.sort_values(stat, ascending=False).reset_index(drop=True)
    pool["_rank"] = pool.index + 1
    me = pool[pool["team"] == team_value]
    if me.empty:
        return None, len(pool)
    return int(me.iloc[0]["_rank"]), len(pool)


def _rank_in_conf(df: pd.DataFrame, season_year: int, stat: str,
                    team_value: str, conf: str | None,
                    meta: pd.DataFrame
                    ) -> tuple[int | None, int]:
    if not conf or meta.empty:
        return None, 0
    schools_in_conf = meta[meta["conference"] == conf]["school"].tolist()
    pool = df[(df["season"] == season_year)
              & (df["team"].isin(schools_in_conf))].copy()
    pool = pool.dropna(subset=[stat])
    if pool.empty:
        return None, 0
    pool = pool.sort_values(stat, ascending=False).reset_index(drop=True)
    pool["_rank"] = pool.index + 1
    me = pool[pool["team"] == team_value]
    if me.empty:
        return None, len(pool)
    return int(me.iloc[0]["_rank"]), len(pool)


def _ord(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th").join([str(n), ""])


def _rank_str(rank: int | None, total: int) -> str:
    if rank is None or not total:
        return "—"
    return f"{_ord(rank)} of {total}"


nat_rank, nat_total = _rank_in(team_df, int(season),
                                  "overall_strength", team)
conf_rank, conf_total = _rank_in_conf(team_df, int(season),
                                          "overall_strength", team,
                                          conference, _meta)
overall_strength = float(row["overall_strength"]) if pd.notna(
    row.get("overall_strength")) else None


# ── Hero band ───────────────────────────────────────────────────
helmet_html = (
    f'<img src="{logo_url}" alt="{team} helmet" '
    f'style="height: 90px; width: auto; '
    f'filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));"/>'
    if logo_url else ""
)
overall_str_str = (
    f"+{overall_strength:.2f}" if overall_strength and overall_strength >= 0
    else (f"{overall_strength:.2f}" if overall_strength is not None else "—")
)

st.markdown(
    f"""
<div style="
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    color: {text_color};
">
    <div style="display: flex; align-items: center; gap: 24px;
                  flex-wrap: wrap;">
        <div style="flex: 0 0 auto;">{helmet_html}</div>
        <div style="flex: 1 1 auto;">
            <div style="font-size: 38px; font-weight: 900;
                          letter-spacing: -0.5px; line-height: 1;">
                {team}
            </div>
            <div style="font-size: 14px; opacity: 0.85; margin-top: 8px;
                          font-weight: 500; letter-spacing: 1px;">
                {int(season)} SEASON ·
                {(conference.upper() + " · ") if conference else ""}
                {class_label.upper()}
            </div>
        </div>
        <div style="flex: 0 0 auto; text-align: right; min-width: 220px;">
            <div style="font-size: 11px; opacity: 0.7;
                          letter-spacing: 1.5px; font-weight: 700;">
                OVERALL STRENGTH
            </div>
            <div style="font-size: 32px; font-weight: 900;
                          line-height: 1; margin-top: 4px;">
                {overall_str_str}
            </div>
            <div style="font-size: 12px; opacity: 0.85; margin-top: 6px;">
                {_rank_str(nat_rank, nat_total)} nationally
            </div>
            {(f'<div style="font-size: 12px; opacity: 0.85;">'
              f'{_rank_str(conf_rank, conf_total)}'
              f' in the {conference}</div>')
             if conference and conf_total > 1 else ""}
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════
# TABS — page body split into four logical groups, mirroring Team.py
# ════════════════════════════════════════════════════════════════

tab_stats, tab_roster, tab_pipeline, tab_comps = st.tabs([
    "📊 Stats & Identity",
    "🎓 Roster",
    "🏆 NFL Pipeline",
    "🔮 Comps",
])


# ─── 📊 STATS & IDENTITY ────────────────────────────────────
with tab_stats:
    st.markdown("#### 📊  Position-group strengths")
    st.caption(
        "Z-scored against all college team-seasons in our database. "
        "Higher = stronger relative to the historical pool."
    )

    GROUPS_DISPLAY = [
        ("qb_strength_z",          "QB"),
        ("wr_te_pass_strength_z",  "Receivers"),
        ("rb_strength_z",          "Running backs"),
        ("ol_strength_z",          "Offensive line"),
        ("def_all_strength_z",     "Defense"),
        ("overall_strength",       "Overall"),
    ]
    cols = st.columns(len(GROUPS_DISPLAY))
    for col, (stat, label) in zip(cols, GROUPS_DISPLAY):
        with col:
            v = row.get(stat)
            if v is None or pd.isna(v):
                col.metric(label, "—")
            else:
                sign = "+" if v >= 0 else ""
                col.metric(label, f"{sign}{v:.2f}")

    # ── Year-over-year — improved / slipped lines ──────────────
    prior_season = int(season) - 1
    prior_row = team_df[(team_df["team"] == team)
                          & (team_df["season"] == prior_season)]
    if not prior_row.empty:
        prior_row = prior_row.iloc[0]
        st.markdown("---")
        st.markdown(
            f"#### 📈  Year over year — vs {prior_season}"
        )
        improved, slipped = [], []
        for stat, label in GROUPS_DISPLAY:
            cur = row.get(stat)
            prev = prior_row.get(stat)
            if pd.isna(cur) or pd.isna(prev):
                continue
            delta = float(cur) - float(prev)
            if delta >= 0.25:
                improved.append((label, delta, cur))
            elif delta <= -0.25:
                slipped.append((label, delta, cur))
        c_imp, c_slip = st.columns(2)
        with c_imp:
            st.markdown("**▲ Improved**")
            if improved:
                for lab, d, cur in sorted(improved, key=lambda x: -x[1]):
                    st.markdown(f"- **{lab}**  ·  +{d:.2f}σ "
                                  f"(now {cur:+.2f})")
            else:
                st.caption("_No groups improved by ≥0.25σ._")
        with c_slip:
            st.markdown("**▼ Slipped**")
            if slipped:
                for lab, d, cur in sorted(slipped, key=lambda x: x[1]):
                    st.markdown(f"- **{lab}**  ·  {d:.2f}σ "
                                  f"(now {cur:+.2f})")
            else:
                st.caption("_No groups slipped by ≥0.25σ._")
    else:
        st.markdown("---")
        st.caption(
            f"_No {prior_season} data on file — year-over-year "
            f"comparison unavailable._"
        )

    # ── Recruiting class ────────────────────────────────────────
    # The "current" recruiting story for a team viewing season N is
    # the INCOMING class arriving for season N+1 (e.g. 2025 page →
    # 2026 class). For a historical season, fall back to the class
    # arriving the next year. If neither is present, walk back to
    # the most-recent class we have for this school.
    rec_df = _load_recruiting()
    rec_year = int(season) + 1
    rec = rec_df[(rec_df["school"] == team)
                  & (rec_df["recruit_year"] == rec_year)]
    if rec.empty and not rec_df[rec_df["school"] == team].empty:
        latest_year = int(rec_df[
            rec_df["school"] == team]["recruit_year"].max())
        rec = rec_df[(rec_df["school"] == team)
                      & (rec_df["recruit_year"] == latest_year)]
        rec_year = latest_year

    st.markdown("---")
    st.markdown(f"#### 🎯  Recruiting class — {rec_year}")
    if rec.empty:
        st.caption("_No recruiting data on file for this program._")
    else:
        n_total = len(rec)
        n_5 = (rec["stars"] == 5).sum()
        n_4 = (rec["stars"] == 4).sum()
        n_3 = (rec["stars"] == 3).sum()
        avg_rating = float(rec["rating"].mean())
        c1_, c2_, c3_, c4_ = st.columns(4)
        c1_.metric("Commits", f"{n_total}")
        c2_.metric("5★", f"{n_5}")
        c3_.metric("4★", f"{n_4}")
        c4_.metric("Avg rating", f"{avg_rating:.4f}")

        top_rec = rec.nsmallest(5, "ranking")[
            ["name", "position", "stars", "rating", "ranking"]
        ].copy()
        top_rec["stars"] = top_rec["stars"].apply(
            lambda s: "★" * int(s) if pd.notna(s) else "—")
        top_rec["rating"] = top_rec["rating"].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—")
        top_rec["ranking"] = top_rec["ranking"].apply(
            lambda v: f"#{int(v)}" if pd.notna(v) else "—")
        top_rec.columns = ["Name", "Pos", "Stars", "Rating",
                              "National rank"]
        st.dataframe(top_rec, hide_index=True,
                       use_container_width=True)


# ─── 🎓 ROSTER ──────────────────────────────────────────────
_ROSTER_SOURCES = [
    ("QB",   "college_qb_all_seasons.parquet"),
    ("WR",   "college_wr_all_seasons.parquet"),
    ("TE",   "college_te_all_seasons.parquet"),
    ("RB",   "college_rb_all_seasons.parquet"),
    ("OL",   "college_ol_roster.parquet"),
    ("Defense", "college_def_all_seasons.parquet"),
]

# Defense rolls up multiple positions into one parquet — map each
# defender's listed_position to the College mode position key so
# clicks land on the correct leaderboard.
_DEF_POS_MAP = {
    "CB": "CB",  "DB": "S",  "S": "S",
    "DE": "DE",  "EDGE": "DE",
    "DT": "DT",  "DL": "DT", "NT": "DT",
    "LB": "LB",  "ILB": "LB", "OLB": "LB",
}


@st.cache_data(show_spinner=False)
def _college_roster_top(team_arg: str, season_arg: int) -> dict:
    out: dict[str, list] = {}
    for pos_label, fname in _ROSTER_SOURCES:
        path = DATA / fname
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path).to_pandas()
        except Exception:
            continue
        if "team" not in df.columns or "season" not in df.columns:
            continue
        sub = df[(df["team"] == team_arg)
                  & (df["season"] == season_arg)]
        if sub.empty:
            continue
        z_cols = [c for c in sub.columns if c.endswith("_z")]
        if not z_cols:
            continue
        sub = sub.copy()
        sub["_avg_z"] = sub[z_cols].mean(axis=1, skipna=True)
        for name_col in ("player_name", "name", "athlete", "player"):
            if name_col in sub.columns:
                break
        else:
            name_col = None
        if name_col is None:
            continue
        sub = sub.dropna(subset=["_avg_z", name_col])
        if sub.empty:
            continue
        top = sub.sort_values("_avg_z", ascending=False).head(3)
        rows = []
        for _, r in top.iterrows():
            if pos_label == "Defense":
                _listed = (r.get("listed_position")
                           or r.get("pos_group") or "LB")
                cm_pos = _DEF_POS_MAP.get(str(_listed).upper(), "LB")
            else:
                cm_pos = pos_label
            rows.append({
                "name": str(r[name_col]),
                "score": float(r["_avg_z"]),
                "cm_pos": cm_pos,
            })
        out[pos_label] = rows
    return out


@st.cache_data(show_spinner=False)
def _roster_row_for(team_arg: str, season_arg: int,
                      fname: str, name: str) -> dict | None:
    """Pull the full row for one player from the position parquet —
    needed to feed render_player_card with the raw stat columns
    it formats into tile values."""
    path = DATA / fname
    if not path.exists():
        return None
    df = pl.read_parquet(path).to_pandas()
    if "team" not in df.columns or "season" not in df.columns:
        return None
    for name_col in ("player_name", "name", "athlete", "player"):
        if name_col in df.columns:
            break
    else:
        return None
    m = df[(df["team"] == team_arg) & (df["season"] == season_arg)
           & (df[name_col] == name)]
    if m.empty:
        return None
    return dict(m.iloc[0])


_ROSTER_STAT_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "QB":  [("pass_yards",   "{:.0f}", "Pass yds"),
            ("pass_tds",     "{:.0f}", "TD"),
            ("pass_ints",    "{:.0f}", "INT"),
            ("pass_ypa",     "{:.1f}", "Y/Att"),
            ("pass_pct",     "{:.1%}", "Comp%")],
    "WR":  [("receptions",   "{:.0f}", "Rec"),
            ("rec_yards",    "{:.0f}", "Rec yds"),
            ("rec_tds",      "{:.0f}", "TD"),
            ("rec_ypr",      "{:.1f}", "Y/R")],
    "TE":  [("receptions",   "{:.0f}", "Rec"),
            ("rec_yards",    "{:.0f}", "Rec yds"),
            ("rec_tds",      "{:.0f}", "TD"),
            ("rec_ypr",      "{:.1f}", "Y/R")],
    "RB":  [("rush_carries", "{:.0f}", "Car"),
            ("rush_yards",   "{:.0f}", "Rush yds"),
            ("rush_tds",     "{:.0f}", "TD"),
            ("rush_ypc",     "{:.1f}", "YPC"),
            ("receptions",   "{:.0f}", "Rec")],
    "OL":  [],
    "Defense": [("tackles_total",   "{:.0f}", "Tkl"),
                ("tfl",              "{:.1f}", "TFL"),
                ("sacks",            "{:.1f}", "Sk"),
                ("interceptions",    "{:.0f}", "INT"),
                ("passes_deflected", "{:.0f}", "PD")],
}


with tab_roster:
    st.markdown("#### 🎓  Top performers")
    st.caption(
        "Top 3 players per position by all-stats average z-score, "
        "this team-season. Click any card to open the full College "
        "mode profile."
    )

    _roster = _college_roster_top(team, int(season))
    if not _roster:
        st.info("No roster data for this team-season yet.")
    else:
        for pk in _roster.keys():
            st.markdown(
                f"""
<div style="font-size: 13px; font-weight: 800; letter-spacing: 1.5px;
             opacity: 0.65; margin: 16px 0 6px 4px;">
    {pk.upper()}
</div>
""",
                unsafe_allow_html=True,
            )
            fname = next((f for label, f in _ROSTER_SOURCES
                           if label == pk), None)
            stat_specs = _ROSTER_STAT_SPECS.get(pk, [])
            for r in _roster[pk]:
                full_row = (
                    _roster_row_for(team, int(season), fname, r["name"])
                    if fname else None)
                view_row = (pd.Series(full_row)
                            if full_row else pd.Series())
                render_player_card(
                    player_name=r["name"],
                    position_label=r["cm_pos"],
                    season_str=f"{team} · {int(season)}",
                    score=r["score"],
                    stat_specs=stat_specs,
                    view_row=view_row,
                    team_label=team,
                    primary_color=primary,
                    secondary_color=secondary,
                    logo_url=logo_url,
                )
                btn_key = f"roster_{team}_{season}_{pk}_{r['name']}"
                if st.button("Open profile  →", key=btn_key,
                                use_container_width=True):
                    st.session_state["college_school_v2"] = team
                    st.session_state["college_season_landing"] = int(season)
                    st.session_state["college_position_top"] = r["cm_pos"]
                    st.session_state["expand_college_player"] = r["name"]
                    st.session_state[
                        f"lb_selected_{r['cm_pos']}"] = r["name"]
                    st.session_state["mode_toggle"] = "College"
                    st.session_state["_college_filter_ctx"] = (
                        team, "🏈 All conferences",
                        int(season), r["cm_pos"],
                    )
                    st.session_state["college_conference"] = (
                        "🏈 All conferences")
                    st.session_state["_force_college_detail"] = {
                        "school": team,
                        "season": int(season),
                        "position": r["cm_pos"],
                        "player": r["name"],
                    }
                    st.switch_page("app.py")


# ─── 🏆 NFL PIPELINE ────────────────────────────────────────
with tab_pipeline:
    st.markdown("#### 🏆  NFL draft history")
    st.caption(
        "Players from this program who've gone on to the NFL Draft. "
        "Source: nflverse + Pro-Football-Reference."
    )

    draft_df = _load_draft_linkage()
    if draft_df.empty:
        st.info("Draft linkage parquet not loaded.")
    else:
        m = draft_df[draft_df["college_team"] == team]
        if m.empty:
            st.caption(
                f"_No NFL draftees on file from {team}._"
            )
        else:
            cur_year = int(season)
            last10 = m[m["draft_year"].between(cur_year - 9, cur_year)]
            n_total = len(last10)
            n_r1 = (last10["round"] == 1).sum()
            n_r2_3 = last10["round"].between(2, 3).sum()
            avg_per_yr = n_total / max(1, last10["draft_year"].nunique())

            c1_, c2_, c3_, c4_ = st.columns(4)
            c1_.metric("Last 10 yrs", f"{n_total}")
            c2_.metric("First-rounders", f"{n_r1}")
            c3_.metric("R2–3", f"{n_r2_3}")
            c4_.metric("Per year (last 10)", f"{avg_per_yr:.1f}")

            st.markdown("---")
            st.markdown("**Recent draftees (last 3 classes)**")
            recent = m[m["draft_year"] >= cur_year - 2].copy()
            if recent.empty:
                st.caption("_No draftees in the last 3 classes._")
            else:
                recent = recent.sort_values(
                    ["draft_year", "overall"]).reset_index(drop=True)
                disp = recent[["draft_year", "round", "overall",
                                "college_player", "position",
                                "nfl_team"]].copy()
                disp.columns = ["Year", "Rd", "Pick", "Player",
                                  "Pos", "NFL team"]
                disp["Pick"] = disp["Pick"].apply(
                    lambda v: f"{int(v)}" if pd.notna(v) else "—")
                disp["Rd"] = disp["Rd"].apply(
                    lambda v: f"R{int(v)}" if pd.notna(v) else "—")
                st.dataframe(disp, hide_index=True,
                                use_container_width=True)

            st.markdown("---")
            st.markdown("**Draftees per year (last 10)**")
            yearly = (last10.groupby("draft_year").size()
                            .reset_index(name="Draftees"))
            yearly.columns = ["Year", "Draftees"]
            st.bar_chart(yearly.set_index("Year"))


# ─── 🔮 COMPS ───────────────────────────────────────────────
with tab_comps:
    st.markdown(
        "#### 🔮  Most comparable college team-seasons"
    )
    st.caption(
        "Cosine similarity across our college team-season database. "
        "Filter set excludes small-sample programs to keep the comp "
        "pool meaningful."
    )

    scope = st.radio(
        "Compare against:",
        options=["full", "offense", "defense"],
        horizontal=True,
        format_func=lambda s: {"full": "Full team",
                                 "offense": "Offensive twins",
                                 "defense": "Defensive twins"}[s],
        key="college_comp_scope",
    )
    include_all_fbs = st.checkbox(
        "🌐  Compare against all FBS teams",
        value=False,
        help=("By default, comps stay within the same tier (Power-4 vs "
              "Group-of-5 vs FCS) so a Big Ten team isn't matched with "
              "a Sun Belt program. Check this to open the pool to every "
              "FBS team-season we have."),
        key="college_comp_all_fbs",
    )
    comps = find_college_team_comps(
        team=team, season=int(season),
        scope=scope, n=3,
        restrict_to_tier=not include_all_fbs,
    )
    if not comps:
        st.info(
            "Not enough data to compute comps for this team-season yet."
        )
    else:
        cc = st.columns(3)
        for col, c in zip(cc, comps):
            comp_primary, comp_secondary = _TEAM_COLORS.get(
                c["team"], _DEFAULT_COLORS)
            comp_text = _readable_text(comp_primary)
            with col:
                st.markdown(
                    f"""
<div style="
    background: linear-gradient(135deg, {comp_primary} 0%, {comp_secondary} 100%);
    border-radius: 14px;
    padding: 20px;
    height: 100%;
    color: {comp_text};
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
    <div style="font-size: 11px; opacity: 0.7; letter-spacing: 1.5px;">
        SIMILARITY {c["similarity"]*100:.0f}%
    </div>
    <div style="font-size: 22px; font-weight: 800; line-height: 1.1;
                 margin-top: 4px;">
        {c["season"]} {c["team"]}
    </div>
    <div style="margin-top: 14px; font-size: 14px; line-height: 1.5;
                 opacity: 0.95;">
        {c["reason"]}
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                if st.button(
                    f"Open {c['season']} {c['team']} →",
                    key=f"cgo_{c['team']}_{c['season']}",
                    use_container_width=True,
                ):
                    st.session_state["college_nav_intent"] = (
                        c["team"], int(c["season"])
                    )
                    st.query_params.update({
                        "team": c["team"],
                        "season": str(c["season"]),
                    })
                    st.rerun()


# ── Click-through to existing College mode leaderboards ───────
st.markdown("---")
_, mid, _ = st.columns([1, 2, 1])
with mid:
    if st.button(
        f"📋  See full {team} leaderboards in College mode",
        use_container_width=True,
    ):
        st.session_state["college_school_v2"] = team
        st.session_state["mode_toggle"] = "College"
        st.switch_page("app.py")
