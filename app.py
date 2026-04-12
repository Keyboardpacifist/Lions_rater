"""
Lions Receiver Rater – Stage 4: Community Algorithms
=====================================================
Everything from Stage 3.1 (two-layer bundle / advanced slider UI)
plus save / load / browse / fork / upvote for user-created algorithms,
persisted in Supabase.

Supabase table: algorithms
  id            uuid          (default gen_random_uuid())
  slug          text          unique
  name          text
  author        text
  description   text
  bundle_weights jsonb
  created_at    timestamptz   (default now())
  upvotes       integer       (default 0)
"""

import json, re, uuid, hashlib, time
from datetime import datetime

import streamlit as st
import pandas as pd
import polars as pl
from pathlib import Path

# ── Supabase client (lazy, cached) ─────────────────────────
# Uses st.secrets for SUPABASE_URL and SUPABASE_KEY set in
# Streamlit Cloud → Settings → Secrets.

from supabase import create_client, Client

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# ============================================================
# Page config & styling
# ============================================================
st.set_page_config(
    page_title="Lions Receiver Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
h1, h2, h3 { color: #0076B6 !important; }
.stSlider [data-baseweb="slider"] > div > div > div > div {
    background-color: #0076B6;
}
.section-divider {
    border-top: 2px solid #B0B7BC;
    margin: 1.5rem 0 1rem 0;
}
.bundle-desc {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: -0.5rem;
    margin-bottom: 0.5rem;
}
.stDataFrame { margin-top: 0.5rem; }
.algo-card {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: #fafbfc;
}
.algo-card h4 { margin: 0 0 0.25rem 0; color: #0076B6 !important; }
.algo-meta { font-size: 0.8rem; color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "data" / "master_lions_with_z.parquet"
    df = pl.read_parquet(data_path).to_pandas()
    return df

# ============================================================
# Stat catalog
# ============================================================
INDIVIDUAL_STATS = {
    "yards_per_target_z": ("Yards per target", "yards_per_target",
        "How many yards he produces every time the ball comes his way."),
    "epa_per_target_z": ("EPA per target", "epa_per_target",
        "Expected points added per target. The single best efficiency metric."),
    "success_rate_z": ("Success rate", "success_rate",
        "% of targets that produced a 'successful' play by EPA standards."),
    "catch_rate_z": ("Catch rate", "catch_rate",
        "% of targets that became receptions. Hands and reliability."),
    "avg_cpoe_z": ("CPOE", "avg_cpoe",
        "Completion % above expected. Catches the catches you're supposed to."),
    "first_down_rate_z": ("First down rate", "first_down_rate",
        "% of targets that picked up a first down. Chain-mover."),
    "yac_per_reception_z": ("YAC per reception", "yac_per_reception",
        "Average yards after catch. Run-after-catch ability."),
    "yac_above_exp_z": ("YAC over expected", "yac_above_exp",
        "YAC vs. what an average receiver would produce on the same catches."),
    "targets_per_snap_z": ("Targets per snap", "targets_per_snap",
        "How often he gets the ball when on the field. QB trust."),
    "yards_per_snap_z": ("Yards per snap", "yards_per_snap",
        "Total yards produced per snap. Combines efficiency and volume."),
    "avg_separation_z": ("Average separation", "avg_separation",
        "Average yards of separation from nearest defender at the catch point."),
}

# ============================================================
# Bundles
# ============================================================
BUNDLES = {
    "reliability": {
        "label": "🎯 Reliability",
        "description": "Catches what's thrown his way and keeps drives alive.",
        "stats": {
            "catch_rate_z": 0.30,
            "avg_cpoe_z": 0.20,
            "success_rate_z": 0.30,
            "first_down_rate_z": 0.20,
        },
    },
    "explosive": {
        "label": "💥 Explosive plays",
        "description": "Turns targets into chunk plays. Big gains, not just dump-offs.",
        "stats": {
            "yards_per_target_z": 0.50,
            "yac_above_exp_z": 0.30,
            "yards_per_snap_z": 0.20,
        },
    },
    "deep_threat": {
        "label": "🔥 Field stretcher",
        "description": "Takes the top off the defense. The 'go deep' guy.",
        "stats": {
            "yards_per_target_z": 0.40,
            "avg_separation_z": 0.30,
            "yards_per_snap_z": 0.30,
        },
    },
    "volume": {
        "label": "📊 Volume & usage",
        "description": "How much of the offense runs through him.",
        "stats": {
            "targets_per_snap_z": 0.50,
            "yards_per_snap_z": 0.50,
        },
    },
    "after_catch": {
        "label": "🏃 After the catch",
        "description": "What happens once he's got the ball in his hands.",
        "stats": {
            "yac_per_reception_z": 0.50,
            "yac_above_exp_z": 0.50,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "reliability": 60,
    "explosive": 50,
    "deep_threat": 30,
    "volume": 60,
    "after_catch": 30,
}

# ============================================================
# Supabase helpers
# ============================================================
def slugify(text: str) -> str:
    """Turn a name into a URL-friendly slug, append short hash for uniqueness."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    short_hash = hashlib.md5(f"{s}{time.time()}".encode()).hexdigest()[:6]
    return f"{s}-{short_hash}"


def save_algorithm(name: str, author: str, description: str,
                   bundle_weights: dict) -> dict | None:
    """Insert a new algorithm row into Supabase. Returns the row or None."""
    sb = get_supabase()
    slug = slugify(name)
    row = {
        "slug": slug,
        "name": name,
        "author": author or "Anonymous",
        "description": description,
        "bundle_weights": bundle_weights,
        "upvotes": 0,
    }
    try:
        resp = sb.table("algorithms").insert(row).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None


def list_algorithms(order_by: str = "upvotes", ascending: bool = False,
                    limit: int = 50) -> list[dict]:
    """Fetch algorithms from Supabase, ordered and limited."""
    sb = get_supabase()
    try:
        resp = (
            sb.table("algorithms")
            .select("*")
            .order(order_by, desc=not ascending)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def get_algorithm_by_slug(slug: str) -> dict | None:
    sb = get_supabase()
    try:
        resp = (
            sb.table("algorithms")
            .select("*")
            .eq("slug", slug)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def upvote_algorithm(algo_id: str, current_votes: int) -> bool:
    """Increment upvotes. Simple last-write-wins; fine for a demo."""
    sb = get_supabase()
    try:
        sb.table("algorithms").update(
            {"upvotes": current_votes + 1}
        ).eq("id", algo_id).execute()
        return True
    except Exception:
        return False

# ============================================================
# Session-state helpers
# ============================================================
if "loaded_algo" not in st.session_state:
    st.session_state.loaded_algo = None        # dict or None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()       # prevent double-votes per session


def apply_algo_weights(algo: dict):
    """Push an algorithm's bundle_weights into session state sliders."""
    bw = algo.get("bundle_weights", {})
    for bk in BUNDLES:
        st.session_state[f"bundle_{bk}"] = bw.get(bk, 0)
    st.session_state.loaded_algo = algo

# ============================================================
# Header
# ============================================================
st.title("🦁 Lions Receiver Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you value, "
    "and watch the Lions receivers re-rank in real time. "
    "_No 'best receiver' — just **your** best receiver._"
)
st.caption(
    "2024 regular season • Compared against all NFL WRs and TEs • "
    "Small samples adjusted toward league average"
)

# ============================================================
# Load data
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error("Couldn't find the data file.")
    st.stop()

# ============================================================
# Handle ?algo= query param → auto-load shared link
# ============================================================
query_params = st.query_params
if "algo" in query_params and st.session_state.loaded_algo is None:
    linked = get_algorithm_by_slug(query_params["algo"])
    if linked:
        apply_algo_weights(linked)
        st.rerun()

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Filters")

positions = st.sidebar.multiselect(
    "Positions",
    options=["WR", "TE"],
    default=["WR", "TE"],
)

min_snaps = st.sidebar.slider(
    "Minimum offensive snaps",
    min_value=0,
    max_value=1000,
    value=100,
    step=25,
    help="Hide players who barely played. Set to 0 to see everyone.",
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode",
    value=False,
    help="Show individual stat sliders (EPA, CPOE, separation, etc.) "
         "instead of plain-English bundles.",
)

st.sidebar.header("What do you value?")

# Show loaded-algo banner
if st.session_state.loaded_algo:
    la = st.session_state.loaded_algo
    st.sidebar.info(
        f"Loaded: **{la['name']}** by {la['author']}\n\n"
        f"_{la.get('description', '')}_"
    )
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.loaded_algo = None

# ── Build effective weights ─────────────────────────────────
effective_weights = {}
bundle_weights = {}

if not advanced_mode:
    st.sidebar.caption("Drag to weight what matters to you. 0 = ignore, 100 = max.")
    for bundle_key, bundle in BUNDLES.items():
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(
            f"<div class='bundle-desc'>{bundle['description']}</div>",
            unsafe_allow_html=True,
        )
        bundle_weights[bundle_key] = st.sidebar.slider(
            bundle["label"],
            min_value=0,
            max_value=100,
            value=DEFAULT_BUNDLE_WEIGHTS.get(bundle_key, 50),
            step=5,
            key=f"bundle_{bundle_key}",
            label_visibility="collapsed",
        )

    for bundle_key, bundle_weight in bundle_weights.items():
        if bundle_weight == 0:
            continue
        for z_col, internal_weight in BUNDLES[bundle_key]["stats"].items():
            effective_weights[z_col] = (
                effective_weights.get(z_col, 0) + bundle_weight * internal_weight
            )
else:
    st.sidebar.caption("Direct control over every underlying stat.")
    for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
        effective_weights[z_col] = st.sidebar.slider(
            display_name,
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help=desc,
            key=f"adv_{z_col}",
        )

# ============================================================
# Filter players & compute weighted score
# ============================================================
filtered = df[df["position"].isin(positions)].copy()
filtered = filtered[filtered["off_snaps"].fillna(0) >= min_snaps]

if len(filtered) == 0:
    st.warning("No players match the current filters. Try lowering the snap threshold.")
    st.stop()

total_weight = sum(effective_weights.values())

if total_weight == 0:
    filtered["score"] = 0.0
    st.info("All weights are zero — drag some sliders to start ranking.")
else:
    score = pd.Series(0.0, index=filtered.index)
    for z_col, w in effective_weights.items():
        if w == 0 or z_col not in filtered.columns:
            continue
        score += filtered[z_col].fillna(0) * (w / total_weight)
    filtered["score"] = score

filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1

# ============================================================
# Main panel: ranked table
# ============================================================
st.subheader("Ranking")

display_df = pd.DataFrame({
    "Rank": filtered.index,
    "Player": filtered["player_display_name"],
    "Pos": filtered["position"],
    "Snaps": filtered["off_snaps"].fillna(0).astype(int),
    "Targets": filtered["targets"].fillna(0).astype(int),
    "Yards": filtered["rec_yards"].fillna(0).astype(int),
    "TDs": filtered["rec_tds"].fillna(0).astype(int),
    "Score": filtered["score"].round(2),
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn(
            "Your Score",
            help="Higher = better. Roughly: 0 = league average, "
                 "+1 = top ~16%, +2 = top ~2.5%.",
            format="%.2f",
        ),
    },
)

# ============================================================
# Detail panel
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected_player = st.selectbox(
    "Pick a player to see how their score breaks down",
    options=filtered["player_display_name"].tolist(),
    index=0,
)

player_row = filtered[filtered["player_display_name"] == selected_player].iloc[0]

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Position", player_row["position"])
    st.metric("Snaps", int(player_row["off_snaps"]) if pd.notna(player_row["off_snaps"]) else 0)
    st.metric("Targets", int(player_row["targets"]) if pd.notna(player_row["targets"]) else 0)
    st.metric("Receiving yards", int(player_row["rec_yards"]) if pd.notna(player_row["rec_yards"]) else 0)
    st.metric("Your score", f"{player_row['score']:.2f}")

with col2:
    if not advanced_mode:
        st.markdown("**How your score breaks down**")
        bundle_rows = []
        for bundle_key, bundle in BUNDLES.items():
            bw = bundle_weights.get(bundle_key, 0)
            if bw == 0:
                continue
            contribution = 0
            for z_col, internal_weight in bundle["stats"].items():
                z = player_row.get(z_col)
                if pd.notna(z):
                    eff_weight = bw * internal_weight
                    contribution += z * (eff_weight / total_weight)
            bundle_rows.append({
                "Bundle": bundle["label"],
                "Your weight": f"{bw}",
                "Contribution": f"{contribution:+.2f}",
            })
        if bundle_rows:
            st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No bundles weighted — drag some sliders.")

        with st.expander("🔬 See the underlying stats"):
            st.caption(
                "Each bundle is a mix of these individual stats. "
                "Z-scores show how each player compares to the league "
                "(0 = average, +1 = top ~16%, +2 = top ~2.5%)."
            )
            stat_rows = []
            for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
                z = player_row.get(z_col)
                raw = player_row.get(raw_col)
                stat_rows.append({
                    "Stat": display_name,
                    "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("**Stat-by-stat breakdown** (z-score vs league)")
        breakdown_rows = []
        for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
            z = player_row.get(z_col)
            raw = player_row.get(raw_col)
            weight = effective_weights.get(z_col, 0)
            contribution = (
                (z if pd.notna(z) else 0) * (weight / total_weight)
                if total_weight > 0 else 0
            )
            breakdown_rows.append({
                "Stat": display_name,
                "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{weight}",
                "Contribution": f"{contribution:+.2f}",
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

# ============================================================
# ▸ STAGE 4: Save / Load / Browse / Fork / Upvote
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Community algorithms")

tab_save, tab_browse = st.tabs(["💾 Save your algorithm", "🌐 Browse community"])

# ── Save tab ────────────────────────────────────────────────
with tab_save:
    if advanced_mode:
        st.info(
            "Saving is available in **bundle mode** (toggle off Advanced in the sidebar). "
            "This keeps saved algorithms portable and comparable."
        )
    else:
        st.markdown(
            "Happy with your slider positions? Save them so others can load, "
            "fork, and upvote your creation."
        )
        with st.form("save_algo_form"):
            algo_name = st.text_input(
                "Algorithm name",
                max_chars=80,
                placeholder="e.g. The Jamo Special",
            )
            algo_author = st.text_input(
                "Your name",
                max_chars=60,
                placeholder="Anonymous",
            )
            algo_desc = st.text_area(
                "Short description (optional)",
                max_chars=280,
                placeholder="Prioritizes explosive YAC monsters over safe possession guys.",
            )
            submitted = st.form_submit_button("Save algorithm")

        if submitted:
            if not algo_name.strip():
                st.warning("Give your algorithm a name first.")
            else:
                saved = save_algorithm(
                    name=algo_name.strip(),
                    author=algo_author.strip(),
                    description=algo_desc.strip(),
                    bundle_weights=bundle_weights,
                )
                if saved:
                    slug = saved["slug"]
                    st.success(f"Saved! Your algorithm slug: **{slug}**")
                    st.code(
                        f"https://lions-rater.streamlit.app/?algo={slug}",
                        language=None,
                    )
                    st.caption("Share that link — anyone who opens it will load your weights.")

# ── Browse tab ──────────────────────────────────────────────
with tab_browse:
    sort_col = st.selectbox(
        "Sort by",
        options=["upvotes", "created_at"],
        format_func=lambda x: "Most upvoted" if x == "upvotes" else "Newest first",
        key="browse_sort",
    )

    algos = list_algorithms(order_by=sort_col, ascending=False, limit=50)

    if not algos:
        st.caption("No community algorithms yet. Be the first to save one!")
    else:
        for algo in algos:
            with st.container():
                st.markdown(
                    f"<div class='algo-card'>"
                    f"<h4>{algo['name']}</h4>"
                    f"<div class='algo-meta'>"
                    f"by {algo['author']} · {algo.get('upvotes', 0)} upvote(s)"
                    f"</div>"
                    f"<p style='margin:0.4rem 0 0 0; font-size:0.9rem;'>"
                    f"{algo.get('description', '') or '<em>No description</em>'}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Show the bundle weights as a compact summary
                bw = algo.get("bundle_weights", {})
                if bw:
                    labels = []
                    for bk, bv in bw.items():
                        if bv and bv > 0:
                            bundle_label = BUNDLES.get(bk, {}).get("label", bk)
                            labels.append(f"{bundle_label} {bv}")
                    if labels:
                        st.caption(" · ".join(labels))

                btn_cols = st.columns([1, 1, 1, 4])

                with btn_cols[0]:
                    if st.button("Load", key=f"load_{algo['id']}"):
                        apply_algo_weights(algo)
                        st.rerun()

                with btn_cols[1]:
                    if st.button("Fork", key=f"fork_{algo['id']}"):
                        # Fork = load weights but clear the loaded_algo reference
                        apply_algo_weights(algo)
                        st.session_state.loaded_algo = {
                            **algo,
                            "name": f"{algo['name']} (fork)",
                            "id": None,  # signal it's unsaved
                        }
                        st.rerun()

                with btn_cols[2]:
                    already_voted = algo["id"] in st.session_state.upvoted_ids
                    vote_label = "Upvoted" if already_voted else "Upvote"
                    if st.button(
                        f"👍 {vote_label} ({algo.get('upvotes', 0)})",
                        key=f"vote_{algo['id']}",
                        disabled=already_voted,
                    ):
                        if upvote_algorithm(algo["id"], algo.get("upvotes", 0)):
                            st.session_state.upvoted_ids.add(algo["id"])
                            st.rerun()

# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "FTN charting via FTN Data via nflverse (CC-BY-SA 4.0) • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)
