"""Build K + P GAS Score tables.

Outputs:
  data/k_gas_seasons.parquet
  data/p_gas_seasons.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

K_MASTER = REPO / "data" / "league_k_all_seasons.parquet"
P_MASTER = REPO / "data" / "league_p_all_seasons.parquet"
K_OUT = REPO / "data" / "k_gas_seasons.parquet"
P_OUT = REPO / "data" / "p_gas_seasons.parquet"

from lib_st_gas import K_SPEC, P_SPEC, compute_st_gas  # noqa: E402


def _normalize(master: pd.DataFrame) -> pd.DataFrame:
    if "season" in master.columns and "season_year" in master.columns:
        master = master.drop(columns=["season"])
    elif "season_year" not in master.columns and "season" in master.columns:
        master = master.rename(columns={"season": "season_year"})
    return master


def _build(master_path: Path, spec, out_path: Path,
             label: str) -> None:
    print(f"→ {label} GAS — loading...")
    master = pd.read_parquet(master_path)
    print(f"  rows: {len(master):,}")
    master = _normalize(master)
    graded = compute_st_gas(master, spec=spec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(out_path, index=False)
    print(f"  ✓ wrote {out_path.relative_to(REPO)}")
    print()
    print(f"=== Top 10 {label} by GAS, 2024 (≥12 games) ===")
    score_cols = ["player_display_name", "recent_team", "games",
                    "gas_score", "gas_label", "gas_confidence"]
    bundle_cols = [c for c in graded.columns
                     if c.startswith("gas_") and c.endswith("_grade")]
    cols = [c for c in score_cols + bundle_cols
              if c in graded.columns]
    s24 = graded[(graded["season_year"] == 2024)
                  & (graded["games"] >= 12)
                  ].nlargest(10, "gas_score")
    print(s24[cols].to_string(index=False))
    print()
    df_y = graded[graded["games"] >= 12][
        ["player_id", "season_year", "gas_score"]
    ].dropna(subset=["player_id"]).sort_values(
        ["player_id", "season_year"]).copy()
    df_y["next"] = df_y.groupby("player_id")["gas_score"].shift(-1)
    df_y["next_season"] = df_y.groupby("player_id"
                                          )["season_year"].shift(-1)
    yoy = df_y.dropna(subset=["next"])
    yoy = yoy[yoy["next_season"] - yoy["season_year"] == 1]
    print(f"{label} YoY r: {yoy['gas_score'].corr(yoy['next']):.3f}  "
          f"(n={len(yoy)})")


def main() -> None:
    _build(K_MASTER, K_SPEC, K_OUT, "K")
    print()
    print("=" * 70)
    print()
    _build(P_MASTER, P_SPEC, P_OUT, "P")


if __name__ == "__main__":
    main()
