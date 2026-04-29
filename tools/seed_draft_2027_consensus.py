#!/usr/bin/env python3
"""Seed the 2027 NFL Draft consensus big board.

Source: Brett's curated big board (April 2026, returned-to-list edits
on 2026-04-29 to re-add 8 of the 9 prospects originally cut). We treat
this as the canonical board for the Draft page until we add scrapers
that aggregate multiple sources.

To re-rank: edit BOARD below in your desired order and re-run. The
script auto-renumbers and dedupes by (player, school) just in case.

Output: data/draft_2027_consensus.parquet

Run:
    python tools/seed_draft_2027_consensus.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "draft_2027_consensus.parquet"

# Final ordered board. Edit this list to re-rank.
# Format: (player, board_position, school)
BOARD = [
    ("Jeremiah Smith",         "WR",   "Ohio State"),
    ("Dante Moore",            "QB",   "Oregon"),
    ("Colin Simmons",          "EDGE", "Texas"),
    ("Leonard Moore",          "CB",   "Notre Dame"),
    ("Arch Manning",           "QB",   "Texas"),
    ("Dylan Stewart",          "EDGE", "South Carolina"),
    ("Drew Mestemaker",        "QB",   "Oklahoma State"),
    ("Cam Coleman",            "WR",   "Texas"),
    ("Jordan Seaton",          "OT",   "LSU"),
    ("David Stone",            "DL",   "Oklahoma"),
    ("Trevor Goosby",          "OT",   "Texas"),
    ("Ellis Robinson IV",      "CB",   "Georgia"),
    ("LaNorris Sellers",       "QB",   "South Carolina"),
    ("Ahmad Moten Sr.",        "DL",   "Miami (FL)"),
    ("Zabien Brown",           "CB",   "Alabama"),
    ("Charlie Becker",         "WR",   "Indiana"),
    ("Sammy Brown",            "LB",   "Clemson"),
    ("Damon Wilson Jr.",       "EDGE", "Miami (FL)"),
    ("Kelley Jones",           "CB",   "Mississippi State"),
    ("Kewan Lacy",             "RB",   "Mississippi"),
    ("Kyngstonn Viliamu-Asa",  "LB",   "Notre Dame"),
    ("KJ Bolden",              "S",    "Georgia"),
    ("Anthony Smith",          "DL",   "Minnesota"),
    ("Jamari Johnson",         "TE",   "Oregon"),
    ("A'Mauri Washington",     "DL",   "Oregon"),
    ("Jacarrius Peak",         "OT",   "South Carolina"),
    ("Matayo Uiagalelei",      "EDGE", "Oregon"),
    ("Koi Perich",             "S",    "Oregon"),
    ("Quincy Rhodes Jr.",      "EDGE", "Arkansas"),
    ("Darian Mensah",          "QB",   "Miami (FL)"),
    ("Ryan Williams",          "WR",   "Alabama"),
    ("Ryan Wingo",             "WR",   "Texas"),
    ("Boubacar Traore",        "EDGE", "Notre Dame"),
    ("Trey'Dez Green",         "TE",   "LSU"),
    ("CJ Carr",                "QB",   "Notre Dame"),
    ("Nick Marsh",             "WR",   "Indiana"),
    ("Cayden Green",           "OT",   "Missouri"),
    ("Julian Sayin",           "QB",   "Ohio State"),
    ("Ahmad Hardy",            "RB",   "Missouri"),
    ("OJ Frederique Jr.",      "CB",   "Miami (FL)"),
    ("William Echoles",        "DL",   "Mississippi"),
    ("Mario Craver",           "WR",   "Texas A&M"),
    ("Jordan Ross",            "DL",   "LSU"),
    ("Jayden Maiava",          "QB",   "USC"),
    ("Austin Siereveld",       "OT",   "Ohio State"),
    ("Kade Pieper",            "IOL",  "Iowa"),
    ("Justin Scott",           "DL",   "Miami (FL)"),
    ("Suntarine Perkins",      "LB",   "Mississippi"),
    ("Yhonzae Pierre",         "EDGE", "Alabama"),
    ("DJ Lagway",              "QB",   "Baylor"),
    ("Brauntae Johnson",       "S",    "Notre Dame"),
    ("TJ Moore",               "WR",   "Clemson"),
    ("Omarion Miller",         "WR",   "Arizona State"),
    ("Rasheem Biles",          "LB",   "Texas"),
    ("Sam Leavitt",            "QB",   "LSU"),
    ("Nate Frazier",           "RB",   "Georgia"),
    ("Chris Peal",             "CB",   "Syracuse"),
    ("Princewill Umanmielen",  "EDGE", "LSU"),
    ("Jayden Jackson",         "DL",   "Oklahoma State"),
    ("Ashton Hampton",         "CB",   "Clemson"),
    ("Will Heldt",             "EDGE", "Clemson"),
    ("Trevor Lauck",           "OT",   "Iowa"),
    ("Anthonie Knapp",         "OT",   "Notre Dame"),
    ("Brendan Sorsby",         "QB",   "Texas Tech"),
    ("Carter Smith",           "OT",   "Indiana"),
    ("Nyck Harbor",            "WR",   "South Carolina"),
    ("Bryant Wesco",           "WR",   "Clemson"),
    ("Josh Hoover",            "QB",   "Indiana"),
    ("Bear Alexander",         "DL",   "Oregon"),
    ("Brice Pollock",          "CB",   "Texas Tech"),
    ("Evan Tengesdahl",        "IOL",  "Cincinnati"),
    ("Lance Heard",            "OT",   "Kentucky"),
    ("Iapani Laloulu",         "IOL",  "Oregon"),
    ("John Henry Daley",       "EDGE", "Michigan"),
    ("Trinidad Chambliss",     "QB",   "Mississippi"),
    ("Blake Frazier",          "OT",   "Michigan"),
    ("Teitum Tuioti",          "EDGE", "Oregon"),
    ("Justice Haynes",         "RB",   "Georgia Tech"),
    ("Drake Lindsey",          "QB",   "Minnesota"),
    ("Mateen Ibirogba",        "DL",   "Texas Tech"),
    ("Kenyatta Jackson",       "EDGE", "Ohio State"),
    ("Maraad Watson",          "DL",   "Texas"),
    ("Adonijah Green",         "DL",   "Louisville"),
    ("Dylan Raiola",           "QB",   "Oregon"),
    ("Marcus Neal Jr.",        "S",    "Penn State"),
    ("Dashawn Spears",         "S",    "LSU"),
    ("Elijah Rushing",         "EDGE", "Oregon"),
    ("Bray Hubbard",           "S",    "Alabama"),
    ("Greg Johnson",           "IOL",  "Minnesota"),
    ("Ty Benefield",           "S",    "LSU"),
    ("LJ McCray",              "EDGE", "Florida"),
    ("Ezomo Oratokhai",        "OT",   "Northwestern"),
    ("A.J. Holmes Jr",         "DL",   "Texas Tech"),
    ("Jelani McDonald",        "S",    "Texas"),
    ("Terrance Carter",        "TE",   "Texas Tech"),
    ("Jyaire Hill",            "CB",   "Michigan"),
    ("Mark Fletcher",          "RB",   "Miami (FL)"),
    ("Niki Prongos",           "OT",   "Stanford"),
    ("Drew Bobo",              "IOL",  "Georgia"),
]

# Map the expert board's position labels to our internal position keys.
_POS_NORM = {
    "QB":   "QB",
    "RB":   "RB",
    "WR":   "WR",
    "TE":   "TE",
    "OT":   "OL",
    "IOL":  "OL",
    "DL":   "DT",
    "EDGE": "DE",
    "LB":   "LB",
    "CB":   "CB",
    "S":    "S",
}


def main() -> None:
    seen: set[tuple[str, str]] = set()
    rows = []
    rank = 0
    for player, board_pos, school in BOARD:
        key = (player.lower(), school.lower())
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        rows.append({
            "expert_rank": rank,
            "player": player,
            "board_position": board_pos,
            "position": _POS_NORM.get(board_pos, board_pos),
            "school": school,
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"✓ wrote {OUT.relative_to(REPO)} · {len(df)} prospects")
    print(f"  positions: {df['position'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
