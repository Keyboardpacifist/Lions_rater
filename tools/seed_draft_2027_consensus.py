#!/usr/bin/env python3
"""Seed the 2027 NFL Draft consensus big board.

Source: expert big board PDF (~Apr 2026, top 100). We treat this as
the canonical board for the Draft page until we add scrapers that
aggregate multiple sources.

Output: data/draft_2027_consensus.parquet

Run:
    python tools/seed_draft_2027_consensus.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "draft_2027_consensus.parquet"

# (rank, player, board_position, school)
# board_position is the position label on the expert board — we map
# it to our internal position keys (DE / DT / OL / etc.) at load
# time so the Draft page can filter by user-friendly position.
BOARD = [
    ( 1, "Arch Manning",          "QB",   "Texas"),
    ( 2, "Dante Moore",           "QB",   "Oregon"),
    ( 3, "Jeremiah Smith",        "WR",   "Ohio State"),
    ( 4, "Colin Simmons",         "EDGE", "Texas"),
    ( 5, "Julian Sayin",          "QB",   "Ohio State"),
    ( 6, "Leonard Moore",         "CB",   "Notre Dame"),
    ( 7, "Dylan Stewart",         "EDGE", "South Carolina"),
    ( 8, "Cam Coleman",           "WR",   "Texas"),
    ( 9, "Jordan Seaton",         "OT",   "LSU"),
    (10, "Trevor Goosby",         "OT",   "Texas"),
    (11, "Ellis Robinson IV",     "CB",   "Georgia"),
    (12, "David Stone",           "DL",   "Oklahoma"),
    (13, "Matayo Uiagalelei",     "EDGE", "Oregon"),
    (14, "Darian Mensah",         "QB",   "Miami (FL)"),
    (15, "Ryan Williams",         "WR",   "Alabama"),
    (16, "KJ Bolden",             "S",    "Georgia"),
    (17, "LaNorris Sellers",      "QB",   "South Carolina"),
    (18, "Jamari Johnson",        "TE",   "Oregon"),
    (19, "CJ Carr",               "QB",   "Notre Dame"),
    (20, "A'Mauri Washington",    "DL",   "Oregon"),
    (21, "Zabien Brown",          "CB",   "Alabama"),
    (22, "Brendan Sorsby",        "QB",   "Texas Tech"),
    (23, "Ahmad Hardy",           "RB",   "Missouri"),
    (24, "Trey'Dez Green",        "TE",   "LSU"),
    (25, "Quincy Rhodes Jr.",     "EDGE", "Arkansas"),
    (26, "Cayden Green",          "OT",   "Missouri"),
    (27, "Drew Mestemaker",       "QB",   "Oklahoma State"),
    (28, "Ahmad Moten Sr.",       "DL",   "Miami (FL)"),
    (29, "Nick Marsh",            "WR",   "Indiana"),
    (30, "Kelley Jones",          "CB",   "Mississippi State"),
    (31, "Kyngstonn Viliamu-Asa", "LB",   "Notre Dame"),
    (32, "Kewan Lacy",            "RB",   "Mississippi"),
    (33, "Austin Siereveld",      "OT",   "Ohio State"),
    (34, "Koi Perich",            "S",    "Oregon"),
    (35, "Carter Smith",          "OT",   "Indiana"),
    (36, "Charlie Becker",        "WR",   "Indiana"),
    (37, "Sammy Brown",           "LB",   "Clemson"),
    (38, "Damon Wilson Jr.",      "EDGE", "Miami (FL)"),
    (39, "Trinidad Chambliss",    "QB",   "Mississippi"),
    (40, "William Echoles",       "DL",   "Mississippi"),
    (41, "Mario Craver",          "WR",   "Texas A&M"),
    (42, "Jayden Maiava",         "QB",   "USC"),
    (43, "OJ Frederique Jr.",     "CB",   "Miami (FL)"),
    (44, "DJ Lagway",             "QB",   "Baylor"),
    (45, "Brauntae Johnson",      "S",    "Notre Dame"),
    (46, "Dylan Raiola",          "QB",   "Oregon"),
    (47, "Omarion Miller",        "WR",   "Arizona State"),
    (48, "Anthony Smith",         "DL",   "Minnesota"),
    (49, "Sam Leavitt",           "QB",   "LSU"),
    (50, "Jordan Ross",           "DL",   "LSU"),
    (51, "Chris Peal",            "CB",   "Syracuse"),
    (52, "Princewill Umanmielen", "EDGE", "LSU"),
    (53, "Jayden Jackson",        "DL",   "Oklahoma State"),
    (54, "Rasheem Biles",         "LB",   "Texas"),
    (55, "Ashton Hampton",        "CB",   "Clemson"),
    (56, "Will Heldt",            "EDGE", "Clemson"),
    (57, "Trevor Lauck",          "OT",   "Iowa"),
    (58, "Yhonzae Pierre",        "EDGE", "Alabama"),
    (59, "A.J. Holmes Jr",        "DL",   "Texas Tech"),
    (60, "Nate Frazier",          "RB",   "Georgia"),
    (61, "TJ Moore",              "WR",   "Clemson"),
    (62, "Ty Benefield",          "S",    "LSU"),
    (63, "Ryan Wingo",            "WR",   "Texas"),
    (64, "Josh Hoover",           "QB",   "Indiana"),
    (65, "John Henry Daley",      "EDGE", "Michigan"),
    (66, "Bear Alexander",        "DL",   "Oregon"),
    (67, "Blake Frazier",         "OT",   "Michigan"),
    (68, "Iapani Laloulu",        "IOL",  "Oregon"),
    (69, "Nyck Harbor",           "WR",   "South Carolina"),
    (70, "Suntarine Perkins",     "LB",   "Mississippi"),
    (71, "Evan Tengesdahl",       "IOL",  "Cincinnati"),
    (72, "Kenyatta Jackson",      "EDGE", "Ohio State"),
    (73, "Teitum Tuioti",         "EDGE", "Oregon"),
    (74, "Brice Pollock",         "CB",   "Texas Tech"),
    (75, "Justin Scott",          "DL",   "Miami (FL)"),
    (76, "Marcus Neal Jr.",       "S",    "Penn State"),
    (77, "Boubacar Traore",       "EDGE", "Notre Dame"),
    (78, "Kade Pieper",           "IOL",  "Iowa"),
    (79, "Anthonie Knapp",        "OT",   "Notre Dame"),
    (80, "Bryant Wesco",          "WR",   "Clemson"),
    (81, "Greg Johnson",          "IOL",  "Minnesota"),
    (82, "Dashawn Spears",        "S",    "LSU"),
    (83, "Drake Lindsey",         "QB",   "Minnesota"),
    (84, "Mateen Ibirogba",       "DL",   "Texas Tech"),
    (85, "Lance Heard",           "OT",   "Kentucky"),
    (86, "Maraad Watson",         "DL",   "Texas"),
    (87, "Jelani McDonald",       "S",    "Texas"),
    (88, "Niki Prongos",          "OT",   "Stanford"),
    (89, "Elijah Rushing",        "EDGE", "Oregon"),
    (90, "Adonijah Green",        "DL",   "Louisville"),
    (91, "Jacarrius Peak",        "OT",   "South Carolina"),
    (92, "Mark Fletcher",         "RB",   "Miami (FL)"),
    (93, "LJ McCray",             "EDGE", "Florida"),
    (94, "Bray Hubbard",          "S",    "Alabama"),
    (95, "Drew Bobo",             "IOL",  "Georgia"),
    (96, "Ezomo Oratokhai",       "OT",   "Northwestern"),
    (97, "Justice Haynes",        "RB",   "Georgia Tech"),
    (98, "Terrance Carter",       "TE",   "Texas Tech"),
    (99, "Jyaire Hill",           "CB",   "Michigan"),
]

# Map the expert board's position labels to our internal position keys.
# Used to make filters / per-position grouping match the rest of the
# College mode.
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
    rows = []
    for rank, player, board_pos, school in BOARD:
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
    print(f"✓ wrote {OUT.relative_to(REPO)} ({len(df)} rows)")
    print(f"  positions: {df['position'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
