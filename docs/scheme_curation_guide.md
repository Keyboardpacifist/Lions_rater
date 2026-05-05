# Scheme curation — your half of the work

You're filling in the football-knowledge layer. Claude is building the data layer in parallel.

## What's in this folder

`data/scheme/curation/` —

| File | Goal | Time estimate |
|---|---|---|
| `oc_signature_concepts.csv` | The 3-5 most distinctive plays/tendencies for each OC + a one-line identity | 1-2 hours |
| `coaching_tree.csv` | Mentor relationships + which "school" each OC came up in | 1-2 hours |

You can open these in Google Sheets / Excel / Numbers. Edit the cells, save back as CSV with the same filename.

## The 15 OCs to focus on first

These are the lighthouse subjects — get these right and the OC page lands. Other OCs can be filled in later.

**Tier 1 priority (must have for v1):**
1. Ben Johnson (DET) — your team, the lighthouse
2. Sean McVay (LA Rams)
3. Kyle Shanahan (SF)
4. Matt LaFleur (GB)
5. Andy Reid (KC)
6. Mike McDaniel (MIA)
7. Greg Roman (most recent: BAL)
8. Brian Daboll (NYG)
9. Josh McDaniels (free agent / next hire)
10. Sean Payton (DEN)

**Tier 2 priority (nice to have):**
11. Bobby Slowik (HOU)
12. Liam Coen (TB / JAX)
13. Drew Petzing (ARI)
14. Brian Schottenheimer (DAL)
15. Kellen Moore (PHI)

I've pre-filled rows for Ben Johnson, McVay, Shanahan, Reid, and Roman as quality bars. Mirror that level of specificity for the rest.

## How to fill `oc_signature_concepts.csv`

For each OC, ask yourself:

1. **What 3-5 plays/concepts come to mind first?** Specific enough that a fellow football fan would say "yeah, that's classic [OC name]."
   - Good: "Pre-snap motion at NFL-leading rate" / "Outside zone with cutback creases"
   - Too vague: "Good play-caller" / "Smart designer"

2. **What's their philosophical lean?** Do they LIVE in 11-personnel, or use heavy sets? Do they go vertical or move the chains? Quick game or longer drop-backs?

3. **What's the situational call you EXPECT from them?** 3rd & 8 — what's the play? Red zone — heavy or empty? 2-minute — tempo or methodical?

4. **The one-liner identity** — 15-20 words capturing their essence. Like a movie tagline.

### Quality-bar example (Ben Johnson, pre-filled):

> Pre-snap motion at NFL-leading rate · Condensed wide-receiver alignments · Outside-zone run scheme · Play-action shot plays off Henry-style heavy sets · Aggressive 4th-down decision-making
>
> *Modern WCO with aggressive intermediate passing and Henry-style run game disguise*

## How to fill `coaching_tree.csv`

For each OC:

1. **`school`** — which philosophical lineage? Pick from:
   - Shanahan tree (Mike Shanahan / Kyle Shanahan / McVay / LaFleur / McDaniel ecosystem)
   - Reid tree (Andy Reid / Holmgren / Pederson / Sirianni)
   - Belichick / Erhardt-Perkins tree (McDaniels / Daboll / O'Brien / Patricia)
   - Harbaugh tree (Greg Roman / etc.)
   - Coryell/Air-Coryell tree (legacy: Norv Turner / Tom Moore / direct descendants)
   - You can hyphenate hybrids: "Reid/Coryell hybrid" if it fits
   - Or invent new categories if needed (these aren't fixed — your call)

2. **`mentor_primary`** — the single biggest influence on their offensive identity (the boss they came up under). Often a HC or OC they served.

3. **`mentor_secondary`** — secondary influence (often a position coach who taught them, or a philosophical hero).

4. **`branched_through`** — career path summarized in 4-7 words. E.g., "MIA → IND → DET" or "WAS (Shanahan) → ATL → LA".

5. **`notes`** — anything else important about how they got their identity.

### Quality-bar example (already pre-filled):

| oc_name | school | mentor_primary | mentor_secondary | branched_through | notes |
|---|---|---|---|---|---|
| Ben Johnson | Shanahan tree | Matt LaFleur | Sean McVay (indirect) | MIA (under Adam Gase) → IND → DET | Promoted from passing-game coordinator to OC at DET in 2022. Most direct Shanahan influence is via LaFleur. |

## When you're stuck

For an OC you don't know well:

1. **Pro Football Reference's coach pages** show year-by-year roles — that gives you the career path.
2. **The Athletic / Sharp Football Analysis articles** often discuss philosophy explicitly.
3. **Beat-writer features** ("Behind Ben Johnson's offense...") usually lay out the signature concepts.
4. **Best move when guessing:** leave a row blank rather than fill it with low-quality info. Claude will surface "no curation yet" rather than misrepresent.

## When you're done

Save the CSVs in place. Claude will pick them up and ingest them into the OC page automatically — no further setup needed.

If you have questions while curating, drop them in chat and I'll resolve before integration.
