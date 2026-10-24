# Time check D1 — at which stage is each missed novel relation lost?

> **Test-label error analysis, disclosed; no method evaluated.** No method,
> threshold or parameter was chosen from it (A20). Anything built after it is
> designed and tuned on training videos only, under its own amendment.

**Reading (SGDet, novel, pooled, A20's rule): the largest group is upstream
(detection, tracking or object naming), at 61.6% (95% CI 50.9–72.2%) of missed
novel relations. The second largest is C1 ("not seen"), at 35.1% (24.7–45.7%).
Joining (A1, 1.7%) and near misses (N1, 1.7%) are small.**

A post-hoc diagnostic (section 5, caveat 1) shows that **most of C1 is not "not
seen"**. In 76% of SGDet C1 cases (80% in PredCls), the GT predicate is in the top
20 but its run continues beyond the GT, at a median of 2.6× (SGDet) or 5.0×
(PredCls) the GT's length. Trimmed to the GT's extent, it would match.

## 1. A20 and the reproduction check

| | SHA256 |
|---|---|
| preregistration.md with A20 (`preregistration.sha256`, line 1) | `71ab531522c6445395653ce94354a2bf233a09fa7c652c77c1c0aae8808a8c1e` |
| A20 text alone (5,877 bytes) | `7015b0d983f618f73c954ec1d58ae240ebe0bc7d7e87a675a76f5c7d2911b736` |
| the first 77,174 bytes (through A19), unchanged | `e1f16c4f285f812143d53fd4af5accdfa0db6be9331da9fe7b48d2cd2787be0d` |

**Step 0** was run before the test relation annotations were opened in this
batch. The repo's evaluator was run on the existing baseline outputs:

| outputs | novel mAP | all mAP | identical to the cache-derived outputs |
|---|---:|---:|---|
| PredCls, Task 4's `predcls_*.json` | **21.698** | 37.201 | yes (200 / 200 videos) |
| SGDet, Task 6's `sgdet_*.json` | **15.786** | 27.185 | yes (200 / 200 videos) |
| SGDet, pilot's `preds_from_pod/full/final_merged_*.json` | 15.786 | 27.185 | same triplets, durations and box values; integer boxes; scores within 1.1e-7 |

**Passed.** The analysis uses session 3's caches as the single source: GT tracks,
detected tracks and tracker classes, per-slot scores for all 132 predicates, and
post-processing re-run from them. SGDet per-slot scores for every predicate are
cached, so the rank analysis covers **both** SGDet and PredCls. The per-video
matcher reproduces the evaluator's mAP exactly in every pass and setting (e.g.
15.786 / 21.698).

**Is there a pre-scoring drop (U3)?** Yes, a narrow one:
- ordered pairs whose tracks overlap < 10 frames are never formed
  (`gen_labels.py:659`), and a tail of < 10 frames is trimmed;
- in PredCls, GT tracks that the repo densifier rejects never enter.

## 2. Step 1 — bucket tables

**Novel pass (primary).** There are 605 novel-split GT relations. SGDet misses 484
of them, and PredCls 361. Each cell is count (share, 95% video-bootstrap CI).

| SGDet | n | U1 tracks not found | U2 wrong object name | U3 pair not scored | A1 lost after scoring | N1 near miss | C1 not seen |
|---|---:|---|---|---|---|---|---|
| pooled | 484 | **262 (54.1%, 44.3–64.2)** | 34 (7.0%, 3.7–11.7) | 2 (0.4%, 0.0–1.5) | 8 (1.7%, 0.5–3.5) | 8 (1.7%, 0.6–3.1) | **170 (35.1%, 24.7–45.7)** |
| long | 305 | 157 (51.5%, 39.7–63.3) | 20 (6.6%, 2.9–12.1) | 2 (0.7%, 0.0–2.4) | 7 (2.3%, 0.4–5.1) | 6 (2.0%, 0.6–4.1) | 113 (37.0%, 25.5–49.2) |
| short | 179 | 105 (58.7%, 44.9–70.6) | 14 (7.8%, 2.8–14.9) | 0 | 1 (0.6%, 0.0–2.0) | 2 (1.1%, 0.0–3.1) | 57 (31.8%, 21.1–43.6) |

| PredCls | n | U3 pair not scored | A1 lost after scoring | N1 near miss | C1 not seen |
|---|---:|---|---|---|---|
| pooled | 361 | 21 (5.8%, 0.8–13.9) | 23 (6.4%, 2.5–11.9) | 21 (5.8%, 2.7–10.4) | **296 (82.0%, 71.2–89.6)** |
| long | 213 | 10 (4.7%, 0.6–10.9) | 17 (8.0%, 2.0–17.5) | 8 (3.8%, 1.5–7.4) | 178 (83.6%, 69.2–92.5) |
| short | 148 | 11 (7.4%, 0.0–19.4) | 6 (4.1%, 1.4–7.7) | 13 (8.8%, 3.2–15.9) | 118 (79.7%, 66.7–90.2) |

**All pass (secondary).** There are 4,835 GT relations. SGDet misses 3,821 and
PredCls 2,936.

| pooled | U1 | U2 | U3 | A1 | N1 | C1 |
|---|---|---|---|---|---|---|
| SGDet (3,821) | 2,119 (55.5%, 45.1–64.5) | 238 (6.2%, 3.1–10.4) | 15 (0.4%, 0.1–0.8) | 94 (2.5%, 1.3–4.0) | 86 (2.3%, 1.5–3.2) | 1,269 (33.2%, 25.5–41.8) |
| PredCls (2,936) | — | — | 194 (6.6%, 1.5–12.6) | 400 (13.6%, 6.3–23.9) | 203 (6.9%, 5.0–9.4) | 2,139 (72.9%, 58.3–83.7) |

Long / short splits for the all pass are in `data/tc_tables.json`. The pattern is
the same, and A1 is larger for long PredCls GT (17.4%).

## 3. Step 2 — inside the buckets (novel pass)

**A1, N1, C1: missed long GT, GT slots**

| | SGDet: GT / GT slots | SGDet: share of slots in top 20 | SGDet: median rank outside | PredCls: GT / slots | PredCls: share in top 20 | PredCls: median rank outside |
|---|---|---:|---:|---|---:|---:|
| A1 | 7 / 39 | 89.7% | 21.5 | 17 / 84 | 90.5% | 24.0 |
| N1 | 6 / 48 | 45.8% | 23.0 | 8 / 71 | 38.0% | 24.0 |
| **C1** | 113 / 360 | **91.1%** | 38.0 | 178 / 557 | **90.7%** | 36.5 |

**N1 and C1: (GT predicate → the predicate ranked first in that GT slot), the 15
most common.** These count GT slots where the top-1 isn't the GT predicate.

| SGDet | slots | kind | | PredCls | slots | kind |
|---|---:|---|---|---|---:|---|
| sit_behind → sit_next_to | 51 | same verb, diff. position | | walk_away → walk_toward | 26 | same verb, diff. position |
| walk_away → walk_toward | 24 | same verb, diff. position | | walk_toward → stand_with | 24 | other |
| walk_toward → stand_with | 17 | other | | stand_with → walk_toward | 21 | other |
| sit_behind → hold | 15 | other | | sit_behind → hold | 21 | other |
| walk_past → walk_toward | 13 | same verb, diff. position | | sit_beneath → sit_next_to | 17 | same verb, diff. position |
| walk_away → stand_with | 12 | other | | stand_with → stand_above | 13 | same verb, diff. position |
| fight → walk_toward | 9 | other | | walk_past → walk_toward | 11 | same verb, diff. position |
| stand_with → stand_above | 8 | same verb, diff. position | | sit_behind → sit_next_to | 11 | same verb, diff. position |
| feed → walk_toward | 8 | other | | drive → above | 11 | other |
| stand_with → walk_toward | 7 | other | | walk_past → stand_with | 10 | other |
| walk_above → sit_behind | 7 | other | | walk_toward → walk_past | 10 | same verb, diff. position |
| walk_past → stand_above | 7 | other | | walk_past → stand_beneath | 10 | other |
| walk_beneath → stand_with | 6 | other | | walk_past → stand_above | 9 | other |
| walk_toward → sit_next_to | 6 | other | | sit_next_to → sit_beneath | 9 | same verb, diff. position |
| stop_beneath → sit_beneath | 6 | same position, diff. verb | | sit_behind → sit_beneath | 9 | same verb, diff. position |

All confusion slots by kind:

| kind | SGDet | PredCls |
|---|---:|---:|
| other | 197 | 342 |
| same verb, different position | 116 | 141 |
| same position, different verb | 20 | 51 |
| bare term vs compound | 7 | 10 |

**U2: (GT class → tracker class), SGDet.** There are 54 class mismatches; **51
involve one of the 10 novel object categories.**

| GT class → tracker class | n |
|---|---:|
| giant_panda → tiger | 7 |
| giant_panda → monkey | 5 |
| antelope → lion | 5 |
| giant_panda → bear | 4 |
| red_panda → monkey | 4 |
| bear → monkey | 4 |
| lizard → turtle | 3 |
| red_panda → fox | 2 |
| red_panda → tiger | 2 |
| giant_panda → red_panda | 2 |

**A1 split** (post-processing re-run with the top-200 cut disabled):

| | (i) no matching instance from association | (ii) removed by the top-200 cut |
|---|---:|---:|
| SGDet | 2 | 6 |
| PredCls | 4 | 19 |

## 4. Step 3 — the reading

**SGDet, novel, pooled: upstream (U1 + U2 + U3) is the largest group, 298 of 484 =
61.6% (95% CI 50.9–72.2%), which points to detection, tracking or object naming,
outside the relation head. Second is C1, 170 = 35.1% (24.7–45.7%).** A1 (8, 1.7%,
0.5–3.5%) and N1 (8, 1.7%, 0.6–3.1%) are small, so neither a learned joining or
ranking step nor a slot re-scoring time head is supported by this reading.

## 5. Surprises and caveats

1. **C1 is mostly "seen, but the run overshoots" (post-hoc).** Step 2 shows the GT
   predicate in the top 20 in about 91% of C1's GT slots. A20's reachability builds
   each run over the pair's **whole** timeline. A predicate that stays in the top
   20 past the GT (persistence is 87%, F2) forms a run far longer than the GT, and
   fails vIoU 0.5. A post-hoc diagnostic (`tc_posthoc.py`; descriptive only,
   changing no bucket) took the best K = 20 run of each C1 GT:

   | C1 | SGDet (170) | PredCls (296) |
   |---|---:|---:|
   | no top-20 run touches the GT (truly not seen) | 20 (12%) | 53 (18%) |
   | **run overlaps; trimmed to the GT extent, it matches (over-long run)** | **129 (76%)** | **237 (80%)** |
   | run overlaps; even trimmed, the boxes don't match | 21 (12%) | 6 (2%) |
   | median run length / GT length | 2.6× | 5.0× |

   In SGDet, that makes about **27% of all missed novel GT (129 / 484) a temporal
   boundary problem.** The predicate is right and the pair is right, but nothing
   cuts the run where the relation ends. Uncut, the run's vIoU falls below 0.5. This
   matches the pilot's finding that loosening vIoU to 0.1 adds about +12 mAP.
   **The pre-registered reading is not changed**, but "C1 = a classifier limit"
   describes only about 12–18% of C1.
2. **Upstream dominates SGDet.** 54% of missed novel GT have no detected track
   pair at vIoU 0.5, and a further 7% have a wrongly named object, 94% of those
   involving a novel object category. The relation head never gets a chance at
   them. Detection and tracking quality bounds SGDet novel mAP more than anything
   downstream.
3. **Joining and the top-200 cut matter little.** A1 is 8 SGDet GT. In both
   settings, most A1 cases are the top-200 cut (ii), not association (i). Fixing
   joining alone has a small ceiling.
4. **The confusions are mostly across verbs ("other"), then positions within a
   verb**: sit_behind → sit_next_to, walk_away → walk_toward. walk_toward,
   stand_with and walk_past absorb many slots.
5. **The definitions shape the numbers.** Reachability takes one best run joined
   across one-slot gaps, and long/short is counted on the GT-track grid. A
   definition that allowed sub-runs would move most of C1 into A1 or N1 (caveat
   1). This is why the post-hoc split is reported next to the pre-registered
   table.
6. **Test labels were read** for this analysis only, and are disclosed. No method,
   threshold or parameter was chosen from them.

## 6. Hashes

| file | SHA256 |
|---|---|
| `tc_run.py` (pinned by A20) | `fae863dd65ee0c09a2cabfdab7a5f59df2e81832e8113c7c0383f7a5238d5996` |
| `tc_posthoc.py` (post-hoc diagnostic, not pre-registered) | `f7394569c8c4f478751a5a9cb664ff5347d8c93f63c023ce210dae7d4c9e80ea` |
| `data/tc_reproduction.json` | `39a47d29806ff3374a303008768437d54e9a598c7028c4407ce5b9dcb30b6d2a` |
| `data/tc_missed.json` (every missed GT with its bucket, ranks and reachability) | `2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2` |
| `data/tc_tables.json` (steps 1–3) | `c3cb77f9752e9315a2e3a33f365fed79cd04c76f7cfa2ee448f3ce715926bf80` |
| `data/tc_posthoc_c1.json` | `e4e95aee42811ef64637ecdb228dae9b56e0a2a99ac651281ae273190974e607` |

Logs: `logs/tc_reproduce.json`, `logs/tc_analyse.json`, `logs/tc_analyse_small.json`
and `logs/tc_posthoc_c1.json`. `data/` is git-ignored.
