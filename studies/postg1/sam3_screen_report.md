# Post-G1 Task 2: the SAM 3 screen on HELD — FAIL; SAM 3 closes as a detector

2026-11-02 · A30 (chain head `b10cf7a7…0aae`, commit `fc7909b`) · results commit: see the Monday summary
· exploration on 157 HELD videos, in-sample for EOV-MMP · not a test result

## Verdict

**No configuration passes with the class gate on (0 of 500), so by A30 SAM 3 closes as a detector.**
There is no second round of rules on HELD. SAM 3's tracker remains a candidate for Option 1 (Task 3).

**Why it fails: the gate alone caps SAM 3's recall gain at +4.90 points, below the +8 bar.**
- On its own, the gate gives: recall gain +4.90 (from +9.80), pair-recall gain +7.42 (from +9.16),
  burden 0.97× (from 1.84×).
- Every other rule only removes tracks, so no configuration with the gate can raise recall above
  +4.90. The gate's 154 removals include **23 tracks that match GT, 16 of them of novel classes**.
  These are SAM 3's own finds of objects that EOV-MMP's R0w tracks do not contain in that video.

**Without the gate, 16 configurations pass** (reported descriptively; A30 does not count them).
- All 16 combine the tiny, edge-sliver and geometry-partner rules.
- The best on A30's selection measure: tiny < 12 + edge (f ≥ 0.5, w < 16) + partner (d = 3), with
  recall +8.68, pair recall +8.79 and burden 1.29×.
- Option 2 without the gate would need all 35 prompts per video, at G1's cost.

**A caveat stated in A30:**
- The screen's gate uses R0w tracks (score ≥ 0.35), stricter than Option 2's planned ≥ 0.1 gate.
- A looser gate would remove fewer tracks, so the gate's recall cost here is an upper bound.
- EOV-MMP's sub-0.35 detections are not cached, so the size of that effect cannot be measured now.
- A30's decision is binding as written.

## 1. Setup and checks

**EOV-MMP R0w on the 157 videos (the reference):**
- recall 67.93% (485 / 714 GT runs);
- pair recall 55.81% (2,363 / 4,234 base relations);
- 0.962 unmatched confident tracks per video.

**SAM 3 with no rule** reproduces G1's 157-video numbers exactly: recall +9.80, pair recall +9.16,
burden 1.84×.

**The scorer:**
- It re-scores each configuration with A29's matching, best-pair and unmatched-track logic.
- It was **identical to `metrics.py`** on 27 configurations re-scored with that code itself: no
  rules, all rules, and 25 drawn with `numpy.random.default_rng(0)`.
- Unit tests: 7 passed.

**The run:** 105 s on the laptop, 0.7 GB of RAM. Output: `results_screen/sam3_screen.json`, which
holds all 1,000 configurations, every per-video file hash and the scorer checks. Log:
`logs/sam3_screen_run.log`.

## 2. The grid: 1,000 configurations

| | count |
|---|---|
| configurations tried | 1,000 (rules 1, 2, 3, 5, 7: 5 × 5 × 5 × 4 × 2) |
| **pass, class gate on** | **0** |
| pass, class gate off (descriptive) | 16 |
| best min(recall gain, pair gain) with the gate on | +4.90 (the gate alone, or with rules that remove no matched track) |

**The 16 passes without the gate.** Gains are in points.

| configuration | recall gain | pair gain | burden | tracks removed |
|---|---:|---:|---:|---:|
| tiny < 12 + edge (0.5, 16) + partner 3 | +8.68 | +8.79 | 1.291 | 88 |
| tiny < 12 + edge (0.8, 16) + partner 3 | +8.68 | +8.79 | 1.305 | 86 |
| tiny < 8 + edge (0.5, 32) + partner 3 | +8.68 | +9.09 | 1.331 | 82 |
| tiny < 12 + partner 3 | +8.68 | +8.79 | 1.331 | 82 |
| tiny < 12 + edge (0.5, 32) + partner 3 | +8.54 | +8.72 | 1.245 | 96 |
| tiny < 12 + edge (0.8, 32) + partner 3 | +8.54 | +8.72 | 1.258 | 94 |
| tiny < 8 + edge (0.5, 16) + partner 2 | +8.40 | +9.12 | 1.318 | 85 |
| tiny < 12 + edge (0.5, 16) + partner 2 | +8.26 | +8.74 | 1.238 | 98 |
| tiny < 12 + edge (0.8, 16) + partner 2 | +8.26 | +8.74 | 1.252 | 96 |
| tiny < 8 + edge (0.5, 32) + partner 2 | +8.26 | +9.05 | 1.272 | 93 |
| tiny < 12 + partner 2 | +8.26 | +8.74 | 1.278 | 92 |
| edge (0.5, 32) + partner 2 | +8.26 | +9.05 | 1.305 | 88 |
| tiny < 8 + edge (0.8, 32) + partner 2 | +8.26 | +9.05 | 1.305 | 88 |
| edge (0.8, 32) + partner 2 | +8.26 | +9.05 | 1.344 | 82 |
| tiny < 12 + edge (0.5, 32) + partner 2 | +8.12 | +8.67 | 1.192 | 106 |
| tiny < 12 + edge (0.8, 32) + partner 2 | +8.12 | +8.67 | 1.205 | 104 |

Edge (f, w) is the touch share and the width limit; partner d is the centre-distance factor.

## 3. The trade-off between burden and the recall gains

The Pareto front: configurations that no other configuration beats on both a lower burden and a
higher min(recall gain, pair gain). These are selected rows; the full front is in the JSON.

| configuration | gate | recall gain | pair gain | burden |
|---|---|---:|---:|---:|
| tiny < 8 + edge (0.5, 16) | off | +9.80 | +9.16 | 1.689 |
| tiny < 8 + edge (0.5, 16) + partner 3 | off | +8.82 | +9.16 | 1.377 |
| tiny < 12 + edge (0.5, 16) + partner 3 | off | +8.68 | +8.79 | 1.291 |
| tiny < 12 + edge (0.5, 32) + partner 2 | off | +8.12 | +8.67 | 1.192 |
| tiny < 16 + edge (0.5, 32) + partner 2 | off | +7.98 | +7.79 | 1.139 |
| tiny < 12 + edge (0.5, 32) + fragment (0.8, 0.8) + partner 2 | off | +6.02 | +6.61 | 1.040 |
| tiny < 8 + edge (0.5, 16) | **on** | +4.90 | +7.42 | 0.887 |
| tiny < 12 + edge (0.5, 32) | **on** | +4.62 | +6.97 | 0.795 |
| tiny < 16 + partner 3 | **on** | +3.64 | +6.16 | 0.649 |
| tiny < 16 + fragment (0.8, 0.8) + partner 3 | **on** | +1.82 | +4.30 | 0.536 |
| tiny < 24 + fragment (0.8, 0.8) + partner 3 | **on** | −0.28 | −0.52 | 0.470 |
| tiny < 24 + edge (0.5, 32) + fragment (0.8, 0.5) + partner 1 | **on** | −11.20 | −13.91 | 0.364 |

**Reading the front:**
- **Without the gate,** the burden falls from 1.84× to about 1.2–1.3× for about 1 point of recall:
  the tiny, edge and partner rules remove mostly unmatched tracks.
- **With the gate,** the burden is low (below 0.9×), but recall never exceeds +4.90.

## 4. Each rule alone

Gains are in points against EOV-MMP's R0w. "Matched tracks lost" and "unmatched tracks removed" count
against SAM 3's no-rule tracks. The last column lists the inspected images (§6) the rule removes.

| rule | setting | recall (gain) | pair recall (gain) | burden | removed | matched lost | unmatched removed | inspected removed |
|---|---|---|---|---:|---:|---:|---:|---|
| none | — | 77.73 (+9.80) | 64.97 (+9.16) | 1.841 | 0 | 0 | 0 | — |
| 1 tiny | t = 8 | 77.73 (+9.80) | 64.97 (+9.16) | 1.781 | 9 | 0 | 9 | 04 |
| 1 tiny | 12 | 77.59 (+9.66) | 64.60 (+8.79) | 1.629 | 33 | 1 | 32 | 04 |
| 1 tiny | 16 | 77.45 (+9.52) | 63.72 (+7.91) | 1.497 | 54 | 2 | 52 | 04, 11, 17, 18 |
| 1 tiny | 24 | 75.21 (+7.28) | 58.90 (+3.09) | 1.245 | 107 | 17 | 90 | 04, 07, 09, 11, 14, 16, 17, 18 |
| 2 edge | (f, w) = (0.5, 16) | 77.73 (+9.80) | 64.97 (+9.16) | 1.728 | 17 | 0 | 17 | 11, 17 |
| 2 edge | (0.5, 32) | 77.59 (+9.66) | 64.90 (+9.09) | 1.656 | 29 | 1 | 28 | 11, 14, 17 |
| 2 edge | (0.8, 16) | 77.73 (+9.80) | 64.97 (+9.16) | 1.768 | 11 | 0 | 11 | 11, 17 |
| 2 edge | (0.8, 32) | 77.59 (+9.66) | 64.90 (+9.09) | 1.695 | 23 | 1 | 22 | 11, 14, 17 |
| 3 fragment | (c, f) = (0.8, 0.5) | 73.81 (+5.88) | 57.46 (+1.65) | 1.470 | 78 | 22 | 56 | 02, 05, 06, 08, 14 |
| 3 fragment | (0.8, 0.8) | 75.63 (+7.70) | 62.92 (+7.11) | 1.596 | 51 | 14 | 37 | 02, 05, 06, 08, 14 |
| 3 fragment | (0.9, 0.5) | 74.79 (+6.86) | 61.08 (+5.27) | 1.497 | 68 | 16 | 52 | 02, 05, 06, 08, 14 |
| 3 fragment | (0.9, 0.8) | 75.77 (+7.84) | 62.92 (+7.11) | 1.636 | 44 | 13 | 31 | 02, 05, 06, 08, 14 |
| 4 score (reference) | s = 0.6 | 75.35 (+7.42) | 61.45 (+5.64) | 1.596 | 52 | 15 | 37 | 10 |
| 4 score (reference) | 0.7 | 68.35 (+0.42) | 50.99 (−4.82) | 1.364 | 114 | 42 | 72 | 02, 04, 06, 10, 11, 15, 16, 17 |
| 4 score (reference) | 0.8 | 53.78 (−14.15) | 34.27 (−21.54) | 0.563 | 300 | 107 | 193 | 16 of the 20 (all but 01, 08, 09, 14) |
| 4 score (reference) | 0.9 | 36.13 (−31.79) | 20.78 (−35.03) | 0.113 | 457 | 196 | 261 | 19 of the 20 (all but 08) |
| 5 partner | d = 1 | 66.81 (−1.12) | 56.16 (+0.35) | 1.159 | 153 | 50 | 103 | 04, 07, 12, 13, 16, 17, 18 |
| 5 partner | 2 | 76.33 (+8.40) | 64.93 (+9.12) | 1.437 | 67 | 6 | 61 | 07, 12, 13, 18 |
| 5 partner | 3 | 76.75 (+8.82) | 64.97 (+9.16) | 1.497 | 56 | 4 | 52 | 07, 12, 13, 18 |
| 6 modelC partner | — | skipped | | | | | | |
| 7 class gate | on | 72.83 (+4.90) | 63.23 (+7.42) | 0.974 | 154 | 23 | 131 | 01, 06, 10, 11, 15, 16, 17, 18 |

**Rule 6 is skipped, as A30 says.** G1 cached modelC scores only for SAM 3 pairs that recall a GT
relation (A29 item 7, box sensitivity), not for every pair of confident tracks. No features were
computed.

**Notes on the single rules:**
- **The tiny and edge rules are nearly free.** At t = 8 or edge (0.5, 16), they remove 9–17
  unmatched tracks and no matched track.
- **The fragment rule is expensive.** It removes 13–22 matched tracks at every setting. Presumably
  these are real objects whose box lies inside a larger track's box (for example, a rider and a
  bicycle). This was not checked.
- **The score cut (reference only) is the worst trade.** At s = 0.7, recall drops by 9.4 points
  while the burden falls only to 1.36×.

## 5. The class gate's removals

The gate removes 154 tracks, none of whose classes appear among EOV-MMP's R0w tracks in that video.

| | matched (SAM 3's no-rule matches) | unmatched | total |
|---|---:|---:|---:|
| novel class | **16** | 19 | 35 |
| base class | 7 | 112 | 119 |
| total | 23 | 131 | 154 |

**By class:**
- bird 30, person 27, giant_panda 15, sheep 14, car 11, dog 9, monkey 9, ball 7, antelope 5;
- bear, cattle, fox and lizard 3 each;
- motorcycle, train, elephant and squirrel 2 each;
- lion, zebra, horse, bicycle, turtle, snake and sofa 1 each.

**The gate is not as harmless as reply 1 expected.** It removes many person, bird and car tracks
(68 of the 154), so EOV-MMP often lacks those classes in a video where SAM 3 finds them. And its 16
novel matched removals carry most of the recall loss.

## 6. The 20 inspected tracks

These are the groups from `studies/rpt/g1/viz_unmatched_observations.md`, with the single rules that
remove each track. No configuration was selected, so none removes them as Option 2.

| # | observed | removed by (single rules at any setting) |
|---|---|---|
| 01 | real object, not the class (flamingo as "bird") | gate; score 0.9 |
| 02 | real object of the class (whale) | fragment (all); score ≥ 0.7 |
| 03 | piece of an object (rider's backside) | score ≥ 0.8 |
| 04 | real object of the class (distant person) | tiny (all); partner 1; score ≥ 0.7 |
| 05 | piece of an object (ram's horn) | fragment (all); score ≥ 0.8 |
| 06 | unclear (possibly a ball) | fragment (all); gate; score ≥ 0.7 |
| 07 | real object of the class (parked car) | tiny 24; partner (all); score ≥ 0.8 |
| 08 | piece of an object (person's back as "elephant") | fragment (all) |
| 09 | real object of the class (pedestrian) | tiny 24; score 0.9 |
| 10 | real object of the class (car) | gate; score ≥ 0.6 |
| 11 | no visible object (frame corner) | tiny ≥ 16; edge (all); gate; score ≥ 0.7 |
| 12 | real object of the class (fox) | partner (all); score ≥ 0.8 |
| 13 | probably the class (person in a boat) | partner (all); score ≥ 0.8 |
| 14 | unclear (edge sliver) | tiny 24; edge w < 32; fragment (all); score 0.9 |
| 15 | no visible object (bush) | gate; score ≥ 0.7 |
| 16 | no visible object (haze) | tiny 24; partner 1; gate; score ≥ 0.7 |
| 17 | no visible object (haze, edge) | tiny ≥ 16; edge (all); partner 1; gate; score ≥ 0.7 |
| 18 | no visible object (smudge) | tiny ≥ 16; partner (all); gate; score ≥ 0.8 |
| 19 | unclear (small shape) | score ≥ 0.8 |
| 20 | unclear (small shape) | score ≥ 0.8 |

**What the 20 show:**
- **The rules aimed at the observed artefacts hit them:**
  - the edge rule removes the frame-corner and haze slivers (11, 17);
  - tiny ≥ 16 removes 11, 17 and 18;
  - fragment removes the pieces (05, 08), but also the whale (02).
- **The partner rule removes real unannotated objects** standing alone: the parked car, the fox and
  the person in a boat (07, 12, 13).
- **None of the rules separates "unannotated real object" from "SAM 3 error" cleanly.** That fits the
  observation that a third of these tracks are real objects HELD does not annotate.

## 7. Files

- `studies/postg1/sam3_screen.py` (pinned in A30), `tests/test_sam3_screen.py`;
- `results_screen/sam3_screen.json` (every configuration, single rules, gate split, scorer checks,
  per-video hashes);
- `logs/sam3_screen_run.log`.
