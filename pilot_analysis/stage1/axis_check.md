# Stage 1 — Task A: measured-axis check

BASE-predicate labels from the TRAINING annotations only. Novel instances are dropped when each file is parsed and are never read after that. Test annotations are not opened. Functions: `measured_axes.py`. Frozen schema: `../stage0e/phi_frozen.json`.

**Split (A0):** 640 FIT / 160 HELD videos, video-level, seed 20260928 (`train_split.json`). **Excluded for missing ECC:** 17 videos. Base instances measured: 22315 (17950 FIT, 4365 HELD).

## A2 — thresholds (fitted on FIT only, frozen in `thresholds.json`)

| threshold | value | how |
|---|---:|---|
| **moving (motion_rate)** — frozen | 0.00447 box-diag/frame | max balanced agreement, stationary (6695) vs moving (7001) subjects; FIT BA 0.753, FIT AUC 0.812 |
| moving (speed, as specified) — rejected | 0.00189 frame-diag/frame | same fit; FIT BA 0.675, FIT AUC 0.725 |
| adjacency gap | 0.3310 box-diag | max balanced agreement, `next_to` family (1037) vs other-spatial-term instances (12246); FIT BA 0.698 |
| relative-motion dead zone | 0.4110 box-diag | 90th percentile of \|Δd\| over FIT co_move (`*_with`) instances (1064) |

Everything else is a sign test with no threshold: horizontal, vertical, depth, contained, comparative, and co_move's direction agreement (cosine > 0).

Adjacency negatives are instances labelled with a different spatial term. They are *not asserted adjacent*, not *asserted separated*, since no predicate asserts `separated`. Agreement on them is therefore a lower bound.

**Why the speed rule was replaced.** The specified moving measure (mean compensated per-frame step ÷ frame diagonal) fails on HELD; see A5. Two things make it weak. Box jitter adds to every step, so it inflates the mean step but cancels out of the net displacement. And dividing by the frame penalises small, distant subjects. I compared alternatives **on FIT only**: mean step ÷ frame diag gives AUC 0.725; net displacement ÷ frame diag 0.787; mean step ÷ box diag 0.763; **net displacement ÷ own box diagonal, per frame: 0.812**. The last one is `motion_rate`. It is the frozen rule for subject_state, object_state and co_move, and HELD was scored once, after the choice was made. `faster` keeps the specified speed, which passes.

## A3 — agreement on HELD

### horizontal

| label ↓ / function → | left | right | none |
|---|---:|---:|---:|
| left | 835 | 34 | 0 |
| right | 35 | 834 | 0 |

Agreement: left 835/869 (96.1%), right 834/869 (96.0%). AUC 0.973 (0.941–0.993).

### vertical

| label ↓ / function → | above | beneath | none |
|---|---:|---:|---:|
| above | 58 | 0 | 0 |
| beneath | 0 | 76 | 0 |

Agreement: above 58/58 (100.0%), beneath 76/76 (100.0%). AUC 1.000 (1.000–1.000).

### depth

| label ↓ / function → | front | behind | none |
|---|---:|---:|---:|
| front | 532 | 50 | 0 |
| behind | 50 | 495 | 0 |

Agreement: front 532/582 (91.4%), behind 495/545 (90.8%). AUC 0.951 (0.911–0.977).

### subject_state — as specified (mean step / frame diag)

| label ↓ / function → | moving | stationary | none |
|---|---:|---:|---:|
| moving | 963 | 476 | 0 |
| stationary | 779 | 984 | 0 |

Agreement: moving 963/1439 (66.9%), stationary 984/1763 (55.8%). AUC 0.685 (0.584–0.768).

### subject_state — corrected (motion_rate)

| label ↓ / function → | moving | stationary | none |
|---|---:|---:|---:|
| moving | 1128 | 311 | 0 |
| stationary | 482 | 1281 | 0 |

Agreement: moving 1128/1439 (78.4%), stationary 1281/1763 (72.7%). AUC 0.810 (0.742–0.870).

### proximity

adjacent: 202/279 (72.4%) of `next_to`-family instances measured adjacent. Negatives measured non-adjacent: 1731/2999 (57.7%). AUC (−gap) 0.737 (0.691–0.784). `contained` has no base carrier; see A4.

### object_state

moving, of `*_with` instances: as specified 79/186 (42.5%); corrected 164/186 (88.2%). Only `moving` is ever asserted, so there is no negative class and no AUC.

### relative_motion

- relative_motion — as specified: co_move 36/186 (19.4%) of HELD `*_with`. HELD output: none 118, co_move 36, approach 18, recede 10, pass 4. All-base output: co_move 632, none 480, approach 76, recede 52, pass 10.
- relative_motion — corrected (motion_rate): co_move 116/186 (62.4%) of HELD `*_with`. HELD output: co_move 116, none 38, approach 18, recede 10, pass 4. All-base output: co_move 758, none 354, approach 76, recede 52, pass 10.

### comparative

Only the positive value is ever labelled, so the negative class is the same instance with subject and object swapped. The 2×2 is then fixed by antisymmetry: labelled-larger cases measured larger = swapped cases measured smaller.

| value | n (HELD) | measured as asserted | tie / no motion | AUC vs swapped |
|---|---:|---:|---:|---:|
| larger | 221 | 209/221 (94.6%) | 0 | 0.989 (0.973–0.998) |
| taller | 206 | 192/206 (93.2%) | 0 | 0.975 (0.945–0.992) |
| faster | 21 | 19/21 (90.5%) | 0 | 0.905 (0.764–1.000) |

## A4 — values with little or no base support

**a. Exact symmetry tests, all base instances (frozen rules):**

- time reversal (approach↔recede, pass/co_move/none fixed): 22315/22315
- subject/object swap (left↔right, above↔beneath, front↔behind): 66945/66945
- subject/object swap inverts larger/taller/faster: 66945/66945
- **all exact**

**b. False-positive rates (report only):**

- recede on HELD co_move (`*_with`) instances: 10/186 (5.4%)
- recede on HELD instances with a stand/sit/lie subject: 26/1486 (1.7%)
- contained on base `ride` (all videos): 1/418 (0.2%)
- contained on all base instances: 2299/22315 (10.3%). Most frequent predicates it fires on: `move_beneath` 158, `play` 122, `stand_front` 107, `stand_next_to` 105, `walk_next_to` 101, `walk_left` 88

**c. LOW-N values** (no threshold was fitted on them, so ALL base videos are usable as well):

- pass ⚠ LOW-N: HELD 1/14 (7.1%); all base 3/53 (5.7%). Function output on all: none 17, co_move 14, recede 11, approach 8, pass 3
- approach ⚠ LOW-N: HELD 1/1 (100.0%); all base 5/12 (41.7%). Function output on all: none 6, approach 5, co_move 1

## A5 — verdicts

Agreement is the LOWEST per-value HELD agreement among values with ≥50 base instances. PASS ≥ 70% with exact symmetry; FIXABLE < 70% with AUC ≥ 0.8; WEAK AUC 0.7–0.8; FAIL AUC < 0.7.

| axis | HELD agreement (per value) | AUC | verdict | novel predicates that depend on it |
|---|---|---:|---|---:|
| horizontal | left 835/869 (96.1%); right 834/869 (96.0%) | 0.973 (0.941–0.993) | **PASS** | 2 |
| vertical | above 58/58 (100.0%); beneath 76/76 (100.0%) | 1.000 (1.000–1.000) | **PASS** | 17 |
| depth | front 532/582 (91.4%); behind 495/545 (90.8%) | 0.951 (0.911–0.977) | **PASS** | 5 |
| proximity | adjacent 202/279 (72.4%) | 0.737 (0.691–0.784) | **PASS** | 10 |
| subject_state — as specified (mean step / frame diag) | moving 963/1439 (66.9%); stationary 984/1763 (55.8%) | 0.685 (0.584–0.768) | **FAIL** | 49 |
| subject_state — corrected (motion_rate) | moving 1128/1439 (78.4%); stationary 1281/1763 (72.7%) | 0.810 (0.742–0.870) | **PASS** | 49 |
| object_state — as specified | moving 79/186 (42.5%) | — | **UNRESOLVED (<70%, no negative class for an AUC)** | 3 |
| object_state — corrected (motion_rate) | moving 164/186 (88.2%) | — | **PASS** | 3 |
| relative_motion — as specified | co_move 36/186 (19.4%) | — | **UNRESOLVED (<70%, no negative class for an AUC)** | 20 |
| relative_motion — corrected (motion_rate) | co_move 116/186 (62.4%) | — | **UNRESOLVED (<70%, no negative class for an AUC)** | 20 |
| comparative=larger | larger 209/221 (94.6%) | 0.989 (0.973–0.998) | **PASS** | 0 |
| comparative=taller | taller 192/206 (93.2%) | 0.975 (0.945–0.992) | **PASS** | 0 |
| comparative=faster | faster 19/21 (90.5%) | 0.905 (0.764–1.000) | **PASS** | 0 |

### Notes on the verdicts

**Pre-registered expectation: held.** subject_state was the weakest axis as specified: HELD agreement 55.8% on stationary, AUC 0.685 (0.584–0.768), which is below 0.70, so FAIL. The corrected `motion_rate` rule, chosen on FIT, passes on HELD, but narrowly: stationary 1281/1763 (72.7%), AUC 0.810 (0.742–0.870). By verb family on HELD, from worst to best: `stop` 128/277, `swim` 15/29, `sit` 111/173, `lie` 85/121, `move` 239/333, `walk` 466/647, `creep` 9/12, `stand` 957/1192, `follow` 58/67, `jump` 42/48, `run` 168/171, `fly` 95/96, `faster` 21/21, `chase` 15/15. **`stop_*` is the weak spot**: most 'stopped' vehicles measure as moving. The four novel `stop_*` predicates depend on subject_state=stationary.

**relative_motion: UNRESOLVED, not PASS.** co_move agrees 116/186 (62.4%) on HELD. It has no negative class, so no AUC is possible, and the pre-registered rules cannot separate FIXABLE, WEAK and FAIL. The other values are worse. `pass` fires on 3/53 (5.7%) of all base `*_past` instances. `approach` fires on 5/12 (41.7%). `recede` has no base example at all. 20 novel predicates depend on this axis. Re-tagging it LEARNED would cover co_move (1254 base instances) and pass (54). It would NOT cover approach (12) or recede (0). So these 12 novel predicates would become unreachable: `away`, `creep_away`, `creep_toward`, `fly_away`, `jump_away`, `move_away`, `move_toward`, `run_away`, `run_toward`, `toward`, `walk_away`, `walk_toward`. As MEASURED they stay reachable, but they rest on a rule that base data cannot validate.

**proximity: PASS on the letter, marginal in substance.** Adjacent agrees 202/279 (72.4%), but the AUC is 0.737 (0.691–0.784), which is WEAK-level, and only 1731/2999 (57.7%) of the other-spatial-term negatives measure non-adjacent. Those negatives are weak: labelled with a different spatial term, not asserted separated. `contained` fires on 10.3% of all base instances. The 2-D overlap it detects is often occlusion (`move_beneath`, `stand_front`). The four `contained` novel predicates will inherit that false-positive rate.

