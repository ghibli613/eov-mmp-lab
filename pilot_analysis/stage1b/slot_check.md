# Stage 1 session 2 — slot-level axis check, relative-motion choice

BASE labels from the TRAINING annotations only: novel instances are dropped when each file is parsed. Test annotations were not opened. Split: session 1's `train_split.json` (640 FIT / 160 HELD videos). Code: `slot_axes.py`, `slot_check.py`; raw numbers are in `slot_check_summary.json`.

## Grid (interface_audit.md B2, on whole tracks from `anno/train`)

6242 ordered pairs, 22584 slots, **13420 with at least one base label** (≥ 15 frames of a base predicate's extent inside the slot, same ordered pair): 10857 FIT, 2563 HELD. 17 videos without ECC are excluded. Slots whose labels assert conflicting values on an axis are left out for that axis: horizontal 10, vertical 0, depth 10, subject_state 102, proximity 0.

## Task 1 — FIT balanced agreement per window, chosen window, HELD

W0 = the slot; W1 = ± 1 slot; W2 = ± 2 slots; W3 = the pair's whole overlap, all clipped to the overlap. Thresholds are refit on FIT at each window, using session 1's rules. Each axis takes the window with the best FIT balanced agreement (ties go to the smaller window). HELD is scored once, at that window.

| axis | W0 | W1 | W2 | W3 | chosen | HELD agreement | HELD AUC (95% CI) |
|---|---:|---:|---:|---:|---|---|---:|
| horizontal | **0.966** | 0.966 | 0.964 | 0.936 | W0 | left 833/867 (96.1%); right 831/866 (96.0%) | 0.971 (0.942–0.991) |
| vertical | **0.996** | 0.996 | 0.993 | 0.991 | W0 | above 68/68 (100.0%); beneath 83/83 (100.0%) | 1.000 (1.000–1.000) |
| depth | 0.831 | **0.833** | 0.833 | 0.819 | W1 | behind 529/578 (91.5%); front 566/617 (91.7%) | 0.951 (0.916–0.977) |
| proximity | 0.694 | 0.699 | 0.696 | **0.700** | W3 | adjacent 175/282 (62.1%) | 0.724 (0.654–0.780) |
| subject_state | 0.738 | 0.743 | **0.744** | 0.722 | W2 | moving 709/937 (75.7%); stationary 965/1330 (72.6%) | 0.799 (0.734–0.852) |
| object_state | **0.822** | 0.808 | 0.805 | 0.810 | W0 | moving 154/178 (86.5%) | — |
| comparative=larger | 0.942 | 0.944 | 0.944 | **0.950** | W3 | larger 237/245 (96.7%) | 0.991 (0.973–0.999) |
| comparative=taller | 0.922 | 0.927 | 0.925 | **0.933** | W3 | taller 216/232 (93.1%) | 0.987 (0.966–0.997) |
| comparative=faster | 0.857 | **0.883** | 0.870 | 0.883 | W1 | faster 17/19 (89.5%) | 0.834 (0.628–1.000) |

Proximity negatives measured as non-adjacent on HELD: 1495/2144 (69.7%).

**Fitted thresholds per window (FIT):**

| window | motion (box-diag/frame) | adjacency gap | D1 dead zone | D2 margin c |
|---|---:|---:|---:|---:|
| W0 | 0.00447 | 0.3097 | 0.3748 | 0.840 |
| W1 | 0.00441 | 0.3404 | 0.4764 | 0.720 |
| W2 | 0.00447 | 0.3306 | 0.5310 | 0.720 |
| W3 | 0.00410 | 0.1975 | 1.0612 | 0.770 |

**subject_state by verb family, HELD, at W2:** `stop` 84/201 (41.8%), `swim` 17/29 (58.6%), `creep` 9/15 (60.0%), `sit` 106/171 (62.0%), `walk` 248/370 (67.0%), `lie` 64/92 (69.6%), `move` 162/232 (69.8%), `stand` 711/866 (82.1%), `follow` 47/57 (82.5%), `jump` 40/44 (90.9%), `run` 73/77 (94.8%), `chase` 16/16 (100.0%), `faster` 16/16 (100.0%), `fly` 81/81 (100.0%)

**Symmetry, at every window (fitted thresholds):**

| window | D1 time reversal | D2 time reversal | swap (3 spatial axes) | swap (comparatives) | D1 = session-1 function |
|---|---:|---:|---:|---:|---:|
| W0 | 13420/13420 | 13420/13420 | 40260/40260 | 40260/40260 | 13420/13420 |
| W1 | 13420/13420 | 13420/13420 | 40260/40260 | 40260/40260 | 13420/13420 |
| W2 | 13420/13420 | 13420/13420 | 40260/40260 | 40260/40260 | 13420/13420 |
| W3 | 13420/13420 | 13420/13420 | 40260/40260 | 40260/40260 | 13420/13420 |

**All exact.**

## Task 2 — relative_motion: D1 distance vs D2 heading

Each definition takes its window by FIT: the mean of three agreements, namely chase/follow (D1: approach or co_move; D2: toward), co_move on `*_with`, and no subject-driven motion on stand/sit/lie subjects. The LOW-N sets (pass, approach) were not used for fitting. D2's margin c is fitted on the same objective.

| HELD | D1 distance (W0) | D2 heading (W3) |
|---|---:|---:|
| **pass** ⚠ LOW-N | 1/10 (10.0%) | 0/10 (0.0%) |
| **chase/follow** (D1: approach or co_move · D2: toward) | 55/77 (71.4%) | 25/77 (32.5%) |
| **co_move** on `*_with` (floor 60%) | 122/178 (68.5%) | 123/178 (69.1%) |
| approach ⚠ LOW-N (D2: toward) | 2/2 (100.0%) | 0/2 (0.0%) |
| false positive: subject-driven motion on stand/sit/lie subjects | 60/1144 (5.2%) | 4/1144 (0.3%) |
| false positive: approach/recede (D1) or toward/away (D2) on `*_with` | 12/178 (6.7%) | 7/178 (3.9%) |

The same, on FIT: pass D1 0/47 (0.0%) vs D2 4/47 (8.5%); chase/follow D1 166/317 (52.4%) vs D2 72/317 (22.7%); co_move D1 656/1132 (58.0%) vs D2 677/1132 (59.8%).

What each definition outputs on HELD chase/follow slots: D1 co_move 50, none 18, approach 5, recede 2, pass 2; D2 none 28, toward 25, co_move 23, pass 1.

**Decision (rule fixed in the brief): D1.** D1 is higher on both pass and chase/follow. co_move 68.5% ≥ 60%.

Two things the rule does not show, recorded so nobody mistakes them for evidence:

- **The pass criterion was decided by a single instance.** On HELD, D1 detects 1/10 (10.0%) and D2 0/10 (0.0%). On FIT the order reverses: D1 0/47 (0.0%), D2 4/47 (8.5%). **Neither definition detects pass.**
- **The chase/follow criterion is asymmetric by construction.** D1 counts approach *or* co_move as a hit, while D2 needs toward. 50 of D1's 55 HELD hits are co_move, and only 5 are approach. D1 wins because chasers co-move at slot scale, not because it detects approach.

**Approach and recede remain unvalidated on base data.** There are 12 base approach instances and 0 recede, and `pass` fails under both definitions. The novel test AP of the 12 `*_away` / `*_toward` predicates (and the `*_past` ones) is their test.

## Task 3 — stopped vs moving vehicles

motion_rate at the chosen subject_state window (W2); move_* slots are positive, stop_* negative; vehicle subjects only (airplane, bicycle, bus, car, motorcycle, train, watercraft):

| slots | move | stop | AUC (95% CI, video-clustered) |
|---|---:|---:|---:|
| all training videos (nothing fitted) | 1465 | 1069 | 0.692 (0.622–0.759) |
| HELD only | 243 | 201 | 0.558 (0.360–0.727) |
