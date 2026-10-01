# Stage 1 session 2 — slot-level check, relative-motion rule, paraphrase freeze

$0: CPU only. The repo is untouched: all output is in this directory. BASE training labels only, and no test annotations. Details: `slot_check.md`. Frozen record: `preregistration.md` (SHA256 in `preregistration.sha256`).

```
python3 pilot_analysis/stage1b/slot_check.py   # ~8 min CPU
python3 pilot_analysis/stage1b/report.py       # writes the .md files
```

## Per-axis: session 1 (per instance) vs slot level (the classifier's grid)

| axis | session 1: agreement / AUC / verdict | slot level: window, agreement (per value), AUC | slot verdict |
|---|---|---|---|
| horizontal | 96.0% / 0.973 / PASS | W0: left 96.1%; right 96.0%, AUC 0.971 (0.942–0.991) | **PASS** |
| vertical | 100.0% / 1.000 / PASS | W0: above 100.0%; beneath 100.0%, AUC 1.000 (1.000–1.000) | **PASS** |
| depth | 90.8% / 0.951 / PASS | W1: behind 91.5%; front 91.7%, AUC 0.951 (0.916–0.977) | **PASS** |
| proximity | 72.4% / 0.737 / PASS | W3: adjacent 62.1%, AUC 0.724 (0.654–0.780) | **WEAK** |
| subject_state | 72.7% / 0.810 / PASS | W2: moving 75.7%; stationary 72.6%, AUC 0.799 (0.734–0.852) | **PASS** |
| object_state | 88.2% / — / PASS | W0: moving 86.5%, AUC — | **PASS** |
| comparative=larger | 94.6% / 0.989 / PASS | W3: larger 96.7%, AUC 0.991 (0.973–0.999) | **PASS** |
| comparative=taller | 93.2% / 0.975 / PASS | W3: taller 93.1%, AUC 0.987 (0.966–0.997) | **PASS** |
| comparative=faster | 90.5% / 0.905 / PASS | W1: faster 89.5%, AUC 0.834 (0.628–1.000) | **PASS** |

Same verdict rules as session 1. Symmetry: **exact at every window**. Grid: 13420 labelled slots (10857 FIT / 2563 HELD).

**Verdicts that changed at slot level:** proximity (PASS → WEAK). Proximity at W3: adjacent 175/282 (62.1%), negatives 1495/2144 (69.7%). subject_state for `stop`: 84/201 (41.8%).

## relative_motion: D1 or D2

| HELD | D1 distance | D2 heading |
|---|---:|---:|
| window (chosen on FIT) | W0 | W3 |
| pass ⚠ LOW-N | 1/10 (10.0%) | 0/10 (0.0%) |
| chase/follow | 55/77 (71.4%) | 25/77 (32.5%) |
| co_move (floor 60%) | 122/178 (68.5%) | 123/178 (69.1%) |
| subject-driven motion on stand/sit/lie (false positive) | 60/1144 (5.2%) | 4/1144 (0.3%) |

**Frozen: D1.** D1 is higher on both pass and chase/follow; co_move 68.5% ≥ 60%. The rule was applied as written, but its evidence is thinner than the table suggests:

- **pass was decided by one instance** (HELD 1 vs 0 of 10; FIT reverses it, 0 vs 4 of 47). Neither definition detects pass.
- **The chase/follow criterion favours D1 by construction.** D1 counts co_move as a hit, and 50 of its 55 hits are co_move; only 5 are approach. D2 has the far lower false-positive rate on stationary subjects.
- **Approach and recede remain unvalidated on base data** (12 and 0 base instances). The novel test AP of `*_away`, `*_toward` and `*_past` is their test.

## Stopped vs moving vehicles

AUC of motion_rate, move_* vs stop_* slots, vehicle subjects, at W2: **0.692 (0.622–0.759)** over all training videos (1465 move / 1069 stop slots; nothing is fitted in this comparison). HELD only: 0.558 (0.360–0.727) (243 / 201). The all-videos CI excludes 0.5, so a separator exists, but it is weak. The limitations note should say *weak*, not *absent*. The HELD-only CI includes 0.5 at this sample size.

## Paraphrases (frozen)

`paraphrases_frozen.json`: 132 predicates × 4, SHA256 `aa2cd527762d5b66f7aabe2f437a6fb1d08fe22b78de0d7d18e1524685f2dda6`. Both files are read-only. It was produced by a fresh subagent whose prompt held only the 132 strings, the three rules, the output format, and an instruction not to use tools. It made no tool calls apart from returning its answer. 10 random examples (seed 20260928):

| predicate | paraphrases |
|---|---|
| `creep_behind` | creeps behind · is creeping behind · crawls in back of · is crawling behind |
| `fly_front` | flies in front of · is flying in front of · flies ahead of · is flying ahead of |
| `lie_beneath` | lies beneath · is lying below · lies under · is lying underneath |
| `lie_next_to` | lies next to · is lying beside · lies alongside · is lying next to |
| `pull` | pulls · is pulling · tugs · drags |
| `sit_above` | sits above · is sitting over · is seated above · sits over |
| `sit_left` | sits to the left of · is sitting to the left of · is seated on the left of · sits on the left side of |
| `swim_left` | swims to the left of · is swimming to the left of · swims on the left of · is swimming on the left side of |
| `touch` | touches · is touching · is in contact with · makes contact with |
| `walk_with` | walks with · is walking with · walks together with · is walking along with |

⚠ For the human read, two systematic meaning shifts. They are recorded, not edited: **"ahead of" appears in all 12 `*_front` predicates.** That is heading-relative, whereas VidVRD front is camera-frame (0b). **"halts" / "comes to a stop" appears in all 8 `stop_*` predicates.** That is an event, whereas VidVRD stop means a stationary vehicle (0c). Minor: `play` → "plays" drops the object; `run_above` → "is running up above" adds "up".

