# Frozen scoring rule for stage 1

Schema: `phi_frozen.json` (132 predicates). Produced by stage 0e; stage 0 closes here.

## Axis roles

| role | axes | trained parameters |
|---|---|---|
| MEASURED | horizontal, vertical, depth, proximity, relative_motion, subject_state, object_state, comparative | none; parameter-free functions of box sequences |
| LEARNED | locomotion, contact_action, posture, contact | attribute head, trained on base only |
| SUPPLIED | subject_kind | none; read off the object classifier's category |

## Unset axes

- **locomotion, posture**: unset **asserts absence**. Licensed by stage 0c Task 2: bare spatial terms overlap a same-spatial compound 4.3% of the time and move_X overlaps manner_X 0.4% of the time, against 70.6% / 59.8% any-label controls.
- **every other axis**: unset is **marginalised out**.
- A predicate with no ingredients at all stays unscorable. The absence rule does not give it a phi.

## MEASURED definitions that changed in 0e

- **proximity=contained — ⚠ DECISION NEEDED.** The brief's definition fails its own acceptance test. All three candidates are parameter-free, evaluated per frame, with the instance taking the majority. Base `ride` (420 instances):

  | definition | ride classed as contained |
  |---|---:|
  | 0d: ≥90% of subject box inside object box (has a threshold) | 8/420 (1.9%) |
  | **brief**: subject-in-object > object-in-subject AND subject centre within object box | **191/420 (45.5%)** |
  | strict nesting: all four subject edges inside the object box | 0/420 (0.0%) |
  | **proposed**: subject-in-object > object-in-subject AND subject top edge not above object top edge | **1/420 (0.2%)** |

  The brief's version fails because a rider's centre (the hip) lies inside the mount's box in 332/420 instances. The rider is also usually the smaller box, so the fraction comparison favours the subject too. What separates ride is that the rider protrudes above the mount (0c: rider top above mount top in 98.5% of frames). The proposed version tests exactly that. Strict nesting separates as well, but a one-pixel box overshoot defeats it, which is a problem for real containment. Neither can be checked on a positive example, because every `contained` carrier is novel.
- **subject_kind**: vehicle = airplane, bicycle, bus, car, motorcycle, train, watercraft; animate = antelope, bear, bird, cattle, dog, domestic_cat, elephant, fox, giant_panda, hamster, horse, lion, lizard, monkey, person, rabbit, red_panda, sheep, snake, squirrel, tiger, turtle, whale, zebra; neither = ball, frisbee, skateboard, sofa.

## Novel predicates with NO learned ingredient (14)

No positive LEARNED value is set. Explicit locomotion=none counts as absence, which the rule treats the same as unset:

`above`*, `away`*, `beneath`*, `move_above`*, `move_away`*, `move_next_to`*, `move_past`*, `move_toward`*, `past`*, `stop_above`*, `stop_beneath`*, `stop_next_to`*, `stop_with`*, `toward`*

⚠ These are parameter-free in their positive evidence only. Under the absence rule, each of them is still scored against the learned locomotion and posture heads, which must say 'absent'. So they are not free of trained parameters. The heads they consult are well supported, with locomotion=none at 8585 base instances, so this does not affect reachability.

## Unreachable novel predicates (5)

`bite`* (contact_action=bite = 0), `feed`* (contact_action=feed = 0), `fight`* (no ingredients), `hold`* (contact_action=hold = 0), `kick`* (contact_action=kick = 0)

## What the rule does to the subset structure

A formal subset A ⊂ B is a real domination only if B's extra attributes are all on marginalised axes. If an extra attribute is locomotion or posture, A asserts its absence and the two conflict. Recomputed on the effective phis (unset locomotion and posture set to `none`):

| | formal phi | under the scoring rule |
|---|---:|---:|
| strict-subset pairs | 333 | 77 |
| novel-subset pairs | 62 | 31 |
| collision groups (all) | 1 | 1 |

Collision groups under the rule: `chase`, `fall_off`, `follow`.

The dominations that remain are listed in `subset_pairs_frozen.csv`, in the column `domination_under_scoring_rule`. Every one of them has its extra attributes on marginalised axes.

## ⚠ Licence check for the absence rule

0c tested bare spatial terms and move_*. The frozen rule applies absence to EVERY predicate with unset locomotion or posture, including the single-morpheme ones. Base training data, same pair and overlapping extent:

| family | instances | overlaps a locomotion compound | overlaps a posture compound | overlaps any other label |
|---|---:|---:|---:|---:|
| bare spatial (left, right, front, behind, next_to) | 1449 | 2.2% | 3.1% | 73.8% |
| move_* (all base) | 1974 | 0.4% | 0.0% | 59.8% |
| stop_* (all base) | 1549 | 0.1% | 0.0% | 66.2% |
| stand_* (all base) | 4676 | 2.0% | — | 75.2% |
| sit_* (all base) | 1508 | 0.7% | — | 64.5% |
| lie_* (all base) | 852 | 0.2% | — | 70.9% |
| walk_* (all base) | 4001 | — | 2.4% | 78.3% |
| run_* (all base) | 657 | — | 1.2% | 83.7% |
| fly_* (all base) | 877 | — | 0.0% | 38.3% |
| swim_* (all base) | 175 | — | 0.6% | 72.6% |
| creep_* (all base) | 176 | — | 0.0% | 42.0% |
| jump_* (all base) | 226 | — | 3.5% | 57.5% |
| `chase` | 83 | 69.9% | 1.2% | 100.0% |
| `fall_off` | 4 | 0.0% | 50.0% | 50.0% |
| `faster` | 89 | 67.4% | 0.0% | 98.9% |
| `follow` | 294 | 71.1% | 1.0% | 99.7% |
| `larger` | 1161 | 30.3% | 40.2% | 96.6% |
| `play` | 556 | 31.1% | 51.3% | 94.2% |
| `ride` | 420 | 0.0% | 80.0% | 80.0% |
| `taller` | 1102 | 29.3% | 45.2% | 99.3% |
| `touch` | 179 | 8.4% | 57.0% | 87.7% |
| `watch` | 608 | 22.9% | 56.9% | 90.8% |

**⚠ DECISION NEEDED — the rule holds for compositional predicates and fails for single-morpheme ones.** Every compositional family conflicts ≤3.5% of the time: the bare spatial terms and all 11 verb families, each on the axes it leaves unset or sets to `none`. Every single-morpheme base predicate with n > 10 conflicts on at least one axis 23–80% of the time: `chase`, `follow` and `faster` with a locomotion compound (~70%), `ride` with a posture compound (80%, riders sit), `touch`, `watch` and `play` with posture (51–57%). For these, an unset locomotion or posture is underspecified, not absent. Scoring it as absent would reject a running dog as `chase`.

**Proposed amendment:** apply absence only to compositional predicates (a bare spatial term, or verb + spatial term). Marginalise every unset axis of single-morpheme predicates. Among the novel ones, that means `bite`, `feed`, `hold`, `kick`, `pull`, `drive` and `fight`. Their base analogues (`touch`, `watch`, `play`, `ride`) co-occur with posture labels 51–80% of the time.

Effect on the subset structure: the 31 novel dominations are **identical** under the universal and the scoped rule (see summary.json). The amendment changes how single-morpheme predicates are scored, not which novel predicates are dominated. Of those 31: 23 are the stationary `*_with` pairs, 7 have a bare spatial term as subset, and 1 is `pull` ⊂ `drive`.

