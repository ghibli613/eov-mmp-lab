# Stage 0c — Task 1: schema v3

Base support = BASE-predicate instances in the 800 training videos; masked novel instances excluded.

## Changes applied

- (a) `stand_with`, `lie_with`, `stop_with`: + proximity=adjacent
- (b) state_change re-tagged LEARNED
- (c) new axis `contact` (LEARNED): contact=contact on every predicate with a contact_action (`hold`, `ride`, `bite`, `touch`, `feed`, `kick`), on the three `*_inside` predicates, and on the `drive`/`pull` recipes
- (d) `drive` = contact=contact, object_state=moving, proximity=overlapping, relative_motion=co_move; `pull` = contact=contact, object_state=moving, relative_motion=co_move
- (e) `fight` left unassigned

## Base support, changed or new values

| axis | tag | value | base train instances | base predicates | masked (not usable) |
|---|---|---|---:|---:|---:|
| contact | LEARNED | contact | 599 | 2 | 311 |
| contact | LEARNED | none | 0 | 0 | 0 |
| state_change | LEARNED | halt | 1549 | 4 | 148 |
| proximity | MEASURED | adjacent | 1342 | 6 | 649 |
| proximity | MEASURED | overlapping | 0 | 0 | 72 |
| relative_motion | MEASURED | co_move | 1254 | 5 | 106 |
| object_state | MEASURED | moving | 1254 | 5 | 106 |

contact=contact base support comes only from `ride` (420) and `touch` (179); every other carrier is novel.

## Reachability (<50 applied to LEARNED axes only)

| | stage 0b | v3, strict | v3, coarse contact fallback |
|---|---:|---:|---:|
| unreachable novel predicates | 7 | 5 | 1 |

- **v3 strict** (every LEARNED ingredient must have ≥50): 5 unreachable — `bite` (contact_action=bite (0)), `feed` (contact_action=feed (0)), `fight` (no ingredients), `hold` (contact_action=hold (0)), `kick` (contact_action=kick (0))
- **v3 fallback** (a contact_action below threshold is dropped in favour of the supported coarse contact=contact): 1 unreachable — `fight`
- `drive` and `pull` are reachable in both views: their only LEARNED ingredient is contact=contact (599).

## Identical-phi collisions

| | stage 0b | v3 | v3 fallback |
|---|---:|---:|---:|
| all-novel | 0 | 1 | 2 |
| base-novel | 0 | 2 | 2 |
| all-base | 1 | 1 | 1 |

**⚠ NEW collisions created by v3: 3**

- base-novel: `lie_next_to`, `lie_with`* — phi {locomotion=none, posture=lie, proximity=adjacent, subject_state=stationary}
- base-novel: `stand_next_to`, `stand_with`* — phi {locomotion=none, posture=stand, proximity=adjacent, subject_state=stationary}
- all-novel: `stop_next_to`*, `stop_with`* — phi {proximity=adjacent, state_change=halt}

**⚠ Further collisions in the fallback view: 1**

- all-novel: `bite`*, `feed`*, `hold`*, `kick`* — phi {contact=contact}

`*` = novel.

## Strict-subset pairs

"Novel-subset" = the SUBSET predicate is novel (a novel predicate dominated by another). This is the stage-0b figure of 87; the 209 reported there counted any pair involving a novel predicate.

| | stage 0b | v3 | v3 fallback |
|---|---:|---:|---:|
| all strict-subset pairs | 392 | 373 | 401 |
| novel-subset pairs | 87 | 65 | 93 |

**⚠ NEW novel-subset pairs in v3: 1**

- `pull`* ⊂ `drive`*

Novel-subset pairs REMOVED by v3: 23

- `lie_with`* ⊂ `lie_above`*
- `lie_with`* ⊂ `lie_behind`
- `lie_with`* ⊂ `lie_beneath`*
- `lie_with`* ⊂ `lie_front`
- `lie_with`* ⊂ `lie_inside`*
- `lie_with`* ⊂ `lie_left`
- `lie_with`* ⊂ `lie_next_to`
- `lie_with`* ⊂ `lie_right`
- `stand_with`* ⊂ `stand_above`*
- `stand_with`* ⊂ `stand_behind`
- `stand_with`* ⊂ `stand_beneath`*
- `stand_with`* ⊂ `stand_front`
- `stand_with`* ⊂ `stand_inside`*
- `stand_with`* ⊂ `stand_left`
- `stand_with`* ⊂ `stand_next_to`
- `stand_with`* ⊂ `stand_right`
- `stop_with`* ⊂ `stop_above`*
- `stop_with`* ⊂ `stop_behind`
- `stop_with`* ⊂ `stop_beneath`*
- `stop_with`* ⊂ `stop_front`
- `stop_with`* ⊂ `stop_left`
- `stop_with`* ⊂ `stop_next_to`*
- `stop_with`* ⊂ `stop_right`

**⚠ Additional novel-subset pairs in the fallback view: 28**

- `bite`* ⊂ `drive`*
- `bite`* ⊂ `lie_inside`*
- `bite`* ⊂ `pull`*
- `bite`* ⊂ `ride`
- `bite`* ⊂ `sit_inside`*
- `bite`* ⊂ `stand_inside`*
- `bite`* ⊂ `touch`
- `feed`* ⊂ `drive`*
- `feed`* ⊂ `lie_inside`*
- `feed`* ⊂ `pull`*
- `feed`* ⊂ `ride`
- `feed`* ⊂ `sit_inside`*
- `feed`* ⊂ `stand_inside`*
- `feed`* ⊂ `touch`
- `hold`* ⊂ `drive`*
- `hold`* ⊂ `lie_inside`*
- `hold`* ⊂ `pull`*
- `hold`* ⊂ `ride`
- `hold`* ⊂ `sit_inside`*
- `hold`* ⊂ `stand_inside`*
- `hold`* ⊂ `touch`
- `kick`* ⊂ `drive`*
- `kick`* ⊂ `lie_inside`*
- `kick`* ⊂ `pull`*
- `kick`* ⊂ `ride`
- `kick`* ⊂ `sit_inside`*
- `kick`* ⊂ `stand_inside`*
- `kick`* ⊂ `touch`

## Task 1(d) test — is drive separable from ride on geometry?

Base `ride` training instances: 420; 420 with ≥2 frames where both boxes are annotated. Motion uses 418 of them (2 lack ECC matrices).

**Box relation (majority over the instance's frames)**

| relation | instances |
|---|---:|
| overlapping, subject centre above | 409/420 (97.4%) |
| contained (>=90% of subject box inside object box) | 8/420 (1.9%) |
| disjoint, subject above | 2/420 (0.5%) |
| overlapping, subject centre not above | 1/420 (0.2%) |

Median fraction of the subject box inside the object box: 0.55. Subject box top above object box top in 98.5% of frames (a contained box cannot do this).

**Co-translation** (net camera-compensated displacement over the extent, moving = >2% of the frame diagonal)

| | instances |
|---|---:|
| both boxes move | 355/418 (84.9%) |
| both move AND directions agree (cosine > 0.7) | 336/418 (80.4%) |
| neither moves | 37/418 (8.9%) |
| subject–object offset stable (drift < 0.25 object-box diagonals) | 370/418 (88.5%) |
| **satisfies the drafted drive recipe's geometry** (overlapping or contained, object moving, co-moving) | **336/418 (80.4%)** |
