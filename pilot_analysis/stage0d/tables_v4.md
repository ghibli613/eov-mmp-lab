# Stage 0d — final tables for schema v4

## Reachability (LEARNED threshold only, strict — no fallback)

| | 0c v3 strict | v4 (Task 1) | **v4 final** (Task 1 + 2B) |
|---|---:|---:|---:|
| unreachable novel predicates | 5 | 5 | 5 |

Unreachable in v4 final: `bite` (contact_action=bite (0)), `feed` (contact_action=feed (0)), `fight` (no ingredients), `hold` (contact_action=hold (0)), `kick` (contact_action=kick (0)).

## Identical-phi collisions

| | 0c v3 strict | v4 (Task 1) | **v4 final** (Task 1 + 2B) |
|---|---:|---:|---:|
| all-novel | 1 | 1 | 0 |
| base-novel | 2 | 2 | 0 |
| all-base | 1 | 1 | 1 |
| novel collision groups (all-novel + base-novel) | 3 | 3 | 0 |

v4 final groups:

- all-base: `chase`, `fall_off`, `follow` — {subject_state=moving}

**⚠ New vs v3: 0**

Removed vs v3: 3:
- base-novel: `lie_next_to`, `lie_with`*
- base-novel: `stand_next_to`, `stand_with`*
- all-novel: `stop_next_to`*, `stop_with`*

## Strict-subset pairs

| | 0c v3 strict | v4 (Task 1) | **v4 final** (Task 1 + 2B) |
|---|---:|---:|---:|
| strict-subset pairs, all | 373 | 373 | 393 |
| novel-subset pairs (subset is novel) | 65 | 65 | 88 |

**⚠ New novel-subset pairs vs v3: 23**:
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

Novel-subset pairs removed vs v3: 0

Of the new pairs, 23 are the ones Task 2 (B) brings back (they were removed in 0c by change (a)); the remaining 0 come from Task 1.

