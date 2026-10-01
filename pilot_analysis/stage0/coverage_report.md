# Stage 0 — coverage report

Support = BASE-predicate relation instances in the 800 training videos. `masked` = novel-predicate instances present in those same videos but excluded from the loss — **not available for training**, shown for reference only.

## (a) Support per (axis, value)

| axis | value | base train instances | base predicates | masked (novel, not usable) |
|---|---|---:|---:|---:|
| horizontal | left | 4124 | 11 | 12 |
| horizontal | right | 4130 | 11 | 9 |
| horizontal | none (never assigned) | 0 | 0 | 0 |
| vertical | above | 434 | 2 | 334 |
| vertical | beneath | 379 | 3 | 389 |
| vertical | none (never assigned) | 0 | 0 | 0 |
| depth | front | 3369 | 11 | 10 |
| depth | behind | 3022 | 8 | 357 |
| depth | none (never assigned) | 0 | 0 | 0 |
| proximity | overlapping ⚠ | 0 | 0 | 41 |
| proximity | adjacent | 1342 | 6 | 529 |
| proximity | separated (never assigned) | 0 | 0 | 0 |
| relative_motion | approach ⚠ | 12 | 2 | 338 |
| relative_motion | recede ⚠ | 0 | 0 | 304 |
| relative_motion | co_move | 1254 | 5 | 8 |
| relative_motion | pass | 54 | 2 | 397 |
| relative_motion | none (never assigned) | 0 | 0 | 0 |
| subject_state | moving | 8556 | 42 | 1418 |
| subject_state | stationary | 8585 | 18 | 1175 |
| object_state | moving | 1254 | 5 | 8 |
| object_state | stationary (never assigned) | 0 | 0 | 0 |
| comparative | larger | 1161 | 1 | 0 |
| comparative | taller | 1102 | 1 | 0 |
| comparative | faster | 89 | 1 | 0 |
| comparative | none (never assigned) | 0 | 0 | 0 |
| locomotion | walk | 4001 | 6 | 606 |
| locomotion | run | 657 | 6 | 161 |
| locomotion | fly | 877 | 5 | 55 |
| locomotion | swim | 175 | 4 | 73 |
| locomotion | creep | 176 | 5 | 56 |
| locomotion | jump | 226 | 6 | 71 |
| locomotion | none | 8585 | 18 | 1175 |
| contact_action | hold ⚠ | 0 | 0 | 76 |
| contact_action | ride | 420 | 1 | 0 |
| contact_action | bite ⚠ | 0 | 0 | 50 |
| contact_action | touch | 179 | 1 | 0 |
| contact_action | feed ⚠ | 0 | 0 | 45 |
| contact_action | kick ⚠ | 0 | 0 | 1 |
| contact_action | none (never assigned) | 0 | 0 | 0 |

⚠ = assigned to at least one predicate but fewer than 50 base training instances. (never assigned) = no predicate sets this value in the draft, so its zero reflects the drafting convention, not thin data.

## (b) Novel predicates, sorted by weakest ingredient

| novel predicate | min support | ingredients (base support) |
|---|---:|---|
| `away` | 0 | relative_motion=recede (0) |
| `bite` | 0 | contact_action=bite (0) |
| `creep_away` | 0 | locomotion=creep (176), relative_motion=recede (0), subject_state=moving (8556) |
| `feed` | 0 | contact_action=feed (0) |
| `fly_away` | 0 | locomotion=fly (877), relative_motion=recede (0), subject_state=moving (8556) |
| `hold` | 0 | contact_action=hold (0) |
| `jump_away` | 0 | locomotion=jump (226), relative_motion=recede (0), subject_state=moving (8556) |
| `kick` | 0 | contact_action=kick (0) |
| `lie_inside` | 0 | locomotion=none (8585), proximity=overlapping (0), subject_state=stationary (8585) |
| `move_away` | 0 | relative_motion=recede (0), subject_state=moving (8556) |
| `run_away` | 0 | locomotion=run (657), relative_motion=recede (0), subject_state=moving (8556) |
| `sit_inside` | 0 | locomotion=none (8585), proximity=overlapping (0), subject_state=stationary (8585) |
| `stand_inside` | 0 | locomotion=none (8585), proximity=overlapping (0), subject_state=stationary (8585) |
| `walk_away` | 0 | locomotion=walk (4001), relative_motion=recede (0), subject_state=moving (8556) |
| `creep_toward` | 12 | locomotion=creep (176), relative_motion=approach (12), subject_state=moving (8556) |
| `move_toward` | 12 | relative_motion=approach (12), subject_state=moving (8556) |
| `run_toward` | 12 | locomotion=run (657), relative_motion=approach (12), subject_state=moving (8556) |
| `toward` | 12 | relative_motion=approach (12) |
| `walk_toward` | 12 | locomotion=walk (4001), relative_motion=approach (12), subject_state=moving (8556) |
| `creep_past` | 54 | locomotion=creep (176), relative_motion=pass (54), subject_state=moving (8556) |
| `jump_past` | 54 | locomotion=jump (226), relative_motion=pass (54), subject_state=moving (8556) |
| `move_past` | 54 | relative_motion=pass (54), subject_state=moving (8556) |
| `past` | 54 | relative_motion=pass (54) |
| `walk_past` | 54 | locomotion=walk (4001), relative_motion=pass (54), subject_state=moving (8556) |
| `swim_behind` | 175 | depth=behind (3022), locomotion=swim (175), subject_state=moving (8556) |
| `swim_beneath` | 175 | locomotion=swim (175), subject_state=moving (8556), vertical=beneath (379) |
| `swim_next_to` | 175 | locomotion=swim (175), proximity=adjacent (1342), subject_state=moving (8556) |
| `creep_above` | 176 | locomotion=creep (176), subject_state=moving (8556), vertical=above (434) |
| `creep_next_to` | 176 | locomotion=creep (176), proximity=adjacent (1342), subject_state=moving (8556) |
| `jump_above` | 226 | locomotion=jump (226), subject_state=moving (8556), vertical=above (434) |
| `jump_behind` | 226 | depth=behind (3022), locomotion=jump (226), subject_state=moving (8556) |
| `jump_with` | 226 | locomotion=jump (226), object_state=moving (1254), relative_motion=co_move (1254), subject_state=moving (8556) |
| `beneath` | 379 | vertical=beneath (379) |
| `lie_beneath` | 379 | locomotion=none (8585), subject_state=stationary (8585), vertical=beneath (379) |
| `run_beneath` | 379 | locomotion=run (657), subject_state=moving (8556), vertical=beneath (379) |
| `sit_beneath` | 379 | locomotion=none (8585), subject_state=stationary (8585), vertical=beneath (379) |
| `stand_beneath` | 379 | locomotion=none (8585), subject_state=stationary (8585), vertical=beneath (379) |
| `stop_beneath` | 379 | locomotion=none (8585), subject_state=stationary (8585), vertical=beneath (379) |
| `walk_beneath` | 379 | locomotion=walk (4001), subject_state=moving (8556), vertical=beneath (379) |
| `above` | 434 | vertical=above (434) |
| `lie_above` | 434 | locomotion=none (8585), subject_state=stationary (8585), vertical=above (434) |
| `move_above` | 434 | subject_state=moving (8556), vertical=above (434) |
| `run_above` | 434 | locomotion=run (657), subject_state=moving (8556), vertical=above (434) |
| `stand_above` | 434 | locomotion=none (8585), subject_state=stationary (8585), vertical=above (434) |
| `stop_above` | 434 | locomotion=none (8585), subject_state=stationary (8585), vertical=above (434) |
| `walk_above` | 434 | locomotion=walk (4001), subject_state=moving (8556), vertical=above (434) |
| `run_next_to` | 657 | locomotion=run (657), proximity=adjacent (1342), subject_state=moving (8556) |
| `fly_behind` | 877 | depth=behind (3022), locomotion=fly (877), subject_state=moving (8556) |
| `fly_front` | 877 | depth=front (3369), locomotion=fly (877), subject_state=moving (8556) |
| `fly_left` | 877 | horizontal=left (4124), locomotion=fly (877), subject_state=moving (8556) |
| `fly_right` | 877 | horizontal=right (4130), locomotion=fly (877), subject_state=moving (8556) |
| `move_next_to` | 1342 | proximity=adjacent (1342), subject_state=moving (8556) |
| `sit_next_to` | 1342 | locomotion=none (8585), proximity=adjacent (1342), subject_state=stationary (8585) |
| `stop_next_to` | 1342 | locomotion=none (8585), proximity=adjacent (1342), subject_state=stationary (8585) |
| `sit_behind` | 3022 | depth=behind (3022), locomotion=none (8585), subject_state=stationary (8585) |
| `lie_with` | 8585 | locomotion=none (8585), subject_state=stationary (8585) |
| `stand_with` | 8585 | locomotion=none (8585), subject_state=stationary (8585) |
| `stop_with` | 8585 | locomotion=none (8585), subject_state=stationary (8585) |

**3 novel predicates have no ingredient assigned** (no schema value fits — see unresolved.md) and cannot be scored: `drive`, `fight`, `pull`.

## (c) Thin ingredients (< 50 base training instances)

| axis=value | base support | novel predicates depending on it |
|---|---:|---|
| contact_action=bite | 0 | `bite` |
| contact_action=feed | 0 | `feed` |
| contact_action=hold | 0 | `hold` |
| contact_action=kick | 0 | `kick` |
| proximity=overlapping | 0 | `lie_inside`, `sit_inside`, `stand_inside` |
| relative_motion=recede | 0 | `away`, `creep_away`, `fly_away`, `jump_away`, `move_away`, `run_away`, `walk_away` |
| relative_motion=approach | 12 | `creep_toward`, `move_toward`, `run_toward`, `toward`, `walk_toward` |

Near the line: relative_motion=pass (54), comparative=faster (89).

**Never assigned by the draft** (zero support because no predicate sets them, not because data is thin): horizontal=none, vertical=none, depth=none, proximity=separated, relative_motion=none, object_state=stationary, comparative=none, contact_action=none.

## (d) Counts over the 61 novel predicates

- all ingredients > 50 base instances: **39**
- all ingredients > 200 base instances: **29**
- at least one ingredient with ZERO base support: **14** — `away`, `bite`, `creep_away`, `feed`, `fly_away`, `hold`, `jump_away`, `kick`, `lie_inside`, `move_away`, `run_away`, `sit_inside`, `stand_inside`, `walk_away`
- no ingredient assigned at all (not counted above): **3** — `drive`, `fight`, `pull`
- remainder (min support 1–50): **5**

## (e) Ingredient combinations no base predicate instantiates (descriptive)

Over the 58 novel predicates with ≥1 ingredient:

- **exact**: no base predicate has an identical phi — **52**
- **containment**: no single base predicate's phi contains all of the novel predicate's ingredients — **41**: `away`, `bite`, `creep_away`, `feed`, `fly_away`, `hold`, `jump_away`, `kick`, `lie_inside`, `move_away`, `run_away`, `sit_inside`, `stand_inside`, `walk_away`, `creep_toward`, `run_toward`, `walk_toward`, `creep_past`, `jump_past`, `walk_past`, `swim_behind`, `swim_beneath`, `swim_next_to`, `creep_above`, `creep_next_to`, `jump_above`, `jump_behind`, `jump_with`, `lie_beneath`, `run_beneath`, `sit_beneath`, `stand_beneath`, `stop_beneath`, `walk_beneath`, `run_above`, `walk_above`, `run_next_to`, `fly_behind`, `fly_front`, `fly_left`, `fly_right`
- the converse — **6 novel predicates have a phi IDENTICAL to a base predicate**, so phi alone cannot separate them:

  - `lie_above` ≡ `sit_above`
  - `sit_behind` ≡ `lie_behind`, `stand_behind`, `stop_behind`
  - `sit_next_to` ≡ `lie_next_to`, `stand_next_to`
  - `stand_above` ≡ `sit_above`
  - `stop_above` ≡ `sit_above`
  - `stop_next_to` ≡ `lie_next_to`, `stand_next_to`

