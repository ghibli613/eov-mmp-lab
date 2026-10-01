# Stage 0b — coverage with MEASURED / LEARNED tags

Base support = BASE-predicate instances in the 800 training videos. `masked` = novel-predicate instances in those videos, excluded from the loss, shown for reference only.

| axis | tag | value | base train instances | base predicates | masked (not usable) |
|---|---|---|---:|---:|---:|
| horizontal | MEASURED | left | 4124 | 11 | 12 |
| horizontal | MEASURED | right | 4130 | 11 | 9 |
| horizontal | MEASURED | none (never assigned) | 0 | 0 | 0 |
| vertical | MEASURED | above | 434 | 2 | 334 |
| vertical | MEASURED | beneath | 379 | 3 | 389 |
| vertical | MEASURED | none (never assigned) | 0 | 0 | 0 |
| depth | MEASURED | front | 3369 | 11 | 10 |
| depth | MEASURED | behind | 3022 | 8 | 357 |
| depth | MEASURED | none (never assigned) | 0 | 0 | 0 |
| proximity | MEASURED | overlapping | 0 | 0 | 41 |
| proximity | MEASURED | adjacent | 1342 | 6 | 529 |
| proximity | MEASURED | separated (never assigned) | 0 | 0 | 0 |
| relative_motion | MEASURED | approach | 12 | 2 | 338 |
| relative_motion | MEASURED | recede | 0 | 0 | 304 |
| relative_motion | MEASURED | co_move | 1254 | 5 | 8 |
| relative_motion | MEASURED | pass | 54 | 2 | 397 |
| relative_motion | MEASURED | none (never assigned) | 0 | 0 | 0 |
| subject_state | MEASURED | moving | 8556 | 42 | 1418 |
| subject_state | MEASURED | stationary | 7036 | 14 | 1027 |
| object_state | MEASURED | moving | 1254 | 5 | 8 |
| object_state | MEASURED | stationary (never assigned) | 0 | 0 | 0 |
| comparative | MEASURED | larger | 1161 | 1 | 0 |
| comparative | MEASURED | taller | 1102 | 1 | 0 |
| comparative | MEASURED | faster | 89 | 1 | 0 |
| comparative | MEASURED | none (never assigned) | 0 | 0 | 0 |
| locomotion | LEARNED | walk | 4001 | 6 | 606 |
| locomotion | LEARNED | run | 657 | 6 | 161 |
| locomotion | LEARNED | fly | 877 | 5 | 55 |
| locomotion | LEARNED | swim | 175 | 4 | 73 |
| locomotion | LEARNED | creep | 176 | 5 | 56 |
| locomotion | LEARNED | jump | 226 | 6 | 71 |
| locomotion | LEARNED | none | 7036 | 14 | 1027 |
| contact_action | LEARNED | hold ⚠ | 0 | 0 | 76 |
| contact_action | LEARNED | ride | 420 | 1 | 0 |
| contact_action | LEARNED | bite ⚠ | 0 | 0 | 50 |
| contact_action | LEARNED | touch | 179 | 1 | 0 |
| contact_action | LEARNED | feed ⚠ | 0 | 0 | 45 |
| contact_action | LEARNED | kick ⚠ | 0 | 0 | 1 |
| contact_action | LEARNED | none (never assigned) | 0 | 0 | 0 |
| posture | LEARNED | stand | 4676 | 5 | 331 |
| posture | LEARNED | sit | 1508 | 4 | 602 |
| posture | LEARNED | lie | 852 | 5 | 94 |
| state_change | MEASURED | halt | 1549 | 4 | 148 |
| state_change | MEASURED | start (never assigned) | 0 | 0 | 0 |
| state_change | MEASURED | none (never assigned) | 0 | 0 | 0 |

⚠ = LEARNED value with < 50 base instances. MEASURED values are not held to the threshold: they are computed from boxes, not learned.

## Novel predicates: reachability

**Unreachable: 7 of 61.**

| novel predicate | reason |
|---|---|
| `bite` | LEARNED ingredient below threshold: contact_action=bite (0) |
| `drive` | no schema value fits (no ingredients) |
| `feed` | LEARNED ingredient below threshold: contact_action=feed (0) |
| `fight` | no schema value fits (no ingredients) |
| `hold` | LEARNED ingredient below threshold: contact_action=hold (0) |
| `kick` | LEARNED ingredient below threshold: contact_action=kick (0) |
| `pull` | no schema value fits (no ingredients) |

**Reachable: 54 of 61.** Of these, 10 depend on a MEASURED value that no base predicate carries, so no base example exists to validate the measurement:

- `away` — relative_motion=recede
- `creep_away` — relative_motion=recede
- `fly_away` — relative_motion=recede
- `jump_away` — relative_motion=recede
- `lie_inside` — proximity=overlapping
- `move_away` — relative_motion=recede
- `run_away` — relative_motion=recede
- `sit_inside` — proximity=overlapping
- `stand_inside` — proximity=overlapping
- `walk_away` — relative_motion=recede

## Reachable novel predicates, weakest LEARNED ingredient

| novel predicate | weakest LEARNED ingredient | base support | all ingredients |
|---|---|---:|---|
| `swim_behind` | locomotion=swim | 175 | depth=behind, locomotion=swim, subject_state=moving |
| `swim_beneath` | locomotion=swim | 175 | locomotion=swim, subject_state=moving, vertical=beneath |
| `swim_next_to` | locomotion=swim | 175 | locomotion=swim, proximity=adjacent, subject_state=moving |
| `creep_above` | locomotion=creep | 176 | locomotion=creep, subject_state=moving, vertical=above |
| `creep_away` | locomotion=creep | 176 | locomotion=creep, relative_motion=recede, subject_state=moving |
| `creep_next_to` | locomotion=creep | 176 | locomotion=creep, proximity=adjacent, subject_state=moving |
| `creep_past` | locomotion=creep | 176 | locomotion=creep, relative_motion=pass, subject_state=moving |
| `creep_toward` | locomotion=creep | 176 | locomotion=creep, relative_motion=approach, subject_state=moving |
| `jump_above` | locomotion=jump | 226 | locomotion=jump, subject_state=moving, vertical=above |
| `jump_away` | locomotion=jump | 226 | locomotion=jump, relative_motion=recede, subject_state=moving |
| `jump_behind` | locomotion=jump | 226 | depth=behind, locomotion=jump, subject_state=moving |
| `jump_past` | locomotion=jump | 226 | locomotion=jump, relative_motion=pass, subject_state=moving |
| `jump_with` | locomotion=jump | 226 | locomotion=jump, object_state=moving, relative_motion=co_move, subject_state=moving |
| `run_above` | locomotion=run | 657 | locomotion=run, subject_state=moving, vertical=above |
| `run_away` | locomotion=run | 657 | locomotion=run, relative_motion=recede, subject_state=moving |
| `run_beneath` | locomotion=run | 657 | locomotion=run, subject_state=moving, vertical=beneath |
| `run_next_to` | locomotion=run | 657 | locomotion=run, proximity=adjacent, subject_state=moving |
| `run_toward` | locomotion=run | 657 | locomotion=run, relative_motion=approach, subject_state=moving |
| `lie_above` | posture=lie | 852 | locomotion=none, posture=lie, subject_state=stationary, vertical=above |
| `lie_beneath` | posture=lie | 852 | locomotion=none, posture=lie, subject_state=stationary, vertical=beneath |
| `lie_inside` | posture=lie | 852 | locomotion=none, posture=lie, proximity=overlapping, subject_state=stationary |
| `lie_with` | posture=lie | 852 | locomotion=none, posture=lie, subject_state=stationary |
| `fly_away` | locomotion=fly | 877 | locomotion=fly, relative_motion=recede, subject_state=moving |
| `fly_behind` | locomotion=fly | 877 | depth=behind, locomotion=fly, subject_state=moving |
| `fly_front` | locomotion=fly | 877 | depth=front, locomotion=fly, subject_state=moving |
| `fly_left` | locomotion=fly | 877 | horizontal=left, locomotion=fly, subject_state=moving |
| `fly_right` | locomotion=fly | 877 | horizontal=right, locomotion=fly, subject_state=moving |
| `sit_behind` | posture=sit | 1508 | depth=behind, locomotion=none, posture=sit, subject_state=stationary |
| `sit_beneath` | posture=sit | 1508 | locomotion=none, posture=sit, subject_state=stationary, vertical=beneath |
| `sit_inside` | posture=sit | 1508 | locomotion=none, posture=sit, proximity=overlapping, subject_state=stationary |
| `sit_next_to` | posture=sit | 1508 | locomotion=none, posture=sit, proximity=adjacent, subject_state=stationary |
| `walk_above` | locomotion=walk | 4001 | locomotion=walk, subject_state=moving, vertical=above |
| `walk_away` | locomotion=walk | 4001 | locomotion=walk, relative_motion=recede, subject_state=moving |
| `walk_beneath` | locomotion=walk | 4001 | locomotion=walk, subject_state=moving, vertical=beneath |
| `walk_past` | locomotion=walk | 4001 | locomotion=walk, relative_motion=pass, subject_state=moving |
| `walk_toward` | locomotion=walk | 4001 | locomotion=walk, relative_motion=approach, subject_state=moving |
| `stand_above` | posture=stand | 4676 | locomotion=none, posture=stand, subject_state=stationary, vertical=above |
| `stand_beneath` | posture=stand | 4676 | locomotion=none, posture=stand, subject_state=stationary, vertical=beneath |
| `stand_inside` | posture=stand | 4676 | locomotion=none, posture=stand, proximity=overlapping, subject_state=stationary |
| `stand_with` | posture=stand | 4676 | locomotion=none, posture=stand, subject_state=stationary |
| `above` | — (all MEASURED) | — | vertical=above |
| `away` | — (all MEASURED) | — | relative_motion=recede |
| `beneath` | — (all MEASURED) | — | vertical=beneath |
| `move_above` | — (all MEASURED) | — | subject_state=moving, vertical=above |
| `move_away` | — (all MEASURED) | — | relative_motion=recede, subject_state=moving |
| `move_next_to` | — (all MEASURED) | — | proximity=adjacent, subject_state=moving |
| `move_past` | — (all MEASURED) | — | relative_motion=pass, subject_state=moving |
| `move_toward` | — (all MEASURED) | — | relative_motion=approach, subject_state=moving |
| `past` | — (all MEASURED) | — | relative_motion=pass |
| `stop_above` | — (all MEASURED) | — | state_change=halt, vertical=above |
| `stop_beneath` | — (all MEASURED) | — | state_change=halt, vertical=beneath |
| `stop_next_to` | — (all MEASURED) | — | proximity=adjacent, state_change=halt |
| `stop_with` | — (all MEASURED) | — | state_change=halt |
| `toward` | — (all MEASURED) | — | relative_motion=approach |
