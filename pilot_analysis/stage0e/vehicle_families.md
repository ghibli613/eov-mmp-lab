# Stage 0e — Task 1: vehicle verb families

Vehicle categories (subjects of base stop_*): airplane, bicycle, bus, car, motorcycle, train, watercraft.

Rule: apply subject_kind=vehicle to a verb family only if every base member with ≥10 instances has ≥95% vehicle subjects. A family rate alone can hide a mixed member.

## `move_*`: family 1954/1974 (99.0%) → **APPLIED**

| base predicate | vehicle subjects | non-vehicle subjects |
|---|---:|---|
| `move_behind` | 359/359 (100.0%) | — |
| `move_beneath` | 291/302 (96.4%) | skateboard 11 |
| `move_front` | 388/394 (98.5%) | skateboard 6 |
| `move_left` | 358/359 (99.7%) | skateboard 1 |
| `move_right` | 402/404 (99.5%) | skateboard 2 |
| `move_with` | 156/156 (100.0%) | — |

## `fly_*`: family 853/877 (97.3%) → **NOT APPLIED**

| base predicate | vehicle subjects | non-vehicle subjects |
|---|---:|---|
| `fly_above` | 21/36 (58.3%) | bird 15 |
| `fly_next_to` | 156/156 (100.0%) | — |
| `fly_past` | 8/9 (88.9%) | bird 1 |
| `fly_toward` | 2/2 (100.0%) | — |
| `fly_with` | 666/674 (98.8%) | bird 8 |

