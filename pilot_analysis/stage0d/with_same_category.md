# Stage 0d — Task 2: is `with` a same-category convention?

BASE training instances only. `same` = subject and object have the same object category. Novel `*_with` / `*_next_to` predicates are masked labels and are excluded (listed).

Base rate over all 22616 base training instances: **59.2%** same-category.

| family | predicate | same / instances | fraction |
|---|---|---:|---:|
| `with` | `fly_with` | 674 / 674 | 100.0% |
| `with` | `walk_with` | 226 / 294 | 76.9% |
| `with` | `move_with` | 154 / 156 | 98.7% |
| `with` | `run_with` | 48 / 88 | 54.5% |
| `with` | `swim_with` | 42 / 42 | 100.0% |
| `with` | **pooled** | **1144 / 1254** | **91.2%** |
| `next_to` | `stand_next_to` | 374 / 479 | 78.1% |
| `next_to` | `walk_next_to` | 281 / 392 | 71.7% |
| `next_to` | `next_to` | 105 / 169 | 62.1% |
| `next_to` | `fly_next_to` | 156 / 156 | 100.0% |
| `next_to` | `lie_next_to` | 77 / 140 | 55.0% |
| `next_to` | `jump_next_to` | 5 / 6 | 83.3% |
| `next_to` | **pooled** | **998 / 1342** | **74.4%** |

Excluded (novel): `with`: jump_with, lie_with, stand_with, stop_with; `next_to`: creep_next_to, move_next_to, run_next_to, sit_next_to, stop_next_to, swim_next_to. No bare `with` exists.

**Same verb, both families base** (with − next_to, video-clustered 95% CI):

| verb | with | next_to | difference | 95% CI |
|---|---:|---:|---:|---|
| fly | 674/674 | 156/156 | +0.0% | +0.0% … +0.0% |
| walk | 226/294 | 281/392 | +5.2% | -13.1% … +21.5% |

## Conclusion: (B) no clear split

The pooled gap (91.2% vs 74.4%) is a verb-mix artefact. `fly_with` is 674 of the 1254 `with` instances, and 666 of those are airplane–airplane (formation flight), so it is 100% same-category. But `fly_next_to` is 100% same-category too, for the same reason. Holding the verb fixed, there is no gap whose CI excludes zero. `next_to` is itself predominantly same-category (74.4%, against a 59.2% base rate). So the second half of criterion (A) fails outright.

Also: no stationary `*_with` is base, and those three are exactly the novel predicates the axis was meant to separate. So even a real gap among moving verbs would have been an extrapolation to them.

**Applied:** stage-0c change (a) reverted. proximity=adjacent removed from `lie_with`, `stand_with`, `stop_with`.

