# Stage 0c — Task 2: label co-occurrence

BASE training annotations only. A hit = the same (video, subject tid, object tid) carries both labels over temporally overlapping extents. Novel compounds are masked labels and are not counted as partners; they are listed as excluded.

`any other label` is the control: the fraction of anchor instances that overlap ANY other base label on the same pair (e.g. `larger`, `watch`). It shows how multi-labelled pairs are in general.

## Bare spatial term vs. verb compounds sharing it

| bare | instances | overlap a compound | fraction | any other label (control) | partners (base) | excluded (novel) |
|---|---:|---:|---:|---:|---|---|
| `left` | 382 | 21 | 5.5% | 67.8% | creep_left, jump_left, lie_left, move_left, run_left, sit_left, stand_left, stop_left, swim_left, walk_left | fly_left |
| `right` | 341 | 16 | 4.7% | 79.2% | creep_right, jump_right, lie_right, move_right, run_right, sit_right, stand_right, stop_right, swim_right, walk_right | fly_right |
| `front` | 291 | 7 | 2.4% | 63.6% | creep_front, jump_front, lie_front, move_front, run_front, sit_front, stand_front, stop_front, swim_front, walk_front | fly_front |
| `behind` | 266 | 11 | 4.1% | 71.4% | creep_behind, lie_behind, move_behind, run_behind, stand_behind, stop_behind, walk_behind | fly_behind, jump_behind, sit_behind, swim_behind |

Reverse direction — of each compound's instances, how many overlap the bare term:

| bare | compound: overlapping / instances |
|---|---|
| `left` | creep_left 0/40; jump_left 1/50; lie_left 0/172; move_left 0/359; run_left 3/141; sit_left 3/418; stand_left 10/1186; stop_left 0/431; swim_left 0/30; walk_left 9/915 |
| `right` | creep_right 0/48; jump_right 0/34; lie_right 1/191; move_right 0/404; run_right 3/151; sit_right 9/396; stand_right 1/1242; stop_right 0/337; swim_right 0/59; walk_right 4/927 |
| `front` | creep_front 1/54; jump_front 0/52; lie_front 1/186; move_front 0/394; run_front 1/127; sit_front 4/296; stand_front 2/865; stop_front 0/308; swim_front 0/44; walk_front 1/752 |
| `behind` | creep_behind 2/31; lie_behind 0/163; move_behind 0/359; run_behind 0/105; stand_behind 4/904; stop_behind 0/473; walk_behind 5/721 |

## move_X vs. manner_X

| move_X | instances | overlap a manner_X | fraction | any other label (control) | partners (base) | excluded (novel) |
|---|---:|---:|---:|---:|---|---|
| `move_behind` | 359 | 0 | 0.0% | 74.4% | creep_behind, run_behind, walk_behind | fly_behind, jump_behind, swim_behind |
| `move_beneath` | 302 | 8 | 2.6% | 7.6% | creep_beneath, jump_beneath | run_beneath, swim_beneath, walk_beneath |
| `move_front` | 394 | 0 | 0.0% | 60.7% | creep_front, jump_front, run_front, swim_front, walk_front | fly_front |
| `move_left` | 359 | 0 | 0.0% | 71.6% | creep_left, jump_left, run_left, swim_left, walk_left | fly_left |
| `move_right` | 404 | 0 | 0.0% | 62.9% | creep_right, jump_right, run_right, swim_right, walk_right | fly_right |
| `move_with` | 156 | 0 | 0.0% | 89.7% | fly_with, run_with, swim_with, walk_with | jump_with |

Reverse direction:

| move_X | manner_X: overlapping / instances |
|---|---|
| `move_behind` | creep_behind 0/31; run_behind 0/105; walk_behind 0/721 |
| `move_beneath` | creep_beneath 0/3; jump_beneath 9/74 |
| `move_front` | creep_front 0/54; jump_front 0/52; run_front 0/127; swim_front 0/44; walk_front 0/752 |
| `move_left` | creep_left 0/40; jump_left 0/50; run_left 0/141; swim_left 0/30; walk_left 0/915 |
| `move_right` | creep_right 0/48; jump_right 0/34; run_right 0/151; swim_right 0/59; walk_right 0/927 |
| `move_with` | fly_with 0/674; run_with 0/88; swim_with 0/42; walk_with 0/294 |

**Pooled:** bare terms 55/1280 (4.3%) overlap a compound (control: 70.6% overlap any other label); move_X 8/1974 (0.4%) overlap a manner_X (control: 59.8%).

## Conclusion

**(A) Largely disjoint.** A bare spatial term shares an overlapping extent with a verb compound of the same spatial value on 4.3% of its instances, although 70.6% of those same instances overlap *some* other label on the pair. The pairs are heavily multi-labelled; they are just not labelled bare-plus-compound. The reverse direction agrees: no compound overlaps its bare term more than a few percent of the time. move_X is disjoint from manner_X (0.4%); the only exceptions are 8 `move_beneath`/`jump_beneath` overlaps.

Annotators picked one label from each family, not an underspecified one alongside a specific one. So bare and move_ labels behave as asserting the absence of the manner their compounds name. **Unset manner axes should be scored as 'asserts absence', not marginalised.**

Scope: this licenses 'absent' only for the axes these two families leave unset (locomotion, and posture for the bare terms). It does not test whether an unset `contact` or `relative_motion` means absence.

