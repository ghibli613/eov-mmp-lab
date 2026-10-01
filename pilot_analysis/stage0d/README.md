# Stage 0d — schema v4 freeze

Data only: base training annotations only. I never opened a test annotation,
used no GPU, and modified nothing outside this directory.

```
python3 pilot_analysis/stage0d/stage0d.py      # ~20 s, CPU
```

| file | contents |
|---|---|
| `phi_v4.json` | **frozen schema v4**: Task 1 changes plus the Task 2 (B) revert |
| `tables_v4.md` | Task 3 tables with every group and pair listed |
| `with_same_category.md` | Task 2 per-predicate table |
| `subset_pairs_v4.csv` | every strict-subset pair; `new_vs_v3` column |
| `summary.json` | all numbers |

## Task 1 — changes applied

- **(a)** Dropped the `state_change` axis.
- **(b)** Added `subject_kind` (LEARNED). The vehicle set is derived as the
  subject categories of base stop_* instances: **airplane, bicycle, bus, car,
  motorcycle, train, watercraft** (car 1,126, watercraft 170, bicycle 86, bus
  52, motorcycle 46, airplane 46, train 23). stop_* is now stationary +
  locomotion=none + subject_kind=vehicle + spatial.
- **(c)** Added proximity=`contained` (MEASURED, ≥90% of the subject box inside
  the object box) for `drive` and the three `*_inside` predicates. **Base
  `ride`: 8/420 (1.9%) contained**, meaning the box is contained in the
  majority of its frames. Counting by median gives the same 8.
- **(d)** Dropped the coarse-contact fallback. Reachability is strict.

| value | base support |
|---|---:|
| subject_kind=vehicle | 1,549 (stop_* only) |
| subject_kind=animate | **0**, since no predicate asserts it |
| proximity=contained | **0**, since all four carriers are novel (MEASURED, so no threshold applies) |
| proximity=overlapping | 0, no longer assigned to anything |
| contact=contact | 599 |

## Task 2 — `with` is not a same-category convention: (B)

| | same-category |
|---|---:|
| all base training instances (base rate) | 59.2% |
| `*_with`, pooled (5 base predicates) | 91.2% (1,144/1,254) |
| `*_next_to`, pooled (6 base predicates, including bare `next_to`) | 74.4% (998/1,342) |
| same verb: `fly_with` vs `fly_next_to` | 100% vs 100% |
| same verb: `walk_with` vs `walk_next_to` | 76.9% vs 71.7%, difference +5.2 pts, CI −13.1 … +21.5 |

The pooled gap is a verb-mix artefact: 674 of the 1,254 `with` instances are
`fly_with`, and 666 of those are airplane–airplane. Once the verb is held
fixed, there is no gap. `next_to` is itself mostly same-category, so criterion
(A) fails on its second half. No stationary `*_with` predicate is base, so a
real gap would still have been an extrapolation to the three novel predicates
it was meant for.

**Applied:** 0c change (a) is reverted, and proximity=adjacent is removed from
`stand_with`, `lie_with` and `stop_with`. The 23 subset pairs it had removed
come back, listed in `tables_v4.md`. Each stationary `*_with` is once again a
strict subset of every same-verb spatial predicate, e.g. `stand_with` ⊂
`stand_left`.

## Task 3 — final tables

Reachability, with the LEARNED threshold only:

| | 0c v3 strict | v4 (Task 1 only) | **v4 final** |
|---|---:|---:|---:|
| unreachable novel predicates | 5 | 5 | **5** |

The five are `bite`, `feed`, `hold`, `kick` (each contact_action value has 0
base instances) and `fight` (no ingredients).

Identical-phi collisions:

| | 0c v3 strict | v4 (Task 1 only) | **v4 final** |
|---|---:|---:|---:|
| all-novel | 1 | 1 | **0** |
| base-novel | 2 | 2 | **0** |
| all-base | 1 | 1 | 1 |

Strict-subset pairs:

| | 0c v3 strict | v4 (Task 1 only) | **v4 final** |
|---|---:|---:|---:|
| all pairs | 373 | 373 | 393 |
| novel-subset pairs (the subset predicate is novel) | 65 | 65 | **88** |

- **Task 1 alone changes no count.** It creates no collision and no subset pair.
- **Collisions:** the three `*_with` ≡ `*_next_to` collisions are gone. The
  only group left is all-base: `chase` ≡ `fall_off` ≡ `follow`.
- **⚠ Novel-subset pairs: 23 new vs v3.** All 23 are the `*_with` pairs that
  Task 2 (B) brings back. Nothing else is new, and no pair was removed. The
  count of 88 is stage 0b's 87 plus `pull` ⊂ `drive`.

## ⚠ Flags (not applied — outside this brief)

1. **`move` is the vehicle counterpart of walk/run, the same way `stop` is for
   stand.** 1,954 of the 1,974 base `move_*` subjects (99.0%) are vehicles; for
   `walk` it is 2 of 4,001. `fly` subjects are 97.3% airplanes. Stage 0b's
   premise that "move_X is an underspecified subset of walk_X" is therefore
   wrong about the data: move and walk have almost disjoint subjects. This fits
   0c Task 2, where they never co-occur. Giving `move_*` subject_kind=vehicle
   would break all 60 `move_X` ⊂ `manner_X` pairs, 26 of which have a novel
   subset.
2. **`animate` is never assigned.** Only `vehicle` has predicate-level support.
   Measured from object labels instead, 5,042 base instances have a vehicle
   subject, across 34 base predicates (the 4 stop_* included). That is the signal available if
   subject_kind is supervised from categories rather than from predicates.
3. **`contained` has no base example.** It joins `recede` as a MEASURED value
   that only novel predicates use. The 1.9% ride figure shows it separates
   drive from ride, but only if drive instances really are contained, and
   that can't be checked on base data.
