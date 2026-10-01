# Stage 0c — final schema and two empirical checks

Data only: no GPU, no checkpoints, no model code, no caching. Nothing outside
this directory was modified. Every empirical figure comes from **base-predicate
instances in the training annotations**. Test annotations were never opened.
Novel-predicate instances were never used as evidence.

```
python3 pilot_analysis/stage0c/stage0c.py      # ~15 s, CPU
```

| file | contents |
|---|---|
| `phi_v3.json` | Task 1 schema as specified |
| `phi_v3_stop_reverted.json` | the same, with stop_* reverted per Task 3's conclusion (C) |
| `schema_v3.md` | Task 1: support, reachability, collisions, subset pairs, ride/drive test |
| `subset_pairs_v3.csv` | every strict-subset pair, with a column marking pairs new since stage 0b |
| `cooccurrence.md` | Task 2 |
| `stop_halt.md` | Task 3 |

**Camera compensation** uses `VidVRD_ECC_train.json`. For 0-based frame `t`,
`ecc[str(t+1)]` warps frame `t-1` into frame `t`. I checked this on 63k box
steps: it gives the lowest residual, and the inverse and off-by-one keys are
worse. 17 of the 800 training videos have no ECC matrices and are excluded from
the speed measurements.

## Task 1 — schema v3

**contact=contact base support: 599** (`ride` 420 + `touch` 179). Every other
carrier is novel.

| | stage 0b | v3 | v3 + coarse-contact fallback ¹ |
|---|---:|---:|---:|
| unreachable novel predicates | 7 | **5** | **1** |
| novel collision groups (all-novel + base-novel) | 0 | **3** ⚠ | **4** ⚠ |
| novel-subset pairs (the subset predicate is novel) | 87 | **65** | **93** |

¹ A contact_action value under 50 instances is dropped whenever the predicate
also carries contact=contact, which has 599. This is my reading of "fallback
channel". The strict column applies the LEARNED threshold as written.

- **Unreachable, strict:** `bite`, `feed`, `hold`, `kick` (their contact_action
  values have 0 base instances) and `fight` (no ingredients). `drive` and `pull`
  are now reachable: their only LEARNED ingredient is contact=contact.
  **With the fallback:** only `fight`.
- **⚠ Three new collisions, all created by change (a).** Adding
  proximity=adjacent makes each stationary `*_with` identical to the matching
  `*_next_to`:
  - `stand_with`\* ≡ `stand_next_to` (base-novel)
  - `lie_with`\* ≡ `lie_next_to` (base-novel)
  - `stop_with`\* ≡ `stop_next_to`\* (all-novel)

  The schema has no value that separates "with" from "next to" for a stationary
  pair.
- **⚠ The fallback adds a fourth collision:** `bite` ≡ `feed` ≡ `hold` ≡ `kick`,
  since all four reduce to {contact=contact}. The coarse channel makes them
  reachable but indistinguishable from each other.
- **Novel-subset pairs:** 87 → 65. Change (a) removes 23, the `*_with` ⊂
  same-posture pairs. One new pair: **`pull` ⊂ `drive`** ⚠. With the fallback
  the count rises to 93: each of `bite`, `feed`, `hold` and `kick` becomes a
  subset of `drive`, `pull`, `ride`, `touch` and the three `*_inside`
  predicates.

### Task 1(d): can geometry separate drive from ride? Not with the recipe as drafted.

Tested on base `ride`, 420 instances, 418 of them with camera motion
available:

| | ride instances |
|---|---:|
| overlapping, with the subject centre above the object's | **97.4%** |
| contained (≥90% of the subject box inside the object box) | **1.9%** |
| subject box top above object box top (by frame) | 98.5% |
| both boxes translate, in the same direction | 80.4% |
| subject–object offset stable over the extent | 88.5% |
| **meets every geometric condition in the drafted drive recipe** | **80.4%** |

Ride is "above + overlapping", almost never contained, and it co-moves. The
drive recipe asks for proximity=**overlapping** + co_move + object moving +
contact, and ride carries contact=contact too. So **drive's full recipe fires
on 4 out of 5 ride instances**. Geometry can separate the two only through
containment, which ride shows 1.9% of the time. The schema has no containment
value, though, so drive was drafted with `overlapping`. Whether drive instances
are actually contained is not testable here, because drive is novel.

## Task 2 — co-occurrence: (A) largely disjoint

| | overlap a same-spatial compound | overlap any other label on the pair (control) |
|---|---:|---:|
| bare `left`/`right`/`front`/`behind` (1,280) | **4.3%** (55) | 70.6% |
| `move_X` vs `manner_X` (1,974) | **0.4%** (8, all `move_beneath`/`jump_beneath`) | 59.8% |

The pairs are heavily multi-labelled in general, just never bare-plus-compound
or move-plus-manner. **Annotators chose one label per family, so bare and
`move_` labels assert that the manner is absent. Unset axes of these families
should be scored as "asserts absence", not marginalised out.** This licenses
that reading only for locomotion (and, for bare terms, posture). It says
nothing about other unset axes.

## Task 3 — is `stop` a halt? (C) No: stop_* cannot be told apart from stand_*

| | stop_* | stand_* (same spatial terms) |
|---|---:|---:|
| before-window clearly faster than extent | 14.6% | 14.9% |
| moving→stationary transition inside extent | 8.3% | 10.7% |
| median speed before the extent | 0.0011 | 0.0017 |

Every threshold-free AUC (stop vs stand, on extent speed, before-window speed,
their ratio, and within-extent deceleration) has a CI that includes 0.5. No
halt shows up inside the extent, and none before it either. Only 404 of the
1,533 stop instances have a usable before-window. Even so, stop's before-window
is *slower* than stand's, so the direction is wrong for (B), not just noisy.
Caveat: the "clearly moving" cutoff is close to box-jitter level, so the rate
columns are coarse; the AUCs don't depend on it.

**⚠ What `stop` actually encodes is a stationary vehicle.** All 1,549 base
stop_* subjects are vehicles (car 1,126, watercraft 170, bicycle 86, …). Only 6
of the 7,036 stand/sit/lie subjects are. That is why the category-matched
control the brief implied is empty. `stop` is the vehicle counterpart of
stand/sit/lie: the label is picked by subject category, not by motion history.

**Reverted per (C)** in `phi_v3_stop_reverted.json`. Each novel stop_*
predicate becomes a strict subset of its posture-marked counterparts, for
example `stop_above` ⊂ `lie_above`, `sit_above`, `stand_above`. Formally they
stay reachable, because their only LEARNED ingredient, locomotion=none, has
7,036 base instances. But nothing in phi or in the annotated extent separates
them from those supersets; only subject category does, and phi has no axis for
that. `stop_next_to` ≡ `stop_with` persists.
