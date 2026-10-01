# Stage 0f — inverse-pair coverage

Source: `pilot_analysis/stage0e/phi_frozen.json` and `data/vidvrd/data/openvoc_pred_class_spilt_info.json`. Evidence: BASE labels in the 800 training videos only. Novel labels were not read, and test annotations were not opened.

## 1. The inverse map

inverse(p) = the predicate whose phi equals phi(p) with its spatial value mirrored (left↔right, above↔beneath, front↔behind), every other axis unchanged. I matched on full phi equality, and cross-checked that against plain string substitution: the two agree for all 132 predicates.

- 70 predicates carry a mirrorable spatial term, and 68 of them have their inverse in the vocabulary.
- The 2 mirrorable predicates whose mirror is not in the vocabulary: `fly_above`, `swim_beneath`.
- 62 predicates have no mirrorable term (next_to, with, toward, away, past, inside, comparatives, actions).

⚠ **The map mirrors the spatial term and keeps the verb, but an argument swap moves the verb onto the other entity.** From A `stand_left` B, swapping the arguments gives *B is right of A*, and says nothing about B standing. So the same-verb inverse is exact only when both entities share the verb, posture and subject_kind. Section 4 measures how often that holds.

## 2. The 61 novel predicates

| novel predicate | axis | inverse | inverse split | base inverse, training instances |
|---|---|---|---|---:|
| `creep_above` | above/beneath | `creep_beneath` | base | 3 |
| `jump_above` | above/beneath | `jump_beneath` | base | 74 |
| `jump_behind` | front/behind | `jump_front` | base | 52 |
| `move_above` | above/beneath | `move_beneath` | base | 302 |
| `sit_behind` | front/behind | `sit_front` | base | 296 |
| `sit_beneath` | above/beneath | `sit_above` | base | 398 |
| `swim_behind` | front/behind | `swim_front` | base | 44 |
| `above` | above/beneath | `beneath` | novel | — |
| `beneath` | above/beneath | `above` | novel | — |
| `fly_behind` | front/behind | `fly_front` | novel | — |
| `fly_front` | front/behind | `fly_behind` | novel | — |
| `fly_left` | left/right | `fly_right` | novel | — |
| `fly_right` | left/right | `fly_left` | novel | — |
| `lie_above` | above/beneath | `lie_beneath` | novel | — |
| `lie_beneath` | above/beneath | `lie_above` | novel | — |
| `run_above` | above/beneath | `run_beneath` | novel | — |
| `run_beneath` | above/beneath | `run_above` | novel | — |
| `stand_above` | above/beneath | `stand_beneath` | novel | — |
| `stand_beneath` | above/beneath | `stand_above` | novel | — |
| `stop_above` | above/beneath | `stop_beneath` | novel | — |
| `stop_beneath` | above/beneath | `stop_above` | novel | — |
| `walk_above` | above/beneath | `walk_beneath` | novel | — |
| `walk_beneath` | above/beneath | `walk_above` | novel | — |
| `away` | — | — | — | — |
| `bite` | — | — | — | — |
| `creep_away` | — | — | — | — |
| `creep_next_to` | — | — | — | — |
| `creep_past` | — | — | — | — |
| `creep_toward` | — | — | — | — |
| `drive` | — | — | — | — |
| `feed` | — | — | — | — |
| `fight` | — | — | — | — |
| `fly_away` | — | — | — | — |
| `hold` | — | — | — | — |
| `jump_away` | — | — | — | — |
| `jump_past` | — | — | — | — |
| `jump_with` | — | — | — | — |
| `kick` | — | — | — | — |
| `lie_inside` | — | — | — | — |
| `lie_with` | — | — | — | — |
| `move_away` | — | — | — | — |
| `move_next_to` | — | — | — | — |
| `move_past` | — | — | — | — |
| `move_toward` | — | — | — | — |
| `past` | — | — | — | — |
| `pull` | — | — | — | — |
| `run_away` | — | — | — | — |
| `run_next_to` | — | — | — | — |
| `run_toward` | — | — | — | — |
| `sit_inside` | — | — | — | — |
| `sit_next_to` | — | — | — | — |
| `stand_inside` | — | — | — | — |
| `stand_with` | — | — | — | — |
| `stop_next_to` | — | — | — | — |
| `stop_with` | — | — | — | — |
| `swim_beneath` | above/beneath | — | — | — |
| `swim_next_to` | — | — | — | — |
| `toward` | — | — | — | — |
| `walk_away` | — | — | — | — |
| `walk_past` | — | — | — | — |
| `walk_toward` | — | — | — | — |

**Transfer set (inverse is base): 7 of 61.** Inverse is novel: 16. No inverse: 38.

## 3. Transfer set — base inverse support

| novel predicate | base inverse | training instances |
|---|---|---:|
| `sit_beneath` | `sit_above` | 398 |
| `move_above` | `move_beneath` | 302 |
| `sit_behind` | `sit_front` | 296 |
| `jump_above` | `jump_beneath` | 74 |
| `jump_behind` | `jump_front` | 52 |
| `swim_behind` | `swim_front` | 44 |
| `creep_above` | `creep_beneath` | 3 |

Total: 1169 base training instances across the 7 inverses.

## 4. Do annotators label the reversed pair? ⚠ the critical number

Anchor: every BASE training instance whose predicate has a mirrorable spatial term. Check the reversed pair (object as subject), same video, temporally overlapping extent, BASE labels only.

| axis | anchors | exact mirror predicate | mirror spatial term, different predicate | SAME spatial term (contradiction) | other base label, no term on this axis | no base label on reversed pair |
|---|---:|---:|---:|---:|---:|---:|
| left/right | 8254 | 51.1% | 48.7% | 0.0% | 0.1% | 0.1% |
| above/beneath | 813 | 0.0% | 73.7% | 0.0% | 13.9% | 12.4% |
| front/behind | 6391 | 43.7% | 50.7% | 0.0% | 4.3% | 1.3% |

**Mirror annotated in any form** (exact or with a different verb), and the exact-mirror rate restricted to anchors whose mirror predicate is base (so an exact match was possible):

| axis | mirror annotated, any verb | exact mirror, when the mirror is base |
|---|---:|---:|
| left/right | 8239/8254 (99.8%) | 4217/8254 (51.1%) |
| above/beneath | 599/813 (73.7%) | — |
| front/behind | 6033/6391 (94.4%) | 2793/5999 (46.6%) |

Most common different-predicate mirrors (anchor → label on the reversed pair):

- left/right: `walk_left → stand_right` 366, `stand_right → walk_left` 363, `walk_right → stand_left` 346, `stand_left → walk_right` 336, `move_right → stop_left` 196, `stop_left → move_right` 196
- above/beneath: `move_beneath → sit_above` 287, `sit_above → move_beneath` 287, `jump_beneath → sit_above` 72, `sit_above → jump_beneath` 72, `move_beneath → fly_above` 32, `fly_above → move_beneath` 32
- front/behind: `stand_behind → walk_front` 338, `walk_front → stand_behind` 332, `walk_behind → stand_front` 284, `stand_front → walk_behind` 273, `stop_front → walk_behind` 147, `walk_behind → stop_front` 147

**Geometry.** The unannotated reverse direction is geometrically the mirror of the annotated one by construction: if A is left of B in the image, B is right of A. So the only question is whether the annotated label agrees with the boxes. Here is that sign test, split by whether the reverse was annotated:

| axis | reverse annotated: label agrees with boxes | reverse NOT annotated: label agrees with boxes |
|---|---:|---:|
| left/right | 7963/8239 (96.7%) | 12/15 (80.0%) |
| above/beneath | 599/599 (100.0%) | 213/214 (99.5%) |
| front/behind | 5072/6033 (84.1%) | 311/358 (86.9%) |

Sign rules (camera frame, per 0b): left = smaller centre x; above = smaller centre y; front = lower bottom edge. Averaged over the anchor's extent.

## Conclusion

**1. The spatial inverse is confirmed by the annotators.** The reversed pair carries the mirror spatial term for 99.8% of left/right anchors and 94.4% of front/behind. The reversed pair carries the SAME term (a contradiction) 1 time in 15458 anchors. above/beneath reaches 73.7%, but that is a lower bound: most above/beneath compounds are novel, their labels were not read, and 12.4% of reversed pairs show no base label at all. Where the reverse is unannotated, the annotated label agrees with the boxes just as often as where it is annotated. The omissions are gaps in labelling, not geometric disagreement.

**2. The verb does not invert. ⚠** The *exact* same-verb mirror appears on only 51.1% of left/right anchors and 46.6% of front/behind anchors, even where that mirror is a base predicate. The other half carry the OBJECT's own verb: `walk_left` ↔ `stand_right`, `move_right` ↔ `stop_left`. On above/beneath the reversed pair of `sit_above` is `move_beneath` or `jump_beneath` (287 and 72 instances). The thing below is a moving entity, not a sitting one. Annotators treat the verb as a property of the subject, as expected under argument swap. The same-verb inverse holds only when both entities happen to share the verb.

**3. So the transfer set adds no reachability under the frozen schema.** 7 of 61 novel predicates have a base inverse. But in `phi_frozen` the part that inverts cleanly, the spatial term, is a MEASURED axis and needs no transfer. The part that would need transfer, the verb or posture, is exactly the part that does not invert. `sit_beneath` gets `beneath` from the boxes and `sit` from all 1508 base sit_* instances. Swapped `sit_above` instances show a moving thing beneath a sitting thing, which teaches nothing about sitting beneath. All 7 are already reachable without the inverse map. `creep_above`'s inverse has only 3 instances anyway.

What the bidirectional annotation does give is a free consistency check. For left/right and front/behind, a prediction on (A, B) implies the mirror spatial term on (B, A), and the annotators agree with that 94–100% of the time. That is a constraint on the measured spatial axes, not a source of verb supervision.

