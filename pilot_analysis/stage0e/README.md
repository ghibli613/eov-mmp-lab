# Stage 0e — apply and freeze

This was data-only work. I used base training annotations only, never opened a
test annotation, used no GPU, and modified nothing outside this directory.
This is the last stage-0 round.

```
python3 pilot_analysis/stage0e/stage0e.py      # ~20 s, CPU
```

| file | contents |
|---|---|
| `phi_frozen.json` | the frozen schema, 132 predicates |
| `scoring_rule.md` | the rule for stage 1, with two items marked ⚠ DECISION NEEDED |
| `vehicle_families.md` | Task 1 per-predicate vehicle rates |
| `subset_pairs_frozen.csv` | every strict-subset pair, plus whether it is a real domination under the scoring rule |
| `summary.json` | all numbers |

## Comparison with v4 final

| | v4 final | **frozen** |
|---|---:|---:|
| unreachable novel predicates | 5 | **5** |
| collisions: all-novel / base-novel / all-base | 0 / 0 / 1 | **0 / 0 / 1** |
| strict-subset pairs | 393 | **333** |
| novel-subset pairs | 88 | **62** |

**Nothing new:** there are no new collisions and no new subset pairs of any
kind. The change is 60 pairs removed, all of them `move_X` ⊂ `manner_X`. 26 of
those had a novel subset, which takes the novel-subset count from 88 to 62. The
five unreachable predicates are still `bite`, `feed`, `hold`, `kick` and
`fight`. The only collision is still all-base: `chase` ≡ `fall_off` ≡ `follow`.

## Conclusion 1 — `move_*` applied, `fly_*` not

The rule applied to each family: every base member with ≥10 instances must be
≥95% vehicle.

- **`move_*`: applied**, to all 11 base and novel predicates. Every base member
  is at least 96.4% vehicle, and the family is 99.0%. The non-vehicle subjects
  are 20 skateboards.
- **`fly_*`: not applied.** The family rate of 97.3% is carried by
  `fly_with` (98.8%) and `fly_next_to` (100%), which are airplane formations.
  **`fly_above` is only 58.3% vehicle: 15 of its 36 subjects are birds.** Fly
  is a genuinely mixed verb, and applying vehicle would make every novel
  `fly_*` reject birds.

## Conclusion 2 — subject_kind from category labels

The sets are derived from the annotations. Vehicle means the subject categories
of base `stop_*`. Animate means the subject categories of base
stand/sit/lie/walk/run/creep/swim, minus the vehicles.

| subject kind | categories | base training instances |
|---|---:|---:|
| vehicle | 7 | **5,042** |
| animate | 24 | **17,304** |
| neither: ball, frisbee, skateboard, sofa | 4 | 270 |

subject_kind is marked **SUPPLIED**: it is read off the object classifier, it
has no attribute head, and it is exempt from the LEARNED threshold. Vehicles
appear as subjects of animate verbs in only 8 instances (e.g. `lie_right` with
a motorcycle).

## Conclusion 3 — the specified `contained` does not separate ride. ⚠

On base `ride` (420 instances), the brief's parameter-free comparative
(subject-in-object > object-in-subject, and subject centre within the object
box) classes **191 of 420 (45.5%) as contained**. Most are person–horse (92)
and person–motorcycle (61). A rider's centre, the hip, is inside the mount's
box in 332 of the 420 instances. So the brief's definition fails its own
acceptance test.

**Proposed instead:** keep the fraction comparison, but replace the centre test
with a top-edge test, *subject top edge not above object top edge*. This is
still parameter-free and classes **1 of 420 (0.2%)** of ride as contained. It
works because riders protrude above their mounts (98.5% of frames in 0c).
Strict box nesting gives 0 of 420, but a single pixel of box overshoot defeats
it. I have not frozen either version: `scoring_rule.md` lists all four
candidates for the chat to choose. No positive example of containment exists in
base data, because every `contained` carrier is novel.

## ⚠ Second decision — the absence rule is licensed only for compositional predicates

The brief's rule makes unset locomotion or posture assert absence everywhere.
0c tested only bare spatial terms and `move_*`, so I checked every family,
counting overlaps with a *different* verb, on the same pair, over overlapping
extents:

- **Holds (≤3.5% conflict):** bare spatial terms and all 11 verb families.
- **Fails (23–80% conflict):** every single-morpheme base predicate with more
  than 10 instances. `chase` 69.9% and `follow` 71.1% overlap a locomotion
  compound. `ride` overlaps a posture compound 80.0% of the time, and `touch`,
  `watch` and `play` 51–57%.

Proposed: apply absence to compositional predicates only, and marginalise for
single-morpheme ones. Among the novel predicates, that affects `bite`, `feed`,
`hold`, `kick`, `pull`, `drive` and `fight`. **This does not change which novel
predicates are dominated:** the same 31 remain under either version. What it
changes is whether, for example, a person who is sitting can score as `hold`.

The parameter-free set is the 14 expected: `above`, `beneath`, `away`, `past`,
`toward`, five `move_*` and four `stop_*`. These have no positive learned
ingredient, but under the absence rule they still consult the learned
locomotion and posture heads, which have to say "absent".
