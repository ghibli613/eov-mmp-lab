# Stage 0b — schema revision

Data only: no GPU, no checkpoints, no inference. Nothing outside this directory
was modified. The script reuses the stage-0 phi rules and then adds the two new
axes. Base support counts exclude the 3,301 masked novel instances.

```
python3 pilot_analysis/stage0b/stage0b.py      # ~1.5 min, CPU
```

| file | contents |
|---|---|
| `phi_b.json` | redrafted phi, all 132 predicates |
| `coverage_b.md` | Tasks 1–2: support per value with MEASURED/LEARNED tags, reachability |
| `collisions_b.md` | Task 3 |
| `reference_frame.md` | Task 4: full tables, every stratum |
| `subset_pairs.csv` | Task 5: every strict-subset pair, with the attributes only the superset has |
| `unresolved_b.md` | new judgement calls (the stage-0 ones still apply) |

## 1. New axes: base support

| axis | tag | value | base train instances | masked novel (not usable) |
|---|---|---|---:|---:|
| posture | LEARNED | stand | 4,676 | 331 |
| posture | LEARNED | sit | 1,508 | 602 |
| posture | LEARNED | lie | 852 | 94 |
| state_change | MEASURED | halt | 1,549 | 148 |
| state_change | MEASURED | start | 0 (never assigned) | 0 |
| state_change | MEASURED | none | 0 (never assigned) | 0 |

`stop_*` is now encoded as state_change=halt. Its subject_state and locomotion
are left unset, because an interval that contains a halt is both moving and
stationary. No predicate asserts `start`. `none` is unassigned under the
convention that an axis the predicate doesn't mention stays unset; see
`unresolved_b.md`.

## 2. Reachability, with the <50 threshold applied only to LEARNED axes

**7 of the 61 novel predicates are unreachable:**

| predicate | reason |
|---|---|
| `bite`, `feed`, `hold`, `kick` | the LEARNED contact_action value has **0** base instances |
| `drive`, `fight`, `pull` | no schema value fits, so the predicate has no ingredients at all |

**54 are reachable.** Stage 0 counted 17 predicates with a zero-support or
missing ingredient. 10 of those are no longer excluded, because the ingredient
they lacked is MEASURED: `recede` for the seven `*away` predicates, and
`overlapping` for `sit_inside`, `lie_inside` and `stand_inside`. No base
predicate carries either value, however. Those 10 predicates depend on a
hand-written box measurement that has no base example to check it against.

## 3. Identical-phi collisions

**Correction to the brief: stage 0 had three all-novel groups, not four.** The
full breakdown of its 10 groups was 3 all-novel, 3 base-novel and 4 all-base.

| stage 0 group | kind | stage 0b |
|---|---|---|
| `lie/sit/stand/stop_beneath` | all-novel | separated |
| `lie/sit/stand_inside` | all-novel | separated |
| `lie/stand/stop_with` | all-novel | separated |
| `lie_above`, `sit_above`, `stand_above`, `stop_above` | base-novel | separated |
| `lie_behind`, `sit_behind`, `stand_behind`, `stop_behind` | base-novel | separated |
| `lie_next_to`, `sit_next_to`, `stand_next_to`, `stop_next_to` | base-novel | separated |
| `*_front`, `*_left`, `*_right` quartets | all-base | separated |
| `chase`, `fall_off`, `follow` | all-base | **still collides** |

**Stage 0b: (a) 0 all-novel groups, (b) 0 base-novel groups.** One all-base
group remains: `chase`, `fall_off` and `follow` all reduce to
{subject_state=moving}. No novel predicate is affected.

## 4. Reference frame: VidVRD front/behind and left/right are camera-frame

Tested on base training instances only: 6,391 front/behind and 8,254
left/right. Each feature is signed so that AUC > 0.5 means it agrees with its
hypothesis. The 95% CIs come from a bootstrap that resamples whole videos.

| front/behind feature | hypothesis | AUC | 95% CI |
|---|---|---:|---|
| (ii) bottom-edge y difference | camera: front = lower in image | **0.894** | 0.864–0.922 |
| (i) log area ratio, subject/object | camera: front = larger | 0.734 | 0.687–0.778 |
| (iii) subject velocity · subject→object | heading: front = moving away from object | 0.523 | 0.494–0.552 |
| (iii-b) object velocity · object→subject *(added)* | heading: front = object approaches | 0.520 | 0.492–0.550 |

| left/right feature | hypothesis | AUC | 95% CI |
|---|---|---:|---|
| signed horizontal offset | camera: left = subject further left in image | **0.971** | 0.956–0.982 |

**Conclusion: both pairs are camera-frame.** The best front/behind predictor is
the difference in ground-contact height, which is a camera-depth cue. The
heading features are at chance. That holds even where heading should matter
most: on instances whose verb says the subject is moving, y-difference scores
0.842 and subject heading scores 0.560 (CI 0.520–0.604). The full per-stratum
tables are in `reference_frame.md`. Left/right is almost fully determined by
image-plane offset, whether the subject is stationary or moving (0.971 and
0.970).

Two things follow:
- Tagging `depth` as MEASURED is justified. It is readable from boxes, but less
  cleanly than left/right (0.89 vs 0.97).
- The heading reading behind `walk_behind = following` is not how VidVRD
  annotated. That weakens the stage-0 proposal of depth=behind for `chase` and
  `follow`.

(iii-b) was added because intrinsic front/behind is normally defined by the
reference object's heading, not the subject's. It was at chance as well.

## 5. Scoring note (no implementation)

- **phi(move_X) is a strict subset of phi(manner_X) for all 60 pairs** where
  both predicates exist, across walk, run, fly, swim, creep and jump. The only
  attribute move_X lacks each time is `locomotion`.
- `subset_pairs.csv` lists all **392** strict-subset pairs; 209 of them involve
  a novel predicate. By type:
  - 237: a bare spatial term inside a verb compound (`left` ⊂ `walk_left`)
  - 72: `chase`, `follow` or `fall_off` inside any moving predicate
  - 60: move_X inside manner_X
  - 23: **`stand_with`, `lie_with` and `stop_with` (all novel) inside every
    same-posture predicate** (`stand_with` ⊂ `stand_left`, and so on)

The last group comes from a gap in the schema. Stationary `with` has no value
to map to, so these three predicates carry posture only and no spatial
ingredient. A scorer that doesn't penalise unmatched asserted attributes will
give `stand_with` at least the score of any of the 8 `stand_*` predicates it is
a subset of, whenever one of them fires. The same goes for `lie_with` and
`stop_with`, and for move_X against each manner_X. Stage 0 proposed
proximity=adjacent for the three `*_with` predicates; adopting it would resolve
their part of this.
