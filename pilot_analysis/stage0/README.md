# Stage 0 — predicate decomposition and ingredient coverage

Data analysis only. No GPU, no inference, no checkpoints. Nothing outside this
directory was modified. Regenerate everything with:

```
PYTHONDONTWRITEBYTECODE=1 python3 pilot_analysis/stage0/stage0.py
```

| file | contents |
|---|---|
| `predicate_inventory.csv` | Task 1: one row per predicate |
| `component_frequency.csv` | Task 2: lexical components |
| `phi_draft.json` | Task 3: ingredient sets; `_evidence` = box / pixel / both / unassigned |
| `coverage_report.md` | Task 4 (a)–(e) |
| `unresolved.md` | every judgement call, plus the identical-phi collisions |
| `summary.json` | the numbers below, machine-readable |

## Where the split came from

`data/vidvrd/data/openvoc_pred_class_spilt_info.json`, the file the code loads
(`utils/paths.py:56`, `PRED_SPLIT_INFO`). The copy at
`configs/VidVRD_pred_class_spilt_info_v2.json` has an identical base/novel
split. Its `cls2id` ordering differs, which does not matter here. Instance
counts come from `data/vidvrd/anno/{train,test}/*.json`, and every per-predicate
count matches `cls2count` in the configs copy.

## Totals

- **132 predicates: 71 base, 61 novel.** Both counts match the expected values.
- 800 training videos, 200 test videos.
- **Training instances:** 22,616 base (usable). There are also **3,301 novel
  instances that are *not available for training***: they appear in the
  training annotations but the loss masks them. None of the support counts
  below include them.
- **Test instances:** 4,230 base + 605 novel = 4,835.
- 41 lexical components. `next_to` is kept as one component; `fall_off` is
  split into `fall` + `off`.

## 4(d) — the answer

Out of the 61 novel predicates:

| | count |
|---|---:|
| every ingredient has > 50 base instances | **39** |
| every ingredient has > 200 base instances | **29** |
| at least one ingredient has **zero** base support | **14** |
| no ingredient assigned (no schema value fits: `drive`, `fight`, `pull`) | **3** |
| weakest ingredient has 1–50 instances (the `toward` family) | 5 |

The 14 zero-support predicates plus the 3 unassigned ones make 17. That is the
same set of 17 open-world predicates that the earlier primitive check
(`../CHECKS.md`) found.

## 4(c) — thin ingredients (< 50 base training instances)

| ingredient | base support | novel predicates that depend on it |
|---|---:|---|
| relative_motion=recede | 0 | `away`, `creep_away`, `fly_away`, `jump_away`, `move_away`, `run_away`, `walk_away` |
| proximity=overlapping | 0 | `lie_inside`, `sit_inside`, `stand_inside` |
| contact_action=hold | 0 | `hold` |
| contact_action=bite | 0 | `bite` |
| contact_action=feed | 0 | `feed` |
| contact_action=kick | 0 | `kick` |
| relative_motion=approach | 12 | `creep_toward`, `move_toward`, `run_toward`, `toward`, `walk_toward` |

Two more are close to the cutoff: relative_motion=pass (54) and
comparative=faster (89).

Eight schema values are never assigned by the draft, because it leaves an axis
unset when the predicate says nothing about it. They are listed separately in
`coverage_report.md` and are not counted as thin.

## Flags for review (details in `unresolved.md`)

1. **The schema cannot tell stand, sit, lie and stop apart.** All four map to
   subject_state=stationary + locomotion=none. As a result, **37 predicates
   fall into 10 groups that share an identical phi**, and **6 novel predicates
   have exactly the same phi as a base predicate**. For example, `sit_behind`
   is identical to `stand_behind`, `lie_behind` and `stop_behind`. A scorer
   that sees only phi cannot separate them. Posture is also visible only in
   pixels, so these predicates really need both evidence groups, even though
   they are marked box-only here.
2. **The draft cannot represent 5 predicates**: `watch`, `play` (base) and
   `fight`, `pull`, `drive` (novel). `chase`, `follow` and `fall_off` share one
   phi (subject_state=moving only).
3. **The reference frame for left/right/front/behind is not confirmed.** If
   VidVRD's front/behind is measured from the object's heading rather than from
   the camera, the `depth` axis cannot be read from boxes.
4. **Evidence groups.** Of all 132 predicates: 61 are box-only, 6 pixel-only,
   **60 need both** (every walk/run/fly/swim/creep/jump compound), and 5 are
   unassigned. Of the 61 novel predicates: 26 box, 4 pixel, 28 both,
   3 unassigned.
5. **4(e), descriptive.** Out of the 58 novel predicates that have at least one
   ingredient, 52 have no exact phi match among base predicates. For 41 of
   them, no single base predicate contains all of their ingredients.
