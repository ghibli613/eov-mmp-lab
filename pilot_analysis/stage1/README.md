# Stage 1, session 1 — measured-axis check and model-interface audit

Cost: $0. CPU only, with no inference and no forward pass. The checkpoint was
opened on CPU, memory-mapped, for parameter names and shapes. The repo is
untouched: all output is in this directory. Task A used base-predicate labels
from the training annotations only. Novel instances are dropped as each file is
parsed, and no test annotation was opened.

```
python3 pilot_analysis/stage1/axis_check.py     # Task A, ~40 s; writes axis_check.md, thresholds.json
```

| file | contents |
|---|---|
| `measured_axes.py` | every MEASURED axis as an importable function of two GT box sequences; stage 2 imports it |
| `thresholds.json` | the three frozen thresholds, how each was fitted, and the rejected one |
| `train_split.json` | 640 FIT / 160 HELD training videos, video level, seed 20260928 |
| `axis_check.md` | Task A in full: A2–A5, 2×2 tables, symmetry tests, false-positive rates |
| `interface_audit.md` | Task B in full: B1–B6 with file:line references |

## A5 — verdict per axis (HELD videos)

| axis | HELD agreement (lowest value) | AUC (95% CI) | verdict |
|---|---|---:|---|
| horizontal | left 96.1%, right 96.0% | 0.973 (0.941–0.993) | **PASS** |
| vertical | above 100%, beneath 100% | 1.000 | **PASS** |
| depth | front 91.4%, behind 90.8% | 0.951 (0.911–0.977) | **PASS** |
| proximity (adjacent) | 72.4% | 0.737 (0.691–0.784) | **PASS, marginal** ⚠ |
| subject_state, as specified | stationary 55.8% | 0.685 (0.584–0.768) | **FAIL** |
| subject_state, corrected (`motion_rate`) | stationary 72.7%, moving 78.4% | 0.810 (0.742–0.870) | **PASS, narrow** |
| object_state, corrected | moving 88.2% (only value labelled) | — | **PASS** |
| relative_motion, corrected | co_move 62.4% | — | **UNRESOLVED** ⚠ |
| comparative: larger / taller / faster | 94.6% / 93.2% / 90.5% | 0.989 / 0.975 / 0.905 | **PASS** |

**Symmetry tests: all exact.** Time reversal holds on 22,315 of 22,315
instances, the three spatial axes under swap on 66,945 of 66,945, and the
comparatives under swap on 66,945 of 66,945. 17 videos have no ECC matrices and
are excluded.

- **The pre-registered expectation held.** subject_state, as specified, was the
  weakest axis and FAILs. By the brief's own logic, a signal with AUC ≥ 0.80
  means the rule is wrong, not the boxes. Comparing definitions **on FIT
  only**, net camera-compensated displacement in the subject's own
  body-lengths per frame (`motion_rate`) reaches AUC 0.812, against 0.725 for
  the specified mean-step ÷ frame-diagonal. Box jitter adds to every step, so
  it inflates the mean step but cancels out of the net displacement; and
  dividing by the frame penalises distant subjects. `motion_rate` is frozen for
  subject_state, object_state and co_move. `faster` keeps the specified speed,
  which passes. HELD was scored once, after the choice was made.
- **Weak spot: `stop_*` agrees only 46%.** 54% of "stopped" vehicles measure as
  moving. `swim` is at 52% and `sit` at 64%. The four novel `stop_*` predicates
  depend on stationary.
- **relative_motion can't get a pre-registered verdict.** co_move has no
  negative class, so no AUC is possible. `pass` fires on 3 of 53 base
  instances, `approach` on 5 of 12, and `recede` has no base example. 20 novel
  predicates depend on this axis. Re-tagging it LEARNED would make 12 of them
  unreachable (`*_away` and `*_toward`, because approach and recede have < 50
  base instances). Kept MEASURED, they rest on a rule that base data cannot
  validate.
- **proximity passes on the letter only.** Its AUC is WEAK-level, and only
  57.7% of the other-spatial-term negatives measure non-adjacent. `contained`
  fires on **10.3% of all base instances**, mostly occlusion (`move_beneath`,
  `stand_front`), but on only 1 of 418 `ride`.

## A2 — thresholds (fitted on FIT, frozen)

| threshold | value | fitted by |
|---|---:|---|
| **moving** (`motion_rate`) | **0.00447** box diagonals per frame | max balanced agreement on stationary (6,695) vs moving (7,001) subjects; FIT BA 0.753, AUC 0.812 |
| adjacency gap | **0.331** mean box diagonals | max balanced agreement on the `next_to` family (1,037) vs other-spatial-term instances (12,246); FIT BA 0.698 |
| relative-motion dead zone | **0.411** mean box diagonals | 90th percentile of \|Δd\| over FIT co_move (`*_with`) instances (1,064) |
| *rejected:* speed as specified | 0.00189 frame diagonals per frame | same fit; FIT BA 0.675, AUC 0.725 |

Everything else is a sign test with no threshold: horizontal, vertical, depth,
contained, the three comparatives, and co_move's direction check (cosine > 0).

## B2 — scoring granularity

**One 132-vector per (ordered pair, 30-frame slot).** A pair is any two
distinct trajectories that overlap by at least 10 frames, and both orderings
are scored. The overlap is cut into 30-frame slots. A final partial slot is
kept if it has at least 10 frames. Each slot's appearance comes from its **mid
frame only**.

The output extent is not produced by the classifier. Post-processing keeps the
top 20 predicates per slot, then chains consecutive slots with the same label
into one instance, whose score is the mean of its slots. Finally the top 200
per video are kept. **The head must emit scores on exactly the same
(pair, slot) grid.**

## B6 — recommended cache set (≈ 3 GB for all 1,000 videos)

Run one GPU pass per video, over **whole** GT trajectories in the `all` split,
through a wrapper script with forward hooks. No model code changes.

1. **Per (pair, slot):**
   - logits over all 132 + the interactiveness logit. One pass reproduces both
     splits exactly.
   - `visual_pre_embeddings` (3,072-d), the output of the temporal transformer.
   - the RoI-pooled slot inputs and the spatial-decoder tokens.
2. **Per pair:** the 3,072-d prompt-conditioning mean. The learned prompt is
   instance-conditioned, so a paraphrase's text embedding differs for each
   pair. Also the subject/object embeddings, categories, and GT boxes (these
   join to `measured_axes.py`).
3. **Per frame:** the CLIP-L frame embedding, and the TagCLIP RoI means per
   (track, frame) and per (pair, frame). This is the frame resolution `modelC`
   throws away: it sees one frame per slot.
4. **Once:** `ctx`, `meta_net` and `token_prefix` / `token_suffix` from the
   checkpoint, plus the three fixed-template text embeddings.
5. **Skip:** the raw per-frame patch grids (≈ 262 GB). If slot-level re-pooling
   is wanted, keep them at mid-frames only (≤ 28 GB).

**⚠ The caching script must handle four things:**

1. **`train_object_trajectories_gt.json` holds fragments, not trajectories**:
   4,834 entries for 2,430 tracks, mostly 30-frame pieces with gaps. Build whole
   tracks from `anno/train` instead.
2. **Pairs longer than 40 slots crash** the temporal decoder (40 position
   embeddings). 10 training pairs exceed that; the longest is 77 slots.
3. JSON frame keys are strings, but the model indexes them with `int`.
4. Handle the 2-frame pre-roll from `add_initial_frames` (pilot §B.9).
