# RB1 — CPU check: does the "unscored track → antelope" fallback also happen at inference?

Date 2026-10-08. Read-only. No repo code changed since the pinned commit
`7e87c9bb8c45cd4699469546c3c4310db20b87b2`. No test data touched.

## Answer

**No, not as antelope.** The two fallbacks differ:

| | training, `format_trajectories` | inference, `format_trajectories_test` |
|---|---|---|
| where (at `3be8147`) | `models/gen_labels.py:462-465`, called at `end2end_model.py:167` | `models/gen_labels.py:619-620`, called at `end2end_model.py:315` |
| condition | no track line has score > 0 | no track line carries a positive score vector (`max(score) > 0`, line 603) |
| class given | `cat = 1` → **antelope, a novel class** | `np.argmax(zeros(35))` = 0 → **airplane, a base class** |
| score | 0 | **NaN**: `max(sssss)/f` with `f = 0` is `0.0/0`; numpy returns NaN with a RuntimeWarning, it does not raise |
| what happens to the label | passed to the relation loss as `sbj_label`/`obj_label`, where `convert_oid_on_base` maps novel ids to −1 and the loss ignores them (`relation_classifier.py:533-534`). Ignored, not used. | **used**: it is the track's name in the output triplet (`gen_labels.py:676-678` → `post_process.py:115`) |

So at inference no track is labelled antelope by default. What does happen when
the condition is met is a **NaN score on an airplane-named track**. That NaN
flows into `curr_clip['score'] = sbj_scr * obj_scr * pre_scr`
(`inference/post_process.py:98`), and the per-video top-200 is
`sorted(..., key=score)` (`post_process.py:108-110`). With NaN keys Python's sort
order is undefined, so a single such track can disturb which relations survive.
It is a release quirk common to R0, R0u, R1, R2 and R3; it is not a novel-label
leak. It is added to the leak list as **L10** (`rb1/leak_list.md`), and the
training-side fallback as L9.

## Demonstration (CPU, the repo's real functions, synthetic track lines)

`python rb1/scripts/fallback_probe.py` → `rb1/logs/fallback_probe_synthetic.txt`.
One scored and one unscored track, per mode:

- training: scored track → `bird`, score 0.9; unscored → **`antelope`, score 0**
- inference: scored track → `bird`, score 0.8; unscored → **`airplane`, score NaN**
- split lookup of the two fallback classes: antelope = novel, airplane = base.

## How often does it happen? Not countable on this machine

Whether a real track has no scored line depends on deep_sort + AFLink output
boxes matching a same-frame detection at IoU > 0.5 (`end2end_model.py:297-306`),
and that needs the detector's outputs on real frames. The detector checkpoint is
3.7 GB and the ViT-L forward exhausts this machine's 7 GB RAM (it killed the WSL
session before). It cannot be counted on CPU, and I did not guess a number.

Why it is probably rare (reasoning, **not a measurement**): deep_sort starts a
track from a detection, and a track's lines come from detections or from
Kalman-updated boxes close to them; a track with **zero** matched lines among its
≥ 12 lines would be unusual.

**Ready for the GPU machine, no repo code change:**
`rb1/scripts/fallback_count_gpu.py --n_videos 2` runs the released end-to-end
checkpoint in eval mode over 2 evenly spaced **training** videos (not test
videos, no GT used by the eval branch, so not a test evaluation) through
`rb1/scripts/fallback_probe.py`, which wraps the two formatting functions only to
count: tracks seen, tracks with no scored line, and how many survive formatting,
per video, with the class they received. Output:
`rb1/logs/fallback_counts_eval.json`. It will run as part of Step 2, after the
machine setup (D.4 tests first). The runner is syntax-checked only; it has not
been run, because it needs the model.

## For the final report

Add L9 and L10 to the leak list (done in `rb1/leak_list.md`), with the measured
counts once Step 2 has produced them.
