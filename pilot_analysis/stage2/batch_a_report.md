# Stage 2, Batch A — recipe head on training videos: report

**Verdict: G2 FAIL.** The zero-shot recipe-only mAP is **28.72**, against random
**10.25**: ratio 2.80 (needed ≥ 3.0), margin +18.47 (needed ≥ 5). As A6 requires,
the batch stops here. Step 4 was not run, and nothing touched the test set.

Scope:
- CPU only, $0.
- Training videos only: 0 novel labels used. 3,301 novel instances were dropped
  at parse: 3,202 in the 783 videos used, 99 in the 17 excluded ones.
- No test annotation, test cache file or test evaluation was opened. The repo is
  unchanged; all code is in `pilot_analysis/stage2/`.
- Every frozen artefact was verified before each step (16 SHA256 checks).

## 1. Evaluator check (step 0a)

**Yes. AP is computed per video, over one ranked list holding every competing
predicate, and then averaged over videos.** Cross-predicate calibration
therefore matters directly. A predicate whose scores run high pushes the others
down every video's list, and both the top-20 and the top-200 cuts act across
predicates.

| what | where |
|---|---|
| loop over videos; per video, one `eval_detection_scores` call and `video_ap[vid] = voc_ap(...)` | `inference/video_relation_detection_openvoc.py:184-192` |
| the video's predictions, all predicates together, sorted by score into one list | `…openvoc.py:119` |
| a prediction matches only a GT relation with the same triplet (vIoU ≥ 0.5 on both trajectories) | `…openvoc.py:126-135` |
| mAP = mean of per-video AP (videos without GT skipped, :185) | `…openvoc.py:207` |
| R@50/100: per-video top-K cut, then recall pooled over all GT | `…openvoc.py:194-197, 209-217` |
| GT and predictions filtered to the target split's predicates | `…openvoc.py:362-365, 369-384` |
| top 20 per slot, across all 132 columns | `inference/post_process.py:19` |
| relation score = subject score × object score × predicate score | `inference/post_process.py:98` |
| top 200 per video, across predicates | `inference/post_process.py:108-110` |
| novel split: novel scores scattered into a zeroed 132-vector, so every base predicate scores 0 | `models/relation_classifier.py:511-514` |

**Who competes in the novel split:** the 61 novel predicates, all in one list per
video: above, away, beneath, bite, creep_above, creep_away, creep_next_to,
creep_past, creep_toward, drive, feed, fight, fly_away, fly_behind, fly_front,
fly_left, fly_right, hold, jump_above, jump_away, jump_behind, jump_past,
jump_with, kick, lie_above, lie_beneath, lie_inside, lie_with, move_above,
move_away, move_next_to, move_past, move_toward, past, pull, run_above,
run_away, run_beneath, run_next_to, run_toward, sit_behind, sit_beneath,
sit_inside, sit_next_to, stand_above, stand_beneath, stand_inside, stand_with,
stop_above, stop_beneath, stop_next_to, stop_with, swim_behind, swim_beneath,
swim_next_to, toward, walk_above, walk_away, walk_beneath, walk_past,
walk_toward.

## 2. Amendment A6

A6 was appended to `pilot_analysis/stage1b/preregistration.md` on 2026-10-05,
before any part score was computed or any network trained.

| | SHA256 |
|---|---|
| preregistration.md with A6 (now in `preregistration.sha256`, line 1) | `0e3ceb5c7101332f41d829d0396a53d3b5d617f140253577d3d5143f49f2eda1` |
| A6 text alone (13,074 bytes) | `1055cab4452c13c12d8a015b15ed5c18bf3ce3c5eef8b5656d9511dc08f33981` |
| the first 10,893 bytes (through A5), unchanged | `43676566c4689cf705f23e3b39a1d3fc424be16183c888432b92abf6cb18a3f7` |

Choices A6 fixed that the brief left open or that differ from it:
- **contact cannot be trained, so it was dropped up front.** No base label
  asserts its absence, and the brief masks unset parts, so it has no negative
  class. contact_action bite, feed, hold and kick have 0 base instances.
- **Adjacency negatives** are slots with a horizontal, vertical or depth label
  and no proximity label. This is stage 1's negative set, the evidence A3 rests
  on.
- **The learned part's input adds the A2/A3 scalars** (D1 stats at W0, D2 stats
  at W3, the gap at W3), because A2 and A3 commit to them. The brief listed only
  the embeddings.
- The 17 videos without ECC are excluded, as in stage 1: 626 FIT, 157 HELD, and
  63 FIT videos for early stopping.
- G1's 30 "instances" are counted as HELD positive slots, the unit the AUC is
  computed on.
- The mix weight w applies to the recipe score: final = EOV^(1−w) · recipe^w.
- The cache is read from `../session3_box_copy/cache_s3` (outside the repo).
  Every file used was checked against `SHA256SUMS_session3.txt`.

## 3. Step 1 — measured and supplied parts

- 783 videos, 8,780 ordered pairs, 13,318 slots. 12,154 slots carry a base label:
  9,044 fit-train, 805 early-stop, 2,305 HELD. The grid re-derived from the
  option-B tracks matches the cache pair for pair.
- **Symmetry is exact on every check.** Swap (left↔right, above↔beneath,
  front↔behind, larger↔smaller, taller↔shorter, faster flips): 53,272 / 53,272
  each. Time reversal (every per-frame fraction, subject_state, object_state and
  faster unchanged; D2 approach↔recede, co_move/pass/none fixed): 53,272 /
  53,272 each, and 13,318 / 13,318 for D2.
- The scores reproduce the frozen rules' HELD agreement on this grid:

  | rule | agreement on HELD |
  |---|---|
  | horizontal | left 96.2%, right 96.0% |
  | vertical | 98.3%, 98.6% |
  | depth | 91.0%, 90.8% |
  | subject_state | moving 75.5%, stationary 72.3% |
  | larger / taller | 95.0% / 93.1% |
  | co_move | 64.5% (A5's option-B figure exactly) |

  No measured part was NaN in any slot.

## 4. G1 — learned-part bar (full version, HELD)

FIT+ and HELD+/− count slots. The AUC is that of the three-seed mean, with a
95% CI from a video-clustered bootstrap. **No gated output fell below 0.75, so G1
dropped nothing.**

| part value | FIT+ | HELD+ | HELD− | AUC (95% CI) | G1 |
|---|---:|---:|---:|---|---|
| locomotion=creep | 129 | 14 | 2,220 | 0.999 (0.998–1.000) | untested (< 30): kept |
| locomotion=fly | 632 | 80 | 2,154 | 0.999 (0.980–1.000) | PASS |
| locomotion=jump | 150 | 38 | 2,196 | 0.992 (0.985–0.997) | PASS |
| locomotion=none | 6,380 | 1,612 | 622 | 0.952 (0.912–0.976) | PASS |
| locomotion=run | 316 | 88 | 2,146 | 0.987 (0.969–0.996) | PASS |
| locomotion=swim | 109 | 24 | 2,210 | 0.982 (0.940–1.000) | untested (< 30): kept |
| locomotion=walk | 1,996 | 396 | 1,838 | 0.968 (0.952–0.980) | PASS |
| posture=lie | 484 | 79 | 2,155 | 0.996 (0.990–0.999) | PASS |
| posture=sit | 986 | 155 | 2,079 | 0.993 (0.985–0.997) | PASS |
| posture=stand | 2,245 | 754 | 1,480 | 0.979 (0.963–0.989) | PASS |
| contact_action=ride | 328 | 67 | 48 | 1.000 (1.000–1.000) | PASS |
| contact_action=touch | 126 | 48 | 67 | 1.000 (1.000–1.000) | PASS |
| relative_motion=pass | 43 | 10 | 172 | 0.809 (0.506–1.000) | untested (< 30): kept |
| proximity=adjacent | 1,028 | 257 | 1,917 | 0.949 (0.926–0.970) | PASS |
| contact=contact | — | — | — | — | not trainable (no negative class): dropped by A6 |
| contact_action=bite / feed / hold / kick | 0 | — | — | — | not trainable (0 base instances): dropped by A6 |

**⚠ These HELD numbers are optimistic.** The inputs are EOV-MMP's features on
training videos, from a model trained on these same videos' labels, including
`ride`, `touch` and `next_to` themselves. The AUCs mostly show that the features
already encode the labels. Every full-version seed early-stopped at epoch 2,
which fits that.

## 5. G2 — zero-shot gate: **FAIL**

Setup:
- The leave-out learned part was retrained without the 10 predicates' labels:
  9,482 labels removed, 5,821 training slots left.
- Evaluation is on HELD (157 videos, 2,494 slots, 2,064 GT relations of the 10),
  PredCls on GT tracks, with only the 10 competing.
- Scores go through the repo's post-processing and
  `eval_relation_detection_openvoc`.

The 10 predicates (fixed in A6): stand_left, stand_right, stand_front, larger,
taller, stand_behind, walk_left, walk_right, walk_behind, walk_front. Their
recipes stay distinct after G1.

| scores (only the 10 compete) | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| **recipe only, leave-out learned part (zero-shot)** | **28.72** | 29.46 | 35.71 |
| random U(0,1), mean of 20 seeds (sd 0.69 mAP) | 10.25 | 16.58 | 25.73 |
| *reference:* recipe only, full learned part (not zero-shot) | 29.73 | 29.99 | 36.29 |
| *upper reference:* EOV-MMP alone (trained on these 10) | 55.77 | 37.74 | 43.94 |

| test | needed | got | |
|---|---|---|---|
| ratio | ≥ 3 × random | 2.80 × | ✗ |
| margin | ≥ random + 5 | +18.47 | ✓ |

**Verdict: FAIL.**

## 6. Step 4 — mix on HELD: **not run**

G2 failed, so A6 and the brief stop the batch before step 4. The 0.25 / 0.5 / 0.75
table does not exist. (`logs/step4_mix_small.json` is the 5-video plumbing run,
not a result.)

## 7. Recipe collisions from dropped parts

- **G1 dropped nothing, so it created no collisions.** The only group with
  identical recipes is the pre-existing all-base one: `chase` ≡ `fall_off` ≡
  `follow` ({subject_state=moving}).
- **The A6 drops leave four novel predicates with no recipe.** contact (no
  negatives) and the zero-support contact_action values empty the recipes of
  `bite`, `feed`, `hold` and `kick`. With `fight`, `play` and `watch`, they get
  the slot's neutral score, so all seven are indistinguishable. Stage 0 had
  already listed bite, feed, hold, kick and fight as unreachable.
- Losing contact also thins `drive`, `pull` and `lie/sit/stand_inside`, but none
  of them collides.

## 8. Surprises and caveats

1. **G2 fails on the ratio, and withholding the labels barely matters.** The
   zero-shot learned part (28.72) is within 1.0 mAP of the full one (29.73). So
   the shortfall is in the recipe score, not in the learned part's
   generalisation. A diagnostic run after the verdict, on the G2 score files
   only (nothing re-run or changed), shows where:
   - Each recipe separates its own labelled HELD slots from the other labelled
     slots well: slot-level AUC 0.80–0.92 for all 10.
   - But on its own slots, a recipe tops the other nine only 14–60% of the time
     when it has four parts (stand_*, walk_*). The one-part measured recipes,
     `larger` and `taller`, top them 86–89% of the time.
   - Those two score about 0.46 on average even on slots that aren't theirs:
     whenever the subject is the bigger box, their fraction is near 1.
   - Within one ranked list per video (section 1), these short, near-binary
     recipes outrank the multi-part ones. The geometric mean removes the length
     penalty only when the parts are equally confident.

   This is the cross-predicate calibration that step 0a flagged.
2. **"Zero-shot" applies only to the learned part's labels.** Its input features
   (`visual_pre_embeddings`, subject/object embeddings) come from EOV-MMP, which
   was trained with the 10 predicates' labels on these same videos.
3. **Three learned values are untested.** creep (14 HELD positives), swim (24)
   and pass (10) are kept under the < 30 rule. pass has AUC 0.81 with a CI of
   0.51–1.00, which supports nothing.
4. **Annotation noise exists in HELD.** In `ILSVRC2015_train_00008004`, for
   example, the annotator swapped subject and object on all four spatial labels
   of a cattle pair. The measured rules are right there and the labels wrong.
5. **The leave-out set leans on two families.** The fixed rule picked 4 `stand_*`,
   4 `walk_*`, `larger` and `taller`. `stand_next_to` was skipped because it is
   the last base carrier of posture=stand (353 FIT instances remain).
6. **Rerunning step 0 would break the pin.** `step0_selection.json` is pinned
   by A6, so `step0_select.py` now refuses to overwrite it.

## 9. Hashes

Trained learned-part models (three seeds each, plus the input normalisation):

| file | SHA256 |
|---|---|
| `data/model_full.pt` | `3feb73909b3812384395bc2981ecf4dbb28440ff1d2ea394a7797e742ccd9e4b` |
| `data/model_leaveout.pt` | `293f4dfa8e87e4ccdded4172cd16c33477d3558b32fe56d2997adeb1aa15daf0` |

Score and data files:

| file | contents | SHA256 |
|---|---|---|
| `data/slots.npz` | measured and supplied part scores, A2/A3 scalars, EOV logits, per slot | `8c9e9c48eb94888eb4f2ff9cbac4075087330e329445f59eb3f693a15e5332ed` |
| `data/emb.npy` | learned-part input embeddings | `9eb54ebb37317f5a85ab848f40b49df5189fc57f3c0f8928bf04c02fd6561469` |
| `data/slots_meta.json` | videos, slot labels (base only) | `01ba90c5933b464e69a540e704f500ed0d4161cfb43623e8cc75c401ce7bc570` |
| `data/learned_full.npy` | learned-part scores, full | `435e4cf499fbd69c5107cb149464fe16144b8d3ae56190c079bce29be864fc3c` |
| `data/learned_leaveout.npy` | learned-part scores, leave-out | `30e78d6d3d663331f2e1564d008539e1dc821dc9af0c8b640f98c9ee95757984` |
| `data/recipe_full_held.npy` | recipe scores, 132 predicates × HELD slots, full | `f7ada222ffd11a38cf968dad17b55ca6e4ed44796b9c20de6da213300689f60e` |
| `data/recipe_leaveout_held.npy` | the same, leave-out | `634f3ce3a7c20127dcc6e6440362aa27cf6d9085cdaec9a83c8bdc87573a8430` |
| `data/held_gt.json` | HELD GT, base only | `78d55c69f9aa2473f4584ae998592dcb67d95ce02a67ab0cd9a8bd8446184dee` |
| `data/held_pairs.json` | HELD pairs for post-processing | `8943fb186707a7908090af759d93815894ad654af6af34061ad52e05a5b783ec` |
| `data/g1.json` | G1 | `4507e3f3b7653008501c641857bc8ee64f0d0c0c392523fe616fcb22a8c104df` |
| `data/g2.json` | G2 | `eba6841c5283edd8a95d8445ad1959df8620303fbd484d60f275a6ac2fa1e9b3` |
| `data/recipes_after_g1.json` | effective recipes, collisions | `f3a4ab84561f60f07ecb38c4d21d697a12a30c49e939cc932298bde8d1c2cc3f` |
| `data/split_leaveout.json` | split file for G2 | `bfcc20fbf1cb2134ab3882bb9cd7cee51ce2746f3a91b41873e30dabbf84eb82` |
| `data/train_full.json` | training record, full | `f61b3de24002e64134731521f2ae8e124809ee8f30f9b32810283c759df82284` |
| `data/train_leaveout.json` | training record, leave-out | `a4929a23ad0876216d3f770eeb36783ea9464a42faf446ecd1f192ae33da832b` |
| `step0_selection.json` | A6's selections (pinned) | `9aa2310b5b2971b082221e0bb727b99c3d94afd19b17c8456c905e1ed170f04c` |

`data/` (about 200 MB) is git-ignored. Every step's command, configuration, seeds,
input and output hashes, code hashes and environment are in `logs/`.

Environment: Python 3.12.13 (conda `repro-next`), torch 2.11.0, numpy 2.5.2,
sklearn 1.9.0, CPU only; repo at `fdeeeb4`, plus the uncommitted `stage2/` code.

Reproduce:
```
python pilot_analysis/stage2/step1_parts.py               # ~5 min
python pilot_analysis/stage2/step2_train.py --version full
python pilot_analysis/stage2/step2_train.py --version leaveout
python pilot_analysis/stage2/step3_g2.py                  # ~1 min
```
