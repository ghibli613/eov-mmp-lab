# Object check O1 — step 1 inventory (interim, before any GPU run)

**A22** is appended and hashed. `preregistration.md` is now
`442623b82cf5392c480e889ab160d5e0606300acc9d588d338e0a806036211f0`; the first
89,462 bytes (through A21) are unchanged. All pins verify. A22 text alone:
`bafa30109ad5682d4d5c61eaf08c1711c493c1b85cdfad85916792e18696299a`.

No test annotation was opened. 1d read D1's existing `tc_missed.json` only.

**Three findings bear on the batch before money is spent:**
1. **The repo already names each track from whole-track evidence.** It does not
   decide per frame (1a). So G-O2's (c) vs (a) compares two pooled evidence
   sources, the raw detector probabilities and the repo's fused detector +
   classifier scores. It is not per-frame vs whole-track.
2. **The shipped detector was fine-tuned end to end** (1b). The repo's
   end-to-end code feeds novel boxes, with their novel labels, as detector
   targets on these same training videos. If that is how the shipped checkpoint
   was trained, training videos are **in-sample for novel objects**. The check
   is then not conservative, and could understate both problems.
3. **Only 23% of D1's U1 misses involve a novel object** (1d). Across U1 + U2,
   novel objects touch **92 of the 484 missed SGDet novel relations (19%)**. That
   bounds what fixing novel objects could recover.

## 1a. How the front end works

| question | answer | where |
|---|---|---|
| frames | Every frame in [begin_fid, end_fid) (stride 1). The range is the min begin / max end of the video's GT trajectories, as at test time: 162,330 of the 244,100 training frames, and every annotated frame lies inside it. (The paper says "every 30 frames"; the repo uses every frame, and session 3 reproduced SGDet that way.) | `end2end_model.py:202`; `dataset.py:82`, `:102`, `:120-134` |
| resolution | Each frame is resized to 336 × 336, ignoring aspect ratio. TagCLIP ViT-L/14@336 gives a 24 × 24 patch grid, the detector's input. | `dataset.py:78`, `:111` |
| detector | Deformable-DETR decoder with 300 queries. The 35 class text features also condition the decoder (`text_query`). Class logit = `cls_layer(query) · text feature`: a plain dot product, then a sigmoid. **There is no cosine and no τ = 0.01 in the detector**; that temperature is the relation head's. | `model.py:454-476`, `:62` |
| all 35 text features at test? | **Yes.** `select_id = range(35)` at test. In training, only the labels present in the frame are used. | `model.py:454`, `:309-312` |
| the 0.35 cut-off | `OVPostProcess`: keep a query iff max sigmoid probability > 0.35. If none passes, keep the frame's single best query. Then keep the top 300 (query, class) entries. `filter` then drops duplicate boxes (IoU > 0.8). | `post_process.py:132`, `:195`, `:221`, `:266`; `end2end_model.py:208-212` |
| per-detection scores | A 35-vector: the detector probability for each of the box's classes in the top 300, fused with modelB (a CoCoOp-style CLIP classifier on the box's RoI mean). The fusion is 0.6 · modelB + 0.4 · detector for antelope, bear, cattle, fox, giant_panda and red_panda, and 0.3 / 0.7 for the rest. (red_panda appears in both lists, so it is fused twice.) | `end2end_model.py:256-264` |
| linking | deep_sort, class-agnostic. Appearance is the RoI mean of the TagCLIP grid; every detection gets confidence 1, so `min_confidence` filters nothing; NMS overlap is 1.0. Camera compensation uses ECC. Then AFLink. Then each track line is matched back to a same-frame detection at IoU > 0.5. Tracks are densified and kept only with ≥ 65% coverage, ≥ 12 frames and > 15% of the range. | `deep_sort_app.py:55`, `:171`; `end2end_model.py:268-319`; `gen_labels.py:381`, `:626` |
| **track class** | **The argmax of the sum of the fused 35-vectors over every track line with a matched detection.** The track score is that sum's max divided by the count. | `gen_labels.py:603-620` |

## 1b. How novel object categories were excluded in training

- **Paper (step 1):** the detector starts from MS-COCO Deformable-DETR "with
  novel categories removed" (docs/01_Architecture.md, Implementation Details).
  There is no step-1 training code in the repo.
- **Shipped checkpoint:** every trainable detector tensor in the end-to-end
  checkpoint differs from the standalone detector checkpoint: decoder 168/168,
  class and box heads, queries, fusion, `pred_embed`. 308 of 755 tensors differ,
  max |Δ| 0.025. Only the frozen TagCLIP backbone (446 tensors) is identical.
  **So step 4, end to end, changed the detector.**
  - End-to-end config: `src_split` = `train_split` = base, epoch 6.
  - Detector checkpoint: 60 epochs, `open_vidvrd`, `all_train_token` False.
- **Repo step-4 code:** the detector's box targets come from
  `train_object_trajectories_gt.json` for every category, via `cls2id`, with no
  split filter (`gen_labels.py:275`, called at `end2end_model.py:69`). The
  frame's labels become the detector's text queries (`model.py:309-312`).
  - That file holds **1,244 novel-category fragments of 4,834**.
  - `src_split`/`train_split` = base filter predicates (`gen_labels.py:247`) and
    the object classifier's prompts, not the detector's boxes.
- **Reading.** If the shipped checkpoint was trained with this code, the
  detector **saw novel objects with their novel labels in these training
  videos**. The repo cannot settle whether it was; the paper's statement covers
  step 1 only. Frames with novel objects were not dropped either way.

## 1c. GT object tracks in the 800 training videos

| category | split | tids | option-B runs | annotated boxes | videos |
|---|---|---:|---:|---:|---:|
| antelope | novel | 114 | 198 | 7,815 | 47 |
| bear | novel | 44 | 124 | 6,045 | 22 |
| cattle | novel | 81 | 137 | 5,970 | 21 |
| fox | novel | 50 | 81 | 4,560 | 30 |
| giant_panda | novel | 48 | 113 | 4,290 | 23 |
| horse | novel | 66 | 100 | 6,450 | 38 |
| lizard | novel | 15 | 25 | 2,340 | 12 |
| motorcycle | novel | 56 | 80 | 3,300 | 46 |
| red_panda | novel | 40 | 65 | 3,765 | 19 |
| watercraft | novel | 67 | 104 | 4,260 | 36 |
| **novel total** | | **581** | **1,027** | **48,795** | 292 |
| base total (25 categories) | | 1,849 | 3,017 | 140,550 | |

Novel is well above 100, so the check is not weak by that test. The full
per-category table, base included and with box counts, is in
`data/o1_step1.json`.

## 1d. D1 cross-reference (SGDet, novel pass, from `tc_missed.json` only)

| D1 bucket | missed GT relations | subject or object novel | share |
|---|---:|---:|---:|
| U1 (no detected pair matches the GT objects) | 262 | 61 | **23%** |
| U2 (a detected object has the wrong name) | 34 | 31 | **91%** |
| U1 + U2 | 296 | 92 | 31% |

There were 484 missed in all.
- **Naming errors are almost all novel-object errors.**
- **The large U1 group is mostly base objects.** A novel-object fix addresses at
  most 92 / 484 = 19% of the missed relations.

## Next: step 2 needs a GPU, and a push

The box clones the repo from GitHub. So the work since `fdeeeb4` has to be
committed and pushed first: A6–A22, and the `stage2` … `objects` folders.

Then `bash pilot_analysis/objects/run_o1_small.sh` runs:
1. setup;
2. 5 training videos with the repo-identity check;
3. the estimate, then a stop.

**Rough full-run estimate before measuring:** 15–25 h on one RTX 4090, about
$6–15. This is extrapolated from session 3's SGDet pass at 0.24 s per processed
frame, plus the two lower cut-offs. The small run replaces it with a measured
figure.
