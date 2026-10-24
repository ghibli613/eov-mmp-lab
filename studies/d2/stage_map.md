# D2 Step 0 — EOV-MMP's front end, stage by stage (test path)

Read from the EOV-MMP working repo (`~/Documents/thesis/source/ov-vidvrd-lab`) at
`fdeeeb4d20145abde48d2f64f9e39f85dcd1fca6`. That is the commit session 3 ran from to
produce the baseline SGDet tracks, and its tracked code has not changed since.
METOR's copy (`5ff9fcb`) is the same on the test path, apart from the fusion call
(`fuse_scores`, bit-identical with RB1's flags off, A23a). Its line numbers are given
in brackets where they differ.

| # | stage | what the code does | where (EOV) [METOR] |
|---|---|---|---|
| 0 | frames | Every frame in [begin_fid, end_fid), stride 1. **begin/end are the min begin / max end of the video's GT object trajectories** (`test_object_trajectories_gt.json`), not the video's length. | `data_loading/dataset.py:120-126`; `models/end2end_model.py:197-202` [205-210] |
| 1 | detector | Deformable-DETR decoder, 300 queries (`--det_token_num`), all 35 class text features at test. Class probability = sigmoid of the logit (A22 §1a). No NMS inside the detector. | `utils/parser_func.py:226`; `post_process.py:158`, `:191` |
| 2 | **score cut** | `OVPostProcess`: keep a query iff its **max over the 35 classes of the sigmoid probability is > 0.35** (strict). If none passes, keep the frame's single best query (the fallback). Then keep the top min(300, n_kept × 35) (query, class) entries, by probability. | `models/methods/detectors/ov_prompt/post_process.py:132`, `:194-195`, `:221-233`, `:264-282`; called at `end2end_model.py:208` [216] |
| 3 | dedupe | `filter` → `unique_boxes`, in the cut's output order (descending probability): a box is dropped if **any later box** has IoU > 0.8 with it, so the later (lower-scored) one survives, and the several entries of one query collapse to one box. Each kept box gets the probability of every entry with an identical box (`torch.equal`). | `models/gen_labels.py:145-162` [146-163]; `end2end_model.py:211-221` [219-229] |
| 3b | clip, fuse | The box is clipped to the image (x1, y1 ≥ 0; x2 ≤ w; y2 ≤ h). The 35-vector is fused with modelB: 0.6/0.4 for classes 1, 3, 8, 12, 14, 23, then 0.3/0.7 for the rest, so red_panda (23) is fused twice and turtle (31) not at all. **This does not affect tracking.** | `end2end_model.py:224-265` [232-273] |
| 4 | **deep_sort** | Every detection gets **confidence 1** (`deep_sort_app.py:55`), so `min_confidence` 0.2 (`:171`), `min_detection_height` 0 and **NMS at overlap 1.0 remove nothing** (overlap ≤ 1 by construction, `preprocessing.py:67`, `:71`). Appearance: the RoI mean of the TagCLIP `patch_proj` grid (`end2end_model.py:235-255`). Vanilla matching (`woC`, `linear_assignment.py:129-137`), cost 0.98 · cosine + 0.02 · Mahalanobis (`MC`, `:202-203`), χ² 95% gate (`:196`), max cosine distance 0.35, budget 100, EMA 0.9. NSA Kalman: measurement noise × (1 − confidence) = 0 (`kalman_filter.py:148-149`), so **an updated track's box is its detection's box**. ECC camera compensation from `VidVRD_ECC_test.json`. IoU stage 0.7, `max_age` 30, `n_init` 3 (`tracker.py:38`). **Output: confirmed tracks with time_since_update ≤ 1**, the Kalman mean as tlwh (`deep_sort_app.py:196-201`), so a track's first 2 frames (tentative) are not output. | `models/tracking/deep_sort_app.py:125-209`; `utils/parser_func.py:302-373`; `models/tracking/deep_sort/tracker.py:38-132`, `track.py:154-185` |
| 5 | **AFLink** | `thrT` (0, 30), `thrS` 75, `thrP` 0.05, `AFLink_epoch20.pth`. It links tracklets end to start by Hungarian matching on the model's cost, relabels ids, then keeps one row per (frame, id). | `end2end_model.py:283-293` [291-301]; `models/tracking/aflink/AppFreeLink.py:73-109` |
| 6 | score assignment | Each tracker row takes the 35-vector of the same-frame detection with the highest IoU **> 0.5**, else zeros. | `end2end_model.py:295-313` [303-321] |
| 7 | **track filter**, in order | (a) 2 frames are added before the first row, copying its box. This makes up for `n_init` 3. (b) **Coverage:** the track is dropped if its rows, the 2 included, are < 65% of its span. (c) **Gap trim:** repeatedly cut at the largest gap > 15% of the current span, keeping the side with more rows. (d) Interpolate over the trimmed range from *all* rows, round, clamp at 0. (e) **Length:** kept iff `len(track) ≥ 12`, where the dict also holds `category` and `score`, so **≥ 10 frames**. (f) **Range:** kept iff frames > 15% of (end_fid − begin_fid), the GT-extent range of stage 0. | `gen_labels.py:363-371` [371-379], `:381-382` [389-390], `:385-413` [393-421], `:416-428` [424-436], `:625-637` [652-664]; `end2end_model.py:314-323` [322-331] |
| 8 | naming | The mean of the 35-vectors over the rows with any score > 0; its argmax is the class, and its max the score. A track with no scored row gets class id 0 (airplane) and a NaN score. | `gen_labels.py:603-622` [630-649] |

**Not as described in the batch brief or in A22 §1a:**
1. The length rule is **≥ 10 frames**, not ≥ 12 (`len(track)` counts 2 non-frame keys).
2. The brief's five stages leave out two steps: the 0.8-IoU **dedupe** between the cut and the tracker, and the **gap trim** inside the filter. D2 labels each separately.
3. The "video's frame range" of the 15% rule is the **GT object extent range** (stage 0), as A22 noted for the processed frames.
4. deep_sort's own confidence, height and NMS filters are no-ops, so the 0.35 cut and the dedupe are the only detection filters before tracking.

**Saved intermediate outputs.** None are usable. Session 3's SGDet cache
(`session3_box_copy/cache_s3_sgdet/test/*.pt`, written by
`pilot_analysis/stage1c/sgdet_cache.py`) stores `model.traj` (the final, filtered,
named tracks), pair logits and features. No raw detections, cut output, tracker
output or filter decisions were saved anywhere on the laptop. A22's O1 dump was
never run (A23 cancelled it). So D2 needs a GPU recording pass (Step 2).
