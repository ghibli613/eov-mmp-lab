# RPT proposal: read-only code check

**For:** the plan's author. **Repo:** `ov-vidvrd-lab`. The code is clean at `fdeeeb4`; only
`pilot_analysis/` differs.
- **Rules kept:** no edits, no GPU, no commits. Read-only.
- **Checkpoints:** their stored configs were read with a restricted unpickler. No tensors were
  loaded and no code from the files was run.
- **Paths:** all are in `ov-vidvrd-lab` unless they say otherwise.

## 1. Entry point for PredCls on given tracks

**The released repo has no PredCls path.** `--train_traj`, `--val_traj` and `--test_traj` are
defined (`utils/parser_func.py:84-92`), but nothing reads them. The only test path is SGDet
(`models/end2end_model.py:195-345`).

**The function that runs the relation head on given tracks** is
`gen_feats_test(video_name, trajectories, split, patch_proj, global_proj, w, h)`
(`models/gen_labels.py:642`), then `modelC(feats, seq_lens)` (`end2end_model.py:326-337`).
Session 3's PredCls fed GT tracks into exactly this
(`pilot_analysis/stage1c/cache_wrapper.py:184-191`, tracks from `gt_tracks.py`).

**What each track needs:**
- `trajectory`: `{frame: [x1, y1, x2, y2]}`, **dense** over [`begin_fid`, `end_fid`). It is
  indexed frame by frame (`gen_labels.py:672`), so a gap raises a KeyError.
- `begin_fid`, `end_fid`: 0-based rows of the frame features.
- `category`: one of the **35 names in `object2id.json`**. It is looked up at `:691`; any other
  name raises a KeyError.
- `score`: a float. It becomes `sbj_scr` / `obj_scr` (`:676-677`) and enters **only the final
  score**.

**What each video needs:**
- `patch_proj` and `global_proj` for every frame;
- `anno/<split>/<video>.json`, read only for `video_id`, `width` and `height` (`:646-648`,
  `:26-30`);
- the video's frame folder, for its frame count.

**No class information enters the features.** The extractor list is fixed to rel, mot, clip and
bbox (`:647`). The class ids feed only `_extract_lan_feat`, which is off.

**Could outside tracks with names enter the same way? Yes,** if all of these hold:
- the names are mapped to VidVRD's 35, or `object2id.json` is extended, which is a data edit;
- the boxes are made dense;
- each pair overlaps by **at least 10 frames**, and the tail rule applies (`:659`, `:662-666`);
- each pair spans **at most 40 clips**, about 1,200 frames. The temporal decoder has 40 positions
  (`models/relation_classifier.py:36`). Session 3 split longer pairs into windows of at most 40
  (`cache_wrapper.py:46`, `:171`). This matters for VidOR.
- a non-VidVRD video gets a small stub annotation file (`video_id`, `width`, `height`).

## 2. Where the subject and object embeddings come from

**The same on both paths:** RoI pooling on frozen CLIP features. **It is not the detector's
`pred_embed`.**
- **The clip feature** (`gen_labels.py:72-118`). Per 30-frame clip, at its **middle frame** (`:82`),
  it is a masked mean over the 24 × 24 TagCLIP *projected* patch grid (`patch_proj`, 768-d) for
  the subject box, the object box and their union. A fourth token is the global CLIP frame
  embedding (`:117`). It is not L2-normalised.
- **Box geometry** comes from the boxes alone (`bbox_feat`, `rel_feat`, `mot_feat`, `:46-70`,
  `:120-143`).
- **The subject and object embeddings** are slots 0 and 1 of the spatial decoder output
  (`relation_classifier.py:99`).
- **`pred_embed`** (`models/methods/detectors/ov_prompt/model.py:260`, `:347`) feeds only the
  detector's training loss (`set_criterion.py:329`). The detector reaches modelC only through
  **boxes**, **track scores** and **names**.
- **The closest faithful substitute for outside boxes is the same function.** Outside boxes go
  through `gen_feats_test` unchanged.

## 3. The frozen boundary, and what can be cached

- **The encoders sit in the dataset, outside `End2End_Model`.** OpenAI CLIP ViT-L/14@336px is
  loaded twice: plain, and as TagCLIP, which is the same weights with a training-free attention
  mask (`data_loading/dataset.py:26-38`, `:80-81`). Both run under `no_grad`
  (`dataset.py:104-114`), and their weights are in no checkpoint.
- **The visual forward:** conv1, then the class and position embeddings, `ln_pre`, the
  transformer, `ln_post`, `proj` (`vlm/backbones/clip_tagclip/clip_model.py:285-310`). **No
  learned visual prompt is injected.**
- **Per frame there are three outputs:**
  - `p`: post-`ln_post` patch tokens, 576 × 1024. This is the detector's input; its test forward
    takes these tokens directly (`model.py:426-431`).
  - `pp`: projected tokens, 576 × 768. These feed the RoI pooling for modelB, deep_sort's
    appearance, and modelC.
  - `gp`: the global CLIP embedding.
- **The learned prompts are text-side.** `CustomCLIP` has a context and a `meta_net`
  conditioned on the visual predicate embedding (`relation_classifier.py:218`, `:268`, `:289`,
  `:455-458`). They are frozen in step 4 with `text_encoder`, `backbone` and `pre_classifier`
  (`cli/common.py:137-142`).
- **Cacheable per video:**
  - `gp`, `p` and `pp` for every frame. None of them depends on any trained weight.
  - For fixed tracks, also the `gen_feats_test` items.

  At fp16, `pp` + `gp` is about 0.9 MB per frame, about 46 GB for the 52,104 test frames.
  Adding `p`, only if the detector must run, is +1.2 MB per frame.

## 4. Rules that assume the released detector, frame range or track format

| # | rule | where |
|---|---|---|
| 1 | frame range = the GT trajectories' extent | `dataset.py:120-135`; loop at `end2end_model.py:202` |
| 2 | keep a query if its max class prob is **> 0.35**, else the single best query | `post_process.py:132`, `:195`, `:221` |
| 3 | dedupe at IoU **> 0.8**; clamp to the image | `gen_labels.py:145-150`; `end2end_model.py:225-232` |
| 4 | 35-d score vector = the detector's prob at its label only, then **per-class fusion** with modelB (0.6/0.4 for 6 novel classes, 0.3/0.7 for the rest, class 31 unfused) | `end2end_model.py:214-221`, `:256-264` |
| 5 | deep_sort: every confidence **= 1**, so `min_confidence` 0.2 and NMS 1.0 do nothing. The appearance is the normalised RoI mean of `pp`; cosine 0.35; budget 100; `max_age` 30; `n_init` 3 | `deep_sort_app.py:54-55`; `end2end_model.py:240-255`; `parser_func.py:355-372`; `tracker.py:38` |
| 6 | AFLink: thrT (0, 30), thrS 75, thrP 0.05 | `end2end_model.py:289-291` |
| 7 | a track frame takes a detection's 35 scores if their IoU is **> 0.5** | `end2end_model.py:303-312` |
| 8 | track format: coverage **≥ 65%** or dropped; gap trim at 15% of the span; **≥ 10 frames**; class = argmax and score = max of the mean fused vector, NaN if no frame was scored | `gen_labels.py:381-382`, `:386-415`, `:626`, `:619-620` |
| 9 | the **15% rule**, `len > 0.15 × (end_fid − begin_fid)` of the GT range | `end2end_model.py:314-319` |
| 10 | pairs: overlap ≥ 10 frames; 30-frame clips; tail rule; ≤ 40 clips | `gen_labels.py:13`, `:659-666`; `relation_classifier.py:36` |
| 11 | top 20 predicates per clip (`clip_top_n`); prior off | `post_process.py:19`; `parser_func.py:75` |
| 12 | greedy merging of consecutive clips with the same predicate; then each video's **top 200** | `post_process.py:54-99`, `:108-110` |

**The final score** (`inference/post_process.py:97-98`):

  score = sbj_scr × obj_scr × mean over merged clips k of [ σ(cos(v_k, t_p) / 0.01) × σ(int_k) ]

- The predicate score is defined at `relation_classifier.py:442`, `:505-524`.
- In the `novel` split, the base predicate columns are zeroed.
- **Track scores enter only as `sbj_scr` and `obj_scr`.** Each is the track's `score` (rule 8),
  carried in `pair_data` (`gen_labels.py:676-677`).

**For outside tracks:** rules 1–9 come before `gen_feats_test`, so only rules 10–12 and the
formula apply. The evaluator ranks within each video only: AP is per video, and R@K counts each
video's top K. So the scale of the outside track scores matters only relative to other tracks in
the same video.

## 5. What stages 3 and 4 were trained on

- **Stage 3** (the relation head, `output/ckpt/…_stage2_new_L14_e2e.pth`).
  - Its stored `config` says `train_traj='gt'`, `val_traj='gt'`, `src_split='base'`; epoch 19
    of 30.
  - **This is weak evidence.** `'gt'` is the parser's default (`parser_func.py:84-92`), and no
    stage 1–3 training code exists (RB1 L8).
  - So "GT trajectories" is likely, but not shown.
- **Stage 4** (`…_end2end_base-001.pth`).
  - Its stored `config` shows `ps='end2end_base'`, `lr=1e-05`, epoch 6 of 20; the same defaults
    are also stored, `train_traj='gt'` among them.
  - **But the released step-4 code trains modelC on detected tracks.** Detections ≥ 0.35 from the
    top-300 output (`end2end_model.py:85-92`) go through deep_sort and AFLink, then
    `format_trajectories`, then the 15% and > 14-frame filter (`:170-176`), then
    `gen_feats(..., 'train')` (`:177`).
  - **Clip labels** come from a vIoU ≥ 0.5 match to GT trajectories (`gen_labels.py:182-207`).
    Predicates come from `relation_instances` (`:540-560` region).
  - **The detector** trains on GT boxes (`end2end_model.py:68-81`).
  - **Data files:** `data/vidvrd/data/train_object_trajectories_gt.json` (frame range and
    detector targets; `dataset.py:64`) and `anno/train/<video>.json` (relations;
    `dataset.py:138-141`).
  - That this code produced the released checkpoint is consistent with its config (`ps`, `lr`,
    batch 1), but cannot be proven.
