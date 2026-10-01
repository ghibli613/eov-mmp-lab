# Stage 1 — Task B: model-interface audit

Code reading only. The checkpoint was opened on CPU, memory-mapped, and read
for parameter names and shapes only (`output/ckpt/…end2end_base-001.pth`).
There was no forward pass and no GPU. EOV-MMP stays frozen, so the question
here is which tensors to save, not where gradients stop.

Paths below are relative to the repo root. `rc` = `models/relation_classifier.py`,
`gl` = `models/gen_labels.py`, `e2e` = `models/end2end_model.py`,
`pp` = `inference/post_process.py`.

---

## B1 — the relation classifier (`modelC`), module by module

`modelC` is called once per **ordered trajectory pair**, batch size 1
(`e2e:337`). `slen` = the number of 30-frame slots in that pair (see B2).

**Inputs**, built per slot by `gen_feats_test` → `FeatExtractor` (`gl:642`, `gl:14-150`):

| stream | shape per call | what it is | where |
|---|---|---|---|
| `clip_feat` | (1, slen, 4, 768) | per slot: masked mean of the TagCLIP 24×24 patch grid (`patch_proj`) over the **subject box, object box and union box** at the slot's **mid frame**, plus the CLIP-L **whole-frame** embedding (`global_proj`) of that frame. Not L2-normalised. | `gl:72-118` (mid frame `gl:82`, frame token `gl:117`) |
| `bbox_feat` | (1, slen, 4, 24) | head box, tail box and their difference, for the same 4 regions | `gl:121-140` |
| `rel_feat` | (1, slen, 42) | relative subject–object geometry at the slot's first, mid and last frame | `gl:45-53` |
| `mot_feat` | (1, slen, 42) | the differences of those three | `gl:54-66` |

The patch grid and frame embedding come from `Dataset_new.__getitem__`, which
encodes **every frame** (stride 1) with CLIP-L (`global_proj`, `dataset.py:109`)
and TagCLIP (`patch_`, `patch_proj`, `dataset.py:111`). Per frame, `patch_proj`
is (1, 576, 768). I inferred that from its consumers, the 24×24 masks and the
768-d tokens, not by running the encoder.

**Forward path** (`Model.forward`, `rc:491-525`, test branch):

| # | module | input → output | where |
|---|---|---|---|
| 1 | `FeatEmbedding` | the 4 streams; `rel ⊕ mot` → 84-d `pos` | `rc:134-154` |
| 2 | `SpatialDecoder` (1 × `TransformerEncoderLayer`, d=768, 8 heads), run **per slot** over its 4 region tokens + `boxEmb(bbox)` + role embedding | (slen, 4, 768) → normalised (slen, 4, 768); `objEmb` on tokens 0/1 → subject/object (slen, 768); relation tokens `norm(x + posEmb(pos))` (slen, 4, 768) | `rc:89-99` |
| 3 | mean over slots | subject/object embeddings (1, 768) each | `rc:148-149` |
| 4 | `TemporalDecoder` (1 × `TransformerEncoderLayer`, d=768), run **over slots**, separately per region stream, with **40** learned positions | (1, slen, 4, 768) → (1, slen, 4, 768); `intHead` on the 3072-d concat → interactiveness logit (1, slen); `relEmb` + norm, reshaped → (1, slen, 3072) / 2 | `rc:45-61`, positions `rc:36`, `rc:153` |
| 5 | L2 normalise | `visual_pre_embeddings` (1, slen, 3072) | `rc:495-498` |
| 6 | `CustomCLIP` prompt learner, conditioned **per pair**: `bias = meta_net(mean over slots of visual_pre_embeddings)` (3072 → 48 → 768), added to 8 of the 16 learned `ctx` tokens; the prompt `[SOS] ctx(16) suffix(name, ".", EOS)` goes through the frozen CLIP text tower | 132 learned-prompt text embeddings (132, 768) | `rc:264-286`, `rc:297-312`, called `rc:500` |
| 7 | `split_text_embeddings`: concat of [fixed subject template, fixed object template, fixed "visual relation" template, learned prompt] | (n_split, 3072) / 2 | `rc:465-488` |
| 8 | logits = `visual_pre_embeddings · textᵀ / 0.01` → sigmoid → × sigmoid(interactiveness); in the `novel` split, scattered into 132 zeros | `pre_scores` (1, slen, 132) | `rc:504-524` |
| 9 | subject/object heads: embedding · object-class text → softmax | (1, 35) each. **Computed but unused**: the evaluated triplet takes its subject/object classes and scores from the tracker (`pair_data`) | `rc:506-509`; `cli/common.py:157-165` |

Checkpoint parameters: `featEmbedding` 16.6 M; `pre_classifier` 92.1 M. The
latter includes a frozen copy of the CLIP text tower. Its learnable part is
`ctx` (16, 768) and `meta_net` (48×3072, 768×48). `token_prefix` (132, 1, 768)
and `token_suffix` (132, 60, 768) are **persistent buffers stored in the
checkpoint**.

---

## B2 — scoring granularity

**The model scores one 132-vector per (ordered pair, 30-frame slot).** It has
no per-frame output.

1. **Pairs** (`gl:655-659`): every ordered pair of distinct trajectories whose
   temporal overlap is ≥ 10 frames. (A, B) and (B, A) are both scored.
2. **Slots** (`gl:661-667`): the overlap `[max(begin), min(end))` is cut into
   30-frame slots. A final partial slot is kept if it has ≥ 10 frames, and
   trimmed otherwise.
3. Each slot's appearance comes from its **mid frame only**. Its geometry comes
   from its first, mid and last boxes.
4. **Temporal extent** (`pp:4-123`): each slot's top-20 predicates get that
   slot's frame range (`pp:19`, `pp:31-38`). `association` then greedily chains
   consecutive slots of the same pair that carry the **same predicate** into
   one instance. Its extent is the union of those slots, and its score is the
   mean of their scores (`pp:54-100`). Triplet score = subject score × object
   score × predicate score (`pp:98`). Finally the top 200 per video are kept
   (`pp:107-110`).

**The head must therefore emit a 132-vector per (ordered pair, slot), on
exactly these slot boundaries.** The output extent is produced downstream by
association, not by the head.

---

## B3 — PredCls interface

**No path in this repo feeds GT trajectories to `modelC`.**

- **Training (step 4, `e2e:55-193`).** `train_object_trajectories_gt.json`
  (`dataset.py:64`) supplies only two things: the video's frame range
  (`dataset.py:120-126`) and the detector's box targets (`gen_box_label`).
  `modelC` trains on **detected and tracked** trajectories (`e2e:167`, `e2e:177`).
  Its labels come from IoU-matching them to GT (`gen_hit_tid` `gl:182`,
  `gen_label` `gl:310`). Training on GT trajectories was MMP's step 3, which has
  no code here.
- **Test (`e2e:195-343`).** Detector → deep_sort → AFLink →
  `format_trajectories_test` (`e2e:315`) → `gen_feats_test` (`e2e:326`) →
  `modelC` (`e2e:337`). The `--test_traj gt` flag is dead code (pilot §B.18).

**How to run it on GT trajectories without modifying model code** (pilot §B.9):
use a wrapper script, in the style of `pilot_analysis/scripts/dump_predictions.py`.
For each dataset item, call `gen_feats_test(video_name, gt_trajectories,
'test', patch_proj, global_proj, w, h)`, then `modelC(feats, seq_lens)`, and
save the outputs. Detection and tracking are skipped entirely. The required
schema is:

```
{'tid': int, 'category': str, 'score': 1.0,
 'trajectory': {int fid: [x1, y1, x2, y2]},   # dense over [begin_fid, end_fid)
 'begin_fid': int, 'end_fid': max(fid) + 1}
```

**Test file.** `data/vidvrd/data/test_object_trajectories_gt.json` exists
(3.4 MB, written by `tools/upstream_data_scripts/gen_trajs.py:168` from the test
annotations). I have not opened it. The pilot also built its own test GT
trajectories (`pilot_analysis/scripts/gt_trajectories.py`) from `anno/test`.
To build one:
1. Collect each `tid`'s boxes from the annotation's per-frame `trajectories`,
   with span `[first, last+1)`.
2. Take the category from `subject/objects`, with score 1.0.
3. Make it dense using the repo's own `add_initial_frames` and
   `interpolate_and_adjust_frames`, as the detected path does. Or use the
   pilot's strict mode, which skips the 2-frame pre-roll that `add_initial_frames`
   adds (pilot §B.9).
4. Convert the frame keys to `int`.

**⚠ Four constraints stage 2 has to handle:**

1. **`train_object_trajectories_gt.json` holds fragments, not trajectories.**
   It has 4,834 entries for 2,430 annotated tracks. The median length is 30
   frames, 2,404 track IDs repeat, and there are gaps. For example, in
   `ILSVRC2015_train_00005005` two 60–135 tracks are stored as 60–90 + 105–135.
   Pairing the fragments gives 10,156 pairs and 12,440 slots, against 6,326 pairs
   and 23,222 slots from whole tracks. **Build whole tracks from `anno/train`
   instead**, the same way as the test side.
2. **40-slot cap.** `TemporalDecoder` has 40 position embeddings (`rc:36`), so a
   pair longer than 1,200 frames raises an index error. On whole GT tracks, 10
   training pairs exceed it (the longest is 77 slots). The wrapper must split
   long pairs into ≤ 40-slot windows or drop them. The test side can't be
   checked without opening test annotations.
3. JSON frame keys are strings, but `gen_feats_test` indexes `trajectory[int]`.
4. The pre-roll: `add_initial_frames` moves `begin_fid` two frames earlier, so a
   track that starts at frame 0 gets `begin_fid = -2` (pilot §B.9).

---

## B4 — tensors to cache (fp16 sizes)

### (a) Head input, at two depths

**There are no per-frame features anywhere after RoI pooling.** `modelC` sees
one frame per slot, and its temporal transformer runs over slots, not frames.
The 4th region token is the **whole-frame CLIP-L embedding**, not a background
box.

| depth | tensor | produced at | granularity | fp16 |
|---|---|---|---|---:|
| 1, before the spatial transformer | `clip_feat` (4×768) + `bbox_feat` (4×24) + `rel`/`mot` (84) | `gl:72-140` | per slot (mid frame) | 6,504 B |
| 1, **per frame** (what the brief asks for) | the same masked means, computed at every frame from `patch_proj` (576×768 per frame, `dataset.py:111`) | new: the per-frame grid exists only transiently in the dataset item | per (trajectory, frame): subject/object 768; per (pair, frame): union 768; per frame: CLIP-L 768 | 1,536 B each |
| between the transformers | spatial-decoder relation tokens (4×768) | `rc:96-99` | per slot | 6,144 B |
| 2, after the temporal transformer | `visual_pre_embeddings` (3072) | `rc:495-498` | per slot | 6,144 B |
| 2 | interactiveness logit | `rc:58` | per slot | 2 B |
| 2 | subject/object embeddings (2×768) | `rc:148-149` | per pair | 3,072 B |

Capture is possible without touching model code, through forward hooks on
`featEmbedding.spatial_decoder`, `featEmbedding.temporal_decoder` and
`featEmbedding`.

### (b) Fusion: the full 132-vector

Cache the **pre-sigmoid logits over all 132** (`rc:504`) plus the
interactiveness logit, per (pair, slot): 266 B. **One `all`-split pass covers
both splits.** `split_text_embeddings` and `CustomCLIP` compute all 132 rows
and then slice, so the novel-split score is exactly
`sigmoid(logit[novel]) × sigmoid(int)`.

**The last place the untruncated vector exists** is `pre_preds` as `modelC`
returns it (`e2e:337-338`). It is consumed in `cli/common.py:157-165`, just
before `process_pred`.

Truncation points, in order:

| # | point | where | still present under GT trajectories? |
|---|---|---|---|
| — | detector score filter, track-length filters (`> 0.15 × span`, `≥ 12` frames) | `e2e:319`, `gl:626` | no, bypassed |
| — | pair overlap ≥ 10 frames; tail < 10 frames trimmed | `gl:659-667` | **yes** (pair level) |
| 1 | `novel` split: base columns hard-zeroed | `rc:511-514` | yes, avoided by caching `all` logits |
| 2 | **per-slot top-20** (`clip_top_n`) | `pp:19` | yes |
| 3 | association: greedy same-label chaining, score averaging | `pp:54-100` | yes |
| 4 | **top 200 per video** (`max_per_video`) | `pp:107-110` | yes |

`--use_prior` would add a class prior at `pp:12-15`. It is off by default
(`utils/parser_func.py:93`).

### (c) Paraphrase test: recomputing a predicate's text features from a new string

**Where the predicate string enters** (in every case after `name.replace("_", " ")`):

1. **The learned prompt.** `Model.__init__` builds `classnames` (`rc:452-453`).
   `PromptLearner.__init__` turns each into `"X X … X {name}."`, with 16 `X`
   placeholders (`rc:211`, `rc:224-226`). This is tokenised and embedded, and
   stored as `token_prefix` / `token_suffix` (`rc:235-236`), which are **saved
   in the checkpoint for the 132 current names**. A new string must rebuild
   its own suffix: tokenise it, then embed it with the CLIP token embedding.
2. **Three fixed templates**, encoded by a freshly loaded, frozen CLIP
   ViT-L/14@336px (`rc:401-416`). They are non-persistent buffers, so they are
   not in the checkpoint:
   - `"An image of a person or object {name} something."`
   - `"An image of something {name} a person or object."`
   - `"An image of the visual relation {name} between two entities."`

**What the recomputation needs:**

- *Learnable, from the checkpoint:* `ctx` (16×768) and `meta_net`.
- *Frozen:* the CLIP text tower (a copy is in the checkpoint under
  `pre_classifier.text_encoder`) and the tokenizer (`vlm/backbones/clip`).
- *Per pair:* the conditioning input, i.e. the mean over slots of
  `visual_pre_embeddings` (3072, 6,144 B) or directly the 768-d `bias`. **The
  learned prompt is instance-conditioned, so a paraphrase's text embedding
  differs for every pair.**
- *Per slot:* `visual_pre_embeddings` and the interactiveness logit, from (a).

**Recipe:**
1. Take `ctx`, and add `bias` at the 8 unmasked positions (`rc:268-275`).
2. Build the prompt `[prefix, ctx, suffix(new name)]` and run it through the
   text tower. Normalise the result.
3. Concatenate it with the three template embeddings of the new name, then
   divide by 2.
4. Compute `logit = v_slot · t / 0.01`, and
   `score = sigmoid(logit) × sigmoid(int)`.

This costs one text-tower pass per (pair, paraphrase), which is cheap on a GPU
and slow on this CPU.

---

## B5 — projected cache size, under GT trajectories

**Counts.**
- **Train:** 800 videos, 244,100 frames, 2,430 whole tracks, 374,190
  track-frames, 6,326 pairs, 674,010 pair-frames, 23,222 slots. Computed from
  `anno/train`.
- **Test:** 200 videos and 52,104 frames (from the frame directories). 583
  trajectories, 1,392 pairs and 8,508 slots come from pilot §B.9, which counted
  them earlier; I did not re-open test annotations. Test track-frames and
  pair-frames are **estimated** from the train per-frame ratios, at ≈ 79.9 k and
  ≈ 143.9 k.

| item | unit | count (train + test) | total |
|---|---|---:|---:|
| depth-1 slot inputs | 6,504 B / slot | 31,730 | 206 MB |
| spatial-decoder tokens | 6,144 B / slot | 31,730 | 195 MB |
| depth-2 `visual_pre_embeddings` | 6,144 B / slot | 31,730 | 195 MB |
| logits (132) + interactiveness | 266 B / slot | 31,730 | 8 MB |
| per pair: conditioning mean + subject/object embeddings | 9,216 B / pair | 7,718 | 71 MB |
| per frame: CLIP-L frame embedding | 1,536 B | 296,204 | 455 MB |
| per (track, frame): RoI mean | 1,536 B | ≈ 454 k | ≈ 697 MB |
| per (pair, frame): union RoI mean | 1,536 B | ≈ 818 k | ≈ 1,256 MB |
| **recommended set, total** | | | **≈ 3.0 GB** |
| ✗ raw TagCLIP patch grids, every frame | 884,736 B / frame | 296,204 | **≈ 262 GB** |
| ✗ detector tokens (`patch_`) | — | — | not needed: GT trajectories bypass the detector |

Per training video, the recommended set averages 2.96 MiB (median 1.64, p95
9.66, max 39.4 MiB). That is 2.32 GiB over all 800.

**Over 100 GB only if the raw patch grids are kept.** Proposal: drop them.
**What that costs:** without re-encoding frames on a GPU, you can no longer
pool new regions (part boxes, other pairs) or try a different pooling scheme.
The per-frame CLIP + TagCLIP encoding is the dominant GPU cost. The pilot's full
test run took 122 min for 200 videos, including the detector, on the rented
box. A middle option is to keep the grids at slot **mid-frames only**, the
frames `modelC` actually uses. That's ≤ 28 GB (upper bound: 31,730 slots, fewer
unique frames), and it keeps re-pooling possible at slot resolution.

---

## B6 — recommended cache set

Run one GPU pass per video, over GT whole tracks and in the `all` split,
through a wrapper script with forward hooks. No model code changes.

1. **Per (ordered pair, slot):**
   - logits over all 132 + the interactiveness logit. *Fusion needs exactly
     this granularity, and one pass serves both splits.*
   - `visual_pre_embeddings` (3072). *This is depth 2, and the logit side of the
     paraphrase recompute.*
   - the depth-1 slot inputs, and the spatial-decoder tokens. *Cheap, and they
     let the head start at either depth.*
   - slot boundaries.
2. **Per pair:** the conditioning mean (3072), subject/object embeddings,
   categories, GT tids, and box sequences. *The paraphrase test needs the
   conditioning. The box sequences join to `measured_axes.py`, which works on
   the same GT boxes.*
3. **Per frame:** the CLIP-L frame embedding, and the TagCLIP RoI means per
   (track, frame) and union RoI means per (pair, frame). *This is the one thing
   `modelC` throws away, since it looks at one frame per slot. The measured
   motion axes, and any learned locomotion or posture head, need frame
   resolution.*
4. **Once:** `ctx`, `meta_net`, `token_prefix` / `token_suffix` from the
   checkpoint, and the three fixed-template embeddings for the 132 names.
   Reload the tokenizer and text tower as needed.
5. **Skip** the raw patch grids (≈ 262 GB). Keep them at mid-frames (≤ 28 GB)
   only if re-pooling at slot resolution is wanted.

The total is ≈ 3 GB. The wrapper has to handle the four B3 constraints:
build whole tracks rather than using the fragment file, the 40-slot cap,
integer frame keys, and the pre-roll.
