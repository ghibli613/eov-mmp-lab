# Data pipeline

VidVRD, its annotation formats, the two datasets, and the indexing traps that
produce a model which trains perfectly and scores zero.

---

## 1. The dataset

**VidVRD** — 1,000 videos (800 train / 200 test), 35 object categories, 132
predicate categories. Frames are **640×360**, and the annotations are in that
same pixel space (verified, not assumed).

Open-vocabulary split: categories are divided `base` / `novel` by
`data/vidvrd/data/openvoc_{obj,pred}_class_spilt_info.json`. **Training sees base
only.** Evaluation runs twice on one set of weights — "all" and "novel" — by
changing which columns of the text bank are used.

VidOR (10,000 videos, 80 objects, 50 predicates) is what
`configs/metor_vidor.yaml` targets, but **the data is not in this workspace**.

### On disk

```
data/vidvrd/
  frames/<video>/%06d.jpg     decoded frames, 1-INDEXED  (42 GB, symlinked)
  anno/{train,test}/*.json    per-video annotation
  videos/                     source .mp4 (4.3 GB)
  data/                       trajectories, splits, ECC matrices, CLIP bank
```

### Annotation format

```jsonc
{ "video_id": "...", "width": 640, "height": 360, "frame_count": 72,
  "subject/objects":     [ {"tid": 0, "category": "dog"}, ... ],
  "relation_instances":  [ {"subject_tid": 0, "object_tid": 1,
                            "predicate": "stand_behind",
                            "begin_fid": 0, "end_fid": 30}, ... ],
  "trajectories":        [ /* per frame, per tid, bbox */ ] }
```

Relations are **temporally scoped** — `begin_fid`/`end_fid` — which is why
predicate labels carry a time axis.

The prepared trajectory files (`train_object_trajectories_gt.json`) key boxes by
frame index as a **string**: `{"0": [x1,y1,x2,y2], "1": [...], ...}`.

---

## 2. Key-frame sampling

Sec. 4.2: *"key frames are sampled every 30 video frames to form 30-frame video
segments."*

One model timestep `t` = one key frame. Measured over the real dataset:

| | |
|---|---|
| total frames | 296,204 |
| key frames at stride 30 | **10,306** (3.5%) |
| key frames per video | median 7, mean 10.3, p90 22, p99 47, **max 184** |

Two consequences:

- **You only need ~1.5 GB of frames, not 42 GB** — which is what makes Colab
  practical → [doc 40](40_Hardware.md).
- **The 184-frame video will OOM.** The model encodes all of a video's key frames
  in one forward with gradients retained. ~10% of videos exceed what a 16 GB GPU
  holds. This is an open defect → [doc 32 §4](32_Not-implemented.md).

---

## 3. Three indexing traps

Each of these produces a model that trains happily and evaluates to nonsense.
All three are pinned by tests.

### 3.1 Files are 1-indexed; trajectories are 0-indexed

`000001.jpg` is the first frame, but its boxes live under trajectory key `"0"`.
The dataset stores `frame_ids = frame_number - 1` and looks up with that. Off by
one and every box is shifted by a frame.

### 3.2 The key-frame grid must match the EOV loader's

Both use `(frame_number - 1) % stride == 0`, so frame 1 is always a key frame.
If the two datasets disagreed, `t` would mean different things in the two
architectures and no A/B between them would be valid.

### 3.3 Boxes must be normalised `cxcywh`

`xyxy_to_cxcywh_normalised(box, w, h)` divides by the frame size and converts.
This matches the sigmoid output of `TrajectoryRegressionHead` and
Deformable-DETR's convention. Feed it `xyxy` and the loss is finite, decreasing,
and meaningless.

---

## 4. Why there are two datasets

`feature_dataset.py` and `frame_dataset.py` sound like duplicates. Their outputs
overlap on `video_name` and `image_size` — and nothing else.

| | `feature_dataset.py` (`Dataset_new`) | `frame_dataset.py` (`METORDataset`) |
|---|---|---|
| Returns | `patch_`, `patch_proj`, `global_proj` — **CLIP features** | `images` — **pixels** |
| Plus | `gt_traj`, `gt_rel`, `begin_fid`, `end_fid` | `gt_boxes`, `gt_box_mask`, `gt_labels`, `pair_labels`, `present_obj/rel` |
| For | `--arch metor-compat` | `--arch metor` |

The reason is structural:

- **EOV's dataset is an encoder.** Its detector consumes patch tokens, so
  `__getitem__` runs CLIP. The detector never sees pixels.
- **METOR cannot use pre-encoded features.** `--cre_mode inject` must run the ViT
  *inside* the training step so gradients reach `c_o`/`c_r`
  ([doc 11 §4](11_Contextual-refinement-encoding.md)); cached features are
  already past that point. And Eq. 5 regresses trajectories, so it needs
  per-frame **box targets** that EOV never had to produce.

Merging them would give one class whose `__getitem__` returns two incompatible
dictionaries depending on a flag — strictly worse than two classes. What *was*
wrong was the naming: the files now say what they return rather than who owns
them.

*(The training scripts, by contrast, **were** merged — one `cli/train.py`
dispatching on `--arch`. → [doc 42 §1](42_Extending.md))*

---

## 5. What `METORDataset` returns

One item = one video.

| Key | Shape | Meaning |
|---|---|---|
| `images` | `(T, 3, 336, 336)` | CLIP-normalised key frames |
| `frame_ids` | `list[int]` | 0-indexed, for trajectory lookup |
| `gt_boxes` | `(n_traj, T, 4)` | normalised `cxcywh` |
| `gt_box_mask` | `(n_traj, T)` | where GT actually exists |
| `gt_labels` | `(n_traj,)` | object category ids |
| `pair_labels` | `dict[(s_row, o_row) → (T, n_rel)]` | per-pair, per-timestep predicates |
| `present_obj`, `present_rel` | `(n_obj,)`, `(n_rel,)` | multi-hot, for Eq. 16 |

**`pair_labels` is keyed by row index into `gt_boxes`, not by `tid`.** The
matcher works in row space; a raw `tid` would silently mis-target every label.

### Batch size 1, enforced

`metor_collate` raises on any batch larger than 1. The paper trains one video at
a time (Sec. 4.2), and the shapes are genuinely ragged — different `T`, different
trajectory counts, different pair sets. Failing loudly beats collating a batch of
2 into something that trains and means nothing.

---

## 6. Other data files

| File | Size | What |
|---|---|---|
| `clip_L14_feat_vidvrd.pkl` | 278 MB | Author-supplied CLIP embeddings of 189,345 ground-truth boxes. Used by OV-DETR's conditional queries. Float16, **not** L2-normalised. |
| `VidVRD_ECC_{train,test}.json` | 42 MB | Per-video camera-motion homographies for deep_sort. Without them a panning shot looks like every object moving. |
| `prior.pkl` | | Predicate co-occurrence counts. `--use_prior` exists but should stay off — a prior from counts cannot generalise to unseen compositions. |
| `openvoc_*_class_spilt_info.json` | | The base/novel splits. Do not edit in place; a test pins them. |
