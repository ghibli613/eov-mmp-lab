# The code as it actually runs

[01_Architecture.md](01_Architecture.md) describes the *paper*. This describes
the *implementation*: the path a video takes through the code, what each stage
computes, and the shape of every tensor along the way.

Read this before changing anything. Read [10_Known-issues.md](10_Known-issues.md)
when something surprises you.

---

## 1. The one idea everything else serves

A normal classifier ends in `nn.Linear(d, n_classes)`. The weight matrix has one
row per class, learned from examples of that class, so the set of classes is
frozen at training time.

This model replaces that matrix with **CLIP text embeddings of the class names**:

```python
prompts = [f"An image of {name}." for name in classnames]
text_embeddings = clip_model.encode_text(clip.tokenize(prompts))
similarity = visual_features @ text_embeddings.T
```

The row for "zebra" is now *computed from the word* "zebra" rather than learned
from zebra pixels. Hand it a class name it has never trained on and it still
produces a row — that is the whole of "open vocabulary", and everything else in
this repository is in service of making the visual side good enough for that
comparison to mean something.

The same trick appears three times: for object categories in
[models/object_classifier.py](../models/object_classifier.py), for predicates in
[models/relation_classifier.py](../models/relation_classifier.py), and inside the
detector, which is distilled against CLIP so that its region features live in the
same space.

---

## 2. The path a video takes

```bash
python -m cli.evaluate --ckpt_path output/ckpt/..._end2end_base-001.pth
```

```
video (≈250 frames)
  │
  ├─ Dataset_new.__getitem__          CLIP + TagCLIP over every frame
  │     → patch_       (F, 576, 1024)   patch tokens, pre-projection
  │     → patch_proj   (F, 576,  768)   patch tokens in the joint text space
  │     → global_proj  (F,       768)   one vector per frame
  │
  ├─ modelA  detector                 300 queries per frame → boxes
  ├─ modelB  object classifier        each box → a category name
  ├─ deep_sort + AFLink               per-frame boxes → trajectories
  │
  ├─ gen_labels                       every (subject, object) pair,
  │                                   cut into 30-frame clips → 4 streams
  ├─ modelC  relation classifier      per clip → 132 predicate scores
  │
  └─ post_process                     clips → instances → ranked triplets
        → vidvrd_eval_api             mAP, R@50, R@100
```

Two things about this are worth internalising early.

**The dataset is an encoder, not a loader.** `__getitem__` returns CLIP features,
not images. That is why constructing one costs GB of VRAM and why iterating it
dominates runtime. It also means the detector never sees pixels: `patch_` *is*
its input.

**`split` is a runtime string, not a second model.** `cli/common.py` runs the
same weights twice, setting `model.modelC.tgt_split` to `"all"` and then
`"novel"`. Only which columns of the text-embedding matrix are used changes. The
"novel mAP" and "all mAP" in every table come from one set of weights.

---

## 3. What the model sees for one pair, in one clip

This is the centre of the system, so it is worth being concrete. Take one
subject-object pair over one 30-frame clip. Four things are computed for it.

### 3.1 Appearance — four regions, one frame

`_extract_clip_feat` in [models/gen_labels.py](../models/gen_labels.py) takes the
clip's **middle frame** and pools CLIP's patch tokens under four masks:

| # | Region | Why it is there |
|---|---|---|
| 0 | the subject's box | what the subject looks like |
| 1 | the object's box | what the object looks like |
| 2 | their **union** box | the space *between* them — where `sit_on` or `ride` actually lives |
| 3 | the whole frame | scene context: a river makes `swim` likely |

Pooling is a weighted average of the 576 patch tokens under a 24×24 binary mask
of the box — the TagCLIP trick, which is how you get a region embedding out of a
frozen CLIP that was only ever trained on whole images.

Result: **(4, 768)**.

Region 2 is UASAN's contribution, absorbed into the CLIP branch. Region 3 is
free context — the frame is already encoded.

### 3.2 Geometry — the same four regions, as numbers

`_extract_bbox_feat` describes each of those four regions with 24 numbers: its
box at the clip's **first** frame (8), at its **last** frame (8), and the
**difference** (8). Each box is given as `[x1, y1, x2, y2, cx, cy, w, h]`,
normalised by frame size.

Result: **(4, 24)** — one geometry vector per appearance vector.

### 3.3 Pair geometry — where the two objects are relative to each other

`vru19_ext_loc_feat` produces 14 numbers for a subject-object box pair:

```
 5  subject : x1, y1, x2, y2, area fraction
 5  object  : x1, y1, x2, y2, area fraction
 4  relative: Δx/w_obj, Δy/h_obj, log(w_sbj/w_obj), log(h_sbj/h_obj)
```

`rel_feat` evaluates that at the clip's **first, middle and last** frame and
concatenates: **(42,)**.

### 3.4 Pair motion — how that geometry changed

`mot_feat` takes the same three snapshots and stores their **differences**:
`mid − begin`, `end − mid`, `end − begin`. Also **(42,)**.

So `rel_feat` says *where things are*, and `mot_feat` says *how that changed* —
the same 14 numbers, in positions and in velocities. This is the entire motion
model of the method. It is enough to separate `walk_left` from `stand_left`
(the boxes move) and `run_left` from `walk_left` (they move faster).

### 3.5 Putting it together

```
per clip:   clip_feat (4, 768)   bbox_feat (4, 24)   rel_feat (42)   mot_feat (42)
```

Each of the four is switchable by its `--<name>` flag. A disabled stream is fed
as **zeros** rather than removed, so shapes and every downstream module are
untouched and the ablation means precisely "this evidence was unavailable".

---

## 4. From those streams to 132 scores

### 4.1 Spatial, then temporal

`SpatialDecoder` works **within one clip**. For each of the four regions it adds
three things — the appearance vector, a projection of that region's 24 geometry
numbers, and a learned **role embedding** saying *which* of the four this is —
then runs a Transformer encoder layer over the set of four. The role embedding is
what lets one shared layer treat "subject" and "union" differently.

`TemporalDecoder` then works **across clips**: for each role independently it
runs a Transformer layer over the clip sequence, with positional embeddings. It
also emits an **interactiveness** score per clip — is this pair interacting at
all? — which later multiplies into every predicate score.

### 4.2 The classifier weights are four text embeddings side by side

`split_text_embeddings` builds the predicate classifier as a concatenation:

```python
pre_text_embeddings = torch.cat([sbj, obj, uni, learned], dim=-1) / 2
```

where the first three are frozen CLIP text embeddings of three different
sentences about the same predicate:

| Block | Sentence | Pairs with |
|---|---|---|
| `sbj` | *"An image of a person or object {p} something."* | the subject region |
| `obj` | *"An image of something {p} a person or object."* | the object region |
| `uni` | *"An image of the visual relation {p} between two entities."* | the union region |
| `learned` | a learned, vision-conditioned prompt | the whole-frame region |

Because the visual side is also 4×768, the dot product decomposes into **a sum of
four matches**, each pairing one visual role with text written from that role's
point of view. Asking "does the subject look like something that is *chasing*?"
and "does the region between them look like *chasing*?" are different questions,
and the model asks both.

Three of the four blocks are frozen CLIP; only the fourth is learned. That ratio,
and the flat `/ 2`, are fixed constants — see
[91_Extension-guide.md](91_Extension-guide.md) §2.

### 4.3 The learned prompt

`PromptLearner` is CoCoOp. There are 16 learnable context token embeddings, and a
small `meta_net` that maps the clip's mean visual feature to a bias vector. An
alternating mask zeroes that bias on every other position:

```python
mask = [[1],[0],[1],[0], ...]          # 16 entries
bias_masked = bias_reshape.masked_fill(mask, 0)
ctx_shifted = ctx + bias_masked
```

So the prompt is **8 static context tokens interleaved with 8 that shift with
what the model is looking at**. The static half carries what "chase" means in
general; the conditioned half adapts it to this pair. That split is the point of
CoCoOp: a fully static prompt overfits the training categories, and a fully
conditioned one has nothing stable to generalise from.

### 4.4 Scoring

```python
pre_scores = sigmoid(visual @ text.T / 0.01) * interactiveness
sbj_scores = softmax(visual_sbj @ obj_text.T / 0.01)
obj_scores = softmax(visual_obj @ obj_text.T / 0.01)
```

Predicates use **sigmoid** because a pair can hold several relations at once —
a dog can be both `run_behind` and `larger_than` a person. Objects use
**softmax** because a box is one thing. τ = 0.01 is the paper's temperature.

---

## 5. Trajectories

Relations are between *trajectories*, so per-frame boxes have to be linked.
[models/tracking/](../models/tracking) runs deep_sort — Kalman-filter motion
prediction plus appearance matching — and then **AFLink**, a small learned
network that repairs breaks by deciding whether two tracklet fragments are the
same object.

Camera motion is compensated first, using the precomputed ECC homographies in
`data/vidvrd/data/VidVRD_ECC_*.json`; without that, a panning shot looks like
every object moving at once.

This stage sets the ceiling on everything downstream: a detection counts only at
**vIoU ≥ 0.5 on both trajectories**, so a good predicate on a poor trajectory
scores zero. It is why EOV's jump in `mAP_o` produced a jump in relation mAP.

---

## 6. From clip scores to a ranked list

[inference/post_process.py](../inference/post_process.py), three small pure
functions:

**`process_pred`** keeps the top `--clip_top_n` (default 20) predicates per clip,
turning dense scores into candidates.

**`association`** stitches candidates across clips into relation *instances*: two
consecutive clips merge when they carry the same predicate, the merged instance
spans both durations, and its score is the mean of its clips'.

**`format_`** computes the final ranking score

```python
score = sbj_scr * obj_scr * pre_scr
```

sorts, and keeps the top `--max_per_video` (default 200). A triplet is only as
confident as its least confident part.

### The metric

[third_party/vidvrd_eval_api/](../third_party/vidvrd_eval_api/):

```python
video_ap[vid] = voc_ap(det_rec, det_prec)
mean_ap = np.mean(list(video_ap.values()))
```

mAP is **averaged over the 200 test videos**, instance-weighted within each — not
macro-averaged over predicate classes. Each video counts the same regardless of
how many relations it contains, and within a video the common predicates
dominate. [20_Benchmark-analysis.md](20_Benchmark-analysis.md) is about what
follows from that.

---

## 7. Three things that will bite you

Short list, kept here because they are properties of the code rather than
research questions; the reasoning is in [10_Known-issues.md](10_Known-issues.md).

**Batch size must be 1.** `PromptLearner.forward` contains
`bias.expand(1, 16, 768)`, which hard-codes the batch dimension. Verified: batch
2 raises `RuntimeError`. This is why the defaults and the released checkpoint
names all say `bs1`, and it is the first thing to fix if you want training to go
faster.

**`modelA` / `modelB` / `modelC` are checkpoint keys.** They are unhelpful names,
and they must stay. Renaming them silently invalidates every released checkpoint,
because `load_state_dict(strict=False)` reports mismatches rather than raising.

**Appearance comes from one frame per clip.** §3.1 samples the middle frame only.
Within a clip the model has no other visual evidence — the other 29 frames
contribute nothing. This is a design fact rather than a bug, but a great deal
follows from it: see [20_Benchmark-analysis.md](20_Benchmark-analysis.md).
