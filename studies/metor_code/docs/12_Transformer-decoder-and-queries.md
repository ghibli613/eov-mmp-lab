# The Transformer decoder and the object queries

**Paper:** Sec. 3.2, Eq. 2 · **File:** [`metor/model.py`](../metor/model.py)

> ⚠ This is the weakest part of the reconstruction. Read §4 before drawing any
> conclusion from a trained model.

---

## 1. What Eq. 2 says

```
O = Decoder(Q̂, H_patch)
```

`N_q` learnable object queries, refined by Eq. 8, decoded against the frozen
CLIP patch tokens of each frame. Output `O` is `(N_q, T, d)` — `N_q` candidate
objects across `T` timesteps, which is exactly what Eq. 3 consumes.

The queries themselves are `nn.Embedding(n_queries, d_model)` at
[model.py:125](../metor/model.py#L125). They are the model's slots: each one
learns to look for "a thing", and which thing is decided per video by the
decoder.

---

## 2. `N_q = 100`, and why that number matters more than it looks

Sec. 4.2: *"The number of object queries `N_q` is set to 100."*

EOV-MMP uses **300** (its own text: *"we use six Transformer layers and 300
object queries"*). This single difference is why:

- **EOV-MMP's released detector checkpoint cannot be loaded into `--arch
  metor`.** `query_embed.weight` is `(300, ·)` against `(100, ·)`. The
  architectures differ, not just the weights. No amount of `strict=False` helps
  — it would load nothing and silently leave the queries random.
- `--arch metor-compat` exists at all ([00_Start-here §4](00_Start-here.md)).

It also changes the pair combinatorics — see
[16_Matching-and-pair-formation](16_Matching-and-pair-formation.md).

---

## 3. What is implemented

`VanillaQueryDecoder`, [model.py:52](../metor/model.py#L52):

```
patch_proj : Linear(1024 → 256)      ViT width → decoder width
decoder    : nn.TransformerDecoder, 6 layers, 8 heads
norm       : LayerNorm
```

Queries are decoded against each frame independently, then transposed to
`(N_q, T, d)`.

Depth 6 and 8 heads are not guesses — EOV-MMP states six layers and MMP states
eight heads ([31_Lineage-evidence](31_Lineage-evidence.md)). Width 256 is
Deformable-DETR's, which is what the paper says it initialises from.

---

## 4. What is NOT implemented — the COCO initialisation

> "The Transformer decoder is initialized with the parameters of an object
> detector pre-trained on the MS-COCO dataset, excluding novel object
> categories." — Sec. 4.2

**This is not wired up.** `VanillaQueryDecoder` is a plain
`nn.TransformerDecoder` starting from random init.

### Why it dominates everything else

Open-vocabulary detection ability does not come from the video data. VidVRD has
800 training videos and 35 object categories, 25 of them base. It comes from:

1. a detector pretrained on COCO's 80 categories with far more data, and
2. CLIP distilled into that detector so its region features live in CLIP's text
   space.

EOV-MMP's own pipeline makes this explicit — its step 1 initialises "the
query-based Transformer decoder, object embedding head, and bounding box
regression head with pre-trained parameters from MS-COCO (excluding novel object
categories in Open-VidVRD), while the relationship classification head is
initialized with random parameters."

Without it, `mAP_o` of 55.09 is out of reach, and since a triplet only counts at
vIoU ≥ 0.5 on **both** trajectories, poor trajectories cap the relation mAP too.
Expect any run of this repository as it stands to fall well short, for this
reason alone and before any [30_Paper-gaps](30_Paper-gaps.md) entry is
considered.

### What doing it properly involves

The real detector is vendored at
[`models/methods/detectors/ov_prompt/`](../models/methods/detectors/ov_prompt/) —
a CLIP-distilled Deformable DETR, complete with its OV-DETR conditional-query
machinery and the CUDA deformable-attention kernels in [`ops/`](../ops/).

Steps:

1. Build it with `--det_token_num 100` instead of 300.
2. Load COCO weights with novel Open-VidVRD categories removed.
3. Feed it `Q̂` from Eq. 8 instead of `self.query_embed.weight` — the hook is
   `query_embeds = self.query_embed.weight` in `OVDETR.forward_train`.
4. Pass it as `decoder=` to `METOR.__init__`
   ([model.py:108](../metor/model.py#L108)), which exists for exactly this.

This is a real project, not a config change. Note also that EOV-MMP never
released the *training* code for its steps 1–3, so there is no reference loop to
copy — only the architecture and the checkpoint format.

Recorded as the headline item in [32_Not-implemented](32_Not-implemented.md).

---

## 5. Why `VanillaQueryDecoder` is here at all

It is not a substitute for the above; it is what lets the rest of the framework
be built, shape-checked and tested at all:

- no CUDA extension to compile,
- runs on CPU, so `tests/test_metor_model.py` works anywhere,
- makes every *other* component's correctness checkable independently of the
  detector question.

Which is the right split: the pieces METOR actually contributes (Φ and Ψ) can be
verified now, and the piece it inherits can be wired later.
