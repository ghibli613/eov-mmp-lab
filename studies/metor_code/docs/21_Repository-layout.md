# Repository layout

Two kinds of code live here, and it is worth knowing which you are reading.

**Ours** — `metor/`, `configs/`, `docs/`, and the METOR tests. This is the paper,
implemented.

**Vendored** — everything else, taken from `../ov-vidvrd-lab`, a cleaned refactor
of EOV-MMP. It is infrastructure: CLIP, the detector, the tracker, the official
metric. METOR builds on it the way the paper builds on its predecessor.

---

## Ours

### `metor/`

All METOR-specific code. Nothing else in the repository is.

| File | What |
|---|---|
| `model.py` | `METOR` — Eq. 1–5 assembled; `VanillaQueryDecoder`; `build_metor()` |
| `modules/contextual_encoding.py` | Φ, Sec. 3.3 → [doc 11](11_Contextual-refinement-encoding.md) |
| `modules/iterative_enhancement.py` | Ψ, Sec. 3.4 → [doc 13](13_Iterative-enhancement.md) |
| `modules/criterion.py` | Eq. 12–16 → [doc 15](15_Training-objective.md) |
| `modules/heads.py` | Eq. 4–5 → [doc 14](14_Scoring-and-trajectory-heads.md) |
| `matcher.py` | query↔GT assignment, pair targets → [doc 16](16_Matching-and-pair-formation.md) |
| `engine.py` | the `--arch metor` training loop |
| `compat.py` | METOR's modules spliced into the EOV stack |

### `configs/`

| File | Contains |
|---|---|
| `metor_vidvrd.yaml` | **only values the paper states**, each with the quotation |
| `metor_vidor.yaml` | same, for VidOR |
| `metor_defaults.yaml` | **only values this reconstruction invented** |
| `VidVRD_*.json`, `VidOR_*.json` | vendored category-split files |

The split is load-bearing — it is what keeps "the paper says 100 queries" visibly
separate from "we picked 10 pairs". `tests/test_metor_config.py` fails if a
value crosses over. → [doc 23](23_Configuration-and-flags.md)

### `docs/`

You are here. → [index](README.md)

---

## Vendored

### `vlm/` — Vision-Language Models

The frozen backbones, ~4.3 MB of code.

| Path | What | Needed |
|---|---|---|
| `backbones/clip/` | OpenAI CLIP. **This is the model METOR freezes and wraps.** | yes |
| `backbones/clip_tagclip/` | TagCLIP — a CLIP variant that yields usable **patch-level** embeddings. It is how a *region* embedding is pooled out of a model only ever trained on whole images. | for the EOV path |
| `backbones/internvideo/` | A video encoder, ~8.8k lines. Currently unreachable — retained because a real video backbone is an obvious future direction. | no |
| `text_encoder.py` | EOV's prompt-learning text side | for the EOV path |

`ContextTokenInjector` reaches into `vlm.backbones.clip`'s `VisionTransformer` —
that is the coupling to know about.

### `models/` — the EOV-MMP stack

| Path | What | Needed |
|---|---|---|
| `methods/detectors/ov_prompt/` | The CLIP-distilled **Deformable DETR**. Also the source for the COCO decoder init METOR needs → [doc 12 §4](12_Transformer-decoder-and-queries.md) | for `metor-compat`, and eventually for `metor` |
| `object_classifier.py` | EOV's vision-guided object classifier | `metor-compat` |
| `relation_classifier.py` | EOV's relation classifier — the module `compat.py` splices into | `metor-compat` |
| `tracking/` | deep_sort + **AFLink**. Links per-frame detections into trajectories. METOR replaces this with Eq. 5. | `metor-compat` |
| `gen_labels.py` | EOV's per-clip feature extraction (4 regions, geometry, motion) | `metor-compat` |
| `end2end_model.py` | EOV's `End2End_Model` — note it calls `.backward()` *inside* its forward | `metor-compat` |

### `ops/` — CUDA kernels

Multi-scale deformable attention (`MSDeformAttn`) as C++/CUDA. Deformable DETR
cannot run without it.

- Build with `ops/make.sh`.
- `ops/test.py` checks the kernel against a PyTorch reference.
- **Not needed for the test suite** or for `--arch metor` with
  `VanillaQueryDecoder`.

### `third_party/`

| Path | What |
|---|---|
| `vidvrd_eval_api/` | **The official VidVRD metric.** mAP and Recall@K. Deliberately untouched — editing it would make every number incomparable to published results. Note mAP is averaged over the 200 test videos and instance-weighted within each, *not* macro-averaged over predicate classes. |
| `detectron2/` | A trimmed subset of detectron2 layers the detector needs |
| `vidvrd_ii_helper/` | Prepares ground truth for evaluation |

### `data_loading/`

Two datasets that look like duplicates and are not.
→ [doc 22](22_Data-pipeline.md)

### `inference/`

`post_process.py` — clip scores → ranked triplets (top-N per clip, associate
across clips, `sbj × obj × pre` ranking). `video_relation_detection_openvoc.py`
drives evaluation. Belongs to the EOV path today.

### `utils/`

| File | What |
|---|---|
| `paths.py` | **Every filesystem path**, env-overridable. Run `python -m utils.paths` to check they resolve. |
| `parser_func.py` | The argparse definition, including the METOR argument group |
| `eov_utils.py` | Logger, `AverageMeter`, focal losses |

### `cli/`

| File | What |
|---|---|
| `train.py` | **The** training entry point; dispatches on `--arch` at [line 60](../cli/train.py#L60) |
| `evaluate.py` | Evaluation (the EOV path today) |
| `common.py` | Shared model-stack construction |
| `config.py` | YAML layering → [doc 23](23_Configuration-and-flags.md) |

### `tools/`

One-off scripts: `extract_frames.py`, `build_clip_object_bank.py`,
`prepare_data.py`, HF up/download, `weights_manifest.json` (checksums for the
author-supplied files).

### `tests/`

→ [doc 41](41_Testing.md). `stub_clip.py` is a 60k-parameter CLIP stand-in that
makes the whole framework testable without loading ViT-L.

---

## Symlinks, not copies

`data/` → `../ov-vidvrd-lab/data/` and `output/ckpt/` →
`../ov-vidvrd-lab/output/ckpt/`.

That avoids duplicating **42 GB of frames** and ~9 GB of checkpoints. The repo
itself is about 10 MB. Override with `VIDVRD_DATA_ROOT`, `VIDVRD_FRAME_DIR`,
`VIDVRD_OUTPUT_ROOT` if your data lives elsewhere.
