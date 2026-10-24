# Configuration and flags

---

## 1. Config layering

```bash
python -m cli.train --config configs/metor_defaults.yaml \
                    --config configs/metor_vidvrd.yaml \
                    --n_iter 3
```

Precedence, lowest to highest:

```
argparse defaults  <  --config files, in the order given  <  flags you typed
```

That last step is the one that matters: **a config file never overrides
something typed on the command line.** Without that rule an ablation sweep would
silently not run. Implemented in [`cli/config.py`](../cli/config.py); pinned by
`test_explicit_flags_beat_config_files`.

### Why the configs are split in two

| File | Contains | Rule |
|---|---|---|
| `metor_vidvrd.yaml`, `metor_vidor.yaml` | **only values the paper states**, each with its quotation | never add an invented value here |
| `metor_defaults.yaml` | **only values this reconstruction invented** | never add a paper value here |

This is what keeps "the paper says `N_q` = 100" visibly distinct from "we picked
10 pairs". `test_paper_and_invented_values_stay_in_separate_files` fails if one
crosses over, and `test_vidvrd_config_matches_the_paper` pins each stated value
as a literal — so drift breaks a test rather than going unnoticed.

---

## 2. What the paper states

All quoted from Sec. 4.2, all in the dataset configs.

| Flag | Value | Quotation |
|---|---|---|
| `--n_queries` | 100 | "The number of object queries N_q is set to 100." |
| `--n_iter` | 2 / 3 | "set to two for VidVRD and three for VidOR" |
| `--alpha` | 0.9 | "The hyperparameter α in Eq. 11 is set to 0.9." |
| `--theta_traj` | 1.0 | "set to 1.0, 0.2, and 0.1, respectively" |
| `--theta_ctx` | 0.2 | ″ |
| `--theta_cst` | 0.1 | ″ |
| `--lr` | 1e-4 | "an initial learning rate of 1e-4" |
| `--max_epoch` | 30 | "trained for a total of 30 epochs" |
| `--batch_size` | 1 | "only one video is processed at a time" |
| `--frame_stride` | 30 | "key frames are sampled every 30 video frames" |
| `--clip_backbone` | ViT-L/14@336px | "the ViT-L/14 variant of CLIP with fixed parameters" |
| `lr_milestones` | 15, 20, 25 | "a multi-step decay schedule is applied at epochs 15, 20, and 25" |
| `lr_gamma` | 0.1 | "reducing the learning rate by a factor of 0.1" |
| `box_filter_threshold` | 0.35 | "filter bounding boxes using a threshold of 0.35" |
| `traj_score_threshold` | 0.2 | "average classification score greater than 0.2" |

**VidOR differs from VidVRD in exactly one hyperparameter**: `n_iter` 2 → 3. A
test asserts that, so if a second difference ever appears it came from somewhere
other than the paper.

---

## 3. What the predecessors settle

Not in METOR, but stated identically by MMP and EOV-MMP. Full quotations in
[doc 31](31_Lineage-evidence.md).

| Flag | Default | Source |
|---|---|---|
| `--n_prompt_vec` | 8 | "eight tokens ... for learnable continuous prompts" |
| `--n_ctx_obj`, `--n_ctx_rel` | 8 | "... and learnable conditional prompts" |
| `--n_heads` | 8 | MMP: "head number ... is set to 8" |
| `--dropout` | 0.1 | MMP: "the dropout rate is set to 0.1" |
| `--n_dec_layers` | 6 | EOV: "we use six Transformer layers" |
| temperature | 0.01 | EOV: "τ in both Eq. 5 and Eq. 10 is set to 0.01" |

---

## 4. What this reconstruction invented

In `metor_defaults.yaml`. Each is a place a faithful reimplementation could
legitimately differ → [doc 30](30_Paper-gaps.md).

| Flag | Default | Note |
|---|---|---|
| `--max_pair_objects` | 10 | → 90 ordered pairs. **The largest unforced choice.** [doc 16 §2](16_Matching-and-pair-formation.md) |
| `--d_model` | 256 | Deformable-DETR's |
| `--st_max_len` | 512 | positional-embedding capacity for the ST transformer |
| `--residual_mapping` | off | a labelled DEVIATION from Eq. 11 |

---

## 5. Mode flags

### `--arch`

| Value | Meaning |
|---|---|
| `metor` (default) | the paper as written; 100 queries; cannot load EOV checkpoints |
| `metor-compat` | METOR's modules on the EOV stack; 300 queries; loads released checkpoints; runs today |

→ [doc 00 §4](00_Start-here.md)

### `--cre_mode`

| Value | Meaning |
|---|---|
| `inject` (default) | really appends context tokens to the frozen ViT — faithful to Eq. 7, costs a ViT-L pass per frame |
| `pool` | attention-pools cached patch embeddings — much cheaper, **an approximation, not equivalent** |

Any number produced with `pool` must say so. → [doc 11 §4](11_Contextual-refinement-encoding.md)

### `--rel_token_position`

| Value | Meaning |
|---|---|
| `end` (default) | what Eq. 10 literally prints |
| `three_quarter` | what **both** predecessors do |

Genuinely ambiguous; worth trying both if you train.
→ [doc 11 §3.3](11_Contextual-refinement-encoding.md)

---

## 6. Ablation recipes

Mapped to the paper's own tables.

### Table 3 — contextual refinement

```bash
python -m cli.train --config configs/metor_vidvrd.yaml               # full
python -m cli.train --config configs/metor_vidvrd.yaml --no_crq      # w/o CRQ
python -m cli.train --config configs/metor_vidvrd.yaml --no_crt      # w/o CRT
python -m cli.train --config configs/metor_vidvrd.yaml --no_crq --no_crt   # w/o CRE
```

| Variant | Their novel mAP |
|---|---|
| w/o CRE | 13.49 |
| w/o CRQ | 15.32 |
| w/o CRT | 14.91 |
| full | 16.43 |

### Table 4 — iteration count

```bash
for n in 0 1 2 3; do
  python -m cli.train --config configs/metor_vidvrd.yaml --n_iter $n --ps iter$n
done
```

| `--n_iter` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Their novel mAP | 15.16 | 16.04 | **16.43** | 16.25 |

### The one you can actually run today

On the EOV stack, on a 16 GB GPU, using the released checkpoints:

```bash
python -m cli.evaluate --arch metor-compat --n_iter 0 \
  --ckpt_path output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth \
  --path_AFLink output/ckpt/AFLink_epoch20.pth

python -m cli.evaluate --arch metor-compat --n_iter 2 ...    # same, then compare
```

`--n_iter 0` is bit-identical to EOV, so the difference is attributable to the
module. → [doc 13 §6](13_Iterative-enhancement.md)

---

## 7. Paths

Everything routes through [`utils/paths.py`](../utils/paths.py). Check it first:

```bash
python -m utils.paths        # prints every path and whether it exists
```

Overridable: `VIDVRD_DATA_ROOT`, `VIDVRD_FRAME_DIR`, `VIDVRD_OUTPUT_ROOT`.

---

## RB1 flags (clean EOV-MMP baseline)

Added for batch RB1 (pre-registration `prereg/preregistration.md`, A23). Every
flag defaults to the released behaviour; tests pin both settings
(`tests/test_rb1_*.py`).

| Flag | Where | Effect |
|---|---|---|
| `--exclude_novel_train` | step 4 training | no novel object category or novel predicate in any training target: detector boxes, relation instances touching a novel object or predicate, hence the interactiveness target. `models/novel_filter.py` |
| `--uniform_fusion` | inference | `0.3·classifier + 0.7·detector` for all 35 classes instead of the released per-class weights. `models/fusion.py` |
| `--no_epoch_eval` | `cli.train` | no test evaluation in the loop; the last epoch is the final model |
| `--epoch_ckpt_dir DIR` | `cli.train` | full-state checkpoint (model, optimizer, scheduler, RNG incl. loader) every epoch |
| `--resume CKPT` | `cli.train` | continue exactly from such a checkpoint |
| `--mirror_dir DIR` | `cli.train` | copy each epoch's checkpoint, video order and log to `DIR`, SHA256-verified |
| `--loader_seed N` | `cli.train` | the shuffle gets its own generator (and the global RNGs are re-seeded after the model is built), so the video order does not depend on what was constructed |
| `--expect_n_iter N` | `cli.train` | abort unless the iterative enhancement module has exactly N layers (0 = none). Guards the compat default of `--n_iter 2` |
| `--small_train_videos N` | `cli.train` | plumbing runs: N evenly spaced training videos |

Not behind a flag: `gen_frame_pre_label` and `gen_box_label` now accept the
annotation format in this repo's files (strings and flat boxes). Before, both
raised on the first frame (`KeyError: 's'`, `TypeError`), so the released step-4
training path could not run on this data at all.
