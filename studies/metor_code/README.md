# METOR — a reconstruction

An implementation of

> Yongqi Wang, Xinxiao Wu, Shuo Yang.
> **METOR: A Unified Framework for Mutual Enhancement of Objects and
> Relationships in Open-vocabulary Video Visual Relationship Detection.**
> IJCAI 2025. [arXiv:2505.06663](https://arxiv.org/abs/2505.06663)

built from the paper, because [the authors'
repository](https://github.com/wangyongqi558/METOR) has never released code.

**Documentation: [`docs/`](docs/README.md)** — 19 documents, indexed by what you
are trying to do.

| If you want to… | Read |
|---|---|
| understand what this is | [00_Start-here](docs/00_Start-here.md) |
| follow the method | [10_Pipeline-overview](docs/10_Pipeline-overview.md) → [11](docs/11_Contextual-refinement-encoding.md) → [13](docs/13_Iterative-enhancement.md) → [15](docs/15_Training-objective.md) |
| find an equation or class | [20_Equation-index](docs/20_Equation-index.md) |
| know what a folder is for | [21_Repository-layout](docs/21_Repository-layout.md) |
| judge whether to trust a number | [33_Verification-status](docs/33_Verification-status.md) |
| run or train it | [40_Hardware](docs/40_Hardware.md) |
| change something | [42_Extending](docs/42_Extending.md) |

---

## What this is, in one screen

METOR's argument is that open-vocabulary VidVRD methods run *objects → relations*
as a cascade, so object errors propagate. It replaces that with one model where
the two refine each other:

```
Eq.1  (H_CLS, H_patch, T_o, T_r, Q̂, C_o, C_r) = Φ(V, Q)    contextual refinement
Eq.2   O                                       = Decoder(Q̂, H_patch)
Eq.3  (Ô_s, Ô_o, R̂)                            = Ψ(O, H_CLS)  iterative enhancement
Eq.4   S_s, S_o, S_r                           = σ(cos(·, T))
Eq.5   τ_s, τ_o                                = M_b(Ô)
```

Two contributions:

- **Contextual refinement encoding (Sec. 3.3)** — learnable object and relationship
  context tokens are appended to the frozen CLIP ViT's input; what comes out
  conditions *both* the object queries and the class text embeddings on this
  video. → [`metor/modules/contextual_encoding.py`](metor/modules/contextual_encoding.py)
- **Iterative enhancement (Sec. 3.4)** — objects produce a relationship
  representation, which is mixed back into the objects, `N_i` times.
  → [`metor/modules/iterative_enhancement.py`](metor/modules/iterative_enhancement.py)

---

## Two architectures, and they are not interchangeable

| | `--arch metor` | `--arch metor-compat` |
|---|---|---|
| What it is | the paper as written | METOR's modules on the EOV-MMP stack |
| Object queries | 100 (paper) | 300 (EOV's) |
| Trajectories | regressed, Eq. 5 | deep_sort + AFLink |
| Losses | 5 (Eq. 12) | EOV's |
| Loads released checkpoints | **no** — shapes differ | **yes** |
| Runs on a 16 GB GPU today | no (see below) | yes |
| Can reproduce Table 1 | in principle | **no** |

`metor-compat` exists because the faithful path has a hard prerequisite that is
not met: the paper initialises its decoder from a COCO-pretrained detector, and
that init is **not wired up here**
([reconstruction notes §4.1](docs/32_Not-implemented.md)).
EOV-MMP never released detector training code either, so there is no shortcut.

What `metor-compat` gives you instead is a **controlled measurement**: hold the
entire EOV stack fixed, switch only `--n_iter`, and the delta is attributable to
the module. `--n_iter 0` is bit-identical to the EOV baseline — pinned by
[`tests/test_compat_equivalence.py`](tests/test_compat_equivalence.py). That is
better evidence than the paper's own +1.70, which changes the module, the query
count, the epoch budget and the loss count simultaneously
([§7](docs/33_Verification-status.md)).

---

## Layout

```
metor/
  model.py                     Eq. 1-5, the unified framework
  compat.py                    METOR's modules on the EOV stack
  matcher.py                   query <-> GT trajectory assignment (unstated in the paper)
  modules/
    contextual_encoding.py     Eq. 6-10   CRE
    iterative_enhancement.py   Eq. 11     IEM
    criterion.py               Eq. 12-16  the five losses
    heads.py                   Eq. 4-5    scoring, trajectory regression
data_loading/
  frame_dataset.py             pixels + trajectory targets (--arch metor)
  feature_dataset.py           EOV's CLIP-feature loader     (--arch metor-compat)
cli/
  train.py                     ONE entry point, dispatches on --arch
  evaluate.py                  evaluation (--arch metor-compat today)
  config.py                    YAML layering
configs/
  metor_vidvrd.yaml            ONLY values the paper states
  metor_vidor.yaml             ditto, VidOR
  metor_defaults.yaml          ONLY values this reconstruction invented
docs/
  30_Paper-gaps.md   read this first: paper's values vs ours
  10_Pipeline-overview.md           where every component and equation lives
  10_Hardware.md               what runs where
```

New here? [`docs/README.md`](docs/README.md) is the navigation
document — it maps every module and every equation to a file and line, explains
what each top-level folder is for, and follows the data through the pipeline in
execution order.

The split between `metor_vidvrd.yaml` and `metor_defaults.yaml` is load-bearing:
it keeps the paper's numbers visibly separate from ours.
`tests/test_metor_config.py` fails if they mix.

Infrastructure (`vlm/`, `models/`, `inference/`, `third_party/`, `utils/`,
`ops/`, `tools/`) is vendored from `../ov-vidvrd-lab`, a cleaned refactor of
EOV-MMP by the same lineage. `data/` and `output/ckpt/` are **symlinks** into
that workspace, so the 47 GB of VidVRD frames and the ~9 GB of checkpoints are
not duplicated.

---

## Setup

`data/` and `output/` are **not in the repository** — they hold 42 GB of VidVRD
frames and ~9 GB of checkpoints. Set them up per
[docs/22_Data-pipeline](docs/22_Data-pipeline.md), or point the environment at an
existing copy:

```bash
export VIDVRD_DATA_ROOT=/path/to/vidvrd
export VIDVRD_OUTPUT_ROOT=/path/to/output
```

Then:

```bash
conda activate repro-next          # torch 2.11.0+cu128
pip install -r requirements.txt
python -m utils.paths              # check every path resolves
python -m pytest tests/            # 137 tests, no GPU or data required
```

The test suite runs without the data (dataset tests skip themselves).

## Running

```bash
# tests — safe anywhere, ~65s, no GPU required
python -m pytest tests/

# the paper's recipe (needs a 24GB GPU; see docs/40_Hardware.md)
python -m cli.train \
    --config configs/metor_defaults.yaml \
    --config configs/metor_vidvrd.yaml

# the controlled ablation, on the EOV stack (needs 16GB)
python -m cli.evaluate --arch metor-compat --n_iter 0 \
    --ckpt_path output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth \
    --path_AFLink output/ckpt/AFLink_epoch20.pth
python -m cli.evaluate --arch metor-compat --n_iter 2 ...   # the delta
```

Config precedence: argparse defaults < `--config` files in order < flags typed.

### Ablation flags

Mapped to the paper's own tables:

| Flag | Paper |
|---|---|
| `--no_crq` | Table 3, "w/o CRQ" |
| `--no_crt` | Table 3, "w/o CRT" |
| `--no_crq --no_crt` | Table 3, "w/o CRE" |
| `--n_iter {0,1,2,3}` | Table 4 |

---

## ⚠ This machine cannot train it

7 GB of RAM and a 4.3 GB GPU. Constructing the real CLIP ViT-L/14@336px and
running a forward/backward **exhausts memory and kills the WSL session**. This
was tried; it is not a bug in the code, the model simply does not fit.

The test suite is built around that: it uses
[`tests/stub_clip.py`](tests/stub_clip.py), a 60k-parameter CLIP-shaped
stand-in, so the whole framework is verifiable here in seconds.
Details in [`docs/40_Hardware.md`](docs/40_Hardware.md).

---

## Status

Verified (run, not reasoned about):

- 137 tests pass
- the framework does a full forward+backward with the shapes Sec. 3 specifies
- gradients reach `c_o` / `c_r` **through the frozen ViT** — the load-bearing
  property of Eq. 7, and the one thing that can silently fail while everything
  still appears to train
- Eq. 11's algebra holds in closed form (`α^N_i` scaling)
- Eq. 12's weighting recomputes by hand
- the dataset reads real VidVRD annotations on the correct frame grid
- compat mode with `--n_iter 0` is bit-identical to EOV

Not verified, and not claimed:

- **that it converges, or scores anything in particular.** Nothing here has been
  trained. Every number in the paper's Table 1 remains unreproduced.
