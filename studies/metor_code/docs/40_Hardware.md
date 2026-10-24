# What runs on what

Measured on this workspace, not estimated.

## This machine

| | |
|---|---|
| GPU | ~4.3 GB usable, under WSL2 |
| System RAM | **7 GB total**, ~5 GB available |
| Python | `conda activate repro-next` (torch 2.11.0+cu128, CUDA available) |

### What works here

```bash
python -m pytest tests/                 # 130 tests, ~65s, no GPU needed
python -m pytest tests/test_metor_model.py   # full framework, stub CLIP, ~5s
```

Everything in `tests/` is designed to run on this machine. The model tests use
`tests/stub_clip.py`, a 60k-parameter CLIP-shaped stand-in, precisely so the
shape algebra of Sec. 3.2–3.3 is checkable without the real backbone.

### What does NOT work here

**Do not construct the real CLIP ViT-L/14@336px and run a forward/backward on
this machine.** It was tried. CLIP ViT-L is ~400M parameters (~1.7 GB in fp32),
and `--cre_mode inject` runs it *inside* the training step over 577+8 tokens per
frame with activations retained for backward. On 7 GB of RAM this exhausts
memory and takes down the WSL session, requiring a reconnect.

This is not a bug in the code. It is a model that does not fit.

Concretely, avoid on this machine:

```bash
python -m cli.train --config configs/metor_vidvrd.yaml   # will OOM
python -m cli.evaluate ...                                     # will OOM
```

The `--cre_mode pool` path is much cheaper (it consumes cached patch features
instead of running the ViT), but the detector and relation stack still exceed
what is available. `../ov-vidvrd-lab/docs/10_Known-issues.md` §2 measured the
EOV datasets alone at 5.32 GB before any model was built.

## What you need to train

| Path | Minimum | Notes |
|---|---|---|
| `--arch metor-compat`, eval only | 16 GB GPU | Loads the released EOV checkpoints; this is the cheap way to a real measured delta. |
| `--arch metor-compat`, fine-tune | 16 GB GPU | EOV's step-4 budget: 5 epochs, batch 1. |
| `--arch metor`, full | 24 GB+ GPU, days | 30 epochs end-to-end from COCO init, batch 1, ~800 videos. And §4.1 of the reconstruction notes has to be done first. |

The paper reports "a single NVIDIA GeForce RTX 4090" (24 GB) for its own
experiments.

## Cost-saving that is not a hack

`--frame_stride 30` is the paper's own setting ("key frames are sampled every 30
video frames", Sec. 4.2) and is the default in `configs/metor_vidvrd.yaml`. It
reduces CLIP forwards by roughly 30×. `METORDataset` implements it on the same
1-indexed grid the EOV loader uses, so the two remain comparable — a half-frame
offset between them would invalidate any A/B.
