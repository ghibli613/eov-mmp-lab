# ov-vidvrd-lab

A research codebase for **open-vocabulary video visual relationship detection**:
given a video, detect object trajectories and the `<subject, predicate, object>`
relations between them — **including object and predicate categories never seen
during training**.

This is a working repository built for extension, not a reproduction package.
Its current baseline is **EOV-MMP** (Wang et al., TPAMI 2025), restructured here
into a layout that can be modified: machine-independent paths, separated train
and evaluate entry points, automated data preparation, and a
[knowledge base](docs/README.md) recording how every part works and where it
diverges from upstream.

The baseline is the starting point. Everything is arranged so components can be
swapped — the detector, the prompting scheme, the relation head — without
untangling the rest.

> **You need ~16 GB of VRAM.** The model peaks at **9.26 GB before a single
> backward pass**, so 4–8 GB cards cannot run it. A 16 GB card (RTX 4080, A4000,
> Colab T4) or better is the requirement. Setup instructions for
> [your own machine](#setup-on-your-own-machine) and for
> [Colab](#setup-on-google-colab) are both below — pick one.

---

## Contents

**Setup — pick one:**

- [Setup on your own machine](#setup-on-your-own-machine) — persistent disk, no
  session limits. Best if you have a 16 GB+ GPU.
- [Setup on Google Colab](#setup-on-google-colab) — no hardware needed, but the
  disk is ephemeral and needs planning.

**Then, common to both:**

- [Get the data](#get-the-data) · [Get the weights](#get-the-weights)
- [Running training](#running-training) · [Running evaluation](#running-evaluation)
- [Troubleshooting](#troubleshooting) · [Credit](#credit-and-provenance)

New to the project? [`docs/README.md`](docs/README.md) explains how the model
works, what every file is, and what is still open.

---

## Layout

```
🧠  models/        end-to-end model
      methods/         deformable-DETR detector, CLIP-distilled
      tracking/        deep_sort + AFLink trajectory association
      util/            boxes, misc
      object_classifier.py    what each detection is
      relation_classifier.py  what relation holds between a pair
📦  data_loading/  video dataset — frames are CLIP-encoded inside __getitem__
🧊  vlm/           frozen backbones: CLIP, TagCLIP, InternVideo
📊  inference/     post-processing, association, open-vocab evaluation
⚙️   ops/           MultiScaleDeformableAttention (CUDA, compiled in step 3)
🔧  utils/         paths, argument parsing, feature helpers
▶️   cli/           entry points — train.py, evaluate.py, common.py
🛠️   tools/         data preparation and weight download
🧪  tests/         55 tests; `python -m pytest tests/ -q`
📚  docs/          knowledge base — start at docs/02_Code-walkthrough.md
💾  data/vidvrd/   videos · frames · anno · trajectories & class splits
📤  output/        ckpt/ and log/   (gitignored)
```

**Every filesystem path is defined in [`utils/paths.py`](utils/paths.py)** and
derived from the repo root, so the project moves between machines cleanly:

```bash
python -m utils.paths        # print every path and whether it exists
```

Two environment variables relocate everything — essential on Colab:

```bash
export VIDVRD_DATA_ROOT=/content/data/vidvrd
export VIDVRD_OUTPUT_ROOT=/content/drive/MyDrive/vidvrd/output
```

---

# Setup on your own machine

Six steps. Linux, or Windows via WSL2.

## 1. Check your hardware

```bash
nvidia-smi
```

Three things in that output matter:

| | |
|---|---|
| **Memory** | needs ≥ 16 GB. Below that, training will OOM |
| **CUDA Version** (top-right) | the *driver's* ceiling — it caps which PyTorch build you can install |
| **Driver Version** | under WSL2 this comes from the **Windows** driver; installing a Linux toolkit will not change it |

You also need **~50 GB of disk**: 4.3 GB videos, ~42 GB decoded frames, ~2 GB
models, plus 7.4 GB if you keep the pretrained weights.

## 2. Create an environment

Python 3.10–3.12. With conda:

```bash
# conda-forge only: Anaconda's default channels are ToS-gated and will refuse
conda create -y -n ovvidvrd --override-channels -c conda-forge python=3.12
conda activate ovvidvrd
```

Or with plain venv, if you would rather not use conda:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

> conda is **strongly** recommended: step 4 needs a CUDA compiler and headers,
> and conda is by far the easiest way to get a matched, isolated toolchain. With
> venv you must install the CUDA toolkit system-wide yourself.

## 3. Install PyTorch, then everything else

PyTorch must match your driver. Take the `CUDA Version` from step 1 and pick the
**highest index at or below it**:

| Your driver reports | Use |
|---|---|
| 12.6 – 12.7 | `cu126` |
| 12.8 | `cu128` |
| 12.9 | `cu129` |
| 13.0 or newer | `cu130` |

Newer drivers run older CUDA runtimes, so under-shooting is safe;
over-shooting fails at import.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install \
  transformers tokenizers huggingface_hub \
  numpy scipy opencv-python-headless matplotlib \
  ftfy regex einops timm fvcore pycocotools scikit-learn tqdm pillow gdown
```

Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda,
           torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `is_available()` is `False`, stop and fix that — nothing downstream works
without it.

## 4. Install a CUDA toolchain

The detector uses Deformable DETR's `MultiScaleDeformableAttention`, a CUDA
extension compiled from source. PyTorch ships a CUDA *runtime* but not a
*compiler*, so you need `nvcc` and headers. (Colab has these already; a normal
machine does not.)

```bash
conda install -y --override-channels -c conda-forge \
  cuda-nvcc cuda-cudart-dev libcusparse-dev libcublas-dev \
  libcusolver-dev libcurand-dev libcufft-dev cuda-version=12.8 \
  "gcc_linux-64=13" "gxx_linux-64=13"
```

Two details that cause most build failures:

- **`cuda-version` pins the toolkit**, which is independent of your driver. It
  must be ≤ what the driver supports, and should match your PyTorch wheel from
  step 3. If you installed `cu130`, use `cuda-version=13.0`.
- **GCC 13 specifically.** CUDA 12.x rejects GCC ≥ 14, and conda's default is
  newer. This is the single most common build error.

## 5. Compile the operator

Find your GPU's compute capability:

```bash
python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))"
```

| GPU | value |
|---|---|
| GTX 16xx, RTX 20xx, T4 | `7.5` |
| A100 | `8.0` |
| RTX 30xx, A10, A40 | `8.6` |
| RTX 40xx, L4, L40 | `8.9` |
| H100 | `9.0` |
| RTX 50xx | `12.0` |

```bash
cd ops
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export TORCH_CUDA_ARCH_LIST="8.6"        # ← from the table above
pip install .                            # not -e, and not `setup.py install`
                                         # (removed in setuptools 80); -e leaves the .so
                                         # in the build tree and the import then fails
cd ..
```

Verify — **import torch first**, or you get
`libc10.so: cannot open shared object file`:

```bash
python -c "import torch, MultiScaleDeformableAttention; print('ok')"
```

## 6. Data and weights

Nothing machine-specific here:

```bash
python tools/prepare_data.py          # ~45 GB, 30-40 min, resumable
python tools/prepare_data.py --check  # status, writes nothing
```

Then [get the weights](#get-the-weights).

That is the whole local setup. Everything persists, so you do this once.

---

# Setup on Google Colab

Use this if you do not have a 16 GB GPU. The trade-off is that Colab's local
disk is **ephemeral** — plan the layout before you start.

## 1. Get a GPU session

Runtime → Change runtime type → **T4 GPU** (or better), then `!nvidia-smi` to
confirm ≥ 16 GB.

## 2. Mount Drive and clone

Drive holds what is slow or impossible to re-fetch; the session disk holds the
big derived data, rebuilt each time.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/<you>/ov-vidvrd-lab.git
%cd ov-vidvrd-lab
```

```python
import os
os.environ['VIDVRD_DATA_ROOT']   = '/content/data/vidvrd'
os.environ['VIDVRD_OUTPUT_ROOT'] = '/content/drive/MyDrive/vidvrd/output'
```

**Outputs go to Drive on purpose** — a disconnected session must not cost you a
training run.

## 3. Install dependencies

**Do not install PyTorch.** Colab ships a build matched to its own driver;
replacing it is slow and frequently breaks CUDA.

```bash
!pip install -q ftfy regex einops timm fvcore pycocotools opencv-python-headless gdown
```

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
```

## 4. Compile the operator

Colab already has `nvcc` matching its torch, so no toolchain install is needed:

```bash
%cd ops
!rm -rf build *.egg-info
!TORCH_CUDA_ARCH_LIST="7.5" pip install .
%cd ..
```

`setup.py install` was removed in setuptools 80 (2025); use pip.

Use `7.5` for T4, `8.0` for A100, `8.9` for L4 — or query it as in the
[local instructions](#5-compile-the-operator).

```python
import torch, MultiScaleDeformableAttention
print('ok')
```

## 5. Stage the videos on Drive (first time only)

The 42 GB of frames cannot live on Drive; the 4.3 GB of videos they derive from
can. Download once, keep forever:

```bash
!python tools/prepare_data.py --steps videos
!mkdir -p /content/drive/MyDrive/vidvrd
!cp -r $VIDVRD_DATA_ROOT/videos /content/drive/MyDrive/vidvrd/
```

Every later session symlinks instead of downloading:

```bash
!mkdir -p $VIDVRD_DATA_ROOT
!ln -sfn /content/drive/MyDrive/vidvrd/videos $VIDVRD_DATA_ROOT/videos
```

Keep the weights on Drive the same way, and symlink `output/ckpt` at them.

## 6. Build the rest of the data, each session

```bash
!df -h /content              # confirm room for ~45 GB
!python tools/prepare_data.py
```

With the videos symlinked it skips straight to decoding frames — 30–40 minutes.

---

# Get the data

One command, on either environment:

```bash
python tools/prepare_data.py           # everything not already done
python tools/prepare_data.py --check   # status, writes nothing
```

Six steps, each skipped when already complete, so an interrupted run just needs
re-running.

| Step | Produces | Size |
|---|---|---|
| `videos` | 1000 `.mp4` from HuggingFace | 4.3 GB |
| `anno` | 800 + 200 annotation JSON | 33 MB |
| `meta` | 11 trajectory/vocabulary files from the MMP repo | 41 MB |
| `frames` | 296,204 JPEGs decoded locally | **~42 GB** |
| `gt` | 2 evaluation GT files, derived | 82 MB |
| `bank` | CLIP object bank | *(use the authors' — see below)* |

Full detail in [`docs/03_Data.md`](docs/03_Data.md). Two things are easy to get wrong:

**Frames are 1-indexed.** `000001.jpg` is trajectory frame 0, because
[`models/gen_labels.py`](models/gen_labels.py) maps ids as `'%06d.jpg' % (fid+1)`.
Do not "fix" this. To re-check an extraction:

```bash
python tools/extract_frames.py --verify     # expect 0 problems
```

**The CLIP object bank must come from the authors.**
`tools/build_clip_object_bank.py` can rebuild an approximation, but theirs has
189,345 exemplars in float16 and unnormalised, against ~1,500 normalised float32
in the reconstruction. Numbers from the reconstruction are **not comparable to
the paper**.

---

# Running training

```bash
python -m cli.train
```

That is the whole command. **The argument defaults are the paper's VidVRD
configuration** — `lr 1e-5`, `src_split base`, `ptm_mode vision_text`, 20 epochs,
`rel`/`mot`/`clip` features on — so a bare invocation reproduces the intended
experiment. Override what you need:

```bash
python -m cli.train --max_epoch 5 --lr 5e-6
```

- `--src_split base` trains on seen categories only. **This is the
  open-vocabulary setting.** Setting it to `all` lets the model see novel
  categories and makes the novel-split score meaningless.
- `--tgt_split all` evaluates on all categories.
- Logs go to `$VIDVRD_OUTPUT_ROOT/log/`, checkpoints to `ckpt/`.
- CLIP ViT-L/14@336px (~900 MB) downloads automatically on first run.

Training starts from the three step-1/2/3 checkpoints — this is **fine-tuning,
not training from scratch**. See [`docs/01_Architecture.md`](docs/01_Architecture.md).

**Measure one epoch before committing to twenty.** Colab sessions are
time-limited, and an epoch is 800 videos × ~260 frames × 2 CLIP-L encoders.
[`docs/10_Known-issues.md`](docs/10_Known-issues.md) documents a commented-out
sampling branch that can cut this by ~30×.

Model selection uses the test set — the convention across this benchmark
(RePro and MMP do the same), but state it in any write-up.

---

# Running evaluation

```bash
python -m cli.evaluate --ckpt_path output/ckpt/<trained>.pth
```

Reports mAP and Recall@50/100 for both the `all` and `novel` predicate splits.

**Evaluation is markedly cheaper than training.** It builds only the test
dataset and does not load the three step-1/2/3 checkpoints — a trained
end-to-end checkpoint already contains those weights. It still needs the AFLink
tracker; trajectory association is part of inference.

(The two CLIP ViT-L/14@336px encoders are now loaded once per process and shared
between the train and val datasets, so building one dataset instead of two no
longer changes that part of the footprint. See docs/10_Known-issues.md §2.)

`--frame_stride 30` applies the paper's sampling and is roughly 30× cheaper;
the default of 1 reproduces upstream. The two do not give the same numbers —
read docs/10_Known-issues.md §3 before choosing.

Any flag changed for training must be repeated here. The feature flags
(`--clip_feat`, `--mot_feat`, `--rel_feat`, `--bbox_feat`) gate which evidence
the relation classifier receives; a disabled stream is fed as zeros, so tensor
shapes are unchanged but the scores are not.

---

# Get the weights

Training starts from four pretrained components, in `output/ckpt/`:

```
checkpoint_vidvrd0059_new_1e-5.pth                              step 1: detector
vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth           step 2: object classifier
baseline_fbce_..._stage2_new_L14_e2e.pth                        step 3: relation classifier
AFLink_epoch20.pth                                              tracker
```

These are the **outputs** of the paper's first three training steps. The code
that produces them was never released, so they cannot be regenerated here — see
[`docs/01_Architecture.md`](docs/01_Architecture.md).

Also required, supplied alongside them:

```
data/vidvrd/data/clip_L14_feat_vidvrd.pkl        278 MB  CLIP object bank
data/vidvrd/data/VidVRD_ECC_{train,test}.json     42 MB  camera-motion matrices
```

**To obtain them:** email the author, whose README invites it —
`1285441164yq@gmail.com`. Their Google Drive folder returns 401 to automated
download; a Baidu link is in their README. **Ask for the trained end-to-end
checkpoints too** — they let you evaluate without training at all.

## Hosting them yourself

Once you have them, put them somewhere you control so every machine — and every
Colab session — can fetch them in one command. A **private** HuggingFace repo is
the simplest option:

```bash
hf auth login          # "hf", not the retired huggingface-cli
python tools/hugging_upload.py --repo <you>/ov-vidvrd-weights \
  --github <you>/ov-vidvrd-lab
```

That hashes every file, writes `MANIFEST.json` and a card, creates the repo
**private by default**, and uploads with `upload_large_folder` — chunked and
resumable, which matters when the detector checkpoint alone is 3.5 GB. It prints
the fetch command when it finishes. Add `--dry-run` to build the manifest
without uploading.

> Keep it private unless the EOV-MMP author agrees otherwise. Their own Drive
> folder is not publicly shared, which is why automated download returns 401.
> Private costs you nothing: `hugging_download.py` reads a token from `HF_TOKEN`
> or the `hf auth login` cache.

The same tool uploads anything else — preprocessed frames, cached features — by
saying where the files should land in a clone:

```bash
python tools/hugging_upload.py --repo <you>/ov-vidvrd-frames --repo-type dataset \
  --bundle ../_frames_bundle --dest data/vidvrd/frames --title "VidVRD frames"
```

## Fetching them

```bash
MANIFEST=https://huggingface.co/ghibli613/ov-vidvrd-weights/resolve/main/MANIFEST.json

python tools/prepare_data.py --manifest $MANIFEST          # data + weights, one command
python tools/hugging_download.py --manifest $MANIFEST      # weights only
python tools/hugging_download.py --manifest $MANIFEST --only eval   # 2.9 GB not 7.4
```

That repo is **private**, so run `hf auth login` first (or set `HF_TOKEN`) on any
machine that needs it — including each Colab session.

Every file is sha256-verified, so a truncated download fails loudly instead of
becoming an unexplained accuracy drop. Correct files are skipped, so it resumes.

The filenames above are defaults. Override rather than renaming your copies:

```bash
--detector_ckpt  --obj_classifier_ckpt  --relation_ckpt  --ckpt_path  --path_AFLink
```

If the ECC files are absent the code still runs, prints a warning, and disables
camera compensation — a real quality loss on moving-camera video, not an
equivalence.

---

# Troubleshooting

| Symptom | Cause |
|---|---|
| `libc10.so: cannot open shared object file` | `import torch` before `MultiScaleDeformableAttention` |
| `No module named 'MultiScaleDeformableAttention'` | the extension was not compiled, or was compiled in another environment / session |
| `CUDA error: no kernel image is available` | `TORCH_CUDA_ARCH_LIST` does not match your GPU — check `torch.cuda.get_device_capability()` |
| g++ "greater than the maximum required version" | GCC ≥ 14 with CUDA 12.8 — install `gxx_linux-64=13` |
| `cuda_runtime_api.h` / `cusparse.h` not found | missing `CPATH`, or the `cuda-*-dev` packages (local only) |
| `Missing required file(s)` at startup | weights not fetched; see [Get the weights](#get-the-weights) |
| `FileNotFoundError: .../clip_L14_feat_vidvrd.pkl` | the CLIP object bank is missing |
| `CUDA out of memory` | < 16 GB of VRAM. Batch size is already 1 |
| `prepare_data.py` stops mid-download | just run it again — completed steps are skipped |
| Everything vanished between sessions | Colab's local disk is ephemeral; only Drive persists |
| conda `CondaToSNonInteractiveError` | add `--override-channels -c conda-forge` |
| `torch.cuda.is_available()` is `False` | the PyTorch build does not match the driver — reinstall from the index for your `nvidia-smi` CUDA version |
| `nvcc: command not found` | the CUDA toolchain step was skipped, or `CUDA_HOME`/`PATH` are not exported in this shell |
| `undefined symbol` importing the extension | it was built against a different PyTorch; delete `ops/build/` and recompile |
| driver upgraded, extension now fails | recompile is *not* needed — newer drivers run older runtimes. Check `torch.cuda.is_available()` first |

---

# Credit and provenance

The baseline method and its original implementation are **not mine**:

| | |
|---|---|
| Method and original code | **EOV-MMP**, Wang, Wu, Yang & Luo, TPAMI 2025 — [repo](https://github.com/wangyongqi558/EOV-MMP-VidVRD), [paper](https://arxiv.org/abs/2409.12499) |
| Trajectory data, class splits | **MMP**, Yang, Wang, Ji & Wu, AAAI 2024 — [repo](https://github.com/wangyongqi558/MMP_OV_VidVRD) (same group) |
| Dataset | **ImageNet-VidVRD**, Shang et al. |
| Pretrained weights | supplied directly by the EOV-MMP author |

What this repository adds is engineering, not method: the restructuring, the
automated data pipeline, the separated entry points, the correctness fixes
recorded in [`docs/11_Port-status.md`](docs/11_Port-status.md), and the analysis in
[`docs/`](docs/README.md).

**If you publish from this**, cite EOV-MMP and MMP for anything inherited, and
be explicit about what you changed. [`docs/10_Known-issues.md`](docs/10_Known-issues.md)
lists the things a reviewer would reasonably ask about — including where this
code and the EOV-MMP paper disagree.
