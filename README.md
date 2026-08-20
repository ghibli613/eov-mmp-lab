# Open-Vocabulary VidVRD (EOV-MMP)

End-to-end open-vocabulary video visual relationship detection: given a video,
detect object trajectories and the `<subject, predicate, object>` relations
between them — including object and predicate categories that were never seen
during training.

The method is **EOV-MMP** ("End-to-end Open-vocabulary Video Visual Relationship
Detection using Multi-modal Prompting", TPAMI 2025). This repository re-houses
that code in a structured layout with machine-independent paths.

> **Before you start:** step 4 (pretrained checkpoints) is currently **blocked** —
> the authors' download link is not publicly accessible. Steps 1–3 work and
> prepare everything else. See [§4](#4-pretrained-checkpoints--currently-blocked)
> and [`docs/known-issues.md`](docs/known-issues.md).

---

## What you need

| | |
|---|---|
| OS | Linux (developed on WSL2) |
| GPU | NVIDIA, **CUDA 12.x driver**. 4 GB works for development; **16 GB (e.g. Colab T4) recommended for training** |
| Disk | **~50 GB** — 4.3 GB videos, ~42 GB decoded frames, ~2 GB models. Peak is ~4 GB higher during download, before the zips are removed. |
| Time | ~1 h of downloads and preprocessing, mostly unattended |

Check your driver first — the CUDA version it reports caps what you can install:

```bash
nvidia-smi        # look at "CUDA Version" in the top-right
```

---

## Layout

```
🧠  models/        end-to-end model
      methods/         deformable-DETR detector, CLIP-distilled
      tracking/        deep_sort + AFLink trajectory association
      util/            boxes, misc, schedulers
🧊  vlm/           frozen backbones: CLIP, TagCLIP, InternVideo + prompt text encoder
📦  data_loading/  video dataset — frames are CLIP-encoded inside __getitem__
📊  inference/     post-processing, association, open-vocab relation evaluation
⚙️   ops/           MultiScaleDeformableAttention (CUDA, you compile it in step 2)
🔧  utils/         paths, arguments, feature helpers
▶️   cli/           entry points

💾  data/vidvrd/   videos · frames · anno · trajectories & class splits
📜  third_party/   detectron2 subset, VidVRD eval API, VidVRD-II helpers
🛠️   tools/         data preparation scripts
📚  docs/          knowledge base — architecture, data, known issues, research
📤  output/        ckpt/ and log/   (created on demand, gitignored)
```

**New here? Read [`docs/`](docs/README.md).** It covers how the model works, what
every data file is, and what is currently blocked.

**Every filesystem path is defined in [`utils/paths.py`](utils/paths.py)** and
derived from the repo root — nothing is hard-coded to a machine. At any point:

```bash
python -m utils.paths        # prints every path and whether it exists
```

Override for a machine with data on another disk:

```bash
export VIDVRD_DATA_ROOT=/mnt/big/vidvrd
export VIDVRD_OUTPUT_ROOT=/mnt/big/output
```

---

## 1. Environment

```bash
# conda-forge only: Anaconda's default channels are ToS-gated and will refuse
conda create -y -n eov --override-channels -c conda-forge python=3.12
conda activate eov

# PyTorch matching YOUR driver. cu128 works with a CUDA 12.8 driver.
# Check https://pytorch.org for the right index-url if yours differs.
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

pip install \
  transformers==5.15.1 tokenizers==0.22.2 huggingface_hub==1.28.0 \
  numpy==2.5.2 scipy==1.18.0 opencv-python-headless matplotlib \
  ftfy regex einops timm fvcore pycocotools scikit-learn tqdm pillow gdown
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Compile the CUDA operator

The detector uses Deformable DETR's `MultiScaleDeformableAttention`, which is a
CUDA extension you must build. This needs a **compiler and CUDA headers**, which
PyTorch does not ship:

```bash
conda install -y --override-channels -c conda-forge \
  cuda-nvcc cuda-cudart-dev libcusparse-dev libcublas-dev \
  libcusolver-dev libcurand-dev libcufft-dev cuda-version=12.8 \
  "gcc_linux-64=13" "gxx_linux-64=13"
```

GCC **13** specifically: CUDA 12.8 rejects GCC ≥ 14, and conda's default is newer.

```bash
cd ops
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export TORCH_CUDA_ARCH_LIST="7.5"     # 7.5 = GTX 1650 / T4; 8.6 = A100/3090
python setup.py build install
cd ..
```

Verify — **import torch first**, or you get `libc10.so: cannot open shared object file`:

```bash
python -c "import torch, MultiScaleDeformableAttention; print('ok')"
```

## 3. Get the data — one command

Everything except the pretrained checkpoints is fetched or computed by a single
script:

```bash
python tools/prepare_data.py
```

It runs six steps in order and **skips any that are already complete**, so if it
is interrupted (or your connection drops mid-download) just run it again.

| Step | What it does | Size | Time |
|---|---|---|---|
| `videos` | 1000 `.mp4` from HuggingFace `shangxd/imagenet-vidvrd` (the source zips are deleted once extraction is verified) | 4.3 GB | ~15 min |
| `anno` | 800 train + 200 test annotation JSON, same dataset | 33 MB | seconds |
| `meta` | trajectories, base/novel splits, vocabularies, predicate prior — from the [MMP repo](https://github.com/wangyongqi558/MMP_OV_VidVRD) that EOV builds on | 12 MB | seconds |
| `frames` | decodes the videos to JPEG (the dataset reads frames, not video) | ~42 GB | ~30 min |
| `gt` | evaluation ground truth, derived from the annotations | small | ~1 min |
| `bank` | per-category CLIP image-embedding bank the detector conditions on | small | ~10 min, needs GPU |

See where you stand at any time — this only reads, never writes:

```bash
python tools/prepare_data.py --check
```

```
  step      have / need
  OK videos    1000 / 1000
  OK anno      1000 / 1000
  OK meta        11 / 11
  OK frames    1000 / 1000
  OK gt           2 / 2
  OK bank         1 / 1
```

Run one step on its own, or force a redo:

```bash
python tools/prepare_data.py --steps frames,bank
python tools/prepare_data.py --steps bank --force
```

The script exits non-zero and prints `INCOMPLETE: ...` if anything is missing, so
it is safe to use in a setup pipeline.

### Two things to know about the data

**Frames are 1-indexed.** `000001.jpg` is trajectory frame 0, because
`models/gen_labels.py` addresses them that way. Do not "fix" this to 0-indexed.
To re-check the extraction against the annotations:

```bash
python tools/extract_frames.py --verify     # expect 0 problems
```

**The CLIP object bank is a reconstruction.**

> ⚠️ The authors never published their `clip_L14_feat_vidvrd.pkl`. Everything fed
> into `tools/build_clip_object_bank.py` is real — real GT boxes, real frames, the
> real CLIP encoder — but their exact recipe (exemplars per class, which frames,
> crop padding) is unknown, so **results built on it are not directly comparable
> to the paper**. Every choice made is recorded in the tool's docstring.

Once the script reports all six steps `OK`, confirm the paths resolve:

```bash
python -m utils.paths
```

## 4. Pretrained checkpoints — **currently blocked**

`prepare_data.py` deliberately does **not** cover these. Training starts from four
pretrained components, which must sit in `output/ckpt/`:

```
checkpoint_vidvrd0059_new_1e-5.pth                                   object detector
baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth
vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth                object classifier
<AFLink model>                                                       pass via --path_AFLink
```

The authors' Google Drive folder returns **401** to automated download — it is not
shared as "anyone with the link". Options:

1. Open the Drive link from their README in a signed-in browser
2. Their Baidu link (in the same README)
3. Email the author — their README invites it

**Nothing can train or evaluate until these exist.**

---

## Running training

```bash
cd cli
bash run_train.sh          # edit the variables at the top first
```

or directly:

```bash
python -m cli.train \
  --dataset vidvrd --batch_size 1 --max_epoch 20 --lr 1e-5 \
  --clip_len 30 --train_traj gt --ptm_mode vision_text \
  --src_split base --tgt_split all \
  --rel_feat True --mot_feat True --clip_feat True \
  --path_AFLink output/ckpt/<aflink>.pth
```

- `--src_split base` — train on base categories only (the open-vocabulary setting)
- `--tgt_split all` — evaluate on all categories
- Logs go to `output/log/`, checkpoints to `output/ckpt/`
- CLIP ViT-L/14@336px (~900 MB) downloads automatically on first run

## Running evaluation

Evaluate a trained checkpoint without training:

```bash
python -m cli.train --eval_only --ckpt_path output/ckpt/<your>.pth \
  --dataset vidvrd --path_AFLink output/ckpt/<aflink>.pth
```

It reports mAP and Recall@50/100 for both the `all` and `novel` predicate splits
and exits. **`--eval_only` does not need the three step-1/2/3 checkpoints** — the
trained end-to-end checkpoint supersedes them; it does still need the AFLink
tracker, since trajectory association is part of inference.

Note `--eval` is a *different*, pre-existing flag belonging to the detector, and
it defaults to `True`. Use `--eval_only`.

`train.py` also evaluates at the end of every epoch and keeps the best-mAP
checkpoint, so model selection uses the test set.

**Model selection uses the test set.** This is the convention across this
benchmark (RePro and MMP do the same), so following it keeps your numbers
comparable — but state it in any write-up.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `libc10.so: cannot open shared object file` | `import torch` before `MultiScaleDeformableAttention` |
| `No module named 'MultiScaleDeformableAttention'` | step 2 not done, or done in a different env |
| `prepare_data.py` stops on a download | just run it again — completed steps are skipped |
| g++ "greater than the maximum required version" | GCC ≥ 14 with CUDA 12.8 — install `gxx_linux-64=13` |
| `cuda_runtime_api.h` / `cusparse.h` not found | missing `CPATH`, or the `cuda-*-dev` packages |
| `FileNotFoundError: .../clip_L14_feat_vidvrd.pkl` | `prepare_data.py` never got to the `bank` step |
| `CUDA out of memory` | 4 GB is not enough for training; use a T4/A100. Batch size is already 1 |
| conda `CondaToSNonInteractiveError` | add `--override-channels -c conda-forge` |

---

## Credit

Method and original code: **EOV-MMP**, Wang et al., TPAMI 2025
([repo](https://github.com/wangyongqi558/EOV-MMP-VidVRD)). Trajectory data and class splits from **MMP**, Yang et al., AAAI 2024
([repo](https://github.com/wangyongqi558/MMP_OV_VidVRD), same authors). Dataset: **ImageNet-VidVRD**,
Shang et al. This repository restructures their code; all method credit is theirs.

[`docs/`](docs/README.md) is the knowledge base:
[architecture](docs/architecture.md) ·
[data](docs/data.md) ·
[known issues](docs/known-issues.md) ·
[port status](docs/port-status.md) ·
[landscape](docs/landscape.md) ·
[research ideas](docs/research-ideas.md)
