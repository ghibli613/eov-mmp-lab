# Port status — provenance and divergence from upstream

Part of the [knowledge base](README.md). This file records where the code came
from and **every way it differs from upstream EOV-MMP**. Current blockers live
in [10_Known-issues.md](10_Known-issues.md); the method itself is in
[01_Architecture.md](01_Architecture.md).

Started 2026-08-20. This repository began as a refactor of **RePro** (ICLR 2023);
RePro's method code was then replaced with **EOV-MMP**'s, keeping the structure.

The RePro history is **not** in this repository — it was deliberately not
published, since the RePro path cannot run here anyway (see §5.5). What follows
records where the EOV code came from and how it was changed.

---

## 1. Where the code came from

| Structure slot | Filled with (from `EOV-MMP-VidVRD/`) |
|---|---|
| `vlm/backbones/clip/`, `clip_tagclip/`, `internvideo/` | `model/clip`, `model/clip_tagclip`, `model/InternVideo` |
| `vlm/text_encoder.py`, `vlm/ptm_encoder.py` | `model/text_encoder.py`, `utils/ptm_encoder.py` |
| `models/methods/` | `model/methods/` — deformable-DETR detector, CLIP-distilled |
| `models/tracking/{deep_sort,aflink,application_util}/` | `model/deep_sort`, `model/AFLink`, `model/application_util` |
| `models/model.py`, `end2end_model.py`, `gen_labels.py`, `util/` | same names under `model/` |
| `ops/` | `model/ops/` — MultiScaleDeformableAttention, flattened so `from ops.modules import MSDeformAttn` resolves |
| `third_party/detectron2/` | `model/util/detectron2` — vendored, moved out of the model tree |
| `data_loading/dataset.py` | `dataset/dataset.py` |
| `inference/` | `utils/video_relation_detection*.py`, `post_process.py`, `format_trajs.py` |
| `utils/` | `utils/utils.py` (as `eov_utils.py`), `parser_func.py`, `model/arguments.py` |
| `cli/train.py`, `cli/evaluate.py`, `cli/common.py` | `train/train.py` (their `run_train.sh` is now argument defaults) |

RePro's `models/`, `data_loading/`, `inference/`, `cli/`, `vlm/alpro*`, `ops/*.py`,
`experiments/`, `tests/`, `docs/` were removed. They are not recoverable from
this repository's history; the upstream RePro code remains at
<https://github.com/Dawn-LX/OpenVoc-VidVRD>.

---

## 2. Fixes applied during the port

**Import rooting.** EOV ran with `sys.path.insert(PROJECT_ROOT/"model")`, so its
modules imported each other as top-level (`from util.misc import ...`,
`import clip`, `import AFLink.config`). ~40 files rewired to package-absolute
imports (`from models.util.misc import ...`, `from vlm.backbones import clip`).
No `sys.path` manipulation remains.

**torchvision version gate (real bug).** `models/util/misc.py` had

```python
if float(torchvision.__version__[:3]) < 0.7 and float(...) != 0.1:
    from torchvision.ops import _new_empty_tensor      # removed years ago
```

`float("0.26.0"[:3])` is `0.2`, so this took the legacy branch on every modern
torchvision and crashed on import. Now compares `(major, minor)` numerically.

**`from model import Classifier` / `from dataset import Dataset_new`**
(`train/train.py`) could never have worked — both packages have empty
`__init__.py`. Repointed at `models.model` and `data_loading.dataset`.

**Hard-coded paths.** `data_loading/dataset.py` pointed at
`/data/wyq/jixf/jixf/VidVRD-baseline/dataset/vidvrd`; checkpoints and logs at
`../output/...` relative to the caller's cwd. These go through
**`utils/paths.py`**, derived from the repo root and overridable per machine:

```bash
VIDVRD_DATA_ROOT=/mnt/big/vidvrd python -m cli.train ...
```

Run `python -m utils.paths` to print every path and whether it exists.

**Hard-coded paths, second pass (2026-08-20).** The paragraph above originally
claimed *all* paths had been converted. That was wrong -- it was true only of the
files touched at the time. A tree-wide grep found 25 more, of which these were
live and would have broken a real run:

| File | Was | Now |
|---|---|---|
| `utils/parser_func.py` | `--clip_feat_path` -> `../dataset/vidvrd/data/...` | `paths.CLIP_FEAT_BANK` |
| `utils/arguments.py` | the **same option** declared again with a different folder *and* filename | `paths.CLIP_FEAT_BANK` |
| `utils/parser_func.py` | `--path_AFLink` -> `../output/ckpt/...` | `paths.AFLINK_CKPT` |
| `inference/video_relation_detection_openvoc.py` | 3 defaults -> `../dataset/vidvrd/data/...` | `paths.*` |
| `models/gen_labels.py` | `ROOT` + 6 literals | `paths.*` (honours `VIDVRD_FRAME_DIR`) |
| `models/end2end_model.py` | 2 unconditional ECC loads | tolerant, see below |

**Correction (2026-08-21).** An earlier version of this file said the duplicate
`--clip_feat_path` "mattered: whichever parser ran last silently won". That was
wrong. `utils/arguments.py` defines a parser, but nothing ever *calls*
`get_args_parser()` — the old `cli/train.py` imported the name and never used
it, and the CLI split removed even that import. Only `utils/parser_func.py`'s
`parse_args()` has ever run.

So there was no runtime conflict. The broken default in `parser_func.py` was
real and needed fixing; the second declaration in `arguments.py` was inert, and
`utils/arguments.py` is dead code — see the audit below.

The eval-time defaults mattered more -- `train.py` calls
`eval_relation_detection_openvoc()` with only three arguments, so all three broken
paths applied, and that code does not run until the **end of the first epoch**. A
full epoch of training would have been lost to a `FileNotFoundError`.

Left alone deliberately: `--coco_path` / `--lvis_path` (never consumed),
InternVideo's `MODEL_PATH` (unreached with `intern_feat=False`), and the paths
inside AFLink's own MOT17 trainer and `dct.py`'s figure code -- all in
`if __name__ == '__main__'` blocks unreachable from `cli/train.py`.

**ECC matrices missing.** `End2End_Model.__init__` unconditionally loaded
`VidVRD_ECC_{train,test}.json`, which EOV never shipped and which cannot be
reconstructed from anything here. Now loads if present, else returns `{}` with a
printed warning. `Tracker.camera_update` guards with `if video in ecc.keys()`, so
an empty dict makes camera-motion compensation a no-op -- tracking still runs.
**This is a real quality loss on moving-camera videos, not an equivalence.**

**`gpu_id=3` in their `run_train.sh`.** Upstream default; on a single-GPU
machine `CUDA_VISIBLE_DEVICES=3` exposes no GPU at all. That script is gone --
see "Entry points split" below.

**Evaluate-only path added.** There was none: `--eval`/`--resume`/`--ckpt_path`
were parsed but unused, so testing required a full training epoch. `cli/train.py`
now has `--eval_only`, which loads `--ckpt_path` and evaluates once. The two
copy-pasted evaluation blocks (identical apart from `tgt_split`) were factored
into `_predict_split()` / `evaluate_model()`, shared by both paths.

Note `--eval` could not be reused: it is a detector flag read in
`methods/detectors/ov_prompt/solq.py` and defaults to `True`.

In `--eval_only` the three step-1/2/3 checkpoint loads are skipped -- a trained
end-to-end checkpoint supersedes them -- so **testing a model you trained does
not require them**. AFLink is still required; association is part of inference.

**Frame indexing — worth knowing.** `models/gen_labels.py:30` is live code
(reached via `end2end_model.gen_feats`) and maps trajectory frame ids to
filenames as `'%06d.jpg' % (fid + 1)` — **1-indexed**, so trajectory frame 0 is
`000001.jpg`. `data_loading/dataset.py` is naming-agnostic (`sorted(os.listdir())`).
Frames are extracted 1-indexed to satisfy both. An earlier 0-indexed extraction
was discarded once this was found.

---

## 3. Data prepared

| | State |
|---|---|
| `data/vidvrd/videos/` | 1000 `.mp4` (4.3 GB), verified: all decode, frame count / width / height match annotations for **all 1000**. The source zips are deleted once extraction is verified complete. |
| `data/vidvrd/anno/{train,test}` | 800 + 200 JSON — the **only** copy |
| `data/vidvrd/data/` | trajectories, class splits, relation GT — copied from `MMP_OV_VidVRD/dataset/vidvrd/data/`; **all five filenames EOV asks for are present with identical names** (EOV builds on MMP) |
| `data/gt_jsons/` | rebuilt from annotations: 200-video test GT, video level (4835 relations) and segment level (2884 segments / 35916 relations) |
| `data/vidvrd/frames/` | 296,204 frames, 1-indexed, **~42 GB** (JPEG q95 on up-to-720p; my 11 GB estimate was low) |

All of it is reproduced by one command on a fresh clone:

```bash
python tools/prepare_data.py          # six steps, resumable, skips what is done
python tools/prepare_data.py --check  # read-only status report
```

The `meta` step pulls MMP's files over raw GitHub URLs rather than cloning; the
11 downloaded files were verified **byte-identical** (sha256) to the ones copied
from a local clone.

### Redundancy removed (2026-08-20)

`data/` went 51 GB -> 47 GB by deleting three things that were pure duplication:

| Removed | Size | Why it was safe |
|---|---|---|
| `videos/vidvrd-videos-part{1,2}.zip` | 4.3 GB | all 1000 `.mp4` verified present at exact byte sizes |
| `vidvrd/annotations/` | 33 MB | `diff -rq` byte-identical to `vidvrd/anno/` |
| `vidvrd/data/predicate_split.json` | 3 KB | referenced nowhere in the repo |

Dropping `annotations/` needed one code change:
`third_party/vidvrd_ii_helper/prepare_gts_for_eval.py` had the tree hard-coded and
now reads `paths.ANNO_DIR` / `paths.VIDEO_DIR`. **Verified by regenerating both GT
files and comparing sha256 — byte-identical to the pre-deletion versions.**

`prepare_data.py` was updated to match, so a fresh clone never creates the
duplicate: `step_anno` writes one copy, and `step_videos` removes each zip once
`_fully_extracted()` confirms every entry landed at the right size.

Still outside `data/`, **not** removed: `~/.cache/huggingface` holds 988 MB of
RePro/ALPro-era models (`vinvl_vg_x152c4` 567 MB, `bert-base-uncased` 421 MB, plus
another 421 MB copy of BERT under the legacy `transformers/` cache layout). This
port is CLIP-based and touches none of them.

Nothing here is synthetic. The placeholder-data generators were deleted earlier
at your request, and no fake data was introduced.

---

## 3b. Cleanup pass (2026-08-20)

A tree-wide sweep for things that would break or embarrass a fresh clone.

**Dead files removed.** Both were unreachable and unrunnable:

| Removed | Why |
|---|---|
| `inference/format_trajs.py` | two syntax errors — `json.dump(,f)` and a function with no body. Unfinished upstream scratch; nothing imported it |
| `models/util/clip_to_timm.py` | imports `models.methods.backbones.clip_vit_det`, which does not exist; also executes at module level and calls `sys.exit(1)` mid-file |

All 146 remaining `.py` files parse cleanly.

**Undefined names in live code.** Found with `ruff --select F821`, each verified
by executing the repaired path:

| File | Bug |
|---|---|
| `third_party/detectron2/layers/batch_norm.py` | `FrozenBatchNorm2d.forward` calls `F.batch_norm` in its eval branch; `F` was never imported |
| `utils/eov_utils.py` | `setup_seed()` calls `random.seed()` with no `import random` |
| `models/util/scheduler.py` | `CosineLRScheduler` warns through an undefined `_logger` |
| `vlm/backbones/clip_tagclip/build_model.py` | the `'CS-'` branch constructs `CLIPSurgery`, a class absent from this package — now raises `NotImplementedError` with a readable message |

Left as-is: `_tokenizer` in InternVideo's `clip_utils/clip.py` (unreached at
`intern_feat=False`) and a `Conv2d` re-export in detectron2 — both vendored, both
on dead paths. Editing them would widen the diff against upstream for no gain.

**`.gitignore` was silently broken.** It used trailing inline comments:

```
output/                     # checkpoints and logs
ops/build/                  # CUDA extension build tree
```

`#` only starts a comment at the **beginning** of a line, so those patterns were
literally `output/                     # checkpoints and logs` and matched
nothing. Build artefacts and anything under `output/` that wasn't `*.pth` or
`*.log` would have been committed. Rewritten with comments on their own lines and
every rule re-verified with `git check-ignore -v`.

**Line endings.** `utils/parser_func.py` is CRLF. An early edit rewrote it as LF
and produced a 703-line diff for a 10-line change. All subsequent edits go
through a helper that detects and restores the original endings.

### Entry points split, shell wrappers removed (2026-08-20)

`cli/train.py` and `cli/evaluate.py` are run as modules from the repository
root -- `python -m cli.train` -- and no longer manipulate `sys.path`. An earlier
version inserted the repo root so `python cli/train.py` also worked, but that
forced the imports below the insert, which linters flag and `ruff --fix` would
"repair" by hoisting them, breaking both scripts. Dropping the shim removed the
hazard and the need for any lint configuration.

`cli/` is now `train.py`, `evaluate.py` and `common.py` (shared setup, model
construction, the evaluation pass). Previously one script did both jobs behind
an `--eval_only` flag.

Splitting them buys more than tidiness: **evaluation no longer builds the train
dataset**, which the old `--eval_only` path constructed and never used. Each
`Dataset_new` holds two CLIP ViT-L/14@336px models, so that is ~2.7 GB of VRAM
reclaimed, on top of skipping the three step-1/2/3 checkpoint loads.

`run_train.sh` and `run_eval.sh` were deleted. They existed to set 25 flags, and
**13 of those differed from the argument defaults** -- so running the Python
directly silently produced a different experiment. Two of the gaps were serious:

| flag | script | old default |
|---|---|---|
| `lr` | 1e-5 | **0.01** (1000x) |
| `src_split` | base | **all** |

`src_split=base` *is* the open-vocabulary setting -- training on `all` lets the
model see novel categories, making the novel-split score meaningless. Anyone
running `python -m cli.train` without the wrapper would have got believable
numbers from an invalid experiment.

The defaults are now the paper's VidVRD configuration, verified to match the old
`run_train.sh` flag for flag. The wrappers had nothing left to add.

### Dead-code audit (2026-08-21)

Measured by importing `cli.train`/`cli.evaluate` and diffing `sys.modules`
against `git ls-files`, then checking every unloaded file for references.

**Removed:** `ops/build/` was **tracked in git** — four compiled artefacts, byte
-identical to `ops/functions/` and `ops/modules/`. `.gitignore` lists
`ops/build/`, but gitignore does not apply to already-tracked files, so they had
survived every earlier cleanup.

### Removed in the 2026-08-22 cleanup

The audit above originally *documented* this dead code rather than deleting it,
on the reasoning that removing files present upstream widens the diff against
EOV. That reasoning was reversed: the diff against upstream is recorded here in
prose, whereas a reader of the code has no way to tell a live module from a dead
one, and two of these were active traps (see below). All were verified
unreachable by an AST import walk from `cli/`, then by grep, then by the test
suite in `tests/test_imports.py`, which fails if any of them is ever imported
again.

| Removed | Lines | Why |
|---|---|---|
| `models/methods/detectors/ov_prompt/solq.py` | 1,393 | alternative detector; its only import was already commented out |
| `models/methods/segmentation.py` | 390 | reachable only through `solq.py` |
| `models/methods/detectors/ov_prompt/dct.py` | 198 | reachable only through `solq.py` |
| `models/methods/detectors/ov_prompt/backbone.py` | 141 | superseded by `models/methods/backbones/clip_backbone.py`, which is what the live detector imports |
| `inference/video_relation_detection.py` | 298 | **trap**: no importer, and it differs from `_ab.py` by 7 lines. The live file is `_openvoc.py`, so editing the obviously-named one had no effect |
| `inference/video_relation_detection_ab.py` | 301 | as above |
| `utils/arguments.py` | 222 | parser never called |
| `utils/video_transform.py` | 751 | byte-identical to `vlm/backbones/internvideo/video_transform.py`, which is the one that gets imported |
| `models/util/scheduler.py` | 248 | no importer |
| `vlm/ptm_encoder.py` | 59 | no importer |
| `configs/VidVRD_class_spilt_info.py` | 52 | a second copy of `VidVRD_class_spilt_info.json`; nothing read it, and two sources of truth for a class split can drift silently |
| `tools/freeze_model.py`, `tools/generate_detections.py` | 400 | deep_sort's TensorFlow feature extractor; imports `tensorflow.contrib.slim`, which does not exist in TF2 |

Total 39,740 → 35,332 tracked Python lines.

Two unreachable, broken branches were also removed from
`models/relation_classifier.py`: `ObjectTextEncoder` and `PredicateTextEncoder`
both branched on `text_encoder == 'intern'` and called a
`build_intern_fixed_prompts()` **defined nowhere**, with the predicate branch
additionally leaving `pre_classifier_weights` unbound. Nothing passed `'intern'`,
so neither could run; they now raise `ValueError` for an unsupported encoder.

### Deliberately kept, though currently unreachable

| Kept | Lines | Why |
|---|---|---|
| `vlm/backbones/internvideo/` | 8,787 | a vendored **video** encoder, orphaned when `ptm_encoder.py` went. Kept because the temporal-degeneracy work in [91_Extension-guide.md](91_Extension-guide.md) may want a real video backbone, and this is already vendored for this task. Delete it if that direction is abandoned |
| `vlm/backbones/{i3d,resnet}.py` | 1,000 | same reasoning, lower value |
| `tools/build_clip_object_bank.py` | — | superseded for VidVRD by the authors' bank, but needed for any new dataset |

**Verified clean:** every module we wrote (`cli/`, our `tools/`, `utils/paths.py`)
reports no unused imports or variables. The ~100 findings in the ported model
tree are upstream style, left alone to keep the diff against EOV readable.

**Four `simple_tokenizer.py` copies** exist across the CLIP variants. That
duplication is upstream's — each vendored backbone carries its own — and
de-duplicating it would diverge for no functional gain.

## 4. Environment

`conda env repro-next` — python 3.12.13, torch 2.11.0+cu128, CUDA available on
the GTX 1650 (sm_75). Added during the port: `matplotlib`, `ftfy`, `regex`,
`scikit-learn`, `fvcore`, `timm`, `opencv-python-headless`, `gdown`, plus
`cuda-nvcc` / `cuda-cudart-dev` 12.8 and `gcc/gxx_linux-64=13`.

GCC 13 specifically: the env had 14.4.0, and CUDA 12.8 refuses `>= 14.0`.

---

## 5. OPEN PROBLEMS

### 5.1 BLOCKER — EOV checkpoints not downloadable

`cli/train.py` loads **four** pretrained checkpoints before training:

```
output/ckpt/checkpoint_vidvrd0059_new_1e-5.pth                            object detector
output/ckpt/baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth
output/ckpt/vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth         object classifier
args.path_AFLink                                                          AFLink tracker
```

The Google Drive folder in their README
(`1IH5fyfZN7on_DMsv55385scd0YrlueTO`) returns **401** to `gdown` — it is not
shared as "anyone with the link", or is rate-limited.

**Nothing can train or test until these are obtained.** Options, in order:

1. Open the Drive link in a browser while signed in; it may be reachable
   interactively even though the API refuses.
2. Their Baidu link: `https://pan.baidu.com/s/1jZbnkYexAZQcGApyBUhQjw?pwd=jfec`
3. Email the author — their README invites it:
   `1285441164yq@gmail.com`

Once downloaded, put them in `output/ckpt/` (the names above) and set
`--path_AFLink`.

### 5.2 CUDA extension — BUILT and verified

`MultiScaleDeformableAttention` compiles and is numerically correct: the CUDA
kernel matches `ms_deform_attn_core_pytorch` to **8.7e-19** max abs difference.

Five failures on the way, each a real incompatibility:

1. no `g++` on PATH -> use the conda toolchain (`x86_64-conda-linux-gnu-g++`)
2. that toolchain was 14.4.0; CUDA 12.8 refuses `>= 14.0` -> installed GCC 13
3. missing `cuda_runtime_api.h` -> `CPATH=$CUDA_HOME/targets/x86_64-linux/include`
4. missing `cusparse.h` -> installed `libcusparse-dev` and friends
5. **source-level:** `ms_deform_attn_cuda.cu` passed `value.type()` to
   `AT_DISPATCH_FLOATING_TYPES`, but modern torch needs a `c10::ScalarType`, not
   `DeprecatedTypeProperties`. Changed to `value.scalar_type()` in both the
   forward and backward dispatch. **This is an edit to vendored CUDA source** --
   note it if you ever diff against upstream EOV.

`ops/test.py` (the op's own gradcheck) OOMs on this 4 GB GPU: it builds a
double-precision Jacobian. That is a VRAM limit, not a kernel fault -- the
forward comparison above is the meaningful check.

Build command:

```bash
cd ops
export CUDA_HOME=$HOME/miniconda3/envs/repro-next
export PATH=$CUDA_HOME/bin:$PATH
export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export TORCH_CUDA_ARCH_LIST="7.5"
python setup.py build install
```

Check the final state with `python -c "import MultiScaleDeformableAttention"`.

### 5.3 How far a real run gets

`cd cli && python train.py --dataset vidvrd --batch_size 1 ...` now:

1. imports the whole tree cleanly (detector, tracker, CLIP stack, dataset)
2. parses and logs its full config
3. creates `output/{ckpt,log}`
4. starts downloading CLIP `ViT-L/14@336px` (934 MB) into the torch cache

so the port is functional up to the point where it needs the four pretrained
checkpoints from 5.1. **No training step has executed yet** — that is blocked on
the checkpoints, not on the code.

### 5.4 Not yet verified
- **`clip_L14_feat_vidvrd.pkl` — RESOLVED, but as a reconstruction.** The
  detector loads it at construction
  (`models/methods/detectors/ov_prompt/model.py:274`) and uses it in
  `forward_train`: one CLIP **image** embedding per category is sampled as an
  "image query", mixed 75% text / 25% image (OV-DETR's scheme). It is
  `category_id -> Tensor(K, 768)` of CLIP ViT-L/14@336px embeddings of GT object
  crops. **The authors did not publish this file.**
  `tools/build_clip_object_bank.py` rebuilds it from real GT boxes, real frames
  and the real CLIP encoder -- nothing synthetic -- but their recipe is unknown:
  exemplars per class, which frames, crop padding, and whether train-only. Those
  choices shift the image queries and therefore the trained detector, so results
  from this bank are **not directly comparable to the paper** until the original
  file is obtained. The tool's docstring records every choice made.
- `models/gen_labels.py` has its own `ROOT = '../dataset/vidvrd'` that should be
  moved onto `utils.paths` — left alone so far to avoid touching label
  generation before anything runs.
- **4 GB of VRAM is the main risk.** EOV runs CLIP ViT-L/14@336px *and* TagCLIP
  *and* a deformable-DETR detector *and* InternVideo. batch_size is already 1. The op's own gradcheck already OOMed here. Expect to need Colab
  (T4, 16 GB) or a bigger card for real training; this machine is fine for
  development and inference-sized work.
- Disk: frames are ~42 GB. There was 916 GB free.

### 5.5 ALPro removed

`vlm/backbones/alpro_weights/` (2.4 GB: the ALPro checkpoint plus
`bert-base-uncased`) was deleted on 2026-08-20. Nothing in the port referenced
it — the port is CLIP-based — and it was gitignored, so the repository is
unchanged.

**Consequence:** the RePro baseline can no longer run here — its equivalence
harness loads ALPro, and both the weights and that code path are gone. This was
a deliberate trade: the port is CLIP-based and never used ALPro. To revive that
baseline, start from upstream RePro rather than from this repository.

---

## 6. When you return

```bash
python tools/prepare_data.py --check      # all six data steps, have / need
python -m utils.paths                     # every path, and whether it exists
python tools/extract_frames.py --verify   # frame counts vs annotations
python -c "import MultiScaleDeformableAttention"   # extension present?
```

Then obtain the checkpoints (5.1) and try:

```bash
python -m cli.train
```
