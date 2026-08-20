# Every data file: what it is, where it comes from, what reads it

All of it is produced by one command:

```bash
python tools/prepare_data.py           # fetch + build everything
python tools/prepare_data.py --check   # read-only status
```

Paths are never hard-coded. Everything resolves through
[`utils/paths.py`](../utils/paths.py), derived from the repo root:

```bash
python -m utils.paths          # print every path and whether it exists
```

Override per machine:

```bash
export VIDVRD_DATA_ROOT=/mnt/big/vidvrd
export VIDVRD_OUTPUT_ROOT=/mnt/big/output
```

---

## The six preparation steps

| Step | Produces | Size | Source |
|---|---|---|---|
| `videos` | 1000 `.mp4` | 4.3 GB | HuggingFace `shangxd/imagenet-vidvrd` |
| `anno` | 800 + 200 annotation JSON | 33 MB | same dataset |
| `meta` | 11 metadata files | 41 MB | the MMP repo, over raw GitHub URLs |
| `frames` | 296,204 JPEGs | ~42 GB | decoded locally from the videos |
| `gt` | 2 evaluation GT files | 82 MB | derived from the annotations |
| `bank` | CLIP object bank | 5 MB | computed from GT crops |

Each step is skipped when already complete, so re-running after an interruption
is cheap.

---

## `data/vidvrd/videos/` — the raw videos

1000 `.mp4`, 800 train / 200 test. The download zips are deleted automatically
once every entry is verified present at the right byte size, so steady-state
disk is 4.3 GB rather than 8.6 GB.

## `data/vidvrd/anno/{train,test}/` — the annotations

One JSON per video: frame count, resolution, object trajectories with
categories, and `relation_instances` (subject, predicate, object, frame range).

**This is the only copy.** There used to be a byte-identical duplicate at
`data/vidvrd/annotations/`; it was removed and the GT builder repointed here.

Read by `data_loading/dataset.py` (via `paths.ANNO_TRAIN_DIR`) and by
`third_party/vidvrd_ii_helper/prepare_gts_for_eval.py`.

## `data/vidvrd/frames/<video>/%06d.jpg` — decoded frames

296,204 JPEGs, quality 95. The dataset reads **frames, not videos**.

> **1-INDEXED.** `000001.jpg` is trajectory frame 0, because
> [`models/gen_labels.py`](../models/gen_labels.py) maps frame ids as
> `'%06d.jpg' % (fid + 1)`. Do not "fix" this to 0-indexed — an earlier
> 0-indexed extraction was discarded once this was found.

Verify against the annotations:

```bash
python tools/extract_frames.py --verify     # expect 0 problems
```

## `data/vidvrd/data/` — metadata, from the MMP repo

EOV builds on MMP and expects these exact filenames. All 11 were verified
byte-identical (sha256) to a local clone of MMP.

| File | What it is |
|---|---|
| `train_object_trajectories_gt.json` | GT tracklets, train — step 3 trains on these |
| `test_object_trajectories_gt.json` | GT tracklets, test |
| `test_object_trajectories_meta.json` | detected (not GT) test tracklets |
| `test_relation_gt.json` | relation GT used by the open-vocab evaluator |
| `openvoc_obj_class_spilt_info.json` | object `base`/`novel` split, `cls2id`, `id2cls` |
| `openvoc_pred_class_spilt_info.json` | predicate `base`/`novel` split |
| `id2object` / `object2id` / `id2predicate` / `predicate2id` `.json` | vocabularies |
| `prior.pkl` | predicate co-occurrence prior, used in post-processing |

(Upstream's spelling of "spilt" is preserved — the code expects it.)

## `data/gt_jsons/` — evaluation ground truth

Two files, derived from the annotations, not downloaded:

- `VidVRDtest_gts.json` — video-level, 4,835 relations
- `VidVRDtest_segment_gts.json` — segment-level, 2,884 segments / 35,916 relations

Regenerate with `python tools/prepare_data.py --steps gt --force`; the output is
byte-identical each time.

The difference from the annotations: annotations are raw per-video labels; the
GT files are those labels **reshaped into the evaluator's format**, including
the 30-frame/stride-15 segment grid used for segment-level scoring.

## `data/vidvrd/data/clip_L14_feat_vidvrd.pkl` — the CLIP object bank

`category_id -> Tensor(K, 768)` of CLIP ViT-L/14@336px **image** embeddings of
ground-truth object crops. The detector samples one per category per step as an
"image query", mixed 75% text / 25% image (OV-DETR's scheme) — see
[`models/methods/detectors/ov_prompt/model.py:274`](../models/methods/detectors/ov_prompt/model.py).

> ⚠️ **This is a reconstruction, not the authors' file.** They never published
> theirs. Every input is real — real GT boxes, real frames, the real CLIP
> encoder — but their recipe is unknown: exemplars per class, which frames, crop
> padding, train-only or not. Those choices shift the image queries and
> therefore the trained detector, so **numbers built on this bank are not
> directly comparable to the paper.** Every choice made is recorded in
> [`tools/build_clip_object_bank.py`](../tools/build_clip_object_bank.py).

Current bank: 35/35 categories, 15–50 exemplars each, L2-normalised, dim 768.

Written with `torch.save`, because the consumer is `torch.load`. An earlier
version of the tool used `pickle.dump`, which `torch.load` cannot read
("Invalid magic number").

---

## What is *not* in the repo

| Missing | Consequence |
|---|---|
| The 4 pretrained checkpoints | cannot train; see [known-issues.md](known-issues.md#1-checkpoints) |
| `VidVRD_ECC_{train,test}.json` | camera-motion compensation disabled |
| Steps 1–3 training code | cannot regenerate the checkpoints yourself |

None of it is faked. Where a file is missing, the code either fails with a clear
message or degrades explicitly and says so.
