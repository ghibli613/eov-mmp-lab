# Span idea — S1 pre-check 2: where the 30-frame training chunks come from

No pre-registration, training or scoring. Training annotations only, base labels
only (novel instances dropped at parse). No test annotation file was opened.

**Answer: case (B).** The official ImageNet-VidVRD training annotations are
themselves chunked: every base instance is exactly 30 frames on a 15-frame grid,
and every gap is a whole number of missing grid chunks. The missing chunks
mostly come in runs, not singly.

## 1. The file the earlier work read

- **Path:** `data/vidvrd/anno/train/<video_id>.json`, 800 files, one per video.
  `utils/paths.ANNO_TRAIN_DIR`, and the `TRAIN_DIR` / `ANNO_TRAIN` constants in
  stage 0 – Batch B2, all point there.
- **Producer: the official ImageNet-VidVRD release.** No EOV-MMP, RePro or MMP
  preprocessing is involved:
  - `tools/prepare_data.py` `step_anno()` downloads `vidvrd-annotations.zip`
    from the HuggingFace dataset `shangxd/imagenet-vidvrd`, the account of
    Xindi Shang, ImageNet-VidVRD's first author.
  - The zip (SHA256 `c394ad77719f2cc712935c25cc9bad2aaad76deca030e8d93e9cd6d710618df2`)
    has the official layout `vidvrd-dataset/{train,test}/*.json`, with entries
    dated 2017-08-13.
  - It is the release that `../repro/datasets/vidvrd-dataset/readme.md` points
    to (xdshang.github.io/docs/imagenet-vidvrd.html).
  - `step_anno()` copies the JSON members unchanged (`shutil.copyfileobj`).
    **All 800 train files are byte-identical to the zip's members.**
  - The only other training-side file on disk, MMP's
    `train_object_trajectories_gt.json`, holds trajectory fragments, not
    relation labels. It was never used for labels.
- **Loading code** (`pilot_analysis/stage2/common.py`, used by Batch A–B2; stage
  0 – 1c read the same files the same way):
  ```python
  ANNO_TRAIN = os.path.join(ROOT, "data", "vidvrd", "anno", "train")
  def load_train_video(path, base):
      d = json.load(open(path))
      rels = [r for r in d["relation_instances"] if r["predicate"] in base]
      n_novel = len(d["relation_instances"]) - len(rels)
      del d["relation_instances"]
  ```
  and `pilot_analysis/stage0/stage0.py:28`: `TRAIN_DIR = "data/vidvrd/anno/train"`.

## 2. The official training annotations

They are the file above, so there is nothing separate to compare against.

| | base instances | min | median | 95th pct | max | share > 30 frames |
|---|---:|---:|---:|---:|---:|---:|
| official train `relation_instances` | 22,616 | 30 | 30 | 30 | 30 | **0.0%** |
| same, merging same-triplet chunks that overlap or abut | 17,643 | 30 | 30 | 60 | 630 | 10.6% |

Every `begin_fid` is a multiple of 15 (22,616 / 22,616).

The pilot's long-instance statistics (`scripts/gt_duration_stats.py` read
`data/vidvrd/anno/test`) come from the **test** folder of the same official
archive. **So the train/test difference is in the official release itself.**
Test annotations were not opened for this check.

## 3. Gaps between consecutive same-triplet chunks of the same pair

There are 10,004 consecutive pairs. Gap = next begin − previous end; missing grid
chunks = (next begin − previous begin) / 15 − 1.

| gap (frames) | missing 15-frame grid chunks | pairs | share |
|---:|---:|---:|---:|
| −15 (overlap) | 0 | 4,302 | 43.0% |
| 0 (abut) | 1 | 671 | 6.7% |
| 15 | 2 | 779 | 7.8% |
| 30 | 3 | 662 | 6.6% |
| 45 | 4 | 889 | 8.9% |
| 60 | 5 | 507 | 5.1% |
| 75 | 6 | 311 | 3.1% |
| 90 | 7 | 425 | 4.2% |
| 105 | 8 | 273 | 2.7% |
| ≥ 120 | ≥ 9 | 1,185 | 11.8% |

- **All 5,031 positive gaps are exact multiples of 15 frames.** Nothing falls off
  the grid.
- They are **not dominated by single missing chunks.**
  - Gaps of exactly 15 or exactly 30 frames are 15.5% and 13.2% of the
    positive gaps.
  - Among pairs with at least one missing chunk, only 25.4% miss 1–2; the
    median is 4 missing chunks (≈ 75 frames).
  - The longest gap is 1,500 frames.

## 4. Which case holds

**(B): chunked, and the gaps are whole missing chunks on the 15-frame grid.**
- (A) is ruled out: the official training annotations are this file, and no
  instance is longer than 30 frames.
- (C) is ruled out: no gap is irregular.

One qualification: the missing chunks usually come in runs (median 4), so
"whole missing chunks" doesn't mean "occasional single dropouts".

The annotations do not record whether a missing chunk means the relation was
absent or just not labelled. Bridging only 1–2 missing chunks would join about a
quarter of the non-contiguous pairs. Merging only overlapping or abutting chunks
leaves 89.4% of training instances at 30 frames.
