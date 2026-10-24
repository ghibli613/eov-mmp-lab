# Stage 1, session 3 — Tasks 3–6 report

Run from commit `fdeeeb4` on the rented RTX 4090 (`OMP_NUM_THREADS=8`), Tasks 3 → 4 → 5 → 6
back to back. No gate tripped and nothing failed.

| task | result | gate | time |
|---|---|---|---|
| 3. full cache, 1,000 videos | 1,000 cached, 0 failed, max fidelity residual **0.0** | any failure / residual > 1e-4 | 4.08 h |
| 4. baseline PredCls | novel **21.70** (paper 21.65) | \|novel − 21.65\| ≤ 3.0: **pass** (+0.05) | < 1 min |
| 5. paraphrase stability | novel spread **10.97**, excl. A4 **11.08** | verdict: **string-sensitive** (both) | 11 min |
| 6. SGDet cache, 200 test videos | novel **15.79**, all **27.18** | within 0.3 of 15.79 / 27.18: **pass** (−0.004 / +0.005) | 3.5 h |

## Task 3 — full cache

- 1,000 videos (200 test, 800 train), 10,318 pairs, 22,066 slots, 2.7 GB.
- Every file matches its manifest SHA256. The worst fidelity residual over all
  1,000 videos is 0.0: the logits rebuilt from the cached tensors reproduce
  modelC's returned scores exactly.
- 995 videos were cached in this run. The other 5 are the Task 2 videos (3
  train, 2 test), kept by the resume logic. The cache code (`cache_wrapper.py`,
  `gt_tracks.py`, `optionB.py`) is byte-identical between the Task 2 fix
  (`b3edbfb`) and this run (`fdeeeb4`), so all 1,000 come from the same code.
- Train tracks: option B (A5). Test tracks: repo densifier.

## Task 4 — baseline PredCls from the cache

| split | mAP | R@50 | R@100 | paper PredCls mAP | diff |
|---|---:|---:|---:|---:|---:|
| novel | **21.70** | 31.57 | 35.54 | 21.65 | **+0.05** |
| all | **37.20** | 28.58 | 34.62 | 39.83 | **−2.63** |

200 test videos, 1,392 GT-track pairs. The scorer equals
`eval_relation_detection_openvoc` exactly (asserted in the run).

**Flag, not a gate:** the all split is 2.63 below the paper. The gate covers
novel only, and novel matches to 0.05. The shipped checkpoint is not exactly
the paper's: its SGDet is 15.79 / 27.18 against the paper's 15.04 / 26.34. The
all-split PredCls gap is still unexplained.

## Task 5 — paraphrase stability

Preconditions and checks:
- `paraphrases_frozen.json` matched its SHA256 (`aa2cd527…`).
- Check 1: rebuilding the text-side buffers (`token_prefix`, `token_suffix` and
  the three template embeddings) from the original strings reproduces the
  checkpoint's buffers with max difference **0.0**.
- Check 2: logits recomputed from the cached fp16 `visual_pre_embeddings` differ
  from the cached logits by at most **0.0023**. The recomputed original scores
  the same as the cached logits (Task 4) to two decimals.

### mAP per condition

| condition | novel | novel, excl. A4 19 | all | all, excl. A4 19 |
|---|---:|---:|---:|---:|
| original (recomputed) | 21.70 | 23.55 | 37.20 | 38.24 |
| set 1 | 20.57 | 22.31 | 25.31 | 27.36 |
| set 2 | 17.03 | 18.90 | 21.62 | 23.32 |
| set 3 | 10.73 | 12.47 | 20.97 | 22.63 |
| set 4 | 12.39 | 12.93 | 16.23 | 17.19 |

Novel R@100: 35.54 (original), 32.89, 37.52, 30.25, 33.72.

### Spreads (A4: both reported, neither replaces the other)

| | with original | sets only |
|---|---:|---:|
| novel, all predicates | **10.97** | 9.84 |
| novel, excl. A4 19 | **11.08** | 9.84 |
| all, all predicates | 20.97 | 9.08 |
| all, excl. A4 19 | 21.05 | 10.16 |

**Verdict (novel spread with original, ≥ 2.0): string-sensitive**, over all
predicates and excluding the A4 19. The known convention shifts (`*_front`,
`stop_*`) don't explain the spread: excluding them changes it by +0.11.

### Per predicate (AP of each predicate scored alone, by the official evaluator)

| split | predicates with GT | moved > 5 AP | with ≥ 10 GT: moved | with ≥ 20 GT: moved |
|---|---:|---:|---:|---:|
| novel | 61 | 41 | 10 of 15 | 5 of 9 |
| all | 132 | 94 | 66 of 80 | 51 of 60 |

Many of the novel predicates have 1–4 test instances, where one hit flipping
moves AP by 25–100 points, so the raw > 5 AP count is noisy. Novel predicates
with ≥ 20 GT instances:

| predicate | GT | spread | AP: orig / s1 / s2 / s3 / s4 |
|---|---:|---:|---|
| sit_next_to | 38 | **23.7** | 51.1 / 41.4 / 47.0 / 30.4 / 27.3 |
| sit_behind | 47 | 9.2 | 50.1 / 47.2 / 52.4 / 48.6 / 43.2 |
| walk_away | 47 | 7.7 | 17.9 / 14.7 / 15.6 / 10.2 / 12.9 |
| walk_past | 34 | 7.0 | 9.8 / 2.8 / 2.8 / 4.5 / 9.0 |
| sit_beneath | 20 | 6.3 | 14.4 / 15.0 / 11.1 / 8.7 / 10.6 |
| stand_with | 64 | 4.8 | 23.0 / 20.9 / 22.7 / 25.7 / 21.6 |
| swim_next_to | 25 | 4.6 | 53.9 / 58.3 / 54.8 / 53.7 / 55.2 |
| stand_above | 35 | 4.5 | 71.7 / 71.7 / 73.8 / 75.3 / 70.8 |
| walk_toward | 43 | 3.3 | 5.6 / 7.6 / 8.1 / 8.3 / 8.9 |

The full table, with all 132 predicates in both splits, is `per_predicate` in
`task5/task5_paraphrase.json`.

### Observation, not part of the verdict

Scored one predicate at a time, the paraphrases barely hurt, yet the novel mAP
halves:

| condition | novel mAP | mean per-predicate AP | instance-weighted |
|---|---:|---:|---:|
| original | 21.70 | 26.09 | 31.77 |
| set 1 | 20.57 | 30.72 | 29.64 |
| set 2 | 17.03 | 27.50 | 30.88 |
| set 3 | 10.73 | 26.40 | 26.86 |
| set 4 | 12.39 | 27.40 | 29.55 |

Read with care, since many predicates have tiny n. Most of the mAP drop does
not seem to be each predicate losing its own ranking. It looks more like the
predicates' scores stop being comparable with each other when they compete in
one ranked list per video (per-slot top-20, per-video top-200). If that holds,
the baseline is string-sensitive mainly in its cross-predicate calibration, and
less in per-predicate discrimination. This is untested, and is a possible
follow-up.

## Task 6 — SGDet cache on detected trajectories

| split | mAP | R@50 | R@100 | pilot SGDet | diff |
|---|---:|---:|---:|---:|---:|
| novel | **15.79** (15.786) | 16.36 | 19.01 | 15.79 | −0.004 |
| all | **27.18** (27.185) | 17.10 | 19.46 | 27.18 | +0.005 |

- 200 test videos, 1,580 detected pairs, 7,066 slots, 676 MB. Max fidelity
  residual **0.0**, and every file matches its manifest SHA256.
- Run with the repo's normal test path (seed 3407, `--frame_stride 1`, the
  shipped `VidVRD_ECC_test.json`, not the stage1c supplement).
- Cached per video: the B6 set on detected pairs, the detected trajectories
  (category, score, boxes), RoI means for tracks and pair unions, and
  `global_proj`.

## Copy-off

`../session3_box_copy/` (outside the repo):
- `cache_s3/`: the Task 3 cache, plus `task4/`, `task5/`, manifests, `once.pt` and
  the Task 2 files;
- `cache_s3_sgdet/`: the Task 6 cache, `sgdet_{novel,all}.json` and `task6_sgdet.json`;
- `SHA256SUMS_session3.txt`: 1,217 files, written on the box;
- `logs/`: `session3.log`, `session3_task2.log`, `session3_rest.log`.

**Verified locally:** `sha256sum -c` passes on all 1,217 files, and the copy
holds exactly those 1,217 files: 200 test + 800 train in `cache_s3`, 200 in
`cache_s3_sgdet`.

**Instance 53673566 is stopped, not destroyed**, and its disk is kept. It
should be destroyed only after the chat confirms the copies are intact.
