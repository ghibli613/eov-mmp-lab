# D2 — at which front-end stage are D1's U1 object tracks lost? (final report)

> **Test-label error analysis, disclosed; no method evaluated.** No method, threshold or
> parameter was chosen from it. Anything built after it is designed and tuned on training
> videos only, under its own amendment.

**Status.** D2 is complete; this is the hard stop. Nothing was built.
- **Step 2 ran on its own machine.** All 200 test videos were recorded, and every
  one is **identical to the baseline** (ground rule 3).
- **Steps 3–4 ran on the laptop.** The test labels were opened only after every
  check had passed.
- The machine is ready to be destroyed. The token was wiped before the recording
  started, and the test annotations were deleted after the run.

**Reading (A25's rule, by relation, pooled, n = 262).** The largest group is
**rescore (O1b), 96 = 36.6% (95% CI 23.4–50.6%)**. By the pre-registered mapping,
this names *relation-aware rescoring of detections before the cut*. The second is
**detector (O1a), 93 = 35.5% (18.9–52.3%)**, and the two are **not separated**
(paired CI of the difference: −26.4 to +29.9 points). Link (O2 + O3) is third:
67 = 25.6% (15.3–39.4%).

**What is robust (descriptive).** Which of O1a and O1b is larger is not robust. It
flips with the score floor (Table 4), and with a small post-hoc relabelling (§6).
What holds under every variant: **about 70% of U1 relations have no good-enough box
above the cut, so they are lost before tracking.** O1a + O1b is 189 of 262 = 72.1% at
every floor, and 181 = 69.1% under the post-hoc relabelling. Only about 26–29% are
lost in tracking or the filter.

Date: 2026-10-10 · METOR (`ghibli613/metor`), `studies/d2/` · EOV code `fdeeeb4` (read only)

---

## 1. Pre-registration and input checks

| item | SHA256 / value |
|---|---|
| ov-pairtime prereg (A24–A24i, closed), final | `981bf067da43c41df8759a260c9bde61014b411809e13bf9838f62f6906b5ce5`, verified, not modified |
| METOR prereg before A25 (after A23a) | `bdfe25e62b20d3d82c743c848197d3448aff9d40fe6e98efdac3fd6134af268b` |
| after **A25** (definitions, reading rule) | `2b54bf65fb43e3150ba4942bc8f14eea76e0ff3953b33568f35299b507de69ea` |
| after **A25a** (code pins) | `f18440cb7737ecb1c562963648ffeae66cb4580799cc510330dabfce714ee175` |
| after **A25b** (folders moved under `studies/`) | `58211c14debd881df31f670ff6722f7ae575558d39863803e40c8b83723d6d6a` |
| after **A25c** (approval; Table 6; own machine; annotations; frame-range finding) — current | `838111d1f669f0eadcb4cde84212db21b08caaefcb652aee101b41d989d7ba60` |
| local commits (not pushed) | A25 `3a54d6e` · code + A25a `b4d53a0` · move + A25b `45c3c3a` · Steps 0–1 report `2afe1d2` · A25c `5db91d6` · this report (next commit) |

**Verified by `analyse.py` before any test label was opened:**
- **A25's input pins:**
  - D1's `tc_missed.json` (`2e4c53b9…`), `tc_tables.json` and `tc_run.py`;
  - `SHA256SUMS_session3.txt`;
  - the test GT files (as bytes);
  - the class splits, the ECC, the checkpoint and AFLink.
- **The 204 SGDet cache files** and **the 200 test annotation files**, as bytes.
- **Code and baseline:** the d2 code and the 22 EOV front-end files against
  `code_pins.json` (`95816845…`), and the baseline tracks
  (`3d9c5c86…`, 658 tracks).
- **Every recording,** against its manifest hash.
- **Every video:** ground rule 3, and the replay check (the repo's own
  `format_trajectories_test` plus the 15% rule, on the recorded rows, reproduces the
  final tracks), with no post-processor mismatch.

**The join.** D1's 262 U1 rows all joined to their GT entries, and every row's
recomputed `best_pair_min_viou` equals D1's. 201 of them are base-only, as in A22
§1d and the brief.

## 2. Execution (Step 2)

| item | value |
|---|---|
| machine | Vast.ai, 1× RTX 4090 (24 GB, driver 570.172.08), 503 GB RAM, 128 cores, 60 GB disk. Vast PyTorch template: Python 3.12.14, torch 2.11.0+cu128, nvcc 12.8. `OMP_NUM_THREADS=8`. |
| upload (laptop → machine; copy, not push) | `git archive fdeeeb4`, 11 metadata files, 200 test annotations and `studies/`: **519 files, all SHA256-equal to the laptop** |
| fetched with the token | the end-to-end checkpoint, AFLink, the CLIP bank and both ECC files, **all equal to the laptop's SHA256**. The frame bundle's `SHARDS.json` and `MANIFEST.json` equal the laptop's copies. **231 shards, each SHA256-checked; 200 videos.** |
| CLIP ViT-L/14@336px | public URL; `3035c92b…` OK (RB1's pin) |
| token | read hidden by the author's run of `fetch_with_token.sh`, held only in that process's `HF_TOKEN`, then unset with token files removed at 04:19 UTC, before any recording. **Every scan found files 0, process environments 0, token files 0:** at setup, before and after the fetch, at the start of every run, and at cleanup. |
| pin check on the machine | `--check_only`: the 22 front-end files, d2 code, baseline tracks, checkpoint, AFLink, ECC and class split all equal their pins. The code root is a `git archive` copy with no `.git`, so the commit is checked through these hashes. |
| 2-video check | `ILSVRC2015_val_00046001` 42.1 s (session 3: 41.6 s); `ILSVRC2015_train_00047020` 13.1 s (13.0 s). **Identical to the baseline; +1.1%,** so the 25% rule passed. Projected 3.5 h, within the 6 h limit. |
| full run | 200/200 videos, **200 identical**, 46,035 frames processed; forward time 10,873 s (session 3: 10,507 s, +3.5%). Ended 08:25 UTC. |
| a disclosed detour | After 7 more videos, two processes were started on disjoint halves of the remaining 191, to use the GPU, which was only 30–45% busy. The code runs cuDNN deterministic, with benchmark off. GPU time-slicing between the processes slowed each more than 3×. **After about 18 min, with no video completed in either, both were stopped and the single run resumed.** They wrote nothing but their run logs (`logs/box/parallel_attempt/`). Cost: about 18 min. |
| test annotations | **Deleted right after the run:** 203 files (`test_relation_gt.json`, `test_object_trajectories_gt.json`, `test_object_trajectories_meta.json` and 200 `anno/test`). Their SHA256 was logged first, and all 203 equal the laptop's pinned copies. 0 files left. **Disclosed:** `cleanup.sh` stopped right after the deletion (`pipefail` on its own `ls` of the deleted files), so its absence check and token scan were run by hand straight after, and noted in `cleanup.log`. |
| copy-off | `copy_back.sh`: 200 videos, 400 recording files, **all equal to the manifest's SHA256** (`data/rec/manifest.jsonl` `37d26c1c…`), plus the machine logs |

## 3. The front end, stage by stage (Step 0; full map with both repos' lines in `stage_map.md`)

EOV `fdeeeb4`, the commit that made the baseline.

| # | stage | code fact | where |
|---|---|---|---|
| 0 | frames | every frame in [begin_fid, end_fid), **the min/max of the GT object trajectory extents** (§4) | `data_loading/dataset.py:120-126` |
| 1 | detector | 300 queries; sigmoid probabilities over 35 classes; no NMS | `post_process.py:191` |
| 2 | **cut** | keep a query iff its **max class probability is > 0.35**; otherwise the frame's single best query; then the top 300 (query, class) entries | `post_process.py:194-195`, `:221-233`, `:264-282` |
| 3 | dedupe | `unique_boxes`: of two boxes with IoU > 0.8, the earlier (higher-scored) one is dropped | `gen_labels.py:145-162`; `end2end_model.py:211-221` |
| 4 | **deep_sort** | confidence is 1 for every detection, so its confidence, height and NMS filters are no-ops; NSA snaps updated boxes to their detections; `n_init` 3, `max_age` 30; only confirmed tracks are output | `deep_sort_app.py:55`, `:171-201`; `kalman_filter.py:148-149` |
| 5 | **AFLink** | `thrT` (0, 30), `thrS` 75, `thrP` 0.05 | `end2end_model.py:283-293` |
| 6 | scores to rows | same-frame detection with IoU > 0.5 | `end2end_model.py:295-313` |
| 7 | **filter**, in order | 2 frames added · **coverage ≥ 65%** · **gap trim** (gaps > 15% of the span) · interpolate and round · **≥ 10 frames** (not 12: `len(track) ≥ 12` counts 2 non-frame keys) · **range > 15% of (end_fid − begin_fid)** | `gen_labels.py:363-428`, `:626`; `end2end_model.py:319` |
| 8 | naming | argmax of the mean fused 35-vector over scored rows | `gen_labels.py:603-622` |

## 4. Finding: the released SGDet test path takes its frame range from the GT

`data_loading/dataset.py` loads `test_object_trajectories_gt.json` (`:67`, `:70`) and
sets the processed range to its tracks' extents:

```python
120        begin_fid = video_len
121        end_fid = 0
122        for t in self.gt_traj[video_name]:
123            if t['begin_fid'] < begin_fid:
124                begin_fid = t['begin_fid']
125            if t['end_fid'] > end_fid:
126                end_fid = t['end_fid']
```

`models/end2end_model.py` runs the detector only inside that range, and measures the
15% rule against it:

```python
197                begin_fid = input['begin_fid'][0]
198                end_fid = input['end_fid'][0]
...
202                for img_id in range(begin_fid,end_fid):
...
314                    min_l = end_fid - begin_fid
...
319                        if len(trajectory['trajectory']) > 0.15*min_l:
```

Every frame is still CLIP-encoded (`dataset.py:98-119`). Frames outside the range are
encoded but never detected.

**Table 7 — frames outside the GT-extent range.** Computed from the recordings' own
metadata (`video_len`, `begin_fid`, `end_fid`); no label file is read for it.

| videos | with frames outside | share of the video outside: median / p75 / max | frames outside / total |
|---:|---:|---|---|
| 200 | 169 | 4.5% / 13.8% / **79.4%** (`ILSVRC2015_train_00295001`) | **6,069 / 52,104 (11.6%)** |

**What this means.**
- The released SGDet numbers detect objects only within the span where GT objects
  exist. Detections, and so false positives, are impossible in 11.6% of test frames.
- The 15% rule's threshold also shrinks with the GT span.

Every SGDet number in this chain shares this property (session 3, D1, RB1's
baseline). A clean SGDet baseline would take the range from the video's length.

## 5. Step 3 tables (U1 relations, SGDet, novel split; video-bootstrap 95% CIs)

**Table 1 — per stage.** Each cell: count (share, CI).

| unit | n | O1a not detected | O1b below the cut | O2 tracking | O3 filter | O4 kept | S0 no missed side |
|---|---:|---|---|---|---|---|---|
| relation | 262 | 93 (35.5%, 18.9–52.3) | **96 (36.6%, 23.4–50.6)** | 29 (11.1%, 6.2–17.5) | 38 (14.5%, 7.9–23.4) | 0 | 6 (2.3%, 0.3–5.6) |
| side | 332 | **128 (38.6%, 17.9–57.1)** | 106 (31.9%, 20.3–46.5) | 43 (13.0%, 7.4–20.9) | 55 (16.6%, 10.0–25.2) | 0 | — |

Sub-labels:

| sub-label | relations | sides |
|---|---:|---:|
| O2-dedupe | 0 | 0 |
| O2-frag (≥ 2 tracks, each holding ≥ 10% of the GT frames) | 22 (8.4%) | 34 (10.2%): 25 split over 2 tracks, 9 over 3 |
| O2-other, untracked | 0 | 0 |
| O2-other, tracked but off | 7 (2.7%) | 9 (2.7%) |
| O3, by rule | see Table 3 | |

76 of the 262 relations have both sides missed.

**Table 2 — by class.**

| | n | O1a | O1b | O2 | O3 | S0 |
|---|---:|---|---|---|---|---|
| relations: base-only | 201 | 67 (33.3%, 13.5–53.2) | 78 (38.8%, 24.2–55.6) | 21 (10.4%, 5.3–17.2) | 30 (14.9%, 6.9–25.9) | 5 (2.5%) |
| relations: involve novel | 61 | 26 (42.6%, 19.1–56.9) | 18 (29.5%, 15.9–39.3) | 8 (13.1%, 4.9–28.9) | 8 (13.1%, 4.4–28.6) | 1 (1.6%) |
| sides: base | 275 | 110 (40.0%, 15.3–60.8) | 88 (32.0%, 19.2–49.6) | 31 (11.3%, 5.8–20.1) | 46 (16.7%, 10.0–28.2) | — |
| sides: novel | 57 | 18 (31.6%, 8.8–47.5) | 18 (31.6%, 20.0–39.6) | 12 (21.1%, 10.1–38.9) | 9 (15.8%, 5.1–31.9) | — |

O4 is 0 everywhere.

**Table 3 — O3, by the rule that removed the track** (55 of 332 missed sides).

| rule | sides | share of missed sides |
|---|---:|---|
| F-cov: coverage < 65% (`:381`) | 29 | 8.7% (3.3–16.0) |
| F-range: ≤ 15% of the GT-extent range (`end2end_model.py:319`) | 23 | 6.9% (2.7–12.7) |
| F-len: < 10 frames (`:626`) | 0 | 0 |
| F-reshape-trim: kept, but the gap trim pushed it below 0.5 | 2 | 0.6% |
| F-reshape-other | 1 | 0.3% |

**Table 4 — sensitivity to the O1a score floor.** Only raw(s)'s floor changes.

| floor | unit | O1a | O1b | O2 | O3 | S0 |
|---|---|---|---|---|---|---|
| 0.01 | relation | 54 (20.6%, 5.5–39.0) | **135 (51.5%, 36.9–66.2)** | 29 | 38 | 6 |
| **0.05** | relation | 93 (35.5%) | **96 (36.6%)** | 29 | 38 | 6 |
| 0.1 | relation | **122 (46.6%, 30.4–61.3)** | 67 (25.6%, 15.8–37.3) | 29 | 38 | 6 |
| 0.01 | side | 80 (24.1%) | **154 (46.4%)** | 43 | 55 | — |
| 0.05 | side | **128 (38.6%)** | 106 (31.9%) | 43 | 55 | — |
| 0.1 | side | **158 (47.6%)** | 76 (22.9%) | 43 | 55 | — |

**Table 5 — funnel, all 587 test GT object tracks** (whole track, cumulative,
vIoU ≥ 0.5).

| | n | F1 detected (≥ 0.05) | F2 passes the cut | F3 passes the dedupe | F4 tracked | F5 kept |
|---|---:|---|---|---|---|---|
| all | 587 | 94.0% (89.8–96.9) | 86.0% (80.1–90.7) | 86.0% | 74.4% (68.8–79.3) | **71.0% (64.6–76.5)** |
| base | 460 | 93.3% | 86.1% | 86.1% | 76.7% | 73.7% (66.4–80.4) |
| novel | 127 | 96.9% | 85.8% | 85.8% | **66.1%** | **61.4% (52.6–70.3)** |

11 tracks are non-monotone: a later stage passes where an earlier one fails.

**Table 6 — the O1b sides: cut scores of the raw(0.05) oracle's boxes** (A25c,
descriptive). Shares: pooled over all GT frames / median across sides; a frame with
no oracle box counts as below.

| | O1b sides | GT frames | median score: pooled / across sides (IQR) | ≥ 0.1 | ≥ 0.2 | ≥ 0.3 | selected by the cut |
|---|---:|---:|---|---|---|---|---|
| all | 106 | 9,600 | 0.185 / 0.147 (0.121–0.292) | 72.9% / 66.7% | 47.0% / 40.0% | 34.2% / 36.7% | 29.9% / 30.0% |
| base | 88 | 7,215 | 0.199 / 0.157 (0.129–0.321) | 72.4% / 66.7% | 49.8% / 44.7% | 39.1% / 36.7% | 35.2% / 36.4% |
| novel | 18 | 2,385 | 0.169 / 0.124 (0.101–0.185) | 74.4% / 61.7% | 38.6% / 21.7% | 19.4% / 10.0% | 13.9% / 7.5% |

Table 7 is in §4.

## 6. Step 4 — the reading

**By relation, pooled: the largest group is rescore (O1b), 96 of 262 = 36.6% (95% CI
23.4–50.6%) → relation-aware rescoring of detections before the cut. The second is
detector (O1a), 93 = 35.5% (18.9–52.3%), not separated from it (paired 95% CI of the
difference −26.4 to +29.9 points). Link (O2 + O3): 67 = 25.6% (15.3–39.4%).**

**Robustness (descriptive).** The A25 rule is applied as registered. The following
only shows how much the order between the top two can be trusted.
- **The floor.** At 0.01, rescore leads, 51.5% to 20.6%. At 0.1, detector leads,
  46.6% to 25.6% (Table 4).
- **Post-hoc: the non-monotone sides.** 10 sides are labelled O1b although a
  pre-filter track reaches vIoU ≥ 0.5 (item 3 in §7). Labelled instead by their
  track, 8 relations change: link 75 (28.6%), rescore 88 (33.6%), **detector 93
  (35.5%) becomes the largest.**
- **What does not move:** O1a + O1b together, the boxes not available above the cut,
  is 189 of 262 = 72.1% at every floor. Link is about 26–29%.

## 7. Surprises and caveats

1. **"Below the cut" mostly means well below it, not just under.** For the O1b
   sides, the oracle's boxes have a pooled median score of 0.185 (Table 6). In 73%
   of GT frames they score ≥ 0.1, but in only 34% ≥ 0.3.
   - About 30% of the frames' oracle boxes *did* pass the cut. These objects are
     partly detected above the cut, just not over enough frames to reach vIoU 0.5.
   - Novel O1b sides sit lower: the median across sides is 0.124, and the median
     side has only 10% of its frames ≥ 0.3.
   - The O1a/O1b boundary is soft. The O1a sides' raw oracle has a median vIoU of
     0.38 at floor 0.05, and 48 of 128 reach 0.5 at floor 0.01. Much of O1a is
     "detected only below 0.05", not "nothing there".
2. **The ranking of the top two is not robust;** see §6. The robust statement is
   that about 72% of U1 relations are lost because no box above the cut covers the
   object well enough. Tracking and the filter account for about a quarter.
3. **Non-monotone cases.** In 10 O1b sides, all with a cut-oracle vIoU of 0.29–0.49,
   the tracker produced a track ≥ 0.5 by interpolating across the frames where the
   cut had no box. The filter then removed it (F-cov 7, F-range 2), or it survived
   below 0.5 (1). These cases sit on the cut/filter border. A25's order labels them
   by the most upstream stage.
4. **The dedupe costs nothing.** O2-dedupe is 0, and the funnel's F2 equals F3. That
   `unique_boxes` keeps the lower-scored of two near-duplicates is harmless here.
5. **Tracking losses are mostly fragmentation** (34 of 43 O2 sides, split over 2–3
   tracks). None are untracked: every O2 object was at least partly tracked.
6. **The filter's losses are coverage and range, not length.** F-cov has 29 sides,
   F-range 23 and F-len 0. The range rule measures against the GT-extent range (§4).
7. **S0 (6 relations) is mostly identity merging.** Five of the six are same-class
   pairs (zebra–zebra ×3, fox–fox, monkey–monkey), plus one person–dog pair. One
   track covers both GT objects, so no pair of distinct tracks exists. That is a
   tracking failure the per-side labels don't see.
8. **The whole-track view differs from the relation view.** As whole tracks, 71% of
   GT objects survive to a final track (Table 5). Novel objects lose most at
   tracking: F3 → F4 drops 19.7 points, against 9.4 for base.
   - U1 sides are evaluated over the relation's extent (median 45 frames), and
     there the losses are mostly before tracking.
   - Short relation spans are where a frame-level detector misses most.
9. **U1 is mostly base objects** (201 of 262). Novel sides are relatively more
   often lost in tracking (O2 21%, against 11% for base).
10. **The released test path takes its frame range from the GT** (§4). 11.6% of test
    frames are never detected. This affects every SGDet number in the chain, the
    baseline's included.
11. **Execution caveats,** all disclosed in §2:
    - the test annotations were on the rented machine, because the released test
      path reads them; they were deleted after the run, which differs from M1's
      A24h rule;
    - the abandoned two-process attempt;
    - `cleanup.sh`'s early exit after the deletion;
    - the code root was a `git archive` copy without `.git`, checked by hash.
12. **Test labels were read** for this analysis only, and this is disclosed. Nothing
    was tuned or chosen from them. The only post-hoc analysis is the relabelling in
    §6, which is descriptive only.

## 8. Hashes of all outputs (paths under `studies/d2/`; `data/` is git-ignored)

| file | SHA256 |
|---|---|
| `data/rec/manifest.jsonl` (200 recordings) | `37d26c1c59bb1f3befafde0c4dab5d532a401987933c2748be80245b2adc28c5` |
| `logs/rec_sha256.txt` (all 404 recording files) | `1328b7c9076cf2a20952293d5aa94a04be3e464574d16cff1ccf8861d5def662` |
| `data/d2_relations.json` (every U1 relation, both sides, all quantities) | `90897c6f5077265169d4a252593b68a09722a7c215d63a5d289e61d080a31a72` |
| `data/d2_funnel.json` | `5b170d82c3652377c514e777d27ef5182a223f75176efb26d889734043e95778` |
| `data/d2_tables.json` (Tables 1–7, the reading) | `c957eeab76fe45c96b81ba86b82e5a3472e2fd8338a0da10d96d7780fcdb770e` |
| `data/d2_tables.md` | `49c11dc2706b36f62343f09c542e916fb304b71845f004c0e7e221bc4e5f8bd5` |
| `logs/d2_analyse.json` (environment, code and input hashes, outputs) | `246c0b39cc0bab0f8222795dcaa02c1500965ac3ebffff0ea69772ade21d67dc` |
| `logs/d2_analyse_stdout.txt` | `2e934f6e91b0a86e376f1520947122c4cdd250d75c29cf369662f6fcd7ffd22c` |
| `logs/box/fetch.log` (token handling, fetch checks) | `21c6183d9cd99b9f8d6ec869c66c47e4be3fe8211a6245633ddb70ba322c83ac` |
| `logs/box/token_scans.log` | `1e5e0454e7c3cc50f7b468f0f4b3ffba691a58af125bb35aa9a737a23e375c7a` |
| `logs/box/setup.log` | `1ffaf1280421c50408cb0bdf14f4069ff2f15a4ed45d5ea8092f81a941940dcd` |
| `logs/box/cleanup.log` / `deleted_test_annotations.sha256` | `dfc2a563bd1df4475338dab3592fe27e01e72431f060bdd11913648dafd46d7c` / `9c3d8941caad13dcaecfc0242abc78700169d9860445b45e27bb99b09f1cccec` |
| `logs/box/run_first_last.log` / `run_full.log` / `run_full2.log` | `a0694564…b7a` / `1ac18e05…3b103` / `84f549bd…09b30a` |
| `logs/box/notes.log` (the parallel attempt) / `run_p1.log` = `run_p2.log` | `a260e17b…4a16` / `6f6522f1…6d2af` |
| `logs/box/frames_meta/SHARDS.json` / `MANIFEST.json` | `86439644…70988` / `f7e2e9e0…2efb` |
| `logs/code_pins.json` (A25c) | `95816845ec3a367ed890796f5018421253a6edf0e78b63a891852c2420b09028` |

**Stop.** D2 is finished. Nothing is built, and the next step is the plan author's
call. The machine can be destroyed now.
