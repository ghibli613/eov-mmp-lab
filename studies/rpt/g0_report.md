# RPT G0 report: SAM 3 set up, code checks, adapter, reproduction, smoke test

2026-10-26 · repo `ov-vidvrd-lab`, commits `5dea6bd`…`b249c1d` (and the one holding this report) · chain head `0d62e838…947b` · nothing pushed

## Verdict

**G0 passes all three of A28's bars.**

| bar (A28) | result |
|---|---|
| 1. HELD GT tracks through the adapter reproduce `cache_s3` (residual ≤ 1e-4) | **PASS, bit for bit**: max \|Δlogit\| = 0.0 on all 5 videos |
| 2. All adapter unit tests pass | **PASS**: 35 of 35 |
| 3. 3 HELD videos run end to end (SAM 3 → adapter → modelC → evaluator) | **PASS**: no error |

A pass allows the G1 amendment to be written; it is not evidence for the method. Four findings matter for G1 (§9, §10):

1. **Runtime.** At SAM 3's defaults, G1 would take about **43 GPU-hours**, **5.4×** G1's bar of 8 GPU-hours. Reusing the backbone across prompts is **bitwise identical** and **1.36×** faster, which brings it to about 32 GPU-hours, still 4×.
2. **GPU memory.** SAM 3 keeps frames on the GPU, about 6.3 MiB per frame. The longest HELD video (2,947 frames) would need about **25 GiB**, which a 24 GB RTX 4090 cannot hold at default settings.
3. **The candidate tier is empty.** Every masklet scores above 0.5. As decided for D1, raw detections in [0.1, 0.5) are recorded; there are many (7,473 on the 3 videos), mostly duplicates around found objects.
4. **An erratum in A28** (no rule changes): masklets do *not* always start at ≥ 0.7. On a frame with no existing masklet, such as frame 0 of every prompt, SAM 3 starts one for every kept detection above 0.5 (`sam3_video_base.py:1522-1535`). Two of the 8 confident tracks start at 0.594 and 0.668.

## 1. Step 1: paths and chain (all as expected; nothing differed)

- **The authors' files** are unchanged from `fdeeeb4`. These are every path tracked at `fdeeeb4` except `pilot_analysis/`; `git diff fdeeeb4` over them is 0 lines, in both HEAD and the working tree.
- **Present:**
  - `pilot_analysis/stage1c/cache_wrapper.py`, `gt_tracks.py`, `pilot_analysis/stage1/train_split.json`;
  - `cache_s3`, at `../session3_box_copy/cache_s3`: 800 train and 200 test tensors. Test files were listed by name only, never opened.
  - The 5 pinned train tensors match `SHA256SUMS_session3.txt` (`2cc2d071…954b`) and session 3's manifest.
- **The chain head after A26m** was `ac77e243…a591`, checked with `studies/migration/verify_after_A26m.py`: all checks passed. The ov-pairtime check was missing from the output, so it had been skipped silently.
- **A26's tests:** `python -m pytest -p paths a26/tests` gave **30 passed**, run from **`studies/`**, where the `paths` plugin resolves.
- **The bundle:** `~/Documents/thesis/archive/ov-pairtime-2026-10-24-a820a0e.bundle` exists (`eb7417df…85a7`, master `a820a0e`).

Evidence: `studies/rpt/g0/logs/step1_checks.txt`.

## 2. SAM 3 licence

The SAM License, last updated 2025-11-19. `LICENSE` SHA256 `4dea99bf…e1be`, unchanged since SAM 3's first commit. The gated Hugging Face `LICENSE` is the same text without its final newline, as shown by its git blob id, `16ee5f63…`.

- **Research use: allowed.** §1.a: *"You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta's intellectual property or other rights owned by Meta embodied in the SAM Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the SAM Materials."*
- **Publication: allowed, with acknowledgement.** §1.b.ii: *"If you submit for publication the results of research you perform on, using, or otherwise in connection with SAM Materials, you must acknowledge the use of SAM Materials in your publication."*
- **Outputs such as tracks: no restriction.** "SAM Materials" means the models, software and documentation; outputs appear only in §3 (warranty disclaimer) and §5.b (litigation). Two cautions:
  - §1.b.iii requires compliance with applicable law, including privacy and data protection;
  - if tracks are ever released, shipping a copy of the licence with them covers the stricter reading of §1.b.i.

## 3. The chain checker

- **What was wrong.** The ov-pairtime check ran only `if os.path.isfile(...)`. After the move, it disappeared without a word: 65 checks instead of 66, and the checker still exited 0.
- **The fix**, in `studies/migration/verify_after_A26m.py` and nothing else. It reads `prereg/preregistration.md` at `a820a0e` from the bundle, through a throwaway bare repository, and requires `981bf067…5ce5`. A missing bundle, one git cannot read, or a different hash is a FAIL with exit 1; the check is never skipped.
- **New SHA256: `3af183caad77d61ea4aca7e7fa04d2da74c354b0d4335797ede98409d0eda1e7`.**
- **Tests** (`studies/rpt/tests/test_chain_checker.py`, 7, all pass):
  - present and matching (a synthetic bundle, and the real one);
  - missing;
  - altered;
  - not a bundle;
  - a bundle without `a820a0e`;
  - `main()` exits 1 with a FAIL line when the bundle is moved away.
- **On the real chain:** 66 checks before A28, 69 after A28, **72** after A28a. Logs: `studies/rpt/g0/logs/verify_after_A26m_fixed.txt`, `studies/rpt/logs/verify_after_A28{,a}.txt`.

## 4. The five code checks (A28 has every file:line)

1. **Gap filling** (`models/gen_labels.py`). The released code:
   - adds 2 frames before the first box, copies of it (`:363-371`);
   - drops a track under 65% coverage (`:380-382`);
   - trims the shorter side of any gap longer than 15% of the span (`:385-413`);
   - fills the remaining internal gaps by **linear interpolation per coordinate**, then rounds and clamps every box (`:416-423`, `round_and_positive` `:426-428`);
   - adds nothing after the last box.

   **Adapter:** fills internal gaps with `:416-423` exactly; no pre-roll, no coverage rule, trim or length filter (rule 8 is replaced; D4).
2. **Score field.**
   - SAM 3 thresholds the detection probability `sigmoid(pred_logits)`. That is **already presence × object**: `pred_logits` = logit(σ(class)·σ(presence)), clamped ±10 (`sam3_image.py:345-353`).
   - The output field is `out_probs` = `obj_id_to_score`: the probability of the detection that started the masklet, set once and never updated (`sam3_video_base.py:1059-1064`; `sam3_video_inference.py:409`, `:450-452`).
   - Frames can be marked detected, propagated or absent, not from the output but from SAM 3's own association. A read-only wrapper records it (`sam3_video_base.py:2011-2017` defines "detected" the same way).
   - **Adapter (D2):** the track score is the mean of `out_probs` over detected frames, which equals the starting detection's probability.
3. **Clip grid: per pair, not per video.** Clips start at each ordered pair's overlap start (`gen_labels.py:657`, `:661-666`, `:693`), and features are read at clip middle frames (`:81-82`). Middle frames are shared only when overlap starts agree modulo 30.
   - **HELD, GT tracks:** 628 of 47,038 frames are read (1.3%).
   - **Cache size per video:** a median of 2.5 MB (max 17.7 MB) when only those frames are cached, against 145 MB (max 2.49 GB) for every frame.
   - **Smoke test, SAM 3 tracks:** 5, 3 and 0 frames read.
4. **Window joining: yes.** Session 3's `run_pair` concatenates the windows' per-clip outputs (`cache_wrapper.py:166-180`), and `eval_from_cache.predict` merges each pair's joined sequence (`:72-77`). No real pair ever exceeded 40 clips; the released SGDet path does not window at all.
5. **Confident cut: 0.5.** Sources:
   - `score_threshold_detection` = 0.5, documented "only keep detections above this threshold" (`sam3_video_base.py:270-272`, `:608`; `model_builder.py:747`);
   - `Sam3Processor(confidence_threshold=0.5)` (`sam3_image_processor.py:17`, `:199`).

   `new_det_thresh` = 0.7 starts masklets, but only on frames that already have masklets (the erratum, §10).

## 5. The amendments

| | label | text SHA256 | chain head after |
|---|---|---|---|
| G0 rules | **A28** (A27 is reserved for tracklet keeping, `agreed_plan.md`) | `363eecc3ddba8d9e6e3eaf284e794ee4706302d2f50a4dddb693af49631402b3` | `b94e198640df54184aecdb25c3c7c5962fb34650e6273b08c534c926cf5dcc9c` |
| checkpoint SHA256 | **A28a** | `fb399ae34cc421a785d41a0358ab13ea0f54496323e54d74d3c67534bab7aeea` | **`0d62e8380a8b8a9e2361717fc210a351fe4f3aaf4d2119a44c946315d753947b`** |

Following your answers to interim report 1, A28 pinned the checkpoint by revision, size and LFS-pointer blob, because Hugging Face hides the SHA256 without access. A28a then pinned `sam3.pt` = `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, and the pointer rebuilt from that hash matches the blob A28 pinned.

Appends log: `studies/rpt/logs/prereg_log.txt` (`54005a42…2c6c`). Backups sit beside it, and the checker verifies them.

## 6. SAM 3 environment, laptop and machine

- **SAM 3:** `facebookresearch/sam3` at **`f1647e8d0c1d280991f4d1b7d974c2704537fe66`**, never edited. Checkpoint `facebook/sam3` revision `3c879f39…`: **SAM 3, not SAM 3.1** (added 2026-03-27). Built with `build_sam3_video_predictor()` at every default.
- **Laptop env** `sam3`: conda-forge Python 3.12.15, torch 2.10.0+cu128, `-e ".[notebooks]"`, psutil, setuptools 78.1.0. Two deviations from a plain install:
  - conda-forge instead of the default channels, which need a ToS acceptance that is the author's to give;
  - setuptools 78.1.0, because SAM 3 imports `pkg_resources`, which setuptools 84 removed.
- **Laptop (GTX 1650, 4 GB, 7.7 GB RAM): SAM 3 does not run at its default settings.**
  - A random-weight probe found 860,055,224 fp32 parameters (3.21 GiB). Moving them to the GPU succeeded only by spilling into WSL shared memory.
  - The real check (`…00010024`, "bicycle") never finished loading the model; the author's WSL session went down.
  - No frame was processed, so peak memory and time per frame could not be measured. It was not retried, since it takes the session down.

  Logs: `studies/rpt/g0/logs/laptop_{memory_probe,check}.txt`.
- **Machine** (approved; one RTX 4090; Vast.ai PyTorch image, Ubuntu 24.04.5, torch 2.11.0+cu128, driver 580.142, 251 GB RAM):
  - setup began 14:25 UTC and all work was done by 15:51 UTC (86 minutes, under the 2-hour budget);
  - the author fetched the weights interactively;
  - token scans showed `files=0 process_environments=0 token_files=0` at every point;
  - no test file was ever on the machine.

  `pip freeze` for both environments, and every machine log: `studies/rpt/g0/logs/box/`.

## 7. The adapter as implemented, and its tests

**Code:** `studies/rpt/adapter.py`, `relation.py`, `sam3_runner.py`, `no_test_labels.py`.

**SAM 3 side:**
- 35 prompts in `object2id` order, `_` replaced by a space, nothing else;
- SAM 3's own multi-prompt protocol, at defaults;
- a read-only recorder for each frame's status, raw [0.1, 0.5) detections and tracker scores.

**EOV-MMP side:**
- one SAM 3 ID is one track;
- the tight mask box, with SAM 3's own box kept beside it;
- internal gaps per check 1;
- per-frame status: detected, propagated or filled;
- the score per D2; tiers at 0.5 and 0.1;
- cross-prompt duplicates at mean mask IoU > 0.7, keeping the higher score (D3);
- `gen_feats_test` for the pairs; session 3's `run_pair` for windows and joining;
- CLIP and TagCLIP on demand per requested frame, memoized, batch 1, fp16, whole video;
- gap-exposure flags;
- an audit hook that refuses to open any test label.

**Unit tests: 35 passed** (`studies/rpt/g0/logs/pytest_rpt.txt`).
- Every case the G0 prompt lists. Internal gaps match the released `interpolate_and_adjust_frames` exactly where its own drop and trim rules do not fire.
- A 95-clip pair gives windows 40/40/15, and the joined output yields one relation over clips 38–42, where window by window would give two.
- Further tests cover the recorder, the raw-detection capture before NMS, the backbone cache, the comparison script, the runner's records and the test-label guard.

## 8. Step 8: reproduction of session 3 — **PASS**

| video | pairs | clips | max \|Δlogit\| | max \|Δint\| | slot inputs | frames read; `global_proj` identical |
|---|---:|---:|---:|---:|---|---|
| `…00008005` | 6 | 18 | 0.0 | 0.0 | 0.0 | 3; 3 |
| `…00010024` | 2 | 4 | 0.0 | 0.0 | 0.0 | 2; 2 |
| `…00024005` | 8 | 30 | 0.0 | 0.0 | 0.0 | 15; 15 |
| `…00025029` | 2 | 2 | 0.0 | 0.0 | 0.0 | 1; 1 |
| `…00027003` | 2 | 4 | 0.0 | 0.0 | 0.0 | 2; 2 |

Pairs, order and windows are identical too. The run took 12 s of GPU time. `studies/rpt/g0/logs/step8_compare.{txt,json}`.

The earlier laptop dry run (GTX 1650; declared, does not count) differed only in fp16 CLIP features (max 0.011), giving a 0.016 logit residual. The bit-exact 4090 run shows the cause was the GPU, not the frames or the code.

## 9. Step 9: smoke test, G1 cost, images, measurement

| | `…00008005` | `…00010024` | `…00024005` |
|---|---:|---:|---:|
| frames | 130 | 64 | 981 |
| GT objects (training annotation) | 3 cattle | 1 bicycle, 1 person | 2 watercraft |
| SAM 3 time, total (s) | 423 | 209 | 3,243 |
| per prompt: no masklet / with masklets (s per frame) | 0.090 / 0.177 | 0.089 / 0.178, 0.137 | 0.094 / 0.118 |
| slowest prompt (s) | cattle 23.0 | bicycle 11.4 | watercraft 115.3 |
| peak GPU allocated / reserved (GiB) | 7.3 / 7.6 | 6.3 / 6.6 | 12.8 / 14.0 |
| confident tracks, by class (score) | cattle ×3 (0.92, 0.96, 0.74) | bicycle ×2 (0.91, 0.71), person ×2 (0.93, 0.59) | watercraft ×1 (0.67) |
| candidate / dropped / no detected frame | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| removed by SAM 3 (hotstart) / duplicates | 0 / 0 | 0 / 0 | 0 / 0 |
| raw [0.1, 0.5) detections, by class | cattle 3,321 | person 619, bicycle 459, car 45, frisbee 10 | watercraft 2,969, person 49, bird 1 |
| ordered pairs / clips / pairs > 40 clips | 6 / 30 / 0 | 12 / 18 / 0 | 0 / 0 / 0 |
| middle frames read / in filled gaps | 5 / 0 | 3 / 0 | 0 / 0 |
| filled frames / frames with several detections | 0 / 1 | 0 / 0 | 0 / 0 |
| relation side (s) | 10.5 (incl. model warm-up) | 1.0 | 0.03 |

What the table shows:
- **SAM 3 finds the annotated objects**, plus extras. The bicycle video gets 2 bicycles and 2 persons for 1 of each; the images show a distant, unannotated person.
- **One annotated object is missed.** The boat video gets 1 watercraft for 2: SAM 3's single masklet covers the yacht, and the second, smaller GT watercraft has no confident masklet. The 2,969 raw watercraft detections show where it went. With only 1 track, no pair could form there.
- **Gaps are rare here.** SAM 3 kept every confident masklet unbroken, so no frame was filled.
- **Most raw low detections sit on found objects.** On average 26 per frame, they are queries around objects SAM 3 already found, so G1's "a missed GT object has such a detection" needs a careful definition (G1 draft).

**Evaluator — smoke test, in-sample, not a result.** Base predicates, the 3 videos' HELD GT (68 relations): mAP 11.11, R@50 2.94, R@100 2.94. The boat video has no predictions, because SAM 3 found no pair there. These numbers are not read, compared or used.

**G1 cost, extrapolated to the 160 HELD videos** (47,038 frames, 35 prompts; `logs/step9_smoke.json`):

| | GPU-hours | × G1 bar (8 GPU-hours) |
|---|---:|---:|
| SAM 3 at defaults (0.094 s per frame per prompt) | 43.1 | 5.4× |
| + relation side | 0.17 | |
| with the backbone reused (1.36×, below) | about 31.7 | about 4.0× |
| needed for the bar | 0.017 s per frame per prompt | |

At $0.35/h, 43 GPU-hours is about $15.

**Images:** 18 in `studies/rpt/g0/viz/`, 6 evenly spaced frames per video, GT in green and SAM 3's confident tracks in colour. The cattle and bicycle boxes fit the GT closely; on the boat video SAM 3's box matches the large GT box.

**Runtime measurement** (A28; smoke video 1; `logs/measure_{reuse,control}.json`):

| comparison | prompts | bitwise identical | time |
|---|---:|---|---|
| default vs backbone reused | 35 | **yes**: masks, boxes, scores, statuses, raw detections | 423 s → 311 s, **1.36×** |
| default vs default again (control) | 3 | **yes** | 39 s vs 36 s |

- The cache served 4,420 hits from 130 computed frames, and held 13.6 GiB in CPU RAM.
- SAM 3 is deterministic here, so the reuse changes nothing.
- The backbone is only about a quarter of the time; the per-prompt detector and tracker dominate.

## 10. Where the proposal or A28 could not be followed as written

1. **The candidate tier is empty at SAM 3's defaults.** Reported in interim report 1. By your answer (D1), raw [0.1, 0.5) detections are recorded read-only. The candidate tier and Verify go to the G2 amendment.
2. **Erratum in A28's text: the "≥ 0.7".** A28 states twice (rules 2 and 5) that every masklet starts at ≥ 0.7. That is true only on frames that already hold a masklet. With none, SAM 3 starts one for every kept detection above 0.5 (`sam3_video_base.py:1522-1535`). Observed: starting scores of 0.594 and 0.668 on frame 0.
   - No rule changes: the score rule is unchanged, and every score is still above 0.5, so the tiers are unchanged.
   - The correction belongs in the G1 amendment.
3. **D2–D4 ambiguities** (score reading, duplicate "merge", pre-roll). Reported in interim report 1 and settled by your answers before A28.
4. **The SAM 3 checkpoint hash** could not be pinned before access was granted. Split into A28 and A28a, as you approved.
5. **The laptop check produced no memory or speed figure**, because the session was lost during model load. The result is "does not run", as you anticipated.
6. **Installation:** conda-forge and the setuptools pin (§6). These do not touch SAM 3's computation; the 4090 environment used SAM 3's README versions.
7. **Weights to the machine:** fetched by the author's interactive token script. Hashes match A23, A28a and CLIP's URL.

Nothing else departed from A28.

## 11. Hashes, status, log

- **All inputs and outputs:** `studies/rpt/g0/logs/sha256_inputs_outputs.txt` (88 lines; tensors that git ignores as `*.pt` are listed there). Key entries:

| file | SHA256 |
|---|---|
| end-to-end checkpoint (A23) | `d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70` |
| CLIP ViT-L/14@336px | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| `sam3.pt` (A28a) | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| `sam3_out/…00008005.json.gz` | `aed5945d1973b8de25876a7069fd5fcfa92ddd271f97ca24abf48380ca705db4` |
| `sam3_out/…00010024.json.gz` | `a2929790527dea7165b1abf3a59ac4d314bd16389799e5743ecf6faf8fdddf3f` |
| `sam3_out/…00024005.json.gz` | `4b963ccb20abbcf2e08b1df3940c14d042ddfd54ca2028d252309f219711b546` |
| `box_out/step8/manifest.json` | `521794d8da08a70af4c6b8f577cf73024ec41c6e9327dd91e4600757c0a99c88` |
| `box_out/step9/manifest.json` | `9cee56c4955b3bf211f326f17d89c8076d9e1d816dbf8a676d96582b313dcad2` |
| `logs/step8_compare.json` | `be42fd854467fc157036aac27d61ebbacb1c044aa0afd5d09d2e3f7a1c3589f1` |
| `logs/step9_smoke.json` | `77b269aab1d2ff7550208b07b722fa5d17abac04a05655d2359f463b0e5863df` |
| `logs/measure_reuse.json` | `f305be79c9127c566c8ca788ce79032f433a74c710973ee085c62da79e10388a` |
| `held_gt_smoke3.json` (subset of `held_gt.json` `78d55c69…4dee`) | `8dab2c87a179c82fffade37b57bd6dd9df37fb6d8e2233c3160e0e85a1a85f49` |

- **Commits since the chain head `ac77e243…`** (repo commit `bc9a87b`), local only:

```
b249c1d RPT G0: steps 8 and 9 on the RTX 4090, the measurement, and the laptop evaluation
7189a9c RPT G0: laptop check — SAM 3 at defaults does not run on the laptop (fails at model load, session lost)
829e7bd RPT G0: A28a pins SAM 3's checkpoint SHA256 (chain head 0d62e838…947b)
de88f3f RPT G0: unit tests (35 passed), step 8 dry run on the laptop (does not count), interim report 2
5dea6bd RPT G0: A28 appended (chain head b94e1986…dcc9c); chain checker reads ov-pairtime from the bundle; adapter, tests and machine scripts (not run)
```

- **`git status`:** clean after the commit holding this report and the G1 draft.
- **SAM 3 commit:** `f1647e8d0c1d280991f4d1b7d974c2704537fe66`.
- **Seeds:** SAM 3 `torch.manual_seed(0)` (no random draws). The EOV-MMP side sets none, as session 3's `cache_wrapper` did.

**Stop.** No G1, no loop code, no test set.
