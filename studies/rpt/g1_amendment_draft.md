# DRAFT — not hashed, not in the chain — for review

#### A29 (draft): RPT G1 — SAM 3's tracks against EOV-MMP's on the 160 HELD videos

Label: A29 would be the next free one. A27 is reserved; A28 and A28a are G0's.

Written after G0 (`studies/rpt/g0_report.md`), before any G1 run. G1 measures the tracker only: no relation score on HELD is read, because EOV-MMP was trained on HELD. **Items marked OPEN are not settled by the proposal; each carries a recommendation for the reviewer.**

##### Fixed from the proposal (§6, version 5), verbatim in substance

| bar | rule |
|---|---|
| **Recall** | GT track recall at vIoU 0.5 is at least **10 points** above EOV-MMP's, overall and on novel object classes (when ≥ 30 instances; OPEN 6). Reported for the confident tier alone, and for confident plus candidates separately (the candidate tier is empty at SAM 3's defaults; G0 §10.1). |
| **Pair recall** | Recall of GT relation pairs is at least **10 points** above EOV-MMP's. |
| **False-track burden** | Unmatched confident tracks per video are at most **1.5×** EOV-MMP's. Also reported: tracks, unmatched tracks and candidate pairs per video, for SAM 3's confident tier and for EOV-MMP. |
| **Box style** | Median box IoU of matched tracks ≥ **0.70**, plus GT and SAM 3 boxes drawn on about 20 HELD frames for a visual check. |
| **Box sensitivity** (secondary) | On HELD base-predicate GT relations that both trackers match, the GT predicate's **top-5** hit rate under three box sources: GT, SAM 3, and EOV-MMP's detector boxes. Pass if SAM 3's boxes are no worse than the detector's by more than **5 points**. |
| **Gap exposure** | Reported: how often a clip's middle frame falls inside a filled gap. |
| **Track counts by *N*** | Reported: tracks per video, binned by the confident track count *N*. |
| **Runtime → cost cap** (replaced; reviewer's answer to `studies/rpt/g0/runtime_proposal.md`) | The proposal's bar was ≤ 1 GPU-hour per 20 videos, with the every-*k*-frames fallback. G0 measured SAM 3 at defaults 5.4× over it. **Replaced by a cost cap of 35 GPU-hours for SAM 3 on all 160 videos, at full SAM 3 fidelity.** The every-*k* fallback (B1) and options B2–B5 are **not used** in G1. **Stop rule:** after the first 20 videos, if the projection for all 160 exceeds 35 GPU-hours, stop and ask the author. |

A G1 fail stops the line (proposal §6, fallback 2). A pass only allows G2.

##### Settled by the reviewer (answers to `studies/rpt/g0/runtime_proposal.md`)

- **A1, backbone reuse across prompts,** and **A2, 2–3 SAM 3 processes per GPU on different videos:** accepted as implementation details. Each is used only after it proves bitwise identical to G0's outputs (below).
- **A3, `torch.compile`** (SAM 3's own `compile=True` option): measured once. Adopted only if bitwise identical; otherwise the size of the difference is reported and it is left out. Never used as a setting change.
- **`offload_video_to_cpu=True`** (SAM 3's own session option; formerly OPEN 2): adopted only if bitwise identical. It is needed for the 2,947-frame video, which does not fit a 24 GB GPU at defaults.
- **B3 (one generic prompt) is not measured now.** B1–B5 are not used in G1.
- **No separate measurement session:** the checks run in G1's first machine hour.

##### G1's first machine hour: the checks, in order

All on smoke video 1, `ILSVRC2015_train_00008005`. Each compares, with `studies/rpt/g0/sam3_compare.py`, against G0's default output `studies/rpt/g0/sam3_out/ILSVRC2015_train_00008005.json.gz` (SHA256 `aed5945d…05db4`): every prompt, every masklet, every per-frame field (mask RLE, boxes, `out_probs`, status, recorded scores) and the raw [0.1, 0.5) detections. "Pass" means bitwise identical.

| # | run | if it passes | if it does not |
|---|---|---|---|
| 0 | **The default run**, exactly as G0 | the machine reproduces G0; the checks can be judged | **stop and ask the author**: a new stack or GPU cannot be checked against G0 |
| 1 | `offload_video_to_cpu=True` | used for all 160 videos | not used; the 2,947-frame video `…00280000` cannot run on this GPU, so **stop before that video and ask the author** (e.g. a 48 GB GPU for it alone, at defaults) |
| 2 | + A1, backbone reuse, on top of every option passed so far | used, under the scheduling rule below | not used |
| 3 | + A2: the same video in 2, then 3, concurrent processes, with every option passed so far; each process's output is compared, and the throughput measured | the largest passing count is used | one process per GPU |
| 4 | + A3, `torch.compile`, on top of everything that passed | used | not used; the size of the difference is reported: max \|Δ\| of `out_probs` and box coordinates, mask IoU, and masklets added or lost |

- **On a mismatch,** that option is dropped and G1 continues with what passed, except at check 0 (stop) and check 1 (stop before the long video). Every check's outcome, time and memory go in the G1 report.
- **Then the first 20 videos** run with the passing options.
- **Scheduling** (identical output; fixed now, not tuned). A1 keeps about 106 MB of CPU RAM per frame. Videos over 1,000 frames therefore run without A1. With A1, concurrent processes share the machine's RAM, holding at most about 150 GB of cache in total. The machine has ≥ 192 GB RAM.

##### Values G0 found (fixed; A28, A28a, G0 report)

- **SAM 3:**
  - code `f1647e8d0c1d280991f4d1b7d974c2704537fe66`;
  - `sam3.pt` `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` (`facebook/sam3` revision `3c879f39…`, not SAM 3.1);
  - `build_sam3_video_predictor()`, every default;
  - SAM 3's own multi-prompt protocol; 35 prompts, `_` replaced by a space.
- **Tiers:** confident cut **0.5** (SAM 3's `score_threshold_detection`); candidate floor **0.1**.
- **Track score:** the mean of `out_probs` over detected frames, which equals the starting detection's probability.
  - **Correction to A28:** that probability is **> 0.5**, not ≥ 0.7. On a frame with no existing masklet, SAM 3 starts one for every kept detection (`sam3_video_base.py:1522-1535`). G0 observed 0.594 and 0.668.
  - No rule changes.
- **Gaps and duplicates:**
  - one SAM 3 ID is one track;
  - internal gaps are filled by `gen_labels.py:416-423`; no pre-roll;
  - cross-prompt duplicates at mean mask IoU > 0.7 keep the higher-scoring masklet unchanged.
- **Relation side and recording:**
  - pairs, windows and features as A28, the adapter as committed (`studies/rpt/adapter.py`, `relation.py`, `sam3_runner.py` at `b249c1d`);
  - raw [0.1, 0.5) detections are recorded read-only.
- **Measured on a 4090:**
  - **Speed:** 0.094 s per frame per prompt at defaults (0.09 when a prompt finds nothing, 0.12–0.18 when it does).
  - **Memory:** about 6.3 MiB of GPU per frame, on top of about 6.5 GiB.
  - **Backbone reuse:** bitwise identical, 1.36× faster.
  - **Determinism:** a rerun is bitwise identical.
- **HELD:** 160 videos, 47,038 frames; 472 GT objects, 348 base and 124 novel. The novel ones by class: antelope 29, cattle 23, horse 20, bear 13, watercraft 11, giant panda 10, motorcycle 7, fox 7, lizard 2, red panda 2. Base GT relations in `held_gt.json`: 4,365 (157 videos).

##### Videos

The 160 HELD videos of `pilot_analysis/stage1/train_split.json` (SHA256 `095dd89c…1f1a`; seed 20260928), sorted. All 160 are used: RPT needs no ECC, unlike stage 2's 157. A bare number below is `ILSVRC2015_train_` + that number (142 videos); the last 18 are ImageNet-VID `val` videos that belong to VidVRD's training split, and are written in full:

```
00008005 00010024 00024005 00025029 00027003 00032000 00040000 00040005
00040030 00050006 00051015 00051016 00052001 00055002 00058003 00060001
00060005 00065002 00065003 00069006 00071013 00071014 00071019 00073037
00074010 00077006 00078000 00097003 00112000 00116002 00119013 00119025
00125006 00125007 00125013 00125014 00135004 00139004 00146005 00149004
00154002 00159010 00161016 00165000 00171001 00171004 00172007 00175001
00181011 00185000 00209003 00209005 00215000 00215006 00215010 00218002
00225001 00227012 00228000 00234001 00234007 00235006 00235007 00237001
00246002 00253025 00265004 00265042 00265063 00265066 00265068 00265074
00270000 00272005 00272034 00280000 00284001 00290008 00290019 00293004
00293005 00300008 00302021 00303008 00308017 00309001 00311002 00339001
00339003 00349000 00353000 00354000 00356001 00371013 00374000 00398004
00407001 00415007 00415008 00435001 00467000 00476002 00477000 00480000
00494000 00532000 00533003 00572000 00588000 00596000 00632001 00636001
00667002 00679000 00683000 00691000 00721000 00750000 00756000 00759000
00760001 00766000 00772004 00780000 00796000 00873000 00882000 00896000
00897005 00912002 00917000 00971000 00978000 00987000 00988001 00989000
01014000 01020000 01025001 01034001 01079000 01108001 ILSVRC2015_val_00007022 ILSVRC2015_val_00008001
ILSVRC2015_val_00020001 ILSVRC2015_val_00026003 ILSVRC2015_val_00027000 ILSVRC2015_val_00028003 ILSVRC2015_val_00035000 ILSVRC2015_val_00035008 ILSVRC2015_val_00035013 ILSVRC2015_val_00037004
ILSVRC2015_val_00046000 ILSVRC2015_val_00078000 ILSVRC2015_val_00082000 ILSVRC2015_val_00103000 ILSVRC2015_val_00123004 ILSVRC2015_val_00135000 ILSVRC2015_val_00150003 ILSVRC2015_val_00169000
```

The "first 20" of the plan are the first 20 IDs above (OPEN 9).

##### Seeds

- SAM 3: `torch.manual_seed(0)`; its inference draws no random numbers, and G0's rerun was bitwise identical.
- The HELD split: seed 20260928 (fixed, stage 1).
- Bootstrap CIs, descriptive only, never a bar: `numpy.random.default_rng(0)`, 1,000 resamples over the 160 videos, the same draws for SAM 3 and EOV-MMP.
- No other randomness.

##### OPEN, each with a recommendation

1. **Runtime — SETTLED** (cost cap of 35 GPU-hours, the stop after 20 videos, A1–A3 as above). Kept here so the numbers stay with it:
   - at defaults G0 projects 43 GPU-hours, over the cap, so G1 needs A1 and/or A2 to pass their checks;
   - with A1 alone, about 32 GPU-hours;
   - A2's gain is unmeasured.
2. **GPU memory — SETTLED** (`offload_video_to_cpu=True`, check 1 above). The numbers: the longest HELD video, `…00280000` (2,947 frames), needs about 25 GiB at defaults; the next longest (1,470 frames) about 16 GiB.
3. **EOV-MMP's tracks on HELD** (every comparison needs them).
   - **Recommendation:** the released front end over the **whole video** (A26's R0w setting, `studies/a26/wrappers.py::whole_video`) as the matched comparison, since SAM 3 searches the whole video. The GT-range front end (R0) is reported beside it.
   - **Note:** EOV-MMP's detector was trained on HELD, so its HELD recall is inflated. That makes G1 conservative for SAM 3; it is disclosed.
   - **Cost:** about 3 GPU-hours (session 3's SGDet ran 200 videos in 3.5 h).
4. **Matching a track to a GT object.**
   - **Recommendation:**
     - GT objects are the annotated tids, whole, over their annotated frames; not session 3's option-B runs.
     - vIoU is the official `viou` (`inference/video_relation_detection_openvoc.py:73`).
     - The match is **class-aware for the bars** (a wrongly named track cannot give the right triplet; U2); class-agnostic recall is reported beside it.
     - Each GT object is matched by at most one track, greedily in descending vIoU.
5. **Pair recall.**
   - **Recommendation:**
     - The unit is a HELD **base** relation instance (`held_gt.json`; novel-predicate instances stay unused).
     - It is recalled if some ordered pair of confident tracks has a subject matching its subject and an object matching its object (class-aware, vIoU ≥ 0.5 over the relation's duration), and the two tracks overlap ≥ 10 frames: the pair rule, so the pair would actually be scored.
6. **"When ≥ 30 instances".**
   - **Recommendation:** the novel group as a whole (124 GT objects) carries the novel bar; per-class recall is reported descriptively. No single novel class reaches 30 (antelope 29).
7. **Unmatched confident track.**
   - **Recommendation:** a confident track matched to no GT object (class-aware, vIoU ≥ 0.5). Real but unannotated objects, such as the distant person G0 saw, count as unmatched for both trackers, so the 1.5× bar stays relative.
8. **Box style, box sensitivity, drawn frames.**
   - **Recommendation:**
     - Box IoU per matched pair is the mean per-frame IoU over shared annotated frames; the bar is the median over pairs.
     - The "about 20 frames" are the middle annotated frame of each of the first 20 HELD videos.
     - Box sensitivity: per clip of each base GT relation that both trackers match, the GT predicate is a hit if it is among the top 5 **base** predicates of modelC's clip scores, with GT, SAM 3 and EOV-MMP boxes in turn on the same clips. This uses a modelC run per box source on HELD, with no score read beyond the hit.
9. **"First 20, then all"** (the cost-cap stop is settled above).
   - **Recommendation for the rest:** besides the 35-GPU-hour projection, stop and ask if any of the first 20 videos crashes. Then run all 160; every bar is judged on all 160, never on the first 20 alone.
10. **Raw [0.1, 0.5) detections for missed GT objects** (D1).
    - G0 found them dense: about 26 per frame on average, pre-NMS, mostly around objects SAM 3 already found.
    - **Recommendation:** for each GT object that no confident track matches, report whether a raw detection **of its class** has box IoU ≥ 0.5 with its GT box on at least one annotated frame, and on what fraction of its frames. Descriptive only, for the G2 amendment; no bar.
11. **Budget and machine.**
    - **The machine:** one RTX 4090 (session 3's GPU, so G0's outputs can be reproduced bitwise at check 0), ≥ 192 GB RAM, the same Vast.ai PyTorch image as G0.
    - **The budget:** at most 35 GPU-hours for SAM 3 (the cap), about 3 for EOV-MMP's tracks (OPEN 3), and about 1 for the first-hour checks. That is at most about 39 GPU-hours, roughly $14 at $0.35/h, to be approved by the author before renting.
    - No test file on the machine; the token as in A28.

**Not changed by this amendment:** anything in A28 and A28a other than the correction above. Any change to the above is a new amendment, written before the result it affects.
