# Option 1 design note: EOV-MMP proposes, a box-prompted tracker tracks

2026-11-02 · post-G1 Task 3 · no runs · nothing here is hashed; the FIT gate below is a draft

**Option 1:** EOV-MMP's detector and deep_sort propose objects; a SAM 2-style tracker, prompted with
boxes, follows each one; EOV-MMP's detections give each track its class and score.

Each point below is checked against EOV-MMP's code (repo root, the authors' files) and the tracker's
API. The API check uses SAM 3's `Sam3TrackerPredictor` at commit `f1647e8d`, which keeps SAM 2's
video-predictor interface; the SAM 2 points are from SAM 2's design. **"Code forces"** marks where
the code requires a change to the agreed design, or a decision.

## 1. Seeds

**Design:** each deep_sort tracklet, before AFLink and the length filter, is seeded at its
highest-score box. The tracker propagates forward and backward from the seed. All seeds are known
before tracking starts, so one tracker session per video.

**What the code gives:**
- **deep_sort rows:** `run()` (`models/tracking/deep_sort_app.py:125-209`) returns rows
  `[frame, track_id, x, y, w, h, -1, -1, -1, -1]`. Only **confirmed** tracks updated in that frame
  are output (line 197), and the box is deep_sort's Kalman-filtered box.
- **Detections:** deep_sort drops detections below `min_confidence` (0.2 by default,
  `utils/parser_func.py:355`; line 171).
- **Scores:** a track box gets a detection score only later, after AFLink
  (`models/end2end_model.py:294-310`). Each track box is matched to the raw detection in that frame
  with IoU > 0.5, and inherits its 35-class score vector.

**Code forces:**
- **(a) Where the seed score comes from.** deep_sort rows carry no score, so "the highest-score box"
  needs the same IoU > 0.5 match, applied to the tracklets **before** AFLink.
  - Proposal: the seed frame is the tracklet's frame whose matched detection has the highest max-class
    score.
  - The seed box is that **raw detection's box**, not the Kalman box. A raw box is the better prompt.
- **(b) Unconfirmed tracklets never appear.** Objects seen for fewer than deep_sort's `n_init`
  frames cannot seed.

**The API:**
- **Box prompts:** `add_new_points_or_box(state, frame_idx, obj_id, box=...)` takes a box
  (`sam3_tracking_predictor.py:180`); SAM 2's predictor has the same call.
- **Forward and backward:** `propagate_in_video(..., reverse=True)` runs backward.
- **Adding objects after propagation starts: no.** A new object id after `tracking_has_started`
  raises `RuntimeError` ("Cannot add new object id … after tracking starts",
  `sam3_tracking_predictor.py:146-168`; SAM 2 is the same). **The design does not need it:** all seeds
  are added first.

**One session per video works, with one detail.**
- The memory selection `select_closest_cond_frames` (`sam3_tracker_utils.py:273`) takes the closest
  conditioning frame **after** the current frame as well. So in a single forward pass from the
  earliest seed, an object seeded later is "tracked" on earlier frames from its future prompt.
- Proposal, which fixes which output counts:
  - forward pass from the earliest seed, then a reverse pass from the latest seed;
  - for each object, keep the forward outputs on frames ≥ its seed and the reverse outputs on frames
    < its seed.

## 2. Same object

**Design:** two tracks with vIoU ≥ 0.5 over their shared frames are merged; the higher seed score
wins. This replaces AFLink.

**The code:** AFLink (`end2end_model.py:283-293`, `thrT=(0, 30)`, `thrS=75`, `thrP=0.05`) is
dropped entirely.
- vIoU over shared frames uses the official `viou` on the two box sequences cut to their shared
  frames, A29's cut convention.
- "Wins" means the winning track is kept unchanged and the other is removed. Nothing is spliced,
  as in G1's duplicate rule.
- **The order is fixed:** pairs in descending vIoU, then by seed score. Transitive chains resolve
  greedily.

## 3. End

**Design:** a track stops after **K** consecutive frames with no EOV-MMP raw detection at IoU ≥ 0.5
(any class, score ≥ the keeping floor), or when the tracker's own score says the object is gone.

**The API:** the tracker returns per-frame `object_score_logits` (`sam3_tracking_predictor.py:541`).
- A non-positive logit means "object absent", and the mask is emptied (SAM 2's `NO_OBJ_SCORE`).
- So "the tracker says gone" means the first frame, walking out from the seed, with an empty
  output mask.
- **Both end rules can be applied after tracking,** by walking outward from the seed in each
  direction. The tracker always runs to the video's ends, so K and the floor need no extra tracking.

**Code forces (c):** raw detections below EOV-MMP's cut are thrown away before deep_sort, in two
places:
- the detector's post-processor keeps only queries with max class probability **> 0.35**
  (`OVPostProcess`, `pruning_threshold=0.35`, `models/methods/detectors/ov_prompt/post_process.py:132`,
  applied at 194-195);
- deep_sort drops anything below `min_confidence` = 0.2.

Both are **constructor or command-line arguments**, so no authors' file is edited, but the detector
must run again with the floor in both places (item 7).

## 4. Keeping

**Design:** low-score tracklets (score between the floor and 0.35) seed only if they last ≥ **N**
frames. Their track score is min(score, 0.35), so they never outrank a confident track.

**The code:** the "score" of a tracklet is the same as now (item 5). A tracklet is low-score if
its score is below 0.35.

**Code forces (d): the length filter and the 15% rule.**
- Today, final tracks need ≥ 10 frames (`format_trajectories_test`: `len(track) >= 12`, which counts
  the `category` and `score` keys, `gen_labels.py:626`), and must cover > 15% of the video's range
  (`end2end_model.py:317-321`).
- Part 1 removes the length filter **for seeds**, but says nothing about **final** tracks.
- **Proposal:** apply both rules to the final tracks, as now. They are part of EOV-MMP's
  final-track definition (A29 item 1), and the relation stage was trained on tracks that passed them.
- **To decide.**

## 5. Class and score

**Design:** the majority class and mean score of the EOV-MMP detections the track covers, as now.
The tracker gives geometry only.

**Code forces (e): "majority class" is not what the code does now.**
- `format_trajectories_test` (`gen_labels.py:586-622`) sums the matched detections' **35-class score
  vectors** over the track.
- Class = the argmax of that sum. Score = its maximum ÷ the number of matched frames.
- That is a score-weighted vote, not a majority of per-frame labels.
- **Proposal:** keep the current rule exactly ("as now"), applied to the raw detections each tracker
  box covers at IoU > 0.5, the same match as today.

## 6. K, N and the floor: a small grid, fixed on HELD at tracker level, then the FIT gate

| | values | note |
|---|---|---|
| K (frames without a detection before a track stops) | 5, 15, 30 | 30 frames is one slot |
| N (frames a low-score tracklet must last) | 10, 20, 30 | 10 = today's minimum track length |
| floor (keeping) | 0.1, 0.2, 0.3 | 0.1 = Option 2's planned gate; 0.2 = deep_sort's own floor |
| keeping off | — | no low-score seeds; the tracking-only row |

- That is 27 combinations plus the keeping-off rows, chosen on HELD by the FIT gate's own two numbers
  (item 8). The selection rule is to be written into the HELD amendment.
- **Cheap if objects are tracked independently.** Track the union of all candidate seeds once (lowest
  floor, smallest N). Then K, N, the floor and merging are all applied afterwards.
- **Code forces (f): that needs independence.**
  - SAM 3's tracker defaults to `non_overlap_masks_for_output=True` (`sam3_tracking_predictor.py`
    `__init__`), which suppresses one object's mask where another scores higher.
  - It must be **off**, and with it off, one object's outputs must not depend on which other objects
    are tracked.
  - The smoke test can check this: the same object's boxes, with and without the other seeds.

## 7. What must be rerun

Below-cut detections are not cached: G1's EOV-MMP side saved final tracks only. So EOV-MMP's front
end must run again, with the floor in the post-processor and in deep_sort (item 3).

**Cost, measured:** G1's EOV-MMP front end (R0w: detector, modelB, deep_sort, AFLink, track
formatting) took **0.112 s per frame** on an RTX 4090. That is 5,288 s for HELD's 160 videos and
47,038 frames (`studies/rpt/g1/eov_out/`, `configs.R0w.seconds`).

| set | videos | frames | detector rerun |
|---|---:|---:|---:|
| HELD | 160 | 47,038 | 1.5 GPU-h |
| FIT subset (item 8) | 160 | about 49,000 (FIT mean 308 frames) | 1.5 GPU-h |
| test | 200 | 52,104 | 1.6 GPU-h |

A lower floor gives more detections, so deep_sort's share rises a little; it runs on the CPU.

## 8. The FIT gate (draft, not hashed)

**The gate:** Option 1 passes if, on the FIT subset against EOV-MMP's own R0w tracks from the same
rerun, **both** hold with keeping on:
- GT pair recall ≥ **+3 points** over EOV-MMP;
- unmatched-track burden ≤ **1.2×**.

**What is measured how:**
- The measures are A29's: D1's best pair for pair recall; class-aware unmatched confident tracks
  per video.
- The **tracking-only** numbers (keeping off) from the same run are reported and decide nothing.
  There is no second gate.

**The FIT subset (proposal):** 160 of the 640 FIT videos, the same size as HELD.
- Drawn with `numpy.random.default_rng(20261102).permutation` over the sorted FIT ids, taking the
  first 160.
- Fixed now, before anything is run on FIT.
- Fresh for every tracker.
- In-sample for EOV-MMP's detector on both sides, so the bias mostly cancels (reply 1, point 3).

## 9. Cost estimate (one RTX 4090; Vast at about $0.40–0.50 per GPU-hour, G1's 8× machine was $0.475)

**The tracker's share is an estimate, not a measurement.** Assumptions:
- **Seeds:** 10–30 per video. Pre-AFLink tracklets are not cached; G1's final R0w averaged 3.1
  tracks per video, and seeds include every confirmed tracklet plus low-score ones.
- **Per frame, per direction:** the image encoder runs once (about 4–8 ms for SAM 2 small or tiny;
  about 29 ms for SAM 3's encoder, item 10), plus about 7 ms per object for memory attention and the
  decoder. Two directions.
- **Totals:**
  - SAM 2 small or tiny: about 0.15–0.45 s per frame for both directions;
  - SAM 3's tracker: about 0.2–0.5 s per frame.

| run | detector | tracker (SAM 2) | total GPU-h | dollars |
|---|---:|---:|---:|---:|
| HELD grid: one tracking pass over the union of seeds; K, N, floor afterwards | 1.5 | 2–6 | **3.5–7.5** | **$1.5–4** |
| HELD grid, if independence fails (9 tracking passes, floor × N) | 1.5 | 18–54 | 20–55 | $10–28 |
| FIT gate (160 videos) | 1.5 | 2–6 | 3.5–7.5 | $1.5–4 |
| test pass (200 videos) | 1.6 | 2–6.5 | 3.6–8 (+ the relation stage, estimated under 1 h) | $2–4.5 |

- **Machine:** one RTX 4090 24 GB, the stack pinned as in G1 (driver recorded).
- **Totals:** about $5–12 if independence holds, or about $15–35 if it does not.
- **Speed carries over only roughly** (see 10): the per-object figure is the least certain number
  here.

## 10. SAM 2 or SAM 3's tracker

| | SAM 2 (2.1; tiny, small, base+) | SAM 3's tracker (`Sam3TrackerPredictor`, `build_tracker`) |
|---|---|---|
| **parameters** | about 39 M, 46 M, 81 M | about 848 M (the 3.4 GB checkpoint; the tracker shares SAM 3's backbone) |
| **memory per frame** | video frames: 1024² RGB float32, about 12 MB per frame on the GPU unless `offload_video_to_cpu` (which works in SAM 2) | 1008² float16, about 5.8 MiB per frame. In detection mode G0's formula was 6.5 GiB + 6.3 MiB per frame, and offload was a no-op there (A29d). **The tracker path loads frames through `sam2_utils.load_video_frames`, which honours offload**, so offload should work in this mode (to check) |
| **memory per object** | stored per object and frame: memory features (64 × 64 × 64, bfloat16, about 0.5 MB) and a 256² low-res mask (float32, about 0.25 MB), so **about 0.75 MB per object-frame**. `offload_state_to_cpu` moves it to RAM | the same design (`num_maskmem=7`, `storage_device`, `offload_state_to_cpu`, lines 80-90). Similar order, to measure. Its **three out-of-memory events on 24 GB** (`…00280000` at 22.79 GiB, `…00225001` at 21.46 GiB, `…00971000` at 22.42 GiB) were in **detection** mode with many objects (A29d item 1), not tracker mode |
| **expected speed (4090 class)** | published SAM 2.1 figures (SAM 2 README, A100, one object): about 91 / 85 / 64 FPS for tiny / small / base+. **From memory, to check** before they are used | encoder about 29 ms per frame. Estimated from G1's backbone reuse: 0.082 against 0.053 s per frame per prompt without and with A1. The tracker per object is likely similar to SAM 2's |
| **setup and validation** | new: a separate environment, a pinned commit and checkpoint, a G0-style bitwise check. About one day | installed and pinned, with its **detection** path checked bitwise in G0/G1. The **tracker** path has never been run here and needs its own check 0. Less work than SAM 2, but not none |
| **the laptop's 4 GB GPU (GTX 1650)** | **yes** for tiny or small, with video and state offloaded: weights about 150–180 MB | **unlikely:** 848 M parameters is about 1.7 GB even in bfloat16, plus 1008² activations; not tried |
| **protocol** | no text model; trained on SA-1B and SA-V masks without class labels; class-agnostic | box mode uses no class names, but the weights were trained with text and concept labels (SAM 3's SA-Co). A weaker story, though no names enter at inference |

**Recommendation: SAM 2, small (or tiny if the smoke test shows small does not fit).**
- **Option 1 needs geometry only.** SAM 2 is the protocol-cleanest component: no text model and no
  class labels in training. That is exactly what Option 1 argues.
- **It is light:** about 20× fewer parameters, and a much cheaper encoder per frame. Cost is
  dominated by per-object steps that both trackers share.
- **It fits the laptop,** so memory can be measured and the independence check (item 6) run for
  free.
- **The cost of choosing it:** a G0-style setup and bitwise check, about one day. SAM 3's head start
  covers its detection path only; its tracker path would need a check too.

## Optional smoke test: not run (needs your OK for downloads)

- **What it would do:** SAM 2 small or tiny in a **separate** conda environment, never EOV-MMP's, on
  3 HELD videos. It would measure memory per frame and per object, and run the independence check
  (item 6). No metric would be recorded.
- **Downloads needed:**
  - **the environment:** cloning the existing `sam3` conda environment needs no download, if its
    torch is acceptable; a fresh one needs torch (about 2–3 GB);
  - **SAM 2's code:** pip from GitHub, a few MB;
  - **one checkpoint:** `sam2.1_hiera_small.pt` (about 185 MB) or `sam2.1_hiera_tiny.pt` (about
    155 MB). Sizes are from memory, to confirm before downloading.
- **Seeds:** pre-AFLink tracklets are not cached. G1's R0w tracks would be used, so the memory figure
  would be a **lower bound**.
- **Speed:** the GTX 1650 has no fast bfloat16, so its speed would not carry over to a 4090. The
  cost estimate above stays based on published speeds.
