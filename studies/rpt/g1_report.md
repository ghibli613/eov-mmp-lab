# RPT G1 report: FAIL — the false-track burden fails even at SAM 3's best case (A29d's futility rule)

2026-10-29 · pre-registration head `c9718301…fac8` (A29d, commit `c230794`) · computation commit
`26eeef1` · nothing pushed

**HELD in-sample tracker comparison.** EOV-MMP's detector was trained on HELD, so its numbers here are
inflated. These are not test results, and HELD relation numbers are not results.

## Verdict

**G1 fails for certain, under A29d item 2.** The **false-track burden** bound over all 160 videos is
**1.81×**, against a bar of **≤ 1.5×**. This holds even with SAM 3 at its best case on the 3 videos
without SAM 3 files.
- **As A29d says:** the 3 videos are not run, and nothing is rented.
- **As A29 says:** "A fail of a binding bar stops the line (proposal §6, fallback 2)." G2 does not
  start.

The other three binding bars pass at their best case, so they decide nothing here. Box style also
passes on the 157 videos.

## 1. The futility computation (A29d item 2)

- **SAM 3:** its 157 videos, plus its best case on `…00280000`, `…00225001` and `…00971000`:
  - all 39 of their GT runs matched;
  - all 156 of their base relations recalled;
  - no unmatched tracks;
  - one box-style pair at IoU 1.0 per GT run.
- **EOV-MMP:** its real tracks (R0w) on all 160 videos, from the 8× machine.

| binding bar | threshold | SAM 3, best case over 160 | EOV-MMP (R0w), over 160 | bound | at best case |
|---|---|---|---|---|---|
| Overall recall | ≥ +10 points | (555 + 39) / 753 = 78.88% | 500 / 753 = 66.40% | +12.48 points | passes |
| Pair recall | ≥ +10 points | (2,751 + 156) / 4,390 = 66.22% | 2,411 / 4,390 = 54.92% | +11.30 points | passes |
| **False-track burden** | **≤ 1.5×** | (278 + 0) / 160 = 1.738 unmatched confident tracks per video | 154 / 160 = 0.963 | **1.81×** | **fails** |
| Box style | ≥ 0.70 | median over 555 + 39 = 594 pairs | — | 0.883 | passes |

**Why the burden cannot recover:**
- SAM 3 has **278** unmatched confident tracks on the 157 videos alone.
- For a ratio of at most 1.5× over 160 videos, it could have at most **231** (1.5 × EOV-MMP's 154).
- So it is 47 tracks over the bar before the 3 missing videos add anything.

## 2. The 157-video numbers (reported only; A29d: not used for any decision)

SAM 3 against EOV-MMP (R0w), on the same 157 videos.

| binding bar | SAM 3 | EOV-MMP (R0w) | value | threshold |
|---|---|---|---|---|
| Overall recall | 555 / 714 = 77.73% | 485 / 714 = 67.93% | +9.80 points | ≥ +10 points |
| Pair recall | 2,751 / 4,234 = 64.97% | 2,363 / 4,234 = 55.81% | +9.16 points | ≥ +10 points |
| False-track burden | 278 / 157 = 1.771 per video | 151 / 157 = 0.962 per video | 1.84× | ≤ 1.5× |
| Box style | median over 555 pairs | — | 0.873 | ≥ 0.70 |

- Every matched pair had at least one annotated frame shared with its track (0 pairs without one).
- On the 157 videos, overall recall and pair recall fall just short of +10 points. Their bounds over
  160 do not, so by A29d they decide nothing.

## 3. What was run

- **SAM 3:** 157 HELD videos on one 8× RTX 4090 machine (A29a), under A29, A29a, A29b and A29c.
  - Check 0 was bitwise identical to G0 on all 8 GPUs.
  - A1 (backbone reuse) passed bitwise and was used.
  - A2 (2 processes per GPU) passed but was turned off by A29c after stop 2 (system RAM).
  - A3 (`torch.compile`) was not bitwise identical and was not used.
  - Offload "passed" only because it is a no-op (A29d item 1); it was used by no video that
    produced a file.
  - The SAM 3 phase used 34.37 billed GPU-hours, under the cap of 48.
- **Videos without SAM 3 files:**
  - `…00280000`: out of GPU memory at 22.79 GiB (stop 1; A29b);
  - `…00225001` and `…00971000`: out of GPU memory at 21.46 and 22.42 GiB (moved under A29c).
- **EOV-MMP:**
  - R0w and R0 tracks for all 160 videos;
  - box-sensitivity runs for the 157 videos.

  Both were computed on the 8× machine with the train ECC file (A29 item 1).
- **Copy-back:** 317 files (157 SAM 3, 160 EOV-MMP) match the machine's SHA256 (commit `6ea77c7`).
  Every file's hash is recorded in `studies/rpt/g1/results_futility/futility.json`.
- **The computation:** `studies/rpt/g1/futility.py`.
  - It reuses A29's matching, pair-recall, unmatched-track and box-style code (`metrics.py`) and
    `analyse.py`'s SAM 3 reader.
  - The bounds go through the same bar function, `metrics.bars`, as the final verdict would have.
  - Pins were checked first.
  - It took 28 s on the laptop.
  - Output: `studies/rpt/g1/results_futility/`; log: `studies/rpt/g1/logs/futility_run.log`.

## 4. Not computed

A29d's fail branch asks only for the verdict, this rule's computation and the 157-video numbers. So
these were **not** computed:
- every report-only A29 quantity;
- the R0 comparison;
- the secondary box-sensitivity bar;
- the bootstrap CIs;
- the raw detections for missed objects, gap exposure and the visual check.

After the verdict, the plan author asked for five descriptive quantities about the unmatched tracks.
They are in section 6. The rest of the list above stays not computed.

## 5. Events and notes

- **Stop 1:** `…00280000` ran out of GPU memory, leading to A29b.
- **Stop 2:** the system RAM was exhausted (A1 caches with 16 processes), leading to A29c.
- **Under A29c:** 2 videos were moved off the 8× machine for GPU memory.
- **A29d item 1:** SAM 3's video offload is a no-op. The cause given in stop report 1 and in A29b
  was wrong. Offload is not carried into G2 or the test session.
- **Cost:**
  - The 8× machine was not destroyed when its work ended at 19:51 UTC. It was found unreachable at
    23:14 UTC. Its total cost is between about $23 and $35; the exact figure from Vast's billing is
    pending.
  - Nothing else was rented. The 48 GB machine was not rented.

## 6. Descriptive, after the verdict

**These were computed after the verdict, at the plan author's request. They describe the fail, decide
nothing and change no verdict.**
- **Data:** the 157 videos with SAM 3 files, for SAM 3 and EOV-MMP (R0w), from the committed files,
  on the laptop.
- **Code:** `studies/rpt/g1/describe_unmatched.py`, reusing A29's matching and cut-vIoU code
  (`metrics.py`).
- **Output:** `studies/rpt/g1/results_descriptive/descriptive.json` (every class, every count);
  log: `studies/rpt/g1/logs/descriptive_run.log`.

### 6.1 Unmatched confident tracks, class-aware and class-agnostic (A29 item 5)

| count, 157 videos | SAM 3 | EOV-MMP (R0w) | SAM 3 ÷ EOV-MMP |
|---|---|---|---|
| class-aware (the bar's count) | 278 (1.771 per video) | 151 (0.962) | 1.84× |
| class-agnostic | 234 (1.490 per video) | 112 (0.713) | 2.09× |

Ignoring class names removes 44 of SAM 3's unmatched tracks and 39 of EOV-MMP's. The ratio grows
rather than shrinks, so misnaming does not explain the excess.

### 6.2 Unmatched confident tracks by class

The 10 classes where SAM 3 has the most unmatched tracks more than EOV-MMP. Each cell is unmatched
tracks / all confident tracks of that class.

| class | SAM 3 | EOV-MMP (R0w) | SAM 3 − EOV-MMP, unmatched |
|---|---|---|---|
| person | 55 / 114 | 18 / 68 | +37 |
| bird | 41 / 64 | 7 / 31 | +34 |
| car | 28 / 60 | 9 / 35 | +19 |
| dog | 13 / 34 | 2 / 26 | +11 |
| elephant | 10 / 20 | 0 / 10 | +10 |
| sheep | 23 / 51 | 14 / 45 | +9 |
| monkey | 23 / 44 | 16 / 39 | +7 |
| antelope | 7 / 17 | 2 / 10 | +5 |
| giant_panda | 5 / 15 | 0 / 0 | +5 |
| ball | 7 / 11 | 4 / 9 | +3 |
| the other 24 classes | 66 / 208 | 79 / 217 | −13 |
| **all** | **278 / 638** | **151 / 490** | **+127** |

Person, bird and car alone account for 90 of the 127 extra unmatched tracks.

### 6.3 What the unmatched tracks overlap

Each unmatched track's best cut-vIoU (A29 item 2) to the video's GT runs:

| group | SAM 3 | EOV-MMP (R0w) |
|---|---|---|
| misnamed: ≥ 0.5 to a run of another class | 46 | 40 |
| poorly placed: best in [0.1, 0.5) | 35 | 67 |
| **no GT object nearby: best < 0.1** | **196** | **44** |
| none of the three: ≥ 0.5 only to a run of its own class, which went to another track | 1 | 0 |
| total | 278 | 151 |

- The three requested groups leave out one case: a track ≥ 0.5 to a run of its own class that lost
  that run to a better track. It is counted apart (1 SAM 3 track).
- No track is ≥ 0.5 to runs of two classes.
- **SAM 3's excess is in the "no GT object nearby" group:** 196 tracks against 44, +152. The
  misnamed groups are similar (46 against 40), and SAM 3 has fewer poorly placed tracks (35 against
  67).

### 6.4 SAM 3's tracks by score

| track score | unmatched | matched | unmatched share |
|---|---|---|---|
| 0.5–0.6 | 37 | 15 | 71% |
| 0.6–0.7 | 35 | 27 | 56% |
| 0.7–0.8 | 121 | 65 | 65% |
| 0.8–0.9 | 68 | 89 | 43% |
| ≥ 0.9 | 17 | 164 | 9% |
| all | 278 | 360 | 44% |

- "Matched" tracks are SAM 3's confident tracks that match at least one GT run (360 tracks, matching
  555 runs).
- No confident track scores below 0.5.
- Unmatched tracks are concentrated below 0.9: 261 of the 278 score under 0.9, against 196 of the
  360 matched tracks.

### 6.5 A visual sample of "no GT object nearby"

- **What:** 20 of SAM 3's 196 such tracks, each drawn on its middle frame. The track is in red, with
  its class and score; the GT runs covering that frame are in green.
- **How they were chosen:** `numpy.random.default_rng(0).choice(196, 20, replace=False)` over the
  tracks sorted by (video, track id).
- **Where:** `studies/rpt/g1/viz_unmatched/`.
- **A closer visual reading** of each image is in `studies/rpt/g1/viz_unmatched_observations.md`.
  It was checked by the thesis author and read from the raw frames, because the drawn outline and
  label hide what is inside small boxes. It is a judgement from one frame per track, not a
  measurement:
  - 6 of the 20 are clearly real objects of the named class with no GT box on that frame, and 1 more
    probably is;
  - 9 are SAM 3 errors: boxes on no visible object (5), small pieces of larger objects (3), and a
    wrong class on a real object (1);
  - 4 are unclear;
  - four are tiny boxes touching the frame edge that persist for 83–149 frames.
  - This corrects an earlier one-line summary here, which named examples from the drawn images that
    the raw frames do not support.
