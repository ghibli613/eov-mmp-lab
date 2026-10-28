# RPT G1: the 8× machine is done, copy-back verified; ready for the 48 GB run

2026-10-28, 20:05 UTC · commits `6ea77c7` (outputs), `eaa627b` (48 GB scripts) · nothing pushed

**The 8× machine can be destroyed.** Every output is on the laptop, with hashes verified and committed.

## 1. D1 under A29c: finished

- **SAM 3:** 157 of the 159 videos have files. The phase ended at 19:24 UTC, at **34.37 billed
  GPU-hours** (cap 48).
- **Moved to the 48 GB run (A29c), 2 of the 3 allowed:**

  | video | frames | failure | when (UTC) |
  |---|---|---|---|
  | `ILSVRC2015_train_00225001` | 836 | CUDA out of memory at 21.46 GiB allocated | 16:50 |
  | `ILSVRC2015_train_00971000` | 521 | CUDA out of memory at 22.42 GiB allocated | 18:20 |

  - Both were short videos with many objects: G0's formula predicts 11.6 and 9.7 GiB for them.
  - Their partial outputs were discarded; the error records are in `logs/box/oom_failed/`.
  - There were no other failures and no stop after A29c.
- **EOV-MMP side:** R0w and R0 for all 160 videos; box sensitivity for the 157 videos with SAM 3
  files. Not for `…00280000`, `…00225001` or `…00971000`, which have no SAM 3 file yet. It ended at
  19:51 UTC.
- **Copy-back:** 317 files (157 SAM 3, 160 EOV-MMP) match the machine's SHA256. Also copied back:
  - the checks' outputs;
  - the 8× EOV-MMP preflight outputs (needed for the 48 GB comparison);
  - every machine log, `moved_to_l40.json` and `sam3_done.json`.
- **Cost of the 8× machine:** about **$23** (rented about 14:05 UTC, destroyed about 20:10, at
  $3.80/h). Vast's billing page has the exact figure.

## 2. A correction: SAM 3's video offload does nothing (stop report 1 and A29b's cause sentence are wrong)

- **What I found:**
  - The offload request does reach SAM 3. But SAM 3's video model then copies every frame to the GPU
    regardless, in `_construct_initial_input_batch` (`sam3_video_inference.py:145-153`,
    `copy_data_to_device(input_batch, device)`).
  - The evidence: check 4's peak with offload equals check 0's peak without it, **7.316072 GiB
    exactly**. Offload had no effect.
- **So:**
  - `…00280000` failed at 22.79 GiB with its frames on the GPU, not because of "per-frame state that
    stays on the GPU" (stop report 1; A29b's "The stop and its cause").
  - Check 4 "passed" only because offload changed nothing.
- **Effect on results: none.** `…00280000` was the only video that asked for offload, and it produced
  no file on the 8× machine.
- **Effect on the 48 GB run: good news.** At defaults, `…00280000` should peak at about 23–25 GiB
  (G0's formula gives 24.6), well under 48 GB.
- **Decision needed:** how to record this. **My suggestion:** a disclosure in `g1_report.md`, with no
  amendment. No rule changes; only a factual statement in A29b's narrative is wrong. If you prefer an
  erratum amendment (A29d), say so and I write it before the 48 GB run.

## 3. The 48 GB run: prepared (approved offer 57063284, fallback 40187935)

- **Videos:** `…00280000` (A29b), plus the two moved videos (A29c: "under the same rules as
  `…00280000`").
- **How they run:**
  - longest first (A29a's order): 2,947, then 836, then 521 frames;
  - each in its own fresh SAM 3 process, at defaults: no offload, A1, A2 or A3;
  - check 0 first (stop and ask if it is not identical to G0);
  - any video failure stops and asks (A29a's rule; there is no further machine to move it to).
- **Box sensitivity** for all three only if the 48 GB machine's EOV-MMP preflight matches the 8×
  machine's (A29b's rule, applied to the moved videos as A29c says).
- **Bars:** R0w and R0 for every bar remain the 8× machine's.
- **Code:** `box48/l40_run.py`, `upload48.sh` (now sends the moved videos' frames and checks its list
  against `moved_to_l40.json`), `run48.sh`, `copy_back48.sh`. Tests with fake SAM 3 and EOV-MMP
  cover the order, the options, a failure stopping, a rerun skipping finished videos, and box
  sensitivity gated by the preflight: 5 passed.
- **Estimate:**

  | step | time |
  |---|---|
  | set-up | about 40 min |
  | check 0 | about 7 min |
  | `…00280000` | about 2.6–3 h |
  | the two moved videos | about 1–1.5 h (no A1, and many objects) |
  | preflight and EOV-MMP side | about 20 min |
  | **total** | **about 5–5.5 h, about $2.5** |

- **The whole of G1:** about **$26**, within the $35 budget.

## 4. After the 48 GB run

Copy-back and verification, then every A29 quantity on the laptop with all 160 videos' SAM 3 files
(the machine that produced each file reported), box sensitivity on 160 or 157 videos as above, then
`studies/rpt/g1_report.md`, verdict first. Then stop.
