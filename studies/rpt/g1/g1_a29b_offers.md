# RPT G1: D1 running, A29b hashed, 3 offers for `…00280000` (approval needed)

2026-10-28, 16:05 UTC · nothing pushed

## D1: running on the 8× machine

- **Restarted** at 15:56 UTC with an explicit exclude list. The driver's
  `--exclude ILSVRC2015_train_00280000` is logged and recorded in `plan.json` and
  `sam3_done.json`.
- **Tests for the change passed** (the driver on 8 fake GPUs with the exclusion, and the EOV-MMP side).
  Commit `51cd6b0`.
- **The stop is archived:** `STOP_video.json` became `stop1_STOP_video.json`, with a note in
  `run_notes.log`.
- **Progress at 16:01 UTC:** 16 of 159 SAM 3 files are done (16 of the first 20). The two other long
  videos are running alone on their GPUs. 7.3 billed GPU-hours used (D3: the wait counts).
- **Next:** the projection after the first 20, then the remaining 139, longest first. After that the
  EOV-MMP side: R0w and R0 for all 160, box sensitivity for the 159.
- **The preflight outputs** (`~/rpt_g1_out/preflight_eov/`) are kept and come back with copy-back.
- **Expected:** the SAM 3 phase ends about 18:00–18:15 UTC, then the EOV-MMP side and copy-back. I
  will tell you when the machine can be destroyed.

## A29b: appended and hashed

- **Chain head:** `85d7d748…047b` → **`6917907926cd989b2d5a71983280043be818049a90802ec403a6da4e2b846e63`**.
- **A29b's text alone:** `dd76dff1c6fdc5963cf882a53403363d11b2a4496a637437b0edd0c3ca3507f7`.
- **Checks:** the old bytes are unchanged; both files are back at 444; `verify_after_A26m.py` passes
  all 81 checks.
- **Commit** `b8989c1`, after a credential scan with 0 hits.
- **It records:**
  - the stop and its cause;
  - D1 and D3;
  - D2 as option (a), restricted to an Ada-generation 48 GB GPU (RTX 6000 Ada, L40S, L40), with the
    same software stack;
  - check 0 required (stop and ask if it fails);
  - `…00280000` as one process at defaults, with no offload, A1, A2 or A3;
  - box sensitivity for that video only if the EOV-MMP preflight there is bitwise identical (apart from
    timing) to the 8× machine's; otherwise 159 videos, disclosed;
  - the R0w and R0 tracks for every bar come from the 8× machine;
  - the cost stays within the $35 budget.
- **One addition beyond your bullets, for your check:** the sentence that the bars' R0w and R0 tracks
  are the 8× machine's for all 160 videos. That follows from D1.

## Three offers for the 48 GB run (1 GPU, verified, reliability ≥ 0.98, CUDA ≥ 12.8)

| # | offer | GPU | $/h | RAM | disk | location | reliability | driver / max CUDA |
|---|---|---|---|---|---|---|---|---|
| 1 | 40187935 | L40 48 GB | **0.457** | 128 GB | 75 GB | Vietnam | 0.991 | **580.142 (G0's driver)** / 13.0 |
| 2 | 57063284 | L40 48 GB | 0.457 | 192 GB | 501 GB | Vietnam | 0.986 | 580.142 / 13.0 |
| 3 | 51528883 | L40 48 GB | 0.495 | 72 GB | 118 GB | New York, US | 0.992 | 570.211 / 12.8 |

- **Other models:** the cheapest verified RTX 6000 Ada is $0.602/h (Türkiye, 128 GB, reliability
  0.982). No verified L40S met the filters.
- **Note:** L40, L40S and RTX 6000 Ada are the same chip generation as the RTX 4090, but different
  models. Bitwise agreement with G0 is not guaranteed; check 0 decides, as A29b says.

**Estimate for the 48 GB run:**
- set-up 30–40 min (the upload is small: 3 videos' frames);
- check 0 about 7 min, the EOV-MMP preflight about 3 min;
- `…00280000` at defaults: about 2.6–3 h (270 s per prompt on a 4090; an L40 may be 10–20% slower);
- the EOV-MMP side about 15 min.

That is about **4 h, under $2**. **The whole of G1:** about $17 for the 8× machine plus about $2,
**about $19**, under the $35 budget.

**Decision needed:** approve one offer. The author rents it after the 8× machine is destroyed.
