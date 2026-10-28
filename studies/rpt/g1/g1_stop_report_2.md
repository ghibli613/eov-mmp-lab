# RPT G1, stop 2: system RAM ran out on the 8× machine (A1 caches) — decision needed

2026-10-28, 16:35 UTC · 8× RTX 4090, $3.80/h · commit on the machine `51cd6b0` · nothing pushed

**The decision needed:** a corrected RAM rule, so D1 can continue. The machine is idle and billed
meanwhile; the cap clock runs (D3).

## 1. What happened

- **The projection after the first 20 passed:** 22.2 projected billed GPU-hours, against the cap of
  48.
  - Measured short-video rate: 0.0552 s per frame per prompt per process.
  - Wall clock: 1.34 h so far, plus 1.43 h simulated remaining.
  - The other 137 videos were released at 16:27:07 UTC.
- **The stop.** At 16:30:52 UTC, the worker running `ILSVRC2015_val_00027000` (953 frames, A1) died.
  The container's memory cgroup records `oom_kill 1`: **the kernel killed it because system RAM was
  exhausted.** Available RAM had fallen from 314 GB to 40 GB within minutes. A29a's rule "any video
  that fails to run stops G1" fired, and every process stopped.
- **The cause: A29a's RAM rule was too loose for 16 processes.**
  - The rule: all A1 caches together at most RAM − 64 GB = 454.9 GB here (cgroup limit 518.9 GB),
    at 106 MB per frame.
  - **Measured instead: 112.1 MB per frame** (all 20 finished videos, 111.6–112.3). At the last
    start, the estimated 430.9 GB of caches was really about 456 GB.
  - The 16 SAM 3 processes (8 GPUs × 2, A2) need about 4 GB each of their own, and used up the
    64 GB margin.
  - The scheduler did apply the rule: a video had just started *without* A1 because of the limit.
    The limit itself was wrong.
- **Lost:** the two long videos (`…00237001`, 1,470 frames; `…00896000`, 1,080), killed after
  about 35 min of work, and the short videos in progress. **Kept:** the first 20 videos' SAM 3 files,
  hashed in the manifest.
- **Clock:** 11.3 billed GPU-hours used at 16:35 UTC, out of 48; about $10 spent on this machine.

## 2. Proposed fix (needs your approval, as an amendment, A29c)

**The A1 RAM rule becomes:** the A1 caches of all running processes, at the measured **112 MB per
frame**, stay at most instance RAM − 64 GB − 8 GB × the number of SAM 3 processes.
- Here that is 518.9 − 64 − 8 × 16 = **326.9 GB**.
- The 8 GB per process is about twice the measured 4 GB.
- Plus a **live guard:** an A1 start also needs `MemAvailable` minus its cache to stay ≥ 64 GB at
  that moment.
- A video that fails either test runs without A1, with the same output (check 1, check 5).

**Effect:** fewer videos run with A1 at once, so it is somewhat slower, but within the cap.
- Remaining work: 138 videos, about 42,000 frames.
- Most still run with A1: 326 GB is about 2,900 frames of cache at a time, across 16 processes.
- My estimate is about **1.5–2 h of wall clock, 12–16 GPU-hours**. With 11.3 used plus the waiting
  time, **about 25–30 of 48 GPU-hours**.
- The driver repeats the projection only if you ask. A29a's projection was made once, and its numbers
  stand.

**Alternatives:**
- **(b) A1 off for the rest.** Same output, but about 1.9× slower. The remaining work would be about
  33 GPU-hours, which with the 11.3 used and the waiting comes close to the 48 cap: likely a stop.
- **(c) A2 off** (one process per GPU, so 8 processes). The current RAM rule would then hold, with
  about 64 GB to spare. A2 gave only 1.04× in check 2, so little speed is lost. This needs no new
  memory figure: fewer processes leave more margin. It is also a simpler change.

**My recommendation:** (c) A2 off for the rest of D1, plus the live `MemAvailable` guard. It is the
smallest change, and A2's measured gain was negligible. Or the proposed rule above, if you prefer to
keep A2.

## 3. What I do on your answer

1. Write the amendment (A29c), then append, hash, check, scan and commit.
2. Change the driver (with a test), then re-sync and restart D1. The 20 finished videos are kept.
3. Then as before: the EOV-MMP side, then copy-back, then I tell you the machine can be destroyed.
