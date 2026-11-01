# Monday summary (post-G1 Part 1)

2026-11-02 · all on the laptop, $0 · nothing pushed · chain head after today:
**`b10cf7a793ea1d457d6d69246c457a4b6636f1376ac5146eba1f5fa0bcff0aae`** (A30)

| task | outcome | decision |
|---|---|---|
| 1. Trimming oracle on test | **stopped before its amendment**; no test label opened | needed from you: which predicted runs it changes |
| 2. SAM 3 screen on HELD | run under A30 | **FAIL: SAM 3 closes as a detector** |
| 3. Option 1 design note | written; no runs | recommends **SAM 2** (small); 6 code-forced points to decide |
| 4. VidOR preparation | not run | waits on Task 1 (runs only on go or weak go) |

## Task 1: trimming oracle — stopped ([trim_oracle_stop.md](trim_oracle_stop.md))

**Why it could not be done as written.** The rule changes "predicted runs that D1's code classifies
as the C1 over-long case", but D1's code classifies **missed GT relations**, not predicted runs.
- Its over-long test (`tc_posthoc.py`, the 129) works on a run it rebuilds itself, joined across
  one-slot gaps.
- The authors' association joins **consecutive** slots only, and the top-200 cut follows.
- So D1's run can span several predicted runs, or none in the top 200.

**The decision needed.** The stop report gives two faithful readings and recommends (A), anchored on
D1's run, so the result connects to D1's 129. It also asks you to confirm seven smaller points:
- T1's predictions must be regenerated from session 3's caches, because they were never saved; the
  reproduction gate is equality with `t1_results.json`;
- a validated copy of D1's code is needed to run it on T1's logits;
- the "all" coverage; the 30-frame slot (recorded); the variants' tie-breaks; the trajectory
  slicing;
- the label: the trimming oracle becomes **A31**, because A30 went to the screen as the next free
  label.

**Also done:** the test-looks ledger, [studies/test_looks_ledger.md](../test_looks_ledger.md).
- 16 test-label accesses so far, from the pilot's baseline reproduction to Q7. Unsure entries are
  marked: session 3's Task 5 and the date of the mirror-miss analysis.
- Test-video uses without labels.
- Looks that were planned but never ran (T2, RB1's evaluation, A26's audit).
- The trimming oracle's entry is pending.

## Task 2: SAM 3 screen — FAIL ([sam3_screen_report.md](sam3_screen_report.md))

**No configuration passes with the class gate on: 0 of 500.**
- The gate alone cuts SAM 3's recall gain from +9.80 to **+4.90** points (bar +8). Every other rule
  only removes tracks, so no gated configuration can recover.
- The gate removes 154 tracks: 131 unmatched, but also **23 matched, 16 of them of novel classes**.
  These are SAM 3's own finds of objects that EOV-MMP's R0w tracks lack in that video.
- Without the gate, 16 configurations pass (tiny + edge + geometry partner; best +8.68 / +8.79 at
  1.29×). These count for nothing under A30.
- Rule 6 was skipped: G1 cached no modelC scores for general SAM 3 pairs.
- The screen's scorer equalled A29's `metrics.py` on 27 configurations, and the no-rule numbers
  reproduce G1's.

**By A30's binding rule, SAM 3 closes as a detector.**
- **The caveat A30 states:** the screen's gate (R0w, score ≥ 0.35) is stricter than Option 2's
  planned ≥ 0.1 gate. The recall cost is therefore an upper bound, which cannot be measured without
  rerunning the detector.
- SAM 3's **tracker** stays a candidate for Option 1.

## Task 3: Option 1 design note ([option1_design.md](option1_design.md))

All ten points were checked against EOV-MMP's code and the tracker API (SAM 3's
`Sam3TrackerPredictor`, which keeps SAM 2's interface).
- **The API fits the design:** box prompts, forward and reverse propagation, a per-frame object
  score. New objects cannot be added after tracking starts, and the design does not need it.
- **Six code-forced points need decisions:**
  - (a) the seed score must come from the IoU > 0.5 detection match, applied before AFLink;
  - (b) unconfirmed deep_sort tracklets cannot seed;
  - (c) the keeping floor needs the detector rerun with its 0.35 pruning threshold and deep_sort's
    0.2 `min_confidence` lowered (both are arguments, so no authors' file is edited);
  - (d) whether final tracks keep the ≥ 10-frame and 15% rules;
  - (e) the current class rule is a score-weighted vote, not a majority vote;
  - (f) the grid is cheap only if objects are tracked independently, which needs SAM 3's
    `non_overlap_masks_for_output` off and a check.
- **The detector rerun is measured from G1:** 0.112 s per frame on a 4090, about 1.5 GPU-h per
  160 videos.
- **Cost estimate:** about $5–12 for the HELD grid, the FIT gate and a test pass if independence
  holds; about $15–35 if not.
- **The FIT gate is drafted, not hashed.** The FIT subset: 160 videos, `default_rng(20261102)`.
- **Recommendation: SAM 2 (small):**
  - it is protocol-clean (no text model);
  - it has about 20× fewer parameters;
  - it fits the laptop's 4 GB GPU;
  - SAM 3's head start covers only its detection path, not its tracker.
- **The optional smoke test was not run:** it needs your OK for downloads. Cloning the `sam3`
  environment needs none; SAM 2's code is a few MB; the checkpoint is about 155–185 MB.

## Task 4: VidOR preparation — not run

Task 4 runs only if Task 1 says go or weak go. Task 1 has no result yet, so nothing was read or
downloaded. No drop list was written.

## Commits (local, not pushed)

| commit | content |
|---|---|
| `f842888` | G1 descriptive: the observation report and `g1_report` §6.5, as you reviewed them (Task 2's input) |
| `fc7909b` | **A30** appended (head `b10cf7a7…0aae`, 90 checks passed); the screen's code and tests; the pack saved at `studies/postg1/monday-pack.md` |
| `b04b714` | Task 2: the screen's results and report (FAIL) |
| `f4c740d` | Task 1: the stop report; the test-looks ledger |
| `7c91a28` | Task 3: the Option 1 design note |
| (this file) | the Monday summary |

**What happens next:**
1. Your answer on Task 1's reading.
2. A31: write, hash, check and commit it.
3. The one guarded run.
4. Then Task 4, if the result is go or weak go.
