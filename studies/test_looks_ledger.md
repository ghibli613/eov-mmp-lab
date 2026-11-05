# Test-looks ledger

Every access to VidVRD's test relation labels in the project, from the existing docs and logs. Created
2026-11-02 for the post-G1 trimming oracle (`studies/postg1/monday-pack.md`, Part 1, Task 1).
- **Kinds of access:**
  - **test evaluation:** a method scored with the official evaluator;
  - **analysis:** a disclosed diagnostic, no method evaluated;
  - **derived:** reads files computed from test labels, but opens no label file.
- **"Unsure"** marks an entry whose test-label use I could not fully confirm from the records.
- **Not counted as label access:** test-video inputs and model outputs read without labels. These
  are listed in the second table.

## Test-label accesses

| # | when | what | labels | kind | what it decided or informed | source |
|---|---|---|---|---|---|---|
| 1 | 2026-09 (pilot, run completed 2026-09-02) | **Baseline reproduction** on the pilot's own dumps: SGDet 27.18 all / 15.79 novel | test relation GT, all and novel | test evaluation | "the baseline reproduces"; the starting point of everything | `pilot_analysis/report.md`, `PILOT-STATUS.md` |
| 2 | 2026-09 (pilot) | **Merge oracle** (H2): oracle-merged runs, **+0.07** mAP | test relation GT | analysis | H2 refuted: post-processing merge costs about 0.07 | `pilot_analysis/report.md` §H2 |
| 3 | 2026-09 (pilot) | **Loose vIoU sweep:** greedy against oracle at vIoU thresholds down to 0.1, about **+12** mAP at 0.1 | test relation GT | analysis | "the failure is upstream"; reconciled later with D1 (over-long runs) | `pilot_analysis/report.md` (vIoU threshold table) |
| 4 | 2026-09 (pilot) | **GT statistics on test:** 4,835 instances, extent counts, "the greedy merge is load-bearing for 77.4% of the test metric" | test relation GT | analysis | the H2 justification; extent figures reused later | `pilot_analysis/PILOT-STATUS.md` (phases 3.1, 3.3) |
| 5 | 2026-09 (pilot era; date **unsure**) | **Mirror misses:** were missed novel relations predicted on the reversed pair? | test relation GT | analysis | descriptive | `pilot_analysis/mirror_misses/mirror_misses.md` |
| 6 | about 2026-10-01 (session 3) | **Task 4:** baseline PredCls from session 3's caches, 21.70 novel / 37.20 all | test relation GT | test evaluation (reproduction) | the cache-based baseline used by every later batch | `pilot_analysis/stage1c/eval_from_cache.py`; `cache_s3/task4/task4_predcls.json` |
| 7 | about 2026-10-01 (session 3) | **Task 5:** paraphrase test, EOV-MMP's test logits recomputed with reworded predicate names | test relation GT (**unsure**: `task5_paraphrase.json` holds the spread; its evaluation path was not re-read) | analysis (**unsure**) | the paraphrase spread (A4) | `stage1b/preregistration.md` (A4, A8 run 7); `cache_s3/task5/` |
| 8 | about 2026-10-01 (session 3) | **Task 6:** SGDet gate from `cache_s3_sgdet`, 15.79 novel / 27.18 all | test relation GT | test evaluation (reproduction) | SGDet baseline for B, T1, D1, E1 | `stage1b/preregistration.md` (A13, A17); `cache_s3_sgdet/` |
| 9 | 2026-10-05 | **Batch B (A8):** the frozen recipe method, PredCls novel **15.37** | test relation GT | **test evaluation 1** | GATE FAIL; method stops | `pilot_analysis/stage2/batch_b_report.md` |
| 10 | 2026-10-05/06 | **Batch B2 (A9):** recipes reorder within each predicate, PredCls novel **21.80** | test relation GT | **test evaluation 2** | below the 24.70 gate; method work ends | `pilot_analysis/stage2/batch_b2_report.md` |
| 11 | 2026-10-06 | **T1 (A17):** type-consistency rules; SGDet **16.15 / 27.57**, PredCls 22.25 / 38.03; random-deletion control; per-predicate diagnostics | test relation GT | **test evaluation 3** | improves the baseline, not METOR; T1 is the current best | `pilot_analysis/final/t1_report.md` |
| 12 | 2026-10-07 | **D1 (A20):** time check, at which stage each missed novel relation is lost; post-hoc C1 split (**129** SGDet over-long) | test relation GT | analysis | upstream dominates; the over-long finding | `pilot_analysis/time_check/time_check_report.md` |
| 13 | 2026-10-07 | **E1 (A21):** nested runs, SGDet 13.84 novel (−2.31 against T1) | test relation GT | **test evaluation 4** | FAIL | `pilot_analysis/boundary/boundary_report.md` |
| 14 | 2026-10-09 | **M1 (A24–A24i, ov-pairtime):** pair-timeline detector against chunk-and-glue, PredCls, 3 seeds; test extents (median 60 frames, 77.4% longer than one slot) | test relation GT, opened once through a guard | **test evaluation** (the fifth actually run; the chain's own numbering differs because RB1's planned evaluation never ran) | G-M1 FAIL | ov-pairtime bundle (`a820a0e`): `m1/step4_report.md`, `m1/M1_RESULTS.md` |
| 15 | 2026-10-10 | **D2 (A25–A25d):** at which front-end stage D1's U1 tracks are lost; the test annotations were on the rented machine and deleted after | test annotations (relations and trajectories) | analysis | descriptive; its recordings are barred from design use (A25d) | `studies/d2/d2_report.md` |
| 16 | 2026-10-11 | **Q7 (A25e):** headroom count, how many of D1's 262 SGDet U1 relations PredCls matches | D1's `tc_missed.json` (computed from test labels); no label file opened | derived | descriptive, for the design review | `studies/prereg/preregistration.md` (A25e); `studies/design_review/` |

**Planned but not run** (no test-label access happened):
- **T2 (A18):** frozen, never run, because the position verifier failed G-P3.
- **RB1's test evaluation (A23):** "the fifth is still ahead".
- **A26's protocol audit (R0u / R0w on test):** the code is ready; there are no results
  (`studies/a26/data/` holds only `r0_reference.json`, the reference predictions).

**No test-label access:**
- **G0 and G1 (A28–A29d):** a guard was installed that raises on any test annotation.
- **The SAM 3 screen (A30):** HELD only.
- **PaD, the split and the naming check (A31–A33b, 2026-11-04/05):** HELD-A and FIT only; both the test-label guard
  and the HELD-B guard were installed in every script; D7 not run.

## Test videos used without labels (not label access)

| when | what | source |
|---|---|---|
| 2026-10-06 | F1 Part B1: track fragmentation over the 200 test videos' detected tracks (model outputs only) | A13, `pilot_analysis/f1/f1_report.md` |
| 2026-10-06 | F2: test model inputs and outputs only; no annotation, no evaluation | A14, `pilot_analysis/f2/f2_report.md` |
| 2026-10-06 | T1 dry run: SGDet path label-free on 5 test videos | A17, `t1_report.md` §1 |
| 2026-10-07 | Position verifier: "label-free test reach" on cached test inputs | A18–A19, `pilot_analysis/position/position_report.md` |

## Closed entry (formerly pending)

| # | when | what | labels | kind | decision | crash rule |
|---|---|---|---|---|---|---|
| 17 | closed 2026-11-04 | **Trimming oracle (post-G1 Task 1):** **not run by decision; no test label accessed.** Its amendment was never hashed (`studies/postg1/trim_oracle_stop.md`); Giang decided not to run it (`studies/postg1/d7-session-prompt.md` v2, decision 1). The label A31 went to the HELD split. | none | — | — | — |
