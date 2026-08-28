# EOV-MMP diagnostic pilot — report

**Status: INCOMPLETE — Phases 1, 2.2/2.3, 3.2/3.3 and 4 are unrun.** They need a
GPU with ≥16 GB; see §Environment. This file is the deliverable named by the
protocol ([`../eov-mmp-pilot-prompt.md`](../eov-mmp-pilot-prompt.md)), with every
section it specifies present and the unrun ones marked as such rather than
omitted. The working record, with full derivations and code references, is
[`PILOT-STATUS.md`](PILOT-STATUS.md); section references below (§B.n, §C.n) point
into it.

---

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, **4096 MiB** (needs ≥16 GB) |
| CPU / RAM | 12 cores / 7 GB (WSL2) |
| Python / torch | 3.12.13 / 2.11.0+cu128, CUDA 12.8 |
| Repo | `ov-vidvrd-lab` @ `3e75adc`, **no tracked file modified** |
| Data / checkpoints | all present locally, 47 GB / 9.1 GB |

Raw artifacts are kept **locally, not committed** (machine-specific, and one
transfer log reached 8.85 MB): `pilot_analysis/logs/` holds `pip-freeze.txt`,
`nvidia-smi.txt`, `git-state.txt` and `commands.md`, the last of which lists
every command run so any of it can be regenerated. Upstream
`../EOV-MMP-VidVRD` was never touched.

**The model runs here but not usefully.** It does not OOM — WSL2 spills into
system RAM — but a single video took >31 minutes without completing, so Phase 1
locally would exceed 200 hours. Measured, not assumed (§A).

## Reproduction table — NOT RUN

| Setting | Published | This run | Verdict |
|---|---|---|---|
| Novel split | mAP **15.64**, R@100 18.18 | — | **not run** |
| All split | mAP **26.88** | — | **not run** |

Target agreed 2026-08-27: the **checkpoint's own reported figures** (26.88 all /
15.64 novel), not the paper's headline 26.34 / 15.04 — the supplied
`..._end2end_base-001.pth` is a specific checkpoint and its author-reported
values are the correct gate (§C.3).

Two protocol issues to settle first: the ±0.5 tolerance is narrower than the gap
between the paper's figures and the supplied checkpoint's own reported values
(26.88 / 15.64), so the gate would trip on a working checkpoint (§C.3); and the
code appears to restrict novel-split object classification to novel categories
where the paper says all (§B.17).

## Predicate partition — COMPLETE, signed off 2026-08-27

All 132 predicates classified, none left over, all five flagged judgement calls
reviewed. Full table: [`predicate_partition_table.md`](predicate_partition_table.md).
Two moved on review: `touch` → `appearance_static` (38 base instances, the only
reassignment that moves a number), `drive` → `geometric_static`. `play`, `feed`
and `pull` kept.

| | one frame | needs time |
|---|---|---|
| **boxes suffice** | `geometric_static` 48.3% (44 preds) | `geometric_dynamic` 45.1% (78 preds) |
| **needs pixels** | `appearance_static` 4.4% (4) | `appearance_dynamic` 2.2% (6) |

**H1 retired, replaced by H1'** (§B.19): test `geometric_dynamic` vs
`geometric_static` on both splits, where the mass and the measured CLIP
degeneracy both are. Appearance groups are descriptive only, with small-*n*
caveats. H1' is properly powered — 76–175 videos per cell, SE ≈ 0.03–0.04,
against the retired H1's ±0.08.

Two cautions on H1's criterion, both raised after the power check:

- **The ≤0.5× bar is a large leap from the mechanism evidence** — the measured
  NN-similarity gap between the two groups is 0.0086. Report the ratio with a
  confidence interval and treat 0.5× as a landmark, not a pass/fail line.
- **Label granularity is a full alternative explanation.** `geometric_dynamic`
  has 78 predicates for 2,183 instances; `geometric_static` 44 for 2,334 —
  half the data per label, twice the classes. **Control for it by reporting AP
  per verb family** (13 families × 8–11 spatial variants), which gives four
  matched pairs — `stand`/`walk`, `sit`/`run`, `lie`/`fly`, `stop`/`creep` — and
  is organised the way the within-family confusion prediction needs to be tested.

## Static / dynamic mAP tables — NOT RUN

Novel and all splits, detected and GT-trajectory conditions: **all blocked on
Phase 1.** The evaluation path itself is built and verified — GT fed back as
predictions returns mAP 1.0000 in all eight group×split cells, using the repo's
own evaluator with no new AP code (§B.8).

## Confusion table for dynamic predicates — NOT RUN

Blocked on Phase 1. `evaluate_v2` already computes the needed per-GT
attribution (`gt2det_ids`, `hit_gt`) but `eval_relation_detection_openvoc`
discards it — its `return hit_infos` branch is commented out (§B.6).

**Pre-registered prediction (§C.6):** errors will concentrate *within* a verb
family — `fly_front` mistaken for `fly_right` — not across verbs. Derived from
§B.11 before any prediction exists.

## Duration statistics — COMPLETE

200 videos, 4,835 GT relation instances. Figure:
[`duration_histogram.png`](duration_histogram.png); per-predicate CSV:
[`duration_by_predicate.csv`](duration_by_predicate.csv).

| | all | base | novel |
|---|---|---|---|
| **span > 1 segment** (>30f) | **77.4%** | 78.1% | 72.4% |
| span > 2 segments | 48.2% | 48.5% | 45.6% |
| span > 4 segments | 24.2% | 24.8% | 19.8% |
| mean duration | 107.2 | 108.9 | 95.3 |

**The merge governs 77.4% of the metric.** Segment length is 30 frames
(`args.clip_len`), and segments are a **partition, not a sliding window** (§B.1).

The protocol's stated expectation — "dynamic predicates last longer" — is
**refuted**: dynamic 80.5 frames vs static 131.9, a ratio of 0.61 (§B.15).

## Three-way merge comparison and fragmentation — SIMULATED, NOT MEASURED

Real numbers need Phase 1. The code is written and validated against synthetic
segment predictions derived from GT (§B.12, §B.16). At a 10% per-segment miss
rate:

| | mAP |
|---|---|
| (a) greedy `association()` | 0.7337 |
| (b) oracle merge | 0.9070 |
| **gap** | **0.1733** |
| (c) same, at relaxed vIoU 0.1 | greedy 0.8800 / oracle 0.9680 |

**The gap splits roughly in half:** ~8.5 mAP is temporal *mislocalisation*
(prediction correct, boundaries wrong — recoverable by better temporal decoding
alone) and ~8.8 mAP is content the greedy merge *destroyed* at any threshold.

**Fragmentation is 0.3–1.6%, not ≥25%** — the merge truncates rather than
fragments, and the remnant scores zero instead of counting as a piece. 22.9% of
long GT instances are lost entirely (§C.5).

## Verdicts

**H1 — dynamic-vs-static predicate error. NOT TESTABLE AS SPECIFIED.**

Not "unsupported" — *unmeasurable* on this benchmark. Its cleanest target,
`appearance_dynamic` ∩ novel, is **25 GT instances across 20 videos, 15 of which
contain exactly one**. Per-video AP over one instance is `1/rank` or 0, giving a
standard error near ±8 mAP against a criterion that turns on a factor of two
(§C.2). Recommendation: test H1 on the base split, where the mass is.

What *can* be said, from mechanism rather than mAP:

- **H1's premise is confirmed and is stronger than stated.** A novel predicate
  has *no learned per-class parameter at all* — its entire representation is
  frozen CLIP text encoding of its name in three hand-written templates (§B.10).
- **H1's predicted direction is reversed.** Measured nearest-neighbour
  similarity within the 61 novel predicates: `geometric_dynamic` 0.9599 (worst),
  `appearance_dynamic` 0.9268 (best), against 0.8560 for object names. The
  frozen-CLIP bottleneck is real and severe, but it lands on the **spatial
  component of compositional names**, not on verbs — CLIP cannot separate
  `move_away` from `move_right` (§B.11). Caveat: 3 of 4 components, and the
  appearance rows are n=5 and n=2.

**H2 — merge loss. STRONGLY SUPPORTED in simulation; unmeasured on real output.**

Oracle-merge gain of 11–26 mAP clears the ≥2 mAP criterion by 5–13× across
plausible miss rates. Its *second* criterion (fragmentation ≥25%) would report a
false negative and should be replaced by a loss rate (§C.5). Mechanism confirmed
by code reading: `association()` requires an exact predicate-string match and
`break`s at the first gap, so one weak segment truncates everything after it
(§B.5).

**A confound neither hypothesis accounts for (§B.15).** `appearance_dynamic` is
both H1's target *and* the longest-lasting group (5.48 segments mean), hence the
most exposed to H2's failure. A poor result there cannot be attributed to H1
without measuring it under oracle merge. Phase 4 does not fix this — GT
trajectories remove detection error, not merge error.

## Surprises and caveats

1. **Phase 4 is already published.** It is the paper's own SGCls/PredCls. EOV-MMP
   reports novel SGDet 15.04 → SGCls 17.96 → PredCls 21.65, so the error
   attribution Phase 4 asks for is computable with **no compute**: +2.92
   detection, +3.69 object classification, remainder relationship
   classification. Phase 4 becomes a reproduction check against 21.65 (§B.17).
2. **`--test_traj gt` is a trap.** It defaults to `"gt"` and is printed in every
   config dump, but **no live code reads it**. Phase 1 numbers are
   detected-trajectory numbers regardless (§B.18).
3. **The temporal grid is not a ceiling.** GT extents sit on a 15-frame grid,
   predictions on a 30-frame one, but zero instances become unmatchable — a
   30-frame GT at offset 15 scores exactly 0.5 and the test is `>=`. 8.6% sit at
   that zero margin (§B.3, §B.4). An initial hypothesis, tested and refuted.
4. **`viou_threshold` is hardcoded** at both call sites, so Phase 3.2(c) is
   impossible through the wrapper without the passthrough in
   `scripts/eval_at_threshold.py` (§B.16).
5. **`max_per_video=200` costs ~0.34 mAP** even with perfect predictions: 218 of
   4,835 instances truncated away in 3 dense videos (§B.12).
6. **`clip_top_n` is 20, not 3.** An earlier draft of the working record said 3,
   taken from a test fixture. Corrected (§B.5).
7. **No novel predicate needs an unseen concept** — 44 of 61 have both
   components in base, all 61 have at least one. The split tests recombination
   (§C.4).
8. **71.5% of test videos contain zero `appearance_dynamic` instances**, and mAP
   averages over videos. Independently reproduces `docs/20`'s 142/200 to within
   one video (§B.7).

## What to run next, in order

1. Your review of the five flagged predicates.
2. Phase 0 sanity gate, `--limit 5`, **both** `--frame_stride 1` and `30` — the
   choice is unsettled and changes cost ~30× (§E).
3. Settle the §B.17 object-split question by inspection.
4. Phase 1 full run via `scripts/dump_predictions.py` (dumps pre- *and*
   post-merge in one pass; unit-tested, §B.13).
5. Phases 2–4, all of which are code-ready and CPU-bound once predictions exist.
