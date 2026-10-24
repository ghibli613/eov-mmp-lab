# Final test T1 — type-consistency rules on the test set: report

**Verdict: improves the baseline, but not METOR.**
- SGDet + rules: **16.15 novel** (METOR 16.75, **−0.60**) and **27.57 all**
  (METOR 27.53, +0.04).
- Gains over EOV-MMP: +0.36 and +0.39 in SGDet; +0.55 and +0.83 in PredCls.
- The gain is attributable to the rules: random deletions of the same number score
  7.4 (PredCls) and 5.9 (SGDet) novel mAP lower.

**This is the third evaluation on the test set**, after Batch B (A8) and Batch B2
(A9). Batches A–B2, S1, F1 and F2 stay reported unchanged. The test run was one
scripted run (exit 0, 176 s, CPU), after A17 was hashed.

## 1. A17 and the pinned artefacts

| | SHA256 |
|---|---|
| preregistration.md with A17 (`preregistration.sha256`, line 1) | `0aff1345a1eb0a6b084be9884b4e78431d2dab9a32399bb4f0ec6a10e592b2b5` |
| A17 text alone (3,896 bytes) | `8c6c9f63c08b7954db6f485cd30a3c995e4879689e8366316ebd077378ef36ee` |
| the first 64,194 bytes (through A16), unchanged | `a9517bd7ec0aef631179b3e8e090b1e5d7b7151738ae40fa39dc5d4eeba7ccc0` |

- **59 pins verified** before the run (latest pin per path). They include
  `t1_run.py`, F2's rules file, and every reused script: `common.py`,
  `span_common.py`, `evaluation.py`, `eval_from_cache.py`, the post-processing and
  the evaluator.
- All 400 cache files read (200 PredCls, 200 SGDet) matched
  `SHA256SUMS_session3.txt`. 0 novel labels were used in training (asserted).
- Before A17, the script ran only in `--dry-run`: PredCls, the rules, the control
  and the diagnostics on HELD; SGDet label-free on 5 test videos without
  evaluation.

**The rules (F2 3a, FIT only):** stand, sit, lie, walk, run, creep, swim →
animate subject; stop → vehicle subject. They cover 73 of the 132 predicates.
Deletion sets the predicate's logit to −∞ (score exactly 0 through Task 4's
`slot_scores`) wherever the pair's subject type contradicts the rule. It applies
before post-processing, in both splits, to base and novel predicates alike.
Subjects are the GT category in PredCls and the tracker's category in SGDet.

## 2. Baseline check (run 1)

| | mAP | reference | diff | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|
| PredCls novel | 21.698 | 21.70 | −0.002 | 31.57 | 35.54 |
| PredCls all | 37.201 | 37.20 | +0.001 | 28.58 | 34.62 |
| SGDet novel | 15.786 | 15.79 | −0.004 | 16.36 | 19.01 |
| SGDet all | 27.185 | 27.18 | +0.005 | 17.10 | 19.46 |

**Passed** (±0.05).

## 3. Results

**PredCls (GT tracks and categories)**

| | novel mAP | R@50 | R@100 | all mAP | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| EOV-MMP (baseline) | 21.70 | 31.57 | 35.54 | 37.20 | 28.58 | 34.62 |
| **+ rules** | **22.25** | 31.57 | 36.03 | **38.03** | 28.62 | 34.73 |
| gain | +0.55 | | | +0.83 | | |
| control: random deletions, same count (5 seeds) | 14.81 (sd 1.52) | 26.84 | 33.49 | — | | |

**SGDet (detected tracks, tracker categories)**

| | novel mAP | R@50 | R@100 | all mAP | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| EOV-MMP (baseline) | 15.79 | 16.36 | 19.01 | 27.18 | 17.10 | 19.46 |
| **+ rules** | **16.15** | 16.36 | 19.01 | **27.57** | 17.08 | 19.59 |
| gain | +0.36 | | | +0.39 | | |
| control: random deletions, same count (5 seeds) | 10.26 (sd 0.88) | 13.55 | 16.63 | — | | |
| METOR (published) | 16.75 | | | 27.53 | | |
| margin vs METOR | **−0.60** | | | +0.04 | | |

Control seeds, novel mAP: PredCls 16.99, 13.29, 13.67, 16.29, 13.82; SGDet 9.31,
9.14, 11.37, 10.68, 10.80. Each control deletes the same number of top-20 novel
entries per video as the rules: 12,607 in PredCls, 11,012 in SGDet.

## 4. Verdict

**Improves the baseline but not METOR.** SGDet + rules gains +0.36 novel and +0.39
all, but stays 0.60 below METOR on novel; the 0.04 above it on all is within
noise. **The gain is attributable to the rules:** the control is 7.44 (PredCls)
and 5.89 (SGDet) novel mAP below them, far outside the 0.3 margin.

## 5. Diagnostics (test labels used for reporting only)

**Deleted top-20 novel entries per verb** (slot level; the same count the control
matched):

| verb | PredCls | SGDet |
|---|---:|---:|
| stop (vehicle) | 4,702 | 3,753 |
| walk | 2,207 | 1,818 |
| run | 1,405 | 1,329 |
| stand | 1,246 | 1,078 |
| lie | 1,208 | 1,241 |
| sit | 788 | 808 |
| creep | 720 | 676 |
| swim | 331 | 309 |
| **total** | **12,607** | **11,012** |

**True positives lost.** Among the baseline's final novel outputs (top 200 per
video), **2,094 (PredCls) and 2,129 (SGDet) break a rule, and none of them
matched a GT relation** (the repo's `evaluate_v2` hit flags, vIoU 0.5). In the
final outputs, the rules removed only false positives.

**Per-predicate novel AP for the 34 rule-governed novel predicates with test
GT:**
- PredCls: 30 unchanged, 2 up (stop_beneath 61.1 → 66.7 with 13 instances;
  walk_above 4.8 → 7.1 with 12), 2 down (lie_above 11.1 → 0.0 with 4 instances;
  walk_past 9.8 → 9.6 with 34).
- SGDet: all 34 unchanged.
- 15 of the 34 have fewer than 5 test instances.

Full table: `data/t1_results.json` → `diagnostics.*.per_predicate_novel_ap`.

**So the gain does not come from the rule-governed predicates' own rankings.** It
comes from the per-video ranked list (Batch A, section 1). Wrong-type predictions,
mostly `stop_*` on people and animals and locomotion verbs on vehicles, were
outranking other predicates' true relations. Deleting them moves those relations
up, and the slot top-20 and video top-200 cuts admit predictions that were
previously crowded out.

## 6. Surprises and caveats

1. **The SGDet novel gain (+0.36) is about a third of the gap to METOR (0.96).**
   The +0.04 over METOR on all is within noise and is not a claim.
2. **The comparison with METOR crosses checkpoints.** Our EOV-MMP baseline
   (15.79 / 27.18) is the shipped checkpoint reproduced in session 3, which is
   already above the paper's EOV-MMP numbers (15.04 / 26.34). METOR's 16.75 /
   27.53 are its published numbers, not re-run here.
3. **The rules are almost free in PredCls, where subject categories are GT.** In
   SGDet they depend on the tracker's category. The gain there is smaller, and no
   per-predicate AP moves, which fits detection and tracking errors limiting what
   a type filter can fix.
4. **lie_above drops from 11.1 to 0.0** (4 instances) although no output true
   positive was deleted. Deletions change which predicates enter each slot's
   top 20, and that changes how association chains them. The same mechanism can
   move a predicate's AP either way.
5. **The random control is a strong negative.** Deleting random top-20 novel
   entries removes many correct high-scoring ones, so it falls far below the
   baseline. It shows that the rules' deletions are targeted. It is not a
   "no-op" baseline: that is the EOV-MMP row.
6. **This is the third test evaluation.** The rules were derived on FIT only,
   validated on HELD (F2) and frozen in A17 before this run; nothing was chosen
   from test numbers. Earlier test evaluations (B, B2) informed which ideas were
   tried, which this report discloses.

## 7. Hashes of new files

| file | SHA256 |
|---|---|
| `t1_run.py` (pinned by A17) | `02e309a006d2736678c7cd61a7ff7b27c9f5d4460021c3c56810a52812bb68dd` |
| `data/t1_results.json` (runs 1–5, verdict) | `5d5acbe2731fa33b78ef7e6279c291b1572b3b25f4aabba700bab41111d7c65a` |
| `data/t1_dry_results.json` (pre-A17 plumbing on HELD) | `df33923563bd5106c279110a75f1c7643b1119dc82501fe746660a12c86bf819` |
| `data/split_one/` (34 one-predicate split files; SHA256 of their sorted hash list) | `017bcf0e2c81abfe3540274c403d793b9da4bbeddb7239663dd55feff84ccf23` |

Logs: `logs/t1_run.json` and `logs/t1_dry.json`; console output in `t1_run.out`.
`data/` is git-ignored.
