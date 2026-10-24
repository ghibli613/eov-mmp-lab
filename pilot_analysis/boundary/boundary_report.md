# Boundary fix E1 — nested runs: report

> **Fourth test evaluation, disclosed. The method was motivated by the D1
> test-label diagnostic and fixed in A21 before any test number.**

**Verdict: FAIL. Nested runs lower test mAP in every setting.**
- **SGDet, (c) rules + nested:** **13.84 novel** (METOR 16.75, **−2.91**) and
  **23.35 all** (METOR 27.53, **−4.18**).
- **Against T1 (b):** SGDet −2.31 novel (95% CI −4.53 to +0.15) and −4.22 all
  (−5.68 to −2.74); PredCls −4.89 novel (−7.41 to −2.62) and −7.06 all
  (−8.93 to −5.33).
- **No configuration beats T1 in any setting.** The best remains (b), T1: SGDet
  16.15 / 27.57.
- **The fix does recover what it targeted.** Of D1's over-long cases, 15 / 129
  SGDet and 36 / 237 PredCls are now matched (T1: 0 and 1). Total GT matched
  rises in every setting.
- **But the added short instances are mostly false positives that rank above
  earlier hits:**
  - 0.35% precision for SGDet novel, against 0.70% for T1's lists;
  - 42–49% of the final top-200 lists.
- **The HELD plumbing (+2.38 mAP) pointed the other way.** HELD is in-sample for
  the checkpoint (section 6).

Run details:
- one scripted run, exit 0, 969 s, CPU;
- before it, A21 was hashed, step 1 (HELD) ran, and step 2 passed;
- earlier reports (B, B2, S1, F1, F2, T1, position, D1) stay unchanged.

## 1. A21 and the pinned artefacts

| | SHA256 |
|---|---|
| preregistration.md with A21 (`preregistration.sha256`, line 1) | `4bb0acbde101a2995156efcc214511144daf50103f92a2d1c382063a976349f3` |
| A21 text alone (6,411 bytes) | `e4a0ca5249ad58f33a48ecc6055760c8fbd35218d7aa0185222242cc32eecba1` |
| the first 83,051 bytes (through A20), unchanged | `71ab531522c6445395653ce94354a2bf233a09fa7c652c77c1c0aae8808a8c1e` |

- **64 pins verified** before every step (latest pin per path). They include
  `e1_run.py`, T1's results (the reference for (b)), D1's `tc_missed.json`, and
  every reused file: `common.py`, `eval_from_cache.py`, `span_common.py`,
  `tc_run.py`, F2's rules, `held_gt.json`, the post-processing and the
  evaluator.
- Every cache file read matched `SHA256SUMS_session3.txt`. 0 novel labels were
  used in training (asserted).
- Before A21, the script ran only on synthetic unit tests: a hand-built pair and
  a synthetic GT file. They checked:
  - nesting and pooling;
  - the dedupe;
  - the instance score;
  - the control's moves;
  - every evaluation path.
- **A18's T2 is withdrawn, not run.** The position verifier failed G-P3, so E1
  took the fourth evaluation's place.

**The method, as fixed in A21:**
1. T1's slot scores, with rule-violating predicates' logits set to −∞.
2. The repo's `process_pred` + `association` with per-slot top-k for
   k = 20, 10, 5, 3, 1, with no top-200 cut.
3. Pool, and deduplicate (pair, predicate, identical extent).
4. The repo's instance score, unchanged; the repo's top-200 cut.

The (e) rows pool k = 20 with a single tighter k. The control (f) moves every
extra to a random slot-aligned position inside its own k = 20 run, keeping its
length, predicate, pair and score (seeds 0–19).

## 2. Step 1 — HELD plumbing (base labels, descriptive only)

157 HELD videos (153 with base GT), base pass:

| | mAP | R@50 | R@100 | GT matched |
|---|---:|---:|---:|---:|
| (b) rules (T1) | 50.23 | 35.44 | 42.73 | 1,998 |
| **(c) rules + nested** | **52.62** | 39.66 | 49.28 | 2,333 |
| (f) control (20 seeds) | 50.98 (sd 0.25; max 51.62) | 37.84 | 46.91 | 2,215.5 |

- **(c) − (b) = +2.38** (95% CI 1.38 to 3.57).
- (b)'s outputs are identical to T1's own path.
- (c) vs (b): 351 GT newly matched, 16 lost.
- 3,693 extras were evaluated, with **381 hits (10.3%)**, about the same as
  (b)'s own list precision (10.8%).

## 3. Step 2 — label-free test checks (no labels, no evaluator)

| | SGDet novel | SGDet all | PredCls novel | PredCls all |
|---|---:|---:|---:|---:|
| **(b) identical to T1's path** | **yes** | **yes** | **yes** | **yes** |
| k = 20 instances (uncut) | 44,137 | 49,717 | 44,339 | 51,844 |
| extras per video: mean / median / max | 126 / 52 / 2,283 | 132 / 49 / 2,305 | 163 / 58 / 2,460 | 168 / 64 / 2,536 |
| (b) final outputs | 24,356 | 25,690 | 22,634 | 24,545 |
| (c) final outputs | 28,678 | 29,377 | 28,668 | 29,479 |
| **extras in (c)'s top 200** | **12,068 (42%)** | **12,308 (42%)** | **13,972 (49%)** | **14,031 (48%)** |
| videos with any extra in top 200 | 178 | 178 | 185 | 184 |
| (b) outputs pushed out | 7,746 | 8,621 | 7,938 | 9,097 |

(c) has more outputs than (b) because many videos had fewer than 200 instances
under T1. Extras fill those lists without pushing anything out.

**Length in frames by origin k** (SGDet novel, in (c)'s top 200; median
[p25–p75]):
- k = 20: 60 [30–105], 16,610 instances;
- k = 10: 30 [30–60], 4,171;
- k = 5: 30 [30–60], 3,746;
- k = 3: 30 [30–60], 2,582;
- k = 1: 30 [30–60], 1,569.

PredCls is similar; its tighter-k medians are 30 with p75 of 77–90.

**Neither stop condition fired:** the identity held everywhere, and extras
entered every setting's top 200.

## 4. Step 3 — the test evaluation

**Reproduction passed.**
- (a) is within ±0.05 of the references: −0.002, +0.001, −0.004, +0.005.
- (b) equals T1's saved mAP exactly (difference 0.0), with outputs identical to
  T1's path, in all four settings.

**SGDet (detected tracks, tracker categories)**

| | novel mAP | R@50 | R@100 | GT matched | all mAP | R@50 | R@100 | GT matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) EOV-MMP | 15.79 | 16.36 | 19.01 | 121 | 27.18 | 17.10 | 19.46 | 1,014 |
| (b) rules (T1) | **16.15** | 16.36 | 19.01 | 122 | **27.57** | 17.08 | 19.59 | 1,015 |
| **(c) rules + nested** | **13.84** | 15.37 | 18.68 | 128 | **23.35** | 15.76 | 19.83 | 1,121 |
| (d) nested, no rules | 12.46 | 15.21 | 18.68 | 128 | 22.87 | 15.80 | 19.81 | 1,118 |
| (e) rules + k = 10 only | 15.91 | 15.70 | 18.84 | 121 | 27.33 | 16.98 | 19.92 | 1,045 |
| (e) rules + k = 5 only | 15.42 | 15.87 | 19.34 | 125 | 26.48 | 16.92 | 19.94 | 1,072 |
| (e) rules + k = 3 only | 13.96 | 15.87 | 19.01 | 127 | 25.16 | 17.06 | 20.10 | 1,086 |
| (e) rules + k = 1 only | 15.50 | 17.19 | 20.00 | 130 | 24.71 | 16.36 | 19.52 | 1,058 |
| (f) control (20 seeds) | 13.01 (sd 0.40) | 14.88 | 17.96 | 123.8 | 22.74 (sd 0.17) | 14.93 | 18.54 | 1,047.4 |
| METOR (published) | 16.75 | | | | 27.53 | | | |
| **(c) margin vs METOR** | **−2.91** | | | | **−4.18** | | | |

**PredCls (GT tracks and categories)**

| | novel mAP | R@50 | R@100 | GT matched | all mAP | R@50 | R@100 | GT matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) EOV-MMP | 21.70 | 31.57 | 35.54 | 244 | 37.20 | 28.58 | 34.62 | 1,899 |
| (b) rules (T1) | **22.25** | 31.57 | 36.03 | 244 | **38.03** | 28.62 | 34.73 | 1,901 |
| **(c) rules + nested** | **17.36** | 30.58 | 37.36 | 267 | **30.97** | 24.61 | 33.49 | 2,012 |
| (d) nested, no rules | 15.51 | 29.42 | 37.02 | 266 | 29.83 | 24.38 | 33.34 | 2,004 |
| (e) rules + k = 10 only | 21.60 | 30.25 | 36.86 | 252 | 37.27 | 28.52 | 34.83 | 1,959 |
| (e) rules + k = 5 only | 21.17 | 31.24 | 38.18 | 263 | 35.92 | 28.25 | 35.14 | 1,987 |
| (e) rules + k = 3 only | 19.25 | 31.24 | 38.35 | 264 | 33.84 | 27.51 | 34.83 | 1,976 |
| (e) rules + k = 1 only | 18.46 | 30.91 | 37.19 | 255 | 33.17 | 26.70 | 33.88 | 1,949 |
| (f) control (20 seeds) | 16.67 (sd 0.33) | 28.93 | 36.11 | 259.5 | 29.78 (sd 0.20) | 23.16 | 31.09 | 1,861.7 |

GT matched counts are from the repo's matcher over the videos with GT for the
pass: 136 videos for novel, 200 for all.

**(c) with its bootstrap CIs** (1,000 video resamples, seed 0):

| | (c) mAP | 95% CI | (c) − (b) | 95% CI |
|---|---:|---|---:|---|
| SGDet novel | 13.84 | 8.92 to 18.72 | −2.31 | −4.53 to +0.15 |
| SGDet all | 23.35 | 19.30 to 27.41 | −4.22 | −5.68 to −2.74 |
| PredCls novel | 17.36 | 12.48 to 22.25 | −4.89 | −7.41 to −2.62 |
| PredCls all | 30.97 | 27.13 to 35.07 | −7.06 | −8.93 to −5.33 |

METOR's 16.75 lies inside (c)'s SGDet novel CI, which is wide: 136 videos. It
does not lie inside the all CI. Either way (c) is below METOR on both, so the
primary claim **FAILS**.

## 5. Match-level breakdown and the control

**(c) vs (b):**

| | GT newly matched | GT lost | unmatched predictions added | extras evaluated | extra hits (precision) | (b)'s list precision |
|---|---:|---:|---:|---:|---:|---:|
| SGDet novel | 16 | 10 | +3,013 | 9,044 | 32 (0.35%) | 0.70% |
| SGDet all | 192 | 86 | +3,581 | 12,308 | 321 (2.6%) | 3.95% |
| PredCls novel | 45 | 22 | +4,413 | 11,078 | 80 (0.72%) | 1.44% |
| PredCls all | 389 | 278 | +4,823 | 14,031 | 771 (5.5%) | 7.74% |
| *HELD (step 1)* | *351* | *16* | *+1,394* | *3,693* | *381 (10.3%)* | *10.8%* |

"Lost" means matched in (b) but not in (c).

**D1's over-long cases** (recomputed by `tc_posthoc.py`'s procedure; the counts
reproduce exactly):

| | over-long cases | matched in (a) | (b) | (c) | (d) |
|---|---:|---:|---:|---:|---:|
| SGDet | 129 (expected 129) | 0 | 0 | **15** | 15 |
| PredCls | 237 (expected 237) | 1 | 1 | **36** | 35 |

**Control.** (c) is above every one of the 20 control seeds in all four
settings:

| | (c) − mean (f) |
|---|---:|
| SGDet novel | +0.83 |
| SGDet all | +0.62 |
| PredCls novel | +0.69 |
| PredCls all | +1.19 |

Placing the shorter instances where the tighter top-k puts them is better than
placing them at random inside the same run. Both are far below T1.

## 6. Reading

1. **The mechanism works as intended, and the gain is small.**
   - The nested instances do recover over-long cases: 12% of D1's SGDet set and
     15% of the PredCls set.
   - GT matched rises in every setting: SGDet all +106, PredCls all +111.
   - The control confirms the placement carries signal.
2. **The cost is larger, and AP is where it shows.**
   - Every run can spawn sub-runs at four tighter k. Each sub-run's score is
     the mean over the slots where the predicate ranks highest, which should
     usually put it at or above its parent run. This was not measured
     separately.
   - On test they hit about half as often as T1's own outputs (novel) or
     two-thirds as often (all).
   - Since GT matched rises while mAP falls, the added false positives must sit
     above earlier hits in each video's ranked list. R@100 is flat or slightly
     up; R@50 and mAP fall.
3. **The single-k rows show the same trade-off in smaller doses.** k = 10 alone
   is nearly neutral: −0.24 SGDet on both novel and all. k = 1 alone has SGDet
   novel's best R@50 / R@100 (17.19 / 20.00). None beats T1 in mAP. These rows
   are descriptive; nothing is chosen from them.
4. **The type rules still help on top of nested runs:** (c) − (d) is +1.38 /
   +0.48 (SGDet) and +1.85 / +1.14 (PredCls).

## 7. Surprises and caveats

1. **HELD and test disagree in sign: +2.38 vs −2.31 to −7.06.** The likely
   reason: EOV-MMP was trained on all 800 training videos, HELD included.
   - On HELD, a predicate's top-1 slots are in-sample and usually right: the
     extras hit as often as T1's own outputs (10.3% vs 10.8%).
   - On test they hit about half as often.

   A21 marked step 1 as descriptive only, and nothing was decided from it. But it
   shows that HELD cannot validate post-processing changes for this checkpoint.
   VidVRD has no split disjoint from the checkpoint's training data except test.
2. **The D1 diagnostic was right about the defect but not about its value.**
   Fixing the boundaries for the 366 over-long cases is possible. But doing it
   without labels means proposing sub-runs for every run, and the wrong ones far
   outnumber the right ones.
3. **The SGDet novel difference's CI just reaches zero** (−4.53 to +0.15). The
   other three settings are clearly negative.
4. **This is the fourth test evaluation.** The method came from a test-label
   diagnostic (D1), was fixed in A21 before any test number for it, and ran
   once. It does not change T1's standing as the best system: SGDet 16.15 /
   27.57, 0.60 below METOR on novel.

## 8. Hashes of new files

| file | SHA256 |
|---|---|
| `e1_run.py` (pinned by A21) | `041e09e13d98999568dc5411f192451cc0cd6db5f488087699913162920b9cbb` |
| `data/e1_step3.json` (the test evaluation, verdict) | `801f4ea47c8617b767856a47490a59471e0660efd29c28a34de92f8afb2401f3` |
| `data/e1_step2.json` (label-free checks) | `7a25f7f0c1be1c69f0227be237876121b5d7f44342f6c0efe1d86ac43bf7f507` |
| `data/e1_step1.json` (HELD plumbing) | `ae20ef84436b2e90116fa7f205779e9e4db0ac0163d417c26d86dfdfb7d91bd1` |
| `data/e1_step1_small.json`, `data/e1_step2_small.json` | `b65bce16…`, `08ce30f0…` |

Logs: `logs/e1_step{1,2,3}.json` and the `*_small.json` plumbing runs. Console
output is in `e1_step{1,2,3}_full.out`. `data/` is git-ignored.
