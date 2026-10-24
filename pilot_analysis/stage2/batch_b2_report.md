# Stage 2, Batch B2 — recipes reorder within each predicate: report

**Verdict: 21.80 novel mAP, above EOV-MMP's 21.70 but below the 24.70 gate.**
Under A9's decision rule, "the method helps but not enough to target METOR; method
work ends." The gain is **+0.10**, small enough that it should not be read as a
real improvement (caveat 1).

**Control verdict: B2 beats shuffled recipes by 3.00** (21.80 vs 18.80), so the
recipes carry real ordering information. But the reordering they produce is only
about as good as EOV-MMP's own.

B2 is a **second attempt, designed after Batch B's test result** (A9). Batch B's
result (15.37, FAIL) stands unchanged. The test run was one scripted run (exit 0),
started after the user's `GO B2-TEST`. SGDet and the rewording run were not run.

## 1. A9 and the pinned artefacts

| | SHA256 |
|---|---|
| preregistration.md with A9 | `86a0f18a16f08edc91536d43dc1ec417dbc8d5787e040c204907d04b3fcb6538` |
| A9 text alone (5,186 bytes) | `80eb10534b7e0ff4e14ffd3d577a47542aab349b443f70c948639309f8e771c6` |
| the first 39,586 bytes (through A8), unchanged | `ca7ad1b7ae949f0a75eae13afc46f52e7c6f2027ae09a387580227e30875ba7d` |
| `b2_run.py` (pinned by A9) | `55ac092fd39783ae016a938cc61e1c9dd99fcb6aa17b1fde58cc4465cb9e0ef0` |

- **All 50 pinned artefacts were verified before every step and after the test
  run.**
- Batch B's four test-side files (`b_test_slots.npz`, `b_test_learned_full.npy`,
  `b_test_pairs.json`, `b_rprime_novel_test.npy`) were verified only in the test
  run, after GO, against `batch_b_report.md`. All matched.
- The novel r′ the script recomputes equals Batch B's stored r′ exactly
  (asserted).
- 0 novel labels were used in training (asserted). Nothing was trained or
  fitted.

## 2. Part 1 (HELD, 71 base predicates, full learned part)

| scores | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| EOV-MMP alone | 50.28 | 35.46 | 42.73 |
| B2 | 49.04 | 35.03 | 42.06 |
| control | 49.04 | 35.03 | 42.06 |

**Part 1: PASS** (49.04 ≥ 50.28 − 2.0; invariant 11,147 / 11,147). The control
is identical to B2 on HELD by construction, as A9 stated: the permutation moves
novel predicates only. Details are in `batch_b2_interim.md`.

## 3. Baseline check (run 1)

| EOV-MMP alone | mAP | R@50 | R@100 | reference | diff | = Task-4 path |
|---|---:|---:|---:|---:|---:|---|
| novel | 21.70 (21.698) | 31.57 | 35.54 | 21.70 | −0.002 | yes, exactly |
| all | 37.20 (37.201) | 28.58 | 34.62 | 37.20 | +0.001 | yes, exactly |

Passed.

## 4. THE GATE (run 2)

| novel split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| EOV-MMP alone | 21.70 | 31.57 | 35.54 |
| **B2** | **21.80** | 31.57 | 36.03 |
| gate | 24.70 | | |

**Verdict: 21.80 is above 21.70 but below 24.70. The method helps, but not
enough to target METOR. Method work ends.**

## 5. Control (run 4)

| novel split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| B2 | 21.80 | 31.57 | 36.03 |
| control: shuffled recipes, same reordering | 18.80 | 29.26 | 34.38 |

**Control verdict: B2 beats the control by 3.00 (≥ 1.0).** Reordering by the
wrong predicate's recipe costs 2.90 against EOV-MMP's own order. Reordering by
the right recipe roughly preserves it (+0.10).

## 6. All split

| all split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| baseline (EOV-MMP alone) | 37.20 | 28.58 | 34.62 |
| B2 (run 3) | 37.13 | 28.52 | 34.46 |
| control (run 4) | 36.55 | 28.42 | 34.19 |

## 7. Reordering keyed by r′ alone (run 5, descriptive)

| key | novel mAP | R@50 | R@100 | all mAP | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| √(EOV × r′) (B2) | 21.80 | 31.57 | 36.03 | 37.13 | 28.52 | 34.46 |
| r′ alone | 19.03 | 27.44 | 33.22 | 33.76 | 25.92 | 31.95 |

Dropping EOV from the key loses 2.77 novel mAP. EOV-MMP's own within-predicate
order holds information that the recipe alone does not.

## 8. Per-predicate novel AP: baseline vs B2

Each AP comes from the official function with only that predicate competing,
as in Batch B.
- ⚠ = fewer than 5 test instances: **30 of the 61 novel predicates**. Read
  them as anecdotes.
- † = no prediction in any of the predicate's GT videos, so AP 0 by
  definition.

Over the 61: 21 go up, 10 go down, 30 are unchanged.

**(a) bare spatial**: 5 predicates with test GT. Mean AP 37.67 → 39.50; median 54.17 → 54.17.

| predicate | test instances | baseline AP | B2 AP | Δ |
|---|---:|---:|---:|---:|
| `above` | 5 | 5.00 | 10.00 | +5.00 |
| `away` | 7 | 0.00 | 0.00 | +0.00 |
| `beneath` | 7 | 62.50 | 66.67 | +4.17 |
| `past` | 6 | 66.67 | 66.67 | +0.00 |
| `toward` | 6 | 54.17 | 54.17 | +0.00 |

**(b) `*_next_to`**: 6 predicates with test GT. Mean AP 34.69 → 32.28; median 40.54 → 39.75.

| predicate | test instances | baseline AP | B2 AP | Δ |
|---|---:|---:|---:|---:|
| `creep_next_to` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `move_next_to` | 8 | 30.00 | 30.00 | +0.00 |
| `run_next_to` | 8 | 9.00 | 10.00 | +1.00 |
| `sit_next_to` | 38 | 51.07 | 49.51 | -1.57 |
| `swim_next_to` | 25 | 53.95 | 49.63 | -4.31 |
| `stop_next_to` | 10 | 64.14 | 54.53 | -9.61 |

**(c) no recipe**: 5 predicates with test GT. Mean AP 32.75 → 33.14; median 33.33 → 30.00.

| predicate | test instances | baseline AP | B2 AP | Δ |
|---|---:|---:|---:|---:|
| `bite` | 4 ⚠ | 0.00† | 0.00† | +0.00 |
| `feed` | 9 | 33.33 | 35.71 | +2.38 |
| `fight` | 8 | 36.67 | 30.00 | -6.67 |
| `hold` | 8 | 93.75 | 100.00 | +6.25 |
| `kick` | 1 ⚠ | 0.00† | 0.00† | +0.00 |

**(d) all others**: 45 predicates with test GT. Mean AP 22.84 → 26.68; median 11.11 → 14.29.

| predicate | test instances | baseline AP | B2 AP | Δ |
|---|---:|---:|---:|---:|
| `creep_above` | 4 ⚠ | 0.00 | 0.00 | +0.00 |
| `creep_away` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `creep_past` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `creep_toward` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `drive` | 3 ⚠ | 33.33 | 50.00 | +16.67 |
| `fly_away` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `fly_behind` | 2 ⚠ | 0.00 | 25.00 | +25.00 |
| `fly_front` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `fly_left` | 2 ⚠ | 50.00 | 50.00 | +0.00 |
| `fly_right` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `jump_above` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `jump_away` | 3 ⚠ | 0.00 | 0.00 | +0.00 |
| `jump_behind` | 8 | 20.00 | 20.00 | +0.00 |
| `jump_past` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `jump_with` | 2 ⚠ | 0.00† | 0.00 | +0.00 |
| `lie_above` | 4 ⚠ | 11.11 | 0.00 | -11.11 |
| `lie_beneath` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `lie_inside` | 1 ⚠ | 50.00 | 100.00 | +50.00 |
| `lie_with` | 2 ⚠ | 13.33 | 14.29 | +0.95 |
| `move_above` | 1 ⚠ | 50.00 | 25.00 | -25.00 |
| `move_away` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `move_past` | 11 | 33.33 | 35.42 | +2.08 |
| `move_toward` | 8 | 35.42 | 58.33 | +22.92 |
| `pull` | 3 ⚠ | 66.67 | 66.67 | +0.00 |
| `run_above` | 1 ⚠ | 100.00 | 100.00 | +0.00 |
| `run_away` | 8 | 30.00 | 40.00 | +10.00 |
| `run_beneath` | 5 | 40.00 | 60.00 | +20.00 |
| `run_toward` | 9 | 3.39 | 2.98 | -0.41 |
| `sit_behind` | 47 | 50.06 | 52.64 | +2.58 |
| `sit_beneath` | 20 | 14.38 | 20.73 | +6.35 |
| `sit_inside` | 3 ⚠ | 100.00 | 100.00 | +0.00 |
| `stand_above` | 35 | 71.73 | 66.49 | -5.24 |
| `stand_beneath` | 6 | 0.00 | 0.00 | +0.00 |
| `stand_inside` | 2 ⚠ | 0.00† | 0.00† | +0.00 |
| `stand_with` | 64 | 23.01 | 23.03 | +0.01 |
| `stop_above` | 2 ⚠ | 8.33 | 8.33 | +0.00 |
| `stop_beneath` | 13 | 61.11 | 61.11 | +0.00 |
| `stop_with` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `swim_behind` | 13 | 12.72 | 15.88 | +3.16 |
| `swim_beneath` | 1 ⚠ | 50.00 | 100.00 | +50.00 |
| `walk_above` | 12 | 4.76 | 7.14 | +2.38 |
| `walk_away` | 47 | 17.93 | 19.74 | +1.81 |
| `walk_beneath` | 17 | 61.67 | 60.00 | -1.67 |
| `walk_past` | 34 | 9.81 | 13.04 | +3.23 |
| `walk_toward` | 43 | 5.57 | 4.73 | -0.84 |

The nine predicates with at least 20 test instances:

| predicate | test instances | baseline | B2 |
|---|---:|---:|---:|
| sit_behind | 47 | 50.1 | 52.6 |
| sit_beneath | 20 | 14.4 | 20.7 |
| sit_next_to | 38 | 51.1 | 49.5 |
| stand_above | 35 | 71.7 | 66.5 |
| stand_with | 64 | 23.0 | 23.0 |
| swim_next_to | 25 | 53.9 | 49.6 |
| walk_away | 47 | 17.9 | 19.7 |
| walk_past | 34 | 9.8 | 13.0 |
| walk_toward | 43 | 5.6 | 4.7 |

5 go up, 3 go down, 1 is unchanged.

## 9. Invariant

**Exact everywhere.** Every (video, competing predicate) pair keeps EOV-MMP's
multiset of score values in every run, and the no-ECC video is unchanged:

| run | checks held |
|---|---:|
| B2, novel | 12,201 / 12,201 |
| B2, all | 26,401 / 26,401 |
| control, novel | 12,201 / 12,201 |
| control, all | 26,401 / 26,401 |
| r′-keyed, novel | 12,201 / 12,201 |
| r′-keyed, all | 26,401 / 26,401 |

1 test video (`ILSVRC2015_train_00250013`, no ECC) kept EOV-MMP's scores
unchanged.

## 10. Surprises and caveats

1. **+0.10 is not evidence of help.** It is smaller than the effect of one
   detection flipping in a few-instance predicate. No uncertainty interval was
   pre-registered, and none is computed after the fact. The decision rule's
   wording ("helps") is applied as written, but the defensible reading is: **B2
   is indistinguishable from EOV-MMP alone on test novel PredCls.**
2. **Batch B's diagnosis is confirmed.** With cross-predicate calibration held
   fixed by construction, the aggregate no longer falls (15.37 in B → 21.80
   here). The per-predicate pattern is the same as in B: 21 up against 10 down,
   and group (d)'s mean rises from 22.84 to 26.68. The recipes do improve the
   order within many predicates, but not enough to move the per-video mAP the
   evaluator reports.
3. **The recipes are informative relative to the wrong recipes, not relative to
   EOV-MMP.** Correct recipes: +0.10 against EOV-MMP's own order. Shuffled:
   −2.90. r′ alone as the key: −2.67. EOV-MMP's within-predicate order is
   already about as good as the recipes can make it.
4. **Predicates without a recipe still change AP.** `fight` goes 36.67 → 30.00
   with no reassigned scores. Other predicates' values moved between slots,
   which changes which predicates make each slot's top 20 and what association
   merges. The invariant holds per video and predicate, not per slot, so the
   post-processing is not invariant.
5. **Small n.** 30 of 61 novel predicates have under 5 test instances.
6. **One run, as fixed.** A9 → Part 1 → interim → GO → one test run (exit 0).
   No code change after A9.

## 11. Hashes of new files

| file | SHA256 |
|---|---|
| `b2_run.py` (pinned by A9) | `55ac092fd39783ae016a938cc61e1c9dd99fcb6aa17b1fde58cc4465cb9e0ef0` |
| `data/b2_part1_results.json` | `466ff6785dc99b24acbf2d27d1ca58f3e70b27c0ddb9cbb4ff8d02d251e01c22` |
| `data/b2_dry_results.json` (test path on HELD, plumbing) | `6ef176864af6c95efa8e670f78903772053e202ebecf619fa16f0d4199ff44e0` |
| `data/b2_test_results.json` (runs 1–5, per-predicate AP, invariant) | `46d0c71774b74946d4d8f53fd67d81fa4043ef46dcbc9ebe70f37d71ebaeae02` |
| `data/b2_test_split_one/` (61 one-predicate split files; SHA256 of their sorted hash list, identical to Batch B's) | `b01e8aeaaf77138c45de09f2b85ab0c9355ff96cfcc7ba6fdb5a41df33b43ca6` |

The run's logs are `logs/b2_test_run.json` and `logs/b2_test.out`; Part 1's and the
dry run's are next to them. `data/` is git-ignored.
