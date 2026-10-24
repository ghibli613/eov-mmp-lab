# Stage 2, Batch B — the frozen method on the test set: report

**Verdict: GATE FAIL.** The method's novel PredCls mAP is **15.37**, against a
gate of **24.70**, and **6.33 below EOV-MMP alone (21.70)**.

As A8 fixes, the method stops and is reported as a negative result: no changes,
no re-runs. The control line: the method beats shuffled recipes by **4.65** novel
mAP (15.37 vs 10.72), so the recipes carry signal. But adding them lowers novel
mAP.

Runs 1–6 were one scripted run (`b_run.py`, one execution, exit 0). Run 7 is
deferred (section 7). SGDet was not run.

## 1. A8 and the pinned artefacts

A8 was appended to `pilot_analysis/stage1b/preregistration.md` on 2026-10-05,
**before any stage-2 read of a test file**. It pins every script the run uses.

| | SHA256 |
|---|---|
| preregistration.md with A8 (`preregistration.sha256`, line 1) | `ca7ad1b7ae949f0a75eae13afc46f52e7c6f2027ae09a387580227e30875ba7d` |
| A8 text alone (8,739 bytes) | `bed306e3e94c72dce8b6992d9591a30189371d7569669076d628b0ba0b455381` |
| the first 30,847 bytes (through A7), unchanged | `2b34d0ef59a167a32f9313b7857355faed631ca41ca87d00f963218ab2d3dadc` |

**Every pinned artefact is unchanged.** All 49 were verified before each step and
again after the run:
- the frozen schema, rules, split and paraphrases;
- A5's tracks and ECC files;
- A6's selection;
- Batch A's 13 outputs;
- A7's LO2 selection;
- A8's coverage, recipes, shipped test ECC, and all 14 scripts, including the
  repo's post-processing and evaluator.

0 novel labels were used in training (asserted again in the run).

Before A8, the code was tested on training data only:
- `b_step1_test_parts.py --plumb-train 5` reproduces step 1's stored values.
- `b_run.py --dry-run` runs the whole pipeline on HELD.
- The EOV score path was confirmed bit-identical to session 3's Task-4 path on
  HELD; see caveat 6.

**Test parts (step 1).**
- 200 videos, **1,392 pairs**, the same as Task 4, matching the cache pair for
  pair (0 pairs over 40 slots); 8,508 slots.
- Swap and time-reversal symmetry is exact on every check: 34,024 / 34,024 per
  rule, and 8,506 / 8,506 for D2.

## 2. Baseline check (run 1)

| EOV-MMP alone | mAP | R@50 | R@100 | reference | diff | = Task-4 path |
|---|---:|---:|---:|---:|---:|---|
| novel | 21.70 (21.698) | 31.57 | 35.54 | 21.70 | −0.002 | yes, exactly |
| all | 37.20 (37.201) | 28.58 | 34.62 | 37.20 | +0.001 | yes, exactly |

**Passed** (±0.05).

## 3. THE GATE (run 2)

| novel split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| EOV-MMP alone (run 1) | 21.70 | 31.57 | 35.54 |
| **method √(EOV × r′)** | **15.37** | 31.24 | 36.86 |
| gate | **24.70** | | |

**Verdict: FAIL.** 15.37 < 24.70, and the method is 6.33 below the baseline.

## 4. Control (run 4)

| novel split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| method | 15.37 | 31.24 | 36.86 |
| control: shuffled recipes (A6 permutation) | 10.72 | 26.61 | 33.39 |

**Control verdict: the method beats the control by 4.65 (≥ 1.0).** The recipes'
content matters relative to a shuffled assignment. Both are below EOV-MMP alone.

## 5. All split (all 132 compete)

| all split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| baseline (EOV-MMP alone) | 37.20 | 28.58 | 34.62 |
| method √(EOV × r′) (run 3) | 38.75 | 29.18 | 35.78 |
| control, shuffled recipes (run 4) | 38.93 | 29.80 | 35.78 |

In the all split the method is +1.55 over the baseline, but the control is higher
still (+1.73). That gain cannot be attributed to the recipes.

## 6. Recipe alone and weight sensitivity (descriptive)

| novel split | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| r′ alone (run 5) | 7.69 | 21.65 | 27.44 |

| mix EOV^(1−w) · r′^w (run 6) | novel mAP | R@50 | R@100 | all mAP | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| w = 0.25 | 18.06 | 32.23 | 37.36 | 40.30 | 29.49 | 35.70 |
| w = 0.5 (the method) | 15.37 | 31.24 | 36.86 | 38.75 | 29.18 | 35.78 |
| w = 0.75 | 12.89 | 28.26 | 35.21 | 34.01 | 26.23 | 32.60 |

Novel mAP falls monotonically as the recipe's weight rises. Every weight is below
EOV-MMP alone (21.70). Not selectable.

## 7. Rewording robustness (run 7): **deferred**

Session 3 saved no reworded per-slot scores (`task5/` holds only
`task5_paraphrase.json`), so EOV-MMP's text side must be re-run under each
wording. Measured on 5 training videos with the run-7 script, on this machine's
CPU:
- 9.57 s per pair-window, plus 58.7 s per set of strings;
- logits within 0.0015 of the cache, so the CPU path is correct.

For 1,392 test pairs × 5 conditions that is **about 18.6 h**, over the 2-hour
limit. As rule 7 and A8 require, run 7 is done later with A8 unchanged
(`b_run7_rewording.py`, pinned), on a GPU (well under $1) or the CPU. Its table
will be added here.

With the gate failed, whether run 7 is still wanted is the chat's call. It
would measure whether the recipe term reduces wording sensitivity, which stays
informative even for a failed method.

## 8. Per-predicate novel AP: baseline (run 1) vs method (run 2)

Each AP comes from the official function with only that predicate competing.
- ⚠ = fewer than 5 test instances: **30 of the 61 novel predicates**. One hit
  changes their AP by 20–100 points, so treat them as anecdotes.
- † = no prediction of the predicate in any of its GT videos, so AP 0 by
  definition.

Over the 61, AP goes up for 24 predicates, down for 15, and is unchanged for 22.

**(a) bare spatial**: 5 predicates with test GT. Mean AP 37.67 → 39.29; median 54.17 → 53.12.

| predicate | test instances | baseline AP | method AP | Δ |
|---|---:|---:|---:|---:|
| `above` | 5 | 5.00 | 10.00 | +5.00 |
| `away` | 7 | 0.00 | 0.00 | +0.00 |
| `beneath` | 7 | 62.50 | 66.67 | +4.17 |
| `past` | 6 | 66.67 | 66.67 | +0.00 |
| `toward` | 6 | 54.17 | 53.12 | -1.04 |

**(b) `*_next_to`**: 6 predicates with test GT. Mean AP 34.69 → 32.25; median 40.54 → 38.47.

| predicate | test instances | baseline AP | method AP | Δ |
|---|---:|---:|---:|---:|
| `creep_next_to` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `move_next_to` | 8 | 30.00 | 28.47 | -1.53 |
| `run_next_to` | 8 | 9.00 | 7.14 | -1.86 |
| `sit_next_to` | 38 | 51.07 | 52.04 | +0.96 |
| `swim_next_to` | 25 | 53.95 | 48.47 | -5.48 |
| `stop_next_to` | 10 | 64.14 | 57.36 | -6.78 |

**(c) no recipe**: 5 predicates with test GT. Mean AP 32.75 → 33.95; median 33.33 → 31.67.

| predicate | test instances | baseline AP | method AP | Δ |
|---|---:|---:|---:|---:|
| `bite` | 4 ⚠ | 0.00† | 0.00 | +0.00 |
| `feed` | 9 | 33.33 | 38.10 | +4.76 |
| `fight` | 8 | 36.67 | 31.67 | -5.00 |
| `hold` | 8 | 93.75 | 100.00 | +6.25 |
| `kick` | 1 ⚠ | 0.00† | 0.00† | +0.00 |

**(d) all others**: 45 predicates with test GT. Mean AP 22.84 → 29.06; median 11.11 → 17.66.

| predicate | test instances | baseline AP | method AP | Δ |
|---|---:|---:|---:|---:|
| `creep_above` | 4 ⚠ | 0.00 | 0.00 | +0.00 |
| `creep_away` | 1 ⚠ | 0.00 | 0.00† | +0.00 |
| `creep_past` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `creep_toward` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `drive` | 3 ⚠ | 33.33 | 33.33 | +0.00 |
| `fly_away` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `fly_behind` | 2 ⚠ | 0.00 | 25.00 | +25.00 |
| `fly_front` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `fly_left` | 2 ⚠ | 50.00 | 100.00 | +50.00 |
| `fly_right` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `jump_above` | 2 ⚠ | 0.00 | 50.00 | +50.00 |
| `jump_away` | 3 ⚠ | 0.00 | 0.00† | +0.00 |
| `jump_behind` | 8 | 20.00 | 20.00 | +0.00 |
| `jump_past` | 2 ⚠ | 0.00 | 50.00 | +50.00 |
| `jump_with` | 2 ⚠ | 0.00† | 0.00† | +0.00 |
| `lie_above` | 4 ⚠ | 11.11 | 0.00 | -11.11 |
| `lie_beneath` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `lie_inside` | 1 ⚠ | 50.00 | 100.00 | +50.00 |
| `lie_with` | 2 ⚠ | 13.33 | 15.38 | +2.05 |
| `move_above` | 1 ⚠ | 50.00 | 33.33 | -16.67 |
| `move_away` | 1 ⚠ | 0.00 | 0.00 | +0.00 |
| `move_past` | 11 | 33.33 | 35.42 | +2.08 |
| `move_toward` | 8 | 35.42 | 56.94 | +21.53 |
| `pull` | 3 ⚠ | 66.67 | 0.00 | -66.67 |
| `run_above` | 1 ⚠ | 100.00 | 100.00 | +0.00 |
| `run_away` | 8 | 30.00 | 40.00 | +10.00 |
| `run_beneath` | 5 | 40.00 | 60.00 | +20.00 |
| `run_toward` | 9 | 3.39 | 3.09 | -0.30 |
| `sit_behind` | 47 | 50.06 | 57.91 | +7.85 |
| `sit_beneath` | 20 | 14.38 | 19.31 | +4.92 |
| `sit_inside` | 3 ⚠ | 100.00 | 66.67 | -33.33 |
| `stand_above` | 35 | 71.73 | 68.15 | -3.57 |
| `stand_beneath` | 6 | 0.00 | 0.00 | +0.00 |
| `stand_inside` | 2 ⚠ | 0.00† | 50.00 | +50.00 |
| `stand_with` | 64 | 23.01 | 22.39 | -0.62 |
| `stop_above` | 2 ⚠ | 8.33 | 8.33 | +0.00 |
| `stop_beneath` | 13 | 61.11 | 69.44 | +8.33 |
| `stop_with` | 2 ⚠ | 0.00 | 0.00 | +0.00 |
| `swim_behind` | 13 | 12.72 | 17.66 | +4.93 |
| `swim_beneath` | 1 ⚠ | 50.00 | 100.00 | +50.00 |
| `walk_above` | 12 | 4.76 | 7.14 | +2.38 |
| `walk_away` | 47 | 17.93 | 16.97 | -0.97 |
| `walk_beneath` | 17 | 61.67 | 60.00 | -1.67 |
| `walk_past` | 34 | 9.81 | 12.93 | +3.12 |
| `walk_toward` | 43 | 5.57 | 8.33 | +2.76 |

**The aggregate and the per-predicate numbers disagree.** Per-predicate AP
improves more often than it falls (group d's mean rises from 22.84 to 29.06), and
R@100 rises (35.54 → 36.86). Yet the single novel mAP falls by 6.33. AP is
computed per video over one ranked list of all 61 novel predicates (Batch A,
section 1). So the method finds as many or more true relations, and ranks them
better *within* each predicate, but ranks predicates worse *against each
other*: false positives of some predicates outrank true positives of others.
This is the cross-predicate calibration problem from Batch A, now on the test
set. The coverage correction did not fix it there.

## 9. Fallback to EOV-MMP only

**1 test video** (`ILSVRC2015_train_00250013`, 2 slots) is missing from the
shipped `VidVRD_ECC_test.json`. As A8 requires, it used EOV-MMP's score alone in
every run.

## 10. Surprises and caveats

1. **The gate failure is not marginal.** The method is below the baseline, not
   just below the gate. Every recipe weight lowers novel mAP, and the recipe
   alone scores 7.69. Training-side checks (Batch A2 step 3: 50.28 → 46.52; the
   dry run: 50.28 → 46.61) had already shown the mix lowering mAP on HELD, where
   EOV-MMP is strong. The test set shows it lowering mAP where EOV-MMP is weak too.
2. **Per-predicate gains, aggregate loss** (section 8). The recipes carry
   real per-predicate signal: 24 up against 15 down, and +4.65 over the shuffled
   control. But the recipe scores are not calibrated across predicates.
3. **The coverage correction boosts the least-validated parts most.** For novel
   predicates, c^−0.25 ranges from 1.30 to 1.97. The largest boosts go to the
   `*_toward` / `*_away` recipes (c ≈ 0.07–0.08), whose measured approach/recede
   part stage 1 flagged as unvalidated on base data (12 and 0 base instances).
   This is a diagnostic from the pinned coverage file, not a tested cause.
4. **The all-split gain belongs to the control.** Method +1.55 against control
   +1.73.
5. **Small n.** 30 of the 61 novel predicates have fewer than 5 test instances.
   The per-predicate table is descriptive only.
6. **The EOV path is bit-identical to Task 4 because of a pre-A8 fix.** A
   first version computed EOV as one float32/float64 matrix. On HELD it gave
   the same mAP, but 10,740 prediction scores differed from Task 4's by up to
   1.2e-7: torch's CPU sigmoid rounds differently with tensor size. Before A8,
   EOV was switched to Task 4's own per-pair `slot_scores`, after which HELD
   predictions were bit-identical. Run 1 then matched the Task-4 path exactly
   on test.
7. **No re-runs, no crashes.** Each test step ran once (step 1 exit 0, the
   single run exit 0), with no code change after A8.
8. **Run 7 is pending** (section 7).

## 11. Hashes of new files

Pinned by A8:

| file | SHA256 |
|---|---|
| `b_step0.json` | `6d659345546da778eff705339cc9fdc571e608759735dd258c655a64255dd499` |
| `data/b_coverage_full.json` | `4d3168190da71f80f794c428908c6d005567403877441187e677ad3a5c447f88` |
| `data/b_run7_timing_cpu.json` | `7dbaed4077ccd6ae8257d4d017e5585db4270c9a1667e1b25722497ceb187654` |

Test outputs:

| file | SHA256 |
|---|---|
| `data/b_test_slots.npz` (test part scores, logits) | `787f18f0376484978b934250e30fe37287b12c9ba77ad41dadc03bd47aa3b073` |
| `data/b_test_emb.npy` | `c1a1edf754b9df22ed7fea72f74d0ac2bf982e1af1366a3e81f9716364ba5279` |
| `data/b_test_learned_full.npy` (learned part on test) | `d08b75a960c45f656274da112d49a878992b12784e95d69f971c0acf6c1aa0a9` |
| `data/b_test_pairs.json` | `b4fd47899a502b27b9e2df30ad066e1c8cd18bd17042defeddf488f981c2a7e6` |
| `data/b_test_parts_summary.json` | `3a650ed97495d78ee78e9fa97db1986bd24b94a9db4196bc3a55fcddcc75c94d` |
| `data/b_results.json` (runs 1–6, per-predicate AP) | `9e0d430198c5663476743e49f7065493982758718b0807173fbcae6e04d25be4` |
| `data/b_rprime_novel_test.npy` (r′ for run 7) | `64fc3a8add4f55ed3f1828c53be3872e46b6898d5928ec0d92a33d9c734e5187` |
| `data/b_split_one/` (61 one-predicate split files; SHA256 of their sorted hash list) | `b01e8aeaaf77138c45de09f2b85ab0c9355ff96cfcc7ba6fdb5a41df33b43ca6` |

Pre-A8 tests on training data:

| file | SHA256 |
|---|---|
| `data/b_dry_results.json` | `e9945b9c619afd37ef6d22b183f7e448956485969355aadaecca6d8b4f1cd2de` |
| `data/b_dry_rprime_novel.npy` | `be24734a9001d3199f1e068e227be3954508392a3f955df771d159421138966b` |
| `data/b_plumb_parts_summary.json` | `0cdb36c913f5a973cf600644113bf558a93f966ac17cdf87e3c8d8c2c92b67d4` |

The scripts' hashes are in A8's table. Logs (commands, configuration, input,
output and code hashes, environment) are in `logs/b_*.json`, with the raw console
output in `logs/b_step1_test.out` and `logs/b_run_test.out`. `data/` is
git-ignored.
