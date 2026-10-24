# Stage 2, Batch B2 — interim report (Part 1, training side)

Part 1: PASS

**Waiting for `GO B2-TEST`.** No test-side file has been opened in this batch:
no annotations, none of Batch B's test outputs, no test evaluation. The test run
is locked in the script until it gets `--go B2-TEST` and this report shows a pass.

## A9

A9 was appended to `pilot_analysis/stage1b/preregistration.md` on 2026-10-05,
before anything in this batch was computed on data. It discloses that B2 is a
second attempt, designed after Batch B's test result, and that Batch B's result
stands.

| | SHA256 |
|---|---|
| preregistration.md with A9 (`preregistration.sha256`, line 1) | `86a0f18a16f08edc91536d43dc1ec417dbc8d5787e040c204907d04b3fcb6538` |
| A9 text alone (5,186 bytes) | `80eb10534b7e0ff4e14ffd3d577a47542aab349b443f70c948639309f8e771c6` |
| the first 39,586 bytes (through A8), unchanged | `ca7ad1b7ae949f0a75eae13afc46f52e7c6f2027ae09a387580227e30875ba7d` |
| `b2_run.py` (pinned by A9) | `55ac092fd39783ae016a938cc61e1c9dd99fcb6aa17b1fde58cc4465cb9e0ef0` |

**All 50 pinned artefacts were verified before each step.** As in every batch
since A, that routine check *hashes* (never parses) two test-side inputs: A5's
`gt_tracks_test.json` and A8's shipped `VidVRD_ECC_test.json`. Batch B's test
outputs (`b_test_*`, `b_rprime_novel_test.npy`) are recorded in A9 outside the
checked table, and are verified only in the test run. 0 novel labels were used in
training (asserted).

The only thing run before A9 was a unit test of the reordering function on
synthetic arrays. It checked the multiset, the key order, both tie-breaks,
untouched no-recipe columns and an untouched no-ECC video.

## HELD table (all 71 base predicates compete, PredCls on GT tracks, full learned part)

| scores | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| EOV-MMP alone | 50.28 | 35.46 | 42.73 |
| **B2** (EOV values reassigned in √(EOV × r′) order) | **49.04** | 35.03 | 42.06 |
| control (shuffled recipes) | 49.04 | 35.03 | 42.06 |

- **The control is identical to B2 here, as A9 states in advance.** A6's
  permutation moves novel predicates only, and only base predicates compete on
  HELD. It carries no information in Part 1.
- B2 is 1.24 below EOV-MMP alone. That is expected: EOV-MMP has seen these
  videos, so any reordering of its scores can only lose ground here.

## Invariant

**Exact everywhere: 11,147 / 11,147 checks for B2, and 11,147 / 11,147 for the
control.** That is every (video, base predicate) pair over HELD's 157 videos,
plus the no-ECC check, which has nothing to do on HELD. B2 moved 115,016 score
values between units without changing a single per-video, per-predicate multiset.

## Pass rule

B2 mAP ≥ EOV-alone − 2.0 and the invariant everywhere: 49.04 ≥ 48.28 ✓,
invariant ✓.

**Part 1: PASS.**

## Also done (training side, after A9)

The test-mode code path was exercised on HELD (`b2_run.py --test --dry-run`, with
base standing in for novel). Every run completed, the invariant held in all 6
run×split combinations, and the per-predicate step worked. Its run-1 check
reproduced A2's HELD baseline of 50.28. This is plumbing only. The control is
again identical in the base split, and its "gate" line means nothing on HELD.

## New files so far

| file | SHA256 |
|---|---|
| `data/b2_part1_results.json` | `466ff6785dc99b24acbf2d27d1ca58f3e70b27c0ddb9cbb4ff8d02d251e01c22` |
| `data/b2_dry_results.json` | `6ef176864af6c95efa8e670f78903772053e202ebecf619fa16f0d4199ff44e0` |

Logs: `logs/b2_part1_run.json`, `logs/b2_dry_run.json`, and the console output
in `logs/b2_part1.out` and `logs/b2_dry.out`.
