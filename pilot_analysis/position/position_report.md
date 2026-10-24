# Position verifier — training-video gates and label-free test reach: report

**Verdict: safe, but with almost no reach. G-P3 FAILS. T2 has no position
component to test.**

| gate | result | |
|---|---|---|
| G-P1 (GT boxes, HELD) | network: **0 / 3,294** instances falsely deleted, on every axis | PASS |
| G-P2 (jittered boxes, IoU 0.70) | horizontal, depth, proximity 0%; **vertical 2 / 139 = 1.44%** | vertical FAILS; 3 axes survive |
| **G-P3 (SGDet incremental reach ≥ 3%)** | **0.46%** (648 of 141,320 SGDet top-20 novel predictions) | **FAIL** |

The untrained measured rule survives the same three axes, but its τ is 0 on all
of them, so it deletes nothing. The rough projection for the network is
0.36 × 0.46 / 7.6 ≈ **+0.02 mAP**.

Scope:
- CPU only, about 25 minutes.
- Training videos: base labels only (3,301 novel instances dropped at parse; 0
  used).
- Test videos: cached model inputs only (pair trajectories, per-slot logits). No
  test relation label was opened, and the evaluator was never run on test.

## 1. A18 (with A19), and the covered values

| amendment | what it does | preregistration.md SHA256 after it |
|---|---|---|
| A18 | the design, before any training or scoring; also freezes T2 (text SHA256 `4861d87cec84e57f13e61e9c19cea4d6392addeb9d2c34dd875e4eef57eab69a`) | `9be2155aa0a154c920263f9dc624631354689f1459de58cbc1c624da12e83bd4` |
| A19 | coding fix: step 4 crashed, before printing anything, on an SGDet test video with no pairs. Such videos (2 of 200) are now skipped and counted | **`e1f16c4f285f812143d53fd4af5accdfa0db6be9331da9fe7b48d2cd2787be0d`** (current) |

The first 68,090 bytes (through A17) are unchanged (`0aff1345…`). 61 pins were
verified before every step (latest pin per path).

**Covered values and the predicates each governs** (`phi_frozen.json`):

| value | base | novel |
|---|---|---|
| left | creep_left, jump_left, left, lie_left, move_left, run_left, sit_left, stand_left, stop_left, swim_left, walk_left | fly_left |
| right | creep_right, jump_right, lie_right, move_right, right, run_right, sit_right, stand_right, stop_right, swim_right, walk_right | fly_right |
| above | fly_above, sit_above | above, creep_above, jump_above, lie_above, move_above, run_above, stand_above, stop_above, walk_above |
| beneath | creep_beneath, jump_beneath, move_beneath | beneath, lie_beneath, run_beneath, sit_beneath, stand_beneath, stop_beneath, swim_beneath, walk_beneath |
| front | creep_front, front, jump_front, lie_front, move_front, run_front, sit_front, stand_front, stop_front, swim_front, walk_front | fly_front |
| behind | behind, creep_behind, lie_behind, move_behind, run_behind, stand_behind, stop_behind, walk_behind | fly_behind, jump_behind, sit_behind, swim_behind |
| adjacent | fly_next_to, jump_next_to, lie_next_to, next_to, stand_next_to, walk_next_to | creep_next_to, move_next_to, run_next_to, sit_next_to, stop_next_to, swim_next_to |

Veto mechanics are T1's: the logit is set to −∞ (score exactly 0) **before**
post-processing, so each slot's top 20 refills. It applies in every slot of every
pair, in both splits, to base and novel predicates.

## 2. Symmetry checks

| check | result |
|---|---|
| features: swap(x(s,o)) = x(o,s), all 13,558 training slots | **13,558 / 13,558 exact** |
| untrained rule: left ↔ right, above ↔ beneath, front ↔ behind, adjacent fixed | **13,558 / 13,558 exact** |
| network, 1,000 random slots: z(s,o) + z(o,s) for horizontal, vertical, depth | max error **0.0** |
| network: z(s,o) − z(o,s) for proximity | max error **0.0** |

## 3. Training and τ

- **Data:** 800 training videos, 13,558 slots, 16,800 base instances with a
  covered value (all with a covering slot).
- **Supervised slots per axis:** horizontal 7,870 (3,933 left), vertical 792,
  depth 6,168, proximity 11,490 (1,313 adjacent).
- **Final model:** seeds stopped at epochs 7, 11 and 9 (early-stop loss
  0.432–0.440).
- **Cross-fitting:** 5 folds × 3 seeds, each trained for that seed's epoch count.

| value | τ (network) | FIT instances | FIT out-of-fold false-delete | HELD instances | HELD slot AUC | τ (untrained) |
|---|---:|---:|---:|---:|---:|---:|
| left | 0.0001 | 3,252 | 7 (0.22%) | 872 | 0.976 | 0.0 |
| right | 0.0001 | 3,258 | 7 (0.21%) | 872 | 0.976 | 0.0 |
| above | 0.5725 | 373 | 0 | 61 | 1.000 | 0.0 |
| beneath | 0.3825 | 301 | 0 | 78 | 1.000 | 0.7 |
| front | 0.0187 | 2,782 | 6 (0.22%) | 587 | 0.947 | 0.0 |
| behind | 0.0187 | 2,477 | 6 (0.24%) | 545 | 0.947 | 0.0 |
| adjacent | 0.0398 | 1,063 | 1 (0.09%) | 279 | 0.807 | 0.0 |

**Why τ is so small for left/right.** The network separates left from right well
(AUC 0.976), but **0.92% of FIT `left` instances get P(left) < 0.01 out of fold**.
The network confidently reads them as right: likely annotation swaps, like the
cattle video found in Batch A. A 0.25% safety margin therefore puts τ at 0.0001.

For above/beneath, k = ⌊0.0025 · n⌋ = 0, so τ is the smallest FIT score itself
(0.57 / 0.38). That is an aggressive veto, which G-P2 duly rejects.

## 4. G-P1 and G-P2 (HELD)

The box jitter uses scale 0.0492 for a mean IoU of 0.85, and 0.1116 for 0.70.

| axis | G-P1, GT boxes (network) | 95% CI | G-P2, IoU 0.85 | G-P2, IoU 0.70 | verdict | untrained: G-P1 / G-P2 (0.70) |
|---|---|---|---|---|---|---|
| horizontal | 0 / 1,744 | 0–0.21% | 0 | 0 | **PASS** | 0 / 0 → PASS (τ = 0) |
| vertical | 0 / 139 | 0–2.6% | 0 | **2 / 139 = 1.44%** | **FAIL** | 1 (0.72%) / 2 (1.44%) → FAIL |
| depth | 0 / 1,132 | 0–0.33% | 0 | 0 | **PASS** | 0 / 0 → PASS (τ = 0) |
| proximity | 0 / 279 | 0–1.3% | 0 | 0 | **PASS** | 0 / 0 → PASS (τ = 0) |
| **pooled** | **0 / 3,294** | 0–0.11% | 0 | 2 / 3,294 | | 1 / 3,294 |

**Overall: PASS for horizontal, depth and proximity; vertical dropped.** The same
three survive for the untrained rule.

**Breakdowns (network, surviving axes):**
- No HELD covered instance has a subject category with ≤ 5 FIT labelled slots,
  so F1's failure mode doesn't arise here.
- 0 deletions in every verb family: stand 0/1,196, walk 0/606, no-verb 0/380,
  stop 0/277, move 0/253, sit 0/124, lie 0/123, run 0/119, jump 0/26, swim 0/23,
  fly 0/16, creep 0/12.

## 5. G-P3: reach on cached test inputs (label-free)

| | SGDet: network | SGDet: untrained | PredCls: network | PredCls: untrained |
|---|---:|---:|---:|---:|
| novel top-20 predictions | 141,320 | 141,320 | 170,160 | 170,160 |
| deleted by the position veto | 792 (0.56%) | 0 | 1,426 (0.84%) | 0 |
| **incremental over the type rules** | **648 (0.46%)** | 0 | 1,236 (0.73%) | 0 |
| (deleted by the type rules, for reference) | 11,012 (7.79%) | | 12,607 (7.41%) | |
| type-rules pipeline outputs (top 200/video) affected / fully removed | 134 / 64 of 24,356 | 0 | 182 / 104 of 22,634 | 0 |

The type-rule shares reproduce F2's 7.8% / 7.4% exactly, a consistency check on
the shared path. 2 SGDet videos have no pairs (A19).

**Top predicates by incremental deletion (SGDet):** move_next_to 222,
sit_next_to 165, run_next_to 82, stop_next_to 69, creep_next_to 65,
swim_next_to 44, jump_behind 1. PredCls is the same set: move_next_to 439,
swim_next_to 187, sit_next_to 178, and so on. **Almost all of the reach is
adjacency;** left/right and front/behind contribute nearly nothing at their τ.

**Gate G-P3: FAIL** (0.46% < 3%). Rough projection: +0.02 mAP.

## 6. Surprises and caveats

1. **Label noise caps a safety-first veto.** About 1% of FIT position labels
   contradict the boxes with high confidence. A false-delete margin of 0.25% then
   forces τ below them, so the veto fires only where the network is even more
   certain than about the noisy labels. The network is accurate (HELD AUC
   0.95–1.00 on spatial axes); the gate fails on reach, not quality.
2. **The untrained rule cannot veto at all.** Some true-v FIT instances have a
   frame fraction of exactly 0, which puts τ at 0. A frame-fraction rule has no
   room below its noisy minimum.
3. **The vertical axis is the fragile one under box noise.** Its τ is aggressive
   because few labels (373 / 301) give k = 0.
4. **What reach remains is adjacency on far-apart pairs.** That is plausible: a
   `*_next_to` prediction on two boxes far apart. But it is only 0.46% of SGDet
   top-20 predictions, and 134 of 24,356 final outputs.
5. **There was no test-label access and no test evaluation.** The projection is
   a rough guide only.
6. **A19 was a crash fix before any step-4 number.** Steps 1–3 were unaffected.

## 7. Hashes

| file | SHA256 |
|---|---|
| `pos_common.py` (pinned by A18) | `d39b2484ac5db90154940588dd950e05e30ebff398543fa948efa3083589111c` |
| `p_run.py` (pinned by A19) | `69cd0228cc450142109e28d460a2e42e37b844fd8c7860e130dfd600addb0987` |
| `data/verifier_final.pt` (3 seeds + normalisation) | `97f087ca3b8c57c15ed6fde11d7624cece5ad66926fec4646fe673544e73f607` |
| `data/verifier_scores.npz` (final and out-of-fold P per slot) | `e099cfd44b2515b35275757e07760963391da4016ec0ce43967702bcb7bef55b` |
| `data/pos_slots.npz` (features, targets, untrained scores) | `579a77a3125ef83423920f264ab09462f5167249396c6db9d3ebb07dd406aa5c` |
| `data/pos_meta.json` | `29b249ef57ae30be7ecea07f20c9b076c62f847afbdcd05236ec5b1c45fc21f2` |
| `data/pos_instances.json` | `a58bd03353d4e050ad249b0c12984cedecbfe495e8c65e6415c481dc23c6a974` |
| `data/step1.json` … `data/step4.json` | `29f827e1…`, `2fd4ccc1…`, `a6ca9e8c…`, `76c84c06…` |

The 15 cross-fitting fold models were used only for the out-of-fold scores in
`verifier_scores.npz` and were not saved separately. Logs: `logs/p_step{1..4}.json`
and the `*_small.json` plumbing runs; console output in `p_step*_full.out`.
`data/` is git-ignored.
