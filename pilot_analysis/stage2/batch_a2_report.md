# Stage 2, Batch A2 — coverage-corrected recipe score, fresh leave-out set: report

**Verdict: G2-LO2 PASS.** The corrected recipe-only mAP is **24.92**, against
random **6.25**: ratio **3.99** (needed ≥ 3), margin **+18.67** (needed ≥ 5),
at α = 0.25.

**Read this with caveat 1.** LO2 also passes without the correction (α = 0:
20.28, which is 3.24× random). On this set the correction adds +4.6 mAP, but the
pass is not caused by it.

Scope:
- CPU only, $0.
- Training videos only: 0 novel labels used, asserted in every script.
- No test annotation, test cache file or test evaluation was opened.
- The repo is unchanged. The new code is in `pilot_analysis/stage2/` and reuses
  Batch A's.
- A6's verdict (FAIL) stands.

## 1. A7, and the A6 artefacts

A7 was appended to `pilot_analysis/stage1b/preregistration.md` on 2026-10-05,
before anything in this batch was trained or scored.

| | SHA256 |
|---|---|
| preregistration.md with A7 (`preregistration.sha256`, line 1) | `2b34d0ef59a167a32f9313b7857355faed631ca41ca87d00f963218ab2d3dadc` |
| A7 text alone (6,880 bytes) | `18f78042e8f93e5998d99ab80d49cb0198411c1806edf629bf77db02623ae0c0` |
| the first 23,967 bytes (through A6), unchanged | `0e3ceb5c7101332f41d829d0396a53d3b5d617f140253577d3d5143f49f2eda1` |

**The A6 artefacts are unchanged.** A7 pins `step0_selection.json` (through A6)
and Batch A's 13 model, score and data files (through A7). Every step re-verifies
all 30 pinned artefacts, and all 30 still match after the last step. Every
output of this batch has a new name (`a2_*`, `*_lo2*`, `es_*`).

## 2. LO2 and its selection

The rule:
- **A6's rule, unchanged:**
  - a non-empty recipe;
  - no shared recipe;
  - every learned output the recipe consults keeps ≥ 50 FIT instances after
    removal;
  - at least 4 with a learned part.
- **A7's additions:**
  - A6's 10 are excluded;
  - at most 3 per family;
  - the 10 with the most HELD instances, taken greedily.
- **Family:** the verb of a verb + spatial predicate. Bare spatial terms and the
  comparatives form one family, "no verb"; any other predicate is its own
  family.

| predicate | family | HELD | FIT | learned part | recipe |
|---|---|---:|---:|---|---|
| `stand_next_to` | stand | 115 | 353 | yes | none, stand, adjacent, stationary |
| `right` | no verb | 105 | 235 | no | right, ¬locomotion, ¬posture |
| `stop_behind` | stop | 96 | 365 | yes | behind, none, vehicle, stationary, ¬posture |
| `left` | no verb | 91 | 287 | no | left, ¬locomotion, ¬posture |
| `move_right` | move | 87 | 314 | no | right, vehicle, moving, ¬locomotion, ¬posture |
| `next_to` | no verb | 82 | 87 | yes | adjacent, ¬locomotion, ¬posture |
| `stop_left` | stop | 76 | 354 | yes | left, none, vehicle, stationary, ¬posture |
| `fly_with` | fly | 72 | 602 | yes | fly, co_move, object moving, moving, ¬posture |
| `move_front` | move | 70 | 302 | no | front, vehicle, moving, ¬locomotion, ¬posture |
| `follow` | follow | 67 | 225 | no | moving |

(¬ = absence factor.)

- 5 of the 10 need a learned part, and the recipes are distinct.
- `ride` (70 HELD instances) was skipped: it is the only carrier of
  contact_action=ride.
- No predicate was skipped by the family cap. A6 had already taken four of the
  five base `stand_*`.
- The weakest consulted output after removal is locomotion=fly, with 177 FIT
  instances.
- The LO2 learned part had 3,898 labels removed and 7,552 training slots left.
  Its seeds early-stopped at epochs 3, 3 and 2.

## 3. α grid (63 FIT early-stop videos)

- Competing: the 61 base predicates outside LO2.
- Learned part: LO2.
- c(p) over all 10,824 FIT slots.
- Data: 920 slots, 1,567 base GT relations.

| α | mAP | R@50 | R@100 |
|---:|---:|---:|---:|
| 0 | 13.50 | 17.79 | 24.85 |
| **0.25** | **22.64** | 26.62 | 33.60 |
| 0.5 | 22.16 | 25.81 | 33.82 |
| 0.75 | 21.11 | 25.37 | 32.65 |
| 1.0 | 19.88 | 23.46 | 30.51 |

**Chosen α = 0.25.** The gain comes almost entirely from moving off α = 0;
from 0.25 to 1.0 the curve is flat to falling.

## 4. G2 on LO2: **PASS**

Setup: HELD, PredCls on GT tracks, only LO2 competing (861 GT relations).
Scores go through the repo's post-processing and
`eval_relation_detection_openvoc`.

| scores (only LO2 competes) | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| **r′, α = 0.25, LO2 learned part** | **24.92** | 27.29 | 34.73 |
| random U(0,1), mean of 20 seeds (sd 0.80) | 6.25 | 13.61 | 20.60 |

| test | needed | got | |
|---|---|---|---|
| ratio | ≥ 3 × random | 3.99 × | ✓ |
| margin | ≥ random + 5 | +18.67 | ✓ |

**Verdict: PASS.**

## 5. Secondary results (not gates)

| | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| LO2, α = 0 (no correction) | 20.28 | 19.16 | 29.27 |
| A6 set, α = 0, A6 leave-out part *(seen during diagnosis, not a test)* | 28.72 | 29.46 | 35.71 |
| A6 set, α = 0.25, A6 leave-out part *(seen during diagnosis, not a test)* | 33.64 | 31.20 | 38.81 |

The A6 set at α = 0 reproduces Batch A's 28.72 exactly. That is a consistency
check on the new code path.

**Each LO2 predicate's top-1 share on its own HELD slots.** This is the share of
the predicate's own labelled HELD slots where its score is ≥ the best of the
other nine.

| predicate | own slots | c(p) | α = 0 | α = 0.25 |
|---|---:|---:|---:|---:|
| stand_next_to | 113 | 0.121 | 61.1% | 69.0% |
| right | 88 | 0.329 | 39.8% | 56.8% |
| stop_behind | 102 | 0.184 | 23.5% | 24.5% |
| left | 73 | 0.339 | 41.1% | 54.8% |
| move_right | 82 | 0.205 | 0.0% | 54.9% |
| next_to | 64 | 0.152 | 0.0% | 0.0% |
| stop_left | 77 | 0.190 | 45.5% | 27.3% |
| fly_with | 72 | 0.147 | 0.0% | 94.4% |
| move_front | 72 | 0.208 | 0.0% | 27.8% |
| follow | 57 | 0.519 | 82.5% | 70.2% |

## 6. Step 3 — mix on HELD (descriptive only)

Setup: the full learned part, c(p) recomputed with it, α = 0.25, all 71 base
predicates competing.

| scores | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| EOV-MMP alone | 50.28 | 35.46 | 42.73 |
| r′ alone | 23.86 | 22.61 | 28.84 |
| EOV^0.75 · r′^0.25 | 49.63 | 35.35 | 42.57 |
| **EOV^0.5 · r′^0.5 (the fixed mix)** | 46.52 | 33.91 | 41.51 |
| EOV^0.25 · r′^0.75 | 39.63 | 29.90 | 37.73 |

On its own training videos EOV-MMP is the strongest, and every mix weight lowers
it. This was expected and selects nothing.

## 7. Surprises and caveats

1. **LO2 is easier than the A6 set, so the pass isn't caused by the correction.**
   - Random scores 6.25 on LO2 against 10.25 on the A6 set, so the ratio test is
     easier here.
   - Uncorrected, LO2 would already pass: 20.28 is 3.24× random, +14.0.
   - The correction's contribution is +4.6 mAP on LO2 and +4.9 on the A6 set.
     On the A6 set it lifts the score from 2.80× to 3.28× random, but that set
     was seen during diagnosis.
   - The defensible reading: the correction helps, consistently, by about 5 mAP
     on both sets. G2-LO2 shows that the recipe beats random by a wide margin on
     a fresh set. It does not show that the correction was needed for that.
2. **Under the geometric mean, a subset recipe beats its superset.** At α = 0,
   `move_right`, `move_front` and `fly_with` never top their own slots (0%).
   Their recipes contain those of `right`, `front` and `follow` respectively.
   Whenever the extra factors score below the mean, the shorter recipe wins.
   The correction fixes most of this: fly_with 0 → 94%, move_right 0 → 55%. It
   costs `stop_left` (45 → 27%) and `follow` (83 → 70%).
3. **`next_to` never tops its own slots, at either α.** The correction doesn't
   reach it. I haven't diagnosed why; it is recorded, not investigated.
4. **`touch` has coverage 0.89, the highest of any predicate.** Its recipe is
   just contact_action=touch, whose output is trained only on ride and touch
   slots and is unconstrained, and high, everywhere else. The correction damps
   it. The same applies to any output trained under a narrow mask.
5. **HELD numbers stay optimistic, as in Batch A.** The features come from
   EOV-MMP, trained on these videos' labels, LO2's included. "Zero-shot"
   applies only to the learned part's labels.
6. **The early-stop videos were used twice.** The α choice used their labels,
   which already chose the LO2 part's early-stopping epoch. A7 allows this; it
   is noted because the α curve is reported as if independent.
7. **G1 was not re-run for the LO2 learned part, as A7 specifies.**
8. **One α (0.25) is used for all predicates, novel ones included.** Nothing is
   fitted per predicate. c(p) needs no labels, so Batch B can compute it for the
   novel predicates the same way.

## 8. Hashes of new files

Model and score files:

| file | SHA256 |
|---|---|
| `data/model_lo2.pt` (LO2 learned part, 3 seeds + normalisation) | `3c2a6c244328766f0bce866e97c6c69ef94b342f5971705dca95673a2b85f19e` |
| `data/learned_lo2.npy` | `666092b2965609804d3e5cacdcb3456cf34f3cfeb0701547a645c15e09b00b73` |
| `data/train_lo2.json` | `177720e69728c796ad6cf3bc66af760b2de952f5a3a28f5ecfef1397f8d16338` |
| `data/a2_coverage_lo2.json` | `b6fcd97ff085a23a9162c873c5f9bab63c0f014f28ace87aa6843464f46f3356` |
| `data/a2_coverage_a6leaveout.json` | `1fe943f24b4281b2ba892d13613b4b9fea0d77b1fe7ac0ca2d692e965b7deeb6` |
| `data/a2_coverage_full.json` | `713e057c40311718dda71b54bb234b44953db07ee548fdc18f3148c2aefa9ef3` |
| `data/a2_alpha.json` | `2fd4b06ed789729ea7d8cd4a259dd659ddadf7bd468c632a8560bad79bf033e2` |
| `data/a2_g2.json` | `131f6e27448e407a0f44a765b1cd232a38770e1b2bbbe8c6c6bd8dff05ba5980` |
| `data/a2_recipe_lo2_held.npy` (r, HELD) | `37657c3a7a301c82a16be95290d497434a90730e1482e18625af9e6ea06590eb` |
| `data/a2_rprime_lo2_held.npy` (r′, HELD) | `51182ee1e8a61c8bd3502673967e7dc464b2301cd116bfff56cc8e192b437059` |
| `data/a2_rprime_full_held.npy` (r′, full part, HELD) | `23a3740185e2a4de6b61e43bbb4406a5c75ee064554085faf55be268c03c6d08` |
| `data/a2_step3.json` | `5656c521a5687fcaf3e6ffc710318b40f588c57cbac6f06d3bad5b92ee2b1fe5` |

Data and split files:

| file | SHA256 |
|---|---|
| `data/es_pairs.json` | `56f8563442146eb314dfe4ae7c052f1dd01f3ebd235a902dd6b2cd2600f0d242` |
| `data/es_gt.json` | `73bb8815c7253bef19b4da010d28bbd5fa39cf91c0f6627b7fb48a003d9bd1dd` |
| `data/split_a2_alpha.json` | `55fd4ac36fee38f30aedcc60d6dc50b65fee52b2dd6cbe8a40865bc731a86eae` |
| `data/split_lo2.json` | `b435ef3ee905a9ce7e29c5e64230b43d75b5f965df3c782e476b438d0c713ea9` |
| `data/split_a6set.json` | `1897dcbe6d867d62e6f0620392cc653232d27944b944caac5b3bfbe7b561748a` |
| `a2_step0_selection.json` (pinned by A7) | `34f7cdced8f7d22eb18b406d511bac516d9e558f0e3a9285fbdcc2ada768864e` |

Logs (command, configuration, seeds, input, output and code hashes, environment)
are in `logs/a2_*.json` and `logs/step2_train_lo2.json`. `*_small` files and logs
are the 5-video plumbing runs. `data/` is git-ignored.

Reproduce:
```
python pilot_analysis/stage2/a2_es_data.py
python pilot_analysis/stage2/step2_train.py --version lo2     # ~1 min
python pilot_analysis/stage2/a2_step1_alpha.py
python pilot_analysis/stage2/a2_step2_g2.py                   # ~40 s
python pilot_analysis/stage2/a2_step3_mix.py
```
