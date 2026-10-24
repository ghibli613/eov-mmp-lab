# Feasibility batch F2 — three checks for a combined final method: report

**Verdict: only Part 3 (type rules) passes. It is the one component for the final
test brief.**

| part | gate | result | |
|---|---|---|---|
| 1. Relation-aware object classes | head top-1 on HELD ≥ 70% **and** disagreement on test ≥ 10% | 68.1% ✗; 24.0% ✓ | **FAIL** (dropped) |
| 2. Temporal smoothing | top-20 persistence ≤ 75% on test (ii) or (iii) | 87.0% / 87.3% | **FAIL** (dropped) |
| 3. Type rules | false-delete ≤ 0.5% **and** reach ≥ 3% on test (ii) or (iii) | 0.07%; 7.4% / 7.8% (8.3% fused) | **PASS** |

Scope:
- CPU only, about 15 minutes in all.
- Training videos: base labels only (0 novel labels used, asserted).
- Test videos: model inputs and outputs only, from session 3's `cache_s3/test`
  and `cache_s3_sgdet/test`, every file checked against `SHA256SUMS_session3.txt`.
  No test annotation was read and no test evaluation was run.

## 1. A14 (with A15, A16) and the pinned artefacts

| amendment | what it does | preregistration.md SHA256 after it |
|---|---|---|
| A14 | the F2 design, before any data (text SHA256 `29b52eea2279ef4472e690901e0410f39e252b120c0fb5662fa7fa1f0dc1b179`) | `1baeb99db95580eb77ab9a7aff44855a632845edec2a67398a3e02c16838f86c` |
| A15 | coding fix: Part 1's small run crashed before any result, because cached `slot_inputs` loaded onto this machine's GPU (`torch.load` without `map_location`) | `4de6a7f7723ad568fa3aae503b00d9ed3b2d5ca03f4a7db942ed1a34148ca560` |
| A16 | coding fix: Part 3's small run crashed while printing, before any number was shown or saved (tuple keys in JSON) | **`a9517bd7ec0aef631179b3e8e090b1e5d7b7151738ae40fa39dc5d4eeba7ccc0`** (current) |

**57 pins verified** (latest pin per path) before every run. Neither fix changed a
rule, threshold or gate. All full runs used the final `f2_run.py`.

## 2. Part 1 — relation-aware object classes

**1-0. The head, reproduced.**
- modelC's subject/object class head (`rc:496-509`): the normalised
  featEmbedding embedding, dotted with `obj_text_encoder.classifier_weights` (CLIP
  ViT-L/14@336px, "An image of {name}."), divided by 0.01 (`rc:442`), then a
  softmax over all 35 categories.
- `once.pt`'s cached text embeddings match the model's within 1.8e-7.
- On a direct CPU forward of modelC over the first 5 fit-train videos (40
  pairs), the re-implemented head matches the model's own scores within
  **1.3e-6 (pass, ≤ 1e-4)**.
- Using the cached fp16 embeddings instead changes probabilities by at most
  3.3e-4, and the top-1 class never changes.

**Object-class protocol.**
- **10 of the 35 categories are novel:** antelope, bear, cattle, fox,
  giant_panda, horse, lizard, motorcycle, red_panda, watercraft.
- The head is **computed but unused**: the evaluated triplet takes the tracker's
  class and score (interface audit step 9).
- The tracker's class comes from the mean, over the track, of each detection's
  35 scores (`gen_labels.py:586-622`). Each detection mixes the detector's
  confidence on its own label with the object classifier's softmax: 0.6/0.4 for
  six novel categories, 0.3/0.7 for the rest, and turtle unfused
  (`end2end_model.py:215-265`). The class is the argmax of that mean, and the
  score its maximum.
- In the novel split, modelC's head scores only the novel classes. It is
  trained on base classes only (`convert_oid_on_base`, `rc:533-534`).

**1a. HELD, GT tracks** (optimistic: modelC was trained on these videos). 742
tracks in 157 videos: **top-1 68.1%, top-3 74.0%.**

| | categories | top-1 |
|---|---|---|
| base categories (25) | airplane, ball, bicycle, bus, car, domestic_cat, elephant, lion, monkey, person, sheep, train, turtle, whale … | 93–100%, rabbit 60% |
| **novel categories (10)** | antelope 0/54, bear 0/39, cattle 0/35, fox 0/14, giant_panda 0/20, horse 0/23, lizard 0/4, motorcycle 0/7, watercraft 0/23, red_panda 6/16 | **≈ 0%** |

The head's misses are almost entirely the novel object categories. It was never
trained to output them.

**1b. Test, detected tracks (label-free).** 658 tracks, 653 of them in at least
one pair.

| | |
|---|---|
| tracks where the tracker's top-1 ≠ the head's | **157 / 653 = 24.0%** |
| … of which the tracker says a **novel** class | **114 (73%)**, e.g. cattle → sheep 14, red_panda → tiger 11, fox → domestic_cat 11, horse → bicycle 6 |
| SGDet predictions (Task 6's path, top 200 per video) with a disagreeing track | 8,612 / 25,784 = 33.4% |
| tracker score, agreeing tracks | median 0.88 (p10 0.63, p90 0.96) |
| tracker score, disagreeing tracks | median 0.54 (p10 0.32, p90 0.78) |
| fused class ≠ tracker class (A14's approximate fusion) | 107 tracks, 83 of them novel-class tracks |

**Gate 1: FAIL.** Top-1 is 68.1% < 70%; disagreement is 24.0% ≥ 10%.

The disagreement condition passes for the wrong reason. Most disagreements are
the head's blindness to novel object classes. Fusion would mostly rewrite correct
novel-class tracks into base look-alikes, which is harmful in an open-vocabulary
evaluation. **Part 1 is dropped.**

## 3. Part 2 — temporal flicker

Novel columns only; each slot's top 20 novel predicates.

| inputs | videos | transitions | persistence | run median / mean | single-slot runs | Spearman mean / median |
|---|---:|---:|---:|---|---:|---|
| (i) HELD, GT tracks | 157 | 982 | 87.3% | 1 / 1.52 | 75.5% | 0.940 / 0.956 |
| (ii) test, GT tracks | 200 | 7,116 | **87.0%** | 2 / 3.67 | 37.8% | 0.940 / 0.953 |
| (iii) test, detected tracks | 200 | 5,486 | **87.3%** | 2 / 3.10 | 39.9% | 0.942 / 0.955 |

**Gate 2: FAIL** (87% > 75% on both test sets). **Part 2 is dropped.**

EOV-MMP's slot-to-slot predicate scores are stable: consecutive slots keep 87% of
their top 20, with rank correlation 0.94. HELD's high single-slot-run share
reflects how short HELD pairs are (982 transitions over 2,494 slots), not
flicker.

## 4. Part 3 — type rules

**3a. Rules on FIT only** (626 videos; ≥ 99.5% one type, ≥ 100 base instances):
- **animate:** stand, sit, lie, walk, run, creep, swim;
- **vehicle:** stop.

`lie` is new compared with S1, which used all 783 videos: on FIT it is 723 animate
to 3 vehicle. Not rules: fly (763 vehicle / 16 animate), jump (123 / 55), move
(1,588 vehicle / 9 neither).

**3b. False-delete rate on HELD: 3 / 4,365 = 0.07%** (≤ 0.5% ✓). All three are
`motorcycle / lie_right`.

**3c. Reach: EOV-MMP's top-20 novel predictions deleted (label-free)**

| inputs | top-20 novel predictions | deleted | share |
|---|---:|---:|---:|
| training videos (783) | 266,360 | 25,489 | 9.6% |
| (ii) test, GT categories | 170,160 | 12,607 | **7.4%** |
| (iii) test, tracker categories | 141,320 | 11,012 | **7.8%** |
| (iii) test, fused categories (approximate fusion) | 141,320 | 11,758 | 8.3% |

**Gate 3: PASS.** 0.07% ≤ 0.5%, and reach ≥ 3% on both test inputs.

## 5. Components that passed

**Only the type rules (Part 3).**
- They delete any prediction whose verb has a rule (stand, sit, lie, walk, run,
  creep, swim → animate; stop → vehicle) when the subject's type differs.
- In PredCls the subject category is the GT category; in SGDet it is the
  tracker's category. Part 1's fusion did not pass, so the tracker's category
  stays.

This, and only this, enters the final test brief.

## 6. Surprises and caveats

1. **The object-class head is blind to novel object classes,** by training
   (base classes only). It is ≈ 100% on base categories and ≈ 0% on the 10 novel
   ones. A head restricted to base categories, or a different fusion, would be a
   post-hoc redesign; it was not tested.
2. **Gate 1's disagreement is mostly that blindness.** 73% of the disagreeing
   tracks are tracker-novel. Their tracker scores are low (median 0.54), so
   these tracks are uncertain either way. The test labels needed to know who is
   right were, by design, not read.
3. **There is no flicker to smooth.** At 87% persistence and Spearman 0.94,
   neighbouring slots already agree. This fits S1's finding that EOV-MMP keeps
   continuing predicates in its top 20.
4. **Reach is not benefit.** The rules delete 7–8% of top-20 novel predictions on
   test inputs. Whether those are false positives can only be shown by the final
   test run. HELD's 0.07% false-delete rate says the rules rarely hit true *base*
   relations. Novel test relations could still break a rule, e.g. an animal
   annotated with a novel `stop_*`, but no training label can show that.
5. **The fusion in A14 was approximate.** The cache lacks the tracker's 35-score
   vector. Moot now that Part 1 is dropped.
6. **A15 and A16 were crash fixes before any result.** No re-run changed a
   number.

## 7. Hashes of new files

| file | SHA256 |
|---|---|
| `f2_run.py` (pinned, A16) | `d2d04aacdbe80fac04d6096e89b5398c17cd3d5015c5bd07dc0107f3ccc2fac0` |
| `data/f2_part1.json` | `12dc80d1d9e7d48ca86b03105014449d9f20ed6d78ba22acbed9c80d60ac8f22` |
| `data/f2_fused_classes.json` (per detected track: tracker, head, fused) | `08fda4e5443f679593f00a28a322a4185247c23d7c7969533684695f50d6db55` |
| `data/f2_part2.json` | `fc38672b3265f66450bfeacfb4a8177f5d041411d5e00f0e5906b6b63ec8e3a4` |
| `data/f2_part3.json` | `68d454cc7706d5425eaa574181acc35cc14faf49d2d508d1941422af87b6fd40` |

Logs: `logs/f2_part{1,2,3}.json` and the `*_small.json` plumbing runs, with the
console output in `f2_part*_full.out` and `f2_part*_small.out`. `data/` and
`data_small/` are git-ignored.
