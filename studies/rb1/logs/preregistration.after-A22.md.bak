# Pre-registration — measured axes and paraphrases for stage 2

**Date:** 2026-09-28. **Status:** frozen. Any later change needs a dated amendment appended below, saying what changed and why. This file's SHA256 is in `preregistration.sha256`.

## Frozen artefacts

| artefact | path | SHA256 |
|---|---|---|
| schema | `pilot_analysis/stage0e/phi_frozen.json` | `2b45efe95f2875ec96f753df5fdfcda450628312e46e2f0fea2db77a0bb0cc2f` |
| session-1 measured axes (definitions) | `pilot_analysis/stage1/measured_axes.py` | `cd852c21ab701815e46536bc0a0ab82fd0cf846679e2267402d7e8ca1773f5a0` |
| slot grid, windows, D1/D2 (definitions) | `pilot_analysis/stage1b/slot_axes.py` | `76aefdea9088b19142262449e2ae51fe2c66b55b06c51bb3ea5926dcd96044f6` |
| per-window fitted thresholds | `pilot_analysis/stage1b/thresholds_slot.json` | `431afc198e8772467176cc9a312c5515c97727ef06f3be93dd241a05af437b45` |
| FIT/HELD split | `pilot_analysis/stage1/train_split.json` | `095dd89ced7260945e76f020a68dcee5793bdd5bca3a6f375f7537dc7c267f1a` |
| paraphrases | `pilot_analysis/stage1b/paraphrases_frozen.json` | `aa2cd527762d5b66f7aabe2f437a6fb1d08fe22b78de0d7d18e1524685f2dda6` |

## Scoring grid

One score vector per (ordered pair, 30-frame slot). A pair is two distinct tracks with ≥ 10 overlapping frames; the overlap is cut into 30-frame slots; a final partial slot is kept if ≥ 10 frames, else trimmed. Stage 2 builds these from **whole** tracks, not from `train_object_trajectories_gt.json`, which holds fragments. Windows: W0 = the slot; W1/W2 = ± 1/2 slots; W3 = the whole overlap; all clipped to the overlap.

## Frozen measured rules

| axis | rule (implementation) | window | threshold |
|---|---|---|---|
| horizontal | sign of mean(subject − object centre x); < 0 = left (`M.horizontal`) | W0 | none |
| vertical | sign of mean centre-y difference; subject higher = above (`M.vertical`) | W0 | none |
| depth | sign of mean bottom-edge y difference; lower = front (`M.depth`) | W1 | none |
| proximity: adjacent | mean gap ÷ mean box diagonal < threshold, and not contained (`M.proximity`) | W3 | gap 0.19752 |
| proximity: contained | in a majority of frames, subject-in-object fraction > object-in-subject fraction AND subject top edge not above object top edge (`M.contained`) | W3 | none |
| subject_state | `M.subject_motion` (net compensated displacement in own box diagonals per frame) > threshold = moving | W2 | 0.004471 |
| object_state | `M.object_motion` > threshold = moving | W0 | 0.004467 |
| relative_motion | **D1**, distance: `S.d1_decide(S.d1_stats(pair), dead_zone, motion)` — pass, then approach/recede on the change in normalised centre distance beyond the dead zone, then co_move (both moving, net directions agree), else none | W0 | dead zone 0.37481; motion 0.004467 |
| comparative: larger | sign of mean log area ratio (`M.larger_score`) | W3 | none |
| comparative: taller | sign of mean log height ratio (`M.taller_score`) | W3 | none |
| comparative: faster | sign of log speed ratio, specified speed (`M.faster_score`) | W1 | none |

Camera compensation: `VidVRD_ECC_train.json`; `ecc[str(t+1)]` maps frame t−1 into frame t (0-based). Symmetry under time reversal and subject/object swap is exact at every window (slot_check.md).

## Scoring rule (carried forward)

Unset locomotion and posture assert absence **for compositional predicates only** (a bare spatial term, or verb + spatial term). Every other unset axis is marginalised. MEASURED axes have no trained parameters. LEARNED: locomotion, posture, contact, contact_action. subject_kind is supplied by the object classifier.

## relative_motion — the decision, and what stays untested

Chosen: **D1** by the rule fixed in the session-2 brief: D1 is higher on both pass and chase/follow; co_move 68.5% ≥ 60%. The HELD evidence is in slot_check.md. The pass criterion was decided by one instance (D1 1/10 (10.0%) vs D2 0/10 (0.0%)). **Approach and recede remain unvalidated on base data (12 and 0 base instances), and pass is undetected by either definition. The novel test AP of the `*_away`, `*_toward` and `*_past` predicates is their test.**

## Paraphrases

`paraphrases_frozen.json`, SHA256 `aa2cd527762d5b66f7aabe2f437a6fb1d08fe22b78de0d7d18e1524685f2dda6`. It was written by a fresh subagent that saw only the 132 predicate strings and the three rules. It is saved verbatim and never edited. Known departures from the annotation meaning are recorded here, not fixed: 12 `*_front` predicates include "ahead of", which is heading-relative, whereas VidVRD front is camera-frame (stage 0b). All 8 `stop_*` predicates include "halts" / "comes to a stop", an event reading, whereas VidVRD stop means a stationary vehicle (stage 0c).

## Amendments

_None._

---

### Amendments appended 2026-09-28

These amendments were appended on 2026-09-28, **before any session-3 output
existed**. Every byte above the `---` line is the original pre-registration:
4,754 bytes, SHA256 `8bec195d2bb9449f0c3052bd9a9b5151ef293d095ce3692cd9848f11ccb16d18`.
To check it, run `head -c 4754 preregistration.md | sha256sum`. The original
sections and the paraphrase files are unchanged.

The evidence cited below comes from `slot_check.md`, i.e. from HELD base-labelled
training videos. No novel label and no test data informed any amendment.

#### A1 (2026-09-28): relative_motion toward / away / co_move, main rule changes from D1 to D2

**Change.** The main rule is now **D2 heading**, with the window and margin
fitted in session 2:
- window **W3** (the pair's whole overlap)
- margin **c = 0.77**
- motion threshold **0.0041003**, refit on FIT at W3

The implementation is `S.d2_decide(S.d2_stats(pair, S.build_heading(...)), c,
motion)` in `slot_axes.py`, unchanged. The schema's `approach` and `recede` are
scored by D2's `toward` and `away`; `co_move` is D2's `co_move`. Pass is
governed by A2, so if D2 returns `pass`, no measured relative_motion value is
emitted.

**Reason.** The frozen comparison was not like-for-like. The session-2 brief
scored D1 on approach **or** co_move, but D2 on toward only, and 50 of D1's 55
chase/follow hits were co_move. Like-for-like on HELD:

| measure | D2 | D1 |
|---|---:|---:|
| toward/approach on chase/follow | 25/77 | 5/77 |
| subject-driven motion on stationary subjects | 4/1144 | 60/1144 |
| co_move | 123/178 | 122/178 |

D2 also matches stage 0f: the verb belongs to the subject. The error was in the
brief.

**D1 remains frozen as a pre-registered ablation**, with the W0 parameters above.
**Main results use D2 whatever the test outcome.**

#### A2 (2026-09-28): relative_motion=pass re-tagged LEARNED

**Change.** `pass` becomes a LEARNED value, with 54 base instances (`run_past`
45, `fly_past` 9). The D1 and D2 statistics (`S.d1_stats`, `S.d2_stats`) become
input features to the head.

**Reason.** Neither measured definition detects it: HELD 1/10 (D1) and 0/10
(D2); FIT 0/47 and 4/47. Session 1's FAIL clause re-tags a value LEARNED when its
base support is ≥ 50.

#### A3 (2026-09-28): proximity=adjacent re-tagged LEARNED

**Change.** `adjacent` becomes a LEARNED value, with 1,342 base instances. The
measured gap (`M.gap_score`) becomes an input feature to the head.
`proximity=contained` stays MEASURED, since it has no base carriers.

**Reason.** The slot-level verdict is WEAK: HELD agreement 62.1%, AUC 0.724.

#### A4 (2026-09-28): the paraphrase test reports two spreads

**Change.** Report the paraphrase spread both over **all 132 predicates** and
**excluding the 20 predicates with known convention shifts**. Both are
reported, and neither replaces the other. The 20 are:
- the 12 `*_front` predicates, for "ahead of": `front`, `creep_front`,
  `fly_front`, `jump_front`, `lie_front`, `move_front`, `run_front`,
  `sit_front`, `stand_front`, `stop_front`, `swim_front`, `walk_front`;
- the 8 `stop_*` predicates, for "halts" / "comes to a stop": `stop_above`,
  `stop_behind`, `stop_beneath`, `stop_front`, `stop_left`, `stop_next_to`,
  `stop_right`, `stop_with`.

`stop_front` belongs to both groups, so the union is 19 distinct predicates.

#### A5 (2026-10-01): train GT tracks are built by option B

**Change.** Train tracks, for the feature cache and for stage 2, are built as
follows:
- Each annotated train tid is split wherever **more than 30 consecutive frames
  lack a box**, and every resulting run becomes its own track.
- Inside a run, the missing frames (gaps of ≤ 30) are linearly interpolated, then
  rounded and clipped at 0 like the repo's `round_and_positive`.
- **Every annotated box is kept unchanged** (189,345 of 189,345, verified). No box
  is invented across a gap of more than 30 frames, and no pre-roll frames are
  added.

**Test tracks are unchanged**: they still use the repo densifier
(`add_initial_frames` + `interpolate_and_adjust_frames`, including its 2-frame
pre-roll). **The ECC supplement for the one test video missing from the shipped
file is kept as computed.**

| artefact | path | SHA256 |
|---|---|---|
| option-B builder | `pilot_analysis/stage1c/optionB.py` (`runs_option_b`) | `ee76aaf1be7e9b17fa17d6f25794f08747b4fe3601ae7cf22105d95f176460e9` |
| track builder | `pilot_analysis/stage1c/gt_tracks.py` | `a2b631c95c5bcb943962ab0ad5492f696d6dd13dd34221abf676cdd93a1ef006` |
| train tracks (option B) | `pilot_analysis/stage1c/gt_tracks_train.json` | `87b40fff514178f77a57d9e142bbbb9363eef34f6b4c22d8eaa88d4c7612636d` |
| test tracks (repo densifier) | `pilot_analysis/stage1c/gt_tracks_test.json` | `45b8b93e61368ea4409f89ad80399ca7a095a0ccc6c511f571cfa4a11856de18` |
| ECC supplement | `pilot_analysis/stage1c/VidVRD_ECC_test_supplement.json` | `22e4c79f3ea881d6c633b52f4c085f1508bbe6e610a91fb4ad51c90f72617057` |

**Reason.** The repo densifier drops any track below 65% frame coverage. That
is **33.5% of train tracks against 0.7% of test tracks**: training annotations
are gappy, and test annotations are not. Option C, interpolating through every
gap, would invent boxes across gaps of 100+ frames, which would corrupt the
motion measurements.

**No test label and no test result informed this change.** The 0.7% figure
comes from test boxes only.

**Effect on the frozen measured axes.** Session 2's grid used whole annotated
spans, from first to last annotated frame. It didn't split at gaps or
interpolate, and measurement skipped frames without both boxes. That differs
from option B, so HELD agreement was re-scored on the option-B grid with the
frozen windows and thresholds, and no refit. The scorer first reproduced
session 2's numbers exactly on the old grid. **No measured axis is worse by more
than 5 points.** The largest change is D2 co_move on `*_with`, from 69.1% to
64.5%, still above the 60% floor. Details: `pilot_analysis/stage1c/optionB_rescore.md`.

Train under option B: 4,044 tracks, 8,926 pairs, 13,558 slots, 12,365 labelled
slots, and 0 pairs over 40 slots.

#### A6 (2026-10-05): stage 2 Batch A — recipe head, gates G1 and G2, the Batch-B control

Written **before any stage-2 part score was computed or any network trained**. The
only inputs used to write it are the training annotations (base labels only; the
3,301 novel instances are dropped at parse, so 0 novel labels are used) and the
frozen files above. No test annotation, test cache file or test evaluation was
opened. The selections below come from `pilot_analysis/stage2/step0_select.py`,
whose output `step0_selection.json` is pinned at the end. Code lives in
`pilot_analysis/stage2/` and imports the repo; each run logs its code hashes.
Where the code and this text disagree, this text governs.

**Data.** Training videos only. The 17 videos without ECC matrices are excluded
from everything, as in stage 1: FIT 626, HELD 157. Early-stop videos: 63 of FIT
(10%), the first 63 of `numpy.random.default_rng(20261005).permutation` over the
sorted FIT ids; the learned part trains on the other 563 ("fit-train"). Grid and
features come from the session-3 cache on GT tracks (`cache_s3/train`, option-B
tracks per A5); each file is checked against `SHA256SUMS_session3.txt` before use.
A slot's labels are the base predicates of the same ordered GT pair whose extent
covers ≥ 15 frames of the slot (`slot_axes.slot_labels`).

**Terms.** *Measured part*: computed from the boxes with the frozen rules; no
training. *Supplied part*: subject_kind=vehicle, read from the subject category
(the vehicle list in `stage0e/scoring_rule.md`); no training. *Learned part*:
locomotion, posture, contact, contact_action, relative_motion=pass (A2),
proximity=adjacent (A3); a small network trained here.

**Measured part scores.** No new fitted number. Each score is the fraction of the
window's frames where the frozen rule's per-frame condition holds; a rule that
exists only at window level gives its 0/1 output. Windows and thresholds are the
frozen ones (original table and A1).

| part | score | window |
|---|---|---|
| left / right | fraction of frames with subject centre x < / > object centre x | W0 |
| above / beneath | fraction with subject centre y < / > object centre y | W0 |
| front / behind | fraction with subject bottom edge lower / higher than the object's | W1 |
| contained | fraction of frames meeting the per-frame condition of `M.contained` | W3 |
| larger / taller | fraction with subject area / height greater than the object's (`M._area`; heights clamped at 1, as in `M`) | W3 |
| subject_state moving / stationary | 0/1: `M.subject_motion` > 0.004471, else stationary | W2 |
| object_state moving | 0/1: `M.object_motion` > 0.004467 | W0 |
| approach / recede / co_move | 0/1: D2 (`S.d2_decide`, c 0.77, motion 0.0041003) returns toward / away / co_move. D2 `pass` sets none of them (A1). | W3 |
| faster | 0/1: sign of `M.faster_score` | W1 |

A part that cannot be evaluated in a slot (no motion data; a tie for faster) is
left out of that slot's recipe. The supplied part is 1 for a vehicle subject,
else 0. **Every part score is floored at ε = 0.01** (and capped at 1).

**Learned part.**

- *Outputs (14):* locomotion creep, fly, jump, none, run, swim, walk; posture
  lie, sit, stand; contact_action ride, touch; relative_motion=pass;
  proximity=adjacent.
- *Not outputs, dropped from every recipe now, before training:*
  contact_action bite, feed, hold and kick have 0 base instances (stage-0
  threshold: 50). **contact** has no negative class: no base label asserts its
  absence, and the brief masks unset parts rather than treating them as
  negative. It can be neither trained nor tested, so it cannot meet G1.
- *Targets, per labelled slot:*
  - locomotion and posture: values a label sets are positive. When no label
    sets the axis, a compositional label implies locomotion=none and no
    posture (the frozen absence rule). An explicit value beats an implied one.
    The other values of an observed axis are negative.
  - contact_action: a set value is positive, the other output negative.
  - pass: observed when a label sets relative_motion; positive iff that value
    is pass.
  - adjacent: positive when a label asserts adjacent; negative when no label
    sets proximity but some label sets horizontal, vertical or depth. That is
    stage 1's negative set (axis_check.md A3, slot_check.md), the evidence A3
    rests on.
  - Every other output of the slot is masked. Slots with no base label are not
    used.
- *Inputs:* the slot's `visual_pre_embeddings` (3,072), plus the pair's subject
  and object embeddings (768 each). **Plus the scalars A2 and A3 commit to:**
  - D1 statistics at W0: first-third and last-third mean distance, mid-third
    minimum, last − first, subject motion, object motion, direction cosine;
  - D2 statistics at W3: mean, first-third and last-third cosine, subject
    motion, object motion, direction cosine;
  - `M.gap_score` at W3.

  The brief lists only the embeddings; A2 and A3 make these inputs too. Each
  scalar is standardised as (x − median) / IQR over fit-train slots and
  clipped to ±5; a NaN becomes 0, with one missing flag per scalar. Then every
  input column is z-scored with the mean and std of all fit-train slots,
  labelled or not. That gives 4,636 inputs.
- *Network and training:*
  - Linear(4,636 → 512), ReLU, dropout 0.3, Linear(512 → 14), one sigmoid per
    output.
  - Loss: BCE over the observed entries only.
  - Adam, lr 1e-3, batch 256, at most 200 epochs.
  - Early stopping: patience 10 epochs on the early-stop videos' masked BCE;
    the best epoch is kept.
  - Training slots: labelled fit-train slots with at least one observed output.
  - Seeds 0, 1 and 2; the three seeds' outputs are averaged.
- *Two versions:* **full**, with all base labels (the version Batch B uses), and
  **leave-out**, identical except that the 10 G2 predicates' labels are removed
  from every slot's label set before targets are built. Slots left with no label
  are not used.

**G1 (learned-part bar), full version, on HELD.** For each output, the positives
and negatives are its targets on HELD slots, and the score is the three-seed
mean. An output with **≥ 30 HELD positive slots needs one-vs-rest AUC ≥ 0.75**.
Below the bar, it is dropped: treated as unset in every recipe. The factor is
removed, and the drop does not trigger the absence rule. Outputs with < 30 HELD
positive slots are kept and reported as untested. A video-clustered bootstrap CI
is reported, not used. Predicates that then share a recipe are listed.

**Recipe score.** The factors of predicate p are:
- the parts phi_frozen sets for p, minus dropped values;
- for compositional p (stage 0e's test: a bare spatial term, or verb + spatial
  term), an absence factor on each of locomotion and posture that phi leaves
  unset. Locomotion absence = p(locomotion=none); posture absence = 1 − the max
  of the kept posture outputs.

The recipe score is the **geometric mean** of the factor scores, each in
[0.01, 1]. A predicate with no recipe gets the slot's mean recipe score over the
predicates that have one (all 132). These are fight, play and watch, plus bite,
feed, hold and kick after the drops above. The same applies in a slot where none
of p's factors can be evaluated.

**Mix.** final = EOV^(1 − w) · recipe^w, where EOV is the repo's per-slot score,
sigmoid(logit) · sigmoid(interactiveness), from the cache. **w = 0.5 is the mix**,
i.e. √(EOV × recipe). w = 0.25 and 0.75 are reported as a sensitivity check
only. They are not selectable.

**Evaluation.** HELD GT is the base relation instances of the 157 HELD videos,
built as `get_relation_insts` builds them
(`third_party/vidvrd_ii_helper/dataset/dataset.py:184-218`). The pipeline is the
repo's:
1. Scores for every cached HELD (pair, slot), with the columns of predicates that
   don't compete set to 0.
2. `process_pred` (top 20 per slot), then `association`, then `format_` (top 200
   per video).
3. `eval_relation_detection_openvoc` with `gt_json` set to the HELD GT.

Step 4 uses `target_split_pred="base"`. G2 uses a copy of the predicate split
file in which the 10 predicates form their own split. The metric is never
reimplemented.

**Leave-out set (G2).** The fixed rule, applied greedily in descending order of
HELD instances (ties by name):
- the predicate has a non-empty recipe;
- no two chosen predicates share a recipe;
- every learned output any chosen predicate's recipe consults (absence factors
  included) keeps ≥ 50 FIT instances after all chosen predicates' labels are
  removed;
- a predicate without a learned part is skipped if the remaining places are
  needed to reach 4 with one.

| predicate | HELD instances | FIT instances | learned part |
|---|---:|---:|---|
| `stand_left` | 339 | 835 | yes |
| `stand_right` | 332 | 889 | yes |
| `stand_front` | 227 | 628 | yes |
| `larger` | 221 | 927 | no |
| `taller` | 206 | 881 | no |
| `stand_behind` | 179 | 720 | yes |
| `walk_left` | 173 | 732 | yes |
| `walk_right` | 147 | 766 | yes |
| `walk_behind` | 127 | 585 | yes |
| `walk_front` | 113 | 629 | yes |

8 of the 10 have a learned part. `stand_next_to` (115 HELD instances) was
skipped: it is the last base carrier of posture=stand, whose FIT support would
fall to 0. `watch` (130) and `play` (109) have no recipe. After the removal, the
weakest consulted outputs are posture=stand with 353 FIT instances and
locomotion=walk with 584.

**G2 (zero-shot gate).**
- Model: the leave-out version, with the recipes as they stand after G1.
- Data: HELD, PredCls on GT tracks; only the 10 compete.
- Comparison: the recipe-only mAP against random scores, U(0,1) per (slot,
  predicate), averaged over seeds 0–19.
- **PASS iff recipe-only mAP ≥ 3 × random and ≥ random + 5 points.**
- Also reported, outside the gate: EOV-MMP alone on the 10 (an upper reference
  only, since it was trained on them), and the full version's recipe-only mAP
  (a reference that is not zero-shot).
- If G2 fails, the batch stops at the report.

**Step 4 (descriptive only).** On HELD, with base predicates competing, report
mAP, R@50 and R@100 for EOV-MMP alone, the recipe alone (full version), and the
mix at w = 0.25, 0.5 and 0.75. Nothing is chosen from these numbers.

**Batch-B control, frozen now and not run now: shuffled recipes.**
- The novel predicates are grouped by their number of parts in
  `phi_frozen.json`.
- Within each group, in increasing part count, `numpy.random.default_rng(0)`
  draws permutations of the name-sorted group until one has no fixed point.
- Each predicate takes the donor's whole recipe, as it stands at Batch B.
- `fight`, the only 0-part predicate, is the one fixed point; it has no recipe.
- The 2-part group (`bite`, `feed`, `hold`, `kick`) has no recipe after the
  drops above, so its shuffle is empty.

Recipient ← donor:

| group (parts) | recipient ← donor |
|---|---|
| 1 | above ← beneath; away ← toward; beneath ← past; past ← above; toward ← away |
| 2 | bite ← hold; feed ← kick; hold ← feed; kick ← bite |
| 3 | creep_above ← swim_next_to; creep_away ← move_next_to; creep_next_to ← run_next_to; creep_past ← walk_past; creep_toward ← walk_beneath; fly_away ← stand_with; fly_behind ← walk_toward; fly_front ← jump_past; fly_left ← run_above; fly_right ← move_toward; jump_above ← walk_above; jump_away ← run_away; jump_behind ← creep_away; jump_past ← creep_above; lie_with ← jump_away; move_above ← creep_past; move_away ← swim_behind; move_next_to ← move_away; move_past ← fly_away; move_toward ← move_above; pull ← fly_front; run_above ← jump_behind; run_away ← fly_left; run_beneath ← swim_beneath; run_next_to ← fly_behind; run_toward ← creep_toward; stand_with ← move_past; stop_with ← lie_with; swim_behind ← run_beneath; swim_beneath ← pull; swim_next_to ← fly_right; walk_above ← creep_next_to; walk_away ← jump_above; walk_beneath ← stop_with; walk_past ← walk_away; walk_toward ← run_toward |
| 4 | drive ← stop_beneath; jump_with ← sit_next_to; lie_above ← stand_above; lie_beneath ← stand_beneath; sit_behind ← stop_next_to; sit_beneath ← lie_beneath; sit_next_to ← sit_behind; stand_above ← stop_above; stand_beneath ← sit_beneath; stop_above ← drive; stop_beneath ← lie_above; stop_next_to ← jump_with |
| 5 | lie_inside ← stand_inside; sit_inside ← lie_inside; stand_inside ← sit_inside |

**Pinned by A6.**

| artefact | path | SHA256 |
|---|---|---|
| scoring rule | `pilot_analysis/stage0e/scoring_rule.md` | `be75addd39d0519c79827ba7c26fb22b8a10897984d434efbc14273c3e9ee5c7` |
| subset pairs | `pilot_analysis/stage0e/subset_pairs_frozen.csv` | `821293aa3044a4a31df12798b7dbbdf41bb1c4abde1cc3fd4154467345586802` |
| step-0 selection | `pilot_analysis/stage2/step0_selection.json` | `9aa2310b5b2971b082221e0bb727b99c3d94afd19b17c8456c905e1ed170f04c` |
| session-3 cache checksums | `../session3_box_copy/SHA256SUMS_session3.txt` | `2cc2d0710f6a92d68ae27ecf74973a9dd122dc1594b9788cff4b0c560567954b` |

#### A7 (2026-10-05): stage 2 Batch A2 — coverage correction of the recipe score, a fresh leave-out set

Written after Batch A's report (A6's G2: FAIL, recipe-only 28.72 vs random 10.25,
ratio 2.80) and **before anything in this batch is trained or scored**. A6's
verdict stands. A6's artefacts are pinned below and are not overwritten; every
output of this batch has a new file name. The inputs used to write A7 are the
training annotations (base labels only; 0 novel labels used) and the frozen
files. No test file was opened. LO2 comes from
`pilot_analysis/stage2/a2_step0_select.py`, whose output
`a2_step0_selection.json` is pinned below. Everything A6 fixed applies unless
changed here.

**Coverage.** For predicate p, c(p) is the mean of its A6 recipe score r(p, s)
over **all FIT slots**, labelled or not: 626 videos, the 63 early-stop videos
included. It uses no label. It is computed for every predicate that has a recipe,
always with the learned part that the run in question uses:
- the LO2 learned part, for the α choice and for G2-LO2;
- A6's leave-out learned part, for the A6-set secondary;
- the full learned part, for step 3.

**Corrected score.** r′(p, s) = r(p, s) / c(p)^α. A predicate with no recipe (or
no evaluable factor in the slot) gets the slot's mean r′ over the predicates that
have one (all 132, as in A6).

**α.** Chosen from {0, 0.25, 0.5, 0.75, 1.0} on the **63 FIT early-stop videos**,
which have been used only for early stopping and never for weights:
- Scores: recipe-only r′ with the LO2 learned part.
- Competing: all base predicates except LO2 (61), with every other column 0.
- Pipeline: the repo's post-processing, then `eval_relation_detection_openvoc`
  with `gt_json` = the early-stop videos' base relations. These are built as the
  HELD GT is, and the competing predicates form their own split in a copy of the
  split file.
- The α with the highest mAP wins. A tie, meaning equal mAP to two decimals,
  goes to the smaller α.
- One α serves every predicate; nothing is fitted per predicate.

**LO2.** A6's rule unchanged:
- a non-empty recipe;
- no shared recipe;
- every consulted learned output keeps ≥ 50 FIT instances after the LO2 labels
  are removed;
- at least 4 with a learned part.

Plus:
- A6's 10 are excluded.
- **At most 3 per family.** The family is the verb of a verb + spatial predicate
  (`stand`, `walk`, `stop`, `move`, `fly`, …). Bare spatial terms and the
  comparatives `larger`, `taller` and `faster` form one family, "no verb". Any
  other predicate (`chase`, `follow`, `ride`, `watch`, …) is its own family.
- Greedy in descending HELD instances, ties by name; the first 10 are taken.

Result:

| predicate | family | HELD instances | FIT instances | learned part |
|---|---|---:|---:|---|
| `stand_next_to` | stand | 115 | 353 | yes |
| `right` | no verb | 105 | 235 | no |
| `stop_behind` | stop | 96 | 365 | yes |
| `left` | no verb | 91 | 287 | no |
| `move_right` | move | 87 | 314 | no |
| `next_to` | no verb | 82 | 87 | yes |
| `stop_left` | stop | 76 | 354 | yes |
| `fly_with` | fly | 72 | 602 | yes |
| `move_front` | move | 70 | 302 | no |
| `follow` | follow | 67 | 225 | no |

5 of the 10 have a learned part. `ride` (HELD 70) was skipped: it is the only carrier of contact_action=ride. FIT support of the consulted learned outputs after the removal: locomotion=fly 177, locomotion=none 7055, posture=lie 726, posture=sit 1288, posture=stand 3072, proximity=adjacent 597.

**LO2 learned part.** A6's network, inputs, normalisation, training settings,
early-stop videos and seeds (0, 1, 2; outputs averaged), retrained with the LO2
predicates' labels removed from every slot's label set. Other labels on those
slots are kept, and slots left with no label are unused. A6's G1 drops (none)
carry over; G1 is not re-run.

**G2 on LO2** has the same setup and **the same thresholds as A6**:
- HELD, PredCls on GT tracks; only LO2 competes.
- Random baseline: U(0,1) per (slot, predicate), mean over seeds 0–19.
- **Primary score: r′ at the chosen α, with the LO2 learned part.**
- **PASS iff r′ mAP ≥ 3 × random and ≥ random + 5 points.**
- If it fails, the batch stops at the report.

**Secondary results (not gates).**
- LO2 with α = 0.
- The A6 set, with A6's leave-out learned part, at α = 0 (which must reproduce
  A6's 28.72) and at the chosen α, labelled "seen during diagnosis, not a test".
- For each LO2 predicate: on its own labelled HELD slots, the share where its
  score is ≥ the maximum of the other nine, at α = 0 and at the chosen α.

**Step 3 (only if G2-LO2 passes; descriptive).** The full learned part, c(p)
recomputed with it, and the chosen α. On HELD with base predicates competing:
EOV-MMP alone, r′ alone, and EOV^(1 − w) · r′^w at w = 0.25, 0.5 and 0.75, giving
mAP, R@50 and R@100. Nothing is chosen from it.

**Pinned by A7.** The LO2 selection, and Batch A's outputs, which must stay
unchanged:

| artefact | path | SHA256 |
|---|---|---|
| Batch A output | `pilot_analysis/stage2/data/model_full.pt` | `3feb73909b3812384395bc2981ecf4dbb28440ff1d2ea394a7797e742ccd9e4b` |
| Batch A output | `pilot_analysis/stage2/data/model_leaveout.pt` | `293f4dfa8e87e4ccdded4172cd16c33477d3558b32fe56d2997adeb1aa15daf0` |
| Batch A output | `pilot_analysis/stage2/data/learned_full.npy` | `435e4cf499fbd69c5107cb149464fe16144b8d3ae56190c079bce29be864fc3c` |
| Batch A output | `pilot_analysis/stage2/data/learned_leaveout.npy` | `30e78d6d3d663331f2e1564d008539e1dc821dc9af0c8b640f98c9ee95757984` |
| Batch A output | `pilot_analysis/stage2/data/recipe_full_held.npy` | `f7ada222ffd11a38cf968dad17b55ca6e4ed44796b9c20de6da213300689f60e` |
| Batch A output | `pilot_analysis/stage2/data/recipe_leaveout_held.npy` | `634f3ce3a7c20127dcc6e6440362aa27cf6d9085cdaec9a83c8bdc87573a8430` |
| Batch A output | `pilot_analysis/stage2/data/slots.npz` | `8c9e9c48eb94888eb4f2ff9cbac4075087330e329445f59eb3f693a15e5332ed` |
| Batch A output | `pilot_analysis/stage2/data/emb.npy` | `9eb54ebb37317f5a85ab848f40b49df5189fc57f3c0f8928bf04c02fd6561469` |
| Batch A output | `pilot_analysis/stage2/data/slots_meta.json` | `01ba90c5933b464e69a540e704f500ed0d4161cfb43623e8cc75c401ce7bc570` |
| Batch A output | `pilot_analysis/stage2/data/held_gt.json` | `78d55c69f9aa2473f4584ae998592dcb67d95ce02a67ab0cd9a8bd8446184dee` |
| Batch A output | `pilot_analysis/stage2/data/held_pairs.json` | `8943fb186707a7908090af759d93815894ad654af6af34061ad52e05a5b783ec` |
| Batch A output | `pilot_analysis/stage2/data/g1.json` | `4507e3f3b7653008501c641857bc8ee64f0d0c0c392523fe616fcb22a8c104df` |
| Batch A output | `pilot_analysis/stage2/data/g2.json` | `eba6841c5283edd8a95d8445ad1959df8620303fbd484d60f275a6ac2fa1e9b3` |
| LO2 selection | `pilot_analysis/stage2/a2_step0_selection.json` | `34f7cdced8f7d22eb18b406d511bac516d9e558f0e3a9285fbdcc2ada768864e` |

#### A8 (2026-10-05): stage 2 Batch B — the frozen method and the one test run

Written after Batch A2's report (G2-LO2: PASS) and **before any test file is
opened in stage 2**. No test annotation, test cache file or test part score has
been read in stage 2. The only test-side inputs used to write A8 are the test
video file names and the keys of the shipped `VidVRD_ECC_test.json`. **Nothing
below may change after test numbers exist.** A demonstrable coding error found
later is fixed in a new amendment, and both the buggy and the fixed numbers are
reported. Everything A6 and A7 fixed applies unless changed here.

**Method (frozen).**
- *Learned part:* `data/model_full.pt` (Batch A; three-seed mean), unchanged, applied with its
  stored input normalisation (`learned_part.py`; it reproduces `learned_full.npy`
  exactly on all 13,318 training slots).
- *Recipes:* `data/recipes_after_g1.json` (A6's recipes; G1 dropped nothing).
- *Coverage:* c(p) for all 132 predicates with the full learned part over all
  10,824 FIT slots: `data/b_coverage_full.json`, identical to A2's
  `a2_coverage_full.json`.
- *Recipe score:* **r′ = r / c^0.25**.
- *Neutral score:* a predicate with no recipe (bite, feed, fight, hold, kick,
  play, watch), or with no evaluable factor in the slot, gets **the slot's mean
  r′ over the predicates competing in that split that have one**. This replaces
  A6's mean over all 132.
- *Final score:* **√(EOV × r′)** for every competing predicate, with EOV =
  sigmoid(logit) · sigmoid(interactiveness) from the test cache. Columns that
  don't compete are 0, as in modelC's novel branch.
- *EOV computation:* EOV is computed per pair by session 3 Task 4's own
  `eval_from_cache.slot_scores`. torch's CPU sigmoid rounds differently with
  tensor size, so this per-pair call is required for run 1 to reproduce Task 4's
  predictions bit for bit. Checked on the 157 HELD videos: the predictions are
  identical to the Task-4 path in both the all and novel modes.
- *Test parts:* measured and supplied parts from the GT test tracks
  (`stage1c/gt_tracks_test.json`, pinned in A5) and the **shipped**
  `VidVRD_ECC_test.json` (stage 1c's supplement is not used). The learned part
  comes from the test cache. Everything is computed by
  `b_step1_test_parts.py`, which reads no annotation file. The swap and
  time-reversal symmetry checks and the pair-for-pair match with the cache are
  required.
- *No ECC:* **1 test video** is missing from the shipped ECC file
  (`ILSVRC2015_train_00250013`). Its slots use **EOV-MMP's score alone, with no
  recipe, in every run**.

**The single run** (`b_run.py`; all 200 test videos, PredCls on GT tracks and
categories, the repo's post-processing and `eval_relation_detection_openvoc` with
the shipped test GT), in this order:
1. **Baseline check:** EOV-MMP alone, novel and all. It must reproduce **21.70
   novel and 37.20 all within ±0.05**, and must equal the session-3 Task-4
   scoring path (`stage1c/eval_from_cache.py`) exactly. Otherwise STOP
   (exit 3), and nothing else runs.
2. **Method, novel split. THE GATE: novel mAP ≥ 24.70.**
3. Method, all split: mAP, R@50 and R@100, reported and not gated.
4. **Control:** shuffled recipes. Each novel predicate takes its donor's r′
   column, using A6's permutation; everything else is identical. Novel and all.
5. Recipe alone (r′), novel. Descriptive.
6. Mix EOV^(1 − w) · r′^w at w = 0.25 and 0.75, novel and all. Descriptive, not
   selectable.
7. **Rewording robustness** (`b_run7_rewording.py`):
   - for the original strings (recomputed) and the 4 sets of
     `paraphrases_frozen.json` (SHA256 checked), EOV-MMP's test logits are
     recomputed exactly as in session 3 Task 5 (`paraphrase_test.set_strings` and
     `recompute`, unchanged);
   - novel mAP for the baseline (EOV′) and the method (√(EOV′ × r′), with r′
     unchanged and the no-ECC video on EOV′);
   - each method's spread = best − worst over the 5 conditions.

   **Run 7 is deferred.** Session 3 saved no reworded per-slot scores (`task5/`
   holds only `task5_paraphrase.json`). On this machine's CPU the recompute takes
   9.57 s per pair-window, plus 58.7 s per set of strings: 5 training videos, 12
   windows, logits within 0.0015 of the cache (`data/b_run7_timing_cpu.json`).
   For about 1,392 test pairs × 5 conditions that is **about 18.6 h**, over the
   2-hour limit. So runs 1–6 form the single run, and run 7 is done later with
   A8 unchanged, on a GPU or the CPU. Its table is then added to the report.

Then, inside the single run, **per-predicate novel AP**, baseline (run 1) against
method (run 2). Each AP comes from the official function on a copy of the split
file in which only that predicate competes. A predicate with no prediction in any
of its GT videos scores 0. Test instance counts are read from the GT file, for
the table only. The groups:
- (a) bare spatial: above, away, beneath, past, toward;
- (b) `*_next_to`: creep, move, run, sit, swim, stop;
- (c) no recipe: bite, feed, fight, hold, kick;
- (d) all other novel predicates.

Group means and medians are reported.

**Decision rules, fixed now.**
- **Gate passes** (run 2 ≥ 24.70): Batch C (SGDet against METOR) gets its own
  brief.
- **Gate fails:** the method stops and is reported as a negative result. No
  changes, no re-runs.
- **If the control (run 4, novel) is within 1.0 mAP of the method (run 2), the
  gain cannot be attributed to the recipes.** The verdict says so, whatever the
  gate says.
- Nothing (weight, α, threshold, recipe or part) is chosen or changed from test
  numbers.

**Tested before A8 on training data only:**
- `b_step1_test_parts.py --plumb-train 5` reproduces step 1's stored values for
  5 training videos: exactly for parts, scalars, logits and embeddings, within
  1e-6 for the learned outputs (float32 batch rounding, at most 1.8e-7).
- `b_run.py --dry-run` runs the whole pipeline on HELD, with the base split
  standing in for novel. Its run-1 check reproduces A2's HELD baseline of 50.28.

**Pinned by A8.** The code and the frozen inputs; any change stops every run:

| artefact | path | SHA256 |
|---|---|---|
| b_step0.json | `pilot_analysis/stage2/b_step0.json` | `6d659345546da778eff705339cc9fdc571e608759735dd258c655a64255dd499` |
| b_coverage_full.json | `pilot_analysis/stage2/data/b_coverage_full.json` | `4d3168190da71f80f794c428908c6d005567403877441187e677ad3a5c447f88` |
| recipes_after_g1.json | `pilot_analysis/stage2/data/recipes_after_g1.json` | `f3a4ab84561f60f07ecb38c4d21d697a12a30c49e939cc932298bde8d1c2cc3f` |
| b_run7_timing_cpu.json | `pilot_analysis/stage2/data/b_run7_timing_cpu.json` | `7dbaed4077ccd6ae8257d4d017e5585db4270c9a1667e1b25722497ceb187654` |
| VidVRD_ECC_test.json | `data/vidvrd/data/VidVRD_ECC_test.json` | `6e7842bee9887c85164b673becc1e61040bfa31bc33b921202e4b2d9649faa2b` |
| common.py | `pilot_analysis/stage2/common.py` | `8a48e7a2a75b22cc9835bb31329e452b35b0d52c8747ddd60e334bce3cdac0bd` |
| evaluation.py | `pilot_analysis/stage2/evaluation.py` | `e3c07de7073fde78127ef674483a4c822a96c6219bfeb90971fd230f1db34c77` |
| learned_part.py | `pilot_analysis/stage2/learned_part.py` | `f82532d7ff0701f502c41460b842b723beb7730dd5db40925d3982a452af2551` |
| a2_step1_alpha.py | `pilot_analysis/stage2/a2_step1_alpha.py` | `61d510474347dbd53584945caf5a7140ebf33d988d5eb908e39b2890c4f8318a` |
| step1_parts.py | `pilot_analysis/stage2/step1_parts.py` | `12a1b8fec3e185261f2bb56258782f6599af9aa995ec8a0028ea03a266de6bec` |
| b_step0.py | `pilot_analysis/stage2/b_step0.py` | `3a4e44f1b03dd316350ada3d1d24bf119a0bdce6d37cb72dc8f40092c03ac1e1` |
| b_step1_test_parts.py | `pilot_analysis/stage2/b_step1_test_parts.py` | `8a75878275dceaf97bda1fe05284dddcf3de981d38a92cf5a1145f5a6c2a49c9` |
| b_run.py | `pilot_analysis/stage2/b_run.py` | `b6f316be4dee324f4ab4dc3286647644577e6935742070897fcea6dc8493a172` |
| b_run7_rewording.py | `pilot_analysis/stage2/b_run7_rewording.py` | `211083954e9bb39275f31f0179b159936922c2daf834ca21e96c1969bb293ce8` |
| eval_from_cache.py | `pilot_analysis/stage1c/eval_from_cache.py` | `60ec4d97cf0ce52b6b0602ed6a8e7029c2c7e602a0dc01e9da3e0b4adebebff3` |
| paraphrase_test.py | `pilot_analysis/stage1c/paraphrase_test.py` | `763332b6400411483bdc13f1eaefb314f45be354029b16c50931788d8df80cf7` |
| cache_wrapper.py | `pilot_analysis/stage1c/cache_wrapper.py` | `9cbabf2e77dcdf99da531ecda229ccd6c23b1f5d33b5e1b734a830baf777b566` |
| post_process.py | `inference/post_process.py` | `f40af46cc95d7bd9b811f63f9dca323c536e4dc2634d0424cbf897e08c0979da` |
| video_relation_detection_openvoc.py | `inference/video_relation_detection_openvoc.py` | `0c2608a83978bce70c82a0b052b0d8457105717f387a07fcf15ad2342fa2b78e` |

#### A9 (2026-10-05): stage 2 Batch B2 — a second method attempt: recipes reorder within each predicate

**Disclosure.** This is a **second attempt, designed after seeing Batch B's test
result**. Batch B's method failed its gate (novel 15.37 vs EOV-MMP 21.70). Its
per-predicate results were mostly better, but its recipe scores broke the
cross-predicate calibration of the per-video ranking. **Batch B's result stays
reported unchanged.** If this attempt fails, method work ends. A9 was written
before anything in this batch was computed on data; the only check before it was
a unit test of the reordering function on synthetic arrays. Everything A6–A8
fixed applies unless changed here. Nothing is trained or fitted in this batch.

**Method (B2).** For each video, and each predicate p that has a recipe
(`recipes_after_g1.json`):
1. Take all of the video's (pair, slot) units. EOV(p, ·) gives a set of score
   values. The key m(p, ·) = √(EOV(p, ·) × r′(p, ·)) gives an order.
2. Sort EOV(p, ·)'s values from high to low and hand them out in key order: the
   unit with the highest m gets the highest EOV value. Ties go to the higher
   EOV, then the lower unit index.
3. Predicates with no recipe keep EOV-MMP's scores unchanged. So does every
   predicate in a video without ECC (`ILSVRC2015_train_00250013` on test).

The inputs are A8's, unchanged:
- EOV is session 3 Task 4's per-pair `slot_scores`.
- r′ = r / c^0.25, with the full learned part and `b_coverage_full.json`.
- A unit whose r′ cannot be evaluated gets the slot's mean r′ over the predicates
  competing in the split that have one.

c(p)'s factor does not change any within-predicate order, so α has no effect.

**Invariant, asserted in code for every video and every competing predicate:**
the multiset of final values equals EOV-MMP's exactly. Videos without ECC must be
unchanged. Within each video every predicate keeps the same scores, so
cross-predicate calibration is unchanged by construction; only the assignment to
pairs and slots changes. A violation stops the run.

**Control.** The same reordering keyed by √(EOV × r′_shuffled), with A6's
permutation: each novel predicate takes its donor's r′ column. **That
permutation moves novel predicates only, so in Part 1 (HELD, base predicates
competing) the control is identical to B2 by construction.** It is reported
there for completeness and carries no information.

**Part 1 (training side: HELD, the full learned part, all 71 base predicates
competing, PredCls on GT tracks).**
- Runs: EOV-MMP alone, B2 and the control, all through the repo's
  post-processing and `eval_relation_detection_openvoc`.
- **Pass rule (a sanity check, not a success test): B2 mAP ≥ EOV-alone mAP −
  2.0, and the invariant holds everywhere.**
- If Part 1 fails, the method ends.
- Then `batch_b2_interim.md` is written and the batch **stops until the user
  replies "GO B2-TEST"**. Until then, no test-side file is opened: no
  annotations, no A8 test files, no test evaluation. The script refuses the test
  run without `--go B2-TEST` and a passed interim report.

**Part 2 (test: 200 videos, PredCls on GT tracks), one scripted run, in order:**
1. **Baseline check:** EOV-MMP alone must reproduce 21.70 novel and 37.20 all
   within ±0.05, and must equal the Task-4 scoring path exactly. Otherwise STOP.
2. **B2, novel split. THE GATE: novel mAP ≥ 24.70.**
3. B2, all split: mAP, R@50, R@100.
4. **Control** (shuffled recipes, same reordering), novel and all.
5. Descriptive: the same reordering keyed by r′ alone, novel and all.

Plus per-predicate novel AP, baseline against B2, computed as in A8 (one
predicate per evaluator call, test instance counts, A8's four groups). The
invariant is asserted on every run.

Test-side inputs are Batch B's files. They are verified in the test run, after
GO, against `batch_b_report.md` section 11:
- b_test_slots.npz: 787f18f0376484978b934250e30fe37287b12c9ba77ad41dadc03bd47aa3b073
- b_test_learned_full.npy: d08b75a960c45f656274da112d49a878992b12784e95d69f971c0acf6c1aa0a9
- b_test_pairs.json: b4fd47899a502b27b9e2df30ad066e1c8cd18bd17042defeddf488f981c2a7e6
- b_rprime_novel_test.npy: 64fc3a8add4f55ed3f1828c53be3872e46b6898d5928ec0d92a33d9c734e5187.
  It must equal the r′ the script recomputes.

**Decision rules, fixed now.**
- **≥ 24.70:** the gate passes; Batch C (SGDet against METOR) gets its own brief.
- **Above 21.70 but below 24.70:** the method helps, but not enough to target
  METOR. The gain is reported, and method work ends.
- **≤ 21.70:** a negative result; method work ends.
- **If the control is within 1.0 novel mAP of B2, the effect cannot be
  attributed to the recipes.** The verdict says so, whatever the gate says.
- One run; no re-runs to change numbers. A crash before any number may be re-run
  and is logged. A demonstrable coding bug found after numbers exist is fixed in
  a new amendment, and both numbers are reported.

**Pinned by A9** (the reused code and inputs stay pinned by A8):

| artefact | path | SHA256 |
|---|---|---|
| b2_run.py | `pilot_analysis/stage2/b2_run.py` | `55ac092fd39783ae016a938cc61e1c9dd99fcb6aa17b1fde58cc4465cb9e0ef0` |

#### A10 (2026-10-06): span idea, Batch S1 — learning "continues vs changes" from training labels

Written before anything in this batch was computed on data. The only checks
before it were unit tests on synthetic sequences: target rule, box features,
model training and prediction, θ rule, clustered bootstrap. **No mAP is measured
in this batch.** The official training annotations are 30-frame chunks on a
15-frame grid (`span/s1_precheck.md`, `s1_precheck2.md`), so longer
predictions would be mis-scored against them. No test annotation, test cache
file or test evaluation is opened. Base labels only.

**Data.** Stage 2's slot grid and slot labels (`stage2/data/slots_meta.json`; a
slot's labels are the base predicates of the same ordered GT pair covering ≥ 15
frames of it): 783 training videos with ECC, A6's splits (fit-train 563,
early-stop 63, HELD 157), A7's LO2 set.

**Targets.** For each ordered pair and consecutive slots t, t+1:
- **continue (0):** both slots have ≥ 1 base label, and the sets are identical;
- **change (1):** both have ≥ 1 base label, and the sets differ;
- **unknown:** either slot has none. Masked, and never a target.

The **LO2 version** removes LO2's labels from every slot's set first; slots left
empty become unknown.

**Inputs per slot**, z-scored with fit-train statistics:
- the learned-part outputs (14). **Full version:** `learned_full.npy`. **LO2
  version: A7's LO2 learned part, `learned_lo2.npy`, trained without LO2's
  labels**, so that the hidden labels do not leak in through this input;
- the 16 measured part scores and the supplied vehicle score;
- PCA-256 of `visual_pre_embeddings`, centred SVD fitted on all FIT slots
  (fit-train and early-stop);
- box deltas (10): the change from the previous slot in subject and object
  centre x and y, log width and log height (each the mean over the slot's frames,
  normalised by the frame size) and in the subject–object gap (the mean
  rectangle gap / the mean box diagonal), plus a first-slot flag (deltas 0).

EOV-MMP's predicate scores are not an input.

**Model.**
- Input linear layer to d = 128, plus learned position embeddings (40).
- A 2-layer transformer encoder: 4 heads, feed-forward 256, dropout 0.2, with
  padding masks.
- For each adjacent pair, a linear classifier on [h_t, h_t+1, |h_t − h_t+1|]
  gives P(change).

**Training.**
- BCE on known transitions only, with pos_weight = continue / change over
  fit-train's known transitions.
- Adam, lr 1e-3, batches of 64 pairs, ≤ 100 epochs.
- Early stopping on the early-stop videos' weighted BCE: patience 10, best
  epoch kept.
- Train on fit-train. Seeds 0, 1 and 2; the three probabilities are averaged.
- Both versions use identical settings.

**θ.** On the early-stop videos' known transitions, each version on its own
targets, the θ in {0.01, …, 0.99} that maximises balanced accuracy; ties go to
the value closest to 0.5. It is used later, at test time, for cutting.

**Evaluation (HELD, known transitions, full-label targets).**
1. **Full version:** AUC, with a 95% CI from a percentile bootstrap that
   resamples whole videos (1,000 resamples, seed 0). Balanced accuracy, precision
   and recall of "change" at θ.
2. **LO2 version:** AUC on HELD known transitions where either slot carries an
   LO2 label in its full set, against the full-label targets. The full version's
   AUC on the same transitions is reported for reference.
3. **Descriptive:**
   - (a) the AUC of the summed absolute change in the 16 measured part scores,
     and of the cosine distance of consecutive PCA-256 vectors;
   - (b) on HELD continue transitions, for each continuing predicate in EOV-MMP's
     top 20 at slot t, the share still in the top 20 at t+1. Top 20 among the base
     predicates, EOV = sigmoid(logit) · sigmoid(interactiveness);
   - (c) the distribution of HELD segment lengths when every transition with
     P(change) ≥ θ (full version) is cut.

**Gate G-S1, both needed:**
1. Full version: HELD AUC ≥ 0.80, **and** the CI's lower bound ≥ 0.75.
2. LO2 version: AUC ≥ 0.75 on the LO2 transitions.

If either fails, the span idea ends.

**Side count (report only).**
- A verb is the first token of a verb + spatial predicate (stage 0e's test).
- Subject types: vehicle, animate or neither, from `scoring_rule.md`'s lists.
- A rule exists where one type is ≥ 99.5% of the verb's base instances, with
  ≥ 100 instances, over the 783 videos.
- Over every slot of those videos, EOV-MMP's top 20 novel predicates are counted
  as violating when the verb has a rule and the slot's subject type differs.

**Order.** `s1_run.py --small` (the first 5 videos of fit-train, early-stop and
HELD), then the full run.

**Pinned by A10:**

| artefact | path | SHA256 |
|---|---|---|
| span_common.py | `pilot_analysis/span/span_common.py` | `0d74fc2cad510b48260f06e749377a9c78ca0f9b8ae79800751ee5032a2da384` |
| s1_run.py | `pilot_analysis/span/s1_run.py` | `a711ab3d19db9f7a328ee93bd0445a25e5971f96b9c4d607fc463e1e232fc0a3` |
| learned_lo2.npy (A2) | `pilot_analysis/stage2/data/learned_lo2.npy` | `666092b2965609804d3e5cacdcb3456cf34f3cfeb0701547a645c15e09b00b73` |

#### A11 (2026-10-06): S1 coding fix — PCA on a fitting set smaller than 256 rows

The `--small` run of `s1_run.py` crashed **before computing any result**:
5 FIT videos give 104 FIT slots, so the SVD has 104 components, not 256. The fix
zero-pads the missing components:
`pca[rr, :len(comps)] = …` replaces `pca[rr] = …`. The full run fits on
10,824 FIT slots, so it is unaffected; the design is unchanged. `s1_run.py`'s
pin moves from `a711ab3d19db9f7a328ee93bd0445a25e5971f96b9c4d607fc463e1e232fc0a3` to the hash below. Every other line of A10 stands.

| artefact | path | SHA256 |
|---|---|---|
| s1_run.py (A11) | `pilot_analysis/span/s1_run.py` | `3cb2a76a7d2d9b059104fa0b7009b10d58af1a75726d9200f959ea01005dbb59` |

#### A12 (2026-10-06): the pin check follows superseding pins

A11 re-pins `s1_run.py`, but `common.verify_frozen` checked every pin row,
including superseded ones, so A10's old row would fail. The checker now keeps
**the last pin of each path in this file**: a later amendment's pin supersedes an
earlier one, as A11 states. This is a change to checking only; it changes no
method, no data and no result. `stage2/common.py` changes in that one function,
and its pin moves from A8's `8a48e7a2a75b22cc9835bb31329e452b35b0d52c8747ddd60e334bce3cdac0bd` to the hash below. Batch B/B2's
recorded runs used the earlier version, whose hash their logs record.

| artefact | path | SHA256 |
|---|---|---|
| common.py (A12) | `pilot_analysis/stage2/common.py` | `0efd5d770812f937fbe589efa78f4e44bcdcb8145b8dd1ed0ed7c1cb7023bc96` |

#### A13 (2026-10-06): feasibility batch F1 — compositional prior (Part A) and track fragmentation (Part B1)

Written before anything in this batch was computed on data. The only checks
before it were unit tests on synthetic data, plus the decomposition of the 132
predicate *names*.

- **Part A** uses training videos and base labels only.
- **Part B1 reads EOV-MMP's detected test tracks and per-slot logits**
  (session 3 Task 6's `cache_s3_sgdet`; model outputs only). It never reads a
  test annotation, runs no evaluation, and **only decides whether a
  track-stitching component is built** in a later batch.

**A1. Decomposition.** `f1_prior.decompose`, from stage 0's tokeniser:
- **verb + spatial predicate:** (verb, spatial), where the spatial term may be
  `with`, `inside` or `next_to`;
- **bare spatial term:** (`none`, spatial);
- **anything else is atomic:** (the predicate itself, `none`).

That gives 29 verb-type components (11 verbs, `none`, 17 atomic) and 13 spatial
components.

**Carriers and neutral components.** In each version, a component has a carrier
if some predicate in that version's training softmax uses it and has ≥ 1 FIT
labelled slot. A component without one is **neutral**: it takes the mean of the
carried components' vectors and biases (verb-type and spatial separately). Every
neutral component is flagged in the report.

**A2. Model.**
- logit(p | a, b) = ⟨W e(a), V_v⟩ + c_v + ⟨MLP([W e(a), W e(b)]), S_s⟩ + d_s, for
  p = (v, s).
- e is the frozen CLIP ViT-L/14@336px text embedding of the category name that
  modelC uses ("An image of {name}."; session 3 `once.pt` `object_text`,
  checked against `SHA256SUMS_session3.txt`).
- W: linear 768 → 64, no bias. MLP: 128 → 128 (ReLU) → 64. V, S ∈ R^64; c and d
  scalars.
- **An atomic predicate uses its own verb-type term plus the shared spatial
  `none` term.** This reads the brief's "verb term only, with s = none" as the
  general formula with s = none, so atomic predicates can depend on the object.
- **Training:** soft-target cross-entropy, uniform over a slot's training labels,
  with the softmax over the version's training predicates (base minus the
  hidden set). Fit-train labelled slots. Adam, lr 1e-3, weight decay 1e-4,
  batch 256, ≤ 300 epochs, early stopping on the early-stop slots' cross-entropy
  (patience 20).
- Seeds 0, 1 and 2; probabilities averaged.
- **Versions:** full; minus-A6; minus-LO2. A minus-version removes the hidden
  labels and takes the hidden predicates out of the training softmax.

**A3. Gate A1.** For each hidden set (A6's 10, with minus-A6; LO2, with
minus-LO2): HELD, PredCls on GT tracks, only the hidden 10 compete, through the
repo's post-processing and `eval_relation_detection_openvoc`.
- **Prior-only:** P(p | a, b) = softmax over the hidden 10, constant within a
  pair.
- **Random:** U(0,1) per (slot, predicate), mean of seeds 0–19.
- **Verb-frequency baseline:** the relative frequency of the predicate's
  verb-type component among FIT base instances with the hidden set removed.
- **PASS iff, on both sets, prior-only mAP ≥ 2 × random and ≥ verb-frequency +
  2.**
- Also reported: per-predicate slot-level AUC on HELD labelled slots.

**A4. Gate A2 (full prior).**
- P(v | a) = Σ over base predicates with verb-type v of P(p | a, b), with the
  softmax over all 71 base predicates. It is averaged over objects b, weighted by
  b's FIT frequency as the object of subject a in base-labelled slots (uniform
  over the 35 categories if a has none).
- **Veto (a, p)** if p's verb-type v has ≥ 100 FIT base instances and
  P(v | a) < 0.005.
- **False-veto rate:** the vetoed share of HELD base GT instances
  (`held_gt.json`, by subject category and predicate). **PASS iff ≤ 0.5%.**
- Label-free: the vetoed share of EOV-MMP's top-20 novel predictions per slot
  over the 783 videos (EOV = sigmoid(logit) · sigmoid(interactiveness)), with
  S1's 8.4% and its rules as references.

**A5. Descriptive.** P(v | a) for the 10 most frequent FIT subjects, and the full
prior's top-5 novel predicates for the 10 most frequent (subject, object) pairs.

**B1.** Over the 200 test videos' detected tracks:
- **Track length:** end − begin, with end exclusive.
- **Stitch candidate:** an ordered pair (A, B) with the same predicted category,
  B's first frame − A's last frame in [1, 30], and IoU(A's last box, B's first
  box) ≥ 0.3 or centre distance ≤ 0.5 × A's last box diagonal.
- Reported: the share of tracks in ≥ 1 candidate, and the share of SGDet
  relation predictions involving such a track. The predictions are Task 6's own
  `eval_from_cache.predict` ("all", top 200 per video), re-derived per video and
  matched to tracks by category and the box at their first frame.
- **Gate B1:** ≥ 10% of tracks involved → a stitching linker is worth building
  later; otherwise Idea B is dropped.

**Order.** `--small` (5 videos per subset; 5 test videos for B1), then full.

**Pinned by A13** (the reused stage-2, A6/A7 and S1 artefacts stay pinned
where they already are):

| artefact | path | SHA256 |
|---|---|---|
| f1_prior.py | `pilot_analysis/f1/f1_prior.py` | `37b5fdff9a6018505b04978f6cd2f2d5ca18212587eaa40858718f7debd4d4d8` |
| f1_tracks.py | `pilot_analysis/f1/f1_tracks.py` | `090c54d2abbc6c522fbf977248d8e8d8465e3a19ba5face990635c391623c817` |
| S1 results (subject-type rules) | `pilot_analysis/span/data/s1_results.json` | `182096795a8dfb144fbaa277b4e0e6c9dd8dbbd05c83d7358815ef3625ec6465` |

#### A14 (2026-10-06): feasibility batch F2 — relation-aware object classes, temporal flicker, type rules

Written before anything in this batch was computed on data. The only checks
before it were synthetic unit tests and code reading.

- **Training videos:** base labels only.
- **Test videos: model inputs and outputs only.**
  - Session 3's `cache_s3/test`: PredCls inputs (GT tracks and categories,
    built from test boxes and categories only), logits and embeddings.
  - `cache_s3_sgdet/test`: detected tracks, logits and embeddings.
  - Every file is checked against `SHA256SUMS_session3.txt`.
- **No test annotation file is read and no test evaluation is run. These checks
  only decide which components enter the final run.**

**Part 1 — relation-aware object classes.**

*The head.* modelC's subject/object class head (`rc:496-509`): the normalised
featEmbedding subject or object embedding, dotted with the object-class text
embeddings, divided by the temperature 0.01 (`rc:442`), then a softmax over the
35 categories ('all' split).
- The text embeddings are `obj_text_encoder.classifier_weights`: CLIP
  ViT-L/14@336px on "An image of {name}.", cached as `once.pt` `object_text`.
- Interface audit step 9: **the head is computed but unused**; the evaluated
  triplet takes the tracker's class and score.
- The tracker's class (`end2end_model.py:215-265`, `gen_labels.py:586-622`):
  - each detection's 35 scores mix the detector's confidence on its own label
    with the object classifier's softmax: 0.6/0.4 for antelope, bear, cattle,
    fox, giant_panda and red_panda; 0.3/0.7 for the others; turtle unfused;
  - a track's class is the argmax of the mean vector, and its score the max of
    that mean;
  - 10 of the 35 categories are novel, and in the novel split modelC's head
    scores only those.

*1-0 check.* A direct CPU forward of modelC on the first 5 fit-train videos'
cached slot inputs. The head re-implemented on the forward's own fp32 embeddings
must equal the model's subject and object scores within **1e-4**. The deviation
when the cached fp16 embeddings are used instead (they are what 1a and 1b use) is
reported, not gated.

*1a (HELD, GT tracks).* For each track, the head's probabilities are averaged over
the pair windows where it is subject or object, weighted by the window's slot
count; cached embeddings are cast to float and renormalised. Reported: top-1 and
top-3 accuracy against the GT category, overall and per category.

*1b (test, detected tracks).*
- Pairs are matched to tracks by category and the box at their first frame.
- Per-track head probabilities as in 1a.
- **Disagreement:** tracker top-1 ≠ head top-1, over tracks that appear in at
  least one pair.
- Also: the share of Task 6's predictions (`eval_from_cache.predict`, "all",
  top 200 per video), matched to tracks as in F1, that involve a disagreeing
  track; and the tracker score on agreeing vs disagreeing tracks.

*Fusion rule (for the final run).*
- A track's class distribution is the normalised geometric mean of the tracker's
  per-track mean class-score vector and the head's per-track probabilities.
- The triplet uses the fused top-1 class and its fused score in place of the
  tracker's class and score.
- **The cache keeps only the tracker's top-1 class and score,** so in F2 (3c,
  fused categories) its vector is approximated as score on the top-1 and
  (1 − score) / 34 elsewhere, floored at 1e-6. The final run must capture the
  real vector (the mean vector in `gen_labels.py:611-619`).

**Gate 1:** 1a top-1 ≥ 70% **and** 1b disagreement ≥ 10%. Otherwise Part 1 is
dropped.

**Part 2 — temporal flicker.**
- Novel columns only; EOV = sigmoid(logit) · sigmoid(interactiveness); each
  slot's top 20 novel predicates; consecutive slots of the same pair.
- **Persistence:** Σ|top20(t) ∩ top20(t+1)| / Σ|top20(t)|.
- **Run lengths:** maximal runs of consecutive slots in which a predicate stays in
  the top 20 (median, mean, share of single-slot runs).
- **Spearman:** consecutive slots' 61-dim novel score vectors, with average ranks
  for ties.
- Inputs: (i) HELD, GT tracks; (ii) test, GT tracks (`cache_s3`); (iii) test,
  detected tracks (`cache_s3_sgdet`).
- **Smoothing rule (for the final run):** for each pair and predicate, the slot
  score becomes the mean of the raw scores at t−1, t and t+1 (those that exist),
  before post-processing. Window 5 is an ablation only.

**Gate 2:** persistence ≤ 75% on (ii) **or** (iii). Otherwise Part 2 is dropped.

**Part 3 — type rules.**
- A verb is the first token of a verb + spatial predicate (S1's definition).
- Subject types come from `scoring_rule.md`'s lists.
- **3a:** a rule exists where one type is ≥ 99.5% of a verb's base instances,
  with ≥ 100 instances, over **FIT only** (626 videos).
- **3b:** the share of HELD base GT instances (`held_gt.json`) that a rule would
  delete, given the GT subject category.
- **3c (reach):** the share of EOV-MMP's top-20 novel predictions per slot that
  the rules delete, on:
  - the 783 training videos;
  - (ii), with the cache's GT categories;
  - (iii), with the tracker's categories, and separately with the fused
    categories.

**Gate 3:** 3b ≤ 0.5% **and** 3c ≥ 3% on (ii) or (iii). Otherwise Part 3 is
dropped.

**Order.** Each part runs `--small` first, then full:
- small for part 1 is 1 video for the 1-0 check, plus 5 HELD and 5 detected-track
  videos;
- small for parts 2 and 3 is 5 videos per set;
- part 3 reads part 1's fused classes.

**Pinned by A14:**

| artefact | path | SHA256 |
|---|---|---|
| f2_run.py | `pilot_analysis/f2/f2_run.py` | `f1b2c23e90aa965d97bbb9d957c09035e7652f529c39174ef1a313a5f70ff094` |

#### A15 (2026-10-06): F2 coding fix — load cached tensors onto the CPU

Part 1's `--small` run crashed **before computing any result**: a device
mismatch in the CPU forward of modelC.
- **Cause:** session 3 saved the cached `slot_inputs` as CUDA tensors, and
  `f2_run.load_cache` called `torch.load` without `map_location`. On this
  machine, which has a GPU, they loaded onto `cuda:0`.
- **Fix:** `map_location="cpu"`. No rule, threshold or gate changes.
- Part 2's small run had completed and is unaffected: it only reads logits,
  which were saved on the CPU. It is re-run with the fixed file anyway.

`f2_run.py`'s pin moves from `f1b2c23e90aa965d97bbb9d957c09035e7652f529c39174ef1a313a5f70ff094` to:

| artefact | path | SHA256 |
|---|---|---|
| f2_run.py (A15) | `pilot_analysis/f2/f2_run.py` | `56db8776d66e2c366977cdc82c1671105e72ac23e0a4786a80a767673a3ca0c9` |

#### A16 (2026-10-06): F2 coding fix — JSON keys in Part 3's output

Part 3's `--small` run crashed while printing its results, **before any number
was shown or saved**. The false-delete breakdown used (subject, predicate) tuples
as dictionary keys, which JSON cannot hold. They are now strings,
`"subject/predicate"`. No rule, threshold or gate changes.

`f2_run.py`'s pin moves from `56db8776d66e2c366977cdc82c1671105e72ac23e0a4786a80a767673a3ca0c9` to:

| artefact | path | SHA256 |
|---|---|---|
| f2_run.py (A16) | `pilot_analysis/f2/f2_run.py` | `d2d04aacdbe80fac04d6096e89b5398c17cd3d5015c5bd07dc0107f3ccc2fac0` |

#### A17 (2026-10-06): final test T1 — type-consistency rules on the test set, one run

**Disclosure.** This is the **third evaluation on the test set**, after Batch B
(A8) and Batch B2 (A9). Batches A–B2, S1, F1 and F2 stay reported unchanged.

Written **before any test annotation is opened or any evaluation is run** in this
batch. Before A17, the script was exercised only in `--dry-run`:
- the PredCls path, the rules, the control and the diagnostics on HELD training
  videos, with the all split standing in for novel;
- the SGDet path label-free on 5 test videos' cached model outputs, without
  evaluation.

**The rules (F2 3a, FIT only; pinned file `f2/data/f2_part3.json`):** stand, sit,
lie, walk, run, creep, swim → the subject must be animate; stop → the subject must
be a vehicle.
- A predicate's verb is the first token of a verb + spatial predicate (S1/F2's
  definition), so 73 of the 132 predicates fall under a rule.
- Subject types: `scoring_rule.md`'s vehicle and animate lists. Any other
  category (ball, frisbee, skateboard, sofa) contradicts both rule types.

**What deletion means.** For every slot of every pair, the logit of each predicate
whose verb has a rule is set to −∞ when the pair's subject type contradicts that
rule. Through session 3's own `eval_from_cache.slot_scores`, that score becomes
exactly 0.
- This happens in both splits, to base and novel predicates alike, **before** the
  repo's post-processing (top 20 per slot, association, top 200 per video).
- Nothing else changes.
- **Subject category:** PredCls uses the GT category in the session-3 test cache
  (Task 4's path). SGDet uses the tracker's category (Task 6's cache and path).

**The single run** (`t1_run.py --go`, in order). Every cache file is checked
against `SHA256SUMS_session3.txt`. All mAP comes from
`eval_relation_detection_openvoc` with the shipped test GT, on predictions from
`eval_from_cache.predict`.
1. **Baseline check:** PredCls 21.70 novel / 37.20 all, and SGDet 15.79 /
   27.18, each within ±0.05. Otherwise STOP.
2. **PredCls + rules:** novel and all; mAP, R@50, R@100.
3. **SGDet + rules:** novel and all; mAP, R@50, R@100.
4. **Control (PredCls and SGDet, novel split):**
   - For each video, n = the number of (pair, slot, predicate) entries in the
     baseline per-slot top-20 novel sets that the rules delete.
   - The control deletes n entries chosen uniformly at random among that
     video's baseline top-20 novel entries (logit → −∞).
   - Seeds 0–4; mean and sd reported.
5. **Diagnostics** (test labels for reporting only):
   - deleted top-20 novel entries per verb;
   - **true positives lost:** baseline novel output predictions (top 200 per
     video) that break a rule and that the evaluator marked as hits at vIoU 0.5
     (the repo's `evaluate_v2` hit flags);
   - per-predicate novel AP, baseline vs rules, for the novel predicates under a
     rule, one predicate per evaluator call, with test instance counts.

**Decision rules.**
- **Beats METOR:** SGDet + rules ≥ 16.75 novel **and** ≥ 27.53 all. The
  margins are reported.
- **Improves the baseline but not METOR:** the gain is reported.
- **No gain, or worse:** reported as is.
- If the control comes within 0.3 novel mAP of the rules, the gain is not
  attributable to the rules.
- One run; no re-runs to change numbers. A crash before any number is printed
  may be re-run and is logged.

**Pinned by A17** (the reused scripts stay pinned where they are: `common.py`
A12, `span_common.py` A10, `evaluation.py`, `eval_from_cache.py`, the
post-processing and the evaluator A8):

| artefact | path | SHA256 |
|---|---|---|
| t1_run.py | `pilot_analysis/final/t1_run.py` | `02e309a006d2736678c7cd61a7ff7b27c9f5d4460021c3c56810a52812bb68dd` |
| F2 rules | `pilot_analysis/f2/data/f2_part3.json` | `68d454cc7706d5425eaa574181acc35cc14faf49d2d508d1941422af87b6fd40` |

#### A18 (2026-10-07): position verifier — build, gate on training videos, label-free test reach; T2 frozen

Written before anything in this batch was trained or scored. The only checks
before it were synthetic unit tests.
- **Training videos: base labels only** (the 3,301 novel instances dropped at
  parse; 0 used).
- **Test videos: cached model inputs only** (pair trajectories, per-slot logits).
  **No test relation label is opened and the evaluator is never run on test in
  this batch.**

**Covered values** (from `phi_frozen.json`; no predicate sets two of them).
Contained, relative motion and comparatives are not covered.

| value | base predicates | novel predicates |
|---|---|---|
| horizontal=left | creep_left, jump_left, left, lie_left, move_left, run_left, sit_left, stand_left, stop_left, swim_left, walk_left | fly_left |
| horizontal=right | creep_right, jump_right, lie_right, move_right, right, run_right, sit_right, stand_right, stop_right, swim_right, walk_right | fly_right |
| vertical=above | fly_above, sit_above | above, creep_above, jump_above, lie_above, move_above, run_above, stand_above, stop_above, walk_above |
| vertical=beneath | creep_beneath, jump_beneath, move_beneath | beneath, lie_beneath, run_beneath, sit_beneath, stand_beneath, stop_beneath, swim_beneath, walk_beneath |
| depth=front | creep_front, front, jump_front, lie_front, move_front, run_front, sit_front, stand_front, stop_front, swim_front, walk_front | fly_front |
| depth=behind | behind, creep_behind, lie_behind, move_behind, run_behind, stand_behind, stop_behind, walk_behind | fly_behind, jump_behind, sit_behind, swim_behind |
| proximity=adjacent | fly_next_to, jump_next_to, lie_next_to, next_to, stand_next_to, walk_next_to | creep_next_to, move_next_to, run_next_to, sit_next_to, stop_next_to, swim_next_to |

**Data.**
- All 800 training videos (boxes only, so ECC isn't needed).
- FIT 640 / HELD 160 (`train_split.json`); early-stop = Batch A's 63 FIT videos;
  fit-train = the other 577 FIT videos.
- One example = one (ordered pair, 30-frame slot) of the session-3 cache: GT
  option-B tracks, slots cut as `process_pred` cuts them.
- The slot's labels are the base predicates covering ≥ 15 of its frames
  (`slot_axes.slot_labels`).

**Features** (15 per frame where both boxes exist; offsets ÷ the mean of the two
box diagonals, absolute heights ÷ frame height):
- centre x and y offsets;
- left, right, top and bottom edge offsets;
- log width and log height ratio (difference of logs);
- IoU, and the intersection over each box's area;
- horizontal and vertical 1-D overlap fractions;
- each box's bottom-edge height.

A slot with fewer than 5 such frames abstains: no veto. The features are z-scored
with fit-train statistics made swap-symmetric: pooled over both pair orders, with
mean 0 on the negated features.

**Network.**
- g: a shared per-frame MLP (15 → 64 → 64, ReLU), masked mean and max pooling,
  an MLP (128 → 64, ReLU), then 4 outputs.
- Horizontal, vertical, depth: z = g(x(s,o)) − g(x(o,s)). Proximity:
  z = g(x(s,o)) + g(x(o,s)).
- P(left) = σ(z_h), P(right) = 1 − σ(z_h); likewise above/beneath,
  front/behind; P(adjacent) = σ(z_p).

**Targets.**
- An axis is supervised in a slot when its labels set exactly one value of it;
  conflicting slots are masked.
- Adjacent: positive when a label sets adjacent; negative when no label sets
  proximity but some label sets horizontal, vertical or depth (the weak
  negatives of `axis_check.md`).
- Loss: class-balanced BCE per axis (weights n / 2n₊ and n / 2n₋ on fit-train),
  summed over axes and normalised by observed entries.

**Training.** Adam, lr 1e-3, batch 256, ≤ 100 epochs, early stopping on the 63
videos' loss (patience 10). Seeds 0, 1 and 2; σ outputs averaged.

**Instance score.** A base instance with a covered value v is scored on its
covering slots: the slots of its ordered GT pair that it covers ≥ 15 frames. The
score is **the minimum over those slots of P(v)**, so a veto in any covering slot
counts as deleting it. Instances with no covering slot are excluded and counted.

**τ_v.**
- 5-fold video-level cross-fitting on all 640 FIT videos: folds from
  `default_rng(20261007)`'s permutation of the sorted FIT list, position mod 5.
- Each fold is trained on the other four, for each seed's best epoch count from
  the final run (fixed, no early stopping), 3 seeds; the out-of-fold
  predictions are pooled.
- τ_v is the (⌊0.0025 · n_v⌋ + 1)-th smallest out-of-fold score among FIT
  instances labelled v: the largest threshold with ≤ 0.25% of them below it.

**Veto rule.** Delete the prediction of predicate p in a slot when p sets a
covered value v on a surviving axis and P(v) < τ_v for that pair and slot (final
model). The mechanics are T1's (A17):
- the logit is set to −∞, so the score is exactly 0 through `slot_scores`;
- **before** the repo's post-processing, so each slot's top 20 refills from the
  remaining predicates;
- in every slot of every pair, in both splits, for base and novel predicates.

**Untrained comparison.** The same procedure with the frozen measured rule's frame
fractions as P(v):
- left/right and above/beneath over the slot (W0);
- front/behind over ± 1 slot (W1);
- adjacent: the fraction of the pair's frames (W3) with gap / mean diagonal below
  the frozen 0.19752.

τ is chosen the same way on FIT (the rule is untrained, so no cross-fitting).

**Gates (network; the untrained rule is reported side by side with its own
survivors).**
- **G-P1, GT boxes:** HELD false-delete = the share of HELD base instances with a
  covered value whose score falls below τ. Clopper–Pearson 95% CIs.
- **G-P2, box noise:** HELD tracks are jittered with seed 0.
  - Per component (centre x × w, centre y × h, log w, log h): a smooth offset
    (knots every 30 frames ~ N(0, s²), linearly interpolated) plus per-frame
    noise N(0, (s/2)²).
  - s is found by bisection so that the mean IoU with the original boxes, over
    all HELD track frames, is 0.85 and 0.70. Pair trajectories are rebuilt from
    the jittered tracks, and τ is unchanged.
- **Survival:**
  - An axis survives if its G-P1 rate ≤ 1.0% **and** its G-P2 rate at 0.70
    ≤ 1.0%.
  - Then, while the survivors' pooled G-P1 > 0.5% or pooled G-P2 (0.70) > 1.0%,
    the survivor with the highest G-P1 rate is dropped.
  - **If no axis survives, the batch stops at the report.**
- **Breakdowns** (surviving axes): subjects whose category has ≤ 5 FIT labelled
  slots, and verb families.
- **G-P3 (label-free reach on cached test inputs, surviving axes):**
  - Measured on SGDet and PredCls, on the novel top-20 entries per slot: the share
    deleted, and the share not already deleted by the type rules
    (**incremental**).
  - Also: the type-rules pipeline's final outputs (top 200 per video) affected
    or removed, the top 10 predicates, and the untrained rule side by side.
  - **PASS iff SGDet incremental reach ≥ 3%.**
  - Projection, a rough guide only: 0.36 × incremental / 7.6%.

**Final test T2 (frozen now, not run now; the fourth test evaluation, disclosed as
such).** PredCls and SGDet through the repo's evaluator:
- (a) EOV-MMP;
- (b) type rules, which must reproduce T1 exactly (22.25 / 38.03 and
  16.15 / 27.57);
- **(c) type rules + position verifier (primary);**
- (d) the position verifier alone;
- (e) type rules + the untrained position rule (its own surviving axes);
- (f) control: type rules + random deletion of as many extra top-20 entries
  per video as (c) deletes, drawn from the entries the type rules keep; mean of
  seeds 0–19.

Reported: novel and all mAP, R@50 and R@100, and a video-level bootstrap 95% CI
(1,000 resamples, seed 0) for (c) and for the paired difference (c) − (b).
**Primary claim:** (c) SGDet against METOR's published 16.75 novel / 27.53 all.
Only axes that survive the gates enter T2.

**Order.** Each step runs `--small` first (5 videos per subset; 5 test videos per
task for step 4), then full.

**Pinned by A18:**

| artefact | path | SHA256 |
|---|---|---|
| pos_common.py | `pilot_analysis/position/pos_common.py` | `d39b2484ac5db90154940588dd950e05e30ebff398543fa948efa3083589111c` |
| p_run.py | `pilot_analysis/position/p_run.py` | `507afb8acbb831283b58f2bb1b1ca30ee7095aa4622ae06024ba955292d78bc2` |

#### A19 (2026-10-07): position verifier coding fix — test videos with no pairs

The full step-4 run crashed on a cached SGDet test video with no subject–object
pair, **before any step-4 number was printed or saved**: featurising an empty
pair list gave a shapeless array. Videos with no pairs are now skipped and
counted; they have no slot, so they contribute nothing to any reach count. Steps
1–3 had completed and are unaffected. No rule, threshold or gate changes.

`p_run.py`'s pin moves from `507afb8acbb831283b58f2bb1b1ca30ee7095aa4622ae06024ba955292d78bc2` to:

| artefact | path | SHA256 |
|---|---|---|
| p_run.py (A19) | `pilot_analysis/position/p_run.py` | `69cd0228cc450142109e28d460a2e42e37b844fd8c7860e130dfd600addb0987` |

#### A20 (2026-10-07): time check D1 — test-label error analysis (disclosed)

**This batch reads test relation labels for error analysis only.**
- **No method, threshold or parameter is chosen from it.**
- The thesis discloses it.
- **Anything built after it is designed and tuned on training videos only,
  under its own amendment.**
- The evaluator runs only on EOV-MMP's existing baseline outputs.

A20 was written before the test relation annotations were opened in this batch.
The only checks before it were synthetic unit tests and a label-free comparison of
prediction files.

**Step 0 (before the labels are opened).** The repo's evaluator, on the existing
baseline outputs, must give novel 21.70 (PredCls; Task 4's `predcls_novel.json`)
and 15.79 (SGDet; Task 6's `sgdet_novel.json`) within ±0.05; all 37.20 / 27.18
is also checked. Otherwise STOP.

**The single source.** Session 3's caches: `cache_s3/test` (GT tracks) and
`cache_s3_sgdet/test` (detected tracks, tracker classes, per-slot logits for all
132 predicates).
- The final outputs re-derived from them by `eval_from_cache.predict` are
  identical to Task 4's and Task 6's dumps (checked label-free, and again in
  step 0).
- The pilot's `preds_from_pod/full/final_merged_*.json` has the same triplets,
  durations and box values. It differs only in integer box storage and in score
  rounding (≤ 1.1e-7). It is evaluated for reference only (novel SHA256
  `b89cba88111e3bf37c1cd6f4f2ddd15be06eb4209a81d6af8ecea94de4c52124`, all `91bb70e84404bf31dcb77daa6c7205c82d876b0757f780e17e6325059fdd45ea`).
- Pre-merge segments are regenerated from the caches with the repo's own
  post-processing.

**Definitions.**
- **Pass:** the official novel split (base predicates zeroed; ranks among the 61
  novel predicates). The all split comes second (ranks among 132).
- **Missed GT:** `gt2det_id = −1` from the repo's matcher
  (`eval_detection_scores_v2`, vIoU 0.5), run per video on the GT and final
  outputs, filtered by the evaluator's own category filter for the pass.
- **GT slots:** the pair's slots that overlap the GT extent by ≥ 15 frames.
- **Long / short:** more than one / exactly one GT slot, counted on the GT-track
  pair in `cache_s3` in both settings. If that pair was never formed, the count is
  ⌊duration / 30⌋, plus 1 if the remainder is ≥ 15 frames.
- **The pair:**
  - PredCls: the GT subject and object tracks, i.e. the cache pair with
    (sub_gt, obj_gt) = the GT's (subject_tid, object_tid).
  - SGDet: the ordered detected-track pair, of any class (s ≠ o), with the highest
    min(subject vIoU, object vIoU) against the GT trajectories, each detected track
    cut to the GT extent. vIoU is the repo's `viou`; ties go to the lowest track
    ids.
- **Rank:** 1 + the number of the pass's predicates with a strictly higher score
  in that slot. Scores are `slot_scores`, the same scores that choose the top 20.
- **Reachable at K** (K ∈ {20, 30, 40}):
  - take the pair's slots, over its whole timeline, with rank ≤ K;
  - group them into runs, joining across gaps of at most one slot;
  - take the run with the largest frame overlap with the GT (ties: earliest); its
    trajectory is the pair's boxes over the run's frames;
  - the GT is reachable iff the repo's `viou` gives min(subject, object) ≥ 0.5
    against it.

**Buckets** (first that applies):
- **U1** (SGDet): the best pair's min vIoU < 0.5.
- **U2** (SGDet): the tracker's class for the subject or object differs from the
  GT's.
- **U3:** pair not scored. The pipeline drops before scoring every ordered pair
  whose tracks overlap < 10 frames (`gen_labels.py:659`), a trimmed tail of < 10
  frames, and, in PredCls, GT tracks the repo densifier rejected. U3 applies if
  the pair was not formed, or if fewer than half of the GT's slots (the long/short
  count) are scored slots of the pair.
- **A1:** reachable at K = 20, but missed. It is split by re-running the repo's
  post-processing with the top-200 cut disabled (`max_per_video` = 10⁹):
  - (ii) the GT is matched in those uncut outputs, so the top-200 cut removed it;
  - (i) otherwise: association produced no matching instance.
- **N1:** reachable at K = 30 or 40, but not at 20.
- **C1:** not reachable at K = 40.

**CIs.** 95% percentile bootstrap that resamples whole videos (1,000 resamples,
seed 0).

**Step 2.**
- For missed long GT in A1, N1 and C1: the share of GT slots with rank ≤ 20, and
  the median rank in the other slots.
- For N1 and C1: the 15 most common (GT predicate → the predicate ranked first in
  a GT slot where that is not the GT predicate) pairs, each labelled, in this
  order, using F1's verb/spatial decomposition:
  - "bare term vs compound": same spatial component, and exactly one has verb
    `none`;
  - "same verb, different position";
  - "same position, different verb";
  - "other".
- For U2: the 10 most common (GT class → tracker class) pairs, subject and object
  counted separately, and how many involve the 10 novel object categories.
- For A1: the counts of (i) and (ii).

**Step 3, the reading rule** (SGDet, novel pass, pooled; the largest of the four
groups names the direction):
- **U1 + U2 + U3, upstream:** detection, tracking or object naming. Outside the
  relation head.
- **A1, after scoring:** joining slots into relations, or the top-200 cut. Points
  to a learned joining or ranking step.
- **N1, near miss:** the right predicate sits just below the cut. Points to a
  trainable time head that re-scores each slot from its neighbours.
- **C1, not seen:** a classifier limit. Points to a stronger model.

The largest group is stated with its share and CI, and so is the second largest.

**Pinned by A20:**

| artefact | path | SHA256 |
|---|---|---|
| tc_run.py | `pilot_analysis/time_check/tc_run.py` | `fae863dd65ee0c09a2cabfdab7a5f59df2e81832e8113c7c0383f7a5238d5996` |

#### A21 (2026-10-07): boundary fix E1 — nested runs; the fourth test evaluation (disclosed)

**Fourth test evaluation, disclosed.** The earlier three were B, B2 and T1.
- **The method was motivated by the D1 test-label diagnostic** (A20 and its
  post-hoc C1 description). In C1, 129 SGDet and 237 PredCls missed novel GT would
  match if the K = 20 run were cut to the GT's extent. The runs are over-long.
- **It is fixed here, before any test number for it exists.** Before A21 the
  script ran only on synthetic unit tests (a hand-built pair and a synthetic GT
  file). No cache file and no label were read.
- **A18's T2 is withdrawn, not run.** The position verifier failed G-P3 (0.46% <
  3%), so T2 has no position component. E1 takes the fourth evaluation's place.
- CPU only. Repo functions only: `process_pred`, `association` and `format_`
  (`inference/post_process.py`); the evaluator; `eval_from_cache.slot_scores`.

**Method (c), type rules + nested runs (primary).**
1. **Slot scores.** T1's (A17): every rule-violating predicate's logit is set to
   −∞ in every slot, then `slot_scores`. The novel pass zeroes the base columns.
2. **Five passes.** For each k ∈ {20, 10, 5, 3, 1}, the repo's `process_pred`
   with `clip_top_n` = k (per-slot top-k), then `association`, per pair. The
   top-200 cut is disabled.
3. **Pool.** The five instance sets are pooled in that order. Duplicates are
   removed by (subject track, object track, predicate, identical extent),
   keeping the highest score (the earlier on a tie).
4. **Score and cut.** The instance score is the repo's, unchanged: sbj_scr ×
   obj_scr × the mean slot score over the instance. Then the repo's `format_`
   cuts the pool to the top 200 per video.

An instance from a smaller k is a sub-run of a k = 20 run: the same pair and
predicate, cut where the predicate leaves the tighter top-k. **k = 20 alone is T1
and must reproduce it exactly.**

**Configurations.** PredCls (`cache_s3/test`) and SGDet (`cache_s3_sgdet/test`),
novel and all passes:
- **(a)** EOV-MMP: no rules, k = 20.
- **(b)** Type rules, k = 20. This is T1.
- **(c)** Rules + nested, k ∈ {20, 10, 5, 3, 1}. **Primary.**
- **(d)** Nested without rules.
- **(e)** Rules + a single tighter k ∈ {10, 5, 3, 1}: the k = 20 set pooled with
  that one k only, with the same dedupe and cut. Four rows.
- **(f) Control.** Every extra instance of (c) is moved to a uniformly random
  slot-aligned position inside the k = 20 run that contains it. An extra is a
  pooled instance not in the k = 20 set.
  - Kept: its length in slots, predicate, pair and score.
  - Its trajectories are the pair's boxes at the new position.
  - Then the same dedupe and cut.
  - Seeds 0–19, mean (with SD).

  The control keeps the number, length and scores of the extras. It removes only
  where the tighter top-k placed them.

**Reported for every configuration:**
- novel and all mAP, R@50 and R@100 (`eval_relation_detection_openvoc`);
- GT matched and unmatched predictions, from the repo's matcher
  (`eval_detection_scores_v2`) run per video on the evaluator's filtered lists.
  The per-video AP mean must equal the official mAP (asserted).

**For (c):**
- a video-level bootstrap 95% CI, 1,000 resamples, seed 0;
- the same for the paired difference (c) − (b).

**Match-level breakdown, (c) vs (b):**
- GT newly matched and GT lost;
- the change in unmatched predictions;
- (c)'s extras that are in its evaluated lists: hits and false positives.

**D1's over-long cases.** These are recomputed by `tc_posthoc.py`'s procedure on
`tc_missed.json`'s C1 rows, mapped to GT indices. The expected counts are 129
SGDet and 237 PredCls; a mismatch is reported, not a stop. Reported: how many
are matched in (a), (b), (c) and (d).

**Step 1 — HELD plumbing** (`--small`: 5 videos, then full).
- **Data:** stage 2's `held_gt.json` (157 HELD videos, base labels only; pinned
  in Batch A), with `cache_s3/train` pairs.
- **Pass:** base. Novel columns are set to −∞, so ranks are among base
  predicates; mode "all"; evaluated on the base split.
- **Run:** (b), (c) and (f), through the same code as step 3. (a), (d) and (e)
  differ from these only in their inputs.
- **Also checked:** (b) is identical to T1's own path (`eval_from_cache.predict`
  with T1's logits).
- **Descriptive only.** EOV-MMP was trained on these videos, so no decision
  depends on step 1.

**Step 2 — label-free test checks** (cached inputs only; no relation label, no
evaluator; `--small` then full). Per task and pass:
- identity: (b)'s final outputs equal T1's path, byte for byte;
- extras per video;
- how many of (b)'s outputs are pushed out of (c)'s top 200;
- instance length (frames) by origin k (the largest k whose set holds the
  instance), in the pool and in the top 200.

**STOP if the identity fails anywhere, or if no extra enters any top 200.**

**Step 3 — the test evaluation, once.** All configurations, both tasks, both
passes; this opens the test relation labels.
- **Reproduction:**
  - (a) within ±0.05 of 21.70 / 37.20 (PredCls) and 15.79 / 27.18 (SGDet);
  - (b) equal to T1's saved mAP (`t1_results.json`, to 1e-9), with its outputs
    identical to T1's path.

  If either fails, the numbers are reported and flagged, but not read.
- **Primary claim:** (c) SGDet against METOR's published 16.75 novel / 27.53 all.
  **PASS iff (c) SGDet beats both.** Reported with:
  - (c)'s CI and whether METOR's number lies inside it;
  - (c) − (b) with its paired CI.
- **Descriptive:** (d), (e), (f) and PredCls. The control is read as
  (c) − mean(f), and the number of the 20 seeds whose mAP reaches (c); no
  threshold.

**Fixed, not tuned:** the k list, its order, the dedupe key and rule, the
control, the bootstrap.

**Pinned by A21** (reused scripts stay pinned where they are: `common.py`,
`eval_from_cache.py`, `span_common.py`, `tc_run.py` (A20), `f2_part3.json`
(A17), `held_gt.json` (Batch A)):

| artefact | path | SHA256 |
|---|---|---|
| e1_run.py | `pilot_analysis/boundary/e1_run.py` | `041e09e13d98999568dc5411f192451cc0cd6db5f488087699913162920b9cbb` |
| T1 results (reference for (b)) | `pilot_analysis/final/data/t1_results.json` | `5d5acbe2731fa33b78ef7e6279c291b1572b3b25f4aabba700bab41111d7c65a` |
| D1 missed rows (C1 cases) | `pilot_analysis/time_check/data/tc_missed.json` | `2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2` |

#### A22 (2026-10-07): object check O1 — are novel objects found but thrown away, and does whole-track evidence fix their names?

**Training videos only.**
- No test annotation is opened and nothing runs on a test video.
- Step 1d reads D1's existing `tc_missed.json`; that is no new test computation.
- **This is not a test evaluation.**

Before A22, nothing ran on a model except synthetic tests. What was done:
- code reading;
- counts from the training annotations: GT tids per category, gaps, and the
  processed frame range (every annotated frame lies inside it);
- a listing of checkpoint keys, with no weights compared and no forward pass;
- synthetic unit tests:
  - the measurements on a hand-built dump of one training video;
  - the front-end copy on synthetic detector outputs, through the repo's
    `OVPostProcess`, deep_sort and AFLink.

**The front end, as the code shows it.** Step 1a gives the line references.
- **Frames.** Every frame in [begin_fid, end_fid) is processed. That range comes
  from the GT trajectory extents, as at test time: 162,330 of the 244,100
  training frames. Frames are resized to 336 × 336, giving a 24 × 24 TagCLIP
  grid.
- **The cut-off.** `OVPostProcess` keeps a query iff its max sigmoid class
  probability is > 0.35. If no query passes, it keeps the frame's single best
  query. It then keeps the top 300 (query, class) entries, and `filter` drops
  duplicate boxes (IoU > 0.8).
- **Detection scores.** Each detection gets a 35-score vector: the detector's
  probabilities for its top-300 classes, fused with modelB, the object
  classifier. The weights are 0.6 / 0.4 for antelope, bear, cattle, fox,
  giant_panda and red_panda, and 0.3 / 0.7 for the rest.
- **Tracking** is class-agnostic: deep_sort on RoI-mean appearance, then AFLink.
- **The track's class is already pooled over the whole track.** It is the
  argmax of the summed fused vectors over every track frame whose box matches a
  detection at IoU > 0.5 (`format_trajectories_test`). So rule (a) is itself a
  track-level rule, and (c) vs (a) compares raw detector probabilities with the
  repo's fused scores, both pooled.

**Definitions.**
- **GT tracks:** A5's option-B runs (`gt_tracks_train.json`, pinned): 4,044 runs
  from 2,430 annotated tids.
  - *Annotated frames* are the frames with a real box.
  - *Novel* means antelope, bear, cattle, fox, giant_panda, horse, lizard,
    motorcycle, red_panda or watercraft.
- **Pipeline tracks:** the repo's final trajectories (after the 0.15 × range
  length filter) at cut-off 0.35.
- **Found:** some pipeline track, **cut to the run's extent**, has repo-`viou`
  ≥ 0.5 with the run. This is A20's U1 convention.
  - *Reason:* training annotations stop at gaps of more than 30 frames and
    between runs. An uncut vIoU would charge a track for following the object
    through unannotated frames.
  - The uncut found rate is reported as well.
- **Found and named:** some pipeline track with cut vIoU ≥ 0.5 has the run's
  class.
- **Localised frame:** an annotated frame where some query's raw box (any of the
  300) has IoU ≥ 0.5 with the GT box, using the repo's `calculate_iou`.
- **Best-matching query (M2, M3):** among the localising queries, the one with
  the highest max class probability (ties: higher IoU, then lower index).
  - *Reason:* DETR puts several queries on one object, and the highest-IoU one is
    often a low-score duplicate. Choosing it would count detected objects as
    thrown away.
  - The highest-IoU variant is reported as a sensitivity row.
- **Kept / passes the cut-off:** the query is in the post-processor's output at
  0.35, including the fallback and the top-300 truncation.
- **Class-probability vector:** the detector's sigmoid probabilities over the 35
  classes (`pred_logits`).
- **Object embedding:** `pred_embed`, the query's 768-d CLIP-distilled
  embedding.
- **Text features:** the detector's own `clip_query`, the 35 CLIP text
  embeddings.
- **Storage.**
  - Boxes and probabilities are saved for every query and frame.
  - Embeddings are saved only for queries that become a detection at any cut-off,
    and for each annotated GT box's best-matching query under both rules. All
    queries' embeddings would be about 112 GB, and M4–M5 need only these.
- **ECC:** the training matrices. 783 of 800 videos have one; the other 17 run
  without camera compensation, as the repo does for a missing key.

**Implementation.** The front end is a verbatim copy of the test branch
(`end2end_model.py:196-317`) without modelC, plus:
- recording of query ids and line → detection matches;
- the association loop restricted to same-frame detections, in the same order;
- memoised modelB similarities. At 0.35 they are computed box by box as in the
  repo; boxes first needed at 0.2 or 0.1 are scored in one batched call per
  frame.

**Measurements.**
- **M1:** found and found-and-named rates per category, plus novel and base
  totals, with video-bootstrap CIs (1,000 resamples, seed 0).
- **M2:** unfound runs, over their annotated frames. Each goes into the first
  group that applies:
  - **(iii)** localised in < 50% of frames;
  - **(i)** otherwise, if the best-matching query is not kept in > 50% of the
    localised frames;
  - **(ii)** otherwise.

  Also reported: how often (ii)'s best query became a detection, i.e. survived
  `filter`.
- **M3:** over all runs, pooled over localised annotated frames, for the
  best-matching query: median, p25 and p75 of (a) its max class probability and
  (b) its GT-class probability, novel vs base. Also the unfound subset, and the
  share of frames whose argmax is the GT class.
- **M4:** on found novel runs (base for reference), using the best-matching
  pipeline track (highest cut vIoU; the first on ties). The frames are the ones
  the repo's own class rule counts: track lines with a matched detection.
  - (a) the repo's class;
  - (b) majority of the per-frame argmax of those detections' probability
    vectors (ties: lowest class index);
  - (c) **primary:** the argmax of their mean probability vector;
  - (d) the argmax of the cosine between their mean embedding and the text
    features.

  Reported: accuracy under each rule, the 10 most common confusions under (a),
  and (c) − (a) with a bootstrap CI.
- **M5 (descriptive only).** Cut-offs 0.2 and 0.1 (`pruning_threshold`, the
  repo's own parameter); everything else unchanged.
  - A track is kept iff the mean of the max class probability over its matched
    detections is ≥ 0.35.
  - Reported: novel and base found rates; tracks per video; extra tracks per
    video (tracks that match no 0.35 track at full vIoU ≥ 0.5); and the share of
    extra tracks that match no GT run (cut vIoU ≥ 0.5).
  - The ungated setting is reported too.

**Gates** (on the point estimates; CIs are reported):
- **G-O1:** among unfound novel runs, group (i) is ≥ 30%.
- **G-O2:** on found novel runs, accuracy (c) − accuracy (a) is ≥ 10 points.

**Reading:**
- both pass → objects ↔ time becomes the core of the new model;
- one passes → only that half carries forward;
- neither passes → relations ↔ time becomes the core.

**Order.**
1. **Step 1, CPU.**
   - 1b: the e2e checkpoint's detector vs the standalone detector checkpoint,
     tensor by tensor; both training configs; novel fragments in the trajectory
     file.
   - 1c: counts of tids and runs per category. **If fewer than 100 novel GT
     tracks: stop and ask.**
   - 1d.
2. **Step 2, GPU, 5 training videos first.** These are the videos with novel
   objects at positions 0, n/4, n/2, 3n/4 and n − 1 of the sorted list.
   - **The repo's own `End2End_Model.forward`** (test_ecc ← training matrices)
     must give trajectories identical to the copy at 0.35: class, extent and
     boxes exact, score within 1e-6. **Otherwise STOP.**
   - Then a runtime, cost and disk estimate. **Ask before the full run.**
   - Full run: all 800 training videos. If more than 1% fail: stop and report.
3. **Step 3, CPU, local.** All pins verified, then M1–M5 and the gates.

**Pinned by A22:**

| artefact | path | SHA256 |
|---|---|---|
| o1_common.py | `pilot_analysis/objects/o1_common.py` | `0a5746ba2a59139ccc86b19c91b764de510b2963f073698e827294da646f4b76` |
| o1_step1.py | `pilot_analysis/objects/o1_step1.py` | `af86e5dc55c30eb050af0ee1e0ca8464aafa660ef3f8828301e346dc2dffaf3a` |
| o1_dump.py | `pilot_analysis/objects/o1_dump.py` | `ff8b93d112d195797342a6a8d4e6d5b32e0443550fbcd2a3fbe236045a516df9` |
| o1_measure.py | `pilot_analysis/objects/o1_measure.py` | `ee5e8087ba20f9957349c9596e8e341cc2963ac86a2cbb9f534d4cdc670ef34e` |
| run_o1_small.sh | `pilot_analysis/objects/run_o1_small.sh` | `2db06bbd96c17a8cbb708056251872f1b5b5e79d5ff0f905e1468bce4ae0d767` |
| run_o1_full.sh | `pilot_analysis/objects/run_o1_full.sh` | `9a29bb4feec16cd38304c4f24867e3edaca6cb3852a4f677cbdaa996c9d4e38d` |
