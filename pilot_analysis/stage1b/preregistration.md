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
