# Feasibility batch F1 — compositional prior (Idea A) and track fragmentation (Idea B): report

**Verdict: all three gates fail.**

| gate | needed | got | |
|---|---|---|---|
| **A1** (prior recognises hidden predicates) | ≥ 2 × random **and** ≥ verb-frequency + 2, on both sets | A6 set: 12.54 vs random 10.25 (**1.22×**) ✗; LO2: 17.06 vs 6.25 (2.73×, +3.50) ✓ | **FAIL** |
| **A2** (veto safety) | false-veto rate ≤ 0.5% | **6.14%** (268 / 4,365 HELD base instances) | **FAIL** |
| **B1** (track fragmentation) | ≥ 10% of detected tracks in a stitch candidate | **3.95%** (26 / 658) | **FAIL**: Idea B dropped |

Scope:
- CPU only, about 2 minutes of training.
- Part A: training videos only, base labels only (0 novel labels used, asserted).
- Part B1 read only EOV-MMP's detected test tracks and per-slot logits (session
  3 Task 6's `cache_s3_sgdet`). No test annotation was opened, and no test
  evaluation was run.

## 1. A13 and the pinned artefacts

| | SHA256 |
|---|---|
| preregistration.md with A13 (`preregistration.sha256`, line 1) | `b43dd969dfd795eb3b529312e808c94ae96350c4141980a9b684b2eaa0ad7436` |
| A13 text alone (5,501 bytes) | `da928ead8612d4529b5ca493526435dd7848c577d7d0a479794f44e5e1645bcc` |
| the first 51,490 bytes (through A12), unchanged | `f86d1355be655cf7dc28a49ce06bc511940a9a8727cf88217cc7d729dfab5776` |

**56 pins verified** before every run (latest pin per path). They include
`f1_prior.py`, `f1_tracks.py`, S1's results (its subject-type rules), A6's and
A7's hidden sets, and stage 2's slots, labels and HELD files. The CLIP
embeddings (`once.pt`) and every `cache_s3_sgdet` file read were checked
against `SHA256SUMS_session3.txt`.

A13 records one interpretation of the brief. **An atomic predicate uses its own
verb-type term plus a shared spatial "none" term**: the general formula with
s = none, so that atomic predicates such as `ride` can depend on the object.

## 2. Decomposition

29 verb-type components (11 verbs, `none`, 17 atomic) and 13 spatial
components. Neutral means no base carrier in the full version.

| verb-type component | predicates: spatial component (`*` = novel) |
|---|---|
| `creep` | above*, away*, behind, beneath, front, left, next_to*, past*, right, toward* |
| `fly` | above, away*, behind*, front*, left*, next_to, past, right*, toward, with |
| `jump` | above*, away*, behind*, beneath, front, left, next_to, past*, right, toward, with* |
| `lie` | above*, behind, beneath*, front, inside*, left, next_to, right, with* |
| `move` | above*, away*, behind, beneath, front, left, next_to*, past*, right, toward*, with |
| `none` | above*, away*, behind, beneath*, front, left, next_to, past*, right, toward* |
| `run` | above*, away*, behind, beneath*, front, left, next_to*, past, right, toward*, with |
| `sit` | above, behind*, beneath*, front, inside*, left, next_to*, right |
| `stand` | above*, behind, beneath*, front, inside*, left, next_to, right, with* |
| `stop` | above*, behind, beneath*, front, left, next_to*, right, with* |
| `swim` | behind*, beneath*, front, left, next_to*, right, with |
| `walk` | above*, away*, behind, beneath*, front, left, next_to, past*, right, toward*, with |
| atomic (own verb-type component, spatial `none`) | `bite`* **(neutral)**, `chase`, `drive`* **(neutral)**, `fall_off`, `faster`, `feed`* **(neutral)**, `fight`* **(neutral)**, `follow`, `hold`* **(neutral)**, `kick`* **(neutral)**, `larger`, `play`, `pull`* **(neutral)**, `ride`, `taller`, `touch`, `watch` |

Spatial components (13): `above`, `away` **(neutral)**, `behind`, `beneath`, `front`, `inside` **(neutral)**, `left`, `next_to`, `none`, `past`, `right`, `toward`, `with`.

In the minus versions, the hidden atomic predicates also become neutral, because
their only carrier is themselves: `larger` and `taller` in minus-A6, `follow` in
minus-LO2.

## 3. Training

Training stopped early on the early-stop slots' cross-entropy, patience 20.

| version | fit-train slots | early-stop slots | epochs run / best (seeds 0, 1, 2) | early-stop CE |
|---|---:|---:|---|---|
| full | 9,044 | 805 | 38/18, 35/15, 36/16 | 3.068, 3.069, 3.076 |
| minus-A6 | 6,187 | 601 | 42/22, 44/24, 41/21 | 2.791, 2.784, 2.795 |
| minus-LO2 | 7,815 | 749 | 38/18, 36/16, 35/15 | 2.934, 2.937, 2.929 |

The CE is over 71, 61 and 61 predicates respectively; the uniform-guess CE would
be ln 71 = 4.26 or ln 61 = 4.11. Per-epoch curves are in
`data/f1_prior_results.json`.

## 4. Gate A1 — hidden predicates from components alone

Setup: HELD, PredCls on GT tracks, only the hidden 10 compete, through the
repo's post-processing and evaluator. The prior uses the matching minus-version.

| set | prior-only mAP | R@50 | R@100 | random (sd) | verb-frequency | ratio to random | over verb-frequency | |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A6 | 12.54 | 17.59 | 27.71 | 10.25 (0.69) | 7.67 | **1.22** | +4.87 | ✗ |
| LO2 | 17.06 | 24.39 | 31.01 | 6.25 (0.80) | 13.56 | 2.73 | +3.50 | ✓ |

**Verdict: FAIL.** The A6 set misses the 2× ratio.

**Per-predicate slot-level AUC (HELD labelled slots):**

| A6 set | n+ | AUC | | LO2 set | n+ | AUC |
|---|---:|---:|---|---|---:|---:|
| stand_left | 303 | 0.498 | | stand_next_to | 113 | 0.714 |
| stand_right | 301 | 0.503 | | right | 88 | 0.560 |
| stand_front | 201 | 0.434 | | stop_behind | 102 | **0.954** |
| larger | 218 | 0.499 | | left | 73 | 0.460 |
| taller | 203 | 0.524 | | move_right | 82 | **0.920** |
| stand_behind | 166 | 0.393 | | next_to | 64 | 0.429 |
| walk_left | 160 | 0.615 | | stop_left | 77 | **0.901** |
| walk_right | 134 | 0.615 | | fly_with | 72 | **0.973** |
| walk_behind | 115 | 0.618 | | move_front | 72 | 0.863 |
| walk_front | 113 | 0.627 | | follow | 57 | 0.508 |

**This pattern is the main finding of Part A.** A prior over (subject category,
object category) recognises *who does what*: the verb. `stop_*`, `move_*` and
`fly_with` reach AUC 0.86–0.97, because only vehicles stop, move or fly in
formation. It cannot recognise *where*: left, right, front, behind or next_to sit
at 0.39–0.56, since the same category pair occurs in every spatial arrangement.
Atomic hidden predicates (`larger`, `taller`, `follow`) are neutral and score
0.50–0.52.

The A6 set is mostly spatial variants of the same two verbs (4 `stand_*`, 4
`walk_*`), so the prior can't separate them. LO2 contains verb contrasts, so it
can.

## 5. Gate A2 — veto safety (full prior)

Rule: veto (a, p) if P(v | a) < 0.005 for p's verb-type v, and v has ≥ 100 FIT
base instances. 19 verb-type components are vetoable (all 12 verb components,
`none` included, plus 7 atomic). The rule vetoes 258 (subject, verb) pairs.

| | |
|---|---|
| HELD base GT instances | 4,365 |
| **falsely vetoed** | **268 (6.14%)**, against a limit of 0.5%: **FAIL** |
| label-free: EOV-MMP's top-20 novel predictions vetoed (783 videos) | 51,079 / 266,360 = **19.2%** (S1's rule-violation share: 8.4%) |

**Where the false vetoes come from** (a diagnostic after the verdict: the saved
prior reloaded, and the 268 reproduced exactly):

| subject / verb-type | false vetoes | FIT labelled slots with this combination | P(v / a) |
|---|---:|---:|---:|
| whale / `none` (bare spatial terms) | 174 | 2 | 0.0031 |
| watercraft / `none` | 48 | 1 | 0.0048 |
| bear / swim | 8 | 7 | 0.0018 |
| antelope / jump | 6 | 0 | 0.0049 |
| person / swim | 5 | 1 | 0.0002 |
| all other combinations (e.g. monkey/jump 4, turtle/taller 4, motorcycle/lie 3, dog/ride 2) | 27 | 0–14 | < 0.005 |

83% of the false vetoes come from bare spatial relations of two rare subject
categories. The prior learned "whales are almost never left of anything" from 2
FIT slots, and HELD happens to contain a whale video full of spatial relations.
Even without those two cases, the remaining 46 vetoes are 1.05%, still over the
limit. The pre-registered rule was not changed.

**Against S1's rules:** the veto covers every S1 rule (stand, sit, walk, run,
creep, swim → animate; stop → vehicle). It also vetoes more: animate subjects for
`stop`, but also e.g. `creep` for person and horse, and `swim` for dog and person.
That is where it overreaches.

## 6. Descriptive (A5)

**P(v | a), full prior, for the 10 most frequent FIT subjects** (components with
P ≥ 0.01, or vetoable):

| subject | stand | walk | sit | lie | run | none | move | stop | fly | ride | larger | taller |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 0.209 | 0.116 | 0.149 | 0.005 | 0.016 | 0.262 | 0.001 | 0.000 | 0.003 | 0.099 | 0.018 | 0.043 |
| car | 0.002 | 0.001 | 0.000 | 0.001 | 0.001 | 0.014 | 0.318 | **0.562** | 0.001 | 0.001 | 0.050 | 0.029 |
| airplane | 0.005 | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 | 0.044 | 0.012 | **0.906** | 0.001 | 0.005 | 0.008 |
| zebra | 0.416 | 0.404 | 0.005 | 0.031 | 0.046 | 0.005 | 0.001 | 0.000 | 0.001 | 0.000 | 0.031 | 0.026 |
| elephant | 0.527 | 0.261 | 0.003 | 0.018 | 0.024 | 0.009 | 0.002 | 0.001 | 0.001 | 0.000 | 0.069 | 0.059 |
| dog | 0.184 | 0.368 | 0.029 | 0.079 | 0.067 | 0.044 | 0.003 | 0.001 | 0.002 | 0.003 | 0.042 | 0.028 |
| monkey | 0.104 | 0.284 | **0.406** | 0.034 | 0.018 | 0.042 | 0.000 | 0.000 | 0.000 | 0.003 | 0.013 | 0.010 |
| bicycle | 0.002 | 0.002 | 0.000 | 0.004 | 0.003 | 0.008 | **0.768** | 0.136 | 0.002 | 0.000 | 0.008 | 0.004 |
| cattle | 0.488 | 0.297 | 0.004 | 0.071 | 0.042 | 0.009 | 0.003 | 0.002 | 0.001 | 0.000 | 0.030 | 0.023 |
| antelope | 0.564 | 0.234 | 0.008 | 0.017 | 0.053 | 0.025 | 0.002 | 0.001 | 0.001 | 0.001 | 0.028 | 0.037 |

**The full prior's top-5 novel predicates for the 10 most frequent FIT (subject,
object) pairs:**

| pair | top 5 novel (P) |
|---|---|
| airplane → airplane | fly_left .28, fly_front .24, fly_right .24, fly_behind .16, fly_away .06 |
| zebra → zebra | stand_inside .16, walk_away .14, stand_with .13, stand_beneath .13, walk_beneath .12 |
| car → car | stop_with .29, stop_next_to .13, move_away .08, move_next_to .07, fly_left .06 |
| person → person | away .11, beneath .10, stand_with .09, stand_inside .08, stand_beneath .08 |
| elephant → elephant | stand_with .25, stand_inside .21, stand_beneath .11, walk_away .10, walk_beneath .05 |
| monkey → monkey | sit_behind .37, sit_inside .13, sit_next_to .12, walk_away .07, walk_above .05 |
| cattle → cattle | stand_inside .17, stand_with .15, stand_beneath .15, walk_away .10, walk_beneath .09 |
| person → bicycle | above .39, stand_above .31, walk_above .16, run_above .02, sit_behind .01 |
| antelope → antelope | stand_inside .21, stand_beneath .18, stand_with .13, walk_away .09, walk_beneath .07 |
| bicycle → person | stop_beneath .52, move_away .11, move_next_to .05, move_past .05, jump_behind .03 |

The verbs are sensible: fly for airplanes, stop and move for vehicles, sit for
monkeys. **The spatial choices often aren't:** `stand_inside` for zebra →
zebra, `stand_beneath` for elephant → elephant. The neutral spatial components
(`inside`, `away`) take the mean spatial vector, which ranks them in the middle
for every pair, high enough to enter many top-5 lists.

## 7. Part B1 — fragmentation of EOV-MMP's detected test tracks (label-free)

| | |
|---|---|
| test videos / detected tracks | 200 / 658; all dense (box per frame) |
| track length (frames) | min 10, quartiles 60 / 111.5 / 225, 95th pct 479, max 900; 5.3% under 30, 38.3% under 90 |
| stitch candidates (same category, B starts 1–30 frames after A, close boxes) | **13**, in 13 videos (at most 1 per video); gaps 1–30 frames |
| tracks in ≥ 1 candidate | **26 / 658 = 3.95%** |
| SGDet relation predictions (Task 6's path, "all", top 200 per video) | 25,784, all matched to their tracks |
| predictions involving a candidate track | 1,687 = 6.5% |

**Gate B1: FAIL** (3.95% < 10%). **Idea B is dropped.** EOV-MMP's tracker
(deep_sort + AFLink) already links almost everything this rule could stitch.

## 8. Surprises and caveats

1. **The prior is a verb prior, not a relation prior.** Category pairs predict
   *who does what* (AUC 0.86–0.97 for vehicle verbs and fly_with) but not *where*
   (≈ 0.5 for left/right/front/behind/next_to). Gate A1's outcome therefore
   depends on the hidden set's make-up. A6's set (8 spatial variants of
   stand/walk, plus larger/taller) fails; LO2's (verb contrasts) passes. A
   test-time benefit would be limited to verb confusions, the same territory as
   S1's side count.
2. **The veto fails on rare subjects, not on the rules it was meant to enforce.**
   Its intended targets (stop → vehicle, walk/stand → animate) behave as S1's
   rules did. But a learned P(v | a) below 0.005 from 1–2 training slots is not
   evidence of impossibility. Whale and watercraft spatial relations make up 83%
   of the false vetoes. A rule restricted to verbs with real support would be a
   different, post-hoc rule; it was not tested.
3. **The label-free veto rate (19.2%) is more than twice S1's rule-violation share
   (8.4%),** because the veto also covers `none` and 7 atomic components. Much of
   the extra is the same overreach that causes the false vetoes.
4. **Neutral spatial components push `inside` and `away` into top lists.** The
   mean-vector rule is not neutral in rank, only in parameter space.
5. **B1 measured the tracks EOV-MMP actually produced on test** (model outputs).
   No test label was read; the gate concerns tracking only.

## 9. Hashes of new files

| file | SHA256 |
|---|---|
| `f1_prior.py` (pinned by A13) | `37b5fdff9a6018505b04978f6cd2f2d5ca18212587eaa40858718f7debd4d4d8` |
| `f1_tracks.py` (pinned by A13) | `090c54d2abbc6c522fbf977248d8e8d8465e3a19ba5face990635c391623c817` |
| `data/f1_prior_results.json` | `94dca45e4e24e5b5f94c9eabd4bfb21cc81f9b4b514f9c88378a49000e0e7964` |
| `data/f1_prior_models.pt` (3 versions × 3 seeds, components, carriers) | `ccbc2a17add1f9c51dbdf55441caf316f02f49825ca0f382cccd54f38a007350` |
| `data/f1_tracks_results.json` | `d7b091a3853d72c69bd29967a8f77b6e4668829286e24ab3091442c204f6f57c` |
| `data/split_A6.json`, `data/split_LO2.json` (identical to A2's split files) | `1897dcbe…`, `b435ef3e…` |

Logs: `logs/f1_prior.json` and `logs/f1_tracks.json`, plus the `*_small.json`
plumbing runs; console output in `f1_*_full.out` and `f1_*_small.out`. `data/`
and `data_small/` are git-ignored.
