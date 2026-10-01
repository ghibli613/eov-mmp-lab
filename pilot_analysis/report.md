# EOV-MMP diagnostic pilot — final report

Run completed 2026-09-02. All phases that do not require modifying the model are
done. The working record, with derivations and code references, is
[`PILOT-STATUS.md`](PILOT-STATUS.md); §-references below point into it.

**Headline: the baseline reproduces, H2 is refuted, and the failure is upstream
of everything the protocol proposed to measure.**

---

## Environment

RTX 4090 (24 GB VRAM, 1007 GiB system RAM), Vast.ai, ~4 h, ~$1.40.
`frame_stride 1`, `clip_len 30`, `clip_top_n 20`, `max_per_video 200`,
checkpoint `..._end2end_base-001.pth` (epoch 6), both predicate splits from one
fused pipeline pass.

Two host requirements neither the paper nor the repo documents, both measured:

- **≥32 GB system RAM.** `dataset.__getitem__` holds a video's per-frame tensors
  twice during `torch.cat`. Colab's 12.7 GB died reproducibly on a 645-frame
  video; the longest test video is 1,234 frames (§B.22).
- **Cap `OMP_NUM_THREADS`.** On a 255-core box, torch's default of one thread per
  core gave **228 s/video**; capping at 8 gave **60**. A 4090 was 2.3× slower
  than a T4 until this was set (§B.23).

## Phase 1 — reproduction: PASSED

| | measured | checkpoint's reported | paper |
|---|---|---|---|
| **all mAP** | **27.18** | 26.88 (+0.30) | 26.34 |
| **novel mAP** | **15.79** | 15.64 (+0.15) | 15.04 |
| all R@50 / R@100 | 17.10 / 19.46 | — | 16.48 / 19.54 |
| novel R@50 / R@100 | 16.36 / 19.01 | — | 16.03 / 18.18 |

200/200 videos, scored over the full test set. Both mAPs within +0.5 of the
checkpoint's own figures — inside even the protocol's original ±0.5. Recall
matches the paper on all four numbers. **Nothing downstream can be dismissed as a
failed baseline.**

Settled on the way: `--frame_stride 30`, which the paper's Implementation Details
imply, costs **45× accuracy for 17% runtime** with these checkpoints and produces
single unmerged 30-frame segments. The released code has that sampling commented
out. Use stride 1 (§B.20).

## Phase 2 — H1′: supported on novel, not on base, and the control is inconclusive

| group | split | inst | videos | mAP | R@50 | R@100 |
|---|---|---|---|---|---|---|
| `geometric_static` | base | 2068 | 175 | **34.11** | 24.18 | 25.53 |
| `geometric_dynamic` | base | 1877 | 151 | **23.34** | 14.06 | 14.81 |
| `geometric_static` | novel | 266 | 76 | **21.97** | 26.69 | 27.44 |
| `geometric_dynamic` | novel | 306 | 81 | **9.33** | 8.82 | 10.78 |
| *appearance_static* | *base* | *206* | *106* | *58.26* | *45.63* | *45.63* |
| *appearance_dynamic* | *base* | *79* | *38* | *24.67* | *29.11* | *29.11* |
| *appearance_static* | *novel* | *8* | *8* | *62.50* | *62.50* | *62.50* |
| *appearance_dynamic* | *novel* | *25* | *20* | *12.10* | *36.00* | *36.00* |

Base rows from the `all` pass, novel rows from the `novel` pass — the
published-comparable setting (§B.19). Italic rows are descriptive: 8 and 25
instances cannot carry a verdict (§C.2).

**H1′ verdict against the ≤0.5× criterion:**

| split | dynamic / static | ratio | verdict |
|---|---|---|---|
| base | 23.34 / 34.11 | **0.684** | not supported |
| novel | 9.33 / 21.97 | **0.425** | **supported** |

The *direction* is consistent — dynamic is worse on both splits — and the gap
widens on novel. But the criterion is met on one split only, and §B.19 argued the
0.5 bar was a large inference from an 0.0086 gap in text separability. Report the
ratios, not the pass/fail.

**The granularity control did not work.** §B.19 flagged that
`geometric_dynamic` spreads its instances over 78 predicates against
`geometric_static`'s 44, so half the data per label — a complete alternative
explanation for the gap. Per verb family, meant to hold that constant, is too
noisy to settle it: matched-pair ratios came out 0.461, 0.223, 0.172, **1.445**,
**2.751**, 0.000, 0.000, 0.000. Several families score exactly 0.00 (`creep`
both splits, `swim` novel, `fly` novel), which is a floor effect rather than a
measurement, and `lie`/`fly` on base runs the *opposite* way at 2.751.

**So the group-level gap is real but its cause is not established.** Label
granularity remains as plausible as verb semantics. Anyone reporting the 0.425
should report this too.

## Phase 2.3 — the pre-registered mechanism prediction: REFUTED

§B.11 measured that CLIP's frozen text embeddings collapse on the *spatial*
component of compositional names (`fly_front` ~ `fly_right` at 0.9798; 38 of 61
novel predicates within 0.95 of a base predicate) and predicted, before any
prediction existed, that novel-predicate errors would cluster **within** a verb
family.

| | |
|---|---|
| within the same verb family | 50 (**39.7%**) |
| across verb families | 76 (**60.3%**) |

**Wrong, and by a clear margin.** Errors are mostly across verb families.

The top confusions say something more interesting than the aggregate:

```
stand_above  -> ride            17   cross-verb
stand_with   -> stand_next_to    7   same-verb
walk_away    -> stand_right      5   cross-verb
stop_beneath -> move_beneath     4   cross-verb
walk_toward  -> stand_left       3   cross-verb
```

`stand_above → ride` is the single largest error and is arguably an **annotation
ambiguity**, not a model failure — a person above a bicycle usually *is* riding
it. And `walk_away → stand_right`, `walk_toward → stand_left` are motion
predicates collapsing into *static* ones: the model sees the pair and the
geometry but not the movement. That supports H1′'s direction while contradicting
its proposed mechanism.

## Phase 3 — H2: REFUTED, and the simulation that suggested otherwise was wrong

| variant | all mAP | novel mAP |
|---|---|---|
| (a) greedy `association()` | 27.18 | 15.79 |
| (b) oracle merge, mean | 27.26 | 15.72 |
| (b) oracle merge, max | 27.04 | 15.37 |
| **oracle gain** | **+0.07** | **−0.07** |

Against a ≥2 mAP criterion: **not supported, by a factor of 30.** The greedy
merge costs essentially nothing.

`merge_simulation.py` predicted an 11–26 mAP gain. That simulation cut GT
instances into segments and dropped some at random, which turns out not to
resemble real model output at all — the caveat in that script's header, that its
fixture could not validate direction, was the operative one. **Do not cite the
simulated figure.**

Relaxed vIoU shows why the merge is not the lever:

| vIoU threshold | greedy | oracle | gap |
|---|---|---|---|
| 0.5 | 27.18 | 27.26 | 0.07 |
| 0.3 | 33.22 | 33.18 | −0.04 |
| 0.1 | 39.23 | 39.28 | 0.06 |

Relaxing localisation gains **12 mAP** (27.18 → 39.23) — so temporal extents
*are* substantially wrong. But the oracle gap stays ~0 at every threshold, so the
wrongness is not in how segments are *merged*. It is in the extents the merge is
given, which come from the detector and tracker.

## The finding that replaces H2

```
GT instances >= 2 segments long: 2834
  lost entirely      2046  (72.2%)
  correctly merged     775  (27.3%)
  fragmented (>=2)      13  ( 0.5%)
```

**72.2% of long ground-truth instances have no matching prediction at all.**
Fragmentation is 0.5%, so §C.5's warning was right that fragmentation is the
wrong metric — but the loss rate reveals something bigger than a merge problem:
for nearly three-quarters of long instances there is nothing to merge, because
the correct triplet was never produced at a vIoU the evaluator accepts.

That is a **detection and classification** failure, upstream of every mechanism
the protocol proposed. Combined with the 12 mAP available from relaxed
localisation, the headroom is in trajectory extents and predicate scoring, not in
post-processing.

## Verdicts

| hypothesis | verdict |
|---|---|
| **Phase 1** reproduction | **PASSED** — 27.18 / 15.79 against 26.88 / 15.64 |
| **H1′** dynamic ≤ 0.5 × static | **supported on novel (0.425), not on base (0.684)**; direction consistent, cause not established |
| **H1′ mechanism** (within-verb-family confusion) | **REFUTED** — 39.7% within vs 60.3% across |
| **H2** oracle-merge gain ≥ 2 mAP | **REFUTED** — +0.07 mAP |
| **H2** fragmentation ≥ 25% | **REFUTED** — 0.5% |

## Surprises and caveats

1. **The pilot's own predictions failed more often than they held.** H2 was
   refuted, its simulated magnitude was misleading, and §B.11's confusion
   prediction was wrong. The two that survived are Phase 1 and H1′'s direction.
2. **The largest single error may be a labelling artifact.** `stand_above → ride`,
   17 instances, where both readings describe the same scene.
3. **`appearance_static` scores highest of any group** (58.26 base, 62.50 novel)
   on 206 and 8 instances. With 8 instances over 8 videos the novel figure is
   noise; the base figure is worth a look, since it contradicts any story where
   needing pixels is the hard case.
4. **Phase 4 needs no compute.** It is the paper's own SGCls/PredCls, already
   published: novel SGDet 15.04 → SGCls 17.96 → PredCls 21.65, giving +2.92 for
   trajectory detection and +3.69 for object classification (§B.17).
5. **A possible code/paper mismatch on the novel split** remains unresolved: the
   code restricts novel-split object classification to novel categories where the
   paper says all (§B.17). It bears on the novel mAP and is worth settling by
   inspection.
6. **Two undocumented host requirements** (§B.22, §B.23) — 32 GB RAM and a capped
   thread pool — are reproducibility findings in their own right.

## What this points at

The protocol set out to test whether the error is semantic (H1) or temporal (H2).
The answer measured here is **neither, as posed**: post-processing costs ~0.07
mAP, and 72% of long instances are never produced at all. The remaining headroom
sits in trajectory extents (12 mAP from relaxed localisation) and in predicate
scoring on motion predicates, which collapse into static ones.

For the thesis, the defensible claims are: the baseline reproduces; dynamic
geometric predicates are roughly 2× worse than static ones on the novel split,
cause undetermined between verb semantics and label granularity; the greedy merge
is not a bottleneck; and the dominant failure is that most long relation
instances are never detected.

---

# Follow-up: what causes H1′?

Two CPU-only analyses on the existing dumps. No new inference. The live question
is whether the dynamic/static gap reflects **(a) motion blindness** — the model
does not encode movement — or **(b) label granularity** — `geometric_dynamic`
spreads over 78 labels against `geometric_static`'s 44, so less supervision and
more competitors per label. (a) motivates an architectural fix; (b) makes H1′ a
benchmark artifact.

## Pre-registration

Written and committed **before** the numbers were computed.

### Analysis 1 — error destination

For every misclassified GT instance, which group does the *predicted* predicate
belong to?

| | under (a) motion blindness | under (b) granularity |
|---|---|---|
| dynamic-GT errors landing on static predicates | **high, above every null** | at the null |
| static-GT errors landing on dynamic predicates | **low** | at the null |
| asymmetry | **large, dynamic→static ≫ static→dynamic** | none; roughly symmetric |
| "static counterpart" rate (same spatial term, motion dropped) | **elevated** | at chance among static labels |

Reasoning: granularity is a *within-group* handicap — it predicts a dynamic label
losing to its 77 dynamic neighbours, not systematic leakage across the group
boundary. Motion blindness predicts exactly that boundary crossing, because
without movement the evidence remaining is the static configuration.

**Nulls to compare against**, all reported:
1. **uniform over labels** — 44/132 = 33.3% static;
2. **the model's own prediction distribution** — the empirical static share of
   its top-1 predictions, which absorbs any bias toward static labels;
3. **GT-frequency-weighted** — the static share of test instances.

Null 2 is the strictest and the one the verdict should rest on.

### Analysis 2 — granularity regression

Does group membership explain per-predicate AP once competition is controlled?

| | under (a) | under (b) |
|---|---|---|
| base: group coefficient after controlling `log(train count)` | **survives, negative for dynamic** | **vanishes** |
| novel: group coefficient after controlling NN text similarity | **survives** | vanishes |
| partial R² of the group indicator | non-trivial | ≈ 0 |

Because the sample is small and heavily censored at AP = 0, effect sizes and
intervals are reported rather than a p-value verdict, and every fit is shown with
and without the zero-AP predicates.

**One input note:** all 61 novel predicates appear in the *training annotations*
(3,301 instances) but are masked out of supervision. So `log(train count)` is a
valid control for base only, and NN similarity is used for novel — as specified.

## Analysis 1 — error destination

Every GT instance whose tracklet pair the model produced, matched at vIoU ≥ 0.5,
with the top-1 predicted predicate over the overlapping segments.

| | 'all' pass | 'novel' pass |
|---|---|---|
| GT instances with a matched pair | 1170 | 133 |
| top-1 correct | 404 (34.5%) | 55 (41.4%) |
| errors | 766 | 78 |
| — dynamic-GT | 280 | 42 |
| — static-GT | 486 | 36 |

**Raw leakage looks exactly like motion blindness:**

| | 'all' | 'novel' |
|---|---|---|
| dynamic-GT errors → **static** predicate | **60.0%** | **57.1%** |
| static-GT errors → **dynamic** predicate | **13.6%** | **8.3%** |
| asymmetry | **+46.4 pts** | **+48.8 pts** |

**Against the nulls it inverts:**

| null for "lands on static" | 'all' | 'novel' |
|---|---|---|
| uniform over 132 labels | 36.4% | 36.4% |
| **the model's own top-1 distribution** | **70.3%** | **65.2%** |
| GT-frequency weighted | 52.7% | 45.3% |
| **observed / model null** | **0.85×** | **0.88×** |

Dynamic-GT errors land on static predicates *less* often than the model's own
baseline rate of predicting static. Conditioned on that baseline, errors in both
groups stay **within** their own group more than chance — the granularity
signature. The +46 point raw asymmetry is the static prior, not a boundary
effect.

**A problem with my own pre-registration.** I named the model's prediction
distribution as decisive without noticing that it is itself the effect under
test: the model predicts static 70.3% of the time against a ground truth that is
52.7% static — a **1.33× over-prediction of static labels**. If the model cannot
see motion, defaulting to static is precisely the expected behaviour, so
conditioning on that distribution may normalise away the signal rather than
control a nuisance. I record this rather than switching to whichever null gives a
cleaner story.

**The sub-check survives that critique.** Among dyn→static errors, how often is
the prediction the *static counterpart* — same spatial term, motion component
dropped (`walk_away` → `stand_away`)?

| | observed | chance among static labels | ratio |
|---|---|---|---|
| 'all' | **32.7%** (55/168) | 7.6% | **4.3×** |
| 'novel' | 4.2% (1/24) | 1.3% | *n = 1, ignore* |

Granularity predicts a dynamic label losing to a *similar* label; it does not
predict that, having crossed into the static group, the winner preserves the
exact spatial configuration. **The rate of crossing is unremarkable; the
structure of the crossings is not.** Geometry read, motion dropped.

## Analysis 2 — granularity regression

**BASE split**, control `log(train instance count)`, 71 predicates, 15 at AP = 0:

| model | R² | group partial R² | group coefficient (dynamic) |
|---|---|---|---|
| AP ~ log(count) | 0.237 | | |
| AP ~ log(count) + group | 0.239 | **0.002** | **−1.96 ± 9.55** (includes 0) |
| *same, zeros dropped (n=56)* | 0.072 | 0.001 | −1.38 ± 10.77 |

`log(train count)` coefficient **+6.14 ± 3.17** — excludes zero. Supervision
explains ~24% of the variance in per-predicate AP; **group membership adds
0.2%.** On base, "dynamic" carries no information once supervision is
controlled. Robust to dropping the zeros.

**NOVEL split**, control nearest-neighbour text similarity (§B.11), 61
predicates, 36 at AP = 0:

| model | R² | group partial R² | group coefficient |
|---|---|---|---|
| AP ~ NN similarity | 0.044 | | |
| AP ~ NN similarity + group | 0.088 | 0.046 | **−8.36 ± 9.77** (includes 0) |
| *same, zeros dropped (n=25)* | 0.073 | 0.038 | −8.11 ± 17.09 |

A substantial point estimate — 8.4 mAP points — but the interval spans zero on a
sample where 36 of 61 predicates sit at the floor. Underpowered.

**Third refutation of the §B.11 line:** NN text similarity explains **R² = 0.044**
of per-predicate AP, with a coefficient of −213 ± 303. Text separability
predicted neither where errors land (Phase 2.3) nor which predicates score badly.

**The novel split cannot answer this question, structurally.** No novel predicate
is supervised, so "granularity" there is not a data-quantity effect at all — it
reduces to competitor count, and the novel output space holds 41 dynamic against
20 static labels. Competitor count is therefore **perfectly collinear with group
membership**, and no regression on this data can separate them. NN similarity was
the wrong proxy, and no better one exists within this design.

## Verdict

| decision-rule condition | result |
|---|---|
| asymmetric leakage into static | **raw yes (+46 pts), but not against the model's own null (0.85×)** |
| group survives the frequency control | **no on base** (partial R² 0.002); **untestable on novel** (collinear) |

**GENUINELY AMBIGUOUS — and leaning granularity on the only split where the
question is answerable.**

Base is well-powered and clean: supervision explains AP, group adds nothing, and
H1′ was not supported there anyway (0.684). Novel is where H1′ held (0.425) and
is exactly where the design cannot separate the hypotheses.

Against that, one piece of evidence resists the granularity reading: the **4.3×
static-counterpart specificity**. Crossing into the static group is not elevated,
but *how* it crosses — preserving the spatial term, dropping the motion — is not
something label crowding predicts.

I am not forcing this either way. Per the decision rule, stating it plainly:
**H1′ is not established as motion blindness, and not dismissed as an artifact.**

## What would resolve it

**A frame-shuffling probe — one GPU pass, no training.** Run the test set with
each video's frames randomly permuted before CLIP encoding, leaving the box
trajectories intact.

| outcome | conclusion |
|---|---|
| dynamic AP **unchanged** | the model never used temporal order — **motion blindness confirmed**, architecture proceeds |
| dynamic AP **drops**, static unchanged | the model does use motion; H1′'s gap is granularity or something else |
| both drop | the probe disturbed appearance too; inconclusive, needs a gentler perturbation |

This is decisive in a way the observational analyses above cannot be, because it
manipulates the variable rather than conditioning on it. It reuses the existing
pipeline — the only change is a permutation in `dataset.__getitem__` — and costs
one ~3 h run at roughly $1.50.

A cheaper partial check, CPU-only on the dumps already here: bin dynamic-GT
instances by actual trajectory displacement and compare AP in the top and bottom
bins. If the model is motion-blind, AP should not vary with how much movement
there actually is. Weaker than the probe, but free.

## Addendum — the free displacement check

Does accuracy on dynamic predicates depend on how much the objects **actually
moved**? Motion blindness predicts *no*: the model would be reading geometry
regardless. Displacement is measured on the ground-truth trajectories (subject
relative to object, normalised by box size), so it is a property of the instance,
not of the model.

'all' pass, 1170 matched GT instances, top-1 accuracy by displacement quartile:

| | Q1 least | Q2 | Q3 | Q4 most | Q4−Q1 |
|---|---|---|---|---|---|
| median displacement | 0.116 | 0.366 | 0.823 | **2.033** | 18× range |
| **dynamic** predicates (n=413) | 34.0% | 24.3% | 33.0% | 37.5% | **+3.5%** |
| *static* predicates, control (n=757) | 37.6% | 31.2% | 32.3% | 42.1% | *+4.5%* |

**Accuracy on dynamic predicates is flat across an 18× range of actual motion.**
The static control is equally flat (+4.5%), so this is not a general "more
displacement is easier" effect that should be subtracted — there is no
displacement effect in either group.

That is what motion blindness predicts, and it is an independent line from the
counterpart specificity.

**But it is weak, and the weakness is quantifiable.** With ~103 instances per
quartile at ~35% accuracy, the 95% interval on a Q4−Q1 difference is about
**±13%**. So this rules out a *large* motion effect and says nothing about a
modest one. Flat here is consistent with motion blindness; it does not establish
it.

## Where the evidence stands

| line | points to |
|---|---|
| raw leakage asymmetry (+46 pts) | motion blindness |
| leakage vs the model's own null (0.85×) | granularity |
| static-counterpart specificity (4.3× chance) | **motion blindness** |
| base regression: group partial R² 0.002 | **granularity** |
| novel regression | untestable (collinear) |
| accuracy flat across 18× displacement | motion blindness (weak) |

Two substantive lines each way, one structurally untestable, and the strongest
single result on each side — the 4.3× counterpart specificity and the base
regression's null group effect — are not in direct contradiction: they concern
different splits and different granularities. **The verdict stands as ambiguous**,
and the frame-shuffling probe is now the clear next step, since it manipulates
temporal order rather than conditioning on it.

---

# H1′ retired

After the two analyses above came out ambiguous, **H1′ was retired as the
motivation**, and the reason matters: on base, the group indicator adds partial
R² **0.002** once `log(train count)` is controlled; on novel, competitor count is
perfectly collinear with group membership, so *no observational analysis on this
data can identify the cause*. Motion blindness is a **different claim** from the
group gap — the model can be motion-blind while the gap itself is
supervision-driven.

A frame-shuffling probe was designed to test motion blindness by intervention.
It was dropped from the plan before it was ever run, and its code has been
removed. The two notes below stand on their own.

## Methodological note — why a probe was necessary

Analysis 1 pre-registered the model's own prediction distribution as the
decisive null. That was a mistake, because **the distribution is itself the
effect under test**: the model predicts static predicates 70.3% of the time
against a ground truth that is 52.7% static, a 1.33× over-prediction. A
motion-blind model would default to static exactly so. Conditioning on that
distribution therefore risks normalising away the signal rather than controlling
a nuisance — and there is no way to tell, from observational data, which it did.

Every statistic in Analysis 1 and 2 conditions on something the hypothesis may
itself have caused. Only an intervention escapes that.

## Also retired: §B.11's mechanism

The nearest-neighbour text-similarity account is now **refuted on three
independent counts** and nothing downstream should cite it:

1. it predicted errors would cluster *within* a verb family — measured 39.7%
   within, 60.3% across;
2. it explains R² = **0.044** of per-predicate AP (coefficient −213 ± 303);
3. its central observation — that compositional names collapse on the spatial
   component — did not predict which predicates score badly.

The measurement itself stands (predicate names really do sit at 0.954 mean NN
similarity against 0.856 for object names). It simply has no demonstrated
consequence for model behaviour.
