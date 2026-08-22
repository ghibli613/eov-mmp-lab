# Research directions — Open-VidVRD

Started 2026-08-20; recommendation revised 2026-08-22 after the measurements in
[20_Benchmark-analysis.md](20_Benchmark-analysis.md).

Every measurement here was computed from `data/vidvrd/anno/` in this workspace
and can be re-run. Claims taken from papers are quoted. Where a direction was
considered and set aside, it is kept below with the reason.

**Current recommendation: Direction A.** It replaced the compositional-gap
measurement (now Direction B, §2) as the lead, and it subsumes what was Idea 2.

---

## 1. Direction A — the benchmark measures neither video nor vocabulary  ★ recommended

### The claim

Not "these methods are bad", but: **VidVRD cannot distinguish a model that
understands video relations from one that reads box trajectories**, and it does
not test open vocabulary either. EOV's own design is quiet evidence — it reaches
state of the art using **one frame of appearance per 30-frame segment** plus
box-corner deltas ([02_Code-walkthrough.md](02_Code-walkthrough.md) §3.1).

### The two measurements

Both from [20_Benchmark-analysis.md](20_Benchmark-analysis.md), reproducible
from the annotations:

```
97.0% of test instances are decidable from one frame + box geometry
 3.0% (143 instances) need to see appearance change over time
  71% of test videos contain zero instance of that kind
       -- and mAP is averaged over videos, so for 71% of the metric a perfect
          temporal model and a blind one are indistinguishable

  61 novel predicates
  44 (72.1%) have BOTH verb and spatial component present in base
  61 (100%)  have at least one
   0          require a genuinely unseen concept
```

The second one is the sharper of the pair: `sit_behind` is "novel" while `sit_*`
and `*_behind` are both base. Four years and five papers have optimised prompt
mechanisms for vocabulary generalisation on a split that tests recombination.

The ICASSP'25 paper is unwitting confirmation. It decomposes predicates into
actional + spatial parts, shares components between base and novel, and reports
19.90 novel mAP — the highest figure printed anywhere on this benchmark. They
read that as a method; it also reads as a symptom.

(That 19.90 is **not** comparable to the rest of the scoreboard: their MMP
baseline scores 16.56 where everyone else reports 12.15, a 4.4 mAP offset — see
[30_Landscape.md](30_Landscape.md) §2. The relevant point here is not their
ranking but that decomposition works this well at all.)

### Why this shape can clear a top venue

The precedent is **Neural Motifs** (Zellers et al., CVPR 2018): image scene-graph
generation turned out to be largely predicting the most frequent predicate given
the object pair, and a "frequency baseline" nearly matched the state of the art.
That paper was a diagnosis, and it is heavily cited. This is the same move one
level up — geometry plus a single frame, on a benchmark that additionally claims
*video* and *open vocabulary*.

A finding that contradicts a stated premise of five published papers, with
code-level evidence, is more citable than any +1.7 mAP. It also explains why the
gains are shrinking (+4.01, +2.27, +2.89, +1.70 — [30_Landscape.md](30_Landscape.md) §6a).

### What the paper contains

A pure critique is a risky submission, so the diagnosis has to carry three more
things:

1. **The geometry-only baseline** — the empirical proof. Object-class text
   embeddings plus box trajectories, no CLIP visual stream at all. If it lands
   near EOV's 26.34 / 15.04, the claim is established *without* depending on
   anyone's predicate taxonomy.
2. **Two corrected protocols**, buildable from existing annotations with no new
   labels: a hard-negative protocol forcing a choice among predicates that share
   box geometry (`walk_left` vs `run_left` vs `creep_left`), and a genuine
   vocabulary split holding an entire verb out of base. Then re-evaluate the
   existing methods on both.
3. **A method** — factorised verb x spatial prediction, which handles all 61
   novel predicates zero-shot by construction, plus real temporal aggregation
   within the clip. Its gains should land where the diagnosis predicts and not
   elsewhere; that is far more convincing than winning a column.

Hooks for all of it: [91_Extension-guide.md](91_Extension-guide.md).

### The go/no-go, and what kills it

Run the geometry-only baseline first — a step-3 experiment on ground-truth
trajectories, retrained with `--clip_feat` off. **If ablating appearance costs 4+
mAP broadly rather than concentrating in a small predicate group, the strong form
of the claim is dead** and the fallback is Direction B.

Note this is *not* an end-to-end experiment: `patch_` is the detector's input, so
the end-to-end path cannot run without CLIP encoding at all
([10_Known-issues.md](10_Known-issues.md) §2). Step-3 training code is not in
this repository; MMP's is public and is the intended starting point.

### Risks to plan around

- **Single dataset.** This workspace has VidVRD only — no VidOR frames, and the
  author's checkpoints are VidVRD-only. Single-dataset papers are routinely
  rejected in this area. Direction A degrades gracefully here: the *diagnosis*
  needs VidOR **annotations only**, which are a small download, and VidOR's
  vocabulary has far more interaction verbs, so the contrast should strengthen
  the argument. Only the method half stays VidVRD-only.
- **The taxonomy is a judgement call.** `swim` is filed as single-frame-decidable
  because water is visible in one frame. The geometry-only baseline is what makes
  the argument independent of that.
- **Position against ICASSP'25 early and explicitly.** A careless reviewer will
  read the factorised method as their decomposed prompts. The difference is that
  they propose decomposition as a method, and this shows the benchmark makes
  decomposition sufficient — say so before they wonder.

---

## 2. Direction B — vocabulary and composition are different axes  (fallback)

The former lead. Still real, and the natural fallback if the geometry-only
baseline comes in high.

**Upgrade it from the original framing:** do not report "unseen triplets" as one
bucket. Report the full 2x2 — seen/novel *category* crossed with seen/unseen
*composition* — and hypothesise the mechanism: prompt tuning buys category
generalisation by binding predicate meaning to base subject-object contexts, so
the two axes trade off. If open-vocabulary methods get measurably *worse* at
composition as they get better at novel categories, that is a finding with a
cause.

Two caveats to state up front. **Zero-shot triplet recall (zsR@K) is already
standard in image SGG** — cite it and claim video + open-vocabulary + the
interaction, not the setting. And the measured tail is thin: 258 triplets / 432
instances, which reviewers will question.

The original measurement follows, unchanged.

### The original measurement (2026-08-20)

##### The measurement

Computed from the annotations (train 800 / test 200 videos):

```
unique triplets            train = 2961    test = 1011
test triplets NEVER seen in train        = 258   (25.5% of test triplets)
their share of test instances            = 432/4835 = 8.9%
of those, all three components seen individually = 258  (100%)
```

Every unseen test triplet is a **pure composition failure**: the subject, the
predicate and the object each appear in training, just never together.

Split by predicate type:

```
unseen composition, predicate is NOVEL :  74 triplets
unseen composition, predicate is BASE  : 184 triplets   <-- the interesting half
                                          318 test instances
```

Examples where the model saw the predicate *and* both objects during training,
but never that combination:

```
<lizard     lie_behind    lizard   >   12 instances   [base predicate]
<lizard     creep_left    lizard   >   10 instances   [base]
<zebra      chase         zebra    >    5 instances   [base]
<monkey     creep_beneath monkey   >    4 instances   [base]
<red_panda  jump_left     red_panda>    4 instances   [base]
<dog        watch         monkey   >    4 instances   [base]
```

#### Why this is a contribution

1. **Orthogonal to open-vocabulary.** These are not novel categories. Failure
   here is compositional, not vocabulary-driven. The entire field reports
   base/novel splits and **has never separated "novel category" from "novel
   composition"**.

2. **The field's founding paper names this problem and then doesn't test it.**
   RePro's abstract: *"conventional prompt tuning is easily biased to certain
   **subject-object combinations** and motion patterns."* Compositional bias is
   the stated motivation; the benchmark split does not measure it.

3. **New evaluation axis + diagnosis + method** is the shape of a top-venue
   paper, rather than a sixth prompting mechanism.

4. **Cheap to falsify.** Re-split the existing test set into
   composition-seen vs composition-unseen, run any existing method, report the
   gap. If the gap is large there is a paper; if small, that is known in days.

#### Next steps if pursued

- build the split for VidVRD; repeat on VidOR (10k videos, richer combinations)
- confirm the gap is not explained by frequency alone (see §Idea 3 caveat)
- check the image-SGG literature for "zero-shot triplet" precedent and cite it
  honestly — the setting exists for images; the contribution is video +
  open-vocabulary

---

---

## 3. Ranking and calibration — necessary, not sufficient

Cheap wins exist and are documented as hooks in
[91_Extension-guide.md](91_Extension-guide.md) §2-4: rebalancing the four-way
text ensemble, calibrating the object-score product that demotes novel objects,
and repairing the association step.

**None of them is a contribution.** They are known techniques transplanted into
this task — F-VLM-style score ensembling, CuPL descriptors, KgCoOp regularisation
— and a reviewer will say so correctly. They belong in the paper as the strong
baseline you had to build, and in the ablation table.

They are still worth doing, for two reasons: the headline number has to stay
competitive while the argument does the work, and §6 of the extension guide
(the pseudo-novel validation split) has to be in place before any of them is
tuned, or every gain is fitted to the test set.

---

## 4. Directions considered and set aside

### Prompting, as an axis — crowded

The first idea considered was **motion-conditioned prompt tuning on CLIP** — on
the basis that RePro (ICLR'23) conditions prompts on motion and no CLIP-branch
method appeared to. Reading the two adjacent papers in full mostly closed it.

**OpenVidVRD** (arXiv 2503.09416, Mar 2025) already:

- extracts motion: *"for every subject-object trajectory pair we extract **motion
  patterns m_s,o encoding their relative positional dynamics and motion trends
  across timestamps**"*, aggregated as `M_s,o = phi_mot({m_s,o})`
- injects it into the spatial Transformer: `Ṽ_k = STran(f̃_vt + R_k + P_k + M_s,o)`
- uses **conditional prompts**: *"learnable continuous prompts ... and learnable
  conditional prompts (dynamically adapting to visual cues)"*
- explicitly discusses CoCoOp's Meta-Net and its limitations

**ICASSP'25 "Decomposed Prompt Learning and Relation Adjustment"** decomposes
predicates into **actional + spatial** patterns (`walk_left` -> action `walk` +
spatial `left`) and shares prompt components between base and novel. This is
*linguistic* decomposition of the label, not trajectory motion — so it does not
collide with motion directly, but it does claim the base→novel prompt
generalisation angle.

**Residual distinction, judged too thin to headline:** OpenVidVRD adds motion as
a *feature in the visual stream*; RePro's motion *selects which prompt bank* is
used. Real, but likely to read as incremental to a reviewer.

**Conclusion: prompting is crowded.** RePro, MMP, EOV-MMP, OpenVidVRD and the
ICASSP paper are all in it. A new prompting mechanism needs to clear five
existing ones.

---

### Fixed temporal granularity — subsumed

Every method inherits VidVRD-II's 2017 choice: **30-frame segments, stride 15**.
Verified across the codebases in this workspace (RePro `SEG_LEN, SEG_STRIDE =
30, 15`; MMP feature dirs named `train_gt_30`, `test_gt_30`).

But predicates have very different natural durations — `walk_past` is transient,
`stand_behind` persists. One granularity for all of them is an untested
assumption. Adaptive or multi-scale temporal windows appear in none of the five.

Cheap first experiment: measure the duration distribution of each predicate in
the annotations, and check how many relation instances are shorter than one
segment or span many.

---

This is now part of Direction A rather than a direction of its own: the
30-frame/stride-15 inheritance is one symptom of the same temporal blindness, and
`--frame_stride` exposes the sampling question directly.

---

### A single-protocol re-evaluation of all five

Same method, same benchmark, different papers:

| Method | novel mAP per METOR | novel mAP per ICASSP'25 |
|---|---|---|
| RePro | 5.87 | 6.10 |
| MMP / OV-MMP | **12.15** | **16.56** |

A 4.4 mAP discrepancy for MMP — larger than most papers' entire claimed gain.
ICASSP's own result (19.90) would top METOR's 16.74 if the tables were
comparable; they are not.

Supporting evidence of protocol instability, all verified:

- **Model selection on the test set, field-wide.** RePro's `SegmentEvaluater`
  and MMP's `scripts/train.py` (two test datasets at lines 69-72; keep-best-mAP
  at 205-206).
- **METOR reports SGDet only**, arguing *"SGCls and PredCls rely on pre-detected
  trajectories"*. RePro reports all three.
- **MMP evaluates against both `meta` (detected) and `gt` trajectories**;
  RePro's in-training evaluator uses the detected path only.
- **A subset ground-truth file sat at the canonical path in this very repo**
  (20 videos where 200 were expected) until it was found and fixed on
  2026-08-20 — an illustration of how silently these numbers can drift.

A single-protocol re-evaluation of all five would be a benchmark/analysis
contribution. Lower ceiling than a method paper, higher citation potential, and
this workspace is unusually well equipped for it (four codebases, a verified
refactor, an equivalence harness).

**Caveat:** requires all five to actually run, which is a large engineering
commitment. Best treated as the analysis section of Idea 1 rather than the whole
paper.

---

---

### Data efficiency of open-vocabulary transfer

VidVRD trains on 800 videos. Nobody reports how performance scales with the
amount of base supervision. If open-vocabulary transfer largely holds at 25% of
the base data, that is a useful, cheap result and a natural ablation to attach
to a larger paper.

---

---

### Other observations that could seed something

- **base/novel is a SEMANTIC split, not a frequency split.** Measured:
  base 71 categories spanning 5..1504 instances; novel 61 spanning 2..339.
  Novel categories are *not* the rare ones (`sit_behind`=339 is novel;
  `fly_toward`=5 is base). So long-tail methods will not close this gap — worth
  stating explicitly, since it is easy to assume otherwise.
- **The association stage is crude and unexamined.** Cascaded methods stitch
  per-segment predictions with a greedy algorithm and a hand-set
  `linkage_threshold = 0.5`. Errors introduced there are never isolated.
- **METOR's stated future work is "audio and 3D information"** — flagged so it
  is not mistaken for open ground.

---

---

## 5. What would change the recommendation

- the geometry-only baseline landing far below EOV -> fall back to Direction B
- VidOR checkpoints arriving from the author -> both directions get much stronger
- someone publishing the temporal-degeneracy analysis first -> Direction B, and
  cite them
- **OpenVidVRD being accepted somewhere** -> the bar moves from 16.74 to 17.95
  novel mAP, and the nearest competitor becomes a paper that already does motion
  features and conditional prompts. It is the one to watch
  ([30_Landscape.md](30_Landscape.md) §8)

---

## Sources

Local: `repro/2302.00268v1.pdf`, `UASAN/3664647.3681061.pdf`,
`MMP_OV_VidVRD/AAAI_2024_...pdf`, `EOV-MMP-VidVRD/2409.12499v2.pdf`,
`METOR/2505.06663v1.pdf`, `_papers/2503.09416.pdf` (OpenVidVRD),
`_papers/2025_ICASSP.pdf` (Decomposed Prompt Learning).

External: [CoCoOp, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)
