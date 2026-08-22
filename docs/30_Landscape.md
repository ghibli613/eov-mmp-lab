# Open-VidVRD: what separates the seven frameworks

Compiled 2026-08-20 from the papers and code in `source/`; extended 2026-08-22
after reading OpenVidVRD and the ICASSP'25 paper in full, which changed the
scoreboard. Every claim here is either quoted from a paper or verified in the
corresponding repository; where a number is unverified it says so.

---

## 1. The lineage

These are not five independent methods. They are one chain, plus a fork at the
backbone.

```
RePro (ICLR'23)  ── the origin, ALPro
   │
   ├── UASAN (ACM MM'24)         ALPro/BERT. README: "contains modified codes
   │                             from OpenVoc-VidVRD" -- a direct fork
   │
   └── MMP / OV-MMP (AAAI'24)    switches to CLIP
          │
          ├── EOV-MMP (TPAMI'25) end-to-end
          │      │
          │      └── METOR (IJCAI'25)  builds on EOV-MMP, same group (BIT)
          │
          ├── OpenVidVRD (arXiv Mar'25)   independent group (SCUT). Region
          │                               captions + spatiotemporal refiner.
          │                               Baselines stop at MMP
          │
          └── Decomposed Prompt (ICASSP'25)  independent group. Predicates
                                             split into actional + spatial
```

The last two are not part of the BIT chain and do not cite EOV-MMP or METOR.
That matters for §2.

METOR's README: *"we will first prioritize the open-sourcing of our TPAMI paper
EOV-MMP ... The open-sourcing for the current work, which builds upon the former,
will follow afterwards."*

---

## 2. The scoreboard, and why it cannot be pooled

VidVRD test set, SGDet / RelDet. The first five rows are METOR Table 1; the last
two are each paper's own table.

| Split | Method | mAP | R@50 | R@100 | mAP_o | Protocol |
|---|---|---|---|---|---|---|
| **Novel** | RePro (ICLR'23) | 5.87 | 12.75 | 16.23 | 10.36 | A |
| | UASAN (MM'24) | 9.88 | 12.80 | 17.68 | 12.15 | A |
| | OV-MMP (AAAI'24) | 12.15 | 13.72 | 15.21 | 14.37 | A |
| | EOV-MMP (TPAMI'25) | 15.04 | 16.03 | 18.18 | 36.31 | A |
| | METOR (IJCAI'25) | 16.74 | 16.72 | 19.43 | 38.91 | A |
| | **OpenVidVRD** (arXiv'25) | **17.95** | 17.69 | 19.01 | — | **A** |
| | Decomposed Prompt (ICASSP'25) | *19.90* | 16.86 | 18.35 | — | **B** |
| **All** | RePro | 21.12 | 12.63 | 15.42 | 18.18 | A |
| | UASAN | 22.93 | 15.74 | 18.89 | 23.74 | A |
| | OV-MMP | 22.10 | 13.26 | 16.08 | 34.61 | A |
| | EOV-MMP | 26.34 | 16.48 | 19.54 | 52.72 | A |
| | METOR | 27.52 | 16.69 | 19.58 | 55.09 | A |
| | **OpenVidVRD** | **28.48** | 17.15 | 19.90 | — | **A** |
| | Decomposed Prompt | 27.87 | 16.53 | 19.54 | — | **B** |

`mAP_o` = object-trajectory detection quality. Watch this column: it is where the
largest single jump happens, and it explains the relation gains.

### The two protocols

Protocol is not stated by any paper; it is inferred from the **baseline rows**,
which is the only cross-check available. Verified by extracting each PDF:

| Baseline | Protocol A (RePro, METOR, OpenVidVRD) | Protocol B (ICASSP'25) |
|---|---|---|
| RePro, novel mAP | 5.87 | 6.10 |
| RePro, all mAP | 21.12 | 21.33 |
| **MMP, novel mAP** | **12.15** | **16.56** |
| MMP, all mAP | 22.10 | 26.80 |

**A 4.4 mAP gap on the same method and the same benchmark.** Under protocol B,
MMP alone (16.56) outscores what METOR reports for EOV-MMP (15.04) — which is a
protocol difference, not a method result.

So the ICASSP'25 headline of 19.90 sits on top of a baseline inflated by ~4.4
relative to everyone else's. It is *not* comparable to the rest of this table and
should never be quoted alongside them without this caveat.

### What is actually state of the art

- **Among peer-reviewed, protocol-A results: METOR**, 16.74 novel / 27.52 all.
- **The highest protocol-A number is OpenVidVRD's**, 17.95 / 28.48. Its baseline
  row matches METOR's exactly, so the comparison is legitimate — but it is an
  **unrefereed arXiv preprint** (IEEE journal format, presumably under review),
  and its own comparison table stops at MMP, so it never claims to beat EOV-MMP
  or METOR. That cross-table comparison is ours, not theirs.
- The ICASSP'25 figure is the largest number printed anywhere and the least
  meaningful of the three.

If you are stating a target to beat: **16.74 novel is the defensible bar, 17.95
is the prudent one.**

---

## 3. What each step actually fixed

### RePro → UASAN — 5.87 → 9.88 (+4.01)

**Diagnosis (UASAN abstract):** *"the correspondence between visual **union
regions** and relation predicates is usually ignored."*

RePro represents a pair by concatenating two *separate* tracklet features
(2048 + 2048 → phi_p). But a predicate like `sit_on` lives in the space between
and around the two objects, not inside either box. UASAN adds:

- a semantic-aware context encoder over object trajectories **+ union regions +
  trajectory motion information**
- a union-relation alignment decoder producing one relation token per union region

**Why this step is the most informative one in the table:** UASAN keeps
ALPro/BERT and stays cascaded. Same backbone, same pipeline. So +4.01 mAP is
*purely representational* -- clean evidence that method still matters
independently of the backbone.

### UASAN → MMP — 9.88 → 12.15 (+2.27)

The backbone switch: **ALPro → CLIP**, plus vision-guided language prompting
(prompts conditioned on visual content) and spatio-temporal visual prompting.

**This number is confounded.** It mixes a stronger, far more widely pretrained
text encoder with a new prompting scheme. No published work separates the two.
See §6.

### MMP → EOV-MMP — 12.15 → 15.04 (+2.89)

The deepest critique in the chain, and the one most often stated imprecisely.
**Corrected 2026-08-22** — an earlier version of this document said RePro, UASAN
and MMP were *"open-vocabulary in the predicate but closed-set in the object
detector ... a novel object is never detected at all"*. That is wrong, and the
paper does not say it.

#### What the predecessors actually did

Verified in their code, not inferred:

| | Localisation + tracking | Object category |
|---|---|---|
| RePro | **closed-set**: VinVL (FasterRCNN) RoI features + Seq-NMS tracking, per its README | **open-vocabulary**: `TrajClsModel_v3` loads text embeddings and carries `num_base` / `num_novel` |
| UASAN | inherits RePro's | inherits RePro's |
| MMP | **closed-set**: consumes pre-computed trajectories | **open-vocabulary**: `ObjectTextEncoder`, `novel_oids`, split-aware scoring |

So all three name objects open-vocabulary. What they inherit from a
closed-vocabulary detector is the **trajectory** — the boxes and the tracking.

#### What EOV actually claims

> *"Such heavy reliance on closed-set trajectory detectors limits their
> generalization capabilities to unseen object categories. Additionally, **the
> domain gap** between the training data of trajectory detectors and that of the
> Open-VidVRD task limits their adaptability **to base categories**. As a result,
> the detected object trajectories are suboptimal, hindering the subsequent
> relationship classification."*

Two failure modes, neither of them "cannot name novel objects":

1. an object the closed-set detector never proposes gets no trajectory at all;
2. trajectory **quality** is poor from domain gap — *including on base
   categories*, which a labelling failure could not explain.

That second point is the tell, and it matches the evidence: `mAP_o` 14.37 → 36.31
is **trajectory quality**, not classification accuracy.

#### The contributions, in the paper's own words

EOV-MMP is a journal extension of the same group's AAAI'24 paper (OV-MMP = MMP),
and it says so, listing precisely what is new. Its three stated contributions:

1. an **end-to-end framework** unifying trajectory detection and relationship
   classification, *"eliminating the need for pre-trained trajectory detectors"*;
2. a **relationship-aware open-vocabulary trajectory detector**, distilling CLIP
   into a query-based Transformer decoder and perceiving relationship context via
   *"a dedicated relationship query and an auxiliary relationship loss"*;
3. an open-vocabulary relationship classifier with **multi-modal prompting** on
   both the visual and language sides.

**Contribution 3 is inherited, not new.** The paper's own "differences from the
preliminary version" names only (1) the end-to-end integration, (2) the
relationship-aware detector, and (3) additional experiments including the
cross-dataset VidOR→VidVRD setting. The multi-modal prompting relationship
classifier is MMP's, carried over — which is why this repository's
`models/relation_classifier.py` is recognisably MMP's `model_stage2.py`
([§7](#7-what-you-can-actually-build-on)).

So: **EOV = MMP's relationship classifier + an open-vocabulary trajectory
detector + end-to-end joint training.** The new contribution is the detector and
the unification; and the detector is exactly the part with no released training
code.

### EOV-MMP → METOR — 15.04 → 16.74 (+1.70)

**Diagnosis:** even end-to-end, EOV-MMP still runs objects → relationships in
sequence, so object errors propagate. METOR Figure 1(c) counts exactly this:
relations classified correctly with GT trajectories but wrongly with detected
ones.

METOR makes it bidirectional:

- **iterative enhancement module** -- objects refine relationships, relationships
  refine objects, alternating
- **contextual refinement encoding (CRE)** -- CLIP-based, refines both text
  features and object queries

Ablations (METOR Tables 3-4):

| Variant | Novel mAP | | Iterations | Novel mAP |
|---|---|---|---|---|
| w/o CRE | 13.49 | | 0 | 15.16 |
| w/o CRQ | 15.32 | | 1 | 16.04 |
| w/o CRT | 14.91 | | **2** | **16.43** |
| **full** | **16.43** | | 3 | 16.25 |

---

## 4. Backbones — the fork in the road

Verified by grep over each repository, not inferred from papers.

| Method | Text encoder | Evidence |
|---|---|---|
| RePro | **ALPro** (BERT tower → 256-d) | `vlm.text_module` = `AlproTextEncoder` |
| UASAN | **ALPro / BERT** (inherits RePro) | `BertConfig.from_pretrained("bert-base-uncased")` in `traj_former_lora.py`, `DistillModels_v1.py` |
| MMP | **CLIP** ViT-B/16 (ViT-L/14@336px in feature extraction) | `clip.load(...)` in MMP's `utils/ptm_encoder.py` |
| EOV-MMP | **CLIP** (+ InternVideo) | `TextEncoder`, `PromptLearner`, `CustomCLIP` in `model/text_encoder.py` |
| METOR | **CLIP** | paper: *"pre-trained vision-language models such as CLIP"* |

```
ALPro branch                CLIP branch
RePro → UASAN               MMP → EOV-MMP → METOR
(5.87 → 9.88)               (12.15 → 15.04 → 16.74)
```

**Consequence:** only UASAN shares RePro's text space. CLIP-branch features and
checkpoints cannot be substituted into a RePro-derived codebase -- different
tokenizer, different embedding dimension, different joint space.

---

## 5. The verified gap: motion

Grep for motion modelling *in the method* (not in the tracker):

| Method | Hits | Where |
|---|---|---|
| RePro | **67** | `ops/motion.py`, `models/` -- motion **selects the prompt** (paper Eq. 8, 6 GIoU groups) |
| MMP | **1** | a CLI flag `--mot_feat`, "use motion location feature or not" -- motion as a concatenated *input feature* |
| EOV-MMP | 50 | **all** in `model/deep_sort/kalman_filter.py` -- constant-velocity model for *tracking association*. `model/text_encoder.py` contains zero motion references |
| METOR | — | not a stated axis; future work is *"audio and 3D information"* |

**Nobody in the CLIP branch conditions the prompt on motion.**

RePro's actual insight was: *"`towards` can be prompted as 'a relation of
[CLASS], moving closer'. In contrast, `eat` and `sit on` as 'a relation of
[CLASS], relative static'"*. The field took the prompting and dropped the motion
conditioning at the CLIP switch.

RePro's own mechanism is crude and unimproved since 2023: **six hand-thresholded
GIoU buckets**, pairwise, ignoring velocity and duration.

---

## 6. Observations worth carrying into a paper

**a. Gains are shrinking.** +4.01, +2.27, +2.89, +1.70. Architectural moves are
getting expensive for diminishing return.

**b. Every step after RePro attacked the *pipeline*,** not the prompt: union
regions → the detector → the cascade. Prompt *conditioning* was last touched in
2023.

**c. The backbone/method confound is unresolved.** The 9.88 → 12.15 step changes
both. Isolating them is a clean, publishable ablation, and it needs exactly the
RePro-derived codebase in this workspace.

**d. base/novel is a SEMANTIC split, not a frequency split.** Measured from
`configs/VidVRD_pred_class_spilt_info_v2.json`:

```
base   71 categories,  5 .. 1504 instances   (rarest: fly_toward=5, fall_off=5)
novel  61 categories,  2 ..  339 instances   (richest: sit_behind=339, sit_next_to=267)
```

Novel categories are *not* the rare ones. So the generalization gap is not a
data-scarcity problem and long-tail methods will not address it.

**e. The base-overfitting problem is stated but unsolved.** RePro's paper:
*"learning the prompt sometimes might break the 'open' knowledge due to
overfitting to the base category training data"* -- exactly CoOp's failure that
CoCoOp / KgCoOp / ProGrad address in the image domain. METOR's numbers show it
persists: novel 16.74 vs all 27.52.

**f. Model selection is on the test set, field-wide.** RePro's `SegmentEvaluater`
and MMP's `train.py` (lines 69-72, 205-206: two test datasets, keep-best-mAP)
both do it. Follow the convention for comparability, but state it.

**g. Evaluation protocol is drifting.** RePro reports SGDet/SGCls/PredCls; METOR
reports **SGDet only**, arguing *"SGCls and PredCls rely on pre-detected
trajectories"*. Pin down the protocol before reporting anything comparative.

---

## 7. What you can actually build on

Verified 2026-08-22 by inspecting each repository in `source/` and each paper's
implementation details. This is the section that decides which baseline to use.

| Method | Code | Checkpoints | Usable as a base here |
|---|---|---|---|
| RePro | full | yes | no — ALPro text space, incompatible with the CLIP branch |
| UASAN | full | partial | no — same ALPro branch |
| MMP | **stage-2 relation training only** (see below) | yes | partly — the only public training loop for anything EOV needs |
| **EOV-MMP** | steps 1-3 unreleased; **step 4 released** | **all four, plus two end-to-end** | **yes — this repository** |
| METOR | **none** | **none** | no |
| OpenVidVRD | none found | none | no |
| Decomposed Prompt | none found | none | no |

### What MMP's training code actually covers

Verified 2026-08-22 by reading the repository; the README's description is more
generous than the code.

`scripts/train.py` is the only file in MMP containing a `backward()`. Mapping it
onto EOV's four steps:

| EOV step | MMP training code? | Evidence |
|---|---|---|
| 1. trajectory detector | **no** | MMP is cascaded. `features.py` *consumes* trajectories — `gen_gt_trajs()` reads `vid_anno["trajectories"]` — and never produces them |
| 2. object classifier (vision-guided prompts) | **no** | MMP's `ObjectTextEncoder` uses fixed hand-written prompts. `CustomCLIP` / `PromptLearner` appears once, on `self.pre_classifier` — the **predicate** classifier. There is no learned object prompt module |
| 3. relationship classifier | **stage 2 only** | see below |
| 4. joint end-to-end | n/a | MMP is not end-to-end; that is what EOV added |

The README describes two stages — stage 1 trains the spatio-temporal vision
module under hand-crafted prompts, stage 2 freezes vision and trains the
vision-guided prompt module. **The shipped script can only run stage 2**, for two
independent reasons:

```python
from model_zoo.model_stage2 import Model      # train.py:36 -- hard-coded
...
elif "featEmbedding" in name:
    param.requires_grad_(False)               # train.py:79 -- unconditional
```

`model_stage1.py` is imported by nothing; the only occurrence of that name
elsewhere is a *checkpoint filename* in `train_vidvrd_openvoc.sh`. And
`featEmbedding` is the spatio-temporal module stage 1 is supposed to train,
frozen on every path. Running stage 1 requires changing both the import and the
freeze block.

**Still the most useful thing available**, because it is a working reference for
training the predicate prompt module on pre-extracted features with ground-truth
trajectories — which is exactly the setting of the geometry-only baseline in
[91_Extension-guide.md](91_Extension-guide.md) §1, and cheap enough for Colab.

One incidental find, stated by neither paper. The four-way text concatenation:

```python
torch.cat([pre_sbj, pre_obj, pre_rel, pre_rel], -1) / 2      # MMP, both stages
torch.cat([sbj,     obj,     uni,     learned],  -1) / 2      # EOV
```

MMP **duplicates** the relation prompt into the third slot. EOV replaced that
duplicate with a dedicated **union-region** prompt and made the fourth slot the
learned one. So the four-way structure predates EOV; EOV's addition to it was the
union prompt. Worth knowing before rebalancing those weights
([91_Extension-guide.md](91_Extension-guide.md) §2).

### Why METOR is not a viable base, despite being the peer-reviewed SOTA

Its repository contains a PDF and a README, and that README still says the
release "will follow afterwards" with no date. Reproducing it from the paper is
not a matter of effort but of feasibility:

- *"We train the entire framework in an **end-to-end manner**"* with **five**
  losses, for **30 epochs** from an MS-COCO-pretrained decoder.
- **The number of object queries is 100.** EOV uses 300. So EOV's released
  detector checkpoint cannot be loaded into a METOR-shaped model at all — the
  architectures differ, not just the weights.

Since EOV's steps 1-2 have no training code either
([01_Architecture.md](01_Architecture.md) §4), reproducing METOR means training
an open-vocabulary detector plus relation model from COCO init, on VidVRD, with
no reference implementation. That is far beyond a Colab budget, and this
workspace cannot yet run EOV's much cheaper 5-epoch step-4 fine-tune.

### And the gain is smaller than the table implies

METOR's own ablation reports **iterations = 0 → 15.16 novel mAP**, against
EOV-MMP's 15.04. METOR *is* EOV plus the iterative enhancement module. So the
route to that +1.6, if it is ever wanted, is to implement that module on top of
the EOV baseline already running here — incrementally and optionally, rather
than as a prerequisite.

Note also that METOR changes the training regime at the same time as adding the
module: 100 queries instead of 300, 30 epochs from COCO, five losses instead of
two. **Its +1.70 is not cleanly attributable to the iterative module**, which is
a confound in their paper and a fair thing to point out.

---

## 8. The two independent 2025 papers

Both were read in full on 2026-08-22. Neither is part of the BIT lineage, and
neither cites EOV-MMP or METOR.

**OpenVidVRD** (arXiv 2503.09416, Mar 2025, SCUT) — prompt-driven semantic space
alignment. Generates **region captions** automatically and encodes them, adds a
spatiotemporal refiner, and uses learnable continuous prompts *plus* learnable
conditional prompts; it discusses CoCoOp's Meta-Net explicitly. It also extracts
motion patterns per trajectory pair and injects them into the spatial
Transformer. Reports SGDet, SGCls and PredCls on VidVRD and VidOR.

**This is the closest published work to the direction in
[90_Research-ideas.md](90_Research-ideas.md), and the strongest protocol-A
number.** Track it: if it is accepted somewhere, the bar moves from 16.74 to
17.95 novel.

**Decomposed Prompt Learning and Relation Adjustment** (ICASSP'25) — decomposes
predicates into **actional + spatial** patterns (`walk_left` → `walk` + `left`)
and shares prompt components between base and novel, then re-ranks candidates by
suppressing implausible component combinations.

Its numbers are protocol B (§2) and not comparable. But the paper is important
for a different reason: **it works precisely because VidVRD's predicates
decompose**, which is the structure
[20_Benchmark-analysis.md](20_Benchmark-analysis.md) measures. They read that as
a method; it also reads as a symptom. Any factorised method must position
against this paper explicitly and early.

---

## 9. Sources

Papers in this workspace:
- `repro/2302.00268v1.pdf` -- RePro (ICLR 2023)
- `UASAN/3664647.3681061.pdf` -- UASAN (ACM MM 2024)
- `MMP_OV_VidVRD/AAAI_2024_open_vocabulary_video_relationship_detection.pdf`
- `EOV-MMP-VidVRD/2409.12499v2.pdf` -- EOV-MMP (TPAMI 2025)
- `METOR/2505.06663v1.pdf` -- METOR (IJCAI 2025)

Read in full 2026-08-22, from the PDFs in `source/_papers/`:
- `_papers/2503.09416.pdf` — [OpenVidVRD](https://arxiv.org/abs/2503.09416)
- `_papers/2025_ICASSP.pdf` — [Decomposed Prompt Learning](https://basurafernando.github.io/papers/2025_ICASSP.pdf)

The protocol-A/B tables in §2 were produced by extracting the results tables
from those PDFs and comparing their shared baseline rows against METOR's.
- [CoCoOp (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)
