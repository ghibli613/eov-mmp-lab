# Open-VidVRD: what separates the five frameworks

Compiled 2026-08-20 from the papers and code in `source/`. Every claim here is
either quoted from a paper or verified in the corresponding repository; where a
number is unverified it says so.

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
          └── EOV-MMP (TPAMI'25) end-to-end
                 │
                 └── METOR (IJCAI'25)  builds on EOV-MMP, same group (BIT)
```

METOR's README: *"we will first prioritize the open-sourcing of our TPAMI paper
EOV-MMP ... The open-sourcing for the current work, which builds upon the former,
will follow afterwards."*

---

## 2. The scoreboard

VidVRD test set, from METOR Table 1 (their numbers for all five).

| Split | Method | mAP | R@50 | R@100 | mAP_o |
|---|---|---|---|---|---|
| **Novel** | RePro (2023) | 5.87 | 12.75 | 16.23 | 10.36 |
| | UASAN (2024) | 9.88 | 12.80 | 17.68 | 12.15 |
| | OV-MMP (2024) | 12.15 | 13.72 | 15.21 | 14.37 |
| | EOV-MMP (2025) | 15.04 | 16.03 | 18.18 | 36.31 |
| | **METOR** (2025) | **16.74** | **16.72** | **19.43** | **38.91** |
| **All** | RePro | 21.12 | 12.63 | 15.42 | 18.18 |
| | UASAN | 22.93 | 15.74 | 18.89 | 23.74 |
| | OV-MMP | 22.10 | 13.26 | 16.08 | 34.61 |
| | EOV-MMP | 26.34 | 16.48 | 19.54 | 52.72 |
| | **METOR** | **27.52** | **16.69** | **19.58** | **55.09** |

`mAP_o` = object-trajectory detection quality. Watch this column: it is where
the largest single jump happens, and it explains the relation gains.

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

**Diagnosis (EOV-MMP abstract):** *"heavy dependence on the pre-trained
trajectory detectors limits their ability to generalize to novel object
categories, leading to performance degradation."*

The deepest critique in the chain. RePro / UASAN / MMP are open-vocabulary in the
*predicate* but closed-set in the *object detector* (VinVL, fixed categories). A
novel object is never detected at all, and no prompting recovers it.

EOV-MMP unifies detection and classification end-to-end:

- query-based Transformer decoder, **CLIP visual encoder distilled** for
  frame-wise open-vocabulary detection
- a **relationship query** embedded in the decoder, plus an auxiliary
  relationship loss, so trajectory detection is relationship-aware
- a trajectory associator

**Evidence it worked:** `mAP_o` 14.37 → 36.31, a 2.5x jump in object detection
quality. The relation gain is a consequence of finally detecting novel objects.

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
| MMP | **CLIP** ViT-B/16 (ViT-L/14@336px in feature extraction) | `clip.load(...)` in `utils/ptm_encoder.py` |
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

## 7. Adjacent work found by search (not in this workspace)

- **OpenVidVRD** (arXiv 2503.09416, Mar 2025) -- prompt-driven semantic space
  alignment, spatiotemporal refiner. **Already references CoCoOp's Meta-Net.**
  Full paper not yet read; overlap needs checking.
- **"Improving Open-vocabulary VidVRD with Decomposed Prompt Learning and
  Relation Adjustment"** (ICASSP'25) -- decomposed prompts based on *"shared
  actional and spatial patterns between base and novel relations"*. Close to
  RePro's compositional axis; overlap needs checking.

Both should be read in full before committing to a direction.

---

## 8. Sources

Papers in this workspace:
- `repro/2302.00268v1.pdf` -- RePro (ICLR 2023)
- `UASAN/3664647.3681061.pdf` -- UASAN (ACM MM 2024)
- `MMP_OV_VidVRD/AAAI_2024_open_vocabulary_video_relationship_detection.pdf`
- `EOV-MMP-VidVRD/2409.12499v2.pdf` -- EOV-MMP (TPAMI 2025)
- `METOR/2505.06663v1.pdf` -- METOR (IJCAI 2025)

External:
- [OpenVidVRD (arXiv 2503.09416)](https://arxiv.org/abs/2503.09416)
- [Decomposed Prompt Learning (ICASSP'25)](https://basurafernando.github.io/papers/2025_ICASSP.pdf)
- [CoCoOp (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)
