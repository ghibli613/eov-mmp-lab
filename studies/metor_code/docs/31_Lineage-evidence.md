# Lineage evidence — what MMP and EOV-MMP state that METOR omits

METOR is the third paper in one group's line:

```
MMP (AAAI'24)  →  EOV-MMP (TPAMI'25)  →  METOR (IJCAI'25)
Wang, Wu, Yang    Wang, Wu, Yang, Luo    Wang, Wu, Yang
```

Same first author, same lab (BIT / Shenzhen MSU-BIT). The first two are public,
code and all. A nine-page IJCAI paper has no room to restate implementation
details its predecessors already published — so when METOR is silent, the right
move is to check whether its ancestors said it.

They usually did, and consistently with each other. This document is the
evidence, quoted.

**Sources**
- MMP — `../ov-vidvrd-lab/_paper/AAAI_2024_open_vocabulary_video_relationship_detection.pdf`
- EOV-MMP — `../ov-vidvrd-lab/_paper/2409.12499v2.pdf` (arXiv 2409.12499v2)

---

## 1. The quotations

### MMP, Implementation Details

> "For all experiments, video frames are sampled every 30 frames. We adopt the
> ViT-B/16 version of CLIP while keeping the parameters fixed. The number of
> Transformer blocks of spatio-temporal visual prompting is set to 1 and 2 for
> the VidVRD dataset and the VidOR dataset, respectively. **The head number of
> multi-head self-attention of Transformer blocks is set to 8, and the dropout
> rate is set to 0.1.** For language prompting, **we set the number of tokens for
> both learnable continuous prompts and conditional prompts to 8. The [CLS]
> token is positioned at 75% of the token length.** For optimization, we use the
> AdamW algorithm with an initial learning rate of 0.001. A multi-step decay
> schedule is applied at epochs 15, 20, and 25, reducing the learning rate by a
> factor of 0.1 each time."

### EOV-MMP, Implementation Details

> "For all experiments, video frames are sampled every 30 frames. We adopt the
> ViT-L/14 version of CLIP with fixed parameters. **For the query-based
> Transformer decoder, we use six Transformer layers and 300 object queries. The
> temperature parameter τ in both Eq. 5 and Eq. 10 is set to 0.01.** The
> threshold ϵ used to filter the bounding boxes is set to 0.35. For the auxiliary
> object classifier, **we use eight tokens each for learnable continuous prompts
> and learnable conditional prompts, positioning the object token [OBJ] at the
> end of the sequence.** ... The number of Transformer blocks for spatio-temporal
> visual prompting is set to one for VidVRD and two for VidOR. For the
> open-vocabulary relationship classifier, we use eight tokens each for learnable
> continuous prompts and learnable conditional prompts, **positioning the
> relationship token [REL] at 75% of the token length. The coefficients γ and δ
> in Eq. 23 are set to 0.2 and 0.1, respectively.**"

### EOV-MMP, spatio-temporal visual prompting

> "We use standard Transformer blocks to model the spatio-temporal relationships
> between objects. **To reduce the computational complexity, we decouple the
> spatio-temporal modeling into separate and successive modules, namely spatial
> modeling and temporal modeling.**"

> "spatial modeling requires combining three key elements: features of the
> subject region, features of the object region, and **features representing the
> background (i.e. the whole image)**."

> "we add two types of learnable embeddings: positional embedding ϱ_k related to
> the normalized bounding box, and **role embedding ρ_k. These two types of
> embeddings are learned and shared across all video frames.**"

> "We then add **temporal embedding θ_t, which is related to frame t and shared
> across all roles.**"

### EOV-MMP, the relationship score and loss

> "ṽ = ψ([v̈_s; v̈_o; v̈_h]), ψ(·) denotes the **relationship mapping layer**"
> — its Eq. 24

> "L_rel = **(1/|C_R_b|) · Σ_{r ∈ C_R_b} BCE(ŷ_rel_r, y_rel_r)**"
> — its Eq. 25

### EOV-MMP, initialisation

> "In step one, we **initialize the query-based Transformer decoder, object
> embedding head, and bounding box regression head with pre-trained parameters
> from MS-COCO (excluding novel object categories in Open-VidVRD)**, while the
> relationship classification head is initialized with random parameters."

---

## 2. What each settles

| Setting | MMP | EOV-MMP | METOR | Used here |
|---|---|---|---|---|
| Attention heads | 8 | — | silent | **8** |
| Dropout | 0.1 | — | silent | **0.1** |
| Continuous prompt tokens | 8 | 8 | silent | `--n_prompt_vec 8` |
| Conditional prompt tokens | 8 | 8 | silent | `--n_ctx_obj/rel 8` |
| Decoder depth | — | 6 layers | silent | **6** |
| Object queries | — | 300 | **100** | 100 (METOR wins) |
| Temperature τ | — | 0.01 | silent | **0.01** |
| Box threshold | — | 0.35 | 0.35 | 0.35 (agree) |
| `[OBJ]` position | — | end | silent | end |
| `[REL]` position | 75% | 75% | Eq. 10 says end | flag, default `end` |
| ST design | decoupled | decoupled | "spatio-temporal Transformer" | factorised |
| Role set | s, o, background | s, o, background | `[O_s; O_o; H_CLS]` | same |
| Roles → relation | — | concat + mapping layer | silent | concat + linear |
| Loss coefficients | — | γ=0.2, δ=0.1 | θ_ctx=0.2, θ_cst=0.1 | **same numbers** |
| ST blocks / iterations | 1 VidVRD, 2 VidOR | 1 / 2 | `N_i` = 2 / 3 | +1 for VidOR, consistently |

That last row is a small but telling pattern: all three papers give VidOR one
more block than VidVRD.

And the loss-coefficient row is the strongest single piece of evidence that the
author reuses settings across papers: EOV-MMP's γ and δ are 0.2 and 0.1, and
METOR's θ_ctx and θ_cst are 0.2 and 0.1.

---

## 3. What this actually changed in the code

Three things, two of them substantive.

### 3.1 A real bug in Eq. 13 — fixed

METOR's Eq. 13 writes `L_rel-ctr = (1/|C^b_r|) BCE(S_r, r̃)`, folding the
summation into `BCE(·)`. EOV-MMP's Eq. 25 writes the same loss out in full —
`(1/|C_R_b|) · Σ_r BCE(·)` — which makes it unambiguous that the factor is a
**mean over categories**.

The first version here applied `reduction="mean"` *and* divided by `n_base_rel`
again, shrinking the predicate term by ~100× on VidVRD. The model would have
trained, the loss would have decreased, and the main signal would have been
missing. Fixed and pinned by a test.
→ [doc 15 §2](15_Training-objective.md)

### 3.2 Context tokens 4 → 8

Was a guess. Both predecessors say eight conditional prompt tokens, and `M(C)`
is the conditional half of METOR's prompt, so 8 is the evidence-backed value.

### 3.3 `--rel_token_position` added — and then settled against the lineage

New flag. METOR's Eq. 10 prints `[REL]` last; both predecessors put it at 75% of
the token length while keeping `[OBJ]` at the end.

**This is the one place where the lineage turned out to be wrong for METOR.**
Figure 3 draws the prompt ending `… [REL]`, agreeing with Eq. 10. So METOR
changed the convention, the default `end` is correct, and `three_quarter` is
retained only to test the predecessors' variant.
→ [doc 33 §1](33_Verification-status.md)

A useful reminder that lineage evidence is a prior, not proof.

### And several guesses became evidence

The factorised spatio-temporal design, the three-role set including the whole
image, concat-then-project, and the role/temporal embedding sharing were all
marked PAPER-GAP on reasoning alone. EOV-MMP states every one of them. They are
now [LINEAGE] in [doc 30](30_Paper-gaps.md).

---

## 4. Where the lineage does *not* help

**Pair formation.** MMP and EOV-MMP both obtain trajectories from a *tracker*
first, then pair the tracked trajectories. There are no object *queries* to pair.
So the question METOR raises — which of 100 queries get paired — is genuinely new
with METOR, and genuinely unanswered.
→ [doc 16 §2](16_Matching-and-pair-formation.md)

**Query↔GT matching.** Same reason: no queries in the predecessors.

**The five-loss objective.** EOV-MMP has three losses on the relation side
(`L_rel + γL_obj + δL_int`); METOR has five and drops the interactiveness term.
The mapping is suggestive but not stated.
