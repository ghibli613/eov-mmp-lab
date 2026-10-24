# Paper gaps — where the paper is silent, and what was chosen

Every entry is a place a different faithful reimplementation could legitimately
differ. **A mismatch in results at one of these points is not evidence of a
bug.** Each is marked `PAPER-GAP` at its definition in the source.

Roughly half turned out to be answerable from the authors' two earlier papers.
Those are marked **[LINEAGE]** and the quotations are in
[doc 31](31_Lineage-evidence.md).

Legend: **[LINEAGE]** = settled by MMP/EOV-MMP · **[FIGURE]** = settled by
reading METOR's own figures · **[OPEN]** = still a genuine choice.

The figures are a real source and were read late — see
[doc 33 §1](33_Verification-status.md). Several things listed here as reasoned choices
turned out to be drawn explicitly in Figs. 2–4.

---

## 1. Contextual refinement encoding — Sec. 3.3

| Gap | Choice | Status | Why |
|---|---|---|---|
| Number of context tokens `c_o`, `c_r` | **8** each | [LINEAGE] | `M(C)` is the conditional half of the prompt; MMP and EOV-MMP both use eight conditional tokens. Flags `--n_ctx_obj`, `--n_ctx_rel`. |
| Size of `v_o`, `v_r` (Eq. 9–10) | **8** | [LINEAGE] | `v` is the continuous half; same two sources. Flag `--n_prompt_vec`. |
| Where `[REL]` sits in the Eq. 10 prompt | `end` | **[RESOLVED — Fig. 3]** | Eq. 10 prints `[…; REL]`, and **Figure 3 draws it**: both prompt rows end `… ▣ [OBJ]` and `… ▣ [REL]`. So METOR departs from its predecessors here, which both put `[REL]` at 75%. The default is right; `--rel_token_position three_quarter` is kept only to test the predecessors' variant. |
| Positional embedding on appended context tokens | none | [OPEN] | Standard VPT practice — the tokens are not at a location in the image. |
| Is `C_o` per-frame or video-level in Eq. 9–10? | per-frame for queries, pooled for text | [OPEN] | Eq. 8 says "in each frame"; Eq. 9–10 say nothing. This is the only reading under which both type-check: a text classifier changing every frame could not be compared against a video-level object score in Eq. 4. |
| Query refiner initialisation | zero-init output projection → `Q̂ = Q` | [OPEN] | The decoder is meant to be COCO-pretrained; a random refiner would corrupt its queries on step 0. |

→ [doc 11](11_Contextual-refinement-encoding.md)

---

## 2. Iterative enhancement — Sec. 3.4

| Gap | Choice | Status | Why |
|---|---|---|---|
| What "spatio-temporal Transformer" is | factorised: spatial over 3 roles, then temporal over T | [LINEAGE] | EOV-MMP: *"we decouple the spatio-temporal modeling into separate and successive modules, namely spatial modeling and temporal modeling."* |
| The third role token | `H_CLS`, the whole image | [LINEAGE] | EOV-MMP's role set is subject, object, and *"features representing the background (i.e. the whole image)"*. |
| How 3 roles become one relationship feature | concatenate, then project | [LINEAGE] | EOV-MMP Eq. 24: `ṽ = ψ([v̈_s; v̈_o; v̈_h])`. Pooling would discard which role contributed what. |
| Role / temporal embeddings | role shared across frames; temporal shared across roles | [LINEAGE] | EOV-MMP states both sharings explicitly. |
| `M_f` initialisation | zero-init final linear | [OPEN] | So an untrained layer adds nothing beyond the `α` rescale. |
| Does the blend carry a residual? | **no** — literally Eq. 11 | [OPEN] | `--residual_mapping` is the alternative, labelled a DEVIATION → [doc 32 §6](32_Not-implemented.md). |

→ [doc 13](13_Iterative-enhancement.md)

---

## 3. Framework and scoring — Sec. 3.2, Eq. 4

| Gap | Choice | Status | Why |
|---|---|---|---|
| **How subject–object pairs are formed** | top-10 queries by learned objectness → 90 ordered pairs | **[OPEN]** | The paper says only that pairs are "derived". All ordered pairs of 100 queries is 9,900 per video. **The largest unforced choice here**, and neither predecessor helps — both pair *tracked trajectories*, not queries. → [doc 16 §2](16_Matching-and-pair-formation.md) |
| Temperature in Eq. 4 | 0.01 | [LINEAGE] | METOR names none; without one the sigmoid is nearly linear. EOV-MMP: *"τ in both Eq. 5 and Eq. 10 is set to 0.01."* |
| Decoder depth | 6 layers | [LINEAGE] | EOV-MMP: *"we use six Transformer layers and 300 object queries."* |
| Attention heads / dropout | 8 / 0.1 | [LINEAGE] | MMP states both. |
| `d_model` | 256 | [OPEN] | Deformable-DETR's, which the decoder is said to initialise from. |
| Query ↔ GT matching | Hungarian, DETR cost, L1 averaged over frames with GT | [OPEN] | Never stated. METOR is query-based and initialises from a DETR, which is trained this way. The temporal averaging is forced. → [doc 16 §1](16_Matching-and-pair-formation.md) |
| sigmoid vs softmax per score | sigmoid for predicates, softmax for objects | [OPEN] | Eq. 4 writes `σ` for all three, but predicates are multi-label and objects are not — and the losses (BCE vs CE) confirm the split. |

---

## 4. Losses — Sec. 3.5

| Gap | Choice | Status | Why |
|---|---|---|---|
| Normalisation of Eq. 13 | mean over base predicate categories | [LINEAGE] | **This was a bug.** EOV-MMP Eq. 25 spells out the summation METOR folds into `BCE(·)`. → [doc 15 §2](15_Training-objective.md) |
| Squashing in Eq. 16 | `(cos + 1)/2` before BCE | [OPEN] | A cosine lives in [−1, 1] and BCE on a negative input is undefined. Some mapping is required; the paper states none. |
| Pooling `C` over tokens in Eq. 16 | max over the 8 context tokens | [OPEN] | Any one token may carry the evidence for a category. |
| Are `Ỹ` sets per-frame or per-video? | per-video | [OPEN] | Eq. 16 says "present in each frame", but `C_o`/`C_r` reach the loss video-pooled (§1), so a per-frame target has nothing to attach to. |

---

## 5. How to use this document

If you train the model and a number disagrees with the paper, work down in this
order:

1. **[doc 32 §1](32_Not-implemented.md)** — the COCO decoder init is missing.
   This dominates everything below and is not a "gap", it is an absence.
2. **§3, pair formation** — the largest genuine choice.
3. **§2, residual** — try `--residual_mapping`.
4. Everything else here is small by comparison.

Do **not** start with `--rel_token_position`: Figure 3 settles it
([doc 33 §1](33_Verification-status.md)).
