# Verification status

What has actually been run, what has not, and what the paper itself leaves
unresolved.

---

## 1. Verified — run, not reasoned about

`python -m pytest tests/` → **137 passed**, ~85 s, no GPU required.

### The framework

- Constructs and completes a **full forward and backward** with the shapes
  Sec. 3 specifies, for `--arch metor` (against a stub CLIP) and for the real
  dataset item pipeline.
- Every output of Eq. 1–5 has the documented shape, for `T` = 1, 2, 4 and 7.
- Text banks come out **L2-normalised**, so the Eq. 4 temperature means the same
  thing for every class.
- All scores and regressed boxes lie in [0, 1].

### The load-bearing property

- **Gradients reach `c_o` and `c_r` through the frozen ViT.** This is the one
  place where a plausible implementation can silently learn nothing: detach
  those tokens and the model still builds, still runs, still trains, and the
  contextual refinement — the paper's largest ablation — simply never happens.
  Nothing else in the suite would notice.
  (`test_context_tokens_receive_gradient_through_the_frozen_vit`)
- The CLIP backbone stays frozen, as Sec. 4.2 requires.
- `v_o`/`v_r` and `M_o`/`M_r` receive gradient.

### The algebra

- **Eq. 11 in closed form**: the subject–object difference is scaled by exactly
  `α^N_i`, checked for `N_i` = 1, 2, 3.
- Zero-initialised `M_f` makes one layer a pure `α` rescale.
- **Eq. 12's weighting** recomputes by hand to the reported total.
- **Eq. 13's normalisation** produces the exact expected value — the test that
  caught the double-division bug.
- `L_cst` is exactly zero for a motionless trajectory (Eq. 15's sign convention).
- Unmatched object labels contribute exactly zero, not NaN.

### The data

- `METORDataset` reads real VidVRD annotations correctly: key-frame grid matches
  the EOV loader's, 1-indexed files vs 0-indexed trajectory keys handled, boxes
  normalised `cxcywh`, base/novel splits disjoint, pair labels in row space.

### Against the paper's own figures

Figures 2–4 are architecture diagrams carrying an explicit ❄ fixed / 🔥 learnable
legend, so they can be checked against the implementation directly. They were
read late — PDF text extraction returns figure labels as scattered fragments
(`ctx obj− / ctr obj−`), which is enough to guess structure from and not enough
to verify one.

**Every structural choice they show was already implemented that way**, so no
code changed:

| Drawn | Confirmed |
|---|---|
| token order `[h_CLS; h_patch; c_o; c_r]`, outputs in the same positions | [doc 11 ④⑥](11_Contextual-refinement-encoding.md) |
| Linear Projection Layer ❄, CLIP ViT ❄, CLIP Text Encoder ❄ | the frozen split |
| ⊕ on Eq. 8 | the residual |
| **one** Relationship Feature Mapping Layer feeding **both** ⊕ | [doc 13 §3](13_Iterative-enhancement.md) — the `α^N_i` property is intended, not a misreading |
| `R̂` branching off *before* `M_f` | what gets scored is the raw transformer output |
| `T_o`/`T_r` each with two outgoing arrows | the same text bank serves scoring *and* Eq. 16 |
| `H_CLS` routed around the decoder | [doc 10 §2](10_Pipeline-overview.md) |

**One documentation ambiguity closed:** Figure 3 draws both prompt rows ending
`… [OBJ]` and `… [REL]`, so the category token is last. That contradicts both
predecessors, which put `[REL]` at 75%. `--rel_token_position end` is correct.
→ [doc 11 ⑨](11_Contextual-refinement-encoding.md)

**What the figures do not touch** are the gaps that matter most: pair formation,
the matching cost, the COCO decoder initialisation. Architecture diagrams do not
describe training.

### Compatibility

- `--arch metor-compat` with `--n_iter 0` is **bit-identical** to the EOV
  baseline.
- Wrapping preserves every released-checkpoint parameter name and shape.
- Inherited: all four EOV checkpoints load with 0 missing / 0 unexpected keys.

→ [doc 41](41_Testing.md) for which test covers what.

---

## 2. Not verified

**Nothing here has been trained.** Not once, not partially, not on a subset.

So none of the following is known:

- whether it converges
- whether the losses balance sensibly at the paper's θ weights
- whether the pair-selection rule supplies enough positives
- whether the Hungarian matcher is stable across epochs
- what mAP it reaches

And it should not be expected to reach the paper's numbers as it stands, because
the COCO decoder initialisation is missing
([doc 32 §1](32_Not-implemented.md)).

A forward pass with the real ViT-L was attempted on the development machine and
exhausted its 7 GB of RAM. That is why the tests use a stub CLIP.
→ [doc 40](40_Hardware.md)

### What "137 tests pass" does and does not mean

It means the code is **what the equations say**, mechanically. It does not mean
the equations were read correctly, that the [doc 30](30_Paper-gaps.md) choices
match the authors', or that the model is any good.

---

## 3. A caveat about the paper's own claim

Worth stating plainly, because it affects what reproducing METOR is worth.

METOR's headline is **15.04 → 16.74 novel mAP (+1.70)** over EOV-MMP. But its own
Table 4 reports **iterations = 0 → 15.16**, against EOV-MMP's 15.04. So the
iterative enhancement module accounts for roughly **+1.3** of the +1.70 — and
only if everything else were held equal.

It is not. METOR simultaneously changes:

| | EOV-MMP | METOR |
|---|---|---|
| object queries | 300 | 100 |
| training | 4-step scheme | 30 epochs end-to-end from COCO |
| losses | 2 | 5 |
| trajectories | tracker | regressed (Eq. 5) |

**So the +1.70 is not cleanly attributable to the paper's stated contribution.**
That is a confound in the paper, not in this reconstruction.

It is also the reason `--arch metor-compat` exists: holding the entire EOV stack
fixed and switching only `--n_iter` measures the module's contribution in a way
the original paper does not. → [doc 13 §6](13_Iterative-enhancement.md)

### And there is a second, related property

Eq. 11 gives both object features the *same* additive term, so the module
provably cannot change how a subject differs from its object — only translate the
pair together while shrinking their difference by `α^N_i`
([doc 13 §4](13_Iterative-enhancement.md)). Nothing in the paper's ablations
separates "mutual enhancement" from "a smoothing prior over the pair", because
varying `N_i` varies both at once.

### Cross-paper comparisons are unreliable here

From `../ov-vidvrd-lab/docs/30_Landscape.md`: MMP alone reports **16.56** novel
mAP under its own protocol — higher than the 15.04 METOR attributes to EOV-MMP.
Rows from different papers in this literature are not reliably comparable, and
protocol differences are not always stated.

---

## 4. If you set out to reproduce this

An honest sequence:

1. Do [doc 32 §1](32_Not-implemented.md) — wire the COCO decoder init.
   Everything before this is uninformative.
2. Do [doc 32 §4 and §5](32_Not-implemented.md) — frame cap and resume, or
   training will not finish.
3. Establish the `--arch metor-compat` baseline first. It uses released
   checkpoints, runs on 16 GB, and gives a *controlled* delta.
4. Only then train `--arch metor`, and report it against the compat delta
   rather than against the paper's table alone.
5. Whatever you report, state the [doc 30](30_Paper-gaps.md) choices you used —
   particularly pair formation.

Also worth stating in any write-up: **model selection on this benchmark uses the
test set.** RePro, MMP and EOV-MMP all evaluate after every epoch and keep the
best. Following the convention keeps numbers comparable, but it should be said
out loud.
