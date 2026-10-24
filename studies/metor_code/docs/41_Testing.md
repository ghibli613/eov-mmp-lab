# Testing

```bash
python -m pytest tests/          # 137 tests, ~85s, no GPU required
python -m pytest tests/test_metor_model.py -q     # the framework alone, ~5s
```

Everything runs on the development machine, including the whole framework —
because the model tests use a **stub CLIP** rather than ViT-L/14. See §3.

---

## 1. What each file covers

| File | Tests | Covers |
|---|---|---|
| `test_metor_model.py` | 24 | `--arch metor` end to end: Eq. 1–5 shapes, ablation switches, gradient flow, `[REL]` position |
| `test_metor_modules.py` | 19 | Eq. 11's algebra, Eq. 12–16, the heads |
| `test_metor_dataset.py` | 16 | `METORDataset` against real VidVRD annotations |
| `test_metor_config.py` | 19 | config layering, and that the paper's values stay separate from ours |
| `test_compat_equivalence.py` | 5 | `--n_iter 0` ≡ EOV; checkpoint keys survive |
| `test_checkpoint_compat.py` | inherited | the four EOV checkpoints load with 0 missing keys |
| `test_imports.py`, `test_config.py`, `test_feature_streams.py`, `test_inference_postprocess.py` | inherited | the vendored EOV path |

---

## 2. The tests that earn their place

Most tests here pin shapes. These five pin something that could otherwise fail
silently.

### `test_context_tokens_receive_gradient_through_the_frozen_vit`

**The single most valuable test in the suite.** `c_o` and `c_r` are new inputs to
a frozen encoder. Detach them — or append them after the transformer instead of
before — and the model still builds, still returns correct shapes, still trains,
and the contextual refinement learns nothing. The paper's largest ablation gap
becomes unreachable and **no other test notices**.

If you touch `contextual_encoding.py`, watch this one.

### `test_relationship_loss_is_a_mean_over_categories_not_a_double_division`

Pins Eq. 13's normalisation in closed form. It exists because the original
implementation divided by `|C_b_r|` twice, shrinking the predicate term ~100× —
the model would have trained with its main signal missing.
→ [doc 15 §2](15_Training-objective.md)

### `test_paper_literal_update_scales_subject_object_difference_by_alpha`

Checks that `Ô_s − Ô_o` is scaled by exactly `α^N_i`. This is both a correctness
check and documentation of a property of Eq. 11 the paper does not mention.
→ [doc 13 §4](13_Iterative-enhancement.md)

### `test_n_iter_zero_matches_the_eov_baseline_exactly`

Asserts bit-identical outputs between compat-with-`--n_iter 0` and plain EOV.
Without this, a measured delta could be plumbing rather than the module — and
the whole point of `--arch metor-compat` evaporates.

It has already caught one real bug: the compat path was applying the relationship
feature even at `--n_iter 0`.

### `test_wrapping_preserves_every_eov_parameter_name`

If attaching METOR's modules renamed or reshaped one EOV parameter,
`load_state_dict(strict=False)` would report it as "missing" and that layer would
load **randomly initialised** instead of erroring. A model like that trains,
evaluates, and quietly scores nonsense.

---

## 3. Why there is a stub CLIP

[`tests/stub_clip.py`](../tests/stub_clip.py) is a ~60k-parameter CLIP-shaped
stand-in: 32×32 images, 8×8 patches, width 64, two transformer layers. It exposes
exactly the attributes `ContextTokenInjector` and `RefinedTextEncoder` reach for.

The real backbone is ViT-L/14@336px — ~400M parameters over 577 tokens per frame
— and `--cre_mode inject` runs it *inside* the training step with activations
retained for backward. Constructing it and running forward+backward costs several
GB. On the 7 GB development machine that exhausts memory and takes down the WSL
session. → [doc 40](40_Hardware.md)

So the tests get the shape algebra of Sec. 3.2–3.3 verified on any machine in
under a second. The tokenizer used is the **real** CLIP BPE tokenizer, so prompt
assembly is exercised for real; only the weights are toy.

What it does **not** do is validate anything about accuracy. It checks wiring.

---

## 4. Adding a test

Put it with its subject:

| Changing | Test file |
|---|---|
| `metor/modules/*` | `test_metor_modules.py` (no CLIP needed) |
| `metor/model.py` | `test_metor_model.py` (uses `stub_clip`) |
| `data_loading/frame_dataset.py` | `test_metor_dataset.py` (needs the data; skips without) |
| `metor/compat.py` | `test_compat_equivalence.py` |
| `configs/*` | `test_metor_config.py` |

Two conventions worth keeping:

- **Pin values, not just shapes,** wherever the paper states a number. A shape
  test would not have caught the Eq. 13 bug.
- **Say what breaks if the test fails,** in the docstring. Every test above
  explains the failure mode it guards, which is what makes them useful to
  someone who did not write them.

---

## 5. Interpreting a pass

137 passing tests means the code is **mechanically what the equations say**.

It does not mean the equations were read correctly, that the
[doc 30](30_Paper-gaps.md) choices match the authors', or that the model works.
Nothing has been trained. → [doc 33](33_Verification-status.md)
