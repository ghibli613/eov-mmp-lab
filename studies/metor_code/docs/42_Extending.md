# Extending — where each change hooks in

Ordered by evidence-per-hour, not by ambition.

---

## 1. Read this before restructuring anything

Two pieces of the layout look like duplication and are not. Both questions have
already been worked through:

### The training scripts *were* merged

`python -m cli.train` is the only entry point; it dispatches on `--arch` at
[cli/train.py:60](../cli/train.py#L60). The two loop **bodies** stay separate —
[`metor/engine.py`](../metor/engine.py) for METOR, the rest of `cli/train.py` for
EOV-MMP.

They are genuinely different procedures: METOR does one forward, one loss, one
backward per video; EOV's `End2End_Model` calls `.backward()` *inside its own
forward*, per frame and per object pair. Folding them together would give one
function with two disjoint halves.

### The datasets were *not* merged

`feature_dataset.py` returns CLIP **features**; `frame_dataset.py` returns
**pixels and box targets**. Their outputs overlap on `video_name` and
`image_size` and nothing else. Merging gives one class returning two incompatible
dicts based on a flag. → [doc 22 §4](22_Data-pipeline.md)

---

## 2. The changes worth making, in order

### 2.1 Wire the COCO decoder initialisation — the prerequisite

**Hook:** `decoder=` on `METOR.__init__`
([model.py:108](../metor/model.py#L108)). It exists for this.

**What:** build the vendored
[`ov_prompt`](../models/methods/detectors/ov_prompt/) Deformable DETR with 100
queries, load COCO weights with novel categories removed, and feed it `Q̂` from
Eq. 8 in place of `self.query_embed.weight`.

**Why first:** every number before this is uninformative.
→ [doc 32 §1](32_Not-implemented.md)

### 2.2 Cap frames per step — required to finish an epoch

**Hook:** `METOR.forward` ([model.py:186](../metor/model.py#L186)), or
`METORDataset.__getitem__`.

**What:** either sample at most `k` key frames per step, or chunk the ViT
encoding with `torch.utils.checkpoint`.

**Why:** ~10% of VidVRD videos exceed any single GPU (max 184 key frames).
Training currently dies partway through an epoch.
→ [doc 32 §4](32_Not-implemented.md)

### 2.3 Add `--resume` — required to finish a run

**Hook:** `train_metor` ([engine.py:127](../metor/engine.py#L127)).

Checkpoints are already written every epoch; only the load path is missing. No
Colab tier finishes 30 epochs in one session. → [doc 40](40_Hardware.md)

### 2.4 An evaluation path for `--arch metor`

**Hook:** a new function converting model outputs to the `pred_rels` structure
[`inference/post_process.py`](../inference/post_process.py) consumes.

**What:** `(P, T, 4)` regressed trajectories + `(P, T, n_rel)` predicate scores →
per-clip candidates → `process_pred` / `association` / `format_`. Mostly
bookkeeping: durations, clip splitting, `sbj × obj × pre` ranking.

**Do not modify** `third_party/vidvrd_eval_api/`. Editing the metric makes every
number incomparable to published results.

---

## 3. Cheap experiments, no retraining

### 3.1 The `[REL]` position

```bash
--rel_token_position three_quarter
```

METOR's Eq. 10 says `end`; both predecessors say 75%. One flag, genuinely
ambiguous, and it changes the text bank.
→ [doc 31 §3.3](31_Lineage-evidence.md)

### 3.2 Pair count

```bash
--max_pair_objects 15      # 210 pairs instead of 90
```

The largest unforced choice in the reconstruction. Changes recall, compute, and
the class balance of Eq. 13's BCE at once.
→ [doc 16 §2](16_Matching-and-pair-formation.md)

### 3.3 The Eq. 11 blend

```bash
--residual_mapping
```

Off by default and labelled a DEVIATION. Changes whether the subject–object
difference is preserved or scaled by `α^k`.
→ [doc 13 §4](13_Iterative-enhancement.md)

### 3.4 The controlled ablation on the EOV stack

The most informative thing runnable today, on a 16 GB GPU with released
checkpoints:

```bash
python -m cli.evaluate --arch metor-compat --n_iter 0 ...
python -m cli.evaluate --arch metor-compat --n_iter 2 ...
```

`--n_iter 0` is bit-identical to EOV, so the delta is attributable to the module
— stronger evidence than the paper's own +1.70.
→ [doc 33 §3](33_Verification-status.md)

---

## 4. Hook points by component

| To change | Edit | Watch this test |
|---|---|---|
| how context is extracted | `ContextTokenInjector` / `PooledContextEncoder` | `test_context_tokens_receive_gradient_through_the_frozen_vit` |
| the prompt layout of Eq. 9–10 | `RefinedTextEncoder.forward` | `test_text_features_are_l2_normalised` |
| how queries are refined | `ObjectQueryRefiner` | `test_query_refiner_starts_as_the_identity` |
| the spatio-temporal design | `SpatioTemporalTransformer` | `test_st_transformer_maps_role_triple_to_relationship` |
| the Eq. 11 update | `IterativeEnhancementLayer.forward` | `test_paper_literal_update_scales_subject_object_difference_by_alpha` |
| any loss | `METORCriterion` | `test_criterion_weights_match_the_paper` |
| pair selection | `METOR.select_pairs` | `test_forward_produces_every_output_with_the_documented_shapes` |
| matching | `TrajectoryHungarianMatcher` | — (add one) |
| data / targets | `METORDataset` | `test_metor_dataset.py` |
| the compat splice | `METORFeatEmbedding` | `test_n_iter_zero_matches_the_eov_baseline_exactly` |

---

## 5. House rules

1. **Never edit `third_party/vidvrd_eval_api/`.** It is the published metric.
2. **Keep paper values and invented values in separate config files.** A test
   enforces it; the separation is what makes this repository honest.
   → [doc 23 §1](23_Configuration-and-flags.md)
3. **Mark new assumptions `PAPER-GAP` in the source** and add them to
   [doc 30](30_Paper-gaps.md).
4. **Check the predecessors before guessing.** MMP and EOV-MMP answered about
   half the open questions, including one that was a real bug.
   → [doc 31](31_Lineage-evidence.md)
5. **Preserve EOV parameter names in compat mode.** A rename does not error — it
   silently loads a randomly-initialised layer.
6. **Don't tune on the test set** beyond the benchmark's existing convention,
   and say what you did. → [doc 33 §4](33_Verification-status.md)
