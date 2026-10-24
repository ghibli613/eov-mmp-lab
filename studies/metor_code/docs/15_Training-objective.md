# The training objective

**Paper:** Sec. 3.5, Eq. 12–16 · **File:**
[`metor/modules/criterion.py`](../metor/modules/criterion.py)

---

## 1. Eq. 12 — the total

```
L = L_rel-ctr + L_obj-ctr + θ_traj·L_traj + θ_ctx·(L_rel-ctx + L_obj-ctx)
```

with **θ_traj = 1.0, θ_ctx = 0.2, θ_cst = 0.1** (Sec. 4.2).

Five terms, and three of them are doing jobs EOV-MMP did not have:

| Term | Eq. | What it is for |
|---|---|---|
| `L_rel-ctr` | 13 | predicate scores — multi-label, so BCE |
| `L_obj-ctr` | 14 | subject and object categories — single-label, so CE |
| `L_traj` | 15 | **new**: METOR regresses trajectories (Eq. 5); its predecessors used a tracker |
| `L_rel-ctx`, `L_obj-ctx` | 16 | **new**: force `C_o`/`C_r` to actually be *about* the categories present, rather than drifting into arbitrary useful features |

Implementation: `METORCriterion.forward`,
[criterion.py:160](../metor/modules/criterion.py#L160). It reports every term
separately in `loss_dict`, so an ablation can see which one moved, and it
**skips** terms whose inputs are absent rather than faking zeros.

`test_criterion_weights_match_the_paper` recomputes Eq. 12 by hand and checks the
total — a silently mis-weighted term would still train, just not to the paper's
objective.

---

## 2. Eq. 13 — relationship contrastive

```
L_rel-ctr = (1/|C^b_r|) · BCE(S_r, r̃)
```

[criterion.py:53](../metor/modules/criterion.py#L53).

### This one contained a real bug

The `1/|C^b_r|` is an **average over predicate categories**, not an extra scale
factor on top of one. METOR folds the summation into `BCE(·)` so it is easy to
misread — but EOV-MMP writes the identical loss out in full (its Eq. 25):

> `L_rel = (1/|C_R_b|) · Σ_{r ∈ C_R_b} BCE(ŷ_r, y_r)`

A sum over categories divided by their count. So `reduction="mean"` *already is*
the `1/|C^b_r|`.

The first version of this file applied `reduction="mean"` **and then divided by
`n_base_rel` again**. On VidVRD that shrank the predicate term by roughly 100×,
effectively deleting the main training signal from Eq. 12 while everything still
appeared to run.

Found by reading EOV-MMP. Fixed, and pinned by
`test_relationship_loss_is_a_mean_over_categories_not_a_double_division`, which
checks the exact value in closed form.

A guard now raises if the score tensor's vocabulary size does not match
`n_base_rel` — passing full-vocabulary scores would normalise against the wrong
count.

Scores arrive **post-sigmoid** from Eq. 4, so this is `binary_cross_entropy`, not
the `_with_logits` form, and it is clamped because a saturated sigmoid gives
`log(0)`.

---

## 3. Eq. 14 — object contrastive

```
L_obj-ctr = CE(S_s, s̃) + CE(S_o, õ)
```

[criterion.py:89](../metor/modules/criterion.py#L89).

Takes **logits**, not the softmax scores of Eq. 4 — `cross_entropy` applies its
own log-softmax and feeding it softmaxed input applies it twice.

Pairs whose subject or object matched no ground truth carry `ignore_index = -1`
and drop out. A video where neither matched contributes exactly zero, not NaN —
unmatched queries are common, so this is the normal case, not an edge case.
Pinned by `test_ignored_object_labels_do_not_contribute`.

Note Eq. 14 carries **no** `1/|C|` normalisation, unlike Eq. 13. That asymmetry
is in the paper and is preserved here.

---

## 4. Eq. 15 — trajectory

```
L_traj = L_box + θ_cst · L_cst

L_box = (1/|T|) Σ_t Σ_{e∈{s,o}} SmoothL1(B_e^(t), B̃_e^(t))
L_cst = (1/|T−1|) Σ_t Σ_{e∈{s,o}} ‖B_e^(t+1) − B_e^(t)‖₁
```

[criterion.py:106](../metor/modules/criterion.py#L106).

- **`L_box`** — Smooth-L1 per frame per entity, over frames where ground truth
  exists (that mask matters: trajectories do not span the whole video).
- **`L_cst`** — temporal smoothness. Note it is computed on **predictions only**.
  It is a *prior*, not a fit: it would be minimised by a completely motionless
  box. `θ_cst = 0.1` is what stops it winning. Pinned by
  `test_trajectory_consistency_is_zero_for_a_static_trajectory`.

Eq. 15 sums over entities inside `L_box`, so the implementation adds the subject
and object terms rather than averaging them
([criterion.py:192](../metor/modules/criterion.py#L192)).

---

## 5. Eq. 16 — the contextual losses

```
L_rel-ctx = BCE(cos(C_r, T_r), R̃)
L_obj-ctx = BCE(cos(C_o, T_o), Õ)
```

[criterion.py:139](../metor/modules/criterion.py#L139).

`R̃` and `Õ` are the multi-hot sets of categories actually present. This asks the
context tokens to be **linearly readable as "which objects / which relations are
here"** — which is what keeps them from becoming arbitrary features that happen
to reduce the other losses.

Two things the paper leaves undefined:

- **`BCE(cos(·), ·)` is not well formed.** A cosine lives in [−1, 1] and BCE on a
  negative input is undefined. Here: `(cos + 1)/2` before the BCE. Some mapping
  into [0, 1] is required; the paper states none.
- **Pooling over context tokens.** There are 8 of them and one label set, so
  they are max-pooled — any single token may carry the evidence for a given
  category.

Both recorded in [30_Paper-gaps §4](30_Paper-gaps.md).

---

## 6. What the loop does with all this

[`metor/engine.py`](../metor/engine.py), `train_one_epoch`
([line 80](../metor/engine.py#L80)):

```
forward the whole video                     model(images=...)
assign queries to GT trajectories           matcher(...)        → doc 16
build per-pair targets                      build_pair_targets(...)
slice scores + labels to base categories    _slice_to_base / _remap_labels
one loss, one backward                      criterion(...)
```

One forward, one loss, one backward **per video**. That is the paper's
"end-to-end manner", and it is the structural difference from EOV-MMP's step 4,
whose `End2End_Model` calls `.backward()` inside its own forward, per frame and
per object pair.

Non-finite losses are logged and skipped rather than propagated — with a
Hungarian matcher and ragged ground truth, one pathological video should not end
a 30-epoch run.

Hyperparameters (Sec. 4.2, all in `configs/metor_vidvrd.yaml`): AdamW, lr 1e-4,
×0.1 at epochs 15/20/25, 30 epochs, batch size 1.
