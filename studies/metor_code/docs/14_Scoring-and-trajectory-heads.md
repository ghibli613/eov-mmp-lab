# Scoring and trajectory regression

**Paper:** Eq. 4 and Eq. 5 · **File:** [`metor/modules/heads.py`](../metor/modules/heads.py)

---

## 1. The open-vocabulary trick, in four lines

A normal classifier ends in `nn.Linear(d, n_classes)`. Its weight matrix has one
row per class, learned from examples of that class — so the class set is frozen
at training time.

METOR replaces that matrix with **text embeddings of the class names**:

```python
similarity = visual_features @ text_embeddings.T
```

The row for "zebra" is computed *from the word* "zebra". Hand it a class name it
has never trained on and it still produces a row. That is the whole of "open
vocabulary", and everything else in the repository exists to make the visual
side good enough for that comparison to mean something.

METOR's addition is that the text side is also conditioned on the video
([11_Contextual-refinement-encoding](11_Contextual-refinement-encoding.md)), so
both ends adapt.

---

## 2. Eq. 4 — scoring

```
S_s = σ(cos(Ô_s, T_o))      S_o = σ(cos(Ô_o, T_o))      S_r = σ(cos(R̂, T_r))
```

`CosineScoreHead`, [heads.py:43](../metor/modules/heads.py#L43).

### Temperature

τ = 0.01. **METOR never states a temperature**, but without one the logits sit
in [−1, 1] and the sigmoid is nearly linear over that range — it cannot produce
confident scores, and it cannot produce the reported numbers.

EOV-MMP states it outright: *"The temperature parameter τ in both Eq. 5 and
Eq. 10 is set to 0.01."* MMP uses the same. See
[31_Lineage-evidence](31_Lineage-evidence.md).

### sigmoid vs softmax

Eq. 4 writes `σ` for all three, but the three are not the same kind of question:

| | Activation | Why |
|---|---|---|
| predicates `S_r` | **sigmoid**, independent | A pair can hold several relations at once — a dog can be `run_behind` *and* `larger_than` a person. Multi-label. |
| subject `S_s`, object `S_o` | **softmax** over classes | A trajectory is one thing. Single-label. |

The `mode` argument selects; the model uses `"sigmoid"` for predicates and
`"softmax"` for objects, matching the whole lineage's behaviour and the loss
functions in [15_Training-objective](15_Training-objective.md) (BCE vs CE).

### A subtlety in what gets scored

`S_r` keeps the time axis, `(P, T, n_rel)`; `S_s`/`S_o` are pooled to `(P, n_obj)`.
A pair's predicate genuinely changes over a video and the post-processor stitches
per-clip predictions; a trajectory's category does not.

### Training scores a smaller vocabulary

At training time the columns are sliced to **base categories only**
([`metor/engine.py:46`](../metor/engine.py#L46)). Novel categories must never be
supervised, or every novel-split number afterwards is meaningless. The labels are
remapped to match the slice at [engine.py:67](../metor/engine.py#L67) — get that
wrong in either direction and the model trains happily against misaligned
columns.

---

## 3. Eq. 5 — trajectory regression

```
τ_s = M_b(Ô_s)        τ_o = M_b(Ô_o)
```

`TrajectoryRegressionHead`, [heads.py:12](../metor/modules/heads.py#L12). A 3-layer
MLP emitting **normalised `cxcywh` through a sigmoid**, so every coordinate is in
[0, 1] by construction — Deformable-DETR's convention, and therefore the
convention of the detector this is meant to initialise from.

A "trajectory" is just the stack of per-timestep boxes: `(P, T, 4)`.

The head is **shared** between subject and object. Eq. 5 writes `M_b` once for
both, and a subject and an object are the same kind of thing — only their role in
the pair differs.

### Why this exists at all

This is a real architectural difference from the predecessors, not a detail.
EOV-MMP and MMP get trajectories from a **separate tracker** (deep_sort +
AFLink). METOR regresses them from the same enhanced features it scores, which
is what makes the framework "unified" — and which is why it needs the trajectory
loss of Eq. 15 that its predecessors did not have.

It is also why `--arch metor` has no evaluation path yet: the vendored
post-processing expects tracker output.
[32_Not-implemented §2](32_Not-implemented.md).

### The last linear is zero-initialised

So every box starts at the centre of the frame rather than at a random corner,
and early training is not dominated by nonsense predictions. DETR's own
initialisation.
