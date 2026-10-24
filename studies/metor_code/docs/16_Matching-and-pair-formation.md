# Matching and pair formation — the two things the paper needs but never describes

METOR is query-based, and it classifies relations between **pairs**. Both facts
create questions the paper does not answer:

1. Which of the `N_q` queries is the dog and which is the frisbee?
2. Which pairs of queries get scored for a relation?

Neither is optional — you cannot train without answering both. Neither appears in
the paper. This document is what was chosen and why, and it is the part of the
reconstruction most likely to differ from the authors'.

---

## 1. Query ↔ ground-truth matching

**File:** [`metor/matcher.py`](../metor/matcher.py), class
`TrajectoryHungarianMatcher` ([line 58](../metor/matcher.py#L58)).

### Why it is needed

Nothing tells query #37 that it is supposed to be the dog. DETR solves this with
a **Hungarian assignment** between predictions and ground truth, computed fresh
each step. METOR inherits that machinery implicitly: it initialises its decoder
from a COCO-pretrained detector, which is trained exactly that way.

### The cost

```
cost = w_box · mean_t |pred_box − gt_box|  −  w_cls · P(gt class)
```

DETR's recipe, adapted from boxes to **trajectories**: the L1 term is averaged
over the frames where a ground-truth box actually exists. That temporal averaging
is the only real departure, and it is forced — DETR matches one box, METOR
matches a sequence. A query that tracks an object well for its whole duration
should beat one that happens to overlap on a single frame.

Trajectories with no boxes on the key-frame grid get `inf` cost and cannot be
matched.

### What is missing relative to DETR

There is **no "no-object" class**. DETR has one and penalises unmatched queries
against it. Here, Eq. 4 scores objects against CLIP text embeddings, which have
no such row — there is nowhere for that term to go. Unmatched queries are simply
not supervised.

Whether the authors did something else here is unknown.

### SciPy, and the fallback

`scipy.optimize.linear_sum_assignment` when available; otherwise a greedy
"repeatedly take the cheapest free cell" assignment
([line 35](../metor/matcher.py#L35)). Not optimal, but deterministic and never
worse than random — it exists so a missing import degrades instead of killing a
training run at hour three.

---

## 2. Pair formation

**File:** [`metor/model.py`](../metor/model.py), `METOR.select_pairs`
([line 173](../metor/model.py#L173)).

### What the paper says

> "From the visual object features `O`, we derive subject-object pairs for
> subsequent recognition"

That is the entire specification. It does not say how many, or which, or on what
basis.

### Why that is not enough

All ordered pairs of 100 queries is **9,900 per video**. Each pair goes through
the full iterative enhancement module over `T` timesteps and is scored against
132 predicates. That is not tractable, so some selection is mandatory — and the
paper does not describe one.

### What is implemented

```python
scores = objectness(obj_feats.mean(dim=1))   # a learned scalar per query
keep   = scores.topk(k).indices              # k = --max_pair_objects, default 10
pairs  = all ordered (i, j), i ≠ j           # → 90 pairs
```

Ordered, not unordered: a relation is not symmetric. `⟨dog, chase, person⟩` and
`⟨person, chase, dog⟩` are different triplets.

`objectness` is a `nn.Linear(d, 1)` at [model.py:162](../metor/model.py#L162),
learned implicitly — it receives gradient only through which pairs get selected
and therefore which pairs contribute to the loss.

### Why this is flagged as the largest unforced choice

- **The count is arbitrary.** 10 → 90 pairs. 15 → 210. 20 → 380. Each changes
  recall, compute, and the class balance of the BCE in Eq. 13.
- **The criterion is invented.** Ranking by a learned objectness score is
  reasonable, but the authors may rank by detection confidence, by matched
  ground truth during training, or take all pairs above a threshold.
- **Training and inference may want different rules.** During training you might
  force ground-truth pairs in (teacher forcing) so the relation head sees
  positives; at inference you cannot. This implementation uses the same rule for
  both, which is simple but may under-supply positives early in training.
- **Neither predecessor helps.** MMP and EOV-MMP both get trajectories from a
  *tracker* first and form pairs over tracked trajectories — there are no
  "queries" to pair. So the lineage evidence that settled so much elsewhere
  ([31_Lineage-evidence](31_Lineage-evidence.md)) is silent here.

If a trained model underperforms and the COCO initialisation
([12_Transformer-decoder-and-queries §4](12_Transformer-decoder-and-queries.md))
has been fixed, this is the next thing to interrogate.

---

## 3. From an assignment to loss targets

`build_pair_targets`, [matcher.py:126](../metor/matcher.py#L126).

Turns a query↔GT assignment into the tensors `METORCriterion` wants:

| Output | Shape | Notes |
|---|---|---|
| `rel_labels` | `(P, T, n_base_rel)` | already sliced to base categories |
| `sbj_labels`, `obj_labels` | `(P,)` | `-1` where the query matched nothing |
| `gt_sbj_boxes`, `gt_obj_boxes` | `(P, T, 4)` | |
| `sbj_box_mask`, `obj_box_mask` | `(P, T)` | false where GT is absent |
| `present_obj`, `present_rel` | `(n_obj,)`, `(n_rel,)` | multi-hot, for Eq. 16 |

Pairs where either side is unmatched are **kept**, with an all-zero relation
label and a false box mask. They are legitimate negatives for the predicate loss
and are excluded from the trajectory loss. Dropping them would bias the BCE
toward positives.

Pair labels are looked up by `(subject_row, object_row)` into the ground-truth
trajectory list — row space, not `tid` space. The dataset builds them that way
deliberately ([22_Data-pipeline](22_Data-pipeline.md)); using a raw `tid` would
silently mis-target every label.
