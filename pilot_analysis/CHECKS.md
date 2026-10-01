# Checks 1 and 2 — the two gates before C1

**Written:** 2026-09-03. **Cost:** CPU only, zero dollars, ~25 min wall clock on
this box. No GPU, no new inference — everything below reuses
`preds_from_pod/full/` and the training annotations.

**Headline:** **C1 is GO.** **C3 is NO-GO** — not because the bias is absent, but
because it does not exist in the metric C3 would be scored on.

Reproduce with:

```
python pilot_analysis/scripts/build_primitive_map.py
python pilot_analysis/scripts/check1_supervision_multiplier.py
python pilot_analysis/scripts/check2_base_novel_bias.py
```

---

## The primitive map

Full table: [`primitive_map_table.md`](primitive_map_table.md). Machine-readable:
[`primitive_map.json`](primitive_map.json).

**132 predicates → 25 action primitives + 15 spatial primitives = 40 classes**
(3.30× compression). Three decomposition bases:

| basis | n | what it is |
|---|---|---|
| `lexical` | 105 | mechanical suffix split, `walk_behind` → (walk, behind). 11 verbs × 12 spatials. No review needed. |
| `degenerate` | 16 | one slot genuinely empty: `beneath` → (—, beneath), `feed` → (feed, —) |
| `entailed` | 11 | one morpheme, two primitives: `chase` → (chase, behind). Hand-authored. |

**Zero collisions** — every predicate has a unique (action, spatial) pair, so a
purely compositional scorer can still separate all 132. This was not guaranteed:
it holds because `chase`/`follow` kept distinct action primitives rather than
both collapsing to `move`.

**Three mappings I am not confident in, flagged for your review** (all
`confidence: low` in the JSON):

- `pull` → (pull, **toward**) — the object moves toward the subject, but `with`
  is equally defensible.
- `drive` → (drive, **inside**) — person-inside-vehicle, but the annotation may
  intend an external view where containment is not visible.
- `fall_off` → (fall, **beneath**) — `off` is not in the spatial vocabulary.
  `away` is the alternative; it would move 4 base instances into `away` and
  break its currently-zero support. Both readings are negligible for the
  numbers below.

Six more are `medium` (`ride`, `touch`, `hold`, `bite`, `kick`, `fight`) — all
contact/proximity entailments. `watch` is `medium` for being left spatial-less.

---

## CHECK 1 — the supervision multiplier — **GO**

Supervision counts **base predicates only**: `Model.forward` trains on
`labels['pre_label'][:, :, base_pids]`, so a novel-predicate instance contributes
zero gradient.

### (b) Vocabulary size

40 primitives for 132 predicates. Not "~100 action primitives" — pooling
happens. The largest primitives carry thousands of instances: `stand` 4676,
`walk` 4001, `right` 4130, `left` 4124, `behind` 3399, `front` 3369.

### (c) The multiplier — 71 base predicates

| | per PREDICATE | per PRIMITIVE (weaker slot) | multiplier |
|---|---|---|---|
| median | 191.0 | 852.0 | **4.37×** |
| geomean | 156.8 | 618.4 | **3.94×** |
| **bottom quartile** (n=18, ≤54 instances) median | 35.0 | 176.0 | **4.46×** |
| **bottom quartile** geomean | 20.3 | 107.0 | **5.28×** |

A predicate is composed from both slots, so the binding constraint is the rarer
one — every figure above is the *minimum* of the two primitives, i.e. the
pessimistic reading.

**The rare tail benefits more than the average** (5.28× vs 3.94× geomean). That
is precisely what C1 needs and it was not guaranteed. Extremes: `creep_beneath`
3 → 176 (58.7×), `jump_next_to` 6 → 226 (37.7×), `fly_above` 36 → 854 (23.7×).

**Five base predicates get 1.00×** — `fall_off`, `chase`, `faster`, `touch`,
`follow`. Their action primitive is a singleton, so nothing pools. C1 will not
help them and should not claim to.

### (d) Novel coverage

- **44 of 61** novel predicates have both primitives with nonzero base support.
  Median base support of the weaker slot: **383 instances** (geomean 274, range
  12–1521).
- **17 of 61** contain a primitive with **zero** base support. **This is exactly
  the pre-stated 17** — the mapping was built independently and reproduced the
  figure, which is a useful independent check on it.
  - unseen **actions** (7): `bite`, `drive`, `feed`, `fight`, `hold`, `kick`, `pull`
  - unseen **spatials** (2): `away`, `inside`
  - the 17: `away`, `bite`, `creep_away`, `drive`, `feed`, `fight`, `fly_away`,
    `hold`, `jump_away`, `kick`, `lie_inside`, `move_away`, `pull`, `run_away`,
    `sit_inside`, `stand_inside`, `walk_away`

### (e) Predicted gain

Fitted on base, zeros kept, n=71: **AP = −13.88 + 6.46·ln(N)**, R² 0.237. (The
report's +6.14 is the same slope in the two-variable model that also carries the
group indicator; using 6.14 instead scales everything below down ~5%.)

| | measured | extrapolated | Δ |
|---|---|---|---|
| base predicates, mean AP | 18.79 | 27.40 | **+8.61** |
| novel, 44 covered | 7.86 | 22.39 | +14.53 |
| novel, 17 uncovered | 17.49 | 17.49 | 0 |
| **novel, all 61** | **10.55** | **21.03** | **+10.48** |

**Read these as a ceiling, not a forecast.** Three reasons, in order of severity:

1. The novel row extrapolates a base-fitted line far outside its support — to
   predicates whose current supervised count is literally zero, where the fit
   predicts AP < 0 and the model actually scores 10.55 off frozen CLIP text alone.
2. A pooled instance is a noisier positive than a native one: a `walk_behind`
   frame is weaker evidence for `behind` than a `behind` frame is.
3. R² is 0.237 with a ±3.17 CI on the slope. This is a correlational fit.

The **base** row (+8.61) is the defensible one — it is in-sample and does not
cross the supervised/unsupervised boundary.

### GO

Rare tail gets 5.28× (criterion: ≥2×). 44/61 novel predicates have substantial
base support at a median of 383 instances. Both conditions met.

---

## CHECK 2 — base/novel score bias — **NO-GO for C3**

### (a) The novel pass scores only the 61 novel predicates

Settled by inspection and verified against the dump.
[`models/relation_classifier.py:511-514`](../models/relation_classifier.py#L511-L514):
text embeddings come from `split_text_embeddings(split='novel')`, so `pre_scores`
has 61 columns, which are then scattered into a `zeros([..., 132])` at
`novel_pids`. **Base predicates carry a hard zero and cannot be predicted.**

Confirmed empirically — across 7066 segments:

| pass | distinct predicates ever in a top-20 list |
|---|---|
| novel | 61 (0 base, 61 novel) |
| all | 132 (71 base, 61 novel) |

This also resolves the predicate half of the §B.17 code/paper question: for
*relationships* the code restricts to novel, matching the paper. The object-side
question in §B.17 is untouched by this.

**Consequence for C3: in the novel split there is no base/novel competition to
calibrate.** Everything below is therefore measured on the **all** pass, the only
place the two compete. 1170 GT instances match a predicted pair at vIoU ≥ 0.5
(base-GT 1037, novel-GT 133).

### (b) Where errors land

| GT group | n | errors | top-1 is base | top-5 is base |
|---|---|---|---|---|
| base-GT | 1037 | 638 | 93.6% | 79.9% |
| novel-GT | 133 | 125 | **95.2%** | 70.4% |
| null: uniform over vocab | | | 53.8% | 53.8% |
| null: GT frequency | | | 88.6% | 88.6% |
| null: model's own top-1 prior | | | 91.8% | — |

Against the 53.8% uniform null this looks like a 1.8× bias. **Against the right
nulls it nearly vanishes.** Base predicates are 88.6% of the GT and 91.8% of the
model's unconditional top-1 output. And the statistic barely moves with the GT
group: 93.6% on base-GT vs 95.2% on novel-GT, +1.6pp. A *specific* seen-unseen
bias would show up as errors landing on base much more often when the truth is
novel. It does not.

### (c) Score distributions

Here the effect is real and large:

| | base-GT instances | novel-GT instances |
|---|---|---|
| top-1 correct | 38.5% | **6.0%** |
| best base score (mean) | 0.5168 | 0.4974 |
| best novel score (mean) | 0.2707 | 0.2969 |
| base > novel within the instance | 96.0% | 89.5% |
| **score of the true predicate** | **0.3768** | **0.2040** |
| median rank of the true predicate | 2 | **6** |

The true label gets **1.85× less score when it is novel**, and sits four ranks
lower. Base predicates outscore novel ones by ~0.20 mean even on instances whose
truth is novel.

### The test that settles it

Mask all 71 base predicates post hoc on the all-split dump:

```
novel-GT top-1:  6.0%  ->  42.1%   (7.0x)
```

That reproduces the separately-run novel-pass dump (41.4%) to within 0.7pp —
independent confirmation that the two dumps are consistent. **The model's ranking
*among* novel predicates is fine. Base competitors are what suppress it.**

So calibrate? One-parameter sweep, multiply every novel score by *k*:

| k | novel-GT top-1 | base-GT top-1 | overall |
|---|---|---|---|
| 1.00 | 6.0% | 38.5% | **34.8%** |
| 1.50 | 19.5% | 28.4% | 27.4% |
| 2.00 | 29.3% | 18.9% | 20.1% |
| 3.00 | 39.1% | 7.3% | 10.9% |
| 4.00+ | 42.1% | 2.5% | 7.0% |

**It is a pure trade.** No *k* improves both groups, and overall accuracy falls
monotonically. This is not miscalibration that a correction recovers; it is
zero-sum competition between 71 trained classifiers and 61 frozen-text ones.

### Why the headroom is small even so

Top-1 accuracy is not the reported metric. For **mAP**, a monotone per-predicate
rescale is a *no-op* — AP for predicate *p* depends only on the ranking of *p*'s
own detections, which a constant factor preserves. The only channel by which
competition costs AP is the top-20 truncation and the greedy association:

- the true predicate survives the top-20 cut for **96.5%** of base-GT instances
  but only **82.0%** of novel-GT ones;
- novel predicates score **8.22** mean AP in the all pass vs **10.55** in the
  novel pass — **2.33 points** lost to competition;
- weighted into the 132-predicate mean, perfect recovery is worth **≈ +1.1
  all-split mAP, and exactly +0.00 novel-split mAP.**

### NO-GO

C3 targets a bias that (i) does not exist in the novel-split metric by
construction, (ii) is not conditioned on GT group once the right null is used,
(iii) cannot be fixed by rescaling without an equal-or-larger base loss, and
(iv) caps out around +1.1 mAP on the one split where it applies. Drop it.

**Keep the measurement as a finding, though** — "removing base competitors moves
novel top-1 from 6.0% to 42.1%" is a clean statement about why the all-split and
novel-split numbers diverge, and it belongs in the thesis whether or not
anything is built on it.

---

## Surprises

1. **Novel predicates have nonzero `train` counts.** `cls2count` reports 3301
   training instances across the 61 novel predicates — they are present in the
   training videos and masked out of the loss, not absent from the data. C1's
   pooling must count base predicates only or it will silently leak supervision
   it does not have. Every number above does.
2. **`away` and `inside` have exactly zero base support.** Two entire spatial
   primitives are unseen — 10 of the 17 open-world novel predicates are unseen on
   the *spatial* slot, not the action slot. C2's spatial head, which the design
   assumes is the easy geometric one, has no training signal for these at all.
3. **`toward` has 12 base instances**, from `jump_toward` (10) and `fly_toward`
   (2), yet five novel predicates depend on it. It is nominally covered and
   effectively unseen.
4. **The 17 uncovered novel predicates outscore the 44 covered ones** — mean AP
   17.49 vs 7.86. This is an artifact, not a reversal: **both medians are 0.00**,
   and the mean is carried by `pull` (AP 66.7 on 3 test instances), `hold` (62.5
   on 8) and `lie_inside` (50.0 on 1). Median test count in that group is 3.
   Worth knowing before anyone reads it as evidence against C1.
5. The post-hoc mask reproducing the separate novel-pass run to 0.7pp is a free
   consistency check on the two dumps that we had not previously done.

---

## Next

Check 1 says GO. Building **C1** next: primitive vocabulary → primitive-level
classifiers trained on pooled base instances → composition to predicate scores.
No further analysis first.
