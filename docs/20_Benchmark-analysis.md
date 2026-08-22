# What this benchmark actually measures

Measured 2026-08-22 from `data/vidvrd/anno/`. These numbers are the empirical
core of the research direction in [90_Research-ideas.md](90_Research-ideas.md);
they are reported here separately so they can be checked without accepting any
argument built on them.

---

## 1. What this benchmark actually measures

Measured on 2026-08-22 from `data/vidvrd/anno/`, reproducible with the snippets
in this section's method notes.

VidVRD's 132 predicates are compositional — `{verb}_{spatial}`, e.g.
`walk_left`, `run_past`, `sit_above`. Decomposing every test instance and asking
what evidence is needed to name it:

| Evidence required | Instances | Share |
|---|---|---|
| Box trajectory only (`walk_left`, `run_past`, `swim_front`) | 2,250 | 46.5% |
| Single frame + box geometry (`stand_behind`, `sit_above`) | 2,013 | 41.6% |
| Box dimensions alone (`taller`, `larger`) | 250 | 5.2% |
| Single frame appearance (`watch`, `ride`, `hold`) | 179 | 3.7% |
| **Appearance change over time** (`bite`, `kick`, `fight`, `play`) | **143** | **3.0%** |

**97% of the test set is decidable from a single frame plus box geometry** — the
exact information EOV's design provides (§3.1). And because mAP is averaged over
videos:

```
test videos containing zero appearance-dynamics instances : 142 / 200  (71%)
```

For 71% of the metric, a perfect temporal model and a blind one are
indistinguishable.

The second measurement concerns the open-vocabulary axis:

```
novel predicates                                            : 61
  both verb and spatial component also appear in base       : 44  (72.1%)
  at least one component appears in base                    : 61  (100%)
  neither appears in base                                   :  0
```

**No novel predicate on this benchmark requires a genuinely unseen concept.**
`sit_behind` is novel while `sit_*` and `*_behind` are both base. The split tests
recombination, not vocabulary.

> Caveat, stated because it is the obvious attack: the taxonomy in the first
> table is a judgement call. `swim` is filed as single-frame-decidable because
> water is visible in one frame; someone may disagree. The way to make the
> argument independent of the taxonomy is the geometry-only baseline in §6.1 —
> an empirical measurement that does not depend on anyone's category labels.

---
