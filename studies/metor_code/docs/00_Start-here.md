# Start here

## What the paper claims

> Yongqi Wang, Xinxiao Wu, Shuo Yang. **METOR: A Unified Framework for Mutual
> Enhancement of Objects and Relationships in Open-vocabulary Video Visual
> Relationship Detection.** IJCAI 2025. arXiv:2505.06663

The task, **Open-VidVRD**, is: given a video, output `⟨subject, predicate,
object⟩` triplets where subject and object are *trajectories* (a box per frame),
and where some object and predicate categories were **never seen in training**.
Categories are split `base` / `novel`; you train on base and are evaluated on
both.

METOR's argument is a diagnosis of its predecessors. They run a **cascade**:
detect objects, freeze that answer, then classify relations from it. When the
detector is wrong the relation classifier inherits the error. The paper's
Figure 1(c) counts exactly this — relations classified correctly from
ground-truth trajectories but wrongly from detected ones.

Its fix is to make the flow **bidirectional**: objects produce a relationship
representation, and that representation is folded back into the objects, several
times, before anything is scored.

Two contributions:

1. **Contextual refinement encoding** (Sec. 3.3) — learnable context tokens are
   appended to the frozen CLIP ViT's input, and what comes out conditions *both*
   the object queries and the class text embeddings on this particular video.
   → [11_Contextual-refinement-encoding](11_Contextual-refinement-encoding.md)
2. **Iterative enhancement** (Sec. 3.4) — the objects↔relationship feedback loop.
   → [13_Iterative-enhancement](13_Iterative-enhancement.md)

Reported: **16.74 novel / 27.52 all** mAP on VidVRD SGDet, against EOV-MMP's
15.04 / 26.34.

---

## What this repository is

A reconstruction from the paper text. [The authors'
repository](https://github.com/wangyongqi558/METOR) contains a PDF and a README
saying a release "will follow", with no date.

Built alongside it: **MMP (AAAI'24) → EOV-MMP (TPAMI'25) → METOR (IJCAI'25)** are
one group's line of work, and the first two *are* public. Their code is vendored
here as infrastructure, and their papers answer about half of what METOR leaves
unstated ([31_Lineage-evidence](31_Lineage-evidence.md)).

---

## Honest status

| Question | Answer |
|---|---|
| Can you read the method here? | **Yes.** Eq. 1–16 are implemented, cross-referenced, and indexed in [20_Equation-index](20_Equation-index.md). |
| Does it run? | **Yes.** 137 tests pass; forward and backward are exercised on stub and on real data. |
| Has it been trained? | **No.** Not once, not partially. |
| Will it reproduce 16.74 / 27.52? | **Not as it stands** — see below. |

**The one caveat that dominates all others:** the paper initialises its
Transformer decoder from a **COCO-pretrained detector**, and that is not wired up
here. The open-vocabulary detection ability comes from that initialisation.
Training this repository's default decoder from random init on 800 VidVRD videos
will not approach the paper's `mAP_o` of 55.09, regardless of everything else.
Full explanation: [32_Not-implemented §1](32_Not-implemented.md).

Everything else that is unverified or assumed:
[33_Verification-status](33_Verification-status.md).

---

## The two architectures

They are **not** interchangeable, and only one of them can run today.

| | `--arch metor` | `--arch metor-compat` |
|---|---|---|
| What it is | the paper as written | METOR's modules on the EOV-MMP stack |
| Object queries | 100 (paper, Sec. 4.2) | 300 (EOV's) |
| Trajectories | regressed, Eq. 5 | deep_sort + AFLink |
| Losses | 5 (Eq. 12) | EOV's |
| Loads released checkpoints | **no** — shapes differ | **yes** |
| Runs on 16 GB today | no | **yes** |
| Comparable to Table 1 | in principle | no |

### Why `metor-compat` exists

Because 100 ≠ 300. EOV-MMP's released detector cannot be loaded into a
METOR-shaped model at all — the architectures differ, not merely the weights.
And EOV-MMP never released detector *training* code either. So the faithful path
requires training an open-vocabulary video detector from COCO init with no
reference implementation to check against.

`metor-compat` takes the other route: hold the entire EOV stack fixed — same
queries, same released checkpoints, same tracker, same post-processing — and
insert only METOR's two modules. Then `--n_iter 0` versus `--n_iter 2` is a
**controlled measurement** of what the iterative enhancement module is worth.

`--n_iter 0` is bit-identical to the EOV baseline, pinned by
`tests/test_compat_equivalence.py`. That matters: it means any delta is
attributable to the module and not to plumbing.

It is worth noting that this is **stronger evidence than the paper's own**. METOR
reports +1.70 over EOV-MMP, but changes the module, the query count (300→100),
the training scheme (4-step → 30-epoch end-to-end) and the loss count (2→5) all
at once. Its own Table 4 puts iterations=0 at 15.16, against EOV-MMP's 15.04 —
so the module accounts for roughly +1.3 of the +1.70, and only if all else were
equal, which it is not. See [33_Verification-status §3](33_Verification-status.md).

---

## Quickest useful thing to do

Not "train the faithful model" — that needs the COCO init first. Instead:

```bash
python -m pytest tests/            # 137 tests, ~85s, no GPU needed

# then, on a 16GB GPU: the controlled ablation
python -m cli.evaluate --arch metor-compat --n_iter 0 ...
python -m cli.evaluate --arch metor-compat --n_iter 2 ...
```

That reproduces the EOV baseline and measures METOR's module against it, for
free, on hardware you can rent by the hour.
Setup details: [40_Hardware](40_Hardware.md),
[23_Configuration-and-flags](23_Configuration-and-flags.md).
