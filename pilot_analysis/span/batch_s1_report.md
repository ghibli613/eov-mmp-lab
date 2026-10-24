# Span idea, Batch S1 — learning "relation continues vs changes" from training labels: report

**Verdict: G-S1 FAIL. The span idea ends.**
- The full span head reaches **HELD AUC 0.696** (95% CI 0.625–0.769), against
  ≥ 0.80 with a lower bound ≥ 0.75.
- The LO2 head reaches **0.670**, against ≥ 0.75.
- A one-line baseline with no learning, the cosine distance between consecutive
  slots' PCA embeddings, scores **0.694**. The head adds nothing over it.

Scope:
- CPU only, about 3.5 minutes of training: 2 versions × 3 seeds of a
  ~0.5 M-parameter model.
- Training videos only; base labels only (0 novel labels used, asserted).
- No test file opened; no mAP computed (by design, A10).

## 1. A10, and the pinned artefacts

| amendment | what it does | preregistration.md SHA256 after it |
|---|---|---|
| A10 | the S1 design, written before any data was touched (text SHA256 `13eaced4061ccec04455710ddeb2fad5a7bb166aaf0127c9e6606721bb1891f7`) | `4410598f044358d71e055ff0dfb946c9f35a504632b49e1b43178f6408e73c80` |
| A11 | coding fix: the `--small` run crashed before any result, because PCA-256 on 104 rows has only 104 components. The missing components are zero-padded; the full run is unaffected | `6b14b0251a9b50455cf3e7c6b19847e45e092c54f961536f4d3c7d4619a83c95` |
| A12 | the pin check keeps the last pin of each path, so A11 can supersede A10's pin. A checking change only; `stage2/common.py` re-pinned | **`f86d1355be655cf7dc28a49ce06bc511940a9a8727cf88217cc7d729dfab5776`** (current) |

**All 53 pins verify, latest pin per path.** That includes every earlier batch's
artefacts, both span scripts, and `learned_lo2.npy`. Every earlier amendment's
bytes are unchanged; the chain is in `preregistration.sha256`.

## 2. Target counts (adjacent slots of the same ordered pair)

| split | transitions | known | change | continue | change share |
|---|---:|---:|---:|---:|---:|
| fit-train | 3,184 | 2,263 | 669 | 1,594 | 29.6% |
| early-stop | 372 | 258 | 56 | 202 | 21.7% |
| HELD | 982 | 771 | 213 | 558 | 27.6% |
| **all** | **4,538** | **3,292** | 938 | 2,354 | 28.5% |

With LO2's labels removed, 2,964 transitions are known (25.1% change), of which
HELD has 651.

Most of the 8,780 ordered pairs are a single slot long, which is why 13,318
slots give only 4,538 transitions.

## 3. θ (early-stop videos, known transitions)

| θ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced accuracy, full | 0.546 | 0.631 | 0.710 | 0.718 | 0.748 | 0.745 | 0.728 | 0.587 | 0.500 |
| balanced accuracy, LO2 | 0.520 | 0.614 | 0.675 | 0.713 | 0.725 | 0.691 | 0.599 | 0.531 | 0.500 |

On the 0.01 grid:
- **full: θ = 0.51.** Early-stop AUC 0.828; balanced accuracy 0.758 (precision
  0.50, recall 0.71).
- **LO2: θ = 0.44.** Early-stop AUC 0.808.

Training stopped early at epoch 1–2 in every seed (patience 10). pos_weight was
2.38 for full and 2.70 for LO2.

## 4. Evaluation (HELD known transitions) and the gate

| | transitions (change) | AUC | 95% CI |
|---|---:|---:|---|
| **full span head** | 771 (213) | **0.696** | 0.625–0.769 |
| **LO2 span head**, on transitions touching an LO2 label | 291 (115) | **0.670** | 0.576–0.767 |
| full span head, on those same LO2 transitions (reference) | 291 (115) | 0.746 | |

Full head at θ = 0.51: balanced accuracy 0.643, precision 0.439, recall 0.559
(TP 119, FP 152, FN 94, TN 406).

**Descriptive (3a): simple baselines, no learning**

| baseline | HELD AUC | LO2 transitions |
|---|---:|---:|
| summed absolute change in the 16 measured part scores | 0.606 | 0.636 |
| cosine distance of consecutive PCA-256 embeddings | **0.694** | **0.733** |

**Descriptive (3b): mechanism.** On the 558 HELD continue transitions there are
931 continuing predicates. 930 (99.9%) are in EOV-MMP's base top 20 at slot t,
and **all 930 (100%) are still there at t+1**. On its own training videos,
EOV-MMP's slot predictions are *not* sparser than the relations they describe.
This says nothing about test, where the pilot found the loss.

**Descriptive (3c): HELD segments at θ (every transition cut).** 1,842 segments:
81% are one slot long, median 1, 90th percentile 2, mean 1.35 slots.

**Gate G-S1:**

| check | needed | got | |
|---|---|---|---|
| full AUC | ≥ 0.80 | 0.696 | ✗ |
| full CI lower bound | ≥ 0.75 | 0.625 | ✗ |
| LO2 AUC | ≥ 0.75 | 0.670 | ✗ |

**Verdict: FAIL. The span idea ends.**

## 5. Side count (EOV-MMP's top-20 novel predicates vs subject-type rules)

Rules from base instances over 783 videos (≥ 99.5% one type, ≥ 100 instances):
- **animate:** stand, sit, walk, run, creep, swim;
- **vehicle:** stop.
- Not rules: fly (97.5% vehicle), lie (99.3% animate), move (99.4% vehicle),
  jump (mixed).

Over 13,318 slots: 266,360 top-20 novel predictions, of which 127,424 have a
verb with a rule. **22,327 of those (17.5%) contradict the rule, which is 8.4%
of all top-20 novel predictions.**

| verb | rule | predictions | violating | share |
|---|---|---:|---:|---:|
| stop | vehicle | 10,743 | 4,017 | **37.4%** |
| creep | animate | 12,843 | 2,510 | 19.5% |
| swim | animate | 2,580 | 555 | 21.5% |
| run | animate | 21,157 | 3,913 | 18.5% |
| walk | animate | 35,708 | 5,776 | 16.2% |
| sit | animate | 14,765 | 1,996 | 13.5% |
| stand | animate | 29,628 | 3,560 | 12.0% |

So EOV-MMP often puts novel `stop_*` on animate subjects, and animate-verb
predicates on vehicles. This is label-free and report only; no effect on mAP was
measured.

## 6. The annotation-format finding (for the thesis)

The official ImageNet-VidVRD release annotates relations differently in its two
splits:
- **Training:** every relation instance is a 30-frame chunk on a 15-frame grid
  (22,616 / 22,616 base instances). Repeated relations appear as overlapping or
  separated chunks; gaps are whole missing chunks, median 4.
- **Test:** long instances; 77% span more than one 30-frame slot.

So the training labels cannot supervise relation extents directly. They only
say whether two neighbouring labelled slots carry the same relations.

(Pre-checks: `s1_precheck.md`, `s1_precheck2.md`. The train files are
byte-identical to the official 2017 archive.)

## 7. Surprises and caveats

1. **The learned head ties a trivial baseline.** Its HELD AUC (0.696) equals the
   cosine distance of consecutive embeddings (0.694). On LO2 transitions the
   baseline does better (0.733 vs the LO2 head's 0.670). Whatever signal there
   is sits in the embedding change itself.
2. **There is a large early-stop → HELD drop** (0.828 → 0.696). The early-stop
   set is small (258 known transitions, 56 changes). Training stopped at epoch
   1–2, which suggests over-fitting from the first epoch. A plausible reason:
   on fit-train slots, the learned-part inputs come from a network trained on
   those slots' labels. They are near-perfect there and noisier on HELD, so the
   head learns to trust an input that degrades at evaluation. This is not
   tested.
3. **The targets are weak by construction.** Only 3,292 transitions are known,
   most pairs are a single slot, and "change" includes any change in the
   label set. That covers annotation chunks starting or ending, not only real
   relation changes.
4. **The mechanism check is uninformative on training videos.** EOV-MMP keeps
   100% of continuing predicates in its top 20 there, because it has seen these
   videos. The pilot's test-set loss can't be examined on training data.
5. **The side count is the one positive by-product.** 17.5% of rule-covered
   top-20 novel predictions break a subject-type rule learned from base labels.
   It was not converted into an mAP test (not asked; it would need its own
   brief).
6. **A11 and A12 were process fixes.** A11 fixed a small-run crash before any
   result; A12 fixed the pin checker, which rejected superseded pins. Neither
   touched a result.

## 8. Hashes of new files

| file | SHA256 |
|---|---|
| `span_common.py` (pinned by A10) | `0d74fc2cad510b48260f06e749377a9c78ca0f9b8ae79800751ee5032a2da384` |
| `s1_run.py` (pinned by A11) | `3cb2a76a7d2d9b059104fa0b7009b10d58af1a75726d9200f959ea01005dbb59` |
| `data/s1_results.json` | `182096795a8dfb144fbaa277b4e0e6c9dd8dbbd05c83d7358815ef3625ec6465` |
| `data/s1_transitions.npz` (targets and probabilities per transition) | `e23a07446a0d864afe81ff028dc0223d66424b78ee1226af49358b2bd5cf6bbd` |
| `data/span_full.pt` (3 seeds + normalisation) | `efbb2d39b9f196fb55ba50d7f63cfdfa532db453b92a1b425e54329848d9d26f` |
| `data/span_lo2.pt` | `7a2b8dc3dcc021cdd75fb6775165e10a7859ec2cbb2323deed69c9ca340369c6` |
| `data/pca_fit.npz` | `8ed364192d88e7d16008008b70c9406fce7b9181a8219e905becacff11bf4bfc` |
| `data_small/s1_results.json` (5-video plumbing run) | `6faaba68d6e2de53cd908d7cc38d968373f5011d539621fb0c2db9e2f1106199` |

Logs: `logs/s1_run.json` and `logs/s1_run_small.json` (configuration,
environment, code, input and output hashes), with the console output in
`s1_full.out` and `s1_small.out`. `data/` and `data_small/` are git-ignored.
