# Brief for the Claude chat: can EOV-MMP be trained in this lab?

Status: assessment only. No training run has been attempted. Everything marked
**measured** comes from the pilot study already reported; everything marked
**estimated** is extrapolation and is flagged as such.

---

## 1. The short answer

**Only step 4 of the paper's four-step scheme is trainable here.**

The paper trains in four stages: (1) the deformable-DETR detector, (2) the
object classifier, (3) the relation classifier with CoCoOp-style prompt
learning, (4) joint end-to-end fine-tuning of all three. The authors lent us
trained checkpoints for steps 1-3 and a fully trained end-to-end checkpoint,
**but not the training code for steps 1-3**. Those checkpoints are the outputs,
not the recipe.

So the two things we can do are:

- **Fine-tune end to end** from the supplied components (`cli/train.py`, step 4).
- **Not** retrain any individual component, and **not** train from scratch.

This matters for how any thesis claim is framed. We cannot say "we retrained
EOV-MMP". We can say "we resumed the authors' step-4 fine-tuning under condition
X and measured the delta", which is a narrower but still publishable kind of
claim.

## 2. The data and weights are all present

Verified on the local machine today, not assumed:

- 800 train annotations, 200 test annotations
- 1000 frame directories, 42 GB total (244,100 train frames / 52,104 test)
- `train_object_trajectories_gt.json` — training uses **ground-truth**
  trajectories, not detected ones
- `VidVRD_ECC_train.json`, both split-info files, the CLIP feature bank
- all four checkpoints (detector, object classifier, relation, AFLink)

Nothing needs to be downloaded or rebuilt. The blocker is not data.

The trainable surface: `text_encoder`, `backbone` and `pre_classifier` are
frozen; everything else receives gradients.

## 3. What it would cost — and why there is no cheap configuration

**Measured anchor:** full-test-set inference runs at **60 s/video** on a rented
RTX 4090 with the thread pool capped (~2-3 h for all 200 videos, both predicate
splits fused).

**Estimated from that anchor:** training is roughly **1.5-2 days per epoch** on
one 4090. The train set is 4.7x the test set in frames, backward adds 2-3x over
forward, and the loop runs a **full test-set evaluation after every epoch**.

| target | epochs | wall clock (est.) | 4090 rental (est.) |
|---|---|---|---|
| match the released `end2end_base` checkpoint | 6 | ~10 days | $75-100 |
| the default schedule | 20 | ~5 weeks | $300-400 |

**The obvious cost lever does not work.** The repo documents `--frame_stride 30`
(the paper's stated sampling) as "~30x cheaper". We **measured** it: stride 1
costs only **17% more runtime** than stride 30, because the detector runs on
every frame either way — CLIP encoding was never the bottleneck. And stride 30
is catastrophic for accuracy (~45x worse; it emits unmerged 30-frame fragments).
So stride 1 is mandatory *and* there is no cheap configuration hiding behind it.

Hardware: 24 GB VRAM (3090/4090) is the right target; a 16 GB T4 is plausible
but untested. Host RAM **>= 32 GB** is a hard requirement we established the
hard way — a 12.7 GB host cannot finish the test set at all, regardless of how
long it runs.

## 4. Three defects in the training loop that block a multi-day run

These are the real blockers, and they are all in ~110 lines of `cli/train.py`:

1. **`--resume` is a dead flag.** It is declared in the argument parser and read
   nowhere in the codebase. `--start_epoch` only shifts the loop counter; it
   loads no weights and no optimizer state.
2. **Only the best-mAP checkpoint is ever written**, and it carries no optimizer
   or scheduler state. A preempted rented instance loses everything since the
   last mAP improvement, with no way to pick up.
3. **There is no `--limit`.** The batch-limiting machinery we built lives in the
   pilot's `dump_predictions.py`, not in the CLI — so there is currently no way
   to time a short training run before committing to a 10-day one.

Together these mean: on a *spot/interruptible* instance the run is essentially
guaranteed to be wasted, and on a reserved instance we would be committing
~$100 on an unmeasured throughput estimate. Fixing all three is a small, purely
additive change (last-epoch checkpointing with optimizer state, a real resume,
a `--limit`) that touches no model-semantic code.

## 5. One methodological point to record either way

`cli/train.py` evaluates on the **test set** after every epoch and keeps the
best-mAP checkpoint — model selection uses the test set. This is the convention
across this benchmark (RePro and MMP do the same), so it is not a deviation, but
it should be stated explicitly in any write-up rather than left implicit.

## 6. What we need decided

1. **Is training in scope for the thesis at all**, or is the contribution the
   diagnostic analysis (the reproduction, the stride finding, the refutation of
   H2, the frame-shuffling probe)? Training is a ~$100 / ~10-day commitment and
   it would be the first thing in this project we cannot iterate on quickly.
2. **If yes, what is the experiment?** "Fine-tune again and get a similar
   number" is not a result. A training run needs a hypothesis attached —
   something the pilot surfaced that a step-4 delta could actually test.
3. **Should the three loop fixes land first regardless?** They are cheap, and
   without them no training run is safely restartable. The proposed sequence is:
   fix the loop, rent a 4090 for one hour, time a limited epoch, and convert the
   estimate in §3 into a measurement *before* anything long is launched.

Note the outstanding item that is already queued: the frame-shuffling probe
(~3 h) has not been run yet. It should probably resolve before any training
decision, since its result bears directly on question 2.
