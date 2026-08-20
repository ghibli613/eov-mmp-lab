# How the model works, and how it is trained

Everything here is read off the EOV-MMP paper (arXiv 2409.12499v2, TPAMI 2025)
and the code in this repository. Where the two disagree, that is noted — those
disagreements are real and matter if you report numbers.

---

## 1. The task

Given a video, output `<subject, predicate, object>` triplets — for example
`<dog, run_behind, person>` — where the subject and object are **trajectories**
(a box per frame), not single boxes.

*Open-vocabulary* means some object categories and some predicate categories are
**never seen in training** and must still be detected. VidVRD splits categories
into `base` (seen) and `novel` (unseen); you train on `base` and evaluate on both.

---

## 2. Four questions, four models

Producing one triplet requires answering four separate questions. EOV trains a
model for each, then joins them.

| | Question | Component in `cli/train.py` | Checkpoint it loads |
|---|---|---|---|
| 1 | *Where* are the objects? | `object_detection_model` | `checkpoint_vidvrd0059_new_1e-5.pth` |
| 2 | *What* is each object? | `object_classifer` | `vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth` |
| 3 | *What relation* holds between a pair? | `relationship_classification_model` | `baseline_fbce_..._stage2_new_L14_e2e.pth` |
| 4 | Which detections are *the same object over time*? | `sort_model` (AFLink) | `--path_AFLink` |

### 2.1 The trajectory detector

A query-based Transformer decoder — Deformable DETR — in
[`models/methods/detectors/ov_prompt`](../models/methods/detectors/ov_prompt).
Six layers, 300 object queries. It localises objects frame by frame; it does not
name them. The visual encoder of CLIP is **distilled** into it, which is what
gives it a chance at novel categories.

It also carries a relationship classification head, so relationship context can
inform detection — the "relationship-aware" part of the paper's title.

### 2.2 The object classifier

[`models/model.py`](../models/model.py), class `Classifier`. This is the
open-vocabulary trick. Instead of a fixed softmax over 35 classes, it classifies
by matching a region's features against **CLIP text embeddings of class names**:

```python
prompts = [f"An image of {name}." for name in classnames]
text_embeddings = model.encode_text(prompts)
```

Because the classifier weights are just text embeddings, you can hand it a class
name at test time that it never trained on. That is what makes the vocabulary
open.

"Vision-guided" means the prompt is **conditioned on the region's visual
features** (`self.text_encoder(visual_feats)`) rather than fixed — the CoCoOp
idea. Eight learnable continuous prompt tokens plus eight learnable conditional
tokens, with the `[OBJ]` token at the end of the sequence.

### 2.3 The relationship classifier

[`models/model_zoo/model_tuing_plus_repro_copy_new_cross_dataset.py`](../models/model_zoo).
Takes a **pair** of object trajectories and predicts the predicate. Same
prompting scheme — eight continuous plus eight conditional tokens — but the
`[REL]` token sits at **75% of the token length** rather than the end.

The features it consumes are named in its checkpoint filename:
`rel` (relative geometry), `mot` (motion), `clip` (CLIP visual), `bbox`.

### 2.4 The tracker

deep_sort plus **AFLink** in [`models/tracking`](../models/tracking). Links
per-frame detections into trajectories. AFLink is a learned post-linker; note it
was trained on **MOT17**, a different dataset, and is the one component with
public training code ([`models/tracking/aflink/train.py`](../models/tracking/aflink/train.py)).

---

## 3. The four training steps

From the paper, Section III-D:

> "We adopt a four-step scheme for training."

| Step | Trains | On | Loss |
|---|---|---|---|
| 1 | Transformer decoder + prediction heads | frames with frame-wise object **and relationship** annotations | `L_o` |
| 2 | auxiliary object classifier | frames with **ground-truth boxes** | `L_o_cls` |
| 3 | open-vocabulary relationship classifier | videos with **ground-truth trajectories** | `L_r` |
| 4 | the whole framework, jointly | end-to-end | `L_o + L_r` |

**`cli/train.py` is step 4 only.** It is fine-tuning, not from-scratch training.
The three `load_state_dict` calls at the top are loading the outputs of steps
1–3. This is why it cannot start from nothing: the joint objective has no signal
to refine if the components are random.

### Hyperparameters (paper, Implementation Details)

| Step | lr | Epochs | Batch | Init |
|---|---|---|---|---|
| 1 | 1e-5 | 10 | 16 | **MS-COCO pretrained**, novel categories removed; relation head random |
| 2 | 1e-3 | 5 | 12 | — |
| 3 | 1e-4, ×0.1 at 15/20/25 | — | 32 | — |
| 4 | 1e-5 | 5 | 1 | steps 1–3 |

Also: frames sampled every 30; CLIP ViT-L/14 **frozen**; τ = 0.01; box filter
threshold ε = 0.35.

**Step 1 is not from scratch either** — it starts from MS-COCO Deformable-DETR
weights, which are publicly available. So the detector is not a mystery
artifact; what is missing is the training loop, not the initialisation.

---

## 4. What is publicly available

Upstream's README says the download link contains "the datasets and models
obtained from the first three training steps", and that "the complete training
and testing code will be released after the paper is officially published". Its
To-Do list still has *Object Detection Training* and *Relationship Detection
Training* open.

Confirmed by inspection: the only training loops in the EOV repository are
`train/train.py` (step 4) and `AFLink/train.py` (the tracker).

**So steps 1–3 have no released code.** That, not the broken download link, is
the real reason you cannot train from scratch here.

### What could be trained anyway

[MMP](https://github.com/wangyongqi558/MMP_OV_VidVRD) — same authors, public,
and what EOV builds on — *does* have complete training code for the relationship
side, and its checkpoint load is conditional:

```python
# MMP_OV_VidVRD/scripts/train.py:39
if args.resume or args.stage2:
    ckpt = torch.load(args.ckpt_path)
```

Without `--stage2` it trains from nothing. MMP's own two stages (train the
spatio-temporal module with hand-written prompts, then freeze the vision side
and train the vision-guided prompt module) correspond to EOV's steps 2 and 3.

MMP also runs on **pre-extracted features** rather than encoding frames live,
which makes it far lighter — see [known-issues.md](known-issues.md#2-vram).

---

## 5. Where the code and the paper disagree

Worth knowing before you report anything.

| | Paper | This code / the checkpoint names |
|---|---|---|
| Step 1 epochs | 10 | filename says `0059` (epoch 59) |
| Step 2 lr | 1e-3 | filename says `lr0.01` |
| Step 3 batch | 32 | filename says `bs1` |
| Step 4 epochs | 5 | `run_train.sh` sets `max_epoch=20` |
| Step 4 schedule | — | `MultiStepLR(milestones=[15,20,25])`, which is *step 3's* schedule |

Step 3's learning rate is the one that does match: paper 1e-4, filename
`lr0.0001`.

Also note the `stage2` inside the relationship checkpoint's filename is **MMP's**
internal stage numbering, not EOV's step numbering. Two schemes in one filename.
