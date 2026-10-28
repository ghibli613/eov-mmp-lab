# DRAFT — not hashed, not in the chain — for review

#### A29 (draft): RPT G1 — SAM 3's tracks against EOV-MMP's on the 160 HELD videos

Label: A29 would be the next free one (A27 is reserved; A28 and A28a are G0's). The chain head
stays `0d62e8380a8b8a9e2361717fc210a351fe4f3aaf4d2119a44c946315d753947b` until this is hashed.

**When this was written, and what had been seen.** It was written after G0
(`studies/rpt/g0_report.md`), before any G1 run. It applies the plan author's answers to
`studies/rpt/g0/runtime_proposal.md` and the review `studies/rpt/rpt-review-g1.md`.

- **G0's 3 smoke videos are in G1.** `ILSVRC2015_train_00008005`, `…00010024` and `…00024005` are 3
  of the 160 HELD videos and of G1's first 20. The definitions below were written after seeing
  SAM 3's output on them. Every bar is judged on all 160, as pre-registered.
- **The other 157 HELD videos:** 45,863 frames (median 175, max 2,947, 3 over 1,000); 465 GT objects
  (346 base, 119 novel) of 34 classes; 4,297 base GT relations in `held_gt.json` (154 of them).

**What G1 reads.** G1 measures the tracker. No relation score on HELD is read, except the secondary
box-sensitivity hit rates, which compare box sources on the same in-sample model. EOV-MMP was
trained on HELD, so its relation scores there are in-sample.

##### Changed from the proposal (before any G1 data; reasons measured or documented)

**1. The runtime bar becomes a cost stop.**
- **Superseded text.** Proposal §6, G1 bars: *"**Runtime.** At most 1 GPU-hour per 20 videos;
  otherwise the pre-registered every-*k*-frames fallback."* Proposal §3.1: *"**Runtime fallback
  (pre-registered).** If G1's runtime bar fails, text detection runs every *k* frames, with SAM 3's
  tracker propagating in between. *k* is fixed in the amendment and never tuned on results."*
- **Reason.** G0 §9 measured SAM 3 at its defaults at 0.094 s per frame per prompt on an RTX 4090.
  That projects 43 GPU-hours for the 160 HELD videos, 5.4× the bar of 8 GPU-hours.
- **Now.** Runtime is **not a pass/fail bar of G1**. It is only a cost stop: a cap of
  **35 GPU-hours**, defined under "The cost cap" below. The every-*k* fallback and options B1–B5 of
  the runtime proposal are not used. SAM 3 runs at full fidelity: every frame, all 35 prompts,
  default settings. Only the implementation options A1–A3 and `offload_video_to_cpu` may be used,
  each only if bitwise identical to G0's output.
- **Binding consequence for the test session.** It runs SAM 3 at exactly the settings G1 validates:
  full fidelity, plus only the options that passed bitwise here. Otherwise G1 would not have
  validated the configuration that is tested.

**2. The novel-object recall bar becomes report-only.**
- **Superseded text.** Proposal §6: *"**Recall.** GT track recall at vIoU 0.5 at least 10 points
  above EOV-MMP's, overall and on novel object classes (when ≥ 30 instances)."* Only the novel part
  changes.
- **Reason.** On HELD, EOV-MMP's detector is not only in-sample. Its step-4 training targets included
  the boxes, class ids and text queries of every training trajectory, novel classes included: leaks
  T1 and T2 of RB1's training-target audit (`studies/prereg/preregistration.md:60-61`, A23). Its
  novel-object recall on HELD is therefore inflated twice, and a 10-point bar against it could stop
  RPT for a reason that has nothing to do with the test set.
- **Now.** Novel-object recall is reported in full, per class too, with no threshold. The overall
  recall bar, which includes the novel GT objects (235 of the 753 GT runs; 124 of the 472 tids), and
  the pair-recall bar stay binding.
- The review marked this the author's decision. **The plan author confirmed it explicitly before any
  G1 data.**

##### The bars

EOV-MMP's tracks are defined in "Definitions", item 1. The GT unit is the GT run (item 3), and
"matched" means matched under item 2's rule, class-aware unless stated. The same rule serves overall
recall, novel-object recall, unmatched tracks, box style and item 8, for SAM 3 and EOV-MMP alike.

| bar | quantity measured | threshold | status |
|---|---|---|---|
| **Overall recall** | fraction of the 753 HELD GT runs (item 3) matched by a SAM 3 confident track (item 2), minus the same fraction for EOV-MMP's tracks (R0w) | **≥ 10 points** | **binding** |
| **Pair recall** | fraction of HELD base GT relations (item 4) that D1's U1/U2 code recalls with SAM 3's confident tracks, minus the same for EOV-MMP's (R0w) | **≥ 10 points** | **binding** |
| **False-track burden** | mean over the 160 videos of unmatched confident tracks per video (item 5), SAM 3 ÷ EOV-MMP (R0w) | **≤ 1.5×** | **binding** |
| **Box style** | median over matched SAM 3 track and GT run pairs (item 2) of the mean per-frame box IoU over shared annotated frames (item 6) | **≥ 0.70** | **binding** |
| **Box sensitivity** | top-5 hit rate of the GT predicate (item 7) with SAM 3 boxes, minus the same with EOV-MMP's boxes | **≥ −5 points** | **secondary**: a fail does not stop the line; it moves v2 before the test (proposal §7) |
| Novel-object recall | overall recall restricted to the 235 novel GT runs (124 tids); per class too | none | **report-only** (change 2) |
| Recall per tid | a tid is recalled if any of its runs is recalled (item 3); 472 / 348 / 124 tids (all / base / novel), beside the run counts | none | report-only |
| Pair recall, scoreable pairs | pair recall with the best pair taken only over track pairs that overlap ≥ 10 frames (item 4); SAM 3 and EOV-MMP | none | report-only |
| Recall, confident plus candidates | as overall recall, with candidate-tier tracks added | none | **report-only**; identical to confident-only by construction (erratum) |
| Class-agnostic counts | every class-aware count above, class-agnostic (item 2) | none | report-only |
| Gap exposure | clips whose middle frame is a *filled* frame of its subject or object | none | report-only |
| Track counts by *N* | tracks per video, binned by confident-track count *N*: 0, 1, 2, 3–4, 5–8, ≥ 9; SAM 3 and EOV-MMP | none | report-only |
| Raw detections for missed objects | item 8 | none | report-only; input to the G2 amendment |

A fail of a binding bar stops the line (proposal §6, fallback 2). A pass only allows G2.

##### Definitions

1. **EOV-MMP's tracks** (formerly OPEN 3).
   - **What:** EOV-MMP's **final** tracks, after its own filters, i.e. the tracks its relation
     stage sees. That is the `trajectories` list after `format_trajectories_test` and the 15% rule,
     as passed to `gen_feats_test` and kept in `self.traj` (`models/end2end_model.py:314-325`).
   - **Primary: R0w,** the whole video (A26's `studies/a26/wrappers.py::whole_video`), matched to
     SAM 3's whole-video search. **R0, the GT-derived range, is reported beside it.**
   - **Leak note:** EOV-MMP's detector was trained on HELD (T1/T2), so its HELD recall is inflated.
     That makes the binding bars conservative for SAM 3.
   - **Camera compensation** (the draft's second-round OPEN 2, settled by the plan author):
     - **The problem.** The released test path gives the tracker the **test** ECC file
       (`end2end_model.py:277`). No HELD video is in it, and a video absent from the file is tracked
       without camera compensation (`tracker.py:55-57`, `end2end_model.py:19-40`).
     - **The fix.** The model's `test_ecc` attribute is set to the contents of the train ECC file,
       `data/vidvrd/data/VidVRD_ECC_train.json` (SHA256
       `202e7ffed90667c4250bf8a0490f20b22b4c7ec620f0cdabb58e0354795d4646`), with no code edited.
       EOV-MMP's HELD tracks then get the same compensation step its test tracks get.
     - **Three HELD videos are absent from that file:** `ILSVRC2015_train_00265063`,
       `ILSVRC2015_train_00679000` and `ILSVRC2015_val_00123004`. They run without compensation, as
       the released code does for missing videos.
     - The front end is fed HELD frames and training-annotation ranges only, never
       `Dataset_new('val')`, which reads test files, under the test-label guard.
2. **Matching a track to a GT run** (formerly OPEN 4; the rule set with the second-round OPEN 1).
   The same rule applies to SAM 3's and EOV-MMP's tracks.
   - **Cut-vIoU.** A track is compared with a run after being **cut to that run's frames**, by D1's
     `cut_viou` convention (`studies/d2/stages.py:147-159`). For a run over [gb, ge) and a track
     over [begin, end), the track is cut to [max(gb, begin), min(ge, end)). The cut-vIoU is the
     official `viou` (`inference/video_relation_detection_openvoc.py:73-114`) of the cut boxes
     against the run over [gb, ge), or 0 if the cut is empty. A match needs cut-vIoU ≥ 0.5.
   - **Assignment.**
     - Candidate (track, run) pairs with cut-vIoU ≥ 0.5 are taken greedily in descending cut-vIoU
       (ties: lower track id first, then lower run index).
     - A pair is accepted if its run is not yet matched, and its track is either not yet matched or
       matched only to runs of the same tid.
     - So **each run is matched by at most one track**, and **one track may match several runs of
       the same tid, but runs of only one tid.**
   - **Class-aware** for the bars: the track's class name equals the run's.
   - **Class-agnostic:** the same rule, ignoring names. Reported beside every class-aware count.
   - **Reason for the cut.** Without it, an unbroken SAM 3 track that bridges a gap of more than 30
     frames would fail vIoU against a single run just for being longer. That would penalise SAM 3 for
     keeping identity through occlusion, while EOV-MMP's tracks, which break at such gaps, would not
     be penalised.
3. **The GT unit: session 3's option-B run** (the draft's second-round OPEN 1, settled by the plan
   author). The unit for overall recall, novel-object recall, unmatched tracks, box style and item 8.
   - **Source:** `pilot_analysis/stage1c/gt_tracks_train.json` (SHA256
     `87b40fff514178f77a57d9e142bbbb9363eef34f6b4c22d8eaa88d4c7612636d`, pinned in A28). Each tid is
     split where more than 30 consecutive frames lack a box, and shorter gaps are interpolated.
   - **On HELD:** 753 runs (518 base, 235 novel) from 472 tids. 149 tids have more than one run, at
     most 8.
   - **Why not whole tids.** The official `viou` needs a dense trajectory over a contiguous span. Many
     training tids have unannotated frames inside their span (session 3 found 33.5% of train tracks
     under 65% coverage).
   - **Reported per run:** the binding overall-recall bar, box style and unmatched tracks.
   - **Reported per tid, report-only:** a tid is recalled if any of its runs is recalled. The tid
     counts (472 / 348 / 124) are reported beside the run counts (753 / 518 / 235).
4. **Pair recall** (formerly OPEN 5): D1's U1/U2 code, unchanged.
   - **D1** (`pilot_analysis/time_check/tc_run.py:232-250`). For a GT relation over [gb, ge), each
     track is cut to [max(gb, begin), min(ge, end)), and its vIoU is the official `viou` of the cut
     boxes against the GT subject (or object) trajectory over [gb, ge); 0 if the cut is empty. The
     pair score is the maximum over ordered track pairs s ≠ o of min(subject vIoU, object vIoU).
     Below 0.5 is **U1**. Otherwise, if the best pair's names differ from the GT triplet's, it is
     **U2**.
   - **The importable copy, pinned in A25a:** `studies/d2/stages.py:147-159` (`cut_viou`,
     `best_viou`, "D1's convention") and `studies/d2/analyse.py:142-148` (`best_pair_min`, "D1's U1
     quantity").
   - **A relation is recalled** (binding) if it is neither U1 nor U2. Class-agnostic recall (not U1)
     is reported beside it. As in D1, the pair rule (overlap ≥ 10 frames) is not part of the bar.
   - **Report-only, scoreable pairs:** the same recall, with the maximum over ordered track pairs
     taken only over pairs whose two tracks overlap ≥ 10 frames, as the pair rule computes it
     (`models/gen_labels.py:659`). Only such pairs are ever scored by the relation head. It is
     reported for SAM 3 and EOV-MMP, so the share of recalled relations that can raise the final
     score is visible.
   - **The relations:** HELD's base GT relations, `pilot_analysis/stage2/data/held_gt.json` (SHA256
     `78d55c69…4dee`, base only; the novel-predicate instances stay unused). It covers 157 of the 160
     videos, because stage 2 dropped the 3 without ECC: `…00265063`, `…00679000`,
     `ILSVRC2015_val_00123004`. For those 3, the same construction (`pilot_analysis/stage2/
     step1_parts.py:40-53`, base relations only) is applied to their training annotations, and
     the result is hashed before any G1 run.
5. **Unmatched confident track** (formerly OPEN 7): a confident track that matches no GT run under
   item 2's rule (class-aware). Real but unannotated objects, such as the distant person in G0,
   count for both trackers, so the bar stays relative. **The class-agnostic unmatched count is reported
   beside it**: a well-placed but misnamed track is unmatched in the bar, and the split shows how
   often.
6. **Box style** (formerly OPEN 8).
   - **Per pair:** for each matched SAM 3 track and GT run pair (item 2), the mean per-frame `iou`
     (`video_relation_detection_openvoc.py:48-70`) over the run's annotated frames (not its
     interpolated ones) where the track has a box.
   - **The bar** is the median over pairs.
   - GT and SAM 3 boxes are drawn on the middle annotated frame of each of the first 20 videos, for a
     visual check.
7. **Box sensitivity** (formerly OPEN 8; this reads modelC scores on HELD).
   - **Which relations:** HELD base GT relations that both SAM 3's and EOV-MMP's tracks recall
     (item 4).
   - **How:** per relation, the pair's clips are built by `gen_feats_test` over the relation's span
     three times: GT boxes, SAM 3's recalled pair cut to the span, and EOV-MMP's likewise. modelC
     runs each (`tgt_split all`, session 3's `run_pair`).
   - **Hit:** a clip is a hit if the GT predicate is among the top 5 **base** predicates of its
     score, `sigmoid(logit) × sigmoid(interactiveness)`. The hit rate is hits ÷ clips per box source.
8. **Raw [0.1, 0.5) detections for missed objects** (formerly OPEN 10; D1's raw recording).
   - **Which runs:** each GT run that no SAM 3 confident track matches (item 2, class-aware).
   - **Quantity:** the fraction of the run's annotated frames that have a raw detection of the **same
     class** with box IoU ≥ 0.5 against its GT box.
   - **Reported:** the distribution of that fraction (deciles, and the share of runs at 0,
     > 0, ≥ 0.25, ≥ 0.5, ≥ 0.75), base and novel separately, not only "at least one frame".
     G0 found raw detections dense (about 26 per frame, pre-NMS).
   - **Use:** report-only. It decides, in the G2 amendment only, between dropping Verify and
     redefining it on raw detections (review §4).

##### The cost cap

- **GPU-hours** are billed machine hours, wall clock × GPUs rented, not summed process time. With
  A2's concurrent processes, billed time is what counts.
- **The cap covers SAM 3:** billed hours from the start of the first-hour checks to the moment the
  last of the 160 videos' SAM 3 files is written. Set-up, the token fetch, EOV-MMP's HELD tracks and
  copy-back count in the budget, not in the cap.
- **Hard stop:** if the actual SAM 3 hours reach **35** before all 160 videos are done, stop and ask
  the author.
- **Projection after the first 20 videos** (frame-weighted, not ×8 of the first 20): hours spent so
  far + Σ over the 140 remaining videos of frames × 35 prompts × *r*(video). *r*(video) is the billed
  seconds per frame per prompt measured for that video's settings:
  - **videos of ≤ 1,000 frames:** the rate over the first 20, all of which are ≤ 1,000 frames (3,371
    frames, max 981), with the options that passed;
  - **the 2 videos of 1,001–2,300 frames** (`…00896000`, 1,080; `…00237001`, 1,470), which run alone
    without A1: the default single-process rate from check 0;
  - **`…00280000`** (2,947 frames), alone, with `offload_video_to_cpu`: the rate from its check.
  - If the projection exceeds **35**, stop and ask the author.
  - Also stop and ask if any of the first 20 videos fails to run (formerly OPEN 9). Every bar is
    judged on all 160, never on the first 20 alone.

##### G1's first machine hour: the checks

All on smoke video 1, `ILSVRC2015_train_00008005`. Each run is compared with
`studies/rpt/g0/sam3_compare.py` against G0's default output
`studies/rpt/g0/sam3_out/ILSVRC2015_train_00008005.json.gz` (SHA256 `aed5945d…05db4`): every prompt,
masklet, per-frame field and raw detection. **Pass means bitwise identical.**

| order | run | pass | not identical |
|---|---|---|---|
| 0 | the default run, as G0 | the machine reproduces G0 | **stop and ask**: nothing else can be judged |
| 1 | A1, backbone reuse | used for videos ≤ 1,000 frames | not used; the difference recorded |
| 2 | A2, 2 then 3 processes at once; each process's output compared; throughput measured | the largest passing count is used, within the scheduling rule | one process at a time; the difference recorded |
| 3 | A3, `torch.compile` (SAM 3's `compile=True`) | used | not used; the size of the difference reported: max \|Δ\| of `out_probs` and box coordinates, mask IoU, and masklets added or lost |
| 4 | `offload_video_to_cpu=True` | used for `…00280000` | `…00280000` is not run on this GPU; **stop before it and ask the author**. The fallback is a 48 GB GPU for that video, at defaults. |

- Each option is checked on its own against G0, in this order.
- An option that is not identical is never used; G1 continues without it, except at checks 0 and 4
  as stated.
- **Check 5, the passing options together** (second-round OPEN 3, settled by the plan author).
  - After checks 1–4, one more run on smoke video 1 with all passing options together, bitwise
    against G0.
  - On a mismatch, drop the option added last (order 1–4) and repeat. Stop when a combination is
    identical, or when none is left; then single-process defaults are used.
- The first 20 videos then run with the combination that passed check 5. Every attempt of every
  check goes in the G1 report, with its outcome, time and GPU memory.

##### Scheduling (identical output; fixed now, not tuned)

- **A1** keeps about 106 MB of CPU RAM per frame, so it is used only for videos of ≤ 1,000 frames.
  Concurrent A1 caches stay below about 150 GB in total.
- **GPU memory.** A process's predicted peak is 6.5 GiB + 6.3 MiB × frames (G0 §9: 7.3 GiB at 130
  frames, 12.8 GiB at 981). Concurrent processes are scheduled so that the predicted peaks sum to
  **under 21 GiB**.
- **Long videos run alone.** Videos over 1,000 frames run alone, without A1. Only `…00280000`, whose
  predicted peak (about 24.6 GiB) exceeds 21 GiB, uses `offload_video_to_cpu`.

##### The machine

- **One RTX 4090,** pinned to G0's stack so that check 0 does not fail on stack drift:
  - NVIDIA driver 580.142; the Vast.ai PyTorch template's `/venv/main` with Python 3.12.15 and torch
    2.11.0+cu128 (CUDA 12.8) for the EOV-MMP side;
  - SAM 3's own venv with torch 2.10.0+cu128;
  - both package sets equal to G0's `pip freeze`: `studies/rpt/g0/logs/box/pip_freeze_main.txt`
    (SHA256 `ed46e4da15a01b48c6c53cbd14fe9a5710a473e51a20b1a225c22063e6250613`) and
    `pip_freeze_sam3env.txt` (`42206283b1e237296b22d7df22d797860f5cfb2501f2bfb79d04dc491ffe58b1`).
  - **Image digest and cuDNN** (second-round OPEN 4, settled by the plan author): G0's logs did not
    record them, so they cannot be pinned. G1 rents the same Vast.ai PyTorch template as G0 and records the image digest and the
    cuDNN version at setup. Check 0, the default run reproducing G0 bitwise, decides whether the
    stack matches; if it fails, stop and ask.
- **≥ 192 GB RAM,** for A1's caches.
- If no such machine is available at about the budget's price, ask the plan author before renting
  anything else.
- No test file on the machine; the token as in A28. Nothing is pushed.

##### Budget (approved by the plan author)

- **Up to about $15 in total,** at about $0.35/h:
  - at most **35 GPU-hours** for SAM 3 (the cap);
  - about **3 GPU-hours** for EOV-MMP's HELD tracks, R0w and R0;
  - plus set-up and copy-back.

##### G1's SAM 3 outputs are kept

Every HELD video's SAM 3 file (the `sam3_runner.py` output) is kept, hashed into a manifest and
committed, or force-added if git-ignored, for reuse in G2. G2 then pays only for the loop's extra
prompts, not another full pass.

##### Values G0 found (fixed; A28, A28a, G0 report)

- **SAM 3:**
  - code `f1647e8d0c1d280991f4d1b7d974c2704537fe66`;
  - `sam3.pt` `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` (`facebook/sam3`
    revision `3c879f39…`, not SAM 3.1);
  - `build_sam3_video_predictor()`, every default; SAM 3's own multi-prompt protocol; 35 prompts, `_`
    replaced by a space.
- **Tiers:** confident cut **0.5** (SAM 3's `score_threshold_detection`); candidate floor **0.1**.
- **Track score:** the mean of `out_probs` over detected frames, which equals the starting
  detection's probability.
- **Erratum to A28.** A28's rule 2 says SAM 3's *"masklet is started only by a kept detection ≥ 0.7
  (`new_det_thresh`) that matches no existing masklet"* and that the track score *"equals the
  probability of the masklet's starting detection (≥ 0.7)"*. Rule 5 says *"at SAM 3's defaults every
  masklet starts from a detection ≥ 0.7"*.
  - **Corrected.** A detection is kept only if its probability is strictly above 0.5:
    `pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)`
    (`sam3/model/sam3_video_base.py:608`; NMS uses the same `>`, `sam3/perflib/nms.py:42`). The
    ≥ 0.7 test (`:211`) applies only when masklets already exist. On a frame with none, such as
    frame 0 of every prompt, every kept detection starts a masklet (`:1522-1535`). G0 observed
    starting scores of 0.594 and 0.668.
  - **The consequence.** A track's score equals its starting detection's probability, and masklets
    start only from detections above 0.5. So **the candidate tier [0.1, 0.5) is empty by
    construction at SAM 3's defaults**, on every video, not only on G0's 3.
  - No rule changes.
- **Gaps and duplicates:**
  - one SAM 3 ID is one track;
  - internal gaps are filled by `gen_labels.py:416-423`; no pre-roll;
  - cross-prompt duplicates at mean mask IoU > 0.7 keep the higher-scoring masklet unchanged.
- **Relation side and recording:**
  - pairs, windows and features as A28, the adapter as committed (`studies/rpt/adapter.py`,
    `relation.py`, `sam3_runner.py` at `b249c1d`);
  - raw [0.1, 0.5) detections are recorded read-only.
- **Measured on a 4090:**
  - **Speed:** 0.094 s per frame per prompt at defaults (0.09 when a prompt finds nothing, 0.12–0.18
    when it does).
  - **Memory:** 6.5 GiB + about 6.3 MiB per frame.
  - **Backbone reuse:** bitwise identical, 1.36× faster.
  - **Determinism:** a rerun is bitwise identical.
- **HELD:** 160 videos, 47,038 frames; 472 GT objects (tids), 348 base and 124 novel. The novel
  ones by class: antelope 29, cattle 23, horse 20, bear 13, watercraft 11, giant panda 10,
  motorcycle 7, fox 7, lizard 2, red panda 2. As option-B runs (item 3): 753 runs, 518 base and 235
  novel (antelope 54, bear 39, cattle 35, watercraft 23, horse 23, giant panda 20, red panda 16,
  fox 14, motorcycle 7, lizard 4). Base GT relations: 4,365 in `held_gt.json` (157 videos), plus
  those of the 3 videos built under item 4.

##### Videos

The 160 HELD videos of `pilot_analysis/stage1/train_split.json` (SHA256 `095dd89c…1f1a`; seed
20260928), sorted. All 160 are used. A bare number below is `ILSVRC2015_train_` + that number (142
videos); the last 18 are ImageNet-VID `val` videos that belong to VidVRD's training split, and are
written in full. The first 20 IDs are G1's "first 20".

```
00008005 00010024 00024005 00025029 00027003 00032000 00040000 00040005
00040030 00050006 00051015 00051016 00052001 00055002 00058003 00060001
00060005 00065002 00065003 00069006 00071013 00071014 00071019 00073037
00074010 00077006 00078000 00097003 00112000 00116002 00119013 00119025
00125006 00125007 00125013 00125014 00135004 00139004 00146005 00149004
00154002 00159010 00161016 00165000 00171001 00171004 00172007 00175001
00181011 00185000 00209003 00209005 00215000 00215006 00215010 00218002
00225001 00227012 00228000 00234001 00234007 00235006 00235007 00237001
00246002 00253025 00265004 00265042 00265063 00265066 00265068 00265074
00270000 00272005 00272034 00280000 00284001 00290008 00290019 00293004
00293005 00300008 00302021 00303008 00308017 00309001 00311002 00339001
00339003 00349000 00353000 00354000 00356001 00371013 00374000 00398004
00407001 00415007 00415008 00435001 00467000 00476002 00477000 00480000
00494000 00532000 00533003 00572000 00588000 00596000 00632001 00636001
00667002 00679000 00683000 00691000 00721000 00750000 00756000 00759000
00760001 00766000 00772004 00780000 00796000 00873000 00882000 00896000
00897005 00912002 00917000 00971000 00978000 00987000 00988001 00989000
01014000 01020000 01025001 01034001 01079000 01108001 ILSVRC2015_val_00007022 ILSVRC2015_val_00008001
ILSVRC2015_val_00020001 ILSVRC2015_val_00026003 ILSVRC2015_val_00027000 ILSVRC2015_val_00028003 ILSVRC2015_val_00035000 ILSVRC2015_val_00035008 ILSVRC2015_val_00035013 ILSVRC2015_val_00037004
ILSVRC2015_val_00046000 ILSVRC2015_val_00078000 ILSVRC2015_val_00082000 ILSVRC2015_val_00103000 ILSVRC2015_val_00123004 ILSVRC2015_val_00135000 ILSVRC2015_val_00150003 ILSVRC2015_val_00169000
```

##### Seeds

- SAM 3: `torch.manual_seed(0)`; its inference draws no random numbers, and G0's rerun was bitwise
  identical.
- The HELD split: seed 20260928 (fixed, stage 1).
- Bootstrap CIs, descriptive only, never a bar: `numpy.random.default_rng(0)`, 1,000 resamples over
  the 160 videos, the same draws for SAM 3 and EOV-MMP.
- No other randomness.

##### OPEN items

None. The second-round OPEN 1–4 of the draft are settled above: the GT unit and matching rule
(Definitions, items 2 and 3), camera compensation (item 1), the combined-options check (check 5)
and the stack record (the machine).

**Not changed by this amendment:** anything in A28 and A28a other than the erratum, and the two
changes listed under "Changed from the proposal". Any change to the above is a new amendment,
written before the result it affects.
