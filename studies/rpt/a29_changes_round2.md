# RPT G1 amendment (A29): your decisions applied, the changed sections

2026-10-28 · repo `ov-vidvrd-lab` · nothing hashed, appended or committed · nothing rented ·
nothing pushed

All of sections 1–5 of your review are applied to `studies/rpt/g1_amendment_draft.md`, and every
settled OPEN label is gone. The chain head is still
`0d62e8380a8b8a9e2361717fc210a351fe4f3aaf4d2119a44c946315d753947b`.

**Sections changed:** change 2; the bars (lead text and table); Definitions items 1, 2, 3, 4 (one
added bullet), 5, 6 and 8; the bullets under the first-hour checks table (check 5); the machine; the
budget; the HELD bullet of "Values G0 found"; "Still OPEN" replaced by "OPEN items" (none).

**Nothing else changed.** I diffed the draft against the version in `a29_for_review.md`, and only
the sections above differ. Matching, unmatched tracks, box style, novel-object recall, item 8 and
the bars table now all use the rule in Definitions, item 2.

## Details I had to fix that your text does not state; please check them

1. **The second tie-break.** You wrote "ties: track id". If one track ties on two runs of different
   tids, track id cannot decide, so I added "then lower run index".
2. **Box-style frames.** The bars table already said "over shared annotated frames", but item 6 said
   "frames where both have a box". With runs, that would include the run's interpolated GT boxes. I
   made item 6 match the table: the run's annotated frames, not its interpolated ones, where the
   track has a box.
3. **Item 8 (raw detections for missed objects).** I moved it to runs as well ("missed run",
   annotated frames of the run), following your instruction that one rule is used throughout.
4. **"Scoreable pairs".** I read this as taking the maximum over ordered track pairs only over pairs
   whose whole tracks overlap ≥ 10 frames, as `gen_labels.py:659` computes it (`end_fid − begin_fid
   ≥ 10`). U1 and U2 are then decided as before.
5. **Counts.** The binding overall-recall bar is over 753 runs (518 base, 235 novel) from 472 tids;
   149 tids have more than one run. Change 2 now gives both figures: "235 of the 753 GT runs; 124 of
   the 472 tids".
6. **Check 5's drop order.** "The option added last" follows the order of checks 1–4.
7. **Two OPEN numberings.** The review's OPEN 3–11 and the draft's later OPEN 1–4 overlap. Labels
   from the later list now read "second-round OPEN n", so that "formerly OPEN 3" (EOV-MMP's tracks)
   is not confused with check 5.

**Facts checked for the new text:**
- `VidVRD_ECC_train.json` has SHA256 `202e7ffe…4646` and holds 783 videos.
- The 3 HELD videos missing from it are the same 3 that are missing from `held_gt.json`.
- `gt_tracks_train.json` covers all 160 HELD videos.

---

### Changed from the proposal: change 2 only

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


### The bars (whole section)

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


### Definitions (items 1–6 and 8; item 7 unchanged)

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
4. **Pair recall**: *unchanged, except one added bullet:*
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
7. *(Box sensitivity: unchanged.)*
8. **Raw [0.1, 0.5) detections for missed objects** (formerly OPEN 10; D1's raw recording).
   - **Which runs:** each GT run that no SAM 3 confident track matches (item 2, class-aware).
   - **Quantity:** the fraction of the run's annotated frames that have a raw detection of the **same
     class** with box IoU ≥ 0.5 against its GT box.
   - **Reported:** the distribution of that fraction (deciles, and the share of runs at 0,
     > 0, ≥ 0.25, ≥ 0.5, ≥ 0.75), base and novel separately, not only "at least one frame".
     G0 found raw detections dense (about 26 per frame, pre-NMS).
   - **Use:** report-only. It decides, in the G2 amendment only, between dropping Verify and
     redefining it on raw detections (review §4).


### First machine hour: the bullets under the checks table

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


### The machine and the budget (both sections)

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


### Values G0 found: the HELD bullet only

- **HELD:** 160 videos, 47,038 frames; 472 GT objects (tids), 348 base and 124 novel. The novel
  ones by class: antelope 29, cattle 23, horse 20, bear 13, watercraft 11, giant panda 10,
  motorcycle 7, fox 7, lizard 2, red panda 2. As option-B runs (item 3): 753 runs, 518 base and 235
  novel (antelope 54, bear 39, cattle 35, watercraft 23, horse 23, giant panda 20, red panda 16,
  fox 14, motorcycle 7, lizard 4). Base GT relations: 4,365 in `held_gt.json` (157 videos), plus
  those of the 3 videos built under item 4.


### OPEN items (replaces "Still OPEN")

##### OPEN items

None. The second-round OPEN 1–4 of the draft are settled above: the GT unit and matching rule
(Definitions, items 2 and 3), camera compensation (item 1), the combined-options check (check 5)
and the stack record (the machine).

---

Stopped here: no hashing, appending, committing or renting. I'm waiting for your go-ahead to hash.
