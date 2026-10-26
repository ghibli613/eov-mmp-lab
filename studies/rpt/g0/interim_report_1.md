# RPT G0: interim report 1 (stopped before step 5)

2026-10-26 · repo `ov-vidvrd-lab`, HEAD `bc9a87b` · no commits yet · nothing governed by the G0 amendment has run

> **Update, 18:15.** The author ran the fetch script and Hugging Face answered **403**: the
> author's account is not yet on `facebook/sam3`'s access list. The author has to request access
> on the model page, and Meta approves it by hand, which can take hours or days.
>
> The script failed cleanly: the token was unset, no file holds it, and no checkpoint was written.
>
> No unofficial mirror will be used: its provenance could not be pinned.
>
> **Option for you, if the wait is long:** A28 could pin the checkpoint by Hugging Face revision
> (`3c879f39…`) and size (3,450,062,241 bytes) now. Its SHA256 would follow in a pin addendum, A28a,
> before any SAM 3 code touches data. A28 could then be hashed now, and the parts that never use
> SAM 3 could go ahead:
> - the adapter's unit tests;
> - step 8, the reproduction of session 3 (on a 4090).
>
> This departs from the G0 prompt (step 5 pins the checkpoint hash), so it is your call. Without
> it, G0 waits for Meta.

G0 is stopped before step 5 for two reasons:
1. **The checkpoint hash.** A28 must pin `sam3.pt`'s SHA256, and `facebook/sam3` is gated. Only the author's token can fetch it. The fetch script is ready and waiting for the author.
2. **Your decision.** Step 4 found several places where the proposal cannot be followed as written. Ground rule 6 says to stop and report, not improvise. G0 can continue on the readings below, each flagged OPEN, but A28 will pin them. Please confirm or change them **before A28 is hashed**.

## 1. Status by step

| step | status | evidence |
|---|---|---|
| 1. paths and chain | **all match** | `studies/rpt/g0/logs/step1_checks.txt` |
| 2. licence | **research use and publication allowed** | §4 below |
| 3. chain checker | **fixed; 7 tests pass; real chain passes (66 checks)** | `studies/rpt/g0/logs/verify_after_A26m_fixed.txt` |
| 4. code checks | **all five answered**, with consequences (§2, §3) | A28 draft (appendix) |
| 5. A28 | drafted, **not hashed**: waits for the checkpoint hash and your reply | appendix |
| 6. SAM 3 environment | installed and pinned; memory probe done; laptop check waits for the checkpoint | §5 |
| 7. adapter and unit tests | **written, not run** (the amendment comes first) | `studies/rpt/{adapter,relation,sam3_runner}.py`, `studies/rpt/tests/test_adapter.py` |
| 8–9 | scripts written, not run; a 4090 will almost surely be needed | §6 |
| 10. G1 draft | not started | |

Step 1 details:
- The authors' files are everything at `fdeeeb4` outside `pilot_analysis/`; `git diff fdeeeb4` over them is 0 lines, in both HEAD and the working tree.
- Present: `cache_wrapper.py`, `gt_tracks.py`, `train_split.json`, and `cache_s3` (800 train and 200 test tensors, at `../session3_box_copy/`). The 5 pinned `cache_s3/train` files match `SHA256SUMS_session3.txt`.
- The chain head is `ac77e243…a591`.
- A26's tests pass: 30 passed, run from `studies/`.
- The bundle exists (`eb7417df…85a7`, master `a820a0e`).

Step 3: the old checker printed nothing for the ov-pairtime check once the folder was gone (65 checks instead of 66) and still exited 0. Now it reads `prereg/preregistration.md` at `a820a0e` from the bundle through a throwaway bare repo. A missing or unreadable bundle, or a different hash, is a FAIL with exit 1. Only that check changed. New checker SHA256: `3af183caad77d61ea4aca7e7fa04d2da74c354b0d4335797ede98409d0eda1e7`.

## 2. Decisions needed before A28 is hashed

### D1. The candidate tier is empty at SAM 3's defaults (the most important finding)

- **Check 5:** SAM 3 keeps a detection if its probability is > **0.5**. The sources:
  - the video model's `score_threshold_detection`, documented "only keep detections above this threshold" (`sam3_video_base.py:270-272`, `:608`; `model_builder.py:747`);
  - the image API's `confidence_threshold=0.5` (`sam3_image_processor.py:17`, `:199`).

  So the confident cut is 0.5.
- **But SAM 3 starts a masklet only from a detection with probability ≥ `new_det_thresh` = 0.7** that matches no existing masklet (`:210-213`; `model_builder.py:750`). So every masklet SAM 3 outputs scores ≥ 0.7 under the literal score, or > 0.5 under any per-frame detection average. Every masklet is confident, and **[0.1, 0.5) is empty by construction**.
- So the Verify action, and G1's "confident plus candidates" recall, have nothing to work on. That changes only if SAM 3 runs with lower thresholds, which the proposal does not specify (and the prompt forbids changing settings).
- **G0 proceeds without candidates** (G0 has no loop). The question is for G1.
- My recommendation, if candidates are wanted: a second, separate SAM 3 pass with `score_threshold_detection` = `new_det_thresh` = 0.1. Masklets that overlap no confident masklet become candidates; the confident tier stays SAM 3 at defaults. It costs a second full SAM 3 pass, on top of the runtime problem in §3.

### D2. The score field (check 2): what "averaged over detected frames" can mean

- The thresholded quantity is the detection probability `sigmoid(pred_logits)`. It is **already presence × object**: `pred_logits` = logit(σ(class) × σ(presence)), clamped ±10 (`sam3_image.py:345-353`; the presence token is on by default).
- The per-object field SAM 3 outputs is `out_probs` = `obj_id_to_score`. It is **set once, from the detection that started the masklet, and never updated** (`sam3_video_base.py:1059-1064`; the code's comment says "first frame detection score", `sam3_video_inference.py:409`). SAM 3's own multi-prompt video evaluation uses exactly this one value per masklet (`sam3_video_inference.py:946-950`; `sam3_video_base.py:2075`, `per_frame_scores = scores`).
- **Detected frames are identifiable**, but not from the output. They come from SAM 3's own per-frame association: a detection started the masklet, or matched it at mask IoU ≥ 0.1. That is the definition SAM 3's masklet confirmation uses (`sam3_video_base.py:1968-1970`, `:2011-2017`). G0 reads it through a read-only wrapper around `run_tracker_update_planning_phase`; SAM 3's code is not edited.
- Per-frame detection probabilities exist, but SAM 3 never attaches them to a masklet. When several detections match one masklet on a frame, SAM 3 defines no single value (`:150`, `:251`).
- **G0 reading (literal):** the per-frame value is `out_probs`; the track score is its mean over detected frames, which equals the starting detection's probability. G0 also records, but does not use, the per-frame detection probabilities and the count of multi-match frames.
- Tiers do not depend on this choice (D1); only the final ranking does.
- My recommendation for G1: keep the literal reading. It is SAM 3's own score and needs no multi-match rule.

### D3. "Merge" of duplicates across prompts is underspecified

The proposal says duplicates are "merged, the higher score keeps its class", but not what the merged track's frames are.
- **G0 reading:** keep the higher-scoring masklet unchanged and remove the other, greedily by score. Each track stays exactly one SAM 3 ID, which fits "one ID = one track".
- The other reading is the union of frames.
- G0 reports every duplicate and the frames the removed one had outside the shared ones.
- I recommend the G0 reading.

### D4. The pre-roll (check 1)

The released gap code (`gen_labels.py`) does five things:
1. adds 2 frames before the first box, copies of it (`:363-371`);
2. drops tracks with < 65% coverage;
3. trims the shorter side of any gap longer than 15% of the span;
4. fills the remaining internal gaps by **linear interpolation** of each coordinate, then rounds and clamps every box (`:416-423`, `round_and_positive`);
5. adds nothing after the last box.

Items 2 and 3 and the length filter are rule 8, which RPT replaces. The pre-roll is neither listed in rule 8 nor kept.
- **G0 reading:** fill internal gaps only, by item 4 exactly (the repo's own `round_and_positive` is called); no pre-roll.
- I recommend no pre-roll: identity, and session 3's train GT tracks had none.

### D5. Rental, a cost decision for the author

The rental would be about 1.5–2 GPU-hours on an RTX 4090, roughly $0.55–0.70 at $0.35/h. It has a pre-registered stop if SAM 3's first smoke video projects the 3 videos beyond 1.25 GPU-hours. Details in §6.

## 3. Early warning for G1 (to be measured in step 9)

- **Runtime.** SAM 3 handles one text prompt per pass. On every reset it clears its feature cache, so 35 prompts mean 35 backbone passes per frame. G1's bar (≤ 1 GPU-hour per 20 videos, i.e. ≤ 8 h for HELD's 47,038 frames × 35 prompts) needs ≤ ~17 ms per frame per prompt. My rough expectation on a 4090 is 80–150 ms, so **the bar is likely missed by 5–9×**, and the pre-registered every-*k*-frames fallback is likely. G0 measures this; it does not pick *k*.
- **A very long HELD video.** One HELD video has 2,947 frames. SAM 3 holds frames on the GPU at 1008², fp16, by default: 6.1 MB per frame, so about 18 GB for that video. It may not fit with the model on 24 GB at default settings. This is a G1 issue.
- **Check 3, the clip grid.** Clips start at each pair's overlap start, not on a video grid (`gen_labels.py:657`, `:661-666`, `:693`), and features are read at clip middle frames (`:81-82`). With HELD's GT tracks, `gen_feats_test` reads 628 of 47,038 frames (1.3%). On-demand features are therefore cheap: a median of 2.5 MB per video, against 145 MB for every frame.
- **Check 4, window joining.** Session 3's `run_pair` concatenates the windows' per-clip outputs, and `eval_from_cache` merges each pair's joined sequence, so windows are joined before the merge. But no real pair ever exceeded 40 clips, and the released SGDet path does not window at all. G0's unit test covers a 95-clip pair.

## 4. Licence (step 2)

The SAM License, last updated 2025-11-19, unchanged since SAM 3's initial commit. The Hugging Face `LICENSE` is the same text, verified by git blob hash, since the file itself is gated.

- **Research use:** allowed. §1.a: *"You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license … to use, reproduce, distribute, copy, create derivative works of, and make modifications to the SAM Materials."*
- **Publication:** allowed, with acknowledgement. §1.b.ii: *"If you submit for publication the results of research you perform on, using, or otherwise in connection with SAM Materials, you must acknowledge the use of SAM Materials in your publication."*
- **Outputs:** no clause restricts redistributing outputs such as tracks. "SAM Materials" covers the models, software and documentation only. Outputs appear only in §3 (warranty disclaimer) and §5.b (litigation). Two cautions: §1.b.iii requires compliance with privacy and data-protection law, and if tracks are ever released, including a copy of the licence covers the stricter reading of §1.b.i.

## 5. Environment and laptop

- **SAM 3:** `facebookresearch/sam3` at `f1647e8d0c1d280991f4d1b7d974c2704537fe66`, cloned to `~/Documents/thesis/source/sam3`, never edited. Its own conda env `sam3`: Python 3.12.15, torch 2.10.0+cu128 (as SAM 3's README installs it), `-e ".[notebooks]"`, psutil.
  - **Not SAM 3.1.** The repo added SAM 3.1 on 2026-03-27 (`facebook/sam3.1`), and `build_sam3_predictor()` now defaults to it. G0 uses `build_sam3_video_predictor()` and `facebook/sam3` at HF revision `3c879f39…`, the November 2025 release.
  - **Two install deviations:**
    - conda-forge instead of the default channels, which now need a Terms-of-Service acceptance that is the author's to give;
    - setuptools pinned to 78.1.0, because SAM 3 imports `pkg_resources`, which setuptools 84 removed.
- **Laptop memory probe** (random weights, no data): 860,055,224 fp32 parameters, 3.21 GiB. Moving the model to the GTX 1650 "succeeds" only by spilling into WSL shared memory (3.49 GiB allocated against 3.21 GiB free), which leaves nothing for frames or activations.
- **The laptop check itself** (`…00010024`, 64 frames, prompt "bicycle", default settings, 30-minute cap) runs after A28.
- **EOV-MMP side:** `repro-next` has torch 2.11.0+cu128, the same build as session 3's box.
- **Frame provenance:** today's cv2 5.0.0 re-decodes the 5 videos' mp4s into JPEGs byte-identical to the laptop's frames (1,300 of 1,300).
- **Housekeeping for the author:** HF token files exist at `~/.cache/huggingface/{token,stored_tokens}`, from an earlier login. G0's scripts never read them (they use a throwaway `HF_HOME`). The author may want to delete them.

## 6. Step 8 needs a 4090; plan if approved

- **Why a 4090.** CLIP runs in fp16 on CUDA, and modelC's temperature is 0.01. A 1e-4 logit bar therefore means agreement to about 1e-6 in cosine, which in practice needs session 3's stack: an RTX 4090 with torch 2.11.0+cu128. A laptop run would be a declared dry run that does not count.
- **On the box** (A26's pattern; scripts in `studies/rpt/g0/box/`):
  1. upload about 150 MB (code archive, the 5 HELD videos' frames, 5 train annotations, class files; **no test file**);
  2. setup: dependencies in the image's `/venv/main`, CLIP from its public URL with a hash check, SAM 3 in its own venv;
  3. the author's interactive token fetch: the end-to-end checkpoint from the private weights repo, and `sam3.pt`;
  4. the run: step 8 → SAM 3 on video 1 → budget check → SAM 3 on videos 2–3 → relation side;
  5. copy back and verify hashes.
- **On the laptop:** compare, evaluate, draw the images, write the report.
- **Time:** about 35 minutes of fixed overhead plus about 55–75 minutes of SAM 3 (41,125 frame-prompt steps).

## 7. What the author does next

1. If the author has no access to `facebook/sam3` yet, request it on the model page (Meta approves manually). Then, in their own terminal:
   `bash studies/rpt/g0/fetch_sam3_ckpt.sh`. The token is typed hidden and held only in `HF_TOKEN`.
2. Bring back your answers to D1–D4, and the author's answer to D5.

Then I append and hash A28 (with your changes, if any), commit after a credential scan, run the unit tests and the laptop check, and ask for the rental formally.

---

## Appendix: A28 draft, not hashed

`{{SAM3_PT_BYTES}}` and `{{SAM3_PT_SHA256}}` will be filled from the fetched checkpoint. The rule texts implement D2–D4 as read above; your answers may change them.

#### A28 (2026-10-26): RPT G0 — SAM 3 set up, the five code checks fixed as adapter rules, a reproduction of session 3 and a 3-video smoke test on HELD; no gate result, no test data

Written before any SAM 3 output on a VidVRD video, any adapter run, the reproduction or the smoke
test exists. No model has been trained or evaluated, no test annotation has been opened, and the
only GPU use so far is a memory probe of SAM 3 with random weights and no data
(`studies/rpt/g0/logs/laptop_memory_probe.txt`). A27 stays reserved for the tracklet-keeping
design (`studies/design_review/agreed_plan.md`); this is the next free label.

**What G0 is.** The first gate of the RPT proposal (`studies/rpt/rpt-proposal.md`, version 5,
SHA256 `1f9f648262c460ccd181e7e60a79ada9b67d5293f7ef8cff2ef746958959cf83`): a frozen SAM 3 replaces
EOV-MMP's detector, tracker and track filter, and its tracks enter the released relation head
unchanged, through `gen_feats_test` + `modelC`, the path session 3 used for GT tracks. G0 sets SAM 3
up, settles the proposal's five code checks, builds the adapter and runs a small smoke test. **It
measures no gate result and makes no design choice.** Where the proposal cannot be followed as
written, G0 reports it and leaves it OPEN for the G1 amendment. It ends with
`studies/rpt/g0_report.md`, a draft G1 amendment (not hashed) and a stop.

##### The chain checker (G0 step 3)

`studies/migration/verify_after_A26m.py` skipped its ov-pairtime check without a word once the
ov-pairtime folder was gone. Only that check changed: the file is now read from the archive bundle
`~/Documents/thesis/archive/ov-pairtime-2026-10-24-a820a0e.bundle` (commit `a820a0e`, through a
throwaway bare repository) and must hash to `981bf067…5ce5`. A missing bundle, one git cannot read,
or a different hash is a FAIL with a non-zero exit; the check is never skipped.

| | SHA256 |
|---|---|
| `studies/migration/verify_after_A26m.py` (fixed) | `3af183caad77d61ea4aca7e7fa04d2da74c354b0d4335797ede98409d0eda1e7` |
| `studies/rpt/tests/test_chain_checker.py` (7 tests: present and matching, synthetic and real; missing; altered; two unreadable bundles; `main()` exits 1 without the bundle) | `8fc440011d2d60750e17b4ee7a3d830ce95791251ac1749882a3b394f33ea13b` |
| `studies/rpt/g0/logs/verify_after_A26m_fixed.txt` (the real chain: 66 checks, all passed, head `ac77e243…a591`) | `26154ba93572ae25bdc36ed502a13902a962d5c95d427dd54efca61838378677` |

##### SAM 3, frozen

- **Code:** `github.com/facebookresearch/sam3` at `f1647e8d0c1d280991f4d1b7d974c2704537fe66`
  (2026-10-23), cloned outside this repo to `~/Documents/thesis/source/sam3`. Never edited.
- **Checkpoint:** `facebook/sam3`, file `sam3.pt`, Hugging Face revision
  `3c879f39826c281e95690f02c7821c4de09afae7`, {{SAM3_PT_BYTES}} bytes, SHA256 `{{SAM3_PT_SHA256}}`.
  It is loaded only from a local path after its SHA256 is checked; SAM 3's own download code is
  never reached.
- **Model:** `build_sam3_video_predictor()` (`sam3/model_builder.py:818`) with every default: SAM 3,
  the November 2025 release. **Not SAM 3.1**, which the repo added on 2026-03-27 (`facebook/sam3.1`,
  `build_sam3_predictor()` defaults to it); the proposal names SAM 3.
- **Licence:** the SAM License (last updated 2025-11-19; `LICENSE` SHA256
  `4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be`, unchanged since the initial
  commit; the Hugging Face repo's `LICENSE` is the same text without its final newline). §1.a grants
  use; §1.b.ii requires that a publication acknowledge the use of SAM Materials. Outputs are not SAM
  Materials; no clause restricts them.
- **Environment:** its own conda env `sam3` (conda-forge Python 3.12.15; the default Anaconda
  channels were not used, since they need a Terms-of-Service acceptance that is the author's to
  give), torch 2.10.0+cu128 as SAM 3's README installs it, `pip install -e ".[notebooks]"`, psutil,
  and setuptools 78.1.0 (`model_builder.py:8` imports `pkg_resources`, which setuptools 84 no longer
  ships). `pip freeze`: `studies/rpt/g0/logs/env/pip_freeze_sam3_laptop.txt`, SHA256
  `7831e667dcbe4decdd0d970c25338efd3e3c62606f31bade73d01d165db7fe7e`. The two sides talk only through
  files.

##### The five code checks, and the rules they fix (G0 step 4)

1. **Gap filling.** The released track code (`models/gen_labels.py`, called by
   `format_trajectories`, `:446-451`, and `format_trajectories_test`, `:605-610`):
   - adds two frames before the first box, copies of it (`add_initial_frames`, `:363-371`);
   - drops a track with boxes on < 65% of its span (`:380-382`), and, while a gap is longer than 15%
     of the span, drops the shorter side of the largest gap (`:385-413`);
   - fills every remaining internal gap by **linear interpolation** of each coordinate between the
     boxes on either side (`np.interp` over the present frames, `:416-419`), then rounds every box of
     the span to integers and clamps it at 0 (`round_and_positive`, `:422-423`, `:426-428`);
   - adds nothing after the last box (`end_fid` = last frame + 1, `:480`, `:635`); keeps a track
     only with ≥ 12 frames, pre-roll included (`:471`, `:626`).

   **Rule.** Internal gaps of a SAM 3 ID are filled by `:416-423` exactly: `np.interp` over the
   frames where SAM 3 output the ID, then the repo's own `round_and_positive` on every frame of
   [first, last]. The coverage rule, the gap trim and the length filter are rule 8, which RPT
   replaces. The **pre-roll** is neither listed in rule 8 nor kept by the proposal, which fills only
   internal gaps: it is **not applied**, and is OPEN for G1.

2. **Score field.**
   - SAM 3 thresholds the **detection probability** `sigmoid(pred_logits)`
     (`sam3/model/sam3_video_base.py:602`): a detection is kept if it is > 0.5
     (`score_threshold_detection`, `:608`, with mask NMS at the same threshold,
     `sam3_image.py:846-861`), and a masklet is started only by a kept detection ≥ 0.7
     (`new_det_thresh`) that matches no existing masklet (`:210-213`). The video post-processing
     (`_postprocess_output`, `sam3_video_inference.py:432-524`) thresholds no score.
   - **Presence × object: yes, already.** With the presence token (on by default:
     `has_presence_token=True` → `supervise_joint_box_scores`, `model_builder.py:680`, `:739`),
     `pred_logits` = logit(σ(class logit) × σ(presence logit)), clamped to ±10
     (`sam3_image.py:345-353`).
   - **The output field** is `out_probs` (`sam3_video_inference.py:450-452`, `:519`) =
     `obj_id_to_score`, set once, to the probability of the detection that started the masklet
     (`sam3_video_base.py:1059-1064`; the code's own comment: "first frame detection score",
     `sam3_video_inference.py:409`), and never updated (`:1079-1080` only marks removals). SAM 3's
     own multi-prompt video evaluation scores each masklet with this one value, on every frame
     (`sam3_video_inference.py:946-950`; `sam3_video_base.py:2049-2075`, `per_frame_scores = scores`).
   - **Detected / propagated / absent:** not in the output, but identifiable from SAM 3's own
     per-frame association: a masklet is *detected* on frame t if it was started by a detection on
     t, or a kept detection was matched to it on t (mask IoU ≥ `assoc_iou_thresh` 0.1); this is the
     definition SAM 3's masklet confirmation uses (`sam3_video_base.py:1968-1970`, `:2011-2017`). It
     is read through a read-only wrapper around `run_tracker_update_planning_phase`; SAM 3's code is
     not edited and its outputs do not change.
   - Per-frame detection probabilities exist inside SAM 3, but are not attached to masklets; when
     several kept detections match one masklet on a frame (allowed: `im_mask`, `:251`, feeds
     `det_to_matched_trk_obj_ids`, `:150`), SAM 3 defines no single value.

   **Rule (as written: "SAM 3's own per-object confidence, averaged over detected frames only").**
   The per-frame value is the masklet's `out_probs` in that frame's output; the track score is its
   mean over detected frames. Since `out_probs` is constant, the score equals the probability of
   the masklet's starting detection, ≥ 0.7. Also recorded per frame, **not used**: the probabilities
   of the matched detections (with a count of frames with more than one) and SAM 3's tracker score.
   Whether G1 should average the per-frame detection probabilities instead, and how a frame with
   several matches would count, is OPEN.

3. **Clip grid.** Clips are not on a video grid. For each ordered pair, clips start at the pair's
   overlap start, `begin_fid = max` of the two tracks' starts (`gen_labels.py:657`), in steps of 30
   (`:13`, `:693`); a tail ≥ 10 frames is the last clip and a tail < 10 is cut (`:661-666`,
   `:694-697`). A clip's features are read at its middle frame, `begin_fid + len(clip)//2`
   (`:81-82`, `:106`, `:117`). Middle frames are shared only by pairs whose overlap starts agree
   modulo 30, always by (s, o) and (o, s). On HELD with session 3's GT tracks
   (`studies/rpt/g0/check3_middle_frames.py`, SHA256 `6843dbff…cede9`; output
   `logs/check3_middle_frames.json`, `44911fb2…f9cd`): 628 of 47,038 frames are read (1.3%, 4.0
   clips per frame read); a cache of only those frames is 2.5 MB per video at the median (max 17.7
   MB) against 145 MB (max 2.49 GB) for every frame, at fp16.
   **Rule.** Features are computed on demand for exactly the frames `gen_feats_test` requests, and
   memoized per (video, frame); never for every frame.

4. **Window joining.** Session 3's `run_pair` (`pilot_analysis/stage1c/cache_wrapper.py:166-180`)
   splits a pair's clip sequence into consecutive windows of ≤ 40 (`:46`, `:171`), runs `modelC`
   once per window (`:173-175`) and concatenates the per-clip outputs in order (`:176`, `:179`);
   `eval_from_cache.predict` then runs `process_pred` → `association` on each pair's whole joined
   sequence (`eval_from_cache.py:72-77`). So per-clip predictions are joined **before** the greedy
   merge. No pair in session 3 had more than 40 clips, so the join ran only in its random-tensor
   check; the released SGDet path does not window at all (`models/end2end_model.py:326-337`).
   **Rule.** `run_pair` is called, not reimplemented; its joined output goes to `process_pred` →
   `association` → `format_`.

5. **Confident cut: 0.5.** SAM 3 keeps a detection if its probability is > 0.5: the video model's
   `score_threshold_detection` ("prob threshold for detection outputs -- only keep detections above
   this threshold", `sam3_video_base.py:270-272`, `:608`; 0.5 in `model_builder.py:747`), the image
   API's `confidence_threshold=0.5` (`sam3_image_processor.py:17`, `:199`; SAM 3's example notebooks
   use `Sam3Processor(model, confidence_threshold=0.5)`), and the predictor API's default
   `output_prob_thresh` 0.5 (`sam3_base_predictor.py:76-79`; passed only to SAM 3.1 models).
   `new_det_thresh` = 0.7 (`model_builder.py:750`) starts a masklet; it does not keep a detection.
   **Consequence, reported and not acted on:** at SAM 3's defaults every masklet starts from a
   detection ≥ 0.7, so under either score reading every masklet is confident and **the candidate tier
   [0.1, 0.5) is empty by construction**. Filling it would need other SAM 3 settings, which the
   proposal does not give. OPEN for G1 and for the loop's Verify action.

##### The adapter (`studies/rpt/`; G0 step 7)

SAM 3 side (its own environment):
- **Prompts:** the 35 names of `data/vidvrd/data/object2id.json` (SHA256 `350c554c…2b37`), in id
  order, `_` replaced by a space; nothing else, no synonym. Each masklet carries its prompt's name.
- **Protocol:** SAM 3's own multi-prompt video protocol (`Sam3VideoInference.forward`,
  `sam3_video_inference.py:928-952`), through the predictor API: one session per video on its JPEG
  folder; per prompt, `add_prompt` at frame 0, then `propagate_in_video` forward from frame 0 over
  every frame, then `reset_session`. Default settings throughout (thresholds 0.5 / 0.7, hotstart
  15 / 8 / 8, image size 1008, frames held on the GPU, bf16 autocast as the predictor sets it, no
  `torch.compile`). `torch.manual_seed(0)`; SAM 3 inference draws no random numbers.
- **Written per video:** `studies/rpt/g0/sam3_out/<video>.json` (gzip): per prompt and masklet,
  per output frame the RLE mask, the tight box of the returned mask, SAM 3's own box, `out_probs`,
  the frame status and the recorded detection and tracker scores; per prompt its time; per video
  the peak GPU memory.

EOV-MMP side (unchanged repo code):
- **Identity:** one (prompt, SAM 3 ID) is one track; a new ID is a new track; nothing is split.
- **Boxes:** on every frame where SAM 3 returns a non-empty mask for the ID, the tight box of that
  mask, `[x1, y1, x2, y2]` in pixels with `torchvision.ops.masks_to_boxes`' convention (the function
  SAM 3 uses for its own box). SAM 3's own box (from the mask before its non-overlap step,
  `sam3_video_inference.py:492-515`) is recorded beside it. A frame with an empty returned mask is
  not an output frame.
- **Gaps:** rule 1. Frames before the first or after the last output frame are not part of the
  track.
- **Status per frame:** *detected* or *propagated* (output frames, rule 2) or *filled* (gap frames).
- **Score and tiers:** rule 2; confident if score ≥ 0.5 (rule 5), candidate if 0.1 ≤ score < 0.5,
  dropped if < 0.1. With no loop in G0, only confident tracks are paired (proposal §3.1: candidates
  are for the loop).
- **Duplicates across prompts:** two masklets from different prompts whose per-frame mask IoU,
  averaged over the frames where both have an output mask, is > 0.7 are duplicates. The proposal
  says they are "merged" and "the higher score keeps its class", but not what the merged track's
  frames are. **Reading used in G0:** the higher-scoring masklet is kept unchanged (frames, boxes,
  score, class) and the other is removed, greedily in descending score (ties: prompt id, then SAM 3
  ID); each kept track is still exactly one SAM 3 ID. Merging into a union of frames is the other
  reading. G0 reports every duplicate and the frames the removed one had outside the shared frames.
  OPEN for G1.
- **Pairs:** `gen_feats_test` (`gen_labels.py:642`) on the confident tracks, split `train` (the HELD
  annotation is read for width, height and the frame count only); its own overlap ≥ 10 and tail
  rules, in its own order.
- **Windows:** rule 4.
- **Features:** CLIP-L and TagCLIP-L through the repo's own encoders, `_shared_encoder` and
  `_transform_resize(336, 336)` (`data_loading/dataset.py`), exactly as session 3's
  `encode_frames` per frame (`cache_wrapper.py:78-98`): batch 1, fp16 on CUDA, frame `f` = the f-th
  file in sorted order; whole video, the GT range of `dataset.py:120-135` bypassed; on demand and
  memoized (rule 3).
- **Gap exposure:** a clip whose middle frame is a *filled* frame of its subject or object is
  flagged and counted.
- **No test labels:** every G0 runner installs an audit hook that raises on opening any path under
  `anno/test`, any `test_*` file of `data/vidvrd/data`, or anything of session 3's test caches.

##### The reproduction of session 3 (G0 step 8)

- **Videos:** the first 5 HELD IDs in sorted order (`pilot_analysis/stage1/train_split.json`,
  SHA256 `095dd89c…1f1a`): `ILSVRC2015_train_00008005`, `…00010024`, `…00024005`, `…00025029`,
  `…00027003`.
- **Input:** their GT tracks from `gt_tracks.load_tracks("train")` (option B,
  `gt_tracks_train.json`, SHA256 `87b40fff…636d`), through the adapter's track check and feature
  wrapper into `gen_feats_test` + `modelC` (session 3's `build_modelC`, from the end-to-end
  checkpoint `d306952a…0e70`, `tgt_split all`, `--frame_stride 1`). No SAM 3 step applies to GT
  tracks.
- **Compared with** `cache_s3/train/<video>.pt` (each matching `SHA256SUMS_session3.txt`, SHA256
  `2cc2d071…954b`, as checked on 2026-10-26), pair by pair in `gen_feats_test` order.
- **Bar:** max |logit − cached logit| ≤ 1e-4 over every pair, clip and predicate column.
  Reported, not bars: the interactiveness-logit residual, the slot-input residuals, and whether
  each requested frame's `global_proj` equals the cached one (frame provenance).
- **Where:** session 3's stack: an RTX 4090, torch 2.11.0+cu128, `OMP_NUM_THREADS=8`. The frames are
  the laptop's JPEGs, which today's cv2 5.0.0 re-decodes from the laptop's mp4 files byte for byte
  (all 1,300 frames of the 5 videos), as session 3 decoded on its box. A run on the laptop's GTX
  1650 (sm_75), if made first to test the code path, is labelled a dry run; its residual is reported
  and does not count.
- **On a fail:** find where the paths diverge (frames, features, windows, ordering) and report.
  Nothing is adjusted to make it pass.

##### The smoke test (G0 step 9)

- **Videos:** the first 3 of the 5: `…00008005` (130 frames), `…00010024` (64), `…00024005` (981).
- **Pipeline:** SAM 3 (35 prompts, every frame of the whole video) → adapter → `gen_feats_test` +
  `modelC` → `process_pred` (top 20 per clip, no prior) → `association` → `format_` (top 200) →
  `eval_relation_detection_openvoc(target_split_pred="base")` with `gt_json` = the HELD GT of these
  3 videos from `pilot_analysis/stage2/data/held_gt.json` (SHA256 `78d55c69…4dee`; built from the
  training annotations, base relations only, so the novel-predicate instances of the training
  videos stay unused). No T1 rules, no loop.
- **Reported per video:** frames; SAM 3 time per prompt and in total; confident, candidate and
  dropped by class; SAM 3's own removals; duplicates; ordered pairs and clips; middle frames inside
  filled gaps; frames with several matched detections; peak GPU memory. Then the G1 cost
  extrapolated to the 160 HELD videos (47,038 frames) against G1's runtime bar, and 18 images (6
  evenly spaced frames per video, GT and SAM 3 boxes) in `studies/rpt/g0/viz/`.
- **The evaluator's numbers are labelled "smoke test, in-sample, not a result"**: EOV-MMP was trained
  on HELD. They are not read, compared or used for anything.

##### The laptop check (G0 step 6)

SAM 3 at its default settings on `…00010024` (the shortest smoke video, 64 frames) with one prompt,
"bicycle" (the class of its GT track 0), on the GTX 1650 (4 GB). Reported: whether it runs, peak
GPU memory, time per frame. Nothing is lowered or changed to make it fit. It is stopped after 30
minutes of wall time if unfinished, and reported as such. If it cannot serve steps 8–9, the
author is asked before any machine is rented, with a time and cost estimate.

##### A rented machine, only if the laptop cannot serve

An RTX 4090 (session 3's GPU), ≥ 32 GB RAM, a Vast.ai PyTorch image with torch 2.11.0+cu128 for the
EOV-MMP side, SAM 3 in its own virtual environment as above, `OMP_NUM_THREADS=8`,
`--frame_stride 1`. Budget: under 2 GPU-hours. If the measured SAM 3 time on the first smoke video
projects the 3 videos beyond 1.25 GPU-hours (which, with about 35 minutes of set-up, fetch,
reproduction and copy-back, keeps the machine under 2 hours), the run stops and the author is asked.
The token, as in
A25c and A26: a read-only Hugging Face token, held only as `HF_TOKEN` in the author's interactive
fetch, never written to a file, log or report, unset after use, with token scans before and after.
No test file is copied to the machine. Nothing is pushed.

##### G0 pass bars

1. The reproduction: max absolute logit residual ≤ 1e-4 against `cache_s3`, on all 5 videos.
2. All adapter unit tests pass (`studies/rpt/tests/`).
3. The 3 smoke videos run end to end, SAM 3 to the evaluator, without an error.

A pass allows the G1 amendment to be written; it is not evidence for the method.

##### OPEN, for the G1 amendment (not decided here)

1. The pre-roll (rule 1).
2. The score: `out_probs` as written, or the per-frame detection probabilities, and how a frame with
   several matched detections counts (rule 2).
3. Candidates: empty at SAM 3's defaults (rule 5); how, if at all, the candidate tier is filled.
4. Duplicates: keep the higher-scoring masklet, or merge into a union of frames.
5. The runtime fallback's *k*, if G1's runtime bar cannot be met (proposal §3.1).

**Order.** 1. This amendment; the checker is run. 2. Laptop: the adapter's unit tests; the laptop
check. 3. If needed, the author approves a machine. 4. Steps 8 and 9 there; copy-back; the machine
is destroyed by the author. 5. Laptop: comparison, evaluation, images, `studies/rpt/g0_report.md`,
`studies/rpt/g1_amendment_draft.md` (not hashed). Stop.

The adapter code and its tests were written after this text was drafted and before it was
appended; none of it had run. Any change to the above is a new amendment, written before the
result it affects.
