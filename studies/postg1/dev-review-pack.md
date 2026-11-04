# Developer review pack: PaD, before implementation

2026-11-03 · one file for Claude Code. Save it in the repo as `studies/postg1/dev-review-pack.md`.

**For Claude Code:** carry out **Part 1** only. Parts 2–6 are the documents it asks you to review. Where background is needed beyond this file, `studies/postg1/` and `studies/rpt/` hold the earlier docs (`trainable-direction-1.md` and replies, `monday-pack.md`, `g1_report.md`).

| Part | Content |
|---|---|
| 1 | **The task:** a read-only developer review |
| 2 | `pad-proposal.md`: the method, PaD (Prompt-and-Distill) |
| 3 | `pad-review-1.md`: the planning chat's review of PaD |
| 4 | `pad-review-2.md`: the other chat's response |
| 5 | `pad-review-3.md`: the planning chat's final reply (plan settled) |
| 6 | `d7-session-prompt.md`: the next session, prepared but not yet run |

---

# Part 1: Developer review of PaD before implementation (read-only)

You are in `~/Documents/thesis/source/ov-vidvrd-lab`. Two planning chats have designed a new method, **PaD (Prompt-and-Distill)** (Part 2), and the next session's prompt (Part 6). Before anything is built, I want **your opinion as the developer who knows the code**: what is feasible, what is harder than the planners think, what they got wrong, and what you would do differently.

**This session is read-only.** No edits to any file except your report, no commits except the report, no GPU, no downloads, no test labels, no HELD-B data (HELD is not split yet; just do not open anything beyond what reading code requires). You may read code, grep, and run small CPU-only inspections (for example printing a module's structure or parameter names from a checkpoint's config) if they help. If something needs a run to answer, say what run, and estimate its cost.

**Read first:** Part 2 (sections 1, 3a, 3, 6 and 8 matter most), Parts 3–5 (the final agreements: **Phase 1 is the method; Phase 2 is an extension**), and Part 6.

## What I want your opinion on

### A. Phase 1, P1: the learned class-free prompt for SAM 3 (the core)
1. **Where can a learned prompt enter SAM 3?**
   - Can token embeddings be fed into SAM 3's frozen text encoder in place of words, CoOp-style? Point to the tokenizer, embedding layer and forward path.
   - Or is it easier to learn the text encoder's **output** (the embedding passed to fusion) directly?
   - Can either be done **without editing SAM 3's code** (wrappers, hooks, subclassing)?
2. **Can gradients reach the prompt?** With SAM 3 frozen and only the prompt trainable: is the path from the detector's outputs (boxes, scores, presence) back to the prompt differentiable? Any `no_grad`, detached tensors or in-place operations in the way?
3. **Training code and losses.**
   - Does the SAM 3 repo ship training code, matching and losses usable with **box-only** targets (VidVRD has no masks)?
   - Can the mask losses be switched off cleanly?
   - What would we have to write ourselves?
4. **Cost.** Rough GPU memory and time to train on about one frame per 30 from the 640 FIT videos: is a 24 GB 4090 enough, or does it need 48 GB?
5. **Using the learned prompt in video inference.** Can a learned embedding replace the text prompt in the video predictor (`build_sam3_video_predictor`, the predictor API G0/G1 used), so the tracker runs with it? Is SAM 3's presence head likely to behave sensibly with a generic or learned prompt?

### B. Phase 1, the other parts
6. **Naming SAM 3's class-free tracks (P4)** with EOV-MMP's auxiliary object classifier (CLIP + VPN prompts, modelB?). Is there code that runs it on **given boxes or tracks** (an SGCls-like path)? The code check found no PredCls path in the released repo. What would naming need, and how faithful would it be to how EOV-MMP names objects?
7. **The learned keep/drop model (P3):** are the planned features (track length, SAM 3 scores, box stability, CLIP-feature consistency, naming margin) easy to get from G1-style outputs? Anything important missing?
8. **Relations (P5):** confirm that SAM 3 tracks can keep entering `gen_feats_test` + `modelC` as in G0/G1, or say what changes with class-free tracks named by P4.

### C. Phase 2 (extension): only feasibility and size
9. **RB1's vendored step-4 code** (`studies/metor_code/`):
   - can it take extra class-free box targets, the generic "object" query, ignore regions, an objectness loss on `pred_ious`, and ignored relation pairs (Part 3, point 1)?
   - how big is it really, and what is the risk, given its GPU path has never run?

### D. The stretch goal
10. **Level 2:** relation loss → prompt, through SAM 3's soft masks on short clips. Realistic in this code, or not? What breaks first?

### E. The next session (Part 6)
11. **Is it implementable as written?** Anything ambiguous, wrong or missing:
    - the HELD split and the HELD-B guard;
    - the screen on HELD-A;
    - the D7 check's arms, measures and memory rule;
    - the scripts and the fake SAM 3 worker.

    Note: Part 6 refers to `monday-pack.md` Part 1 for Task 2's definition; that file is in `studies/postg1/`.
12. **Would you change the order of work?** For example, test P1's gradient path cheaply before the D7 check.

### F. Overall
13. **Effort:** realistic days of work for Phase 1, and for Phase 2.
14. **The top 3 risks** you see as a developer, and what you would check first to retire each cheaply.
15. **Anything the planners got wrong or missed.**

## Deliverable
`studies/postg1/dev_review_pad.md`:
- answer A–F point by point, with file:line references where you looked;
- mark each point **feasible / feasible with work / doubtful / blocker**;
- put a short summary at the top: your overall verdict, and the changes you would make before starting.

Commit only that file, after a credential scan. **Then stop.**


---

# Part 2: `pad-proposal.md` (the method)

# Proposal: PaD, Prompt-and-Distill a foundation tracker for open-vocabulary video relation detection

Version 1 · 2026-11-03 · Bùi Trường Giang · merges ideas D7 (prompting SAM 3) and C6 (distilling SAM 3). All other ideas stay in `trainable-direction-1.md` as fallbacks.

## 1. The idea in one paragraph
EOV-MMP uses its foundation model, CLIP, in two ways: it **prompts** CLIP (learned prompt vectors, used directly for naming and relations), and it **distils** CLIP into its own box decoder. PaD does the same with **SAM 3**, the foundation model that finds and tracks objects.
- **Phase 1, Prompt:** SAM 3 stays frozen, and we learn a small class-free prompt that makes it find *the kind of objects VidVRD annotates*, in one pass per frame instead of 35, and with far less junk than G1's 35-name version.
- **Phase 2, Distill:** the prompted SAM 3 becomes a teacher, and EOV-MMP's own fast detector learns from its boxes. At test time SAM 3 is gone, so the final model runs at EOV-MMP's speed while carrying SAM 3's object-finding skill.

Everything learned is trained on VidVRD's training videos, base classes only.

### Why merge prompting and distillation
**The model:** EOV-MMP needs CLIP in both ways because each covers what the other cannot.
- CLIP cannot output boxes, so EOV-MMP trains its own decoder and **distils** CLIP into it, making the decoder's embeddings CLIP-like.
- It also **prompts** CLIP directly for naming and relations.

SAM 3 is the foundation model for the "where" side, so the same two uses carry over.

**Each idea alone has a flaw; the other fixes it:**

| Idea alone | Its flaw | How the other fixes it |
|---|---|---|
| **Prompting alone (D7)** | SAM 3 must run at test time, so it stays slow and heavy (G1 ran out of memory on 24 GB GPUs) | **Distillation** moves the skill into EOV-MMP's own detector, and SAM 3 is gone at test time |
| **Distillation alone (C6)** | the teacher would be the 35-name SAM 3, whose junk (1.8× unmatched tracks in G1) the student would learn; running it on the 640 FIT videos costs about 100–140 GPU-hours | **Prompting** first gives a cleaner teacher, and with one learned prompt instead of 35 names, running it on FIT is about 35× cheaper (about 5–6 GPU-hours) |

**The order matters.** Prompt first: the teacher must be good before it teaches. Then distil, which gives speed and one network. The Phase 1 gates check the teacher before anything is spent on Phase 2.

## 2. Why: what the measurements say
- **Most misses happen before any relation is scored.** Of the novel relations EOV-MMP misses on SGDet, 54% have no matching track pair, and about 62% of those losses happen in EOV-MMP's **non-learned** parts: the fixed 0.35 class-score cut, the coverage/range filters, and the hand-made tracker (deep_sort + AFLink).
- **EOV-MMP learns "where" from base-class boxes only.** Its CLIP distillation teaches "what is in this box", not "where are the objects". So objects outside the 35 base-class experience are missed or scored low.
- **SAM 3 knows "where" far better.** In G1 it found +10 to +12 points more annotated objects and pairs than EOV-MMP. But it also produced 1.8× more unmatched tracks: tiny boxes, edge slivers, fragments, empty background, and real objects VidVRD leaves unannotated. It needs 35 passes per frame.
- **METOR (the bar, 16.74)** removed the tracker with video-level queries, but still keeps and drops objects by **class score** (average > 0.2, boxes at 0.35). Its object detection gain over EOV-MMP is small (mAP_o 38.9 vs 36.3).

PaD attacks the gap neither paper closes: **learning what counts as an object separately from what it is called**, with a foundation model's localisation.

## 3a. Architecture at a glance

**Legend:** 🆕 **new**, built and trained by us · 🔒 **frozen** foundation model · ♻️ **reused** from EOV-MMP, unchanged · ❌ **removed** from EOV-MMP

**Phase 1 system (main contribution):**

| Step | Component | Status | Trained by us? |
|---|---|---|---|
| 1 | SAM 3 image and text encoders | 🔒 frozen | no |
| 2 | **Learned class-free prompt** for SAM 3 (P1) | 🆕 new | **yes** |
| 3 | SAM 3's box finder (driven by the prompt) | 🔒 frozen | no |
| 4 | SAM 3's tracker | 🔒 frozen (replaces EOV-MMP's tracker) | no |
| 5 | **Track filter + learned keep/drop** (P3) | 🆕 new | **yes** |
| 6 | CLIP image and text encoders | 🔒 frozen | no |
| 7 | Naming: auxiliary object classifier (VPN + learned prompts on CLIP) | ♻️ reused | no (already trained) |
| 8 | Relation classifier: spatio-temporal visual prompting, VPN, object and relationship mapping layers, relationship indication layer | ♻️ reused | no at first; *optional fine-tune later* |
| 9 | T1 subject-type rules | ♻️ ours, already done (+0.36) | rules |
| 10 | Clip merging, top-200 per video | ♻️ unchanged | — |

**Phase 2 system (fast version, no SAM 3 at test time):**

| Step | Component | Status | Trained by us? |
|---|---|---|---|
| 1 | CLIP image encoder | 🔒 frozen | no |
| 2 | **Query-based box decoder, retrained** with the prompted SAM 3's boxes as extra teaching targets (class-free) | ♻️→🆕 retrained | **yes** |
| 3 | **Objectness head** ("is this an object?", separate from the class; starts from the unused `pred_ious`) | 🆕 new | **yes** |
| 4 | Tracking: deep_sort + AFLink | ♻️ reused | no |
| 5 | **Learned keep/drop** (P3), refit on the new tracks | 🆕 refit | **yes** |
| 6–10 | Naming, relations, T1, merging | ♻️ as in Phase 1 | as in Phase 1 |

During training only, the Phase 1 system acts as **the teacher**: it runs on the 640 training videos to produce the boxes the decoder learns from.

**Removed from EOV-MMP in Phase 1:**

| EOV-MMP part | Why it goes |
|---|---|
| ❌ Query-based box decoder + box head | SAM 3 finds the boxes (it returns, retrained, in Phase 2) |
| ❌ Object embedding head (CLIP distillation) | no decoder queries in Phase 1 |
| ❌ Relationship query + relationship classification head | they only helped train the decoder |
| ❌ deep_sort + AFLink | SAM 3 tracks |
| ❌ The fixed 0.35 cut and the length/coverage filters | replaced by the learned keep/drop |
| ❌ Decoder class score in the name fusion | no decoder score exists; naming uses the auxiliary classifier alone |

## 3. The system

### Phase 1: Prompt (the main contribution)

| # | Component | Status | Trained by us |
|---|---|---|---|
| P1 | **SAM 3 detector with a learned class-free prompt** | new | **yes** |
| P2 | SAM 3's tracker | new to the pipeline (replaces deep_sort, AFLink, the length filter) | no, frozen |
| P3 | **Track filter + learned keeping** | new (replaces the fixed 0.35 cut) | **yes** |
| P4 | Naming: EOV-MMP's auxiliary object classifier (CLIP + VPN prompts) | reused | already trained |
| P5 | Relations: EOV-MMP's relation classifier (spatio-temporal visual prompting, mapping layers, VPN, relation indication layer) | reused | already trained |
| P6 | T1 subject-type rules | done (+0.36) | rules |
| — | Clip merging and top-200 per video | unchanged | — |

**P1, the learned prompt:**
- **What is learned:** a few (4–16) prompt vectors fed into SAM 3's frozen text encoder in place of words, as CoOp/EOV-MMP do with CLIP. One shared prompt, no class names, so **one pass per frame instead of 35**.
  - variant: learn the text encoder's output directly;
  - optional later: a conditional (VPN/CoCoOp-style) prompt, if the plain one drifts.
- **Training data:** frames sampled from the 640 FIT training videos (about one per 30 frames), with VidVRD's **base-class boxes** as targets.
- **Loss:** SAM 3's own detection matching, with box losses (L1, GIoU) and its object/presence score. Masks are not used, because VidVRD has boxes only.
- **Novel objects in training videos:** option (b). They are unlabelled background like any unannotated area. No novel annotation is used, not even locations.
- **Guards against over-fitting to base classes:**
  - few tokens;
  - initialise from the best hand-written prompt;
  - a KgCoOp-style penalty keeping the learned prompt near the hand-written one;
  - the prompt is chosen by base-object recall and burden only.

**P3, keep or drop after tracking:**
- **First, a geometric filter:** the tiny-box, edge-sliver and fragment rules from the HELD-A screen.
- **Then a small learned classifier** per track, with inputs: track length, SAM 3 scores, box stability, CLIP-feature consistency, and the naming margin. Kept tracks get a capped score.
- **Labels:** "matches a base GT object" = 1. Everything else, novel objects included, is unlabelled (option (b)).
- Trained and cross-validated on **HELD-A**.

**Naming, P4:** SAM 3's boxes have no class. Each track is named by EOV-MMP's auxiliary object classifier, as in EOV-MMP's SGCls setting. EOV-MMP's decoder class score no longer exists, so its score fusion is not used.

### Phase 2: Distill (the speed extension)

| # | Component | Status | Trained by us |
|---|---|---|---|
| D1 | **Teacher labels:** the Phase 1 system run on the 640 FIT videos | new | — (inference) |
| D2 | **Student detector:** EOV-MMP's query-based decoder, retrained with the teacher's boxes as extra targets | retrained | **yes** |
| D3 | **Objectness head** on the student: keep a box if class score > 0.35 **or** objectness is high. Starts from the decoder's existing, unused IoU head (`pred_ious`). | new | **yes** |
| D4 | Tracking for the student: EOV-MMP's tracker, plus learned keeping (P3) refit on the student's tracks | reused + refit | refit |
| D5 | P4–P6 as in Phase 1 | reused | — |

- **Class-free distillation:** the teacher's boxes teach **where** (box and objectness targets) and never **names**. Naming stays with CLIP, as now.
- **Teacher boxes that match no base GT** (novel objects, unannotated objects) are used as positives only after P3's filter, so the student does not learn the teacher's junk.
- **Training** uses RB1's clean step-4 code (R2-style: FIT only, leaks removed), so HELD stays a fair check.
- **At test time:** the student only. No SAM 3, at EOV-MMP's speed.
- **The honest limit:** the student is a per-frame detector, so tracking returns to EOV-MMP's tracker. Phase 2 trades some of Phase 1's tracking quality for speed.

### Phase 3: stretch goals (only if Phases 1–2 work)
- **Relation classifier fine-tuned on the new tracks** (P5 adapted to SAM 3's or the student's tracks), with RB1's clean code.
- **Level 2, end-to-end:** the relation loss trains P1's prompt through SAM 3's soft masks, on short clips. It needs about a 48 GB GPU, and it is research-heavy.

## 4. What is new
- **Against EOV-MMP:** a foundation tracker, prompt-tuned, replaces the trained-on-35-classes decoder and the hand-made tracker. Keep/drop is learned and separate from naming.
- **Against METOR:** METOR still keeps and drops by class score; PaD decides "is it an object?" separately from "what is it?". PaD also uses no trained trajectory regression; SAM 3 tracks.
- **The method pattern:** **prompt-tune a foundation model to the benchmark, then distil the tuned model into a fast detector.** EOV-MMP's two uses of CLIP, carried over to a localisation foundation model.

## 5. Protocol and data rules
- **Learned components** use VidVRD **base-class** annotations only, with option (b) for novel objects.
- **Disclosed:** SAM 3's own pre-training has seen the novel object classes. The gain is reported split into relations with novel-class objects and with base-class objects only. Most of the upstream gain is expected on base objects.
- **HELD is split into two halves:**
  - **HELD-A (80 videos):** prompt choice, the screen, and learned-keeping training;
  - **HELD-B (80):** sealed until one pre-registered, tracker-level go/no-go.
- **One test run at the end,** with its primary row named in advance. Every rule is fixed and hashed before the data it governs is seen.

## 6. Gates, in order

| # | Gate | Data | Pass rule | Cost |
|---|---|---|---|---|
| 0 | Task 2 screen (geometric rules) | HELD-A, G1 outputs | gives P3's starting filter (its verdict closes only the 35-name option) | $0 |
| 1 | **D7 check:** hand-written generic prompts, no class names | 20 (or 40) HELD-A videos | the better generic arm keeps **≥ half** of the 35-name SAM 3's class-agnostic recall gain over EOV-MMP | $1–5 |
| 2 | Prompt training (P1) | FIT to train, HELD-A to choose | class-aware recall and pair recall above the generic prompt, burden at or below the 35-name SAM 3's (exact numbers fixed before training) | about $10–30 |
| 3 | **Phase 1 go/no-go** (P1 + P3) | sealed HELD-B, tracker-level | recall gain ≥ +8, pair-recall gain ≥ +8, burden ≤ 1.35× vs EOV-MMP R0w | small |
| 4 | Phase 2 (student) | HELD-B, tracker-level | the student keeps a pre-fixed share of the teacher's recall gain at a burden ≤ 1.35× (share fixed before training) | about $100–200 (with R2) |
| 5 | **Test,** once | VidVRD test | pre-registered rows; primary row named in advance | small |

## 7. Expected result and risks
- **Phase 1:** roughly **+0.5 to +1.5** SGDet novel over T1 (16.15), so about **16.7–17.7** against METOR's 16.74. A possible win, not a safe one.
- **Phase 2:** keeps part of that gain, but at EOV-MMP's speed with one trained network.
- **Main risks:**
  1. SAM 3 may need class names to find things. Gate 1 checks this for $1–5.
  2. A prompt trained on base classes may suppress novel objects. The guards help; it is reported, not hidden.
  3. Some unannotated real objects will still be found, which is a burden. P3 targets it.
  4. The student keeps only part of the teacher's skill.
  5. The trimmer is out, so the large over-long loss (27% of misses) stays. If results disappoint, it comes back (fallback).

## 8. Cost and time (rough)
- **Phase 1:** about **$20–40**, about 3–4 weeks.
- **Phase 2:** about **$150–250** including R2, about 3–4 more weeks.
- With one learned prompt, SAM 3 on all 160 HELD videos is about 1–2 GPU-hours, and on the 640 FIT videos about 5–6. That is far cheaper than G1.

## 9. Fallbacks (in `trainable-direction-1.md`)
- **If gate 1 fails:** the objectness-head route (C4/C5) on R2.
- **Further down:** Option 1 (EOV-MMP detects, SAM 2 tracks), Option 2 (rule-filtered SAM 3), and the VidOR trimmer if results are poor.

## 10. Open questions for Giang
1. **The name:** "PaD (Prompt-and-Distill)" or another?
2. **Phase 2:** part of the thesis method, or kept as an extension for the paper?
3. **The primary test row:** the Phase 1 system (slower, likely stronger), or the Phase 2 student (fast, one network)? Fixed before the test.


---

# Part 3: `pad-review-1.md`

# Review of PaD (`pad-for-review.md`, `pad-proposal.md`), planning chat, 2026-11-03

Sources:
- `d2_report.md`, `o1_step1_inventory.md`, RB1 `step1_report.md`;
- `unified-model-plan.md` (Q7, A27), `rpt-g1-outcome.md`;
- published work (links at the end).

## Summary
**Phase 1 is sound and should be the method.** Phase 2's case is much weaker than the proposal says, for four reasons:
1. **The speed argument mostly disappears once Phase 1 works.**
   - With one prompt, SAM 3 on the 200 test videos is about 1–2 GPU-hours (≈ 50k frames × 0.094 s, plus tracking), against EOV-MMP's ≈ 3.5 h.
   - G1's out-of-memory came from keeping every frame on the GPU (6.3 MiB per frame), which offloading fixes (G0 measured it). It did not come from the 35 prompts. *(Corrected in Part 4: offloading is a no-op in this code, per A29d.)*
   - A published training-free method (DART) also batches all class prompts through SAM 3 with a shared backbone.
2. **The analogy breaks at its reason.** EOV-MMP distils CLIP *because CLIP cannot output boxes*. SAM 3 can, so Phase 1 has no gap that distillation must fill.
3. **The teacher adds little where the score is.**
   - Upstream gains come from base objects (O1: 23% of U1 involves a novel object; Q7: 7 of 54 novel-object relations recoverable).
   - FIT's base objects already have GT boxes, so the student gets those without a teacher.
   - Under option (b), the prompt is trained on FIT to skip everything unannotated. So **on its own training videos the teacher's extras are smallest** (see 3).
   - What the student would mostly gain is the objectness head, which is C4.
4. **The student keeps only part of Phase 1's gain, and it sits on R2,** which may score below the released 15.79. As the primary test row it would likely give the lower number.

**Recommendation:**
- Phase 1 is the method and the default primary row.
- Phase 2 is an extension, kept or dropped at gate 4 by a share rule fixed in advance (6).

This is Giang's decision; it also answers the proposal's open questions 2 and 3.

**Small fix to proposal §2:**
- "About 62% in non-learned parts" holds at score floor 0.05 only.
- The filter losses are coverage and the GT-range rule, not length.
- The robust statement is "about 72% of U1 have no good box above the cut".

## 1. Phase 2 in the code
**Feasible, in RB1's vendored copy (`studies/metor_code/`), not the authors' root files.** RB1 already wraps step 4 with flags and keeps the released parameter names.

**What changes, roughly in order of difficulty:**
1. **Text and exemplar conditioning (RB1 audit T2).** The detector's training queries are the text and exemplar features of the labels present in the frame. A class-free target has no label, and using a novel name would be a leak. A design choice is needed, for example one generic "object" text query added in training and at test. That changes the architecture and breaks bit-compatibility with R2, so R2 must be rerun with it as the baseline.
2. **Matching and class loss.**
   - Teacher targets enter Hungarian matching with box (and objectness) costs only.
   - The class loss must be **masked** on queries matched to them. Otherwise they become negatives for all 35 classes.
3. **Ignore regions.** Deformable DETR has none. Mask the class and objectness losses for unmatched queries overlapping an ignore box.
4. **Objectness on `pred_ious`.**
   - Check whether `set_criterion` already trains it. If it does, extend its targets; if not, add an IoU-regression loss on matched queries only (OLN style, no background term).
   - Change the keep rule in the post-processor (`post_process.py:194-195`): class score > 0.35 or objectness ≥ t.
5. **The frame gate (T3).** Frames with only teacher boxes are now trained on, which changes the frame set. Log the count.
6. **The relation half of step 4.** Step 4 trains the relation classifier jointly on the detector's tracks. Pairs involving teacher-only objects have no GT relations, so by RB1's F3 they become **negative pairs**: more suppression of relations on exactly the objects the teacher added. **Ignore those pairs in the relation and interactiveness losses.**
7. **The teacher-label loader:** boxes, keep probabilities and hashes per FIT frame.

**Size:** a few hundred lines in the vendored copy, plus tests.

**The bigger risk:**
- RB1's step-4 path has **never run on a GPU**. The released path crashed on this data, and RB1's fix was tested on CPU only.
- RB1's 5-video GPU check must pass before Phase 2 is costed.
- Phase 2 needs at least two step-4 runs: R2, the baseline, and the student.

## 2. The student loses SAM 3's tracking
**Expect roughly 30–60% of Phase 1's gain to survive.**
- In D2, about 72% of U1 is lost before tracking and about 26% at tracking and the filter.
- A student that matched the teacher's per-frame boxes would keep at most about 70%.
- The extra low-score boxes will fragment more under deep_sort, and **novel objects lose most at tracking** (D2 Table 5: −19.7 points from detection to tracking, against −9.4 for base).

**Do not add association learning now.**
- A re-ID head trained on the teacher's track identities is feasible and is not METOR's territory, but fragmentation is the smaller link loss (F1: 3.95% stitch candidates; D2: 22 fragmented relations).
- The larger link losses are the **coverage filter (29 sides) and the GT-range rule (23)**.
- **So in Phase 2, P3 must replace the fixed filters too, not only refit beside them.** The GT-range rule is also a protocol leak and has to go.

If gate 4 shows large tracking losses, a middle version is the student plus SAM 2 tracking (Option 1). It is still a foundation tracker at test time.

## 3. The teacher's junk
**P3's filter alone is not enough, and the problem is not only junk.**
- **The in-sample teacher.** The prompt is trained on FIT under option (b), so on FIT it has learned to stay quiet on unannotated objects. Its extras on FIT will understate what it finds on new videos.
  - **Fix: cross-fit.** Train two prompts, each on half of FIT, and let each label the other half. That means 2 prompt trainings, about $10–30 more.
- **How teacher boxes enter the student:**

  | teacher box | box loss | objectness loss | class loss | relation loss |
  |---|---|---|---|---|
  | matches base GT | as GT | positive | as GT | as GT |
  | passes P3, no base GT match | yes | **soft positive, weight = P3's keep probability** | masked | pairs ignored |
  | rejected by P3 or the geometric rules | none | **ignored, not negative** | masked | — |

  P3 is trained positive-unlabelled on benchmark labels, so its rejects include real unannotated objects, novel ones among them. Making them negatives would teach the student to suppress real objects. The student does not make SAM 3's slivers itself, so no negatives are needed to unlearn them.
- **The check:** gate 4's burden bar (≤ 1.35× against R2) catches inherited junk.

## 4. Exposure in Phase 2
- **In information, it is no stronger than Phase 1.** Only part of SAM 3's knowledge passes through pseudo-boxes on training videos, and SAM 3 is gone at test.
- **In form, it resembles the known T1 leak:** novel-object boxes as detector targets on training videos, minus human annotation and names. Reviewers will look at it for that reason.
- **Disclose:**
  - the student's box targets include class-free pseudo-boxes from a frozen foundation model on training videos;
  - no human novel annotation and no novel names are used;
  - teacher boxes are never filtered, weighted or counted with novel GT during training.
- **Report:**
  - after training, how many teacher extras land on novel-class objects (descriptive);
  - the student **without extras** (the C4-only head, 6);
  - the relation gain split by novel-object and base-only relations.
- **A point in PaD's favour, worth a sentence in the thesis:** Phase 1 removes the released detector, which was trained on human novel boxes with their labels (T1/T2), and Phase 2 on R2 does too. On that point both are cleaner than EOV-MMP itself.

## 5. R2's timing
**Agreed: R2 waits for gate 3.** Two notes:
- R2 is also the fallback route if **gate 1 fails** (C4/C5). In that branch it becomes the next step at once, and its 10 days are on the critical path.
- RB1's 5-video GPU check is R2's own first gate (see 1).

## 6. Thresholds

### Gate 2, on HELD-A, P1 alone (before P3)
Gate 3 needs +8 / +8 after P3, and P3 only removes tracks, so P1 needs headroom. All four must hold:
1. **Against EOV-MMP R0w, class-aware** (naming with the auxiliary classifier): recall gain ≥ **+10** and pair-recall gain ≥ **+10** points.
2. **Against the better generic arm from gate 1, class-agnostic:** recall ≥ **+2** points at a burden no higher, **or** burden lower by ≥ **0.15×** at a recall no lower. Otherwise the learned part adds nothing, and the honest contribution is the hand-written prompt.
3. **Burden (class-aware) ≤ the 35-name SAM 3's** on the same HELD-A videos.
4. **Novel-object recall:** reported, not gated.

HELD-A also chooses the prompt, so gate 2 is optimistic by design. Gate 3 on HELD-B is the binding one.

### Gate 4, on HELD-B, tracker-level, class-aware
- **Common base: R2**, which is out-of-sample on HELD.
- **Share** = (student − R2) / (Phase 1 system − R2), computed on recall and on pair recall.
- **Burden ≤ 1.35× against R2,** in every case.
- **The share rule:**
  - **≥ 0.8:** the student may be the primary test row;
  - **0.5–0.8:** the student is reported as the fast version, and Phase 1 stays the primary row;
  - **< 0.5:** Phase 2 is dropped from the method and reported as a negative result.
- **Teacher contribution:** the student must beat a **C4-only head** by ≥ **+2** pair-recall points: an objectness head trained on R2's frozen query outputs with base GT only, which is cheap. Otherwise report that distillation adds nothing beyond objectness.
- **HELD-B's second use:** disclose it; nothing is tuned on HELD-B.

## 7. The next session (`d7-session-prompt.md`)
**One change is required.**
- Step 1 says HELD-B is sealed "until one pre-registered go/no-go for 'D7 + learned keeping'", and a later amendment may only make it stricter. That forbids gate 4 on HELD-B later.
- Write the roles now: **HELD-B serves gate 3 (Phase 1) and, only if gate 3 passes, gate 4 (student against R2), each pre-registered, with nothing tuned on HELD-B.** *(Applied in Part 6.)*

**Optional:** call Steps 2 and 4 "PaD gates 0 and 1" in the summary.

Nothing else needs to change. Naming without the decoder score is caught by gate 2, which is class-aware.

## 8. Prior work
- **The pattern "prompt-tune a foundation model, then distil it":** **PromptKD** (CVPR 2024) prompt-tunes a large CLIP teacher on labelled data, then distils it into a small CLIP student on unlabelled images. It works on classification, not detection.
- **SAM-generated class-agnostic pseudo-labels training a detector:** **SOS** (ECCV 2024) prompts SAM with object priors, filters and merges the masks with the base annotations, and trains a class-agnostic Mask R-CNN for open-world segmentation, with large precision gains. That is very close to Phase 2, minus learned prompts. **Open World DETR** uses class-agnostic pseudo-labels for unknown objects in a DETR-style detector.
- **SAM 3 speed and compression:**
  - **DART** (arXiv 2603.11441): training-free multi-class SAM 3 (shared backbone, batched prompts, box-only), up to 5.6× faster. This weakens both speed arguments.
  - **EfficientSAM3** (arXiv 2511.15833): distils SAM 3's encoders into small backbones and MobileCLIP. This is compression, not detection.
- **I know of nothing that learns a class-free SAM 3 prompt and distils it into a DETR-style detector, or applies either step to video relations.**

**The novelty is the combination and the task.** Each half has a clear precedent: CoOp-style prompting for Phase 1, SOS-style pseudo-labelling for Phase 2. Frame it that way.

## Links
- [PromptKD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html)
- [SOS: Segment Object System (ECCV 2024)](https://fiona.uni-hamburg.de/a3c1f3ec/wilms-etal-eccv-2024.pdf)
- [Open World DETR](https://arxiv.org/pdf/2212.02969)
- [DART: Detect Anything in Real Time](https://arxiv.org/pdf/2603.11441)
- [EfficientSAM3](https://huggingface.co/Simon7108528/EfficientSAM3)


---

# Part 4: `pad-review-2.md`

# Response to `pad-review-1.md` (other planning chat, 2026-11-03)

**I accept the review almost entirely**, including the main conclusion: Phase 1 is the method, and Phase 2 is an extension decided at gate 4 by the share rule. There is **one factual correction** (offloading), which changes the hardware note but not the conclusion, and **one small addition** to the next session. Giang decides the primary row; I will put your recommendation to him.

## Summary points
1. **Speed: agreed, with a correction.**
   - One prompt puts SAM 3 near EOV-MMP's cost (about 1–2 GPU-hours for the test set), so speed is not a reason for Phase 2. DART points the same way.
   - **But offloading does not fix the out-of-memory problem.** A29d recorded that SAM 3's `offload_video_to_cpu` is a **no-op** in this code: `sam3_video_inference.py:145-153` copies every frame to the GPU regardless, and check 4's peak equals check 0's, 7.316072 GiB. G0's check 4 "passed" only because nothing changed.
   - Also, two of G1's three out-of-memory videos were short and crowded (521 and 836 frames, 21.5–22.4 GiB), so memory grows with the number of objects, not only frames. One prompt does not remove that.
   - **Consequence:** Phase 1 at test time may need a 48 GB GPU (L40 or similar) for long or crowded videos. That is cheap (G1 priced one at $0.457/h), but it is a hardware requirement, and the test-session amendment must plan for it.
2. **The analogy breaks at its reason:** agreed. SAM 3 outputs boxes, so there is no gap that distillation must fill.
3. **The teacher adds little where the score is:** agreed. FIT's base objects already have GT, and under option (b) the teacher's extras on FIT are suppressed. The student's main gain would be the objectness head, which is C4.
4. **The student on R2 likely gives the lower number:** agreed.

**The §2 fix:** accepted. Use "about 72% of U1 have no good box above the cut"; the filter losses are coverage and the GT-range rule.

## Point 1, the code: accepted
Especially the two items I had missed:
- **the detector's text and exemplar conditioning (T2),** which needs a generic "object" query and therefore an R2 rerun with it as the baseline;
- **the relation half of step 4,** where teacher-only objects would become negative pairs and must be ignored.

**RB1's untested GPU path is the main practical risk.** Its 5-video GPU check comes first in any Phase 2 or R2 plan. Phase 2 therefore needs at least two step-4 runs (R2 with the generic query, and the student), plus that check.

## Point 2, tracking losses: accepted
- About 30–60% of the gain is expected to survive. No association learning now.
- **P3 replaces the coverage filter and the GT-range rule** in Phase 2, not only refits beside them. The GT-range rule is a protocol leak.
- The student + SAM 2 middle version is noted as a fallback.

## Point 3, the teacher: accepted
- the cross-fit teacher (two prompts on FIT halves);
- the table of how teacher boxes enter the losses;
- P3's rejects are **ignored, not negative**.

## Point 4, exposure: accepted
The disclosure and the reports as written, and the sentence in PaD's favour (both phases drop the T1/T2-trained detector).

## Point 5, R2's timing: accepted
Including that R2 moves onto the critical path at once if gate 1 fails.

## Point 6, thresholds: accepted
Gates 2 and 4 exactly as proposed. One note for Giang: gate 2's **+10 / +10 class-aware against R0w** asks P1 alone to beat what the 35-name SAM 3 reached in G1 (+9.8 / +9.2 on 157 videos, class-aware). That is demanding, but it is the right headroom for gate 3's +8 / +8 after P3. If gate 2 fails only on this criterion, that is information, not a reason to lower it.

## Point 7, the next session: accepted, plus one addition
- **HELD-B's roles,** to be written in Step 1 now: gate 3 (Phase 1) and, only if gate 3 passes, gate 4 (student against R2). Each is pre-registered, and nothing is tuned on HELD-B.
- **Addition (memory):** in the D7 check, a video that runs out of GPU memory on the 4090 is skipped and replaced by the next video in the fixed order, recorded, and never retried on another GPU within the check. This is not result-based (no arm's numbers are seen), and it keeps the check at one machine and one cost. Videos over 1,000 frames are already excluded.
- "PaD gates 0 and 1" naming in the summary: yes.

## Point 8, prior work: accepted
The framing is: **the novelty is the combination and the task.** Phase 1 follows CoOp-style prompting, applied to a promptable concept segmenter and made class-free. Phase 2 follows SOS-style class-agnostic pseudo-labelling, with a prompt-tuned teacher. Cite PromptKD, SOS, Open World DETR, DART and EfficientSAM3.

## Next
I will:
- update `d7-session-prompt.md` (HELD-B roles, the memory rule, gate names);
- update `pad-proposal.md` to version 2 once Giang has chosen the primary row.


---

# Part 5: `pad-review-3.md`

# Reply to `pad-review-2.md` (planning chat, 2026-11-03)

**Agreed, and the PaD plan is settled** apart from Giang's choice of the primary row.

**Offloading: your correction is right, and mine was wrong.** A29d recorded that `offload_video_to_cpu` is a no-op in this code, and G1's crowded short videos show memory grows with the number of objects. A 48 GB GPU for long or crowded videos is the right plan.

## Two notes for `d7-session-prompt.md` and later amendments
1. **The out-of-memory rule in the D7 check applies to every system.**
   - A video that runs out of memory in either arm is dropped from **all four** systems (arm A, arm B, the 35-name SAM 3 and EOV-MMP R0w) and replaced by the next video in the fixed order.
   - The "half of the 35-name gain" reference is computed on the **final** list.
   - Disclose that the skipped videos are the crowded ones, so the check slightly under-samples crowded scenes.
2. **Skipping is acceptable only for the cheap check.** Gate 3 (HELD-B), the FIT teacher runs and the test session must not skip videos: dropping crowded videos would bias a binding gate.
   - Plan the 48 GB GPU in advance for those runs, in their amendments.
   - Write "no skips" into each of them.

*(Both notes are applied in Part 6.)*

## Gate 2
Keep +10 / +10 as written. If it fails only on that criterion, record it as information, as you said.


---

# Part 6: `d7-session-prompt.md` (the next session, not yet run)

# Next session: HELD split, SAM 3 screen on HELD-A, (trimming oracle), D7 check preparation

> **DECISION (set by Giang): run Task 1, the trimming oracle? → NO.**
> Skip Step 3 entirely, and say so in the summary. The trimmer is out of the plan for now; it comes back only if the main results are poor. No test-label access happens in this session.

You are in `~/Documents/thesis/source/ov-vidvrd-lab`. Read first:
- `studies/postg1/monday-pack.md`: Part 1 defines Task 1 and Task 2 in full;
- `studies/postg1/trainable-direction-1.md`, `-reply-1.md`, `-reply-2.md` and `-reply-3.md` (I will copy them in): the agreed new direction.

**In short:** the lead direction is now **D7**, a learned, class-free prompt for SAM 3 (so detection takes one pass per frame instead of 35), plus a learned keeping model, both trained on VidVRD. Today does:
- the HELD split;
- the SAM 3 screen on HELD-A only, which becomes D7's track-level filter;
- optionally, the trimming oracle;
- the **preparation** of a small paid D7 check.

**No rental happens without my approval. Nothing is pushed.**

## Ground rules
1. **Never edit the authors' code** (files from `fdeeeb4`). New code goes in `studies/postg1/` or `studies/d7/`. Never push. Credential scan before every commit.
2. **Test labels:** touched only in Step 3, only if the decision above is YES, and only after its amendment is committed. Otherwise the test-label guard stays on.
3. **HELD-B is sealed after Step 1.** Add a guard (an audit hook, like the test-label guard) that refuses to read HELD-B's annotations, G1's SAM 3 outputs for HELD-B videos, and EOV-MMP's R0w tracks for HELD-B videos. It stays on for this whole session.
4. **Amendments first.** Each step that decides something gets its amendment, appended after the current chain head (A29d, `c9718301…fac8`, unless later ones exist), with the next free labels. Hash it, check it with `verify_after_A26m.py`, and commit it **before** what it governs runs.
5. **No design choices beyond what is written here.** If something cannot be done as written, stop that step and report.
6. **Never install into or upgrade EOV-MMP's Python environment. No downloads without my OK,** with the size stated first. **Never run SAM 3 on the laptop** (it brought the WSL session down in G0).
7. **Rentals, if I approve one later:** set an auto-stop; copy back and verify every hash; then tell me at once so I destroy the machine.

## Step 1: split HELD into A and B (CPU)
Write the amendment first. It fixes:
- **The split:** the 160 HELD videos into **HELD-A (80)** and **HELD-B (80)**.
  - Stratify by two flags: "has at least one novel-class GT object", and "longer than the HELD median frame count". That gives 4 strata.
  - Inside each stratum, sort by video ID and shuffle with `numpy.random.default_rng(20261102)`; alternate assignment A, B, A, B…
  - Record where the 3 videos without SAM 3 files (`…00280000`, `…00225001`, `…00971000`) fall.
- **The roles:**
  - **HELD-A:** prompt selection, the screen, and learned-keeping training (cross-validated inside HELD-A).
  - **HELD-B:** sealed. It serves at most two pre-registered, tracker-level gates, with nothing ever tuned on it:
    - **PaD gate 3**, the Phase 1 system (learned prompt + learned keeping) against EOV-MMP R0w;
    - **only if gate 3 passes, PaD gate 4**, the Phase 2 student against R2.
  - Both uses are disclosed. No other use of HELD-B is allowed.
- **HELD-B's gate type, fixed now (thresholds as stated; a later amendment may only make them stricter):**
  - It is **tracker-level, not mAP**, since modelC is in-sample on HELD.
  - It uses G1's matching (A29 rules) against EOV-MMP R0w on HELD-B:
    - overall recall gain ≥ +8 points;
    - pair recall gain ≥ +8 points;
    - burden ≤ 1.35×.
  - Class-aware matching is binding, since the final system names objects with CLIP. Class-agnostic numbers are reported.
- **Learned keeping follows option (b).** SAM 3 tracks that match a novel GT object are unlabelled like any other unmatched track, and never excluded. Novel GT is used only to report suppression.

Then build the split, write `studies/d7/held_split.json`, hash it, commit, and install the HELD-B guard.

**Deliverable:** the split's counts per stratum, the 3 no-file videos' halves, and the amendment's hash.

## Step 2: Task 2, the SAM 3 screen, on HELD-A only (CPU)
Run **Task 2 of `monday-pack.md` Part 1** exactly as written there, with these changes. All of them go in its amendment:
- **Data:** HELD-A videos with SAM 3 files only (about 78). The criterion is unchanged (+8 / +8 / ≤ 1.35× against EOV-MMP R0w); disclose the smaller sample.
- **Restated closing rule:** a fail closes **Option 2 and C6b** (the 35-name SAM 3 with rules, and its boxes as a teacher). **It does not close D7,** which uses SAM 3 differently and has its own check (Step 4).
- **Carried over, already in Part 1:**
  - the full 1,000-configuration grid;
  - a pass counts only with the class gate on;
  - the fixed selection rule;
  - rule 6 is skipped if modelC scores for SAM 3 pairs were not cached;
  - "for each rule", not "for each single setting".
- **The 20 inspected tracks:** report only those whose video is in HELD-A. Do not read or report HELD-B's.
- **Its role now:** besides the verdict, report the geometric rules' effect (tiny, edge sliver, fragment) on unmatched and matched tracks. They are the starting point for D7's track-level filter.

**Deliverable:** `studies/postg1/sam3_screen_report.md`, verdict first.

## Step 3: Task 1, the trimming oracle (only if the decision above is YES)
Run **Task 1 of `monday-pack.md` Part 1** exactly as written: the amendment first; one guarded run; the ledger; the crash rule; the binding decision rule (go if (ii) − T1 ≥ +2.0 **and** (iii) − T1 ≥ +1.0). Do not start Task 4 (VidOR prep), even on go; that is a later session.

**Deliverable:** `studies/postg1/trim_oracle_report.md`.

## Step 4: prepare the D7 check (CPU here; the run itself needs my approval)
**Purpose:** a cheap kill check. Can SAM 3 find VidVRD's objects **without class names**? If even hand-written generic prompts keep a good part of what the 35 class names found, a *learned* class-free prompt is worth building.

Write its amendment first, fixing:
- **The videos:**
  - 20 HELD-A videos with SAM 3 files, taken in the order of `numpy.random.default_rng(20261103)` over the sorted HELD-A list;
  - **the guard (computed now, from G1's committed outputs, before any rental):** compute the 35-name SAM 3's **class-agnostic** object-recall gain over EOV-MMP R0w on those 20 videos. If it is below **+5 points**, use **40 videos** (the next 20 in the same order). Record the gain and the final list. No arm's result exists at this point.
- **The arms**, both SAM 3 at G1's settings (defaults, `facebook/sam3` revision and SHA256 as A28/A28a, the G1 stack):
  - **(A)** the single prompt "object";
  - **(B)** the four prompts "person", "animal", "vehicle" and "object", one pass each. Their masklets are merged by A29's duplicate rule (mean mask IoU > 0.7, the higher score kept).
- **Check 0 first on the machine:** smoke video 1 at defaults with the 35 names, bitwise against G0. Stop and ask if it is not identical.
- **The measures,** all **class-agnostic**, with G1's A29 matching code (cut-vIoU on option-B runs), for four systems on the same videos: arm A, arm B, the 35-name SAM 3 (G1's files) and EOV-MMP R0w (G1's files):
  - object recall, overall and split base / novel;
  - pair recall (not U1);
  - unmatched confident tracks per video (burden).
- **The rule (binding):** D7 continues if the **better arm's class-agnostic object-recall gain over EOV-MMP is at least half** of the 35-name SAM 3's gain on the same videos. Burden and pair recall are reported, not gated: lowering the burden is what the learned prompt is for.
- **Machine:** one RTX 4090, ≥ 64 GB RAM, the same Vast.ai template as G1. One SAM 3 process per GPU. No A2; A1 only if its bitwise check passes as in G1. Videos over 1,000 frames are excluded from the list by construction.
- **Memory rule (fixed now):** SAM 3's `offload_video_to_cpu` is a no-op (A29d), and memory grows with the number of objects.
  - A video that runs out of GPU memory in **either arm** is dropped from **all four systems** (arm A, arm B, the 35-name SAM 3 and EOV-MMP R0w), and replaced by the next video in the fixed order. It is recorded, and never retried on another GPU within this check.
  - No arm's numbers are seen when this happens, so the replacement is not result-based.
  - The guard and the "half of the 35-name gain" reference are computed on the **final** list. If a replacement changes the list after the guard was computed, recompute the guard on the final list and report both.
  - **Disclose** that skipped videos are the crowded ones, so the check slightly under-samples crowded scenes.
  - **This skip rule applies only to this cheap check.** PaD gate 3 (HELD-B), the FIT teacher runs and the test session must **not** skip videos, since dropping crowded videos would bias a binding gate. Their amendments must plan a 48 GB GPU in advance for long or crowded videos, and state "no skips". Note this in the D7 check's amendment, so it carries forward.
- **Cost estimate:** about 20 × 300 frames × 5 prompts × 0.094 s ≈ 0.8 GPU-hours, plus setup. State the estimate for 40 videos too.

Then:
- write the scripts in `studies/d7/check/`: upload (HELD-A videos' frames only; no test file, no HELD-B file), setup (G1's pinned stack), run (check 0, both arms), copy-back with hashes, and the laptop analysis;
- test them locally with a **fake SAM 3 worker** (never real SAM 3 on the laptop);
- commit.

**Then stop and ask me before renting,** with:
- the final video list (20 or 40) and the guard's number;
- the time and cost estimate;
- 3 matching offers (price, RAM, location, reliability).

## Final deliverable
`studies/d7/session_summary.md`, with one paragraph per step:
- the split, including HELD-B's two roles;
- Task 2's verdict (**PaD gate 0**: the source of P3's filter);
- Task 1's decision ("skipped by decision");
- the D7 check's readiness (**PaD gate 1**), its guard number and the cost estimate.

Also give the new chain head and the commits. **Then stop.**
