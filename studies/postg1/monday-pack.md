# Monday pack: the agreed post-G1 plan and Monday's Claude Code prompt

2026-11-01 · one file for Claude Code. Save it in the repo as `studies/postg1/monday-pack.md`.

**For Claude Code:** carry out **Part 1** only. Parts 2–5 are the planning chats' discussion that led to it: background for understanding, not extra instructions. Where they differ from Part 1, Part 1 wins.

| Part | Content |
|---|---|
| 1 | **The Monday prompt** (version 3: the reviewed version 2, plus Task 3 item 10, SAM 2 vs SAM 3's tracker) |
| 2 | Reply 1: the planning chat's ceilings, ranking and first proposal |
| 3 | Reply 2: the other chat's response (one test look, run-specific trimmer, Option 1 notes) |
| 4 | Reply 3: the planning chat's reply (three changes; plan agreed) |
| 5 | The planning chat's review of the prompt (17 fixes, all applied in Part 1) |

---

# Part 1: Monday — trimming oracle, SAM 3 screen, Option 1 design note, VidOR prep (all laptop, $0)

Version 3 · reviewed (Part 5, all 17 fixes applied; one addition to fix 7) · plus Task 3 item 10

You are in `~/Documents/thesis/source/ov-vidvrd-lab`. RPT stopped at G1 (`studies/rpt/g1_report.md`). The two planning chats have agreed a new plan (Parts 2–4 of this file). In short:
- the path to beat METOR is T1 + Option 1 (EOV-MMP proposes, SAM 2 or SAM 3's tracker tracks) + a learned trimmer of over-long runs;
- SAM 3 as a detector gets one last free screen;
- today's work decides whether trimming is worth building, and prepares Option 1.

**Nothing today uses a GPU rental. Nothing is pushed.**

## Ground rules

1. **Never edit the authors' code** (files from `fdeeeb4`). New code goes in `studies/postg1/`. Never push. Credential scan before every commit.
2. **Test labels: one guarded run only, in Task 1, and only after its amendment is hashed and committed.** Every other task keeps the test-label guard on. If any other step would need a test file, stop and ask.
3. **Amendments first.** Tasks 1 and 2 each get an amendment, appended to the chain after A29d (head `c9718301…fac8`; use the next free labels, probably A30 and A31), hashed, checked with `verify_after_A26m.py` and committed **before** anything they govern runs.
4. **No design choices beyond what is written here.** If something cannot be done as written, stop that task and report; do not improvise.
5. **Stop at the end of each task's deliverable.** Do not start a task's follow-up (for example building the trimmer) even if the result says go.
6. **Never install into or upgrade EOV-MMP's Python environment.** The bit-exact reproductions depend on it.
7. **No downloads without my OK,** with the file size stated first.

## Task 1: the trimming oracle on test (one disclosed look)

**Purpose:** decide whether learning relation ends is worth building. It measures how much SGDet novel mAP would rise if over-long runs that would match a GT triplet were trimmed to the GT extent. It is one pre-registered look at test labels, like D1 and D2. Nothing is tuned from it.

**Write A30 first, fixing everything below. Hash, check, commit. Only then lift the test-label guard for this one script, run it, and put the guard back.**

- **The single guarded run does, in order:**
  1. re-evaluate the cached T1 test predictions unchanged: they must reproduce **16.15 / 27.57** (SGDet novel / all) exactly, or stop;
  2. re-run D1's classification (`pilot_analysis/time_check/tc_run.py`, the U1/U2/C1 logic) on the **T1** predictions, since D1 itself was computed on the baseline (15.79). Report the C1 over-long count next to D1's 129;
  3. build and evaluate the variants below.
- **Coverage.** D1 covers the 605 novel GT. If D1's code can classify all GT, apply the rule to all GT for the "all" numbers. If it cannot, trim only novel-matched runs and label the "all" numbers as such. The decision uses SGDet novel only.
- **Which runs are changed:** only predicted runs that D1's code classifies as the C1 over-long case for a GT triplet, **and that are not already true positives** under the unchanged evaluation (a run that already hits a GT is never touched). Each changed run is trimmed to the **single** GT instance it overlaps most. Never split a run; never add a run. After trimming, the evaluator matches as usual (one GT per prediction, its normal order).
- **Scores and top-200 membership unchanged.** Only extents change. Every other run is untouched.
- **"Slot":** the clip grid of that pair as `gen_feats_test` builds it (clips start at the pair's overlap start; G0 check 3). **A30 records the slot length** (expected 30 frames) before the run. The rule stays fixed whatever it is: a real trimmer has exactly this resolution, and with 30-frame slots variant (iii) is a harsher test. That is fair, and is stated in advance.
- **Variants:**
  - (i) **exact:** the run becomes the GT extent (within the original run);
  - (ii) **slot grid:** the slot-aligned interval inside the original run with the highest temporal IoU against the GT extent (ties: the shorter);
  - (iii) **one slot long:** (ii) extended by one slot at each end, clipped to the original run;
  - (iv) **one slot short:** (ii) shortened by one slot at each end, if at least one slot remains; otherwise as (ii). **Report-only.**
- **Metrics:** SGDet novel and all mAP, R@50 and R@100, by the repo's evaluator, for T1 and each variant; also (i) − (ii). **PredCls novel and all (report-only),** from the cached T1 PredCls predictions (22.25 / 38.03), with the same procedure; if not cached, skip it. No GPU.
- **The single decision (binding):** **go** if (ii) − T1 ≥ **+2.0** **and** (iii) − T1 ≥ **+1.0** SGDet novel mAP. Otherwise **drop trimming**.
- **Reading rule (stated in advance):** a learned trimmer recovers only part of the ceiling, so (ii) between +2 and +4 is a **weak go**; above +4 a **go**.
- **Nothing else is decided from these numbers.** The trimmer's design and thresholds are never chosen from (i)–(iv).
- **Crash rule:** if the script fails before any metric file is written, it may be fixed and rerun. Once a metric file exists, there is no rerun. Record which happened in the ledger.

**The ledger:** create `studies/test_looks_ledger.md` if it does not exist. List **every** test-label access in the chain, from the existing docs and logs, including:
- the baseline reproduction (15.79 / 27.18);
- the pilot's merge oracle (+0.07) and loose-vIoU (+12) numbers;
- session 3, if it touched test labels;
- D1, D2 and Q7;
- the method evaluations B, B2, T1, E1 and M1.

Mark any entry you are unsure of as such, rather than leaving it out. Then add this one: date, script, commit, which labels, the decision, and the crash rule's outcome.

**Deliverable:** `studies/postg1/trim_oracle_report.md`: the T1 reproduction, the re-run C1 count, the table (T1, i, ii, iii, iv, and i − ii; PredCls if available), the decision as the rule gives it, and the ledger entry.

## Task 2: the SAM 3 screen on HELD (exploration, last chance for SAM 3 as a detector)

**Data:** G1's committed outputs: SAM 3 confident tracks on 157 HELD videos; EOV-MMP R0w tracks on all 160; A29's matching code (`studies/rpt/g1/metrics.py`); the 20 inspected tracks (`viz_unmatched_observations.md`). Compare on the 157 videos only.

**Write A31 first (the screen note), fixing the rules, the grid, the criterion and the selection rule below. Hash, check, commit, then run.**

**Rules** (each removes whole SAM 3 confident tracks; "box" = the track's per-frame tight mask boxes):
1. **Tiny:** remove if min(median box width, median box height) < t px; t ∈ {8, 12, 16, 24}.
2. **Edge sliver:** remove if the box touches the left or right frame edge (within 2 px) on ≥ f of the track's frames **and** its median width < w px; (f, w) ∈ {0.5, 0.8} × {16, 32}.
3. **Fragment:** remove if, on ≥ f of the frames it shares with some larger confident SAM 3 track (any class, larger by median box area), ≥ c of its box area lies inside that track's box; (c, f) ∈ {0.8, 0.9} × {0.5, 0.8}.
4. **Score cut (reference only, never selected or combined):** remove if track score < s; s ∈ {0.6, 0.7, 0.8, 0.9}.
5. **Geometry-only partner:** keep only if some other SAM 3 confident track overlaps it in time by ≥ 10 frames and, on ≥ half of those frames, their box centres are within d × the larger box's diagonal; d ∈ {1, 2, 3}.
6. **modelC partner (flagged, never selected or combined; in-sample):** keep only if its best base-predicate score with any partner track is ≥ q; q ∈ {0.3, 0.5}. **Only if G1 cached modelC scores for SAM 3 pairs;** otherwise skip it and say so. Do not compute features.
7. **Class gate:** keep only if EOV-MMP's R0w tracks contain its class in the same video. One setting.

**The grid (full search, fixed now):** each of rules 1, 2, 3, 5 and 7 is off or at one of its settings: 5 × 5 × 5 × 4 × 2 = **1,000 configurations**. Rules 4 and 6 are computed alone and reported, never combined.

**Criterion ("worth validating"):** a configuration passes if, on the 157 videos against EOV-MMP R0w:
- overall recall gain ≥ +8 points,
- pair recall gain ≥ +8 points,
- false-track burden ≤ 1.35×.

**A pass counts only with the class gate (rule 7) on.** Option 2 is defined as SAM 3 prompted only with the classes EOV-MMP finds in each video, which is the gate by construction. Passes without rule 7 are reported descriptively, with the note that Option 2 would then need all 35 prompts at G1's cost. Note in the report that the screen's gate uses R0w tracks (score ≥ 0.35), stricter than Option 2's planned ≥ 0.1 gate, so the screen is conservative here.

**Selection rule for the FIT validation (fixed now):** among passing configurations with rule 7 on, the one that goes forward is the one with the highest min(recall gain, pair-recall gain); ties by the lower burden, then by fewer rules on. No other configuration is carried forward.

**Fragility:** if only one or two configurations pass, flag the pass as fragile.

**Decision (binding, stated now):** pass → the selected configuration becomes Option 2, to be validated later on a fresh FIT subset. Fail → **SAM 3 closes as a detector**; no second round of rules on HELD. (SAM 3's tracker remains a candidate for Option 1; Task 3 item 10.)

**Report:**
- how many configurations were tried, how many pass (with and without rule 7), and the trade-off curve between burden and the recall gains;
- for each single rule at each setting: recall, pair recall, burden, matched tracks lost, unmatched tracks removed;
- rules 4 and 6 alone, as above;
- the class-gate split: tracks removed by novel vs base class, with matched vs unmatched counts in each cell (by construction the gate removes only classes EOV-MMP does not have in the video);
- which of the 20 inspected tracks the selected configuration removes, and which each single rule's settings remove.

**Deliverable:** `studies/postg1/sam3_screen_report.md`, verdict first.

## Task 3: Option 1 design note (no runs, except an optional smoke test)

Write `studies/postg1/option1_design.md`. Check each point against EOV-MMP's code and the tracker's API, and say where the code forces a change:

1. **Seeds:** each deep_sort tracklet *before* AFLink and the length filter, seeded at its highest-score box; the tracker propagates forward and backward from the seed. All seeds are known before tracking starts, so one session per video. Check whether the API allows adding objects after propagation starts (the design should not need it).
2. **Same object:** two tracks with vIoU ≥ 0.5 over shared frames are merged; the higher seed score wins. This replaces AFLink.
3. **End:** a track stops after K consecutive frames with no EOV-MMP raw detection (any class, score ≥ the keeping floor) at IoU ≥ 0.5, or when the tracker's own score says the object is gone.
4. **Keeping:** low-score tracklets (score between the floor and 0.35) seed only if they last ≥ N frames. Their track score is min(score, 0.35).
5. **Class and score:** the majority class and mean score of the EOV-MMP detections the track covers, as now. The tracker gives geometry only.
6. **K, N and the floor:** propose a small grid, to be fixed on HELD at tracker level (exploration), then the FIT gate.
7. **What must be rerun:** below-cut detections are not cached, so EOV-MMP's detector must run again to get them. Estimate its cost.
8. **The FIT gate (draft, not hashed):** GT pair recall ≥ +3 points over EOV-MMP **and** unmatched-track burden ≤ 1.2×, both with keeping on. The tracking-only numbers (keeping off) from the same run are reported, deciding nothing; there is no second gate. Propose the FIT subset (size, fixed seed).
9. **Cost estimate** for the HELD grid, the FIT gate and a test-set pass: GPU-hours and dollars, with the machine type.
10. **SAM 2 or SAM 3's tracker.** SAM 3 also tracks from box prompts (its tracker derives from SAM 2). Compare the two for Option 1, with no runs beyond the optional SAM 2 smoke test:
    - memory per frame and per object (SAM 3's figures from G0/G1's logs, including its three out-of-memory events on 24 GB GPUs);
    - expected speed;
    - setup and validation work (SAM 3 is already installed, pinned and checked bitwise in G0/G1; SAM 2 would need all of that);
    - whether each runs on the laptop's 4 GB GPU;
    - the protocol story (SAM 2 has no text model; SAM 3 in box mode uses no class names).

    Recommend one, with reasons. The rest of the design note applies to either.

**Optional smoke test,** in a **separate** Python environment (never EOV-MMP's), and only after asking me about any download with its size:
- If a small SAM 2 model runs on the laptop's 4 GB GPU, run it on 3 HELD videos, only to measure speed and memory. Record no metric.
- **Seeds:** use the pre-AFLink tracklets if cached. G1's R0w tracks come after AFLink and the length filter, so they are fewer than the design's seeds; if you must use them, state that the memory figure is a lower bound.
- **Report memory per frame and per object.**
- **Speed does not carry over:** the GTX 1650 has no fast bf16. Base the cost estimate on published 4090-class speeds, or state the scaling factor used.
- If it does not fit, say so and estimate from published numbers.

## Task 4: VidOR preparation — only if Task 1 says go or weak go

If Task 1 says drop, skip this task and say so.

1. **The drop list, from predicate names only, before any VidOR data is read.**
   - The VidVRD-novel predicate list comes from the split file the repo's evaluator uses; name the file.
   - VidOR's predicate names come from its paper or website list, not from data files.
   - Drop every VidOR predicate that matches or overlaps a VidVRD-novel predicate: exact match, synonym, or the same verb with another preposition. **When in doubt, drop.**
   - Write it as `studies/postg1/vidor_droplist.md`, hash it, commit.
2. **Then, from VidOR's training annotations only:**
   - **Ask me before downloading,** with the file size. Download the training annotations only; if the archive bundles validation, do not extract the validation part (val stays reserved). No videos yet.
   - Report the download size of the videos per training part, and the disk needed.
   - Report the frame rates, so a 30-frame VidVRD clip can be matched in seconds.
   - Report relation extent statistics after dropping the listed predicates: median length in seconds and in matched slots, and the share longer than one slot. Compare with VidVRD's test extents already disclosed in M1's report (median 60 frames, 77.4% longer than one slot). **No new test look. This comparison is descriptive and decides nothing.**

**Deliverable:** `studies/postg1/vidor_prep.md`.

## Final deliverable

`studies/postg1/monday_summary.md`: for each task, one paragraph and its decision, with links to the reports, the new chain head, and the commits. **Then stop.**

---

# Part 2: Reply 1 — the planning chat's response to the other chat's position (2026-10-31)

Sources: `stage2-results.md` (D1, E1, S1), `unified-model-plan.md` (D2, Q7), `rpt-g1-outcome.md`. A conversion used below, rough and labelled as such: on SGDet novel, recall 20.0% ↔ 15.79 mAP and on PredCls 40.3% ↔ 21.70 mAP, so about **0.29 mAP per recall point**. 1 recall point = about 6 of the 605 novel GT relations.

## Agreement first
- Deciding by ceiling × feasibility × cost: agreed, with one caveat (below).
- Over-long runs are the biggest single loss, and the current plan ignores them: **agreed, and this is the most important correction to my ranking.** D1 itself called it "the largest fixable loss found in the project".
- Moving Option 1 (EOV-MMP proposes, SAM 2 tracks) up: agreed.
- Splitting the class gate's effect: agreed.
- No second round of rule-tuning on HELD if the screen fails: agreed.

## Point 1: ceilings
| Oracle | Missed GT it targets (of 605) | Recall ceiling | Rough mAP ceiling | Free to measure? |
|---|---|---|---|---|
| (a) perfect tracking for EOV-MMP's own detections (tracking 11.1% + filter 14.5% of U1) | 67 GT; about 40% of these become hits once tracked, the PredCls hit rate | +4.4 points | **about +1.3** | Not quite: needs the relation stage on oracle tracks. Free if Q7 or D2 left prediction files; otherwise about 1–3 GPU-hours |
| (b) (a) + perfect keeping of below-cut boxes (+36.6% of U1) | + 96 GT, × about 0.4 | +10.7 points | **about +3.1** | as (a) |
| (c) perfect trimming of over-long runs (C1 over-long) | **129 GT** (D1: the right predicate on the right pair, in the top 20 in about 91% of GT slots; "trimmed to the GT extent, it would match") | **+21.3 points** | **about +6 (upper bound)** | **Yes:** post-processing of cached test predictions, CPU |
| (d) perfect object names (U2) | 34 GT (51 of 54 class mismatches involve a novel class) | at most +5.6 points | at most about +1.6 | **Yes:** rename matched tracks, CPU |

- The buckets are exclusive, so recall gains add. mAP does not add linearly, because everything competes in each video's top-200.
- **The caveat on "ceiling × feasibility":** the time side has the largest ceiling *and* the worst record. The merge oracle (+0.07), the span head (AUC 0.696 against a trivial 0.694), E1's added runs (−2.31) and **E1's single tighter cuts (k = 10, 5, 3, 1 as replacements, which also failed to beat T1)** all point the same way. Replacing runs label-free has already been tried and failed. Only a *learned* boundary from real extents is untried. Feasibility must be weighted by that record, not only by the ceiling.

**The +0.07 and +12 numbers are consistent** (D1 reconciled them). The merge oracle joins fragments, which are rare (0.5%), so it gains nothing. Loosening vIoU to 0.1 accepts over-long runs as matches, so it gains a lot. +12 is a looser bound than (c): it also forgives badly placed boxes and partial overlaps. The true trimming ceiling is below +12, plausibly around +4–6. Only recall-level trimming has been measured (D1's 129 GT); **the mAP effect has not.** That is the first thing to compute.

## Point 2: learning relation ends from VidOR
**Worth it, but gate it hard.** It is the one untried route at the biggest ceiling, and it needs no SAM 3 and no supervisor question.

What makes it feasible:
- VidOR's relations have real start and end frames.
- EOV-MMP's features are frozen CLIP and TagCLIP, read only at clip middle frames (about 1.3% of frames, per G0). So extracting features on a VidOR training part is cheap: likely under 5 GPU-hours. The model is small. **Total about $5–15**, plus the VidOR download and 2–3 weeks of work.

Three conditions:
1. **Drop VidOR relations whose predicates match VidVRD-novel ones** (e.g. bite, feed, hold, kick, chase, if present) with a mapping fixed before any data is seen.
2. **Replace, never add**; the E1 lesson.
3. **Validate on a held-out VidOR training split, never VidOR val** (reserved). The gate must beat the trivial baseline that S1 tied, the cosine distance between consecutive clip embeddings, by a clear margin. Proposal: ≥ +0.05 AUC on continue-versus-change.

**A risk to state now:** over-long runs arise because a *predicate* stays in the top 20 after its relation ends, while the pair's other relations may continue. A predicate-agnostic "the pair's relation changes here" model may cut the wrong runs. It may need to be run-specific: given this predicate's run, where does it end?

## Point 3: Option 1 moves up
**Agreed**, and it should be the default clean component.
- SAM 2 takes boxes, not names. Its training data has no class labels, so it brings no novel-name knowledge. That is a weaker exposure than EOV-MMP's own CLIP.
- **But its ceiling is small:** about +1.3 mAP for perfect tracking alone, about +3.1 with perfect keeping. Realistically about +0.5–1.
- On FIT, both sides use EOV-MMP's in-sample detections, so the in-sample bias mostly cancels and the comparison is fair.

## Point 4: the class gate
**Agreed.** The extras are mostly person, bird and car, classes EOV-MMP usually detects in the same video, so the gate barely touches them, while it can delete SAM 3's unique novel-object finds. Report it split by novel versus base, and by whether the class is present in EOV-MMP's tracks. The geometric error rules (tiny boxes, edge slivers, fragments) target the observed errors better than the gate does.

## Point 5: the margin
**Agreed:** +9.8 and +9.2 unfiltered against a +8 floor leaves only about 1–2 points to lose. Fixed now: if the screen fails, SAM 3 closes, with no second round on HELD.

## Point 6: FIT is fresh only for SAM — I disagree that it needs a stricter threshold
- The in-sample bias on FIT runs **against SAM 3** on both bars. EOV-MMP's recall is inflated, and its false-track count is flattered, because its detector learned to ignore unannotated objects in these very videos. **So a pass on FIT is conservative and trustworthy; a fail is not fully conclusive.** No stricter threshold is needed.
- A VidOR part makes EOV-MMP out-of-sample, but changes the annotation style and needs an 80 → 35 class mapping. Use it as an optional fairness check, not as the gate.
- For Option 1 the issue barely arises: both sides use EOV-MMP's detections.

## Point 7: the combined path, and my ranking
The pieces target different buckets: T1 (ranking, +0.36 measured), Option 1 (tracking and filter U1, about +0.5–1 estimated) and trimming (C1 over-long, unknown).

- Without trimming: about 16.15 + 0.5–1 ≈ **16.7–17.2**. That is a coin flip against 16.74.
- If a learned trimmer recovers even 20–30% of over-long cases without adding runs (about +1–1.5): **about 17.5–18.**
- **So trimming decides whether there is a safe margin.** A combination of T1, Option 1 and trimming without SAM 3 is more realistic than Option 2, *if* trimming's ceiling and gate hold.

| Rank | Step | Kill criterion (proposed; to agree) |
|---|---|---|
| 1 | **Free oracles on Monday** (below) | the trimming ceiling at slot resolution < +2 mAP SGDet novel → drop trimming |
| 2 | **Trimming from VidOR extents** | on held-out VidOR, continue-versus-change AUC does not beat the cosine baseline by ≥ 0.05 → drop |
| 3 | **Option 1, EOV-MMP proposes and SAM 2 tracks** (in parallel; safe and additive) | FIT tracker gate: GT pair recall not ≥ +3 points over EOV-MMP at a burden ≤ 1.2× → drop |
| 4 | **Option 2, SAM 3 filtered** | the screen as defined; if it fails, SAM 3 closes |
| — | **Tracklet keeping (A27)** | subsumed by Option 1 plus a keeping rule |

## The cheapest first step, on Monday: all CPU, $0, one Claude Code session
1. **Trimming oracle**, pre-registered as a disclosed analysis like D1 and D2. On the cached T1 test predictions, report SGDet and PredCls novel and all mAP with:
   - (i) runs that match a GT triplet trimmed to the exact GT extent;
   - (ii) the same, trimmed to slot resolution, which is the realistic best a slot-based trimmer can do;
   - (iii) (ii) with ±1-slot boundary jitter. If the gain collapses here, a learned trimmer has little room.

   Scores unchanged, nothing added.
2. **Naming oracle (d):** tracks matched to a GT object (class-agnostic, vIoU ≥ 0.5) get the GT name; report mAP.
3. **(a)/(b) mAP:** if Q7 or D2 left prediction files, re-evaluate them for mAP. Otherwise report the GPU estimate only, and rent nothing.
4. **The SAM 3 screen on HELD,** with the class gate split by novel/base and by class present/absent.

These oracles use test labels for analysis only: no method or parameter is chosen from them except go or no-go. Disclose them like D1 and D2. They add looks at the test labels, not test evaluations of a method.

---

# Part 3: Reply 2 — the other chat's response to Reply 1 (2026-10-31)

Point by point. Agreements are short; disagreements say why and propose a fix.

## Overall
Agreed on the main shift: **trimming over-long runs decides whether there is a safe margin**, and the path to beat METOR is most likely T1 + Option 1 (SAM 2) + learned trimming, with no SAM 3. My main disagreement is about **test-label looks** (Monday step): I propose exactly one, not three.

## Point 1: ceilings
- **The table: agreed.** The 0.29 mAP-per-recall-point conversion is rough (mAP depends on ranking inside each video's top-200), but adequate for go/no-go. The reconciliation of +0.07 vs +12 is convincing.
- **Your caveat about the time side's record: agreed, and it matters.** Label-free replacements (k = 10, 5, 3, 1) already failed. Only a *learned* boundary from real extents is untried.

## Point 2: learning ends from VidOR
Agreed, with your three conditions and your run-specific warning. Two further risks, which change the design:

1. **Annotation-convention mismatch.** A VidOR-trained trimmer learns *VidOR's* idea of where a relation ends. VidVRD's test extents follow VidVRD's annotators, and VidVRD's training labels are 30-frame chunks, so no VidVRD training data shows VidVRD's long-extent convention. The VidOR held-out gate cannot check transfer to VidVRD; only the test set can. **Fix:** prefer a *conservative* trimmer (cut only where the predicate's own clip score clearly drops), and use the trimming oracle's ±1-slot jitter variant (iii) to see how much boundary error the gain tolerates. If (iii) collapses, a trimmer trained on another dataset's convention is unlikely to land within tolerance.
2. **Run-specific, not pair-level.** Agreed with your warning, and I would make it the design, not a risk: given a predicate's run on a pair, predict from that predicate's own per-clip scores (and the clip embeddings) whether the run continues. EOV-MMP's open-vocabulary head can score VidOR's predicates, so this input exists on VidOR. Predicate identity must not be an input (it would not transfer to VidVRD-novel predicates).

**Practical check before committing:** the VidOR download size and disk needs for a training part. Fix the novel-predicate drop list from names only, before any VidOR data is read.

## Point 3: Option 1 moves up
Agreed: SAM 2 takes boxes, has no class knowledge, and is the protocol-cleanest component. Two notes:
- **Most of its ceiling is in keeping below-cut boxes** (+3.1 with perfect keeping against +1.3 for tracking alone). Keeping is exactly where E1-style crowding returns. The design must say which below-cut boxes seed SAM 2 tracks, and score them so they cannot crowd confident pairs (as RPT's `min(score, cut)` rule did).
- **The seeding design is open:** which detections start a SAM 2 track (each new deep_sort track's first confident box?), when two seeds are the same object, and when a track ends. This should be a short design note before the FIT gate is written.

## Point 4: class gate
Agreed. Geometric rules first; gate split by novel/base and by class present/absent.

## Point 5: margin
Agreed: if the screen fails, SAM 3 closes, with no second round on HELD.

## Point 6: FIT
**I accept your argument.** The in-sample bias on FIT works against SAM 3 on both bars, so a pass is conservative. No stricter threshold; the VidOR part is an optional fairness check.

## Point 7: combined path and ranking
Agreed with the ranking: 1 oracle, 2 trimming, 3 Option 1 in parallel, 4 SAM 3 screen, A27 subsumed. Two additions:
- **The pieces interact through the top-200.** Option 1 adds pairs; trimming changes run lengths. The final test session must include each piece alone and combined (pre-registered rows), so the paper can show which part carries the gain.
- **Kill criteria: agreed as proposed**, with one change to Option 1's: GT pair recall ≥ +3 points over EOV-MMP **and** an unmatched-track burden ≤ 1.2×, **both measured with the keeping rule on**, since that is where its gain and its risk are.

## Monday: where I disagree — one test-label look, not three
The project's rule is that test labels are touched only for disclosed diagnostics (D1, D2) and one final method evaluation. Each extra look is small, but they add up, and three go/no-go decisions taken from test labels is a real forking path that a reviewer can raise.

- **Keep (1), the trimming oracle on test**, as one pre-registered, disclosed analysis like D1. It has no substitute: HELD's GT relations are 30-frame chunks, so a trimming oracle on HELD would be meaningless, and VidOR has another convention. Pre-register:
  - the three variants (i)–(iii) exactly as you wrote them;
  - **the single decision:** go if (ii), slot resolution, gains ≥ +2 mAP SGDet novel; otherwise drop trimming;
  - **no tuning from it:** the trimmer's design and thresholds are never chosen from (i)–(iii). Variant (iii) is reported, not used to tune tolerance;
  - recorded in a test-looks ledger next to D1 and D2.
- **Drop (2), the naming oracle, for now.** U2 is SAM 3's territory; if the screen fails, SAM 3 closes and (d) decides nothing. Run it only if the screen passes, and then on HELD (GT names exist there) rather than on test.
- **Drop (3), the (a)/(b) mAP oracles on test.** Option 1's decision is the FIT gate, not a test ceiling, and the recall-level ceilings from D2/Q7 already exist. A test-label look that decides nothing should not happen.
- **Keep (4), the SAM 3 screen on HELD**, as agreed.
- **Add:** a design note for Option 1's seeding and keeping (no runs), and the VidOR size/disk check plus the novel-predicate drop list (names only).

## Also before any rental
- Supervisor meeting next week: present T1 + Option 1 + trimming, and ask whether SAM 2 as a box-to-track front end is acceptable (it is class-agnostic, so it is a weaker question than SAM 3's).
- The follow-up email to METOR's first author.

## Questions back to you
1. Do you accept one test look instead of three? If not, which of (2)/(3) changes a decision, and how?
2. Do you agree the trimmer should be run-specific with predicate identity excluded, and conservative by design?
3. Do HELD's or VidOR's annotations give any way to check VidVRD's long-extent convention without test labels? If you know of one, it would remove the main risk in point 2.
4. Option 1's seeding: do you have a preferred design, or should Claude Code propose one from the code?

---

# Part 4: Reply 3 — the planning chat's reply to Reply 2 (2026-10-31)

Sources: `stage2-results.md` (S1 pre-checks, D1, E1), `ov-pairtime-proposal.md` (M1 test extents), `rpt-g1-outcome.md`.

## Summary
We agree on almost everything. I propose three changes:
1. **One test look: accepted.** But variant (iii) should be part of the single decision, redefined as "one slot too long", because a conservative trimmer errs in that direction.
2. **"Conservative" must not mean "cut where the predicate's own score drops".** That rule is close to E1's single tighter cuts, which already failed on test. The learned trimmer must beat that rule, as a second baseline on VidOR.
3. **In the final test session, name one primary row in advance** (the full combination). The single-piece rows are report-only ablations.

**Proposed close.** If you accept these, I consider the plan agreed. You write Monday's Claude Code prompt, and I review it before it runs.

## Point by point
- **Overall, points 1, 4, 5 and 6:** agreed, nothing to add.
- **Point 2, convention mismatch:** agreed. It is the main risk; see Q3 for what can and cannot be checked.
- **Point 2, run-specific design:** agreed; see Q2 for the inputs.
- **Point 2, practical checks:** agreed, with three additions:
  - The drop list follows "when in doubt, drop". A leak costs more than lost data.
  - **Use VidOR's GT tracks to compute the trimmer's training features.** Then no detection or tracking runs on VidOR; only the frozen features at clip middles and modelC's scores are needed. The known gap: the trimmer trains on cleaner tracks than SGDet gives it on VidVRD.
  - Match clip duration: resample VidOR so that a 30-frame clip covers the same time as on VidVRD.
- **Point 3:** agreed. Two facts for the design note:
  - Keeping needs a short rerun of EOV-MMP's detector, because below-cut detections are not cached. G1 already flagged this for the ≥ 0.1 gate. Include it in Option 1's cost.
  - The keeping floor is chosen on HELD at the tracker level, not from D2's test numbers.
  - Seeding: see Q4.
- **Point 7, Option 1's kill criterion** (pair recall ≥ +3 points and burden ≤ 1.2×, both with keeping on): agreed. The tracking-only numbers (keeping off) from the same run are reported but decide nothing. There is no second gate.
- **Point 7, test rows:** agreed, with the primary-row condition. SGDet novel bootstrap CIs are about ±5 mAP (E1). Picking the best of four test rows afterwards would be exactly the forking path you warned about.
- **Supervisor and the follow-up email:** agreed.

## Monday: one look, accepted, with the decision rule changed
- **(2) Naming oracle: drop entirely, not just defer.** G1 showed SAM 3 does not name better (46 misnamed tracks against EOV-MMP's 40), and relation-aware naming is closed. No candidate method exists whose fate (d) could decide.
- **(3) The (a)/(b) mAP oracles: drop.** Agreed.
- **(1) Trimming oracle: pre-registration details**, added to yours:
  - **Matching:** D1's code, so the result connects to D1's 129 GT.
  - **Trim, never split:** each matched run is trimmed to the single GT extent it overlaps most. Splitting a run around two GT instances adds instances, which was E1's failure.
  - **Scores and top-200 membership unchanged:** only the extents change.
  - **Variants:**
    - (i) exact GT extent;
    - (ii) snapped to the slot grid;
    - **(iii) slot grid, then one slot longer at each end** (within the original run);
    - (iv) one slot shorter at each end, report-only.

    This replaces symmetric random jitter. A conservative trimmer stops late, so "one slot too long" is the error that matters.
  - **The single decision:** go if (ii) ≥ +2 **and** (iii) ≥ +1 SGDet novel mAP. It is still one look and one decision. It uses (iii) the way your point 2 does, as the check on boundary tolerance, not to tune anything.
  - **Reading rule, stated now:** a learned trimmer recovers only part of the ceiling. So (ii) between +2 and about +4 is a weak go. Writing this down now avoids arguing about it afterwards.
  - **Also report (i) − (ii).** If it is about 0, test boundaries sit on the slot grid, which suggests the test extents were built from chunks.

## Answers to your questions

### Q1. One look instead of three?
Yes. Neither (2) nor (3) changes a decision.

### Q2. Run-specific, predicate identity excluded, conservative?
**Run-specific: yes.**

**Predicate identity excluded: yes, and by default the predicate's text embedding too.** VidOR has few predicates to learn from. M1 showed that transfer to unseen predicates through a text head is weak and unstable across seeds (novel 5.7 / 7.5 / 9.0). Proposed inputs, all predicate-agnostic:
- the run's own score in each clip, divided by the run's peak, so it is scale-free;
- its percentile rank among all predicates in that clip. The rank is the direct cause of over-long runs (the top-20 cut). Use a percentile because the two vocabularies differ in size;
- the cosine change between consecutive clip embeddings of the pair.

**Conservative: yes, but defined as an operating point, not a hand rule.**
- Fix the cut threshold on held-out VidOR at high precision for "this run has ended" (for example ≥ 0.9), before any VidVRD result exists.
- A hand rule such as "cut where the run's own score drops" is close to E1's single tighter cuts (top-k replacements), which failed on test.
- **So the VidOR gate gets two trivial baselines:**
  - the cosine change (S1 tied it at 0.694);
  - the own-score or rank drop (the E1-like rule).

  The learned trimmer must beat both by ≥ 0.05 AUC. If it cannot beat the second, it has learned nothing that E1 did not already try.

### Q3. A way to check VidVRD's long-extent convention without test labels?
**No clean check exists.** What the data allow:
- **HELD, extents: no.** Every training instance is exactly 30 frames on a 15-frame grid, with whole-chunk gaps (median 4 missing chunks). The gaps are unlabelled, not "the relation ended". Merging chunks cannot rebuild VidVRD's long extents.
- **HELD, ends: weak.** S1's 3,292 known transitions (adjacent labelled chunks where the labels change) are real decisions by VidVRD's annotators, at chunk level. But modelC's scores on HELD are in-sample, so a trimmer that reads scores cannot be judged there honestly. Descriptive at most; not a gate.
- **VidOR against test statistics already disclosed: free but weak.** M1's report already gives VidVRD's test extents: median 60 frames, 77.4% longer than one slot. Compare VidOR's training extents in seconds and in slots, after matching clip duration. This costs no new test look.

  If VidOR's extents are much longer, a VidOR trimmer will stop late. That is the conservative direction, and (iii) prices it.
- **Inside the agreed look:** (i) − (ii), as above.

So the transfer risk stays until the final test. (iii) is the best proxy, which is why it should bind.

### Q4. Option 1's seeding
**A preferred design.** Claude Code should check it against the code and the SAM 2 API, then write the note, with no runs:
1. **Seeds:** each deep_sort tracklet before AFLink and the length filter, seeded at its highest-score box. First boxes are often partial, as objects enter the frame. SAM 2 propagates forward and backward from the seed.
   - All seeds are known before SAM 2 starts, so one SAM 2 session per video suffices.
   - Check whether the API allows adding objects after propagation starts. This design does not need it.
2. **Same object:** two SAM 2 tracks with vIoU ≥ 0.5 over their shared frames are merged, and the higher seed score wins. This replaces AFLink.
3. **End:** a track stops in either case:
   - after K consecutive frames with no EOV-MMP raw detection (any class, score ≥ the keeping floor) at IoU ≥ 0.5;
   - when SAM 2's own presence score says the object is gone.

   Requiring the detector's agreement stops SAM 2 drifting onto background. That is the same failure as SAM 3's long edge slivers (83–149 frames).
4. **Keeping:** low-score tracklets seed only if they last ≥ N frames. Their track score is min(score, 0.35), so they never outrank a confident track.
5. **Class and score:** the majority class and mean score of the EOV-MMP detections the track covers, as now. SAM 2 contributes geometry only.
6. **K, N and the floor:** a small grid fixed on HELD at the tracker level (exploration), then the FIT gate.

**Cost note:** SAM 2 encodes each frame once and adds a small cost per object, so there is no 35× factor as with SAM 3. Claude Code can check whether a small SAM 2 model runs on the laptop's 4 GB GPU for a 3-video smoke test. The rental estimate is then measured, not guessed.

## Monday list (proposed final)
1. **Trimming oracle on test.**
   - The pre-registration note is committed and hashed first.
   - One disclosed look, with the decision rule above.
   - An entry in the test-looks ledger.
2. **The SAM 3 screen on HELD**, as agreed.
3. **Option 1 design note** (Q4), with no runs. Optional: the laptop smoke test of SAM 2.
4. **VidOR:**
   - first, the drop list from predicate names only, hashed;
   - then the download size per part and the training-extent statistics (Q3).
5. **Not on Monday:** the follow-up email to METOR's author and the material for the supervisor meeting.

---

# Part 5: The planning chat's review of the prompt (2026-10-31; all 17 fixes are applied in Part 1)

**Verdict:** the plan is as agreed. Fix the items below before it runs. Items 1–3 and 7–8 are the important ones; the rest are short additions.

## Task 1: trimming oracle

1. **The T1 reproduction is itself a test-label look, so it runs after A30.** The prompt puts "re-evaluate unchanged; must reproduce 16.15 / 27.57" under Inputs, before "Write A30 first". Make it the first step of the single guarded run, after A30 is hashed and committed.
2. **Never trim a run that is already a hit.** A run can match one GT (vIoU ≥ 0.5) while also being "over-long" for another GT instance of the same triplet. Trimming it to the second can lose the first. Rule: runs that are true positives under the unchanged evaluation are left alone. After trimming, the evaluator matches as usual (one GT per prediction, its normal order).
3. **D1's buckets were computed on the baseline (15.79), not on T1.** Re-run D1's classification on the T1 predictions inside the same run, and report the over-long count next to D1's 129.
   - D1 covers the 605 novel GT only. If D1's code can classify all GT, apply the rule to all GT for the "all" numbers. If it cannot, trim only the novel-matched runs and label the "all" numbers as such. The decision uses SGDet novel only, so nothing changes in the rule.
4. **Crash rule.** If the script fails before any metric is written, it may be fixed and rerun. Once a metric file exists, there is no rerun. Record this in the ledger entry.
5. **Add PredCls novel and all, report-only** (agreed in reply 1), from the cached T1 PredCls predictions (22.25 / 38.03). If they are not cached, skip it; no GPU.
6. **Record the slot length in A30 before the run** (15 or 30 frames, as `gen_feats_test` builds clips). The rule stays fixed whatever it is: a real trimmer has exactly this resolution. With 30-frame slots, (iii) is a harsher test. That is fair, and saying so in advance avoids arguing later.

## Task 2: SAM 3 screen

7. **The combinations as written will fail mechanically.** Each rule's "selected value" is its most aggressive setting that still keeps recall ≥ +8. Combining two such settings removes more matched tracks, so recall drops below +8. A real pass could be missed for that reason alone.
   - **Fix:** search the full grid. Each of rules 1, 2, 3, 5 and 7 is off or at one of its settings: 5 × 5 × 5 × 4 × 2 = 1,000 configurations, seconds on CPU.
   - Report:
     - how many configurations were tried;
     - how many pass;
     - the trade-off curve between burden and recall.
   - If only one or two configurations pass, flag the pass as fragile.
   - A lucky pick is caught later by the fresh FIT check. The screen is a futility check, so it should not kill SAM 3 through its own design.
   - *(Added by the other chat when applying: a fixed selection rule for which passing configuration goes forward; see Part 1, Task 2.)*
8. **A pass counts only with the class gate (rule 7) on.** Option 2 is defined as SAM 3 prompted only with the classes EOV-MMP finds in each video, which is the gate by construction. A pass without rule 7 is reported descriptively, with the note that Option 2 would then need all 35 prompts at G1's cost.
   - Also note that the screen's gate uses R0w tracks (score ≥ 0.35), which is stricter than Option 2's planned ≥ 0.1 gate. So the screen is conservative here.
9. **Rule 6 needs modelC scores for SAM 3 pairs.** If G1 did not cache them, skip rule 6 and say so; it is report-only. Do not compute features.
10. **Wording:** "for each single setting, its selected value" should read "for each rule, its selected value".

## Task 3: Option 1 design note

11. **The smoke test runs in a separate Python environment.** Never install into or upgrade EOV-MMP's environment: the bit-exact reproductions depend on it.
12. **Treat the smoke-test numbers carefully:**
    - **Speed:** the GTX 1650 has no fast bf16, so its speed does not carry over to a 4090. Base the cost estimate on published 4090-class speeds, or state the scaling factor used.
    - **Memory:** report memory per frame and per object.
    - **Seeds:** G1's R0w tracks come after AFLink and the length filter, so there are fewer of them than the design's seeds. Use the pre-AFLink tracklets if cached; otherwise state that the memory figure is a lower bound.

## Task 4: VidOR prep

13. **Downloads need the user's OK first, with the file size.**
    - Download the training annotations only. If the archive bundles validation, do not extract the validation part.
14. **Name the source of the VidVRD-novel predicate list:** the split file the repo's evaluator uses.
15. **State that the extent comparison is descriptive and decides nothing.**

## Ledger and setup

16. **The ledger should list every test-label access in the chain, not only the names given.** That includes:
    - the baseline reproduction;
    - the pilot's merge oracle (+0.07) and loose-vIoU (+12) numbers;
    - G0's session 3, if it touched test labels;
    - D1, D2, Q7, B, B2, T1, E1 and M1.

    Mark any entry it is unsure of rather than leaving it out.
17. **Before Monday, the user copies the three reply docs into `studies/postg1/`.** Claude Code cannot read the project docs. *(Superseded: this one file now contains them, as Parts 2–4.)*
