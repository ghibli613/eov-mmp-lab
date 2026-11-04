# Next session (v2): HELD split, A30 scope, naming check, D7 check preparation

> **DECISIONS (set by Giang):**
> 1. **Task 1, the trimming oracle: NO.** It is not run. Mark its pending entry in `studies/test_looks_ledger.md` as "not run by decision; no test label accessed". No test-label access happens in this session.
> 2. **A30's scope: option (a).** A30's verdict stands for the 35-name SAM 3 with rules. PaD Phase 1 is a system A30 did not test. This is written after A30's result and disclosed (Step 2).

You are in `~/Documents/thesis/source/ov-vidvrd-lab`. Read first:
- `studies/postg1/dev_review_pad.md`, your own review. This session applies it as amended by `studies/postg1/dev-review-reply-1.md` (I will copy it in).
- `studies/postg1/monday-pack.md` (Task 2, A30's rule) and `studies/postg1/sam3_screen_report.md`.
- `studies/postg1/dev-review-pack.md` Part 2, the PaD proposal.

**Phase names (fixed):**

| phase | what it is | at test time |
|---|---|---|
| **Phase 1, Prompt** | the main method | SAM 3 |
| **Phase 2, Filtered Distill** | EOV-MMP's detector learns from Phase 1's boxes, cleaned by P3 | EOV-MMP's detector |
| **Phase 3, Split Distill** | box-only distillation plus gap fill by SAM 3's tracker; EOV-MMP's own score judges junk, no P3 | EOV-MMP's detector |
| **Phase 4, Improve** | further ideas, only if needed | — |

**Gates:** 0–3 are Phase 1; 4 is Phase 2; 5 is Phase 3.

**In short:** Phase 1 is the lead method. It has:
- a learned class-free prompt for SAM 3 (P1);
- a learned keep/drop model (P3);
- naming by EOV-MMP's auxiliary classifier, modelB (P4);
- EOV-MMP's relation classifier with T1 (P5).

This session does four things, all on the laptop, for $0:
- **Step 1:** splits HELD;
- **Step 2:** writes A30's scope amendment;
- **Step 3:** runs a naming check;
- **Step 4:** prepares a small paid D7 check (PaD gate 1).

**No rental happens without my approval. Nothing is pushed.**

## Ground rules
1. **Never edit the authors' code** (the files from `fdeeeb4`).
   - New code goes in `studies/postg1/` or `studies/d7/`.
   - Never push.
   - Run a credential scan before every commit.
2. **Test labels:** the test-label guard stays on all session.
3. **HELD-B is sealed after Step 1** (the guard is defined in Step 1). It stays on for this whole session.
4. **Amendments first.**
   - Each step that decides something gets its amendment, appended after the chain head: **A30, `b10cf7a7…0aae`**, unless later ones exist.
   - Labels are the next free ones, from **A31**.
   - Hash each amendment, check it with `verify_after_A26m.py`, and commit it **before** what it governs runs.
5. **No design choices beyond what is written here.** If something cannot be done as written, stop that step and report.
6. **Environment:**
   - Never install into or upgrade EOV-MMP's Python environment.
   - No downloads without my OK, with the size stated first.
   - **Never run SAM 3 on the laptop.**
   - CLIP and modelB may run on the laptop's GPU in EOV-MMP's existing environment, with nothing installed.
7. **Rentals, if I approve one later:**
   - set an auto-stop;
   - copy back and verify every hash;
   - then tell me at once so I destroy the machine.

## Step 1: split HELD, extracts, guard, FIT counts (CPU)
Write the amendment first. It fixes the items below.

**The split.** The 160 HELD videos go into **HELD-A (80)** and **HELD-B (80)**.
- **Flags:**
  - "novel": the video has at least one novel-class GT object. This reads all 160 HELD annotations; it happens before the guard is installed, and the amendment says so.
  - "long": the frame count, from the video's frame folder, is **strictly above** the median of the 160.
- **Strata, in this fixed order:** (novel, long), (novel, not long), (not novel, long), (not novel, not long).
- **Procedure:**
  - one generator, `numpy.random.default_rng(20261102)`;
  - for each stratum in that order: sort its video IDs, then take `rng.permutation` of them;
  - concatenate the four permuted lists in stratum order;
  - assign A, B, A, B… along the whole list, starting with A. The alternation does not restart per stratum, so the split is exactly 80/80.
- **Record** where the 3 videos without SAM 3 files fall: `…00280000`, `…00225001`, `…00971000`.

**The roles:**
- **HELD-A:** prompt selection, the naming check, and P3's training (cross-validated by video inside HELD-A).
- **HELD-B:** sealed. It serves at most three pre-registered, tracker-level gates, with nothing ever tuned on it:
  - **gate 3:** the Phase 1 (Prompt) system against EOV-MMP R0w;
  - **only if gate 3 passes, gate 4:** the Phase 2 (Filtered Distill) student against R2;
  - **only if gate 3 passes, gate 5:** the Phase 3 (Split Distill) student against R2.
  - Each gate gets its own amendment before it runs. All uses are disclosed. No other use is allowed.

**Gate 3, fixed now.** A later amendment may only make it stricter.
- It is tracker-level, not mAP, since modelC is in-sample on HELD.
- It uses G1's matching (A29 rules), class-aware, against EOV-MMP R0w on HELD-B:
  - overall recall gain ≥ +8 points;
  - pair recall gain ≥ +8 points;
  - burden ≤ 1.35×.
- Class-agnostic numbers are reported.
- **No skips:** gate 3 runs every HELD-B video, on a 48 GB GPU planned in advance.

**P3 follows option (b).** SAM 3 tracks that match a novel GT object are unlabelled like any other unmatched track, and never excluded. Novel GT is used only to report suppression.

**Disclosure, in the amendment:**
- HELD-B is sealed **from now on**, not unseen. G1, A29d's futility rule, the descriptive analysis (including the 20 inspected tracks) and A30's screen all used it.
- P3's starting rule family (tiny, edge sliver, fragment) was seen on all of HELD.

**Extracts, then the guard:**
1. Write hashed **FIT-only** and **HELD-A-only** extracts of every aggregate file that contains HELD-B. At least these (list any others you find):
   - `train_object_trajectories_gt.json`;
   - `pilot_analysis/stage2/data/held_gt.json`;
   - `studies/rpt/g1/results_futility/*.json`;
   - `studies/rpt/g1/results_descriptive/descriptive.json`;
   - `studies/postg1/results_screen/sam3_screen.json`;
   - `studies/rpt/g1/viz_unmatched/`;
   - `held_gt_missing3.json`.
2. Install the HELD-B guard: an audit hook like the test-label guard. It refuses:
   - HELD-B's per-video annotations, G1's SAM 3 outputs and EOV-MMP outputs for HELD-B videos;
   - **every aggregate above, by path.**

   Later code reads the extracts only.

**FIT counts (report only, from the FIT-only extract):** on FIT's frames sampled 1 in 30, report:
- the share of sampled frames with no GT box;
- the share of sampled (frame, track) pairs inside an annotation gap of that track;
- the distribution of track coverage.

**Commit:**
- `studies/d7/held_split.json`, hashed;
- the extracts and their hashes;
- the guard and its tests.

**Deliverable:**
- the counts per stratum;
- the halves of the 3 videos without SAM 3 files;
- the FIT counts;
- the amendment's hash.

## Step 2: A30's scope amendment (no run)
**Do not re-run Task 2.** Write one amendment stating:
- **A30's verdict stands for the system it tested:** the 35-name SAM 3 with fixed rules and the class gate. Option 2 and C6b (that system's boxes as a teacher) are closed.
- **PaD Phase 1 is a system A30 did not test:**
  - one learned prompt, trained on FIT;
  - a learned keep/drop model;
  - naming by modelB;
  - **no class gate.** A30's fail came from that gate (+9.80 → +4.90).
- **This scope statement was written after A30's result,** and the thesis says so in the method chapter.
- **P3 is a learned model,** fitted on P1's own HELD-A tracks and judged on HELD-B. Its rule family is reused and its thresholds are refitted; no threshold from the 35-name tracks is carried over.
- **The geometric rules' effect is already in `sam3_screen_report.md`.** It is the descriptive starting point for P3; nothing new is computed.

**Deliverable:** the amendment's hash.

## Step 3: the naming check (laptop, $0; CLIP and modelB only, no SAM 3)
**Purpose:** gates 2 and 3 are class-aware, and modelB has never named objects alone. Measure that before renting anything.

Write the amendment first. It fixes the items below.

**Data:**
- G1's 35-name SAM 3 tracks on HELD-A videos with SAM 3 files, from the HELD-A extract and per-video files;
- the same confident, deduplicated track set that G1 counted (A29 rules).

**Naming (P4), as in your review §B6:**
- for each track, on **every detected frame**:
  - the box's pooled TagCLIP `patch_proj`, as `studies/rpt/relation.py` computes it;
  - then modelB, loaded as EOV-MMP loads it, then `new_order`;
- the track's 35-vector is the mean over its detected frames; its name is the argmax;
- no fusion with any decoder score.

**Measures,** class-aware with A29's matching, on the same videos:
- **(i)** the tracks with **the prompt names** (G1's);
- **(ii)** the same tracks with **modelB's names**;
- **(iii)** EOV-MMP R0w.

For each, report:
- object recall, overall and base / novel;
- pair recall;
- class-aware burden.

Also report:
- modelB's naming accuracy on the tracks matched class-agnostically, base / novel;
- modelB's top-1 on HELD-A GT boxes, base / novel.

**Stop rule (binding, stop and report; not a kill):**
- **If (ii)'s object-recall gain over (iii) is less than half of (i)'s gain:** stop. Do not prepare the rental request in Step 4; report. Naming is then the bottleneck, and P4 is redesigned first.
- **If (i)'s gain is ≤ +2 points:** stop and report. The check cannot discriminate.

**Disclose:** modelB and R0w are both in-sample on HELD's base objects; the novel split is the honest part.

**Deliverable:** `studies/d7/naming_check_report.md`, verdict first, with runtime.

## Step 4: prepare the D7 check, PaD gate 1 (CPU here; the run needs my approval)
**Purpose:** a cheap kill check. Can SAM 3 find VidVRD's objects **without class names**?

Write the amendment first. It fixes the items below.

**The videos:**
- 20 HELD-A videos with SAM 3 files, in the order of `numpy.random.default_rng(20261103)` over the sorted HELD-A list. There is **no length exclusion**.
- **The guard, computed now from G1's committed files, before any rental:**
  - compute the 35-name SAM 3's **class-agnostic** object-recall gain over R0w on those 20 videos;
  - if it is below **+5 points**, use **40 videos** (the next 20 in the same order);
  - if it is still below +5 on 40, or ≤ 0, **stop and report**: the check cannot discriminate.
  - Record the gain and the final list. No arm's result exists at this point.

**The arms,** both SAM 3 at G1's settings (defaults, `facebook/sam3` revision and SHA256 as A28/A28a, the G1 stack):
- **(A)** the single prompt "object";
- **(B)** the four prompts "person", "animal", "vehicle" and "object", one pass each. Their masklets are merged by A29's duplicate rule (mean mask IoU > 0.7, the higher score kept).
- **The code:** a new entry point for non-class prompts, reusing `run_video`, G1's driver and machine scripts, and `studies/rpt/tests/fake_sam3_worker.py`. The adapter gets a class-free path.

**Machine and memory:**
- One 48 GB Ada-generation GPU (L40, L40S or RTX 6000 Ada), ≥ 64 GB RAM, the G1 template and stack.
- One SAM 3 process. No A2; A1 only if its bitwise check passes, as in G1.
- **No skips.** If a video runs out of memory even on 48 GB, stop and report.

**Check 0, first on the machine:**
- the first G0 smoke video that is in **HELD-A**, at defaults with the 35 names, compared with G0's reference output;
- **if bitwise identical:** G1's 35-name files are the reference;
- **if not:** do not stop. Run the 35-name SAM 3 on the final list on this machine, and use that as the reference for the rule. Report G1's numbers alongside. The guard is not recomputed, since it only fixed the list.

**The measures,** all **class-agnostic**, with A29's matching (cut-vIoU on option-B runs). Four systems on the same videos: arm A, arm B, the 35-name reference and EOV-MMP R0w.
- object recall, overall and base / novel;
- pair recall (not U1);
- burden (unmatched confident tracks per video);
- **recall at an equal track budget (reported, not gated):** per video, each system's confident tracks are cut by A29's track score to the 35-name reference's count of confident tracks;
- **per pass:** seconds per frame, masklet count and peak GPU memory.

**The rule (binding):**
- "The better arm" is the arm with the higher class-agnostic object-recall gain over R0w.
- D7 continues if that gain is **at least half** of the 35-name reference's gain on the same videos.
- Burden, pair recall and equal-budget recall are reported, not gated.

**Cost estimate:**
- 20 videos: about 1–2 GPU-hours (0.12–0.25 s per frame-pass), plus about 1 hour of setup.
- 40 videos: about 2–4 GPU-hours, plus setup.
- If check 0 is not bitwise, the 35-name reference rerun adds about 4.5 GPU-hours (20 videos) or 9 (40).
- State all of these in dollars at the offers' prices.

**Not in this rental:** the P1 smoke step and the `forward_text` wrapper's bitwise check. They come first on P1's first rental.

**Then:**
- write the scripts in `studies/d7/check/`:
  - upload (HELD-A videos' frames only; no test file, no HELD-B file);
  - setup (G1's pinned stack);
  - run (check 0, the reference if needed, both arms);
  - copy-back with hashes;
  - the laptop analysis;
- test them locally with the **fake SAM 3 worker**;
- commit.

**Then stop and ask me before renting,** unless Step 3 or the guard stopped. Bring:
- the final video list (20 or 40) and the guard's number;
- the time and cost estimate, both check-0 cases;
- 3 matching offers (price, RAM, location, reliability).

## Final deliverable
`studies/d7/session_summary.md`, one paragraph per step:
- **the split:** HELD-B's two roles, the disclosure, and the FIT counts;
- **A30's scope amendment:** PaD gate 0 is recorded as "A30, scoped"; nothing was run;
- **the naming check:** its verdict and numbers;
- **the D7 check's readiness (PaD gate 1):** its guard number and cost estimate;
- **Task 1:** "not run by decision", with its ledger entry closed.

Also give the new chain head and the commits. **Then stop.**
