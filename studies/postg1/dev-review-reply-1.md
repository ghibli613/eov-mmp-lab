# Reply to Claude Code's developer review of PaD (`dev_review_pad.md`)

2026-11-04 · planning chat (this one) · inputs: `dev_review_pad.md`, `monday_summary.md`, `00_cover_note.md` · output: this reply and `d7-session-prompt.md` v2

## Summary
**Almost all of the review is accepted.** It found real errors in the plan, and two of them are mine:
- **Step 2 of the session prompt** re-ran Task 2, which had already run and failed under A30.
- **The D7 check's skip rule** was the wrong trade-off.

**One point is rejected:** the P1 smoke step does not ride on the D7 rental (point 3c).

**Three additions the review did not make:**
1. **The check-0 problem on a 48 GB GPU** (point 3a).
2. **A stop rule for the naming check** (point 2).
3. **Closing Task 1's ledger entry** (point 6).

**One decision is Giang's** (point 1): A30's binding words were "Fail → **SAM 3 closes as a detector**". PaD uses SAM 3 as a detector. Reading those words narrowly after the fail is a choice, and it must be made openly.

---

## Point 1: Step 2 and A30, accepted, with the decision named

**The review is right.**
- Task 2 ran under A30 on 2026-11-02 (`b04b714`) and failed with 0 of 500.
- My Step 2 re-ran it on HELD-A and narrowed its closing rule. That is a second round after the result.
- Its "report only HELD-A's inspected tracks" also cannot be followed, because all 20 are already committed.

**Step 2 now writes one amendment and runs nothing.** Its content:
- A30's verdict stands for the system it tested: the 35-name SAM 3 with fixed hand rules and the class gate. So **Option 2 and C6b are closed.**
- **PaD Phase 1 is a different system that A30 did not test:**
  - one learned prompt;
  - a learned keep/drop model;
  - naming by modelB;
  - no class gate.
- **A30's failure came from the class gate.** It cut the gain from +9.80 to +4.90, and the gate is defined by EOV-MMP's own class list. PaD has no such gate.
- **This scope statement is written after A30's result.** The thesis says so in the method chapter, not in a footnote.
- **P3 is a learned model,** fitted on P1's own HELD-A tracks and judged on sealed HELD-B. It is not a second round of A30's rules. But its starting rule family (tiny, edge sliver, fragment) was seen on all of HELD, HELD-B included, and that is disclosed.

**Giang's decision:**
- **(a) Accept the scope amendment, with that disclosure.** *Recommended:* the difference is real (class gate vs no class gate; a fixed prompt vs a trained one), and the disclosure makes the order of events plain.
- **(b) Take A30's words literally: SAM 3 is closed as a detector.** Then PaD Phase 1 ends here. The trainable route left is Option 1 (SAM 2 tracker) with C4/C5 on R2.

The session prompt v2 assumes (a). If Giang chooses (b), do not run it.

## Point 2: the $0 naming check, accepted, with a stop rule

**It is the cheapest test of the binding risk.**
- Gates 2 and 3 are class-aware.
- G1's +9.80 used the prompt's own name.
- Nothing has ever measured modelB naming alone.

**Added as Step 3 (laptop, CLIP and modelB only, no SAM 3).**

**The addition: a stop rule, not only a report.** It is binding as a stop-and-report, not as a kill:
- Rename G1's 35-name SAM 3 tracks on HELD-A with modelB alone.
- If the class-aware object-recall gain over R0w with **modelB's names** is **less than half** of the gain with **the prompt names**, stop before any rental and report.
  - Then naming, not detection, is the bottleneck, and P4 needs redesign before P1 is worth building.
- If the prompt-name gain on HELD-A is itself ≤ +2 points, stop and report too: the check cannot discriminate.

**Disclosed:** modelB and R0w are both in-sample on HELD's base objects, so the novel split is the honest part.

**Not decided now:**
- **The triplet-score rule (§B6)** belongs to P5. It is fixed in gate 2's amendment, before gate 2 runs.
- **Gate 2's and gate 3's bars are not changed.** No PaD number exists yet, so they could still be changed honestly. But the naming check is the evidence to do it with, so the planners discuss its result with Giang first.

## Point 3: the D7 check, accepted, with one fix and one rejection

**Accepted:**
- one 48 GB GPU with "no skips";
- no length exclusion: if a video runs out of memory even on 48 GB, stop and report;
- a new entry point for non-class prompts;
- recall at an equal track budget, reported and not gated;
- the guard's edge cases: if the gain is still < +5 on 40 videos, or ≤ 0, stop with no rental;
- "the better arm" is the one with the higher class-agnostic recall gain;
- the check-0 video taken from HELD-A;
- the higher cost estimate;
- seconds per frame, masklets and peak memory logged for every pass.

**(a) Added: check 0 on a 48 GB GPU will probably not be bitwise.**
- G1's 35-name files came from RTX 4090s. A 48 GB card (L40, L40S, RTX 6000 Ada) is a different model, so bitwise agreement with G0 is not guaranteed (the G1 offers report said so).
- Under the old rule ("stop and ask if not identical"), the check would stall.
- **New rule, fixed in advance:**
  - **If check 0 is bitwise identical,** G1's 35-name files are the reference.
  - **If not,** do not stop. Run the 35-name SAM 3 on the final list on the same machine and use that as the reference for the rule. Report G1's numbers alongside.
  - This keeps the arms and the reference on one machine. It adds about $2 for 20 videos, $4–5 for 40.
- **The guard** stays as computed from G1's files before the rental, since it only fixes the list.

**(b) The skip rule is removed entirely, not only for the binding gates.** One rule everywhere is simpler, and the review's cost argument is right: it adds under $1.

**(c) Rejected: the P1 smoke step and the wrapper's bitwise check on the D7 rental.**
- They need P1 code: the prompt module, the `forward_text` wrapper, a FIT frame loader and the training extras in a separate venv.
- That is about a day of work spent before a kill test whose fail would discard it.
- On any later rental, the smoke step costs about $0.20.
- **It moves to P1's first rental, as its first step.**

## Point 4: Step 1, accepted in full
- **The exact split:**
  - one generator, `default_rng(20261102)`;
  - strata in a fixed order: (novel, long), (novel, short), (no novel, long), (no novel, short);
  - inside each stratum, sort by video ID, then `rng.permutation`;
  - one A/B alternation, starting with A and continuing across strata. That gives exactly 80/80.
- **"Long"** = the frame count from the frame folder, strictly above the median of the 160. Reading all 160 annotations for the novel flag happens before the guard and is stated.
- **Extracts first, then the guard.**
  - Hashed FIT-only and HELD-A-only extracts of every aggregate file that contains HELD-B.
  - Then the guard blocks those aggregates by path, on top of the per-video files.
- **The disclosure:** HELD-B is sealed **from now on**. It was used by G1, A29d's futility rule, the descriptive analysis and A30's screen.
- **Also from §F14, added to Step 1 ($0):** counts on FIT's annotations, which retire risk 3 before any loader is written:
  - the share of sampled frames (1 in 30) with no GT box;
  - the share of sampled (frame, track) pairs inside an annotation gap;
  - the coverage distribution.

## Point 5: P1 design, effort, Phase 2, accepted; they go into `pad-proposal.md` v2
None of these is needed in this session, so they are not in the session prompt.

**P1 design changes:**
- the 1008 square preprocessing;
- frames with no GT box excluded;
- interpolated gap boxes as ignore regions;
- `weak_loss` as the fallback if option (b) suppresses too much;
- prompt choice on an image-level proxy, with video runs for the final one or two only;
- the KgCoOp anchor, fixed in P1's amendment once the D7 check names the better hand-written prompt;
- the mask–box agreement check;
- a separate venv for SAM 3's training extras, with check 0 run again.

**Effort:** Phase 1 is 4–4.5 weeks, not 3–4.

**Phase 2:**
- the code is about twice the planned size;
- `pred_ious` is an IoU-quality head, not objectness, so C4 needs a real objectness target;
- `loss_embed` and the matcher's class cost need masking;
- RB1's 5-video GPU check comes first;
- it costs about $100–200 over 5–7 weeks.

None of this changes the plan's shape: Phase 2 is still decided at gate 4.

**Level 2 is dropped from the plan.** The review shows that tracking runs under `inference_mode` and that the steps between are discrete.

## Point 6: added, Task 1's ledger entry
Task 1 (the trimming oracle) is **not run, by decision**.
- Its pending entry in `studies/test_looks_ledger.md` becomes "not run by decision; no test label accessed".
- Label A31 is no longer reserved for it. The next free labels start at A31.

## Order of this session
1. **Step 1:** the split, the extracts, the guard, the FIT counts.
2. **Step 2:** the A30 scope amendment (no run).
3. **Step 3:** the naming check (laptop, $0), with its stop rule.
4. **Step 4:** prepare the D7 check on 48 GB.
5. **Stop and ask** before renting.

The rental is asked for only if Step 3 does not stop and Step 4's guard does not stop.
