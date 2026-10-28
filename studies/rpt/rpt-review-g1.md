# Review of the G1 amendment draft (A29), after G0 (planning chat, 2026-10-26)

Inputs: `g0_report.md`, the A29 draft (not hashed), `runtime_proposal.md`.

## Verdict
**Hash A29 after the changes in section 5.** None of them changes a bar's threshold except one recommended change (5.8), which needs the author's decision. G0 is a clean pass; the bit-exact reproduction of session 3 is the strongest possible G0 result.

## 1. OPEN items

| # | Decision | Notes |
|---|---|---|
| 3. EOV-MMP's tracks | **Accept, with a clarification** | Use EOV-MMP's *final* tracks, after its own filters: the tracks its relation stage would see. R0w (whole video) is the matched comparison, with R0 beside it. |
| 4. Matching | **Accept** | Also report class-agnostic counts beside every class-aware one, as planned. |
| 5. Pair recall | **Accept, with a change** | Use the **same matching code as D1's U1 bucket**, so G1's pair recall connects directly to the diagnosis behind RPT. State exactly how vIoU is computed over the relation's span (the track cropped to the GT relation's frames). |
| 6. Novel group | **Accept the grouping**, but see 5.8 on whether the novel bar should bind at all. |
| 7. Unmatched track | **Accept.** Report the class-agnostic unmatched count beside it: a well-placed but misnamed track counts as unmatched in the bar, which is right for the burden, but the split shows why. |
| 8. Box style and sensitivity | **Accept**, but fix the preamble (5.5): box sensitivity does read modelC scores on HELD. |
| 9. First 20, then all | **Accept.** Disclose that the first 20 include G0's 3 smoke videos (5.9). |
| 10. Raw detections for missed objects | **Accept, with a change.** Raw detections are dense (about 26 per frame), so "IoU ≥ 0.5 on at least one frame" will be met too easily. Report the distribution of the fraction of annotated frames with such a detection, not only "at least one". This measurement decides Verify's fate in G2 (section 4), so it is worth doing well. |
| 11. Budget and machine | **Accept, with two additions:** pin the machine stack (5.4) and define GPU-hours (5.2). |

## 2. Replacing the runtime bar: acceptable, but rewrite how it is written

**Acceptable**, for four reasons:
1. It was a feasibility bar, not a test of the hypothesis. In the proposal, failing it triggered a fallback (every-k frames), not a kill.
2. It is changed before any G1 data exists, in a new dated amendment, with the measured reason (G0 §9: 5.4× over).
3. The change keeps SAM 3's output exactly as the proposal intended. The fallback would have changed what SAM 3 sees, which is the thing G1 tests.
4. The only outcomes seen so far are G0's 3 smoke videos, which are disclosed.

**Not yet written correctly:**
- It sits in the table headed "Fixed from the proposal, verbatim in substance". It is not verbatim. Move it to its own section, "Changed from the proposal", which:
  - quotes the superseded text: the runtime bar in proposal §6, and the runtime fallback in §3.1;
  - gives the reason (G0 §9);
  - states that runtime is no longer a pass/fail bar of G1, only a cost stop.
- The cap needs the fixes in 5.2 and 5.3.
- **Binding consequence to state:** the test session runs SAM 3 at the same settings G1 validates (full fidelity, plus only the options that passed bitwise). Otherwise G1 would not validate the configuration that is tested.

## 3. The A28 erratum: correct in substance; add two things
1. Name the corrected sentences (A28 rules 2 and 5) and quote them.
2. State the consequence, which matters more than the correction itself: **the track score equals the starting detection's probability, and masklets only start from detections above 0.5. So the candidate tier [0.1, 0.5) is empty by construction at SAM 3's defaults, not just in G0's 3 videos.** Cite the comparison line for "above" (`>` or `≥`) with its file:line.

## 4. The plan

### Test-session cost
- Each full SAM 3 pass over the test set is likely 35–45 GPU-hours, and the session needs about two: the whole video, plus the GT-derived range for row i.
- With the loop runs and A26's EOV-MMP runs, the session is **about 80–100 GPU-hours, roughly $30–45**, instead of $14–24. That is still small against the budget.
- Update the proposal's cost table.
- Keep row i's separate GT-range pass. It is the like-for-like METOR comparison, and about $12–16 is worth it.

### Reuse G1's outputs
**Keep and hash G1's SAM 3 outputs on HELD, and reuse them in G2.** G2 then only pays for the loop's extra prompts, not another 35 GPU-hours. Write this into A29.

### Verify, given the empty tier
Verify as designed has nothing to verify: no masklet ever scores below 0.5. There are three options for the G2 amendment:
- **(a) drop Verify**, and let the loop be Extend and Re-find;
- **(b) redefine Verify on raw [0.1, 0.5) detections:** a raw detection of a class, not overlapping any existing masklet, is promoted by a box prompt to SAM 3's tracker when it pairs plausibly with a confident track. This is close to Re-find seeded by raw detections;
- **(c) lower SAM 3's detection threshold.** Rejected: it changes SAM 3's settings and invalidates G1.

G1's OPEN 10 measurement tells whether (b) can reach missed objects at all. Decide between (a) and (b) in the G2 amendment, from that measurement only.

**Effect on the claim:** the loop's novelty now rests mostly on Extend and Re-find. The G2 amendment should say so.

### Smoke videos inside HELD
G0's 3 smoke videos are 3 of the 160 HELD videos and of G1's first 20. The OPEN definitions were written after seeing them. Bars are judged on all 160, as pre-registered. Add one descriptive line on the other 157.

## 5. Changes before hashing
1. **Move the cost cap** out of the "verbatim" table into "Changed from the proposal" (section 2), with the superseded text quoted and the binding consequence for the test session.
2. **Define the cap precisely:**
   - GPU-hours = billed machine hours (wall clock × GPUs rented), not summed process time. This matters with A2's concurrent processes.
   - Add a hard stop-and-ask if *actual* SAM 3 hours reach 35 before the 160 videos finish.
   - The projection after 20 videos is **frame-weighted**: Σ over all 160 videos of frames × 35 prompts × the measured seconds per frame per prompt for that video's settings. Use the A1 rate only for videos of ≤ 1,000 frames, and the measured offload rate for long videos. Not a plain ×8 of the first 20.
3. **GPU memory in the A2 scheduling rule.** The rule covers CPU RAM only. Three processes on ~1,000-frame videos would need about 3 × (6.5 GiB + 6.3 MiB × frames), which is more than 24 GiB. Schedule so that the predicted peak sum stays under a fixed margin (for example 21 GiB), using G0's per-frame figures; long videos run alone.
4. **Pin the machine stack to G0's:** the same image digest, and the torch, CUDA and driver versions from G0's `pip freeze` and logs. Otherwise check 0 can fail on stack drift alone, and stop G1 for no scientific reason.
5. **Fix the preamble.** "No relation score on HELD is read" is contradicted by the box-sensitivity bar, which reads modelC's top-5 hits on HELD. Say: no relation score is read except the secondary box-sensitivity hit rates, which compare box sources on the same in-sample model.
6. **OPEN 3:** EOV-MMP's final, filtered tracks; R0w primary, R0 beside.
7. **OPEN 5:** D1's U1 matching code, with vIoU over the relation's span stated exactly.
8. **Recommended, the author's decision: make the novel-object recall bar descriptive.**
   - On HELD, EOV-MMP's detector is not only in-sample. Its step-4 training targets included the novel-class boxes of these very videos (leak T1/T2). Its novel-object recall on HELD is therefore inflated twice, and a 10-point bar against it can kill RPT for a reason unrelated to the test set.
   - The overall recall bar (which includes the 124 novel objects) and the pair-recall bar stay binding.
   - This changes a pre-registered bar, but before any G1 data, for a reason documented since RB1's audit.
   - If the author prefers to keep it binding, keep it, and write down that the risk is accepted.
9. **Disclose the smoke videos** inside HELD and the first 20; add the 157-video descriptive line.
10. **Erratum:** quote A28's sentences, cite the "above 0.5" comparison line, and state that the candidate tier is empty by construction.
11. **Keep and hash G1's SAM 3 outputs** for G2.
12. **OPEN 10:** report the distribution of the fraction of frames, not only "at least one".
13. **OPEN 7:** report class-agnostic unmatched beside the bar.
