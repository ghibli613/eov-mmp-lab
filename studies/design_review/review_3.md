# Design review 3 — what the full record changes (addendum to reviews 1 and 2)

For the plan's author.
- **Why this exists.** The author asked me to firm up my review by reading the rest of
  the record. Reviews 1 and 2 relied on code reading plus a few reports. For the
  closed batches, they took your summary on trust.
- **What I've now read:** M1's results, reports and measured GPU estimate; RB1's leak
  list and fallback report; and the pilot's reports for B, B2, S1, F1, F2, the
  position verifier, E1, T1, the mirror-miss analysis, the trainability brief, the
  status file and the final report. Also the EOV docs `30_Landscape` and
  `90_Research-ideas`.
- **Rules:** reading only. No GPU, no new computation, no test-label use.

**Net effect.** My position on the merged plan (review 2) stands, with **four upgrades**:
- **E1-safe scoring** for kept tracklets, pre-registered (§1.1);
- a **stronger prior against the relation gate** (§1.2);
- **"clean step 4", not "clean model"**, for R2 (§1.3);
- an **honest target statement**: 16.74 is the defensible bar and 17.95 the prudent
  one. The expected keeping gain doesn't reach 17.95 (§1.4).

RB1's cost goes up, and I'd order its runs R2 → R1 → R3 (§1.5).

---

## 1. What changes

### 1.1 E1's failure is the main risk for tracklet keeping. Design against it now.

**What happened in E1** (`ov-vidvrd-lab/pilot_analysis/boundary/boundary_report.md`):
- Nested runs **recovered what they targeted.** 15 of 129 over-long SGDet cases were
  now matched, against 0 for T1.
- Yet **mAP fell in every setting** (SGDet novel 13.84, against T1's 16.15). The
  added instances were mostly false positives that **ranked above earlier hits**:
  42–49% of the final top-200 lists, at 0.35% precision against T1's 0.70% (SGDet
  novel).
- "GT matched rises while mAP falls."

Tracklet keeping adds candidates in the same way, so the same mechanism applies:
- **The ranking rule:** instance score = subject-track score × object-track score ×
  mean predicate score (`inference/post_process.py:97-98`), then each video's
  top 200 (`:108-110`).
- **What the plan has that E1 didn't.** E1's extras carried their pair's full score.
  A kept tracklet's track score can be set deliberately. **A27 must pre-register
  it**, e.g. keep probability × the tracklet's mean fused score, so weak tracklets
  can't outrank confident pairs. Leaving it to the released naming rule
  (`gen_labels.py:619`) is a gamble.
- **NaN guard.** A track with no scored row gets a NaN score at inference, and NaN
  keys make the top-200 sort undefined (RB1 `leak_list.md` L10; `fallback_check_report.md`).
  Tier-2 tracklets from weak boxes are exactly where this can happen. A27 needs an
  explicit rule.
- **The ranking effect can't be seen before test.** The outside videos have no
  relations, and HELD is in-sample (§1.6). So A27 should choose the conservative
  scoring scheme in advance, not a tuned one.

### 1.2 "Relations help objects" has failed once already. The 2(e) gate is not a formality.

- **F2 Part 1, relation-aware object classes,** failed its gate: head top-1 68.1% on
  HELD against ≥ 70% (`f2/f2_report.md`). It's a different target (naming, not
  keeping), but the same idea.
- **Related relation-side attempts** also stopped: F2's temporal smoothing failed,
  and the position verifier was "safe, but with almost no reach" (+0.02 mAP
  projected).
- **The prior** is therefore that relation features add little to object decisions
  in this stack. Keep the gate exactly as review 2 specified, and **write the
  fallback framing into A27 now** ("learned tracklet keeping"), so dropping the
  relation framing isn't a post-hoc choice.

### 1.3 R2 is a clean *step 4*, not a clean model

- **The leak list says it directly.** "L8, stage 1–3 checkpoints: no training code …
  whether they saw novel data is unknown; cannot be fixed; R2/R3 are clean for step 4
  only" (`studies/rb1/leak_list.md`).
- **So does the trainability brief:** "Only step 4 … is trainable here … We cannot say
  'we retrained EOV-MMP'" (`pilot_analysis/trainability-brief.md` §1).
- **The M1 gap stays unattributed.** M1's S arm (9.92 novel / 32.55 all PredCls)
  differs from EOV-MMP (21.70 / 37.20) in four ways at once:
  - the head: a new 9.1M-parameter slot classifier on frozen TagCLIP features;
  - the data: FIT's 640 videos only;
  - the cleaning: novel tracks and relations removed;
  - no prompt learning or step-4 fine-tuning.

  M1's reports never analyse the gap (`m1/handoff/step3_report.md`,
  `m1/step4_report.md`). One hint: the gap is −11.8 on novel but only −4.6 on all,
  which fits novel-specific leakage better than a generally weaker head. That's
  not proof.
- **What R2 − R1 measures:** only the step-4 part.
- **Wording for the merged plan's test session:** "keeping on R2" should read
  "keeping on a clean step-4 fine-tune of the authors' stage 1–3 checkpoints". A fully
  clean claim still needs our own stages 1–3 (step 5 of the plan).

### 1.4 The target, stated honestly

From `docs/30_Landscape.md` §2, with my additions:

| reference | novel / all SGDet | note |
|---|---|---|
| EOV-MMP, as METOR's Table 1 reports it | 15.04 / 26.34 | protocol A |
| METOR's own ablation, 0 iterations | 15.16 novel | "METOR is EOV plus the module" (§7) |
| **released EOV checkpoint, as we run it** | **15.79 / 27.18** | session 3 and the pilot. The pilot's reference for the same checkpoint was 15.64 / 26.88 (`PILOT-STATUS.md` B.24). |
| T1 (released + type rules) | 16.15 / 27.57 | the chain's best |
| **METOR** | **16.74 / 27.52** | "the defensible bar" |
| **OpenVidVRD** (arXiv 2503.09416, protocol A) | **17.95 / 28.48** | "the prudent bar"; unrefereed |
| Decomposed Prompt (ICASSP'25) | 19.90 / 27.87 | protocol B: its MMP baseline sits 4.4 above protocol A's; not comparable |

**Consequences:**
- **On the published protocol:**
  - our starting point (T1, 16.15) is 0.6 short of METOR;
  - review 2's back-of-envelope keeping gain (about +1 novel on R0) would at best
    edge past METOR;
  - **it would not reach 17.95.**
- **"Beat METOR" should name its protocol and its bar.** I'd state both bars in A27,
  and set expectations: beating 16.74 on protocol A is plausible but marginal;
  17.95 is out of reach for keeping alone.
- **METOR's gain lines up with object-trajectory quality** (mAP_o 36.31 → 38.91 novel
  in its table), which fits a front-end focus.

### 1.5 RB1's cost is higher than my estimate. Order the runs.

- **The measured price:** the pilot's full 200-video run cost **about $1.40 for about
  4 h**, roughly **$0.35/h** on a Vast.ai 4090 (`pilot_analysis/report.md`). It's the
  only measured price in the chain.
- **An independent estimate:** the trainability brief puts step-4 training at **1.5–2
  days per epoch** on one 4090. That includes a test evaluation after every epoch,
  which RB1 removes (`--no_epoch_eval`).
- **No cheap lever:** stride 30 saves only about 17% of runtime and destroys the
  released model's output (pilot `PILOT-STATUS.md` B.20; the brief §3).
- **My revised range:** about **110–220 GPU-h per configuration**. R1 + R2 + R3 is
  about **330–660 GPU-h, $115–400** at $0.35–0.60/h. That's most of a "hundreds of
  dollars" budget.
- **Order:** run **R2 first** (the plan needs it as a base), then **R1** (the control
  that makes R2 − R1 readable). **R3 last, and optional:** METOR's own ablation puts
  the module at +1.6 from its iteration-0 baseline, so it's worth measuring, but it's
  not on the keeping plan's critical path.
- **A23's 5-video measurement** stays the gate before any full run.
- **Not comparable:** M1's measured cost (about 3 h in all) came from training small
  heads on cached features (`m1/logs_box/step1_gpu_estimate.md`).

### 1.6 HELD is unusable for decisions on R0. Now with numbers.

- **B2:** EOV-MMP alone scores **50.28 base mAP on HELD** in PredCls
  (`stage2/batch_b2_report.md` §2), far above its test numbers.
- **E1:** its HELD plumbing showed **+2.38 mAP** while test went down (E1 §6).

Review 2's rule stands, now with evidence: on R0, HELD may only be used for
descriptive suppression counts, never to choose anything.

---

## 2. What the record confirms (no change)

- **The failure is upstream,** and has been known since the pilot.
  - Of the GT instances spanning ≥ 2 segments, 72.2% had no matching prediction.
    Oracle merging gained +0.07. Relaxing vIoU to 0.1 gained +12 mAP. "The headroom
    is in the detector, the tracker and predicate scoring" (`PILOT-STATUS.md` B.24).
  - The mirror-miss analysis agrees: 81.5% of missed novel GT have no prediction on
    either pair direction (`mirror_misses/mirror_misses.md`).
  - D2 then put the loss mostly before tracking.
- **Fragmentation is small.** F1 found only 3.95% of detected tracks were stitching
  candidates (`f1/f1_report.md`), consistent with D2's small O2-frag share. **Offline
  linking will contribute little.** The keeping gain must come from the weak-box
  tier (O1b) and the filter rescues (O3).
- **The chain's own scoring paths filter "novel" by predicate only** (review 2 §1);
  T1, E1 and D1 all use the same evaluator.

---

## 3. A ready fallback framing

`docs/90_Research-ideas.md` already recommends a measurement contribution: "the
benchmark measures neither video nor vocabulary". It also sets aside "a single-protocol
re-evaluation". The chain has since built most of that material:
- the test frame range comes from the GT (D2);
- the fusion weights favour exactly the novel classes (L7);
- novel boxes and labels reach detector training (L1–L2);
- novel predicates reach the interactiveness target (L6 / T6);
- HELD is in-sample (§1.6);
- two incompatible protocols in the published table (±4.4 mAP).

**If A26 or RB1 show large protocol effects**, a clean-protocol benchmark plus "what
these leaks are worth" is a ready alternative thesis contribution. That's better than
chasing a bar that the leaks themselves may inflate. I'd name this fallback in the
plan now, with the go/no-go being A26's and R2 − R1's results.

---

## 4. Items for A27, beyond review 2

1. The **track-score rule for kept tracklets**, chosen conservatively before any data
   (§1.1), plus a **NaN guard** (L10).
2. The **relation-feature gate** as specified, **with the fallback framing written
   in** (§1.2).
3. **"Clean step 4"** wording for R2 throughout (§1.3).
4. **Both bars** (16.74 defensible, 17.95 prudent, protocol A), with expectations set
   (§1.4).
5. **RB1: R2 → R1 → R3,** with the 5-video cost gate and the revised range (§1.5).
6. **No HELD mAP for any decision on R0** (§1.6).

## 5. What I still haven't read

- the pilot's stage 0–1 documents (the predicate schema and the recipes behind B and
  B2);
- M1's technical amendments A24b–A24g;
- EOV's `10_Known-issues.md` and `20_Benchmark-analysis.md` in full.

I don't expect them to change the plan. If you want them covered, say so.

**Stop.** Reading only. Nothing was built, run or trained.
