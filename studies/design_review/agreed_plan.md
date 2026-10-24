# Agreed plan — relation-aware tracklet keeping on EOV-MMP

The merged result of design reviews 1–3, `vidor_check.md`, the plan author's acceptances,
and the author's final instructions (2026-10-11). This document is the reference for the
amendments that follow: A26 (the protocol audit), A27 (the keeping step), and RB1's own
amendments.

---

## 1. Goal, bars and the realistic target

- **The method:** relation-aware **tracklet** keeping on the EOV-MMP front end.
  - Keep confident tracks exactly as today.
  - Recover weak-box tracklets (D2's O1b) and filter-removed tracks (D2's O3).
- **The bars, SGDet novel / all, protocol A:**
  - **16.74 / 27.52 (METOR), the defensible bar;**
  - **17.95 / 28.48 (OpenVidVRD, arXiv), the prudent bar.**
  - The protocol must be named with every comparison.
- **The realistic target:** about **+1 novel mAP on R0**, the released model, from
  keeping. This is review 2's back-of-envelope.
  - Starting from T1 (16.15), that would at best edge past 16.74. It won't reach 17.95.
  - On R2, the clean step-4 base, the size is unknown.
- **The headroom limit (Q7, A25e):** perfect tracks would recover only 106–108 of the
  262 U1 relations with the released relation head. U1 relations that involve a novel
  object almost never come back (7 / 54).
  - The novel split is predicate-defined, so base-object relations count toward novel
    mAP (review 2 §1).

## 2. Bases

| base | what | claim it supports |
|---|---|---|
| **R0** | the released EOV-MMP end-to-end checkpoint (`--arch metor-compat --n_iter 0`) | the published protocol, compared with METOR's numbers, with every leak disclosed |
| **R2** | RB1's clean **step-4** fine-tune of the authors' stage 1–3 checkpoints | a clean step 4 only. **Not a clean model:** stages 1–3 have no training code, and their exposure to novel data is unknown (RB1 L8). |
| R0u, R0w, R0uw | R0 with uniform fusion, with the whole video, or with both (A26) | how much the test-time protocol choices are worth |

Keeping is **refit per base** on the outside set. The recipe is the same; the fit is
not.

## 3. Steps

| # | step | where | cost (est.) | gate / output |
|---|---|---|---|---|
| **0** | **A26, the protocol audit:** R0u; R0 over the whole video (R0w); both (R0uw); against R0 (re-run as the identity check). Readings fixed in advance. | rented 4090 | ≈ 15 GPU-h (12.5–17), machine 13–18 h, **≈ $5–11** | the size of the test-path protocol effect sets the framing (§5) |
| 1 | **A27, design and pre-register the keeping step** (CPU) | laptop | — | the frozen recipe, the relation gate, the scoring rule |
| 2 | **Outside set:** one **VidOR train part** (official host, MD5-verified), restricted to videos with **≥ 2 annotated objects**; recording passes with R0 and later R2; split by video into a fit half and a check half | rented 4090 | about 10–15 GPU-h per base, **≈ $4–9 each** | the keeping fit, the relation gate, the suppression counts |
| 3 | **RB1 in parallel:** the PredCls amendment, training frames, the **5-video cost gate**, then **R2 → R1 → R3** (R3 optional) | rented 4090 | ≈ 110–220 GPU-h per configuration; **$115–400** for all three | R2 = the clean base; R2 − R1 = the step-4 leak |
| 4 | **One pre-registered test session:** keeping on R0 (the published protocol, and the whole video) and on R2; baselines R0, R2 (and R0u / R0w / R0uw from A26); T1; METOR's and OpenVidVRD's published numbers | rented 4090, then the laptop | ≈ 20 GPU-h, **≈ $7–12** | the result |
| 5 | **Our own detector:** only if R2 ≪ R1 **and** a fully clean claim becomes necessary | — | 2–4 months, $100+ | — |

**Total GPU, steps 0–4:** about **$135–445**. RB1 is about 80% of it, and its 5-video
gate decides the real figure. **Calendar:** about 2.5–3.5 months. Prices assume the one
measured rate, $0.35/h, up to $0.60/h.

## 4. The keeping step (A27 will freeze these)

- **Tracking: two tiers, offline extension.**
  - Tier 1 (boxes > 0.35) goes through today's deep_sort and AFLink, **unchanged and
    verified bit-identical** on non-test videos.
  - Tier 2 (boxes in [0.1, 0.35], deduplicated against tier 1 at IoU > 0.8) goes to a
    **separate** tracker, giving candidate tracklets.
  - Candidates attach to tier-1 ends **offline**, within the AFLink window. Online
    extension would change tier-1 tracks (`track.py:164-177`).
- **The keeping step decides on:**
  - candidate tracklets;
  - proposed extensions;
  - tier-1 tracks the fixed filter removed (coverage, range).

  The tracks the filter kept stay as they are.
- **Features, class-blind:**
  - **the baseline set (A):** `pred_ious`, persistence, length, geometry, motion, the
    AFLink link probability;
  - **the relation set (B = A + …):** modelC's interactiveness, the max base-predicate
    score over partners, the number of plausible partners.
  - **Excluded:** class probabilities, the cut score, names, raw `pred_embed` and RoI
    vectors (review 2, Q4).
- **Labels:** a match to a **base**-class GT track at vIoU ≥ 0.5. Unmatched tracklets
  are **unlabeled** (positive-unlabeled learning), not negative. Novel-class GT is used
  only for **descriptive suppression counts**.
- **The relation gate:** B − A ≥ **+2 AP** on the outside check half, **and** the paired
  video-bootstrap CI's lower bound is > 0. Otherwise the step is reported as **learned
  tracklet keeping**. That fallback framing is written into A27 in advance.
- **Scoring, the E1 lesson:** a kept tracklet's track score is fixed **conservatively,
  in advance** (e.g. keep probability × its mean fused score), so weak tracklets can't
  outrank confident pairs in the per-video top 200. There's an explicit **NaN guard**
  (RB1 L10).
- **Refit per base:** the same recipe, fitted separately on R0's and R2's outside
  recordings.

## 5. Decision points

| when | evidence | decision |
|---|---|---|
| after A26 | **Δuw = R0uw − R0** (novel mAP, with CI) | **large** (\|Δuw\| ≥ 1.0, the CI excludes 0): the clean protocol becomes a primary protocol in step 4; if the clean protocol scores lower, the fallback framing moves up. **Moderate** (0.3 ≤ \|Δuw\| < 1.0, the CI excludes 0): report both protocols. **Small** (otherwise): the published-protocol comparison stands, with disclosure. Exact wording in A26. |
| after RB1's 5-video gate | the measured cost per epoch | run R2 (then R1, then R3) only within the budget the author approves |
| after R2 and R1 | R2 − R1 | large → the leak story is central (fallback framing); small → the method framing stands |
| on the outside check half | the relation gate | keep or drop the "relation-aware" framing |
| before step 4 | everything above | one amendment freezes every test configuration |

## 6. Risks

1. **The realized gain is small** (≈ +1 or less), and extra candidates flood the
   ranking, as E1 showed. Mitigations: conservative scoring and a tier-1-safe design.
2. **The relation features add nothing** (F2 Part 1 failed before). Mitigation: the
   gate and the pre-written fallback framing.
3. **RB1's cost** comes in at the top of its range. Mitigation: the 5-video gate; R3 is
   optional.
4. **Outside-set transfer.** VidOR's videos are longer, blurrier and in a different
   domain. Mitigations: the ≥ 2-object rule, and reporting suppression and AP per
   subset.
5. **Protocol effects are large** (A26), so "beat METOR" loses meaning. Mitigation: the
   fallback framing (§7).

## 7. The fallback framing

A **clean-protocol benchmark and measurement contribution**
(`ov-vidvrd-lab/docs/90_Research-ideas.md` Direction A, plus "a single-protocol
re-evaluation"). Its material:
- the GT-derived test frame range (D2, A26);
- the fusion weights favouring the novel classes (L7);
- novel boxes, labels and predicates in training (L1, L2, L6);
- in-sample HELD;
- two incompatible published protocols (±4.4 mAP);
- what the clean protocol does to the published bar (A26, R2 − R1).

It's triggered by step 0's or step 3's decision points.

## 8. VidOR

- **A VidOR evaluation is deferred.** It depends on obtaining VidOR-trained EOV-MMP
  models (none are released; `vidor_check.md`). That decision is pending with the
  thesis author.
- **VidOR val stays untouched,** reserved for a possible pre-registered VidOR session.
- The outside check set uses one **VidOR train** part only.
- **ImageNet-VID is dropped:** its official host is gone, and the VidOR train part
  replaces it.

## 9. Standing rules carried forward

- **A25d:** D2's test recordings and their derivatives are never used for design or
  tuning.
- **HELD on R0:** descriptive only, never for a decision (B2 and E1 evidence).
- **Test labels** are opened only in pre-registered evaluation steps, on the laptop.
- **On the machines:** no tokens during runs; token scans before and after; test
  annotations deleted after runs and the deletion logged.
- **Commits** are local, nothing pushed. `studies/` is copied, not pushed.
