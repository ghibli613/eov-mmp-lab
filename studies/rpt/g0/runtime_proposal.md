# Proposal: SAM 3 runtime options before G1

2026-10-26 · after the G0 report (`studies/rpt/g0_report.md`) · for the G1 amendment

## The problem

At SAM 3's defaults, G1 on the 160 HELD videos is projected at **43 GPU-hours** (about $15). That is **5.4×** G1's bar of 1 GPU-hour per 20 videos (8 GPU-hours). The bar needs 0.017 s per frame per prompt; G0 measured 0.094 s.

## Where the time goes (measured in G0, one RTX 4090)

- **Per frame per prompt: about 0.09 s,** even when the prompt finds nothing. Most of the 35 prompts find nothing in a given video, yet each still costs a full detector pass on every frame.
- **The tracker adds time only for prompts that find objects:** 0.12–0.18 s per frame for those.
- **The image backbone is about a quarter of the time.** Reusing it across prompts gave **1.36×**, bitwise identical.
- **The GPU was underused:** about 77% busy, with 6–13 GiB of 24 GiB in use.

**Why SAM 3 cannot work like EOV-MMP's detector.**
- EOV-MMP's detector proposes objects once per frame and compares them with all 35 class-text embeddings only at the end (late fusion). So one pass covers every class.
- SAM 3 fuses the text prompt into the image features early (`use_early_fusion=True`). Its encoder, its queries and its presence token all depend on the prompt. Only the backbone is prompt-free, so 35 names need 35 passes.
- That early fusion is plausibly why SAM 3 finds objects EOV-MMP misses. Speed and the thing G1 tests trade off against each other.

## Options

### A. Same output, faster: engineering, no method change

| | option | expected | how it would be verified |
|---|---|---|---|
| A1 | **Backbone reuse across prompts** (read-only wrapper, as measured in G0) | **1.36×, measured** | Already bitwise identical on smoke video 1; re-checked on the first G1 video. The CPU cache needs about 106 MB per frame, so for very long videos (the 2,947-frame one would need about 300 GB) run without it. |
| A2 | **2–3 SAM 3 processes per GPU**, on different videos | about 1.3–1.6× throughput (estimate) | Each process computes exactly the same thing. Check bitwise against G0's smoke outputs, and measure throughput. |
| A3 | **`torch.compile`**: SAM 3's own `compile=True` option (default off) | perhaps 1.5–2× (unmeasured) | Probably *not* bitwise identical, because fused kernels round differently. If it differs, report how much, and treat it as a setting change (option B) rather than an implementation detail. |

A1 and A2 together could plausibly bring G1 to **about 20–25 GPU-hours (about $7–9)** with provably unchanged output. That is still over the bar.

### B. Faster, but SAM 3's output changes: a method decision

| | option | expected | risk |
|---|---|---|---|
| B1 | **The pre-registered fallback**: detection every *k* frames, SAM 3's tracker in between | large for the detector share | Objects appearing between detection frames are found late. Running it needs a read-only wrapper on SAM 3's detector step; *k* fixed in advance. |
| B2 | **Skip absent prompts**: SAM 3's own presence score on a few sampled frames decides whether to run the full pass for that prompt | very large (most prompts are empty) | Misses objects that appear only in unsampled parts of the video, which is a recall loss in exactly what G1 measures. |
| B3 | **One generic prompt** (e.g. "object"), then name each track among the 35 classes with a classifier (e.g. CLIP), as EOV-MMP does | up to about 35× on detection | SAM 3 is trained for specific noun phrases, so vague prompts likely lose recall. Naming moves back to a classifier, and U2 (wrong object name, 7% of misses) is part of what RPT was meant to fix. |
| B4 | **Every 2nd frame** | 2× | Skipped frames are interpolated, which affects tracking quality. |
| B5 | **SAM 3.1** (Object Multiplex) | probably small for us | It speeds up tracking many objects; our cost is per-prompt detection. It is also a different model from the one the proposal names. |

Excluded by the ground rules: editing SAM 3's code, lowering its resolution, or dropping object classes.

## Recommendation

1. **Adopt A1 and A2** for G1. Both keep SAM 3's output identical, verified bitwise at the start of G1's machine.
2. **Measure A3** the same way, once. Adopt it only if bitwise identical; otherwise report the differences and leave it out.
3. **Replace G1's runtime bar with a cost cap**, about 25 GPU-hours with A1 and A2 (roughly $9 at $0.35/h), rather than use B1–B4. Those change what SAM 3 sees, and G1 exists to test whether SAM 3, as it is, finds objects EOV-MMP misses.
4. **If you want B3 kept as a future option, measure it cheaply first,** with no decision now: on G0's 3 smoke videos, how many GT objects and how many of the 35-prompt confident tracks does a single generic prompt also find? The match would be class-agnostic, so no classifier is needed for this measurement. The prompt wording must be fixed before the run.

## A small measurement session, if approved

About **0.5–0.75 GPU-hours** (about $0.25) on one 4090, on G0's 3 smoke videos, before G1:
- A2: throughput with 2 and 3 processes per GPU, plus a bitwise check against G0's outputs;
- A3: `torch.compile`, plus a bitwise check (or the size of the difference) and the speed-up;
- B3 (only if you want it): the generic-prompt coverage count above, with the prompt wording fixed by you in advance.

These would be pre-registered in the G1 amendment, or in a short addendum to it, before they run.

## Decisions needed from you

1. Accept A1 and A2 as implementation details, given bitwise verification?
2. Measure A3?
3. Cost cap instead of the runtime bar, and at what level?
4. Measure B3, and if so with what prompt wording?
5. Run the small measurement session before G1, or fold it into G1's first machine hour?
