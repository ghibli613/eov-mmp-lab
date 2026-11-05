# Session summary: HELD split, A30 scope, naming check (STOP), D7 not run

2026-11-04 to 2026-11-05 · `studies/postg1/d7-session-prompt.md` (v2), as amended by Giang's decisions during
the session · chain head **`6cc4a37145aaecf659e4eb66327c8cc1b6e8e95de560696591bb7a3a4b486d81`** (after A33b)
· nothing pushed · one rental (an L40 for 1.45 h, about $0.80), destroyed

## Step 1: the split (A31)

**The split.** HELD is split 80/80 into HELD-A and HELD-B (`studies/d7/held_split.json`, SHA256
`02415b25…f18a`).
- The median length is 171.5 frames.
- The strata:

  | stratum | videos | HELD-A / HELD-B |
  |---|---|---|
  | novel, long | 26 | 13 / 13 |
  | novel, short | 31 | 16 / 15 |
  | not novel, long | 54 | 27 / 27 |
  | not novel, short | 49 | 24 / 25 |

- Of the 3 videos without SAM 3 files, `…00280000` is in HELD-B; `…00225001` and `…00971000` are in
  HELD-A. So HELD-A has 78 videos with SAM 3 files.
- 18 HELD videos have `ILSVRC2015_val_` IDs, and the guard and every scan handle both ID forms.

**HELD-B's roles:** sealed; at most gate 3 (Phase 1 against R0w), and only if gate 3 passes, gates 4
and 5 (the Phase 2 and Phase 3 students against R2). Each gate gets its own amendment, and nothing is
tuned on HELD-B.

**Gate 3 is fixed:** tracker-level, class-aware, against R0w on HELD-B; recall ≥ +8, pair recall ≥ +8,
burden ≤ 1.35×; no skips, on a 48 GB GPU.

**The disclosure:** HELD-B is sealed **from now on, not unseen**. G1, A29d's futility rule, the
descriptive analysis and A30's screen all used it. P3's starting rule family was seen on all of HELD.

**Extracts and guard:**
- 36 extracts (FIT-only and HELD-A-only) of the data aggregates, hashed.
- `held_b_guard.py` then refuses HELD-B per-video paths and every aggregate. It also installs the
  test-label guard. 13 tests pass.
- Logs, run records and the pilot's aligned-array folders are sealed without extracts, since no later
  code reads them.

**FIT counts** (frames sampled 1 in 30):
- 70.4% of sampled frames have no GT box, and 52.7% inside track spans;
- 51.2% of in-span (frame, track) pairs fall in an annotation gap;
- 34.2% of tracks have coverage under 65%.

A descriptive extra check showed the gaps are in VidVRD's own per-frame annotation. **P1's training
has to handle empty frames and gaps.**

## Step 2: A30's scope (A32)

Giang chose option (a), and A32 is written after A30's result:
- **A30's verdict stands for the 35-name SAM 3** with fixed rules and the class gate, so Option 2 and
  C6b stay closed.
- **PaD Phase 1 is a system A30 did not test:** one learned prompt, learned keep/drop, modelB names,
  no class gate. A30's fail came from that gate (+9.80 → +4.90).
- **The thesis states the order of events** in the method chapter.
- **PaD gate 0 is recorded as "A30, scoped".** Nothing was run.

## Step 3: the naming check (A33, A33a, A33b): STOP

**Verdict: STOP.** On the 78 HELD-A videos, class-aware object-recall gain over R0w:
- **G1's 35-name SAM 3 tracks with the prompt names:** +5.73 points;
- **the same tracks named by modelB alone:** **−0.52**, below half of +5.73.

So naming is the bottleneck. **The cause is novel naming:**

| modelB accuracy | base | novel |
|---|---|---|
| on matched tracks | 97.8% | **26.3%** |
| top-1 on GT boxes | 98.1% | **35.0%** |

- The novel recall gain falls from +28.8 with prompt names to −14.4 with modelB's names.
- Base recall (+5.1) and base pair recall (+4.6) improve.
- **The confusions:** modelB mostly picks a close base class, e.g. watercraft → airplane, red panda →
  monkey, antelope → sheep, zebra or bird, bear → dog, sheep or tiger, giant panda → person,
  motorcycle → person or bicycle. It gets 0 of 41 matched tracks right for bear, giant panda, red
  panda and motorcycle. Cattle, fox and horse are named well.
- **The pooled CLIP features were not saved.** Recomputing them takes about 10 minutes of TagCLIP on
  an L40 plus about 30–40 minutes of rental setup (about $0.40, no token), or about 1.8 h on the
  laptop.
- **The run:**
  - a shape bug stopped the first laptop run before any result (A33a);
  - Giang stopped the laptop rerun after 5 videos, and moved the check to a rented L40 (A33b);
  - the L40 ran it in 40 minutes, and all 78 outputs copied back with matching hashes.
- Report: `studies/d7/naming_check_report.md`.

## Step 4: the D7 check (PaD gate 1): not run

Under A33's rule, a stop means the rental request is not prepared. **Giang decided: destroy the
machine, do not run D7.** So there is no guard number, no video list and no D7 cost.

**What exists:**
- **The code, prepared and tested but not pre-registered:**
  - `studies/d7/check/`: worker, driver, class-free adapter path, video selection, analysis;
  - 16 tests, including the driver end to end with a fake worker, run in a copy of the exact upload
    tree.
- **A34, drafted, not appended** (`studies/d7/amendments/A34.draft.md`). If D7 is ever run, A34 must
  first be updated and appended.

## Task 1: the trimming oracle

**Not run by decision.** The ledger's entry 17 is closed: "not run by decision; no test label accessed"
(`studies/test_looks_ledger.md`). No test label was opened in this session; the test-label guard was
installed in every script.

## Chain and commits

| amendment | chain head after it |
|---|---|
| A31 | `fec77dd3…e007` |
| A32 | `ceedec74…872b` |
| A33 | `e6b1e908…c1bd` |
| A33a | `99744ee7…f50f` |
| A33b | **`6cc4a371…6d81`** |

`verify_after_A26m.py` passed after each.

Commits: `d99cb0c`, `5de5d10`, `7ab1874`, `c68a2b5`, `088282c`, `4625943`, `afcab09`, `52d704a`, and
this session's results commit.

## For the planners

- **P4 needs a redesign before P1 is worth building.** modelB alone cannot name novel objects. The
  35-name SAM 3 tracks are found well class-agnostically (novel recall 85.6% against R0w's 64.9%); the
  naming loses the gain.
- **EOV-MMP's own fused naming is weak on novel objects too.** R0w's novel recall is 36.9%
  class-aware against 64.9% class-agnostic.
- **Gate 2's +10 / +10 class-aware bar looks out of reach with modelB naming.**
- **One gap seen in passing.** The test-label guard (`studies/rpt/no_test_labels.py`) does not list
  `pilot_analysis/stage1c/gt_tracks_test.json`, which holds test GT. Nothing read it in this session.
  It should be added to the guard before any session that could touch it.
- **Rental lessons, now in the scripts:**
  - the auto-stop's key test must be read-only; `vastai start` on a running instance restarted the
    container;
  - `vastai destroy` needs `-y` without a terminal;
  - check that the GPU name is exactly the planned model ("RTX A6000" is not "RTX 6000 Ada").
