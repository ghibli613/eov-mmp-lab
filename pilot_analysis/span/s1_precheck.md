# Span idea, Batch S1 — pre-check: training labels cannot supply span supervision

**Status: stopped before A10.** Nothing was pre-registered, scored or trained.
The only work done was counting training annotations: base labels only, no test
file opened. The 50 pinned artefacts still verify.

## The finding

S1 measures and learns relation extents on **training videos** (HELD for the
ceiling, FIT for the cut targets). The annotation convention there does not
match the test set's.

| | training annotations (base) | test annotations (pilot, phase 3.1 / 3.3) |
|---|---|---|
| instance length | **every one of 22,616 instances is exactly 30 frames** | 77% span more than one 30-frame slot; 2,834 instances are ≥ 2 segments long |
| consecutive same-triplet instances | 10,004 pairs: 4,302 overlap (mostly by 15 frames), 671 abut, 5,031 have a gap | long single instances |
| HELD, if same-triplet chunks that overlap or abut are merged | 4,365 chunks become 3,075 instances; **only 13.3% are longer than 30 frames** (median 30, 95th percentile 90) | — |

Training relations are annotated as 30-frame chunks on a 15-frame stride. Half
of consecutive chunks of the same relation have a gap between them, and we can't
tell whether the gap is unannotated or the relation really stopped. The long
instances that the 77% / 72%-lost / +12 mAP findings are about exist only in the
test annotations.

## What this does to S1 as specified

- **Step 1 ceiling on HELD.**
  - With chunk GT, the oracle "cut where the GT set changes" cuts at nearly
    every chunk boundary, so (b) ≈ (a).
  - Any longer prediction, such as hand rule (d) or pooled segments, fails
    vIoU 0.5 against a 30-frame GT chunk.
  - **G-S1 would very likely fail for an annotation-format reason, not because
    the idea fails.** That would be a false negative.
- **Step 2 cut targets.** "The pair's set of base relations changes" would mark
  chunk boundaries, which are annotation artefacts, plus gaps of unknown meaning.
  A head trained on them would learn the training annotation grid, not the
  test-style relation extents it is meant to recover.
- Step 0 (the instance-builder check) and the side count are unaffected.

## Options (the chat's decision)

1. **Merge training chunks, then run S1 as specified.** Pre-register a merge
   rule: same triplet, overlapping or abutting chunks become one instance. Use
   it for the HELD GT and for the cut targets. Cheap and honest, but HELD stays
   87% single-chunk, so the ceiling is small by construction. A G-S1 failure
   would still be hard to interpret.
2. **Merge across gaps up to k frames.** This makes training labels look more
   like test, but it invents relations in unannotated gaps, and k is a new free
   choice.
3. **A declared development split of the test set.** For example, a fixed
   random 50 of the 200 test videos for the ceiling, θ and the head's
   validation, with the final evaluation on the other 150. This is the only
   source of test-style extents. But it uses test labels during development, and
   the final numbers would cover 150 videos, not comparable with METOR's
   200-video results.
4. **Park the span idea.** Record the annotation mismatch as a finding: the
   pilot's timing-loss figures describe a test-set property that the training
   annotations cannot supervise.

**My recommendation is 4, or 3 only if a reduced-test protocol is acceptable for
the thesis.** Option 1 costs little but would very likely end in an
uninterpretable G-S1 failure. Option 2 adds an unvalidated free parameter at the
core of the supervision.
