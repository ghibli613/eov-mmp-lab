# Post-G1 Task 1, the trimming oracle: stopped before its amendment — one decision needed

2026-11-02 · nothing hashed for Task 1 · **no test label opened** · ground rule 4: "If something
cannot be done as written, stop that task and report"

## The decision needed

**Which predicted runs does the oracle change?** The rule reads "only predicted runs that D1's code
classifies as the C1 over-long case for a GT triplet". D1's code does not classify predicted runs,
and its "run" is not always a predicted run.

**What D1's code actually does** (`tc_run.py` `analyse`, then `tc_posthoc.py`, which gave the 129):
- It classifies **missed GT relations**.
- For each C1 GT, it rebuilds a run on one pair: the GT pair in PredCls, the best-vIoU detected pair
  in SGDet.
- The rebuilt run covers the slots where the GT predicate ranks ≤ 20, **joined across gaps of up to
  one slot** (`tc_run.runs`, `max_gap=1`). It then tests whether that run, clipped to the GT extent,
  reaches min vIoU ≥ 0.5.

**How predicted runs are formed** (the authors' `inference/post_process.py`, `association`):
- A predicate's run on a pair joins **consecutive** slots only. A one-slot gap ends it.
- The video's top-200 cut follows (`format_`).

**So D1's run is not always a predicted run:**
- **(a)** When D1's run spans a one-slot gap, it covers **two or more** predicted runs.
- **(b)** The predicted run may have been cut by the top-200 limit. Since membership is unchanged,
  it then stays out.

How often (a) and (b) occur on T1 cannot be known without opening test labels.

**Two faithful readings.**

| | A: anchored on D1's run (recommended) | B: D1's test applied to each predicted run |
|---|---|---|
| **Changed runs** | for each GT that D1's code, re-run on T1, puts in C1 over-long: the top-200 instances from **D1's pair**, with the GT predicate, lying inside D1's run and overlapping the GT | every top-200 instance, not a true positive, with the GT's triplet, that overlaps a GT D1's code puts in C1, and that, clipped to that GT's extent, reaches min vIoU ≥ 0.5 (D1's posthoc test) |
| **When several qualify** | each qualifying instance is changed, to the GT it overlaps most | each is changed, to the qualifying GT it overlaps most |
| **For** | the literal reading of "D1's code classifies"; ties the result to D1's 129 | closer to what a real trimmer acts on: actual predicted runs |
| **Against** | when D1's run spans a gap, one GT can trim several instances | in SGDet it may also trim runs from other track pairs (duplicates), which D1 never classified |

**My recommendation is A,** because Part 4 asks that "the result connects to D1's 129 GT". Either
way, the report would give the number of GT whose D1 run is (a) split into several instances or
(b) outside the top 200.

## Smaller points the amendment would fix (please confirm)

1. **T1's test predictions are not stored as files.**
   - `t1_run.py` computed them in memory and kept only metrics (`data/t1_results.json`).
   - The guarded run would **regenerate** them from session 3's cached slot scores (the 400 cache
     files, checked against `SHA256SUMS_session3.txt`). It would use `t1_run.py`'s rule
     (`with_rules`) and `eval_from_cache.predict`.
   - The reproduction gate is equality with `t1_results.json` at full precision: SGDet
     16.14878 / 27.57078, PredCls 22.24980 / 38.02963 (novel / all mAP). Otherwise, stop.
2. **D1's classification on T1 needs a copy of D1's code.**
   - `tc_run.py` (pinned by A20) and `tc_posthoc.py` hard-code the baseline logits and cannot be
     edited.
   - The copy, in `studies/postg1/`, changes only two things: the predictions come from T1's
     logits, and the per-slot ranks use T1's logits.
   - **Validation inside the same guarded run:** the copy, given the baseline logits, must
     reproduce D1's published counts before it is used on T1:
     - SGDet novel: 484 missed, with U1 262, U2 34, U3 2, A1 8, N1 8 and C1 170, of which 129 are
       over-long;
     - PredCls novel: 361 missed, 237 over-long.

     This re-reads test labels only to reproduce known numbers, within the one run.
3. **The "all" numbers:** D1's code does classify all GT (`tc_run`'s "all" pass; its SGDet buckets
   are in D1's report). The over-long split is applied to the all-pass C1 rows with all predicates'
   columns. So "all" trims all GT, as the rule asks when coverage exists.
4. **The slot (A31 records it):** 30 frames, starting at the pair's overlap start
   (`gen_labels.py:661-666`). A tail of ≥ 10 frames forms a shorter last slot; a tail of < 10 frames
   is dropped from the pair.
5. **The variants' tie-breaks:**
   - **(i):** the instance ∩ the GT extent.
   - **(ii):** among unions of consecutive slots inside the instance, the highest temporal IoU in
     frames with the GT extent. Ties go to the shorter, then to the earlier.
   - **(iii):** (ii) plus one slot at each end, clipped to the instance.
   - **(iv):** (ii) minus one slot at each end, if at least one slot remains; otherwise as (ii).
   - **"Overlaps most":** the most frames of temporal overlap; ties go to the earlier GT.
6. **Trimmed trajectories:** the instance's own subject and object boxes, sliced to the new extent.
   Its score and its place in the top 200 are unchanged.
7. **The label:** A30 went to the SAM 3 screen, the next free label. The trimming oracle's amendment
   becomes **A31**.

## Ready, waiting for the decision

- The ledger, `studies/test_looks_ledger.md`: 16 test-label accesses so far, plus test-video uses
  without labels and planned looks that never ran.
- **What happens after the decision:**
  1. I write A31 with the chosen reading and the points above, then hash, check and commit it.
  2. I write the script, testing it on synthetic data and HELD-format inputs only.
  3. I lift the guard for that one script, run it once, put the guard back, and write the report.
- **Cost:** CPU only, minutes; no GPU.
- **Task 4 waits on Task 1's decision,** because it runs only if Task 1 says go.
