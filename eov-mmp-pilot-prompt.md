# Prompt for Claude Code — EOV-MMP Diagnostic Pilot Study

Copy everything below the line into Claude Code, after filling in the three placeholders marked `<<...>>`.

---

## Context

You are assisting with a diagnostic pilot study for a Master's thesis on Open-Vocabulary Video Visual Relation Detection (Ov-VidVRD). The machine you are on has:

- The **EOV-MMP** codebase at `<<PATH_TO_EOV_MMP_REPO>>` — this is unpublished code lent privately by the paper's authors ("End-to-end Open-vocabulary Video Visual Relationship Detection using Multi-modal Prompting", arXiv:2409.12499). It has two main modules: a relationship-aware open-vocabulary trajectory detector and an open-vocabulary relationship classifier built on frozen CLIP with multi-modal prompting.
- Pretrained checkpoints at `<<PATH_TO_CHECKPOINTS>>`.
- The **ImageNet-VidVRD dataset** (1,000 videos; 35 object categories; 132 predicate categories; JSON annotations with per-frame trajectories and temporally-bounded relation instances) at `<<PATH_TO_VIDVRD>>`.
- A CUDA GPU.

The study tests two hypotheses about why current Ov-VidVRD methods fail on novel predicates:

- **H1 (semantic):** classification error on the novel split is concentrated in *dynamic/kinematic* predicates (chase, push, pull, run toward...) rather than *static/spatial* ones (above, next to, hold...), because the CLIP text embedding is the only semantic prior and CLIP is known to be weak on verbs.
- **H2 (temporal):** the sliding-segment + greedy-merge pipeline loses substantial performance relative to what the segment-level predictions could support, because temporal extents are produced by a heuristic rather than decoded by the model.

Your job is to produce the **measurements**, not to fix anything.

## Ground rules — read carefully

1. **Never modify any file inside the EOV-MMP repo.** All new code goes in a separate directory `./pilot_analysis/` (scripts, outputs, logs). If you need to hook into their pipeline, import from it or copy the minimal function into your own file with a comment noting its origin.
2. **All mAP / Recall numbers must come from the repo's own evaluation code** (or the standard VidVRD evaluation script it calls). Do not reimplement the metric. For subgroup metrics (Phase 2), filter the *prediction and ground-truth predicate sets* fed to the official evaluator; do not write your own AP computation.
3. **Scale up gradually.** Every inference stage runs first on ~5 videos to validate shapes, paths, and output formats. Only after a clean small run do you launch the full test set. Print an estimated total runtime after the small run and ask before launching anything projected to exceed 2 hours.
4. **Hard stop on reproduction failure.** If Phase 1 misses the published numbers beyond tolerance, stop, write up what you observed, and report. Do not continue to Phases 2–4 and do not "fix" the discrepancy by changing evaluation settings.
5. **Log everything.** Save exact commands, config values used, git status/diff of the repo (to prove it is unmodified), environment (`pip freeze`, CUDA/PyTorch versions, GPU model) into `pilot_analysis/logs/`.
6. Explore the repo first (README, configs, train/inference entry points, dataset loaders) and adapt to its actual structure — the module and flag names assumed below may differ. In particular, find where the temporal **segment length** is defined (expected to be around 30 frames) and record its actual value; it parameterizes Phase 3.

## Phase 0 — Sanity gate (~minutes)

Load the checkpoint(s), run inference on 3–5 test videos, and verify: no shape/vocabulary mismatches; the prediction format contains, per relation instance, `(subject_category, predicate, object_category, subject_trajectory, object_trajectory, temporal_extent [t_start, t_end], confidence)`. Confirm which vocabulary split configuration (base/novel predicate lists) the config selects and that it matches the paper's novel-split protocol. Report findings before proceeding.

## Phase 1 — Reproduction gate

Run full inference + official evaluation on the VidVRD test set for both settings the paper reports:

| Setting | Published (paper Table) |
|---|---|
| Novel split | mAP 15.04, R@100 18.18 |
| All split | mAP 26.34 |

Tolerance: within ±0.5 mAP absolute. Slightly different numbers with a documented cause (e.g., a config the author noted) are acceptable if the cause is written down. **Save the raw prediction files** — they are reused by every later phase:

- `pilot_analysis/preds/final_merged.json` — post-merge relation instances with temporal extents.
- `pilot_analysis/preds/segments_raw.json` — pre-merge per-segment predictions `(video_id, segment_index, frame_range, triplet, confidence)`. If the pipeline does not naturally expose pre-merge predictions, add a dump hook in *your own* wrapper around their inference loop.

## Phase 2 — Static vs. dynamic predicate breakdown (tests H1)

1. Generate `pilot_analysis/predicate_partition.json`: every one of the 132 VidVRD predicates assigned to one of `{static_spatial, dynamic_kinematic, ambiguous}`, each with a one-line rationale. Rule of thumb: classifiable from a single frame → static; requires motion over time → dynamic; contact verbs like `hold`/`ride` that are visually static once established → static; `walk toward`, `chase`, `push` → dynamic. **Print the partition table and ask the user to review/correct it before computing final numbers** (it becomes a thesis artifact).
2. Compute per-predicate AP on the novel split (official evaluator, one predicate group at a time), then aggregate: mAP over static-novel vs. mAP over dynamic-novel. Also report the same on the all split.
3. Failure characterization: for each *missed* dynamic-novel ground-truth instance, record the top-5 predicates the model did predict for that tracklet pair — a table of what dynamic predicates get confused with.

**Decision criterion (record the verdict explicitly):** H1 supported if dynamic-novel mAP ≤ 0.5 × static-novel mAP.

## Phase 3 — Temporal extent analysis (tests H2)

1. **Data-only statistics** (from GT annotations, no model): distribution of GT relation-instance durations (frames); fraction exceeding 1×, 2×, 4× the segment length found in Phase 0; per-predicate mean duration (join with the partition — expectation: dynamic predicates last longer). Output a histogram figure + CSV.
2. **Merge-loss measurement:** evaluate three prediction sets with the official evaluator: (a) the actual post-merge output (= Phase 1 numbers); (b) oracle-merged: group the raw segment predictions using GT temporal extents (assign each segment prediction to the GT instance it temporally overlaps, take the union extent, max/mean confidence — document the choice); (c) segment-level ceiling: evaluate with temporal-IoU threshold relaxed, to bound how much signal exists before merging. The gap (b) − (a) is the price of the greedy merge.
3. **Fragmentation count:** for each GT instance ≥ 2 segments long, how many disjoint predicted instances of the correct triplet overlap it (1 = correctly merged; ≥2 = fragmented).

**Decision criterion:** H2 supported if oracle-merge gain (b) − (a) ≥ 2 mAP on the all split, or fragmentation affects ≥ 25% of long GT instances.

## Phase 4 — Classification oracle (isolates classification from detection)

Bypass the trajectory detector: feed **ground-truth trajectories** from the annotations into the relationship-classification stage (the repo's cascaded ancestry means the classifier consumes tracklet pairs — find that interface). Re-run the Phase 2 breakdown under GT trajectories. This yields the cleanest H1 measurement (no detection noise) and, by comparison with Phase 2, a two-way error attribution: (detection + pairing) vs. (classification). If the interface makes this genuinely infeasible, document why and skip — Phases 1–3 stand alone.

## Deliverables

`pilot_analysis/report.md` containing: environment + reproduction table with pass/fail; the reviewed predicate partition; static/dynamic mAP tables (novel and all, detected and GT-trajectory conditions); confusion table for dynamic predicates; duration statistics + histogram; the three-way merge comparison and fragmentation rate; an explicit verdict line per hypothesis (H1 supported / not supported; H2 supported / not supported) against the criteria above; and a short "surprises and caveats" section for anything anomalous. Keep all JSON/CSV dumps under `pilot_analysis/` — they will be reused for the thesis figures.

Work through the phases strictly in order, reporting at each gate.
