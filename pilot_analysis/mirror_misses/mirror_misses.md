# Missed novel-split relations: was the mirror predicted on the reversed pair?

**Data:** the existing prediction dumps (`preds_from_pod/full`) and the **test** relation GT. This is an error analysis of the evaluated model, so it needs test data, unlike stage 0. Anything designed from it has been informed by the test set.

**Definitions.** Novel-split GT means test relations whose predicate is novel, which is the official filter. *Missed* uses the official matcher: greedy by score, exact triplet, min(subject, object) vIoU ≥ 0.5, first over all dumped predictions (≤200 per video) and then within each video's top 100. The *reversed pair* means predictions whose subject trajectory matches the GT object and whose object trajectory matches the GT subject, both at vIoU ≥ 0.5. That also guarantees an overlapping extent. Categories are not required to match.

**Controls,** per missed GT, both exact expectations rather than samples: *uniform* = the chance that a predicate drawn uniformly from the pass's other candidates was predicted on the reversed pair; *marginal* = the same, with the draw weighted by how often the model emits each predicate. 95% CIs resample whole videos.

## Novel pass — the official novel split

Novel-split GT instances whose predicate has a mirror in the vocabulary: **211**. Of these, 95 have a mirror this pass cannot emit (a BASE mirror, which the novel pass hard-zeroes) and are excluded below.

| GT set | n | exact mirror on reversed pair | mirror spatial term, any predicate | control: uniform random predicate | control: model-marginal random predicate | reversed pair has any prediction |
|---|---:|---:|---:|---:|---:|---:|
| missed (all dumped) | 81 | **3.7% (0.0%–11.1%)** | 7.4% (1.6%–16.1%) | 1.7% (0.4%–3.9%) | 2.3% (0.6%–5.4%) | 16.0% (7.5%–29.0%) |
| missed (R@100) | 81 | **3.7% (0.0%–10.4%)** | 7.4% (2.2%–15.9%) | 1.7% (0.4%–3.8%) | 2.3% (0.6%–5.7%) | 16.0% (7.3%–27.7%) |
| detected (reference) | 35 | **82.9% (71.8%–93.8%)** | 97.1% (90.0%–100.0%) | 27.0% (24.5%–29.4%) | 36.0% (32.0%–39.7%) | 97.1% (90.9%–100.0%) |

**Where the missed GTs went** (all-dumped level), with trajectories matched at vIoU ≥ 0.5 and any category:

| missed GT | rate |
|---|---:|
| forward pair (A, B) has any prediction | 17.3% (8.7%–31.7%) |
| forward pair has the GT predicate, but the triplet still missed (category error) | 4.9% (0.0%–13.3%) |
| reversed pair (B, A) has any prediction | 16.0% (7.5%–29.0%) |
| **neither direction has any prediction** | **81.5% (68.1%–90.4%)** |

Conditional on the reversed pair having a prediction (n = 13): exact mirror 3/13 (23%). The controls on the same rows are 11% uniform and 14% model-marginal.


Missed GT by predicate (all-dumped level): mirror predicted / missed:

`stand_above`→`stand_beneath` 0/16, `walk_above`→`walk_beneath` 0/12, `walk_beneath`→`walk_above` 0/11, `stop_beneath`→`stop_above` 0/8, `stand_beneath`→`stand_above` 0/6, `above`→`beneath` 0/5, `beneath`→`above` 0/4, `lie_above`→`lie_beneath` 1/4, `run_beneath`→`run_above` 1/3, `fly_behind`→`fly_front` 0/2, `fly_left`→`fly_right` 0/2, `stop_above`→`stop_beneath` 0/2, `lie_beneath`→`lie_above` 0/2, `fly_right`→`fly_left` 0/2, `run_above`→`run_beneath` 1/1, `fly_front`→`fly_behind` 0/1

## All pass — base predicates also scored (secondary)

Novel-split GT instances whose predicate has a mirror in the vocabulary: **211**. Of these, 0 have a mirror this pass cannot emit.

| GT set | n | exact mirror on reversed pair | mirror spatial term, any predicate | control: uniform random predicate | control: model-marginal random predicate | reversed pair has any prediction |
|---|---:|---:|---:|---:|---:|---:|
| missed (all dumped) | 170 | **3.5% (1.1%–7.5%)** | 11.2% (6.5%–17.7%) | 1.2% (0.7%–2.1%) | 2.1% (1.2%–3.7%) | 20.6% (14.7%–28.8%) |
| missed (R@100) | 174 | **4.6% (2.0%–7.5%)** | 13.2% (8.6%–20.0%) | 1.4% (0.8%–2.3%) | 2.4% (1.5%–3.9%) | 22.4% (15.9%–31.6%) |
| detected (reference) | 41 | **82.9% (72.5%–92.5%)** | 100.0% (100.0%–100.0%) | 10.8% (8.9%–12.7%) | 16.3% (13.7%–19.2%) | 100.0% (100.0%–100.0%) |

**Where the missed GTs went** (all-dumped level), with trajectories matched at vIoU ≥ 0.5 and any category:

| missed GT | rate |
|---|---:|
| forward pair (A, B) has any prediction | 21.2% (14.8%–29.4%) |
| forward pair has the GT predicate, but the triplet still missed (category error) | 1.8% (0.0%–4.5%) |
| reversed pair (B, A) has any prediction | 20.6% (14.7%–28.8%) |
| **neither direction has any prediction** | **78.2% (69.4%–85.2%)** |

Conditional on the reversed pair having a prediction (n = 35): exact mirror 6/35 (17%). The controls on the same rows are 6% uniform and 10% model-marginal.


Missed GT by predicate (all-dumped level): mirror predicted / missed:

`sit_behind`→`sit_front` 4/42, `sit_beneath`→`sit_above` 0/17, `stand_above`→`stand_beneath` 0/16, `swim_behind`→`swim_front` 0/13, `walk_above`→`walk_beneath` 0/12, `walk_beneath`→`walk_above` 0/11, `stop_beneath`→`stop_above` 0/8, `jump_behind`→`jump_front` 0/8, `stand_beneath`→`stand_above` 0/6, `above`→`beneath` 0/5, `beneath`→`above` 1/5, `run_beneath`→`run_above` 0/4, `creep_above`→`creep_beneath` 0/4, `lie_above`→`lie_beneath` 1/4, `fly_behind`→`fly_front` 0/2, `fly_left`→`fly_right` 0/2, `stop_above`→`stop_beneath` 0/2, `lie_beneath`→`lie_above` 0/2, `jump_above`→`jump_beneath` 0/2, `fly_right`→`fly_left` 0/2, `run_above`→`run_beneath` 0/1, `move_above`→`move_beneath` 0/1, `fly_front`→`fly_behind` 0/1

## Conclusion

**The mirror explains almost none of the misses.** On missed novel-split GT, the model put the exact mirror predicate on the reversed pair 3.7% of the time in the novel pass (n = 81), and 3.5% in the all pass (n = 170). The random-pairing control is 1.7%–2.3% and 1.2%–2.1% respectively. That is above chance in the all pass, but on the order of 2–3 percentage points of misses. In the novel pass the CIs overlap.

**The reason is that the pair is usually absent altogether.** 81% (novel pass) and 78% (all pass) of missed GTs have NO dumped prediction on either direction of the pair, meaning no prediction whose trajectories match the GT boxes at vIoU ≥ 0.5, under any predicate or category. Where the reversed pair is covered, the mirror rate rises to 3/13 and 6/35, against 14% / 10% model-marginal controls. That is elevated, but it rests on small counts.

**When the model detects a relation, it is direction-consistent.** 83% of detected novel GT also carry the exact mirror on the reversed pair, against a 36% control. So the model is not confusing directions. It is failing to produce the pair in the first place.

**Build implication.** A mirror or argument-swap mechanism could recover at most the few percent of misses where the reversed pair is covered and the forward one isn't. The ceiling on these misses is pair coverage, which matches the pilot's earlier finding that the failure is upstream. One caveat on 'absent': the dump keeps ≤200 predictions per video and the top 20 predicates per segment. So an absent pair is either a tracking or proposal failure, or a pair cut by truncation. The pre-merge `segments_raw_*.json` dumps can separate those two cases; that has not been done here.

