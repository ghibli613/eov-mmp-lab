# Stage 0c — Task 3: is `stop` a halt?

BASE training annotations only. Subject speed = mean camera-compensated per-frame displacement of the subject box centre, as a fraction of the frame diagonal. (i) = the annotated extent; (ii) = the window of equal length immediately before it (needs ≥3 tracked steps there).

Instances: stop_behind, stop_front, stop_left, stop_right (stop); control stand_behind, stand_front, stand_left, stand_right (stand, matched on spatial term). Reference for 'clearly moving': τ = 0.0019/frame, the 25th percentile of extent speed over base walk_* and run_* instances (so 75% of labelled walking/running exceeds it).

Excluded: no ECC for video: 126.

## Rates

| | instances | with a before-window | (ii) clearly exceeds (i) ¹ | transition inside extent ² | extent speed ≥ τ | median speed (i) | median speed (ii) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stop_* | 1533 | 404 | 59/404 (14.6%) | 127/1533 (8.3%) | 639/1533 (41.7%) | 0.0016 | 0.0011 |
| stand_* (spatial-matched) | 4149 | 1222 | 182/1222 (14.9%) | 443/4149 (10.7%) | 1572/4149 (37.9%) | 0.0015 | 0.0017 |
| walk_*/run_* (reference) | 4596 | 1756 | 124/1756 (7.1%) | 421/4596 (9.2%) | 3449/4596 (75.0%) | 0.0030 | 0.0030 |

¹ before-window speed ≥ τ AND ≥ 2× the extent speed.  ² first third of the extent ≥ τ AND last third ≤ half the first third.

Category-matched control: stand instances resampled to stop's subject-category mix; covers 0/1533 (0.0%) of stop instances (the rest have subject categories never seen standing). Stop's top subject categories: car 1126, watercraft 170, bicycle 86, bus 52, motorcycle 46, airplane 30.

## Threshold-free: can the speeds tell stop_* from stand_*?

AUC > 0.5 = the feature is higher for stop. 95% CI from a video-clustered bootstrap.

| feature | n | n stop | AUC | 95% CI |
|---|---:|---:|---:|---|
| extent speed (i) | 5682 | 1533 | 0.517 | 0.427–0.611 |
| before-window speed (ii) | 1626 | 404 | 0.469 | 0.271–0.649 |
| log(before / extent) | 1626 | 404 | 0.404 | 0.277–0.528 |
| first-third minus last-third speed | 5682 | 1533 | 0.462 | 0.404–0.518 |
## Subject categories — the control the brief assumed does not exist

Base training instances, subject category by verb:

| verb | instances | vehicle subjects (categories seen with stop) | top categories |
|---|---:|---:|---|
| stop | 1549 | 1549/1549 (100.0%) | car 1126, watercraft 170, bicycle 86, bus 52, motorcycle 46, airplane 46 |
| stand | 4676 | 0/4676 (0.0%) | person 671, elephant 626, antelope 544, bird 438, cattle 427, zebra 407 |
| sit | 1508 | 0/1508 (0.0%) | person 533, monkey 467, rabbit 126, lion 93, giant_panda 75, domestic_cat 71 |
| lie | 852 | 6/852 (0.7%) | domestic_cat 126, lion 95, dog 91, giant_panda 70, lizard 68, cattle 66 |

Vehicle categories: airplane, bicycle, bus, car, motorcycle, train, watercraft. Every stop_* subject is a vehicle; stand/sit/lie subjects almost never are. A category-matched stand_* control is therefore empty (coverage 0/1533 (0.0%) above), and the stop-vs-stand comparison is necessarily a vehicles-vs-animals comparison.

## Conclusion

**(C) stop_* is indistinguishable from stand_* on both windows.** The before-window clearly exceeds the extent in 14.6% of stop vs 14.9% of stand. A within-extent moving→stationary transition shows in 8.3% vs 10.7%. Median before-window speed is lower for stop (0.0011 vs 0.0017). Every threshold-free AUC has a CI that includes 0.5. Neither (A) nor (B) holds: no halt is visible inside the extent or before it. Only 404 of 1533 stop instances have a usable before-window at all, so (B) is also weakly powered — but the direction is wrong for it, not merely noisy.

What `stop` actually encodes is **stationary + vehicle subject**. It is the vehicle counterpart of stand/sit/lie, chosen by subject category, not by motion history. state_change is empty as an axis. Subject category separates stop from stand/sit/lie almost perfectly (1549/1549 (100.0%) vs 6/7036 vehicle subjects), but phi has no axis for it.

### Consequence of reverting stop_* to stationary

Written to `phi_v3_stop_reverted.json`; `phi_v3.json` keeps the Task 1 spec. After reversion, stop_X = {subject_state=stationary, locomotion=none, spatial}; stand_/sit_/lie_X add posture. Each novel stop_* predicate then becomes a strict subset of:

- `stop_above`* ⊂ `lie_above`*, `sit_above`, `stand_above`*
- `stop_beneath`* ⊂ `lie_beneath`*, `sit_beneath`*, `stand_beneath`*
- `stop_next_to`* ⊂ `lie_next_to`, `lie_with`*, `sit_next_to`*, `stand_next_to`, `stand_with`*
- `stop_with`* ⊂ `lie_next_to`, `lie_with`*, `sit_next_to`*, `stand_next_to`, `stand_with`*

Under the LEARNED-threshold rule they are reachable: their only LEARNED ingredient is locomotion=none, which has 7036 base instances. They are not reachable in the sense that matters. Nothing in phi or in the annotated extent distinguishes them from their posture-marked supersets. A scorer can only separate them through subject category, which is outside phi. New collisions from the reversion: 0. The v3 collision `stop_next_to` ≡ `stop_with` persists.

