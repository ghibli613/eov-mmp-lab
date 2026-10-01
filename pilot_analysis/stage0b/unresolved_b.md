# Stage 0b — new judgement calls

Stage-0 calls in `../stage0/unresolved.md` still apply unless superseded here.

| predicate(s) | as drafted / best guess | reason for doubt |
|---|---|---|
| stop_* | state_change=halt; subject_state and locomotion unset | Stage 0 drafted stop as subject_state=stationary + locomotion=none. With a halt value available, the interval spans both states, so both axes are now unset. Unverified: whether VidVRD stop_* instances actually contain the moving->stationary transition inside their temporal extent, or simply label an already-stopped subject. If the latter, halt is MEASURED-false on most instances and stop should revert to stationary. |
| stand_*, sit_*, lie_*, and all moving verbs | state_change left unset | A sustained posture or gait arguably asserts state_change=none over the relation's extent. Left unset under the stage-0 convention (silent -> unset). Consequence: `none` and `start` are assigned to no predicate. |
| ride | posture=sit | Riding a bicycle, horse or motorbike is usually seated, but standing on a skateboard is also annotated as ride in similar datasets. Left unset. |
| *_inside | posture as drafted (sit_inside -> sit, etc.) | Posture is lexical here, so no doubt about the value; noted only because these three were an all-novel collision group in stage 0. |
| chase, follow, fall_off | still identical: {subject_state=moving} | Neither new axis touches them. All three are base, so this collision does not affect novel scoring, but the stage-0 proposals (relative_motion, depth, object_state) would separate them. |
