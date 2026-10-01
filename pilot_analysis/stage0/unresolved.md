# Stage 0 — unresolved judgement calls

Nothing below is in `phi_draft.json`. Each row is a proposal for review: the best guess, and why it was not drafted.

**Drafting convention.** An axis the predicate is silent about is left *unset*. `none` is written only where the predicate asserts absence (`stand` → locomotion=none). Evidence group: a predicate needs pixels if it sets locomotion or contact_action to a non-`none` value; `locomotion=none` on stationary verbs is implied by subject_state=stationary and does not count.

| predicate(s) | best guess | reason for doubt |
|---|---|---|
| stand_* / sit_* / lie_* / stop_* | schema needs a posture axis (stand | sit | lie), pixel-readable | The schema has no axis separating stand, sit, lie and stop: all four map to subject_state=stationary, locomotion=none. Every stand_X / sit_X / lie_X / stop_X quartet therefore gets an IDENTICAL phi -- see the collision table below. Several of these collisions pair a novel predicate with a base one (e.g. sit_behind vs stand_behind), so a phi-only scorer cannot tell them apart. Posture is also a pixel property, so these predicates are marked box-only here while really needing both. |
| stop_* | subject_state=stationary (as drafted) | `stop` may denote the transition moving -> stationary, not a state. If so it needs a temporal value the schema lacks. |
| move_* | locomotion left unset (as drafted) | `move` is manner-unspecified, so phi(move_X) is a strict subset of phi(walk_X), phi(run_X), ... A scorer will rank move_X at least as high as each of them whenever they fire. |
| left / right / front / behind (and every *_left etc.) | camera-frame (image-plane) relations | Reference frame unconfirmed. If VidVRD annotates front/behind relative to the object's heading (walk_behind = following), `depth` is not a depth axis and is not readable from boxes as the schema assumes. Camera-frame front/behind is only indirectly box-readable (occlusion, box size, y-position). |
| stand_with, lie_with, stop_with | relative_motion=none, proximity=adjacent | `with` for a stationary pair means 'together'; co_move needs motion, and the schema has no 'accompany/together' value. relative_motion left unset. |
| *_with (moving verbs) | proximity=adjacent | co_move and object_state=moving are drafted as definitional; adjacency is likely but not entailed (a flock flying 'with' each other can be separated), so proximity is left unset. |
| inside, sit_inside, lie_inside, stand_inside | proximity=overlapping (as drafted) | Containment is stronger than overlap; the schema has no 'contained' value so `inside` shares proximity=overlapping with any overlapping pair. |
| toward, away, past (bare) | subject_state=moving | Bare forms do not say which entity moves; the relative motion could come from the object. subject_state left unset. |
| chase | object_state=moving, relative_motion=approach, depth=behind | subject_state=moving is definitional. relative_motion could equally be co_move (pursuit at a constant gap); depth=behind depends on the reference-frame question above. |
| follow | object_state=moving, relative_motion=co_move, depth=behind | As chase, with a constant gap -> co_move rather than approach. |
| faster | object_state=moving | 'faster than' a stationary object is degenerate but not impossible in the annotations; left unset. |
| ride | vertical=above, proximity=overlapping, relative_motion=co_move, subject_state=moving, object_state=moving | Only contact_action=ride is drafted. Rider-above-mount is typical but a parked bicycle can be 'ridden' in a static frame. |
| touch, hold, bite, feed, kick | proximity=overlapping (hold, bite, touch, kick); adjacent (feed) | Contact implies the boxes meet, but 2-D boxes of touching objects can overlap or merely abut. Left unset. |
| watch | no axis fits; nearest is none | Gaze direction is not in the schema. No ingredient assigned -> the predicate is unscorable under phi as drafted. Base predicate. |
| play | no axis fits | Too broad for any schema value. No ingredient assigned. Base predicate. |
| fight | contact_action=? (no 'fight' value); proximity=adjacent | The contact_action list has no value for fight. No ingredient assigned. NOVEL predicate -> uncoverable under the current schema. |
| pull | contact_action=hold?, relative_motion=co_move or approach | No 'pull' value; pulling usually involves holding, but that is an inference. No ingredient assigned. NOVEL predicate. |
| drive | contact_action=ride?, proximity=overlapping | No 'drive' value; ride is the nearest but conflates rider and driver. No ingredient assigned. NOVEL predicate. |
| fall_off | vertical=beneath (end state) or relative_motion=recede | Tokenises to [fall, off]; neither is a schema value. Only subject_state=moving drafted. `fall` is not a locomotion value either. Base predicate, 4 training instances. |
| next_to (tokenisation) | treated as ONE lexical component | Splitting on '_' would give [next, to]; kept whole in component_frequency.csv. `fall_off` IS split into [fall, off]. |

## Identical-phi collisions (10 groups, 37 predicates)

Predicates in one row are indistinguishable under the drafted phi. `*` = novel.

- `lie_above`*, `sit_above`, `stand_above`*, `stop_above`*
- `lie_behind`, `sit_behind`*, `stand_behind`, `stop_behind`
- `lie_beneath`*, `sit_beneath`*, `stand_beneath`*, `stop_beneath`*
- `lie_front`, `sit_front`, `stand_front`, `stop_front`
- `lie_left`, `sit_left`, `stand_left`, `stop_left`
- `lie_next_to`, `sit_next_to`*, `stand_next_to`, `stop_next_to`*
- `lie_right`, `sit_right`, `stand_right`, `stop_right`
- `chase`, `fall_off`, `follow`
- `lie_inside`*, `sit_inside`*, `stand_inside`*
- `lie_with`*, `stand_with`*, `stop_with`*
