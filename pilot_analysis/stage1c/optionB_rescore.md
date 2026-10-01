# Option-B grid — re-score of the frozen measured axes (HELD, no refit)

Frozen windows and thresholds come from `stage1b/thresholds_slot.json` and preregistration A1–A3. The scorer is first run on session 2's grid, where it must reproduce session 2's HELD numbers: **reproduced exactly**.

| metric | session 2 grid | option-B grid | change (points; + = worse) |
|---|---:|---:|---:|
| horizontal=left | 833/867 (96.1%) | 751/781 (96.2%) | -0.1 |
| horizontal=right | 831/866 (96.0%) | 750/781 (96.0%) | -0.1 |
| vertical=above | 68/68 (100.0%) | 60/60 (100.0%) | +0.0 |
| vertical=beneath | 83/83 (100.0%) | 74/74 (100.0%) | +0.0 |
| depth=behind | 529/578 (91.5%) | 468/512 (91.4%) | +0.1 |
| depth=front | 566/617 (91.7%) | 502/546 (91.9%) | -0.2 |
| subject_state=moving | 709/937 (75.7%) | 651/856 (76.1%) | -0.4 |
| subject_state=stationary | 965/1330 (72.6%) | 849/1171 (72.5%) | +0.1 |
| object_state=moving | 154/178 (86.5%) | 148/172 (86.0%) | +0.5 |
| proximity=adjacent *(LEARNED since A3, reference only)* | 175/282 (62.1%) | 169/257 (65.8%) | -3.7 |
| proximity=non-adjacent negatives *(LEARNED since A3, reference only)* | 1495/2144 (69.7%) | 1286/1915 (67.2%) | +2.6 |
| comparative=larger | 237/245 (96.7%) | 206/218 (94.5%) | +2.2 |
| comparative=taller | 216/232 (93.1%) | 188/203 (92.6%) | +0.5 |
| comparative=faster | 17/19 (89.5%) | 16/18 (88.9%) | +0.6 |
| D2 co_move on *_with | 123/178 (69.1%) | 111/172 (64.5%) | +4.6 |
| D2 toward on chase/follow | 25/77 (32.5%) | 26/71 (36.6%) | -4.2 |
| D2 subject-driven on stand/sit/lie (false positive) | 4/1144 (0.3%) | 7/988 (0.7%) | +0.4 |
| D1 co_move on *_with | 122/178 (68.5%) | 112/172 (65.1%) | +3.4 |
| D1 approach-or-co_move on chase/follow | 55/77 (71.4%) | 50/71 (70.4%) | +1.0 |

**Stop rule** (any measured axis worse by > 5 points): not triggered.

## Train counts

| grid | tracks | pairs | slots | labelled slots | (with ECC) | pairs > 40 slots |
|---|---:|---:|---:|---:|---:|---:|
| session 2 (whole spans, gaps skipped) | 2430 | 6326 | 23222 | 13659 | 13420 | 10 |
| **option B** | 4044 | 8926 | 13558 | 12365 | 12154 | 0 |
