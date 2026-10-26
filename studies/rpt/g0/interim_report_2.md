# RPT G0: interim report 2 (A28 hashed; waiting for SAM 3 access)

2026-10-26 · repo `ov-vidvrd-lab` · commits `5dea6bd` (A28) and the one holding this report · nothing pushed

Your answers to interim report 1 are all applied. **No decision is needed from you now.** G0 is
waiting only for the author's access to `facebook/sam3`.

## 1. A28: appended, hashed, checked, committed

- **Chain head:** `ac77e243…a591` → **`b94e198640df54184aecdb25c3c7c5962fb34650e6273b08c534c926cf5dcc9c`**.
  - The A28 text alone hashes to `363eecc3ddba8d9e6e3eaf284e794ee4706302d2f50a4dddb693af49631402b3` (23,124 bytes).
  - The old bytes are unchanged, the appended bytes equal the text, and both files are back at 444.
  - Log and backups are in `studies/rpt/logs/`. They sit one level under `studies/`, so the checker's snapshot and hash-list checks cover them.
- **Checker:** `verify_after_A26m.py` passes on the new chain with **69 checks**: the earlier 66, plus the after-A26m line, the snapshot and the hash-list backup (`studies/rpt/logs/verify_after_A28.txt`).
- **Commit:** `5dea6bd`, after a credential scan of the staged diff. Gitleaks is not installed, so the scan was a grep of the 6,749 added lines for 11 token and key patterns; all came back 0.

## 2. How your answers were applied

- **Checkpoint hash:** **Hugging Face does not expose the SHA256 without access.** The file's HEAD request, its raw LFS pointer and `paths-info` all answer 401, with no `X-Linked-ETag`, and the tree API masks the LFS oid.
  - A28 pins the revision `3c879f39…`, the size of 3,450,062,241 bytes, and the **git blob id of `sam3.pt`'s LFS pointer**, `5b7c2eaba7c1e2a81f26bc7476bfe80e43e979e4`.
  - That pointer is a standard 135-byte, three-line file that contains `oid sha256:<hash>`, so the blob id already commits to the SHA256.
  - A28a will pin the SHA256 after the download. It will also check that the pointer rebuilt from that hash and the size hashes to `5b7c2eab…`, all before any SAM 3 code touches data.
- **D1 (raw detections):** feasible read-only and implemented. A wrapper on the detector's `forward_grounding` clones the logits and boxes as they return. That is necessary: NMS then lowers the logit of every detection ≤ 0.5 by 1e4, in place (`sam3_image.py:846-861`). A wrapper on `_build_multigpu_buffer_next_chunk` supplies the frame index without a GPU sync. The runner keeps `[frame, x1, y1, x2, y2, p]` for p in [0.1, 0.5), per prompt; these are pre-NMS, since SAM 3 never runs NMS on them. The smoke test counts them by class. Added to A28's OPEN list, for the G2 amendment.
- **D2–D4:** fixed in A28 as agreed (literal `out_probs`; keep the higher-scoring duplicate; no pre-roll).
- **D5:** recorded in A28: wait up to 2 days for access, then one machine for steps 8–9; otherwise you are asked about a step-8-only machine.
- **Runtime measurement:** feasible read-only and implemented.
  - `forward_image` (`vl_combiner.py:81`) depends on the image alone and is recomputed for every frame of every prompt.
  - A wrapper caches its output per frame and gives later prompts a GPU copy with the same tensor sharing.
  - **Sizing:** on random weights the output is **212.6 MiB per frame in bf16**, about half of it position encodings. Those are kept once wherever they are bitwise equal. Smoke video 1 (130 frames) therefore needs about 14 GB of CPU RAM; it would not fit on the GPU next to the model.
  - **What runs:** a reuse run of all 35 prompts on smoke video 1, and a determinism control (the default run again on its first 3 prompts), so that a difference can be told apart from GPU non-determinism.
  - **Comparison:** `studies/rpt/g0/sam3_compare.py` compares masks, boxes, scores, statuses and raw detections bitwise, and reports the speedup.
  - **Limits:** it runs last, only if fewer than 95 minutes have passed since setup began, and is never used for G0's outputs.

## 3. Results so far (parts that do not use SAM 3)

**Unit tests: 35 passed** (`studies/rpt/g0/logs/pytest_rpt.txt`): 28 for the adapter and runner, 7 for the checker.

| case from the G0 prompt | test |
|---|---|
| internal gap filled per check 1 | equal to the released `interpolate_and_adjust_frames` where its own drop and trim rules do not fire; rounding checked |
| leading and trailing absence | no pre-roll, nothing after; nothing dropped where the released code would drop the track (50% coverage) |
| ID change makes two tracks | yes; same-prompt masklets are never deduplicated |
| names | underscore mapping, round trip of all 35, runner and adapter agree; unknown names raise |
| duplicate merge just above and below 0.7 | IoU 0.709 merges (the higher score keeps its class, extra frames counted); 0.695 does not |
| score over detected frames only | per-frame statuses checked; a masklet with no detected frame gets no score and is not paired |
| tiers | 0.5 confident, 0.4999 candidate, 0.1 candidate, 0.0999 dropped |
| 95-clip pair | session 3's `run_pair` gives windows 40/40/15; the joined output yields one relation over clips 38–42; window by window would give two |
| filled middle frame flagged | real `gen_feats_test` on a synthetic video; the middle frame 15 inside a gap is flagged; each frame is encoded once |

Further tests cover:
- the read-only recorder, its pass-through, and the raw-detection capture before NMS;
- the backbone cache: copies, sharing kept, position encodings stored once, pass-through;
- the comparison script;
- the runner's records;
- the test-label guard.

**Step 8 dry run on the laptop's GTX 1650.** Declared in A28 as a code-path test that does not count. It completed all 5 videos in 9.7 minutes, with peak RAM 4.8 GB, and read only 23 frames.

| | result |
|---|---|
| pairs, their order, windows | **identical** to `cache_s3` on all 5 videos |
| geometry inputs (`rel_feat`, `mot_feat`, `bbox_feat`) | **identical** (max diff 0.0) |
| CLIP features (`global_proj`, fp16) | differ on all 23 frames; max 0.011 |
| logits | max 0.0158 (int-logit 0.0052): misses 1e-4, as expected off the 4090 |

The code path matches session 3. The divergence enters only at fp16 CLIP. That fits the GPU architecture (sm_75 against session 3's sm_89), but on the laptop it cannot be told apart from frame bytes. The 4090 run will settle it, since its frame-provenance check compares each requested frame's `global_proj` with the cache. Logs: `studies/rpt/g0/logs/step8_dryrun_laptop*.{txt,json}`. The dry run's tensors (`dryrun_laptop/step8/*.pt`, 512 KB) are git-ignored by the repo's `*.pt` rule; their SHA256s are in the committed `manifest.json`.

## 4. What happens next

1. The author requests access on `https://huggingface.co/facebook/sam3` (Meta approves by hand), then runs `bash studies/rpt/g0/fetch_sam3_ckpt.sh` in their own terminal.
2. I write A28a with the SHA256 and the blob check, hash it, and commit.
3. The laptop check: SAM 3 on `…00010024`, "bicycle", default settings, a 30-minute cap.
4. The machine (approved): steps 8 and 9, then the measurement.
5. The report, and the G1 draft.

If access has not come within 2 days (by 2026-10-28), I will ask you about a step-8-only machine.
