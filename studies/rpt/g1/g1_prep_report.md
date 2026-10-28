# RPT G1: A29 hashed, G1 prepared, machine search (decisions needed before renting)

2026-10-28 · repo `ov-vidvrd-lab` · commits `d3fe998` (A29) and `a6ae56b` (G1 preparation) · nothing
rented, nothing pushed

## Step 1: A29 hashed

- **A29's text SHA256** (the appended bytes, with their leading newline, 26,508 bytes):
  `09345b004da68d859b05022ea7ef8a2574ef2d7acfce091caf55ff082c9a830b`.
- **New chain head:** `0d62e838…947b` → **`b0f1ec509d3b77d03c58db8f98e63f6e3df480fa34c4e7dbb2447225c8552fd0`**.
- **Checks:** the old 110,690 bytes are unchanged, the appended bytes equal the text, and both
  prereg files are back at 444.
- **Backups and log** are in `studies/rpt/logs/`: `preregistration.after-A28a.bak`,
  `preregistration.sha256.before-A29.bak` and `prereg_log.txt`.
- **Checker:** `verify_after_A26m.py` **passes all 75 checks**: the earlier 72, plus the after-A28a
  line and the two new backups (`studies/rpt/logs/verify_after_A29.txt`).
- **Commit:** `d3fe998`. The credential scan (grep, 11 patterns; gitleaks is not installed) found 0
  hits. Committed with A29: the review, `a29_for_review.md`, `a29_changes_round2.md` and the final
  draft.
- **Edits beyond removing the header:**
  - the title says `(2026-10-28)` instead of `(draft)`;
  - the label line now reads "A29 is the next free one … appended … after A28a, whose head is
    `0d62…`", replacing "would be … stays … until this is hashed".

  Nothing else changed.

## Step 2: G1 prepared on the laptop (no GPU)

All new code is in `studies/rpt/g1/`. No existing file was edited.

1. **The 3 missing videos** (`held_gt_missing3.py`).
   - Stage 2's construction (`step1_parts.py:40-53`, imported, not copied) is first re-run on the
     157 videos in `held_gt.json`, and **reproduces it exactly**.
   - It then gives 7, 14 and 4 base relations for `…00265063`, `…00679000` and
     `ILSVRC2015_val_00123004`: 25 in all.
   - **`held_gt_missing3.json`, SHA256
     `64c0233f9619d023917247078303be062ddddd6156adf39abadad107beb54afd`**, committed before any G1
     run.
2. **The analysis** (`metrics.py`, `analyse.py`). It imports D1's `cut_viou`/`best_viou`
   (`studies/d2/stages.py`) and the official `viou`/`iou`; nothing is re-implemented. It covers:
   - cut-vIoU run matching with A29's assignment and tie-breaks;
   - unmatched tracks;
   - box style on annotated frames;
   - pair recall (U1/U2, plus scoreable pairs);
   - item 8;
   - track counts by *N*, gap exposure, tid recall and the class-agnostic counts;
   - the bootstrap CIs and the verdict table.

   **Dry run** on G0's 3 smoke videos, with stand-in EOV-side files: the code path only, no result.
   Its gap-exposure clip count (48) equals G0's step-9 count, which was computed with real features.
3. **Unit tests: 71 passed** (`studies/rpt/g1/logs/pytest_rpt_g1.txt`). 36 are new and cover:
   - each case you listed: a track bridging two runs of one tid; a track over two tids; one track
     per run; the tie-breaks; a partial cut; annotated frames only; scoreable pairs under 10 frames;
   - D1's quantity equality;
   - item 8;
   - the scheduling rule and the projection;
   - check 5's drop order;
   - the worker's offload switch;
   - the box-sensitivity inputs through the repo's `gen_feats_test`;
   - **the machine driver end to end, with a fake SAM 3 worker**, for each case:
     - all checks pass;
     - check 0 mismatches (stop);
     - A3 mismatches (the size of the difference is reported, and G1 continues);
     - check 5 drops offload, then stops before `…00280000`;
     - the 35-hour cap is reached (stop);
     - a first-20 video fails (stop).
4. **The machine scripts** (`box/`):
   - `upload.sh`: the 160 HELD videos' frames (6.7 GiB) and annotations, and the data EOV-MMP's
     front end reads. **No test file**, not even the test ECC.
   - `setup.sh`: stops unless the driver is 580.142, the RAM is ≥ 192 GB, and Python, torch and
     both package sets equal G0's `pip freeze`. It records both cuDNN versions.
   - `fetch_with_token.sh`: as G0's.
   - `run.sh`, which runs:
     - `g1_driver.py`: checks 0–5, the cap, the projection, scheduling and every stop;
     - `sam3_worker.py`: the committed runner, unchanged, with A1, A3 and offload switches;
     - `eov_held.py`: R0w and R0 with the train ECC, then box sensitivity.
   - `copy_back.sh`: every file checked against the machine's SHA256.
5. **EOV-MMP's HELD tracks** (`eov_held.py`). It runs D2's front end unchanged, with only these
   differences:
   - the loader is `Dataset_new(args, 'train')`, restricted to HELD;
   - `model.test_ecc` is the train ECC (the 3 absent videos run without compensation);
   - R0w is A26's `whole_video`.

   It checks every pin first: the checkpoint, AFLink, the train ECC, the train trajectories,
   `held_gt.json` and the missing-3 file, plus A25a's 22 front-end files. All match on the laptop.

## Step 3: the machine — two problems, so I have not rented

1. **No RTX 4090 offer has G0's driver, 580.142.** A29 pins it, and `setup.sh` stops on any other
   driver. Today's 1× 4090 offers run 580.178/173/159/126/82, 590, 595, 610, 615 or 565. The closest
   branch with enough RAM is 580.178.04 (New York, $0.93/h). A29 already says check 0 (bitwise
   reproduction of G0) decides whether the stack matches, for the image and cuDNN.
   **Decision needed:** may the driver pin be relaxed, with check 0 deciding as it does for the image
   and cuDNN, or must G1 wait for a 580.142 machine?
2. **No ≥ 192 GB offer near $0.35/h.** The ≥ 192 GB RAM, 1× 4090 offers today:

| offer | $/h | RAM | disk | location | driver | verified | reliability |
|---|---|---|---|---|---|---|---|
| 52126082 | **0.602** | 192 GB | 2,878 GB | Romania | 595.71.05 | yes | 0.994 |
| 53748556 | 0.736 | 257 GB | 734 GB | Norway | 595.91.07 | yes | 0.988 |
| 46640475 | 0.804 | 257 GB | 490 GB | Texas, US | 615.71.09 | yes | 0.992 |
| 57209633 | 0.930 | 343 GB | 1,925 GB | New York, US | 580.178.04 | yes | 0.974 |
| 34171098 | 0.938 | 193 GB | 1,060 GB | Singapore | 595.84 | yes | 0.994 |

   **Expected total**, assuming:
   - SAM 3: 25–35 GPU-hours (G0's 43 h at defaults ÷ 1.36 for A1 ≈ 32 h; A2 and A3 unmeasured);
     the cap is 35;
   - EOV-MMP side: about 3 h;
   - set-up, preflight and copy-back: about 1.5 h.

   That is **about 30–40 h**:
   - **$18–24** at $0.602/h (Romania, 192 GB);
   - $22–29 at $0.736/h (Norway, 257 GB).

   Your approved budget was "up to about $15". **Decision needed:** approve one of these, or say what
   else to do. I will not rent until you approve.

## Points where I had to choose (please confirm)

1. **Check 5 and offload.**
   - Check 5 runs all passing options together, offload included, as written.
   - The videos then use offload only for `…00280000`, and A1 only for ≤ 1,000 frames, per the
     scheduling rule.
   - If check 5 drops offload, G1 runs the other 159 videos and stops before `…00280000`.
2. **Check 2:** 3 processes are tried only if 2 pass. Note that with the 21 GiB rule, 3 processes
   fit together only on videos of about 240 frames in total, so A2 = 3 is rarely used.
3. **A failure of any video stops G1**, not only one in the first 20. A missing video would make the
   bars unjudgeable on all 160.
4. **Box sensitivity uses EOV-MMP's R0w tracks** (A29's primary). Its relation selection is
   recomputed on the laptop and must match the machine's.
5. **Item 8's "box IoU ≥ 0.5"** uses the official `iou` (`video_relation_detection_openvoc.py:48-70`).
6. **A preflight before the cap's clock starts.** EOV-MMP's side runs on 2 smoke videos with G0's
   SAM 3 files, to catch errors before the 30-hour SAM 3 run. It counts in the budget, not the cap.
7. **The EOV-MMP side runs after all 160 SAM 3 files exist**, since box sensitivity needs them.
8. **The image digest is not visible inside a Vast.ai container.** `setup.sh` records the OS and the
   container label, and the author should note the digest from the instance page when renting.

Stopped here: nothing is rented and nothing has run on a GPU.
