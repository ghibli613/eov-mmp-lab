# RPT G1, stop 1: `…00280000` does not fit on an RTX 4090, even with offload — decision needed

2026-10-28, 15:50 UTC · machine: Vast.ai, 8× RTX 4090, rented by the author · commit on the machine
`e619a7d` · nothing pushed

**The decision needed now:** how to handle one video. The machine is idle and billed meanwhile, and
the cap clock keeps running (section 4).

## 1. What happened

- **The stop.** At 15:42:52 UTC, `ILSVRC2015_train_00280000` (2,947 frames) ran alone on GPU 0 with
  `offload_video_to_cpu=True` and failed with **CUDA out of memory**. A29a's rule "any video that
  fails to run stops G1" fired. The driver stopped every process; the other running videos were
  killed before writing (no partial files).
- **The memory.**
  - The first prompt (`airplane`) completed: all 2,947 frames in 270.6 s, but already at a peak of
    **22.19 GiB allocated**, with no masklet started.
  - The second prompt failed at 22.79 GiB allocated, on a card with 23.54 GiB.
- **Why check 4 did not catch it.** Offload moves only the video frames to the CPU; SAM 3's per-frame
  inference state stays on the GPU and grows with the video's length. Check 4 runs on smoke video 1
  (130 frames), where that state is small, so it passed bitwise at 7.3 GiB. G0's memory formula
  (6.5 GiB + 6.3 MiB × frames = 24.6 GiB for this video) already said it would not fit without
  offload; offload saved less than needed.
- **What is kept:** 11 of the first 20 videos finished and are on disk, each with its SHA256 in the
  manifest (1,017 frames). The other 149 videos (46,021 frames) have not been run or are not
  complete.

## 2. Everything before the stop went as planned

| | result |
|---|---|
| set-up | both environments equal G0's `pip freeze` (`/venv/main` `ed46e4da…`; SAM 3's venv); every uploaded file and fetched weight matched its SHA256 |
| unit tests, on the machine | 77 passed; 2 failed only because the laptop-only archive bundle (`~/Documents/thesis/archive/…bundle`) is not uploaded (the chain checker; it passed on the laptop) |
| EOV-MMP preflight, GPU 0 vs GPU 1 | **identical** on both smoke videos, apart from the timing fields |
| **check 0, all 8 GPUs** | **identical to G0 on every GPU** (360–389 s each, against G0's 423 s; peak 7.32 GiB) |
| check 1, A1 | identical; 200 s against 389 s (1.9×) |
| check 2, A2 with 2 processes | identical; throughput 1.04× |
| check 2, A2 with 3 processes | all 3 processes ran out of GPU memory, so not identical (the 21 GiB rule had predicted this) |
| check 3, A3 `torch.compile` | **not identical**: max \|Δ out_probs\| 0.0032; max \|Δ box\| 26 px; mask IoU min 0.961, mean 0.997, below 1 on all 390 shared masks; no masklet or frame added or lost. Not used. |
| check 4, offload | identical (on the 130-frame smoke video) |
| check 5, A1 + A2 (2) + offload, at the videos' 6 OpenMP threads | identical, so this is the configuration in use |

The checks used 3.9 billed GPU-hours. The first 11 videos ran at 0.063 s per frame per prompt per
process, with 2 processes per GPU: about 2.7× G0's single-GPU throughput.

## 3. Set-up incidents, already resolved (for the record)

1. **The template version.** The current "PyTorch (Vast)" image (`vastai/pytorch`, default tag) ships
   torch 2.14.1+cu132. `setup.sh` installed G0's exact package set instead. Removing the image's CUDA
   13 packages first was needed, because they share folders with CUDA 12's.
2. **The driver** is 610.43.02, recorded as A29a allows. Check 0 then passed on all 8 GPUs.
3. **EOV-MMP's detector operator** (`MultiScaleDeformableAttention`) had to be compiled, as the repo's
   `tools/setup_rented_box.sh:144-149` does. That needed nvcc 12.8, from `cuda-toolkit-12-8` in
   NVIDIA's apt repository (no driver package). G0 never ran the detector. `/venv/main` is G0's
   freeze plus this one package.
4. **A test file, removed.** The author's weight fetch (`--only eval`, as in G0) also downloaded
   `VidVRD_ECC_test.json` (camera motion of test videos, not labels). It was deleted unopened and
   logged (`~/rpt_g1_box/test_file_removed.log`).
5. **The hard-coded `'test'` annotation folder.** EOV-MMP's forward pass passes `'test'` to
   `gen_feats_test` (`end2end_model.py:326`), which reads the video's annotation for its id, height
   and width only (`gen_labels.py:25-29`). The test-label guard blocked the attempt in the preflight;
   nothing was opened. For HELD videos it now reads the training annotation, by module-attribute
   shadowing from outside. The final tracks are set before that call (`:325`).
6. **The weight fetch** was run by the author before the upload had finished. Its final hash check
   ran later: all 5 files OK; token scans 0.

## 4. Cost and clock

- **Machine:** $3.80/h ($0.475 per GPU-hour, within A29a's $0.50); about $7 spent so far.
- **The cap's clock:** 5.73 billed GPU-hours at 15:50 UTC. It keeps running while the machine waits
  (A29a defines it as billed wall clock × 8), at **8 GPU-hours and $3.80 per hour of waiting**.
- **The rest of the plan, for 159 videos:** at the measured rate, about 1.6–1.8 h of wall clock, or
  13–15 GPU-hours. Then the EOV-MMP side (about 0.5 h) and copy-back.
  - **About $17–19 in total**, plus the waiting time and whatever `…00280000` needs.
  - **About 20 GPU-hours for the SAM 3 phase**, well under 48, unless the wait is long.

## 5. The decisions

**D1. The other 159 videos.** My recommendation: run them now, on this machine, with `…00280000`
left out, and decide that video separately (D2). This is what A29a already prescribes when offload
does not pass for that video ("`…00280000` is not started; the other 159 run; then stop and ask").
The code change is an explicit exclude list in the driver, recorded in its log.

**D2. `…00280000`** (14 GT runs from 12 tids, all *cattle* (a novel class); 59 base GT relations).
Options:

- **(a) A29's own fallback: "a 48 GB GPU for that video, at defaults."**
  - A separate short rental, e.g. an RTX 6000 Ada or A6000: about 1 h, about $1. One process,
    defaults, no offload (G0's formula predicts about 24.6 GiB).
  - Open question: check 0 cannot be expected to reproduce G0 bitwise on another GPU model. Should it
    run anyway, and must it pass? If it must and fails, this option ends there.
- **(b) SAM 3's `offload_state_to_cpu=True`** (also its own start-session option,
  `sam3_base_predictor.py:140-147`), with or without the video offload, on an RTX 4090 of this
  machine.
  - Not pre-registered: it needs a short amendment, then a bitwise check on smoke video 1 like check
    4 (about 6 min), then the video (likely slower: state moves over PCIe).
  - Same GPU model as everything else, and the 503 GB of RAM is ample.
- **(c) Leave `…00280000` out of G1.** An amendment would judge every bar on 159 videos. This removes
  14 of the 753 GT runs and 59 of the base relations, all *cattle*.

**D3. The clock while waiting.** Should the time spent waiting for this decision count toward the 48
GPU-hour cap, as A29a defines it, or be excluded? This matters only if the wait is long.

**What I will do on your answer:** for D1-yes, the driver change (with a test), commit, re-sync, and
restart. The 11 finished videos are kept and not rerun; the projection after the first 20 runs as
A29a says. For D2 (b), I draft the amendment for you first.
