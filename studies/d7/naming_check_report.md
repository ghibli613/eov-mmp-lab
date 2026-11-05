# The naming check (A33, A33a, A33b): STOP, naming is the bottleneck

2026-11-05 · HELD-A, 78 videos, in-sample for modelB and R0w on base objects · not a test result · run on a
rented L40 (A33b), analysed on the laptop · D7 not run, by A33's rule and Giang's decision

## Verdict

**STOP. modelB alone loses all of the 35-name SAM 3's recall gain, so the D7 rental request is not
prepared and P4 is redesigned first** (A33's binding stop-and-report rule).

The rule runs on class-aware object recall (A29 run recall) gain over EOV-MMP R0w, on the same 78
videos:

| tracks named by | gain | half of (i) | rule |
|---|---|---|---|
| (i) the 35 prompt names | **+5.73** | +2.86 | above +2, so the check can discriminate |
| (ii) modelB alone | **−0.52** | | below +2.86: **stop** |

**The cause is novel naming.** On SAM 3 tracks that match a GT object (class-agnostic), modelB names
97.8% of base objects correctly, but only **26.3% of novel objects**.

## 1. The three systems

Class-aware unless marked. Recall counts are hits / GT runs.

| | (i) prompt names | (ii) modelB names | (iii) R0w |
|---|---|---|---|
| object recall | 71.6% (275/384) | 65.4% (251/384) | 65.9% (253/384) |
| base | 74.0% (202/273) | **82.8%** (226/273) | 77.7% (212/273) |
| novel | 65.8% (73/111) | **22.5%** (25/111) | 36.9% (41/111) |
| pair recall (2,188 base relations) | 56.3% | **62.0%** | 57.4% |
| burden (unmatched confident tracks per video) | 1.68 | 1.76 | 0.95 |
| class-agnostic object recall | 84.9% | 84.9% (same tracks) | 74.2% |
| class-agnostic novel recall | 85.6% | 85.6% | 64.9% |

| gain over R0w, points | overall | base | novel | pair recall | burden ratio |
|---|---|---|---|---|---|
| (i) prompt names | +5.73 | −3.66 | +28.83 | −1.14 | 1.77× |
| (ii) modelB names | −0.52 | +5.13 | −14.41 | +4.57 | 1.85× |

- **The tracks find the objects:** class-agnostically, the 35-name SAM 3 recalls 84.9% against R0w's
  74.2% (novel: 85.6% against 64.9%).
- **Naming decides the outcome.** modelB's names raise base recall (+5.13 over R0w) and pair recall
  (+4.57, base relations only). But they recover only 25 novel runs, against the prompt names' 73
  and R0w's 41.
- **On HELD-A the prompt-name gain is +5.73,** below G1's +9.80 on all 157 HELD videos. The guard of
  +2 is passed comfortably.

## 2. Naming accuracy

| | base | novel |
|---|---|---|
| modelB, on tracks matched class-agnostically (by the GT run's class) | 97.8% (226/231) | **26.3%** (25/95) |
| the prompt names, on the same matches | 87.4% (202/231) | 76.8% (73/95) |
| modelB top-1 on HELD-A GT boxes (every annotated frame of every run) | 98.1% (13,056/13,305) | **35.0%** (1,842/5,265) |

modelB's base accuracy is in-sample (it was trained on the training videos), so it is optimistic. The
novel split is the honest part.

## 3. Which classes modelB picks for novel objects (HELD-A only; descriptive)

From `studies/d7/naming_confusions.py`; output `studies/d7/naming_check/confusions.json`. 9 of the 10
novel classes occur in HELD-A; lizard does not.

| novel class | GT runs (videos) | boxes correct | modelB's most common picks, per box | matched tracks: modelB's names | matched tracks: prompt names |
|---|---|---|---|---|---|
| antelope | 38 (7) | 32% of 1,290 | antelope 415, **sheep 246**, **bird 184**, **zebra 176**, fox 112 | zebra 10, antelope 8, sheep 5 (of 31) | sheep 18, antelope 10 |
| watercraft | 12 (4) | 42% of 915 | **airplane 404**, watercraft 382, train 101 | airplane 4, watercraft 2 (of 6) | watercraft 6 |
| red_panda | 16 (1) | 22% of 810 | **monkey 364**, red_panda 178, **tiger 148**, fox 83 | **monkey 14** (of 14) | red_panda 14 |
| bear | 11 (2) | 1% of 600 | **dog 155**, **sheep 146**, **tiger 141**, whale 73 | **tiger 10**, person 1 (of 11) | bear 10, person 1 |
| giant_panda | 12 (3) | 0% of 420 | **person 121**, bird 65, red_panda 54, zebra 45 | person 6, red_panda 4, tiger 1 (of 12) | giant_panda 12 |
| cattle | 2 (1) | 93% of 390 | cattle 363, sheep 19 | cattle 2 (of 2) | cattle 2 |
| horse | 9 (3) | 59% of 360 | horse 211, **zebra 90**, cattle 28 | horse 7, zebra 2 (of 9) | horse 9 |
| fox | 7 (4) | 94% of 300 | fox 283, bird 11 | fox 6 (of 6) | fox 6 |
| motorcycle | 4 (4) | 2% of 180 | **person 115**, **bicycle 53**, car 7 | bicycle 2, person 2 (of 4) | motorcycle 4 |

**The most common box confusions, over all novel classes:**
- watercraft → airplane (404);
- red_panda → monkey (364);
- antelope → sheep (246);
- antelope → bird (184);
- antelope → zebra (176);
- bear → dog (155);
- red_panda → tiger (148);
- bear → sheep (146);
- bear → tiger (141);
- giant_panda → person (121);
- motorcycle → person (115);
- antelope → fox (112).

**What this shows (descriptive):**
- **modelB mostly picks a visually or contextually close base class.** Water and sky vehicles become
  airplane or train. Animals become sheep, zebra, tiger, dog or monkey. A motorcycle's box, usually
  with its rider, becomes person or bicycle.
- **Three novel classes are named well:** cattle, fox and horse.
- **Four are almost never right:** bear, giant_panda, red_panda and motorcycle get 0 of 41 matched
  tracks.
- **The samples are small and clustered.** red_panda is one video. Each class has 2–38 runs.

## 4. Were the pooled CLIP features per track saved?

**No.**
- **What the outputs store, per video:**
  - per track: modelB's mean 35-vector, and modelB's top-1 for every detected-frame box;
  - per GT run: the same.
- **What was not saved:** the pooled TagCLIP features (768-d per box) were computed in memory and
  discarded.

**Recomputing them** means only TagCLIP's frame encoding plus the pooling; modelB is not needed.
- **On an L40:** this run's TagCLIP encoding took **552 s** for the 20,036 frames (0.028 s per frame);
  modelB took the other 1,846 s.
  - So about **10 minutes of compute.**
  - Plus the rental's setup, about 30–40 minutes, mostly the 2.5 GiB frame upload.
  - About **$0.40** at this L40's price. No Hugging Face token is needed: CLIP's weights are public.
- **On the laptop's GTX 1650:** about **1.8 hours** at the measured 0.33 s per frame, for $0.
- **Size:** 59,647 boxes × 768 in fp16 is about 92 MB. Per-track means (294 × 768) are negligible.

## 5. Run, protocol and cost

- **The first laptop run (A33)** stopped at its first box on a shape bug. It produced no output and
  was fixed before any result (A33a: the missing `squeeze(0)` of `end2end_model.py:253`).
- **The laptop rerun under A33a** was stopped by Giang after 5 of 78 videos. Its outputs are kept in
  `naming_check/laptop_stopped/`, not analysed and not used (A33b).
- **The rented L40 (A33b):**
  - Vietnam, instance 59255065, $0.546/h with its 200 GB disk;
  - G1's `setup.sh` passed every check: the 21,762 data and 370 code files hashed, `/venv/main`
    equal to G0's EOV-MMP freeze, `~/sam3env` equal to G0's SAM 3 freeze;
  - Giang's token fetch: every weight OK, token scans 0;
  - **the naming check took 40.0 minutes** (2,402 s), against about 9 hours on the laptop;
  - **the copy-back matched:** 78 of 78 outputs against the machine's SHA256.
- **Incidents, both handled:**
  - the container restarted 22 s after the auto-stop's first key test (`vastai start instance` on a
    running instance). The test is now read-only (`vastai show instance`), and the watchdog was
    re-armed with the original deadline;
  - `vastai destroy` needs `-y` when run without a terminal.
- **Destroyed** at 12:47 UTC from inside, with the per-instance key; the port closed within 20 s.
- **Cost:** about 1.45 h × $0.546 ≈ **$0.80**. An earlier rental (an RTX A6000, an Ampere card that
  did not meet the Ada requirement) was destroyed within minutes, with nothing run on it.

## 6. Disclosure

- **In-sample:** modelB and R0w both are, on HELD's base objects, so their base numbers are
  optimistic. The novel split is the honest part.
- **HELD-A only.** HELD-B was not touched: the guard was installed in every script.
- **The confusions in §3** are descriptive, asked for after the verdict, and decide nothing.

## Files

- `studies/d7/naming_check/videos/` (78), `manifest.jsonl`, `run.txt`: the L40's outputs and log;
- `studies/d7/naming_check/results.json`: the measures and the verdict;
- `studies/d7/naming_check/confusions.json`: §3;
- `studies/d7/rental/box_logs/`: the machine's setup, fetch, auto-stop and run logs;
- code: `studies/d7/naming_check.py` (pinned in A33a), `studies/d7/naming_confusions.py`.
