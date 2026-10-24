# D2 — at which front-end stage are D1's U1 object tracks lost?

Pre-registration: **A25** (definitions, reading rule) and **A25a** (code pins) in
[`../prereg/preregistration.md`](../prereg/preregistration.md). This is a disclosed
test-label error analysis: no method is evaluated, and nothing here fixes a parameter of
any future model.

| file | what |
|---|---|
| `stage_map.md` | Step 0: EOV-MMP's front end, stage by stage, with line references |
| `recorders.py` | Step 1: read-only hooks (post-processor hook + four wrappers) |
| `record_run.py` | Step 2: the GPU recording pass, with ground rule 3 checked after every video |
| `stages.py` | oracle links, pre-filter tracks, the filter replay, the stage classifier, the bootstrap |
| `analyse.py` | Steps 3–4: verification, then the test labels, then the tables and the reading |
| `baseline_tracks.py` | the baseline's final tracks from session 3's SGDet cache, and ground rule 3's comparison |
| `pin.py` | writes `logs/code_pins.json` (A25a; re-pinned by A25b, A25c) |
| `box/` | Step 2 on its own machine (A25c): upload, setup, the author's token fetch, run, cleanup, copy-back, token scan, expected hashes |
| `common.py` | A25's constants, the input pins, logging |
| `tests/` | stubs and synthetic cases; CPU only, no CLIP |

`data/` (git-ignored) holds `baseline_tracks.json` and, later, the analysis outputs.
`logs/` holds the pins, the test logs and the prereg append log.

## Laptop (CPU)

Since A25b the study folders live under `METOR/studies/`, so the `d2` commands run from
there.

```bash
PY=~/miniconda3/envs/repro-next/bin/python          # python 3.12, torch 2.11.0+cu128 (CPU use only)
cd METOR/studies
$PY -B -m d2.baseline_tracks                          # -> d2/data/baseline_tracks.json
$PY -B -m d2.pin                                      # -> d2/logs/code_pins.json
$PY -B -m pytest -p no:cacheprovider d2/tests -v      # on the EOV code root (default)
D2_CODE_ROOT=$(dirname $PWD) $PY -B -m pytest -p no:cacheprovider d2/tests -v   # on METOR's vendored copy
cd .. && $PY -B -m pytest -p no:cacheprovider tests -q   # METOR's own suite, from the repo root
```

The d2 tests put the EOV code root first on `sys.path`, so they must run in a session of
their own. A session that also collects `tests/` fails.

## Step 2 — GPU (only after the estimate is approved)

The run uses its own machine (A25c): one RTX 4090 with ≥ 32 GB RAM and ≥ 60 GB disk, on
the Vast.ai PyTorch template, with `OMP_NUM_THREADS=8`. The scripts are in `box/`; the
machine's layout is `~/ov-vidvrd` (EOV code and data), `~/METOR/studies`, `~/d2_box`
(logs) and `~/d2_rec` (recordings).

| # | where, who | command |
|---|---|---|
| 1 | laptop | `bash studies/d2/box/upload.sh root@HOST PORT`: `git archive fdeeeb4`, `box/upload_files.txt`, `studies/` (copy, don't push) |
| 2 | machine, Claude | `bash ~/METOR/studies/d2/box/setup.sh`: preflight, the 519 uploaded files' SHA256, dependencies, CUDA operator, CLIP |
| 3 | machine, **the author** | `ssh -t … 'bash ~/METOR/studies/d2/box/fetch_with_token.sh'`: the token is read hidden, held only in that process's `HF_TOKEN`, then unset and scanned for; weights, CLIP bank, ECC, frames, all SHA256-checked |
| 4 | machine, Claude | `run.sh check`, then `run.sh first_last` (ground rule 3; the 25% timing rule; stop and ask if the projection is over 6 h), then `nohup setsid bash run.sh full` |
| 5 | machine, Claude | `cleanup.sh`: deletes the test annotations right after the run, after logging their SHA256 |
| 6 | laptop | `bash studies/d2/box/copy_back.sh root@HOST PORT`: recordings to `d2/data/rec/`, logs to `d2/logs/box/`, every file checked against the manifest |

- **Any track difference from the baseline** writes `d2_rec/STOP_ground_rule_3.json` and
  exits 3. Nothing is fixed or rerun without approval.
- **After the verified copy-off,** the author destroys the machine.

## Steps 3–4 — laptop

```bash
cd METOR/studies && $PY -B -m d2.analyse --rec_dir d2/data/rec
```

It verifies everything first:
- the A25 input pins, the 204 cache files and the 200 annotation files (as bytes);
- the code pins and the baseline;
- every recording, with ground rule 3 and the replay check on every video.

**Only then** does it open the test labels. It writes `data/d2_tables.{json,md}`,
`data/d2_relations.json`, `data/d2_funnel.json` and `logs/d2_analyse.json`. The report
is `d2_report.md`, and then D2 stops.
