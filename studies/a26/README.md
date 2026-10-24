# A26 — the protocol audit (R0u, R0 over the whole video, both)

Pre-registration: [`../prereg/preregistration.md`](../prereg/preregistration.md), A26. Plan:
[`../design_review/agreed_plan.md`](../design_review/agreed_plan.md), step 0.

The released EOV-MMP checkpoint, four ways, on the 200 test videos (SGDet):

| config | frame range | fusion |
|---|---|---|
| R0 | the GT-derived range, as released | per-class, as released |
| R0u | the GT-derived range | uniform 0.3 / 0.7 (A23's R0u) |
| R0w | the whole video | per-class |
| R0uw | the whole video | uniform |

The reading is fixed in A26 and applies to one quantity: Δuw = R0uw − R0, SGDet novel mAP.

## Files

- `common.py`: every number A26 fixes.
- `wrappers.py`:
  - `whole_video` sets the frame range to the whole video;
  - `set_fusion` switches the fusion;
  - `BothSplits` gives both predicate splits from one pass (pilot B.21).
- `tracks.py`: the identity-gate comparisons (copies of D2's, so no D2 code runs on the machine).
- `run.py`: the GPU run, one encoding per video and four forwards.
- `evaluate.py`: the laptop evaluation, with the official evaluator, the paired bootstrap and the
  reading.
- `estimate.py`: the time and cost estimate, and what the machine needs.
- `make_reference.py`: session 3's R0 predictions for the 2 gate videos, written to
  `data/r0_reference.json`.
- `pin.py`: writes `logs/code_pins.json`, which `run.py` checks on the machine.
- `tests/`: CPU only, with stubs. Run from `studies/`: `python -m pytest a26/tests`.
- `box/`: the machine flow.

## Commands

**Laptop:**
```bash
cd METOR/studies
python -m a26.estimate
python -m a26.make_reference && python -m a26.pin     # already done for A26; rerun only with a pin addendum
```

**Machine** (only after the author approves the spend):
```bash
bash studies/a26/box/upload.sh root@HOST PORT                    # laptop; METOR tree clean
ssh ... 'bash ~/METOR/studies/a26/box/setup.sh'
ssh -t ... 'bash ~/METOR/studies/a26/box/fetch_with_token.sh'    # the author, interactively
ssh ... 'bash ~/METOR/studies/a26/box/run.sh check'
ssh ... 'bash ~/METOR/studies/a26/box/run.sh gate'               # R0 identity on 2 videos, timing, projection
ssh ... 'nohup setsid bash ~/METOR/studies/a26/box/run.sh full > ~/a26_box/run_full.log 2>&1 &'
ssh ... 'bash ~/METOR/studies/a26/box/cleanup.sh'                # test annotations deleted, logged
bash studies/a26/box/copy_back.sh root@HOST PORT                 # laptop
```

**Laptop, after the copy-off:** `python -m a26.evaluate`. This writes `data/a26_results.json`.
