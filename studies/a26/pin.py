#!/usr/bin/env python3
"""Laptop: the pins A26 cites -> a26/logs/code_pins.json. run.py checks them all on the machine.

    cd METOR/studies && python -m a26.pin

metor_code  every tracked .py file of METOR outside studies/ and tests/ (the code the run imports)
a26_code    every a26/*.py
inputs      the checkpoint, AFLink, the test ECC, the class splits (the laptop's copies, which the
            machine's must equal), D2's baseline tracks (session 3's, extracted before any
            recording) and the gate reference (make_reference.py)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from a26.common import AFLINK, CKPT, EOV_ROOT, LOGS, METOR_ROOT, REFERENCE  # noqa: E402
from a26.run import sha256  # noqa: E402


def main():
    files = subprocess.check_output(["git", "-C", METOR_ROOT, "ls-files", "*.py"], text=True).split()
    metor = {f: sha256(os.path.join(METOR_ROOT, f)) for f in sorted(files)
             if not f.startswith(("studies/", "tests/"))}
    a26 = {f: sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")}
    meta = os.path.join(EOV_ROOT, "data", "vidvrd", "data")
    inputs = {"ckpt": os.path.join(EOV_ROOT, CKPT), "aflink": os.path.join(EOV_ROOT, AFLINK),
              "ecc_test": os.path.join(meta, "VidVRD_ECC_test.json"),
              "obj_split": os.path.join(meta, "openvoc_obj_class_spilt_info.json"),
              "pred_split": os.path.join(meta, "openvoc_pred_class_spilt_info.json"),
              "baseline_tracks": os.path.join(os.path.dirname(HERE), "d2", "data", "baseline_tracks.json"),
              "reference": REFERENCE}
    pins = dict(metor_code=metor, a26_code=a26, inputs={k: sha256(p) for k, p in inputs.items()})
    os.makedirs(LOGS, exist_ok=True)
    out = os.path.join(LOGS, "code_pins.json")
    with open(out, "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True)
    print(f"{out}: {len(metor)} METOR files, {len(a26)} a26 files, {len(inputs)} inputs; sha256={sha256(out)}")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
