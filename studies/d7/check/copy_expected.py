#!/usr/bin/env python3
"""A33b's one named exemption from A31's guard: copy G0's two pinned package lists
(studies/rpt/g0/logs/box/pip_freeze_main.txt and pip_freeze_sam3env.txt, in a sealed folder) to
studies/d7/check/expected/.

Each file is a pip freeze. It is refused unless it names no video at all (either ID form) and is the
file committed with G0 (its `git hash-object` equals the blob at HEAD). Runs WITHOUT the HELD-B guard, by
the exemption, and reads those two files only.

    python studies/d7/check/copy_expected.py
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FILES = ["pip_freeze_main.txt", "pip_freeze_sam3env.txt"]
SRC_DIR = "studies/rpt/g0/logs/box"
DST_DIR = "studies/d7/check/expected"
ID = re.compile(rb"ILSVRC2015_(?:train|val)_\d{8}")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def main():
    os.chdir(REPO)
    os.makedirs(DST_DIR, exist_ok=True)
    for name in FILES:
        src, dst = f"{SRC_DIR}/{name}", f"{DST_DIR}/{name}"
        data = open(src, "rb").read()
        if ID.search(data):
            sys.exit(f"!! STOP: {src} names a video; it is not copied")
        if git("rev-parse", f"HEAD:{src}") != git("hash-object", src):
            sys.exit(f"!! STOP: {src} differs from the committed one")
        shutil.copyfile(src, dst)
        print(f"copied {src} -> {dst}: {len(data.splitlines())} lines, SHA256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
