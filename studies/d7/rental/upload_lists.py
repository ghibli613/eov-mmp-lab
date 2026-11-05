#!/usr/bin/env python3
"""A33b: what goes to the rented machine, as lists. Writes, into STAGE:
- code_files.txt: committed files (git ls-files at HEAD) of an explicit set of code folders, plus the 78
  HELD-A videos' G1 SAM 3 files and G0's check-0 reference (...00010024, HELD-A);
- data_files.txt: EOV-MMP's class files, the 78 videos' training annotations and every frame of the 78;
- expected_code.sha256 and expected_data.sha256 (relative paths, for G1's setup.sh step 1);
- pip_freeze_main.txt, pip_freeze_sam3env.txt (A33b's exemption copies) and G1's expected_fetch.sha256.
Refuses any path that names a HELD-B video (either ID form) or looks like a test file, and any HELD-A
video without a G1 SAM 3 file. Runs with the HELD-B guard installed.

    python studies/d7/rental/upload_lists.py STAGE
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
D7 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(D7))
sys.path.insert(0, D7)
import held_b_guard  # noqa: E402
held_b_guard.install()

CODE_DIRS = ["cli", "configs", "data_loading", "inference", "models", "ops", "tools", "utils", "vlm", "third_party",
             "requirements.txt", "studies/rpt/g0/box", "studies/rpt/g1/box", "studies/d7"]
CODE_GLOBS = [":(glob)studies/*.py", ":(glob)studies/d2/*.py", ":(glob)studies/rpt/*.py", ":(glob)studies/rpt/g0/*.py",
              ":(glob)studies/rpt/g1/*.py",
              ":(glob)pilot_analysis/stage1c/*.py"]
CLASS_FILES = ["object2id.json", "id2object.json", "predicate2id.json", "id2predicate.json",
               "openvoc_obj_class_spilt_info.json", "openvoc_pred_class_spilt_info.json"]
CHECK0_REF = "studies/rpt/g0/sam3_out/ILSVRC2015_train_00010024.json.gz"
ID = re.compile(r"ILSVRC2015_(?:train|val)_\d{8}")
TEST = re.compile(r"/anno/test/|test_relation_gt|test_object_trajectories|ECC_test|gt_tracks_test|cache_s3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_files(specs):
    out = subprocess.run(["git", "ls-files", "--", *specs], capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l]


def main(stage):
    os.chdir(REPO)
    split = json.load(open("studies/d7/held_split.json"))
    held_b = set(split["held_b"])
    videos = [v for v in sorted(split["held_a"]) if os.path.exists(f"studies/rpt/g1/sam3_out/{v}.json.gz")]
    assert len(videos) == 78
    code = git_files(CODE_DIRS) + git_files(CODE_GLOBS)
    code = [p for p in code if not p.startswith("studies/d7/naming_check/")]          # outputs are made there
    code += git_files([f"studies/rpt/g1/sam3_out/{v}.json.gz" for v in videos]) + git_files([CHECK0_REF])
    code = sorted(set(code))
    assert len([p for p in code if p.startswith("studies/rpt/g1/sam3_out/")]) == 78
    data = [f"data/vidvrd/data/{f}" for f in CLASS_FILES]
    data += [f"data/vidvrd/anno/train/{v}.json" for v in videos]
    for v in videos:
        d = f"data/vidvrd/frames/{v}"
        data += [f"{d}/{f}" for f in sorted(os.listdir(d))]
    bad = [p for p in code + data if set(ID.findall(p)) & held_b or TEST.search("/" + p)]
    if bad:
        sys.exit(f"!! STOP: the upload would include HELD-B or test files: {bad[:5]}")
    os.makedirs(stage, exist_ok=True)
    for name, lst in (("code_files.txt", code), ("data_files.txt", data)):
        with open(os.path.join(stage, name), "w") as f:
            f.write("\n".join(lst) + "\n")
    for name, lst in (("expected_code.sha256", code), ("expected_data.sha256", data)):
        with open(os.path.join(stage, name), "w") as f:
            for p in lst:
                f.write(f"{sha256(p)}  {p}\n")
    for f in ("pip_freeze_main.txt", "pip_freeze_sam3env.txt"):
        shutil.copyfile(f"studies/d7/check/expected/{f}", os.path.join(stage, f))
    shutil.copyfile("studies/rpt/g1/box/expected_fetch.sha256", os.path.join(stage, "expected_fetch.sha256"))
    size = sum(os.path.getsize(p) for p in data) / 2**30
    print(f"code files {len(code)}, data files {len(data)} ({size:.2f} GiB), videos {len(videos)}; "
          f"no HELD-B and no test path")


if __name__ == "__main__":
    main(sys.argv[1])
