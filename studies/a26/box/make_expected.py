#!/usr/bin/env python3
"""Laptop, at upload time: what the A26 machine must reproduce, in `sha256sum -c` format.

    cd METOR/studies && python -m a26.box.make_expected OUT_DIR [METOR_COMMIT]

OUT_DIR/upload_files.txt        the data files upload.sh copies (relative to the EOV repo): the class
                                splits, id maps, prior, train trajectories, and the test annotations
                                the pipeline reads (test_relation_gt.json,
                                test_object_trajectories_gt.json, anno/test/*.json)
OUT_DIR/expected_eov.sha256     `git archive fdeeeb4` of the EOV repo, plus those data files
                                (checked in ~/ov-vidvrd)
OUT_DIR/expected_metor.sha256   `git archive METOR_COMMIT` of METOR without studies/d2/data/rec
                                (checked in ~/METOR)
OUT_DIR/expected_fetch.sha256   what fetch_with_token.sh pulls (weights, CLIP bank, ECC, the frame
                                bundle's SHARDS.json / MANIFEST.json), against the laptop's copies
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from a26.common import AFLINK, CKPT, EOV_COMMIT, EOV_ROOT, METOR_ROOT, THESIS  # noqa: E402
from a26.run import sha256  # noqa: E402


META = "data/vidvrd/data"
META_FILES = ("id2object.json", "id2predicate.json", "object2id.json", "openvoc_obj_class_spilt_info.json",
              "openvoc_pred_class_spilt_info.json", "predicate2id.json", "prior.pkl",
              "train_object_trajectories_gt.json", "test_relation_gt.json", "test_object_trajectories_gt.json")
FETCHED = (f"{META}/clip_L14_feat_vidvrd.pkl", f"{META}/VidVRD_ECC_train.json", f"{META}/VidVRD_ECC_test.json",
           CKPT, AFLINK)
FRAMES_META = os.path.join(THESIS, "_frames_bundle_test")
EXCLUDE_METOR = ":(exclude)studies/d2/data/rec"


def archive_hashes(repo: str, commit: str, *pathspec: str) -> list[tuple[str, str]]:
    tar = subprocess.check_output(["git", "-C", repo, "archive", "--format=tar", commit, *pathspec])
    out = []
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                out.append((hashlib.sha256(tf.extractfile(m).read()).hexdigest(), m.name))
    return out


def upload_list() -> list[str]:
    anno = sorted(os.listdir(os.path.join(EOV_ROOT, "data/vidvrd/anno/test")))
    return [f"{META}/{f}" for f in META_FILES] + [f"data/vidvrd/anno/test/{f}" for f in anno]


def main():
    out = os.path.abspath(sys.argv[1])
    commit = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    os.makedirs(out, exist_ok=True)
    up = upload_list()
    with open(os.path.join(out, "upload_files.txt"), "w") as f:
        f.write("\n".join(up) + "\n")
    eov = archive_hashes(EOV_ROOT, EOV_COMMIT) + [(sha256(os.path.join(EOV_ROOT, p)), p) for p in up]
    metor = archive_hashes(METOR_ROOT, commit, ".", EXCLUDE_METOR)
    fetch = [(sha256(os.path.join(FRAMES_META, f)), f"../a26_box/frames_meta/{f}") for f in ("MANIFEST.json", "SHARDS.json")]
    fetch += [(sha256(os.path.join(EOV_ROOT, p)), p) for p in FETCHED]
    for name, rows in (("expected_eov.sha256", eov), ("expected_metor.sha256", metor), ("expected_fetch.sha256", fetch)):
        with open(os.path.join(out, name), "w") as f:
            f.writelines(f"{h}  {p}\n" for h, p in rows)
    n_test = sum(p.startswith("data/vidvrd/anno/test/") or "/test_" in p for p in up)
    print(f"{out}: {len(up)} data files ({n_test} test annotations), {len(eov)} EOV files, "
          f"{len(metor)} METOR files ({commit}), {len(fetch)} fetched files")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
