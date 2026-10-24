#!/usr/bin/env python3
"""Laptop: the SHA256 the D2 machine must reproduce (A25c), in `sha256sum -c` format, with paths
relative to the machine's EOV work dir (~/ov-vidvrd).

    cd METOR/studies && python -m d2.box.make_expected

expected_upload.sha256  every file of `git archive fdeeeb4`, the metadata and test annotations
                        copied from the laptop (upload.sh)
expected_fetch.sha256   what fetch_with_token.sh pulls from Hugging Face (weights, CLIP bank,
                        ECC), checked against the laptop's copies, plus the frame bundle's
                        SHARDS.json / MANIFEST.json (saved under d2_box/frames_meta/)
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile

from d2.common import CODE_ROOT, EOV_COMMIT, THESIS, sha256

HERE = os.path.dirname(os.path.abspath(__file__))
META = "data/vidvrd/data"
FROM_HF_META = ("clip_L14_feat_vidvrd.pkl", "VidVRD_ECC_train.json", "VidVRD_ECC_test.json")
CKPTS = ("baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth", "AFLink_epoch20.pth")
FRAMES_META = os.path.join(THESIS, "_frames_bundle_test")


def upload_list() -> list[str]:
    """The metadata and annotation files upload.sh copies (relative to the EOV repo)."""
    meta = sorted(f for f in os.listdir(os.path.join(CODE_ROOT, META)) if f not in FROM_HF_META)
    anno = sorted(os.listdir(os.path.join(CODE_ROOT, "data/vidvrd/anno/test")))
    return [f"{META}/{f}" for f in meta] + [f"data/vidvrd/anno/test/{f}" for f in anno]


def main():
    tar = subprocess.check_output(["git", "-C", CODE_ROOT, "archive", "--format=tar", EOV_COMMIT])
    up = []
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                up.append((hashlib.sha256(tf.extractfile(m).read()).hexdigest(), m.name))
    up += [(sha256(os.path.join(CODE_ROOT, p)), p) for p in upload_list()]
    fetch = [(sha256(os.path.join(CODE_ROOT, META, f)), f"{META}/{f}") for f in FROM_HF_META]
    fetch += [(sha256(os.path.join(CODE_ROOT, "output/ckpt", c)), f"output/ckpt/{c}") for c in CKPTS]
    fetch += [(sha256(os.path.join(FRAMES_META, f)), f"../d2_box/frames_meta/{f}") for f in ("SHARDS.json", "MANIFEST.json")]
    with open(os.path.join(HERE, "upload_files.txt"), "w") as f:
        f.write("\n".join(upload_list()) + "\n")
    for name, rows in (("expected_upload.sha256", up), ("expected_fetch.sha256", fetch)):
        with open(os.path.join(HERE, name), "w") as f:
            f.writelines(f"{h}  {p}\n" for h, p in sorted(rows, key=lambda r: r[1]))
        print(f"{name}: {len(rows)} files")


if __name__ == "__main__":
    main()
