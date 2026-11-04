#!/usr/bin/env python3
"""A31 §5(a): FIT-only and HELD-A-only extracts of every data aggregate that holds HELD-B.

    python studies/d7/make_extracts.py        # from the repo root, BEFORE the HELD-B guard exists

Writes studies/d7/extracts/<source path with '/' -> '__'>.<fit|held_a>.<ext>, SHA256SUMS and
manifest.json. Every extract is checked to contain no HELD-B video ID (both ID forms); a FIT extract
also contains no HELD ID at all. The test-label guard is installed first.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob
import hashlib
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "studies", "rpt"))
import no_test_labels  # noqa: E402
no_test_labels.install()

OUT = "studies/d7/extracts"
SPLIT_SHA = "02415b25cfcbd398337c76331a5e4b54306bd8ec9c7579220503a0a87078f18a"
ID = re.compile(r"ILSVRC2015_(?:train|val)_\d{8}")

KEYED = [   # ground truth keyed by video at the root
    "data/vidvrd/data/train_object_trajectories_gt.json",
    "pilot_analysis/stage1c/gt_tracks_train.json",
    "pilot_analysis/stage1c/gt_tracks_train_optionA.json",
    "pilot_analysis/stage2/data/held_gt.json",
    "pilot_analysis/stage2/data/held_pairs.json",
    "studies/rpt/g1/held_gt_missing3.json",
    "studies/rpt/g0/held_gt_smoke3.json",
    "pilot_analysis/stage2/data_small/held_gt.json",
    "pilot_analysis/stage2/data_small/held_pairs.json",
    "pilot_analysis/stage2/data/held_gt_small5.json",
    "pilot_analysis/boundary/data/held_gt_small.json",
    "pilot_analysis/f1/data_small/held_gt_small.json",
]
RESULTS = {  # results files: only the per-video entries of HELD-A are kept; pooled numbers mix the halves
    "studies/rpt/g1/results_futility/futility.json": ("files",),
    "studies/rpt/g1/results_descriptive/descriptive.json": ("files", "viz_sample.drawn"),
    "studies/postg1/results_screen/sam3_screen.json": ("files", "inspected_20"),
}
JSONL = ["studies/rpt/g1/logs/box/sam3_manifest.jsonl"] + sorted(glob.glob("studies/rpt/g1/eov_out/manifest_g*.jsonl"))
IMAGES = "studies/rpt/g1/viz_unmatched"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_in(text: str) -> set:
    return set(ID.findall(text))


def name(src: str, subset: str) -> str:
    base, ext = os.path.splitext(src)
    return os.path.join(OUT, base.replace("/", "__") + f".{subset}{ext}")


def keep_entry(entry, keep: set, held_b: set) -> bool:
    found = ids_in(json.dumps(entry))
    return bool(found & keep) and not found & held_b


def results_extract(d: dict, parts, keep: set, held_b: set) -> dict:
    out = {"_note": "A31 §5(a): per-video entries of HELD-A only; pooled numbers mix both halves and are left out"}
    for part in parts:
        node = d
        for k in part.split("."):
            node = node[k]
        if isinstance(node, dict) and all(ID.fullmatch(k) for k in node):
            out[part] = {k: v for k, v in node.items() if k in keep}
        elif isinstance(node, dict):
            out[part] = {k: v for k, v in node.items() if keep_entry(v, keep, held_b)}
        else:
            out[part] = [e for e in node if keep_entry(e, keep, held_b)]
    return out


def main():
    os.chdir(REPO)
    if sha256("studies/d7/held_split.json") != SPLIT_SHA:
        raise SystemExit("!! STOP: held_split.json differs from its pin")
    split = json.load(open("studies/d7/held_split.json"))
    held_a, held_b = set(split["held_a"]), set(split["held_b"])
    tsplit = json.load(open("pilot_analysis/stage1/train_split.json"))
    fit, held = set(tsplit["fit"]), set(tsplit["held"])
    assert held == held_a | held_b and not fit & held
    os.makedirs(OUT, exist_ok=True)
    rows = []

    def record(src, subset, path, n, kind):
        text = open(path, "rb").read().decode("utf-8", errors="replace") if not path.endswith(".jpg") else os.path.basename(path)
        found = ids_in(text)
        if found & held_b or (subset == "fit" and found & held):
            raise SystemExit(f"!! STOP: {path} contains a HELD-B (or, for FIT, a HELD) video ID")
        rows.append(dict(source=src, source_sha256=sha256(src) if os.path.isfile(src) else None, subset=subset,
                         extract=path, extract_sha256=sha256(path), videos=n, kind=kind))

    for src in KEYED:
        d = json.load(open(src))
        assert isinstance(d, dict) and all(ID.fullmatch(k) for k in d), src
        for subset, keep in (("fit", fit), ("held_a", held_a)):
            sub = {k: v for k, v in d.items() if k in keep}
            if not sub:
                continue
            path = name(src, subset)
            with open(path, "w") as f:
                json.dump(sub, f)
            record(src, subset, path, len(sub), "ground truth keyed by video")
    for src, parts in RESULTS.items():
        sub = results_extract(json.load(open(src)), parts, held_a, held_b)
        path = name(src, "held_a")
        with open(path, "w") as f:
            json.dump(sub, f, indent=1)
        record(src, "held_a", path, len(sub["files"]), "results: per-video entries only")
    for src in JSONL:
        lines = [l for l in open(src) if l.strip()]
        kept = [l for l in lines if (lambda s: s & held_a and not s & held_b)(ids_in(l))]
        path = name(src, "held_a")
        with open(path, "w") as f:
            f.writelines(kept)
        record(src, "held_a", path, len(kept), "manifest lines of HELD-A videos")
    img_dir = os.path.join(OUT, "viz_unmatched.held_a")
    os.makedirs(img_dir, exist_ok=True)
    for f in sorted(os.listdir(IMAGES)):
        vids = ids_in(f)
        if vids and vids <= held_a:
            dst = os.path.join(img_dir, f)
            shutil.copyfile(os.path.join(IMAGES, f), dst)
            record(os.path.join(IMAGES, f), "held_a", dst, 1, "image of a HELD-A video")

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(dict(label="A31 §5(a): extracts", held_split_sha256=SPLIT_SHA, extracts=rows), f, indent=1)
        f.write("\n")
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        for r in rows:
            f.write(f"{r['extract_sha256']}  {os.path.relpath(r['extract'], OUT)}\n")
    for r in rows:
        print(f"{r['subset']:7s} {r['videos']:4d}  {r['extract']}")
    print(f"{len(rows)} extracts")


if __name__ == "__main__":
    main()
