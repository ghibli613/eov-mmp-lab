#!/usr/bin/env python3
"""Stage 2 Batch A2 -- pairs and ground truth for the 63 FIT early-stop videos (A7's
alpha choice), in the same form as step 1's HELD files.

    python pilot_analysis/stage2/a2_es_data.py [--limit 5]

Ground truth: the videos' BASE relations from the training annotations (novel
dropped at parse), built by step1_parts.gt_relation_insts (= get_relation_insts).
Pairs: from the cache, each with its row range in data/slots.npz.
Writes data/es_pairs.json and data/es_gt.json (with --limit: *_small.json).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os

import numpy as np

import common as C
from step1_parts import gt_relation_insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    C.verify_frozen()
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    data = os.path.join(C.OUT, "data")
    d = np.load(os.path.join(data, "slots.npz"))
    videos = json.load(open(os.path.join(data, "slots_meta.json")))["videos"]
    es = C.splits()["es"][:a.limit or None]
    C.verify_cache(es)
    pairs, gt, nov = {}, {}, 0
    for v in es:
        vi = videos.index(v)
        rows = np.where(d["video"] == vi)[0]
        assert (d["subset"][rows] == 1).all()
        _, _, _, rels, n_nov, ann = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{v}.json"), base)
        nov += n_nov
        gt[v] = gt_relation_insts(ann, rels)
        cache = torch.load(os.path.join(C.CACHE, "train", f"{v}.pt"), weights_only=False)
        pairs[v] = []
        for pi, cp in enumerate(cache["pairs"]):
            r = rows[d["pair"][rows] == pi]
            assert len(r) == len(cp["logits"]) and (np.diff(r) == 1).all() and (d["slot"][r] == np.arange(len(r))).all()
            assert np.allclose(d["logits"][r], cp["logits"].numpy())
            pairs[v].append(dict(sbj_cls=cp["sbj_cls"], obj_cls=cp["obj_cls"], sbj_scr=cp["sbj_scr"], obj_scr=cp["obj_scr"],
                                 duration=list(cp["duration"]), sbj_traj=[list(map(float, b)) for b in cp["sbj_traj"]],
                                 obj_traj=[list(map(float, b)) for b in cp["obj_traj"]], rows=[int(r[0]), int(r[-1]) + 1]))
    sfx = "_small" if a.limit else ""
    paths = [os.path.join(data, f"es_pairs{sfx}.json"), os.path.join(data, f"es_gt{sfx}.json")]
    json.dump(pairs, open(paths[0], "w"))
    json.dump(gt, open(paths[1], "w"))
    assert all(r["triplet"][1] in base for g in gt.values() for r in g)
    C.write_log("a2_es_data" + sfx, dict(limit=a.limit), [os.path.join(data, "slots.npz"), C.CACHE_SUMS], paths,
                extra=dict(novel_dropped_at_parse=nov, novel_labels_used=0))
    print(f"{len(es)} early-stop videos, {sum(len(p) for p in pairs.values())} pairs, "
          f"{sum(len(g) for g in gt.values())} base GT relations; novel dropped at parse {nov}, used 0")


if __name__ == "__main__":
    main()
