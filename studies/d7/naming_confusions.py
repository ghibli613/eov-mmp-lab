#!/usr/bin/env python3
"""After A33's naming check, descriptive (asked by Giang, decides nothing): which classes modelB picks for
the novel GT objects of HELD-A, from the copied-back per-video outputs. Three views per novel class:
- per GT box (each annotated frame of each GT run): modelB's top-1;
- per GT run: the argmax of the run's mean vector (as tracks are named);
- per SAM 3 track matched class-agnostically to a run of that class (A29): modelB's name, and the prompt's.
Also the L40's time split (TagCLIP encoding against modelB). The HELD-B guard is installed first.

    python studies/d7/naming_confusions.py      # -> studies/d7/naming_check/confusions.json
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import collections
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "studies", "rpt"), os.path.join(REPO, "studies", "rpt", "g1"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
import held_b_guard  # noqa: E402
held_b_guard.install()


def main():
    import adapter as A
    import metrics as M
    from utils import paths
    os.chdir(REPO)
    sp = json.load(open(paths.OBJ_SPLIT_INFO))
    id2cls = sp["id2cls"]
    novel = {c for c, s in sp["cls2split"].items() if s == "novel"}
    argmax = lambda vec: id2cls[str(max(range(35), key=lambda i: vec[i]))]  # noqa: E731  (numpy.argmax: first max)
    rows = {json.loads(l)["video"]: json.loads(l) for l in open("studies/d7/naming_check/manifest.jsonl")}
    opt_b = json.load(open("studies/d7/extracts/pilot_analysis__stage1c__gt_tracks_train.held_a.json"))
    box, run, trk, trk_prompt = (collections.defaultdict(collections.Counter) for _ in range(4))
    runs_n, vids = collections.Counter(), collections.defaultdict(set)
    enc = mb = 0.0
    for v in sorted(rows):
        nc = json.load(gzip.open(f"studies/d7/naming_check/videos/{v}.json.gz", "rt"))
        enc += nc["seconds_encode"]
        mb += nc["seconds_modelB"]
        for g in nc["gt_runs"]:
            if g["category"] in novel:
                runs_n[g["category"]] += 1
                vids[g["category"]].add(v)
                box[g["category"]].update(id2cls[str(x)] for x in g["per_box_top1"])
                if g["mean_vector"] is not None:
                    run[g["category"]][argmax(g["mean_vector"])] += 1
        out = A.tracks_from_sam3(json.load(gzip.open(f"studies/rpt/g1/sam3_out/{v}.json.gz", "rt")))
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        runs = M.runs_of(opt_b[v], anno)
        by_rid = {r["rid"]: r for r in runs}
        names = [argmax(t["mean_vector"]) for t in nc["tracks"]]
        for rid, (tid, _) in M.match(M.tracks_from_list(out["repo"]), runs, class_aware=False).items():
            c = by_rid[rid]["category"]
            if c in novel:
                trk[c][names[tid]] += 1
                trk_prompt[c][out["repo"][tid]["category"]] += 1
    per = {c: dict(gt_runs=runs_n[c], videos=len(vids[c]), boxes=sum(box[c].values()),
                   box_top1_correct=box[c][c], box_top=box[c].most_common(5), run_names=run[c].most_common(5),
                   matched_tracks=sum(trk[c].values()), matched_tracks_modelB=trk[c].most_common(5),
                   matched_tracks_prompt=trk_prompt[c].most_common(5))
           for c in sorted(runs_n, key=lambda c: -sum(box[c].values()))}
    tot = collections.Counter()
    for c in box:
        tot.update({(c, p): n for p, n in box[c].items() if p != c})
    res = dict(label="after A33, descriptive: modelB's choices for HELD-A's novel GT objects (decides nothing)",
               novel_classes_present=sorted(runs_n), novel_classes_absent=sorted(novel - set(runs_n)),
               per_class=per, top_box_confusions=[[c, p, n] for (c, p), n in tot.most_common(12)],
               l40_seconds=dict(tagclip_encode=round(enc, 1), modelB=round(mb, 1),
                                frames_encoded=sum(r["frames"] for r in rows.values()),
                                boxes=sum(r["boxes"] for r in rows.values())))
    with open("studies/d7/naming_check/confusions.json", "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps(dict(top_box_confusions=res["top_box_confusions"], l40_seconds=res["l40_seconds"],
                          absent=res["novel_classes_absent"]), indent=1))


if __name__ == "__main__":
    main()
