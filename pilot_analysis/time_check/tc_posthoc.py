#!/usr/bin/env python3
"""Time check D1 -- POST-HOC diagnostic of the C1 bucket (not pre-registered; it changes no
bucket and no reading; it only describes C1).

    python pilot_analysis/time_check/tc_posthoc.py

Step 2 showed that in C1 ("not reachable at K = 40") the GT predicate is in the top 20 in ~91%
of the GT slots. A20's reachability builds the run over the pair's WHOLE timeline, so a
predicate that stays in the top 20 beyond the GT can form a run much longer than the GT and
fail vIoU 0.5. For each missed novel C1 GT this reports, at K = 20: whether any run overlaps
the GT; the best run's length / the GT's length; and whether the same run, clipped to the GT's
extent (an oracle trim using the GT boundaries), would reach min vIoU >= 0.5. Repo viou only.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import collections, json, os

import numpy as np

import tc_run as T


def main():
    from inference.video_relation_detection_openvoc import viou
    T.C.verify_frozen()
    phi, split = T.C.load_phi()
    names = json.load(open(os.path.join(T.C.ROOT, "data", "vidvrd", "data", "id2predicate.json")))
    missed = json.load(open(os.path.join(T.OUT, "tc_missed.json")))
    gt_all = json.load(open(T.C.TEST_GT))
    cols = [j for j, p in enumerate(names) if split[p] == "novel"]
    res = {}
    for task in ("sgdet", "predcls"):
        rows = [r for r in missed["novel"][task]["rows"] if r["bucket"] == "C1"]
        vids = sorted({r["video"] for r in rows})
        cache = T.load_cache(task, set(vids))
        pc = T.load_cache("predcls", set(vids)) if task == "sgdet" else cache
        stats = collections.Counter()
        ratios = []
        for r in rows:
            v = r["video"]
            cands = [x for x in gt_all[v] if x["triplet"] == [r["subject"], r["predicate"], r["object"]] and list(x["duration"]) == r["duration"]]
            g = cands[0]
            # the same pair as in tc_run (duplicate GT with the same triplet and extent: the one whose GT pair exists)
            if task == "predcls":
                ok = [(x, p) for x in cands for p in cache[v]["pairs"] if (p["sub_gt"], p["obj_gt"]) == (x["subject_tid"], x["object_tid"])]
                if not ok:
                    stats["unresolved (duplicate GT)"] += 1
                    continue
                g, pair = ok[0]
            else:
                tracks = cache[v]["tracks"]
                gb, ge = g["duration"]

                def best(key):
                    sc = {}
                    for tid, t in tracks.items():
                        lo, hi = max(gb, t["begin_fid"]), min(ge, t["end_fid"])
                        sc[tid] = 0.0 if hi <= lo else viou(t["boxes"][lo - t["begin_fid"]:hi - t["begin_fid"]], [lo, hi], g[key], [gb, ge])
                    return sc
                vs, vo = best("sub_traj"), best("obj_traj")
                _, _, _, bs, bo = max((min(vs[s], vo[o]), -s, -o, s, o) for s in tracks for o in tracks if s != o)
                idx = collections.defaultdict(list)
                for tid, t in tracks.items():
                    for i, b in enumerate(t["boxes"]):
                        idx[(t["category"], t["begin_fid"] + i)].append((tid, b))
                hit = lambda cat, f, box: [tid for tid, b in idx.get((cat, f), []) if max(abs(x - y) for x, y in zip(b, box)) < 1e-6]
                pair = next(p for p in cache[v]["pairs"]
                            if hit(p["sbj_cls"], p["duration"][0], p["sbj_traj"][0]) == [bs] and hit(p["obj_cls"], p["duration"][0], p["obj_traj"][0]) == [bo])
            S = T.E4.slot_scores(pair["logits"], pair["int_logit"], "all", None).numpy()[:, cols]
            j = cols.index(names.index(r["predicate"]))
            ranks = [int(1 + (S[t] > S[t, j]).sum()) for t in range(len(S))]
            sl = T.pair_slots(len(pair["sbj_traj"]))
            b0 = pair["duration"][0]
            gb, ge = g["duration"]
            best_run, bo_ = None, 0
            for a, e in T.runs([k <= 20 for k in ranks]):
                o = T.overlap(b0 + sl[a][0], b0 + sl[e - 1][1], gb, ge)
                if o > bo_:
                    best_run, bo_ = (a, e), o
            if best_run is None:
                stats["no run overlaps the GT (not seen)"] += 1
                continue
            f0, f1 = b0 + sl[best_run[0]][0], b0 + sl[best_run[1] - 1][1]
            ratios.append((f1 - f0) / (ge - gb))
            lo, hi = max(f0, gb), min(f1, ge)
            s = viou(pair["sbj_traj"][lo - b0:hi - b0], [lo, hi], g["sub_traj"], [gb, ge])
            o = viou(pair["obj_traj"][lo - b0:hi - b0], [lo, hi], g["obj_traj"], [gb, ge])
            if min(s, o) >= T.THR:
                stats["a run overlaps; trimmed to the GT extent it would match (over-long run)"] += 1
            else:
                stats["a run overlaps; even trimmed it does not match (boxes / coverage)"] += 1
        R = np.array(ratios) if ratios else np.zeros(1)
        res[task] = dict(c1=len(rows), **dict(stats), run_over_gt_length=dict(median=float(np.median(R)), p25=float(np.percentile(R, 25)),
                                                                             p75=float(np.percentile(R, 75)), share_over_2x=float((R > 2).mean())))
        print(task, json.dumps(res[task]), flush=True)
    path = os.path.join(T.OUT, "tc_posthoc_c1.json")
    json.dump(res, open(path, "w"), indent=1)
    T.write_log("tc_posthoc_c1", dict(k=20, note="post-hoc, descriptive only"), [T.C.TEST_GT, os.path.join(T.OUT, "tc_missed.json")], [path])


if __name__ == "__main__":
    main()
