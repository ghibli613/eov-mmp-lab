#!/usr/bin/env python3
"""Object check O1 (A22), step 3 -- measurements M1-M5 and gates G-O1 / G-O2 (CPU, local).

    python pilot_analysis/objects/o1_measure.py --dump DIR [--tag small]

Training videos only. Definitions are A22's; see o1_common.py for GT runs, frame IoU and vIoU.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, glob, json, os, time

import numpy as np

import o1_common as O

G_O1, G_O2 = 0.30, 0.10


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def best_query(iou, prob_rows, rule):
    """Among localising queries (IoU >= 0.5): 'score' = highest max class probability (ties: higher
    IoU, then lower index) -- A22's best-matching query; 'iou' = highest IoU (ties: lower index)."""
    loc = np.where(iou >= O.LOC_IOU)[0]
    if not len(loc):
        return None
    if rule == "iou":
        return int(loc[np.lexsort((loc, -iou[loc]))[0]])
    pm = prob_rows[loc].max(1)
    return int(loc[np.lexsort((loc, -iou[loc], -pm))[0]])


def track_view(t):
    return dict(begin=t["begin"], end=t["end"], boxes=t["boxes"])


def names(t, rec, thr):
    """Rules (a)-(d) for one pipeline track (A22)."""
    det = rec["det"][thr]
    hits = [(f, i) for f, i in t["hits"] if i >= 0]
    out = dict(a=t["category"], n_hits=len(hits))
    if not hits:
        out.update(b=None, c=None, d=None)
        return out
    probs = np.stack([rec["prob"][f - rec["begin_fid"], int(det["qid"][i])].astype(np.float32) for f, i in hits])
    votes = np.bincount(probs.argmax(1), minlength=35)
    out["b"] = O.CATS[int(np.argmax(votes))]                     # ties: lowest class index
    out["b_tie"] = int((votes == votes.max()).sum() > 1)
    out["c"] = O.CATS[int(probs.mean(0).argmax())]
    ek = {(int(a), int(b)): j for j, (a, b) in enumerate(rec["emb_key"])}
    embs = np.stack([rec["emb"][ek[(f, int(det["qid"][i]))]].float().numpy() for f, i in hits])
    m = embs.mean(0)
    tf = rec["clip_query"].float().numpy()
    cos = (tf @ m) / (np.linalg.norm(tf, axis=1) * np.linalg.norm(m))
    out["d"] = O.CATS[int(cos.argmax())]
    return out


def mean_max_prob(t, rec, thr):
    det = rec["det"][thr]
    v = [float(rec["prob"][f - rec["begin_fid"], int(det["qid"][i])].astype(np.float32).max()) for f, i in t["hits"] if i >= 0]
    return float(np.mean(v)) if v else 0.0


def per_video(path):
    rec = load(path)
    v = rec["video"]
    runs = O.gt_runs(v)
    b0, b1 = rec["begin_fid"], rec["end_fid"]
    prob = rec["prob"].astype(np.float32)
    kept = rec["kept"][0.35]
    tr = rec["tracks"][0.35]
    det35 = {(int(f), int(q)) for f, q in zip(rec["det"][0.35]["frame"], rec["det"][0.35]["qid"])}
    rows = []
    for r in runs:
        vc = [O.viou_cut(track_view(t), r) for t in tr]
        vf = [O.viou_full(track_view(t), r) for t in tr]
        best = int(np.argmax(vc)) if vc else -1
        found = bool(vc) and vc[best] >= O.VIOU_THR
        named_any = any(x >= O.VIOU_THR and t["category"] == r["category"] for x, t in zip(vc, tr))
        ann = sorted(r["ann"])
        outside = sum(1 for f in ann if not (b0 <= f < b1))
        frames = []
        for f in ann:
            if not (b0 <= f < b1):
                continue
            k = f - b0
            iou = O.frame_iou(r["ann"][f], rec["boxes"][k])
            q = best_query(iou, prob[k], "score")
            if q is None:
                frames.append(None)
                continue
            qi = best_query(iou, prob[k], "iou")
            gi = O.CATS.index(r["category"])
            frames.append(dict(q=q, maxp=float(prob[k, q].max()), gtp=float(prob[k, q, gi]), kept=q in kept[k],
                               det=(f, q) in det35, argmax_gt=int(prob[k, q].argmax()) == gi,
                               qi_kept=qi in kept[k], qi_maxp=float(prob[k, qi].max())))
        loc = [x for x in frames if x is not None]
        row = dict(video=v, key=list(r["key"]), category=r["category"], novel=r["novel"], frames=len(ann), outside=outside,
                   found=found, named=bool(named_any), found_full=bool(vf) and max(vf) >= O.VIOU_THR,
                   best_viou=float(vc[best]) if vc else 0.0, localized=len(loc),
                   loc_share=len(loc) / len(frames) if frames else 0.0,
                   not_kept_share=float(np.mean([not x["kept"] for x in loc])) if loc else None,
                   not_kept_share_iou=float(np.mean([not x["qi_kept"] for x in loc])) if loc else None,
                   det_share=float(np.mean([x["det"] for x in loc])) if loc else None,
                   maxp=[x["maxp"] for x in loc], gtp=[x["gtp"] for x in loc],
                   argmax_gt_share=float(np.mean([x["argmax_gt"] for x in loc])) if loc else None)
        if found:
            row["names"] = names(tr[best], rec, 0.35)
        rows.append(row)
    m5 = {}
    for thr in O.THRESHOLDS[1:]:
        ts = rec["tracks"][thr]
        keep = [t for t in ts if mean_max_prob(t, rec, thr) >= 0.35]
        res = {}
        for label, sel in (("gated", keep), ("ungated", ts)):
            fr = [any(O.viou_cut(track_view(t), r) >= O.VIOU_THR for t in sel) for r in runs]
            extra = [t for t in sel if not any(O.viou_full(track_view(t), track_view(b)) >= O.VIOU_THR for b in tr)]
            nogt = [t for t in extra if not any(O.viou_cut(track_view(t), r) >= O.VIOU_THR for r in runs)]
            res[label] = dict(tracks=len(sel), found=[bool(x) for x in fr], extra_tracks=len(extra), extra_no_gt=len(nogt),
                              count_minus_base=len(sel) - len(tr))
        m5[str(thr)] = res
    base_nogt = sum(1 for t in tr if not any(O.viou_cut(track_view(t), r) >= O.VIOU_THR for r in runs))
    return dict(video=v, rows=rows, m5=m5, base_tracks=len(tr), base_tracks_no_gt=base_nogt,
                checks=rec.get("checks"), repo_check=rec.get("repo_check"), seconds=rec.get("seconds"))


def q(x):
    x = np.asarray(x, dtype=float)
    return dict(n=int(len(x)), median=float(np.median(x)), p25=float(np.percentile(x, 25)), p75=float(np.percentile(x, 75))) if len(x) else dict(n=0)


def summarise(V):
    rows = [r for v in V for r in v["rows"]]
    out = {}
    # M1
    m1 = {}
    for c in O.CATS:
        s = [r for r in rows if r["category"] == c]
        if s:
            m1[c] = dict(novel=c in O.NOVEL, runs=len(s), found=float(np.mean([r["found"] for r in s])),
                         found_named=float(np.mean([r["named"] for r in s])), found_uncut=float(np.mean([r["found_full"] for r in s])))
    for nv, lab in ((True, "NOVEL"), (False, "BASE")):
        s = [r for r in rows if r["novel"] == nv]
        g = [r["video"] for r in s]
        m1[lab] = dict(runs=len(s), found=float(np.mean([r["found"] for r in s])), found_ci=O.boot_ci([r["found"] for r in s], g),
                       found_named=float(np.mean([r["named"] for r in s])), found_named_ci=O.boot_ci([r["named"] for r in s], g),
                       found_uncut=float(np.mean([r["found_full"] for r in s])))
    out["M1"] = m1
    # M2
    def group(r, key="not_kept_share"):
        if r["frames"] == 0 or r["loc_share"] < 0.5:
            return "iii"
        return "i" if r[key] > O.MAJ else "ii"
    m2 = {}
    for nv, lab in ((True, "novel"), (False, "base")):
        s = [r for r in rows if r["novel"] == nv and not r["found"]]
        g = collections.Counter(group(r) for r in s)
        gi = collections.Counter(group(r, "not_kept_share_iou") for r in s)
        m2[lab] = dict(unfound=len(s), **{k: g[k] for k in ("i", "ii", "iii")},
                       share={k: g[k] / len(s) if s else None for k in ("i", "ii", "iii")},
                       share_i_ci=O.boot_ci([group(r) == "i" for r in s], [r["video"] for r in s]),
                       sensitivity_highest_iou_query={k: gi[k] for k in ("i", "ii", "iii")},
                       ii_best_query_became_detection_median=float(np.median([r["det_share"] for r in s if group(r) == "ii"]))
                       if g["ii"] else None)
        m2[lab]["per_category"] = {c: dict(collections.Counter(group(r) for r in s if r["category"] == c))
                                   for c in sorted({r["category"] for r in s})}
    out["M2"] = m2
    # M3
    m3 = {}
    for nv, lab in ((True, "novel"), (False, "base")):
        s = [r for r in rows if r["novel"] == nv]
        m3[lab] = dict(all_runs=dict(max_class=q([x for r in s for x in r["maxp"]]), gt_class=q([x for r in s for x in r["gtp"]]),
                                     share_argmax_is_gt=float(np.mean([r["argmax_gt_share"] for r in s if r["argmax_gt_share"] is not None]))),
                       unfound_runs=dict(max_class=q([x for r in s if not r["found"] for x in r["maxp"]]),
                                         gt_class=q([x for r in s if not r["found"] for x in r["gtp"]])))
    out["M3"] = m3
    # M4
    m4 = {}
    for nv, lab in ((True, "novel"), (False, "base")):
        s = [r for r in rows if r["novel"] == nv and r["found"]]
        acc = {k: float(np.mean([r["names"][k] == r["category"] for r in s])) if s else None for k in "abcd"}
        conf = collections.Counter((r["category"], r["names"]["a"]) for r in s if r["names"]["a"] != r["category"])
        m4[lab] = dict(found_runs=len(s), accuracy=acc, no_hits=sum(r["names"]["n_hits"] == 0 for r in s),
                       b_ties=sum(r["names"].get("b_tie", 0) for r in s),
                       c_minus_a=(acc["c"] - acc["a"]) if s else None,
                       c_minus_a_ci=O.boot_ci([(r["names"]["c"] == r["category"]) - (r["names"]["a"] == r["category"]) for r in s],
                                              [r["video"] for r in s]),
                       confusions_a=[[a, b, n] for (a, b), n in conf.most_common(10)],
                       per_category={c: {k: float(np.mean([r["names"][k] == c for r in s if r["category"] == c])) for k in "abcd"}
                                     for c in sorted({r["category"] for r in s})})
    out["M4"] = m4
    # M5
    m5 = {}
    nov = np.array([r["novel"] for r in rows])
    for thr in O.THRESHOLDS[1:]:
        m5[thr] = {}
        for label in ("gated", "ungated"):
            fr = np.array([x for v in V for x in v["m5"][str(thr)][label]["found"]])
            ex = [v["m5"][str(thr)][label]["extra_tracks"] for v in V]
            ng = sum(v["m5"][str(thr)][label]["extra_no_gt"] for v in V)
            m5[thr][label] = dict(novel_found=float(fr[nov].mean()), base_found=float(fr[~nov].mean()),
                                  tracks_per_video=float(np.mean([v["m5"][str(thr)][label]["tracks"] for v in V])),
                                  count_minus_base_per_video=float(np.mean([v["m5"][str(thr)][label]["count_minus_base"] for v in V])),
                                  extra_tracks_per_video=float(np.mean(ex)), extra_tracks=int(sum(ex)),
                                  extra_share_no_gt=ng / sum(ex) if sum(ex) else None)
    m5["0.35 (repo)"] = dict(novel_found=out["M1"]["NOVEL"]["found"], base_found=out["M1"]["BASE"]["found"],
                             tracks_per_video=float(np.mean([v["base_tracks"] for v in V])),
                             share_tracks_no_gt=sum(v["base_tracks_no_gt"] for v in V) / max(1, sum(v["base_tracks"] for v in V)))
    out["M5"] = m5
    # gates
    s1 = out["M2"]["novel"]["share"]["i"]
    s2 = out["M4"]["novel"]["c_minus_a"]
    p1, p2 = s1 is not None and s1 >= G_O1, s2 is not None and s2 >= G_O2
    reading = {(True, True): "both pass: objects <-> time becomes the core of the new model",
               (True, False): "G-O1 only: the rescue half carries forward",
               (False, True): "G-O2 only: the naming half carries forward",
               (False, False): "neither passes: relations <-> time becomes the core instead"}[(p1, p2)]
    out["gates"] = dict(G_O1=dict(share_i=s1, threshold=G_O1, ci=out["M2"]["novel"]["share_i_ci"], passed=p1),
                        G_O2=dict(c_minus_a=s2, threshold=G_O2, ci=out["M4"]["novel"]["c_minus_a_ci"], passed=p2),
                        reading=reading)
    out["videos"] = len(V)
    out["annotated_frames_outside_processed_range"] = sum(r["outside"] for r in rows)
    out["repo_identity"] = [v["repo_check"] for v in V if v["repo_check"] is not None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    O.C.verify_frozen()
    os.makedirs(O.OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dump, "*.pt")))
    man = {json.loads(l)["video"]: json.loads(l)["sha256"] for l in open(os.path.join(a.dump, "manifest.jsonl"))}
    for f in files:
        v = os.path.basename(f)[:-3]
        assert O.sha256(f) == man[v], f
    V = []
    for i, f in enumerate(files, 1):
        V.append(per_video(f))
        if i % 50 == 0:
            print(f"  {i}/{len(files)} videos, {time.time() - t0:.0f}s", flush=True)
    res = summarise(V)
    sfx = f"_{a.tag}" if a.tag else ""
    path = os.path.join(O.OUT, f"o1_step3{sfx}.json")
    json.dump(res, open(path, "w"), indent=1, default=str)
    rows_path = os.path.join(O.OUT, f"o1_rows{sfx}.json")
    json.dump([dict(r, maxp=None, gtp=None) for v in V for r in v["rows"]], open(rows_path, "w"), default=str)
    O.write_log(f"o1_step3{sfx}", dict(G_O1=G_O1, G_O2=G_O2), [os.path.join(a.dump, "manifest.jsonl"), O.GT_RUNS],
                [path, rows_path], extra=dict(seconds=round(time.time() - t0, 1)))
    print(json.dumps(dict(M1_novel=res["M1"]["NOVEL"], M1_base=res["M1"]["BASE"], M2=res["M2"]["novel"]["share"],
                          M4=res["M4"]["novel"]["accuracy"], gates=res["gates"]), indent=1, default=str))


if __name__ == "__main__":
    main()
