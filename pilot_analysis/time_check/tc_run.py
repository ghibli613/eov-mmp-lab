#!/usr/bin/env python3
"""Time check D1 (A20) -- at which stage is each missed novel relation lost?

    python pilot_analysis/time_check/tc_run.py --reproduce          # step 0: evaluator on the existing outputs only
    python pilot_analysis/time_check/tc_run.py --analyse [--small]  # steps 1-3 (opens the test relation GT)

TEST-LABEL ERROR ANALYSIS, DISCLOSED; NO METHOD IS EVALUATED OR CHOSEN. The evaluator runs only
on EOV-MMP's existing baseline outputs. Single source: session 3's caches (PredCls: cache_s3/test,
GT tracks; SGDet: cache_s3_sgdet/test, detected tracks), whose re-derived final outputs are
identical to Task 4's and Task 6's dumps. Repo functions only: eval_from_cache.predict /
slot_scores (post-processing), viou and eval_detection_scores_v2 (matcher),
eval_relation_detection_openvoc (mAP).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, glob, json, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("stage2", "stage1c", "f1"):
    sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", sub))
import common as C                 # noqa: E402
import eval_from_cache as E4       # noqa: E402
from f1_prior import decompose     # noqa: E402

CLIP, HALF, THR, KS, N_BOOT = 30, 15, 0.5, (20, 30, 40), 1000
REF = dict(predcls=dict(novel=21.70, all=37.20), sgdet=dict(novel=15.79, all=27.18))
SOURCES = dict(predcls="cache_s3/test", sgdet="cache_s3_sgdet/test")
DUMPS = dict(predcls="cache_s3/task4/predcls_{}.json", sgdet="cache_s3_sgdet/sgdet_{}.json")
PILOT = os.path.join(C.PA, "preds_from_pod", "full", "final_merged_{}.json")
NOVEL_OBJ = {"antelope", "bear", "cattle", "fox", "giant_panda", "horse", "lizard", "motorcycle", "red_panda", "watercraft"}
OUT = os.path.join(HERE, "data")
SUMS = None


def write_log(name, config, inputs, outputs, extra=None):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rel = lambda p: os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
    rec = dict(step=name, time=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(sys.argv), config=config,
               env=C.env_info(), code={rel(os.path.join(HERE, f)): C.sha256(os.path.join(HERE, f))
                                       for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
               inputs={rel(p): C.sha256(p) for p in inputs}, outputs={rel(p): C.sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    json.dump(rec, open(os.path.join(HERE, "logs", f"{name}.json"), "w"), indent=1, default=str)


def checked(rel):
    global SUMS
    if SUMS is None:
        SUMS = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    f = os.path.join(C.BOX_COPY, rel)
    assert C.sha256(f) == SUMS[rel], rel
    return f


def load_cache(task, videos=None):
    import torch
    out = {}
    for f in sorted(glob.glob(os.path.join(C.BOX_COPY, SOURCES[task], "*.pt"))):
        v = os.path.basename(f)[:-3]
        if videos is not None and v not in videos:
            continue
        d = torch.load(checked(f"{SOURCES[task]}/{v}.pt"), weights_only=False, map_location="cpu")
        out[v] = dict(pairs=d["pairs"], tracks=d["tracks"])
    return out


def official(preds, split):
    from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
    m, rec = eval_relation_detection_openvoc(target_split_pred=split, prediction_results=preds)
    return dict(mAP=100 * float(m), R50=100 * float(rec[50]), R100=100 * float(rec[100]))


# ---------------------------------------------------------------------------
def reproduce():
    """Step 0: the repo's evaluator on the existing baseline outputs, and the identity of the
    cache-derived outputs with them. Reads no test label itself."""
    res, ok = {}, True
    for task in ("predcls", "sgdet"):
        cache = load_cache(task)
        for split in ("novel", "all"):
            dump = json.load(open(checked(DUMPS[task].format(split))))
            r = official(dump, split)
            r["ref"], r["diff"] = REF[task][split], r["mAP"] - REF[task][split]
            regen = json.loads(json.dumps(E4.predict({v: c["pairs"] for v, c in cache.items()}, split)))
            r["cache_outputs_identical_to_dump"] = regen == dump
            ok &= abs(r["diff"]) <= 0.05 and r["cache_outputs_identical_to_dump"]
            res.setdefault(task, {})[split] = r
            print(f"  {task} {split}: mAP {r['mAP']:.3f} (ref {r['ref']}) identical to cache outputs: "
                  f"{r['cache_outputs_identical_to_dump']}", flush=True)
    for split in ("novel", "all"):
        res.setdefault("pilot_dump", {})[split] = official(json.load(open(PILOT.format(split))), split)
    res["ok"] = bool(ok)
    return res


# ---------------------------------------------------------------------------
def pair_slots(n):
    k = n // CLIP + (1 if n % CLIP >= 10 else 0)
    return [(i * CLIP, n if i == k - 1 else (i + 1) * CLIP) for i in range(k)]


def runs(q, max_gap=1):
    """qualifying slot flags -> [(first, last+1)] joining runs separated by <= max_gap slots."""
    idx = [i for i, x in enumerate(q) if x]
    out = []
    for i in idx:
        if out and i - out[-1][1] <= max_gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(r) for r in out]


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def gt_slot_idx(pair, g):
    b0 = pair["duration"][0]
    return [k for k, (s0, s1) in enumerate(pair_slots(len(pair["sbj_traj"])))
            if overlap(b0 + s0, b0 + s1, g["duration"][0], g["duration"][1]) >= HALF]


def reachable(pair, ranks, K, g):
    """A20: slots with rank <= K, runs joined across gaps <= 1 slot, the run overlapping the GT
    most, trajectory = the pair's boxes over it; reachable iff min(sub, obj) vIoU >= 0.5."""
    from inference.video_relation_detection_openvoc import viou
    sl = pair_slots(len(pair["sbj_traj"]))
    b0 = pair["duration"][0]
    best, bo = None, 0
    for a, e in runs([r <= K for r in ranks]):
        o = overlap(b0 + sl[a][0], b0 + sl[e - 1][1], *g["duration"])
        if o > bo:
            best, bo = (a, e), o
    if best is None:
        return False
    f0, f1 = sl[best[0]][0], sl[best[1] - 1][1]
    dur = [b0 + f0, b0 + f1]
    s = viou(pair["sbj_traj"][f0:f1], dur, g["sub_traj"], g["duration"])
    o = viou(pair["obj_traj"][f0:f1], dur, g["obj_traj"], g["duration"])
    return min(s, o) >= THR


def label_confusion(gp, top):
    (v1, s1), (v2, s2) = decompose(gp), decompose(top)
    if s1 == s2 and s1 != "none" and ((v1 == "none") != (v2 == "none")):
        return "bare term vs compound"
    if v1 == v2 and v1 != "none" and s1 != s2:
        return "same verb, different position"
    if s1 == s2 and s1 != "none" and v1 != v2:
        return "same position, different verb"
    return "other"


def boot_share(mask, vids, seed=0):
    mask, vids = np.asarray(mask, float), np.asarray(vids)
    uv = np.unique(vids)
    if not len(uv):
        return (None, None)
    by = {u: np.where(vids == u)[0] for u in uv}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[u] for u in rng.choice(uv, len(uv))])
        bs.append(mask[idx].mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------------------
def analyse(small):
    from inference.video_relation_detection_openvoc import eval_detection_scores_v2, viou
    phi, split = C.load_phi()
    names = json.load(open(os.path.join(C.ROOT, "data", "vidvrd", "data", "id2predicate.json")))
    gt_all = json.load(open(C.TEST_GT))                     # test relation labels: opened here, after A20 and step 0
    vids = sorted(gt_all)[:5 if small else None]
    caches = {t: load_cache(t, set(vids)) for t in SOURCES}
    obj_split = json.load(open(C.ROOT + "/data/vidvrd/data/openvoc_obj_class_spilt_info.json"))["cls2split"]
    tc = {c for c in obj_split if c != "__background__"}
    out = {}
    for pass_name in ("novel", "all"):
        pc_ = {p for p, s in split.items() if pass_name == "all" or s == pass_name}
        cols = [j for j, p in enumerate(names) if p in pc_]
        keep = lambda r: r["triplet"][0] in tc and r["triplet"][1] in pc_ and r["triplet"][2] in tc
        for task in ("sgdet", "predcls"):
            cache = caches[task]
            preds = E4.predict({v: cache[v]["pairs"] for v in vids}, pass_name)
            old = E4.POST.max_per_video
            E4.POST.max_per_video = 10 ** 9
            uncut = E4.predict({v: cache[v]["pairs"] for v in vids}, pass_name)
            E4.POST.max_per_video = old
            rows, ap = [], []
            for v in vids:
                gts = [g for g in gt_all[v] if keep(g)]
                if not gts:
                    continue
                pr = [r for r in preds.get(v, []) if keep(r)]
                prec, rec, _, g2d = eval_detection_scores_v2(gts, pr, THR)
                from inference.video_relation_detection_openvoc import voc_ap
                ap.append(voc_ap(rec, prec))
                g2d_uncut = eval_detection_scores_v2(gts, [r for r in uncut.get(v, []) if keep(r)], THR)[3]
                pairs, tracks = cache[v]["pairs"], cache[v]["tracks"]
                gt_pairs = {(p["sub_gt"], p["obj_gt"]): k for k, p in enumerate(caches["predcls"][v]["pairs"])}
                if task == "sgdet":
                    idx = collections.defaultdict(list)
                    for tid, t in tracks.items():
                        for i, b in enumerate(t["boxes"]):
                            idx[(t["category"], t["begin_fid"] + i)].append((tid, b))
                    hit = lambda cat, f, box: [tid for tid, b in idx.get((cat, f), []) if max(abs(x - y) for x, y in zip(b, box)) < 1e-6]
                    pair_of = {}
                    for k, p in enumerate(pairs):
                        s_, o_ = hit(p["sbj_cls"], p["duration"][0], p["sbj_traj"][0]), hit(p["obj_cls"], p["duration"][0], p["obj_traj"][0])
                        assert len(s_) == 1 and len(o_) == 1
                        pair_of[(s_[0], o_[0])] = k
                for gi, g in enumerate(gts):
                    if g2d[gi] >= 0:
                        continue
                    gb, ge = g["duration"]
                    gp = gt_pairs.get((g["subject_tid"], g["object_tid"]))
                    n_gt = len(gt_slot_idx(caches["predcls"][v]["pairs"][gp], g)) if gp is not None else \
                        (ge - gb) // CLIP + (1 if (ge - gb) % CLIP >= HALF else 0)
                    row = dict(video=v, predicate=g["triplet"][1], subject=g["triplet"][0], object=g["triplet"][2],
                               duration=[gb, ge], n_gt_slots=n_gt, long=n_gt > 1)
                    bucket, pair = None, None
                    if task == "sgdet":
                        def best_viou(traj_key):
                            sc = {}
                            for tid, t in tracks.items():
                                lo, hi = max(gb, t["begin_fid"]), min(ge, t["end_fid"])
                                sc[tid] = 0.0 if hi <= lo else viou(t["boxes"][lo - t["begin_fid"]:hi - t["begin_fid"]], [lo, hi], g[traj_key], [gb, ge])
                            return sc
                        vs, vo = best_viou("sub_traj"), best_viou("obj_traj")
                        cand = [(min(vs[s], vo[o]), -s, -o, s, o) for s in tracks for o in tracks if s != o]
                        if cand:
                            m_, _, _, bs, bo_ = max(cand)
                        else:
                            m_, bs, bo_ = 0.0, None, None
                        row.update(best_pair_min_viou=m_)
                        if m_ < THR:
                            bucket = "U1"
                        else:
                            row.update(tracker_subject=tracks[bs]["category"], tracker_object=tracks[bo_]["category"])
                            if tracks[bs]["category"] != g["triplet"][0] or tracks[bo_]["category"] != g["triplet"][2]:
                                bucket = "U2"
                            else:
                                k = pair_of.get((bs, bo_))
                                if k is None or len(gt_slot_idx(pairs[k], g)) < n_gt / 2:
                                    bucket = "U3"
                                else:
                                    pair = pairs[k]
                    else:
                        if gp is None:
                            bucket = "U3"
                        else:
                            pair = pairs[gp]
                    if pair is not None:
                        S = E4.slot_scores(pair["logits"], pair["int_logit"], "all", None).numpy()[:, cols]
                        j = cols.index(names.index(g["triplet"][1]))
                        ranks = [int(1 + (S[t] > S[t, j]).sum()) for t in range(len(S))]
                        reach = {K: reachable(pair, ranks, K, g) for K in KS}
                        gsl = gt_slot_idx(pair, g)
                        row.update(reach={str(K): bool(x) for K, x in reach.items()}, gt_slot_ranks=[ranks[t] for t in gsl],
                                   gt_slot_top1=[names[cols[int(np.argmax(S[t]))]] for t in gsl])
                        if reach[20]:
                            bucket = "A1"
                            row["a1"] = "ii: removed by the top-200 cut" if g2d_uncut[gi] >= 0 else "i: no matching instance from association"
                        elif reach[30] or reach[40]:
                            bucket = "N1"
                        else:
                            bucket = "C1"
                    row["bucket"] = bucket
                    rows.append(row)
            n_total = sum(1 for v in vids for g in gt_all[v] if keep(g))
            out.setdefault(pass_name, {})[task] = dict(gt_total=n_total, missed=len(rows), mAP_from_matcher=100 * float(np.mean(ap)),
                                                       rows=rows)
            print(f"  {pass_name} {task}: GT {n_total}, missed {len(rows)}, mAP via matcher {100 * np.mean(ap):.3f} | "
                  f"{collections.Counter(r['bucket'] for r in rows)}", flush=True)
    return out


def tables(an):
    res = {}
    for pass_name, d in an.items():
        for task, x in d.items():
            rows = x["rows"]
            buckets = ["U1", "U2", "U3", "A1", "N1", "C1"] if task == "sgdet" else ["U3", "A1", "N1", "C1"]
            t = {}
            for grp, sel in (("pooled", rows), ("long", [r for r in rows if r["long"]]), ("short", [r for r in rows if not r["long"]])):
                vids = [r["video"] for r in sel]
                t[grp] = dict(n=len(sel), **{b: dict(count=sum(r["bucket"] == b for r in sel),
                                                    share=(sum(r["bucket"] == b for r in sel) / len(sel)) if sel else None,
                                                    ci95=boot_share([r["bucket"] == b for r in sel], vids)) for b in buckets})
            res.setdefault(pass_name, {})[task] = dict(gt_total=x["gt_total"], missed=x["missed"], mAP_from_matcher=x["mAP_from_matcher"], buckets=t)
            if pass_name != "novel":
                continue
            # step 2
            longs = [r for r in rows if r["long"] and r["bucket"] in ("A1", "N1", "C1")]
            s2 = {}
            for b in ("A1", "N1", "C1"):
                rk = [k for r in longs if r["bucket"] == b for k in r["gt_slot_ranks"]]
                s2[b] = dict(gt=sum(r["bucket"] == b for r in longs), gt_slots=len(rk),
                             share_top20=(sum(k <= 20 for k in rk) / len(rk)) if rk else None,
                             median_rank_outside_top20=float(np.median([k for k in rk if k > 20])) if any(k > 20 for k in rk) else None)
            conf = collections.Counter()
            for r in rows:
                if r["bucket"] in ("N1", "C1"):
                    for top in r["gt_slot_top1"]:
                        if top != r["predicate"]:
                            conf[(r["predicate"], top)] += 1
            s2["confusions_N1_C1"] = [dict(gt=a, top1=b, slots=n, kind=label_confusion(a, b)) for (a, b), n in conf.most_common(15)]
            s2["confusion_kinds_all"] = dict(collections.Counter(label_confusion(a, b) for (a, b), n in conf.items() for _ in range(n)))
            if task == "sgdet":
                u2 = collections.Counter()
                for r in rows:
                    if r["bucket"] == "U2":
                        if r["tracker_subject"] != r["subject"]:
                            u2[(r["subject"], r["tracker_subject"])] += 1
                        if r["tracker_object"] != r["object"]:
                            u2[(r["object"], r["tracker_object"])] += 1
                s2["U2_class_pairs"] = [dict(gt=a, tracker=b, n=n, novel_object=a in NOVEL_OBJ or b in NOVEL_OBJ) for (a, b), n in u2.most_common(10)]
                s2["U2_involving_novel_objects"] = sum(n for (a, b), n in u2.items() if a in NOVEL_OBJ or b in NOVEL_OBJ)
                s2["U2_mismatches"] = sum(u2.values())
            s2["A1_split"] = dict(collections.Counter(r.get("a1") for r in rows if r["bucket"] == "A1"))
            res[pass_name][task]["step2"] = s2
    p = res["novel"]["sgdet"]["buckets"]["pooled"]
    rows = an["novel"]["sgdet"]["rows"]
    vids = [r["video"] for r in rows]
    groups = dict(upstream=["U1", "U2", "U3"], A1=["A1"], N1=["N1"], C1=["C1"])
    g = {k: dict(count=sum(p[b]["count"] for b in bs), share=sum(p[b]["count"] for b in bs) / max(p["n"], 1),
                 ci95=boot_share([r["bucket"] in bs for r in rows], vids)) for k, bs in groups.items()}
    order = sorted(g, key=lambda k: -g[k]["count"])
    res["reading"] = dict(groups=g, largest=order[0], second=order[1])
    return res


def main():
    ap = argparse.ArgumentParser()
    gr = ap.add_mutually_exclusive_group(required=True)
    gr.add_argument("--reproduce", action="store_true")
    gr.add_argument("--analyse", action="store_true")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    C.verify_frozen()
    os.makedirs(OUT, exist_ok=True)
    if a.reproduce:
        res = reproduce()
        path = os.path.join(OUT, "tc_reproduction.json")
        json.dump(res, open(path, "w"), indent=1)
        write_log("tc_reproduce", dict(ref=REF), [C.CACHE_SUMS, PILOT.format("novel"), PILOT.format("all")], [path])
        if not res["ok"]:
            sys.exit("!! reproduction failed -- STOP")
        print("reproduction OK")
        return
    rep = json.load(open(os.path.join(OUT, "tc_reproduction.json")))
    assert rep["ok"], "step 0 has not passed"
    an = analyse(a.small)
    tb = tables(an)
    sfx = "_small" if a.small else ""
    paths = [os.path.join(OUT, f"tc_missed{sfx}.json"), os.path.join(OUT, f"tc_tables{sfx}.json")]
    json.dump(an, open(paths[0], "w"))
    json.dump(tb, open(paths[1], "w"), indent=1, default=str)
    write_log("tc_analyse" + sfx, dict(small=a.small, thr=THR, ks=KS, half=HALF, boot=N_BOOT), [C.TEST_GT, C.CACHE_SUMS], paths,
              extra=dict(seconds=round(time.time() - t0, 1)))
    print(json.dumps(tb["reading"], indent=1))


if __name__ == "__main__":
    main()
