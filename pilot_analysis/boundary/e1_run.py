#!/usr/bin/env python3
"""Boundary fix E1 (A21) -- nested runs: each predicted relation is also offered at shorter
lengths, cut where its predicate drops out of a tighter per-slot top-k.

    python pilot_analysis/boundary/e1_run.py --step 1 [--small]   # HELD plumbing (base labels, descriptive)
    python pilot_analysis/boundary/e1_run.py --step 2 [--small]   # label-free checks on cached test inputs
    python pilot_analysis/boundary/e1_run.py --step 3             # THE test evaluation (once)

FOURTH TEST EVALUATION, DISCLOSED. The method was motivated by the D1 test-label diagnostic
(over-long runs in C1) and fixed in A21 before any test number.

Method (c): per-slot scores after the type rules (T1, A17: logit -> -inf); for k in
{20, 10, 5, 3, 1} the repo's process_pred (per-slot top-k) and association, with no top-200
cut; pool the five sets in that order; deduplicate (subject track, object track, predicate,
identical extent) keeping the highest score; the repo's instance score unchanged; the repo's
format_ (top 200 per video). k = 20 alone is T1 and must reproduce it exactly. Repo functions
only: nothing in the post-processing or the evaluator is reimplemented.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, glob, json, os, time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("stage2", "stage1c", "span", "time_check"):
    sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", sub))
import common as C            # noqa: E402
import eval_from_cache as E4  # noqa: E402
import span_common as SC      # noqa: E402
import tc_run as T            # noqa: E402  (pair_slots, runs, overlap, load_cache: D1's own functions)

KS = (20, 10, 5, 3, 1)
REF_A = dict(predcls=dict(novel=21.70, all=37.20), sgdet=dict(novel=15.79, all=27.18))
METOR = dict(novel=16.75, all=27.53)
D1_EXPECTED = dict(sgdet=129, predcls=237)
TOL, N_BOOT, BOOT_SEED, CTRL_SEEDS, BIG = 0.05, 1000, 0, tuple(range(20)), 10 ** 9
SOURCES = dict(predcls="cache_s3/test", sgdet="cache_s3_sgdet/test")
RULES_FILE = os.path.join(C.PA, "f2", "data", "f2_part3.json")
T1_RESULTS = os.path.join(C.PA, "final", "data", "t1_results.json")
HELD_GT = os.path.join(SC.DATA2, "held_gt.json")                 # stage 2 step 1: HELD, base labels only
TC_MISSED = os.path.join(C.PA, "time_check", "data", "tc_missed.json")
KEEP = ("sbj_cls", "obj_cls", "sbj_scr", "obj_scr", "sbj_traj", "obj_traj", "duration", "logits", "int_logit", "sub_gt", "obj_gt")
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


def load_pairs(sub, videos=None):
    """{video: [pair dicts]} from session-3 cache files, each checked against SHA256SUMS."""
    import torch
    global SUMS
    if SUMS is None:
        SUMS = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    out = {}
    for f in sorted(glob.glob(os.path.join(C.BOX_COPY, sub, "*.pt"))):
        v = os.path.basename(f)[:-3]
        if videos is not None and v not in videos:
            continue
        assert C.sha256(f) == SUMS[f"{sub}/{v}.pt"], f
        out[v] = [{k: p[k] for k in KEEP if k in p} for p in torch.load(f, weights_only=False, map_location="cpu")["pairs"]]
    return out


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------
class Pipeline:
    """The repo's post-processing, called per pair so every instance keeps its pair."""

    def __init__(self):
        phi, split = C.load_phi()
        self.id2pre, self.obj2id, self.base_ids = E4.meta()
        self.names = self.id2pre
        self.novel_cols = [j for j, p in enumerate(self.names) if split[p] == "novel"]
        rules = json.load(open(RULES_FILE))["rules"]["rules"]
        assert rules == {"stand": "animate", "run": "animate", "walk": "animate", "sit": "animate", "stop": "vehicle",
                         "lie": "animate", "creep": "animate", "swim": "animate"}
        verb = lambda p: (C.S0.tokenise(p)[0] if C.compositional(p) and len(C.S0.tokenise(p)) == 2 else None)
        self.ruled = {j: rules[verb(p)] for j, p in enumerate(self.names) if verb(p) in rules}

    def deleted_cols(self, cat):                       # T1 (A17), verbatim
        t = SC.subject_type(cat)
        return [j for j, typ in self.ruled.items() if t != typ]

    def logits(self, p, pass_, rules):
        L = p["logits"].clone().float()
        if rules:
            cols = self.deleted_cols(p["sbj_cls"])
            if cols:
                L[:, cols] = float("-inf")
        if pass_ == "base":                            # HELD plumbing: novel predicates out
            L[:, self.novel_cols] = float("-inf")
        return L

    @staticmethod
    def mode(pass_):
        return "novel" if pass_ == "novel" else "all"

    def instances(self, cache, pass_, rules, k):
        """{video: [(pair index, instance)]}: process_pred with per-slot top-k, association, no cut."""
        from inference.post_process import association, process_pred
        a = SimpleNamespace(clip_len=30, clip_top_n=k, use_prior=False, max_per_video=BIG)
        out = {}
        for v, pairs in cache.items():
            lst = []
            for pi, p in enumerate(pairs):
                pd = dict(sbj_cls=p["sbj_cls"], obj_cls=p["obj_cls"], sbj_scr=p["sbj_scr"], obj_scr=p["obj_scr"],
                          sbj_traj=list(p["sbj_traj"]), obj_traj=list(p["obj_traj"]), duration=list(p["duration"]))
                s = E4.slot_scores(self.logits(p, pass_, rules), p["int_logit"], self.mode(pass_), self.base_ids)
                lst.extend((pi, inst) for inst in association(process_pred(a, self.id2pre, self.obj2id, None, s, pd)))
            out[v] = lst
        return out

    def t1(self, cache, pass_):
        """T1's own path (eval_from_cache.predict with T1's logits): the identity reference for (b)."""
        return E4.predict(cache, self.mode(pass_), logits_of=lambda v, k, p: self.logits(p, pass_, True))


def key(pi, inst):
    return (pi, inst["pre_cls"], tuple(inst["duration"]))


def pool(sets):
    """sets: list of {video: [(pi, inst)]}, in k order -> one list per video, deduplicated by key,
    keeping the highest score (the earlier one on a tie)."""
    out = {}
    for v in sets[0]:
        seen, order = {}, []
        for s in sets:
            for pi, inst in s[v]:
                kk = key(pi, inst)
                if kk not in seen:
                    seen[kk] = (pi, inst)
                    order.append(kk)
                elif inst["score"] > seen[kk][1]["score"]:
                    seen[kk] = (pi, inst)
        out[v] = [seen[kk] for kk in order]
    return out


def cut(pooled):
    """the repo's format_: top 200 per video."""
    from inference.post_process import format_
    return {v: format_(E4.POST, [inst for _, inst in lst]) for v, lst in pooled.items()}


def extras(pooled, set20):
    """{video: {id(sbj_traj)}} of pooled instances that are not k = 20 instances. format_ keeps
    the instance's own trajectory list, so the ids identify extras in the formatted output."""
    out = {}
    for v, lst in pooled.items():
        b = {key(pi, inst) for pi, inst in set20[v]}
        out[v] = {id(inst["sbj_traj"]) for pi, inst in lst if key(pi, inst) not in b}
    return out


def slot_range(pair, inst):
    """instance -> (first slot, last slot + 1, slot table) on its pair."""
    sl = T.pair_slots(len(pair["sbj_traj"]))
    assert len(sl) == len(pair["logits"])
    d0 = pair["duration"][0]
    i = next(t for t, (s0, s1) in enumerate(sl) if d0 + s0 == inst["duration"][0])
    j = next(t for t, (s0, s1) in enumerate(sl) if d0 + s1 == inst["duration"][1]) + 1
    return i, j, sl


def control(cache, set20, pooled, seed):
    """(f): every extra instance moved to a uniformly random slot-aligned position inside the k = 20
    run that contains it (same pair, same predicate), keeping its length in slots, predicate, pair
    and score; its trajectories are the pair's boxes at the new position. Then the same dedupe."""
    rng = np.random.default_rng(seed)
    out, unplaced = {}, 0
    for v in sorted(pooled):
        pairs = cache[v]
        bkeys = {key(pi, inst) for pi, inst in set20[v]}
        runs = collections.defaultdict(list)
        for pi, inst in set20[v]:
            i, j, _ = slot_range(pairs[pi], inst)
            runs[(pi, inst["pre_cls"])].append((i, j))
        new = list(set20[v])
        for pi, inst in pooled[v]:
            if key(pi, inst) in bkeys:
                continue
            p = pairs[pi]
            i, j, sl = slot_range(p, inst)
            cand = [(a, b) for a, b in runs[(pi, inst["pre_cls"])] if a <= i and j <= b]
            if not cand:
                unplaced += 1
                new.append((pi, inst))
                continue
            a, b = cand[0]
            s = int(rng.integers(a, b - (j - i) + 1))
            f0, f1 = sl[s][0], sl[s + (j - i) - 1][1]
            d0 = p["duration"][0]
            new.append((pi, dict(inst, duration=[d0 + f0, d0 + f1], sbj_traj=list(p["sbj_traj"][f0:f1]),
                                 obj_traj=list(p["obj_traj"][f0:f1]))))
        out[v] = new
    return pool([out]), unplaced


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def filt(split):
    """the official evaluator's category filter."""
    phi, sp = C.load_phi()
    oc = {c for c in json.load(open(os.path.join(C.ROOT, "data", "vidvrd", "data", "openvoc_obj_class_spilt_info.json")))["cls2split"]
          if c != "__background__"}
    pc = {p for p, s in sp.items() if split == "all" or s == split}
    return lambda r: r["triplet"][0] in oc and r["triplet"][1] in pc and r["triplet"][2] in oc


def evaluate(preds, split, gt_path, extra_ids=None):
    """official mAP / R@50 / R@100 (eval_relation_detection_openvoc); per video, the repo's matcher
    (eval_detection_scores_v2: the official one plus gt2det) for AP, matched GT and false positives.
    The per-video mean must equal the official mAP."""
    from inference.video_relation_detection_openvoc import eval_detection_scores_v2, eval_relation_detection_openvoc, voc_ap
    m, rec = eval_relation_detection_openvoc(target_split_pred=split, prediction_results=preds, gt_json=gt_path)
    keep = filt(split)
    gt = json.load(open(gt_path))
    ap, mset, matched, fp, ex = {}, {}, 0, 0, collections.Counter()
    for v, g in gt.items():
        g = [x for x in g if keep(x)]
        if not g:
            continue
        pr = sorted([dict(x) for x in preds.get(v, []) if keep(x)], key=lambda x: x["score"], reverse=True)
        prec, rc, hs, g2d = eval_detection_scores_v2(g, pr, 0.5)
        ap[v] = float(voc_ap(rc, prec))
        mset[v] = [i for i, d in enumerate(g2d) if d >= 0]
        matched += len(mset[v])
        fp += int((~np.isfinite(hs)).sum())
        if extra_ids is not None:
            for x, h in zip(pr, hs):
                if id(x["sub_traj"]) in extra_ids[v]:
                    ex["in_list"] += 1
                    ex["hit" if np.isfinite(h) else "false_positive"] += 1
    assert abs(np.mean(list(ap.values())) - float(m)) < 1e-9, (np.mean(list(ap.values())), m)
    r = dict(mAP=100 * float(m), R50=100 * float(rec[50]), R100=100 * float(rec[100]), gt_matched=matched, false_positives=fp,
             predictions=sum(1 for v in gt for x in preds.get(v, []) if keep(x)))
    if extra_ids is not None:
        r["extras_evaluated"] = dict(ex)
    return r, ap, mset


def boot(ap_c, ap_b=None, seed=BOOT_SEED):
    """video-level bootstrap 95% CI of the mean AP (or of the paired difference), in points."""
    vids = sorted(ap_c)
    x = np.array([ap_c[v] for v in vids]) - (np.array([ap_b[v] for v in vids]) if ap_b is not None else 0.0)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))]


def identical(a, b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def run_configs(PL, cache, pass_, gt_path, eval_split, full=True, seeds=CTRL_SEEDS):
    """(a)-(f) on one cache and pass, and the identity of (b) with T1's own path."""
    rule_settings = (True, False) if full else (True,)
    sets = {r: [PL.instances(cache, pass_, r, k) for k in KS] for r in rule_settings}
    pooled_c = pool(sets[True])
    ex_c = extras(pooled_c, sets[True][0])
    cfg = {}
    if full:
        cfg["a"] = cut(sets[False][0])
    cfg["b"] = cut(sets[True][0])
    cfg["c"] = cut(pooled_c)
    if full:
        cfg["d"] = cut(pool(sets[False]))
        for i, k in enumerate(KS[1:], 1):                  # (e): k = 20 plus a single tighter k
            cfg[f"e_k{k}"] = cut(pool([sets[True][0], sets[True][i]]))
    ident = identical(cfg["b"], PL.t1(cache, pass_))
    res, aps, msets = {}, {}, {}
    for name, preds in cfg.items():
        res[name], aps[name], msets[name] = evaluate(preds, eval_split, gt_path, extra_ids=ex_c if name == "c" else None)
    ctrl, unplaced = [], 0
    for s in seeds:
        pc, u = control(cache, sets[True][0], pooled_c, s)
        unplaced += u
        ctrl.append(evaluate(cut(pc), eval_split, gt_path)[0])
    res["f_control"] = dict(seeds=list(seeds), unplaced=unplaced, per_seed_mAP=[r["mAP"] for r in ctrl],
                            sd_mAP=float(np.std([r["mAP"] for r in ctrl])),
                            **{x: float(np.mean([r[x] for r in ctrl])) for x in ("mAP", "R50", "R100", "gt_matched", "false_positives")})
    res["b_identical_to_T1_path"] = ident
    res["c_ci95"] = boot(aps["c"])
    res["c_minus_b"] = res["c"]["mAP"] - res["b"]["mAP"]
    res["c_minus_b_ci95"] = boot(aps["c"], aps["b"])
    mb, mc = msets["b"], msets["c"]
    res["c_vs_b"] = dict(newly_matched=sum(len(set(mc[v]) - set(mb[v])) for v in mc),
                         lost=sum(len(set(mb[v]) - set(mc[v])) for v in mb),
                         false_positives_added=res["c"]["false_positives"] - res["b"]["false_positives"],
                         extras_in_c=res["c"]["extras_evaluated"])
    res["videos_evaluated"] = len(aps["c"])
    return res, cfg, sets, pooled_c, msets


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def step1(a):
    """HELD plumbing: base pass on stage 2's base-only HELD GT (descriptive only). Exercises every
    step-3 code path except (a), (d), (e) and the D1 count, which differ only in their inputs."""
    gt = json.load(open(HELD_GT))
    held = sorted(gt)[:5 if a.small else None]
    gt_path = HELD_GT
    if a.small:
        gt_path = os.path.join(OUT, "held_gt_small.json")
        json.dump({v: gt[v] for v in held}, open(gt_path, "w"))
    phi, split = C.load_phi()
    assert all(split[x["triplet"][1]] == "base" for v in held for x in gt[v])
    PL = Pipeline()
    cache = load_pairs("cache_s3/train", set(held))
    assert sorted(cache) == held
    res, cfg, sets, pooled, _ = run_configs(PL, cache, "base", gt_path, "base", full=False,
                                            seeds=CTRL_SEEDS[:2] if a.small else CTRL_SEEDS)
    res["held_videos"], res["novel_labels_used"] = len(held), 0
    print(json.dumps({k: res[k] for k in ("b", "c", "f_control", "b_identical_to_T1_path", "c_ci95", "c_minus_b_ci95", "c_vs_b")},
                     indent=1, default=str), flush=True)
    return res, [gt_path] if a.small else []


def step2(a):
    """label-free checks on the cached test inputs: no relation label, no evaluator."""
    PL = Pipeline()
    res, ok_id, any_extra = {}, True, False
    for task, sub in SOURCES.items():
        cache = load_pairs(sub)
        if a.small:
            cache = {v: cache[v] for v in sorted(cache)[:5]}
        for pass_ in ("novel", "all"):
            sets = [PL.instances(cache, pass_, True, k) for k in KS]
            b = cut(sets[0])
            ident = identical(b, PL.t1(cache, pass_))
            ok_id &= ident
            pooled = pool(sets)
            c = cut(pooled)
            ex = extras(pooled, sets[0])
            origin = {}
            for k, s in zip(KS, sets):                     # the largest k whose set holds the instance
                for v, lst in s.items():
                    for pi, inst in lst:
                        origin.setdefault((v,) + key(pi, inst), k)
            length = {(v, id(inst["sbj_traj"])): (origin[(v,) + key(pi, inst)], len(inst["sbj_traj"]))
                      for v, lst in pooled.items() for pi, inst in lst}
            c_ids = {v: {id(x["sub_traj"]) for x in c[v]} for v in c}
            pushed = sum(1 for v in b for x in b[v] if id(x["sub_traj"]) not in c_ids[v])
            n_extra = [len(ex[v]) for v in sorted(pooled)]
            in_top = [sum(1 for x in c[v] if id(x["sub_traj"]) in ex[v]) for v in sorted(pooled)]
            any_extra |= sum(in_top) > 0
            pool_by_k, top_by_k = collections.defaultdict(list), collections.defaultdict(list)
            for (v, i), (k, n) in length.items():
                pool_by_k[k].append(n)
            for v in c:
                for x in c[v]:
                    k, n = length[(v, id(x["sub_traj"]))]
                    top_by_k[k].append(n)
            summ = lambda d: {str(k): dict(n=len(x), frames_median=float(np.median(x)), frames_mean=float(np.mean(x)),
                                           frames_p25=float(np.percentile(x, 25)), frames_p75=float(np.percentile(x, 75)))
                              for k, x in sorted(d.items(), reverse=True)}
            res.setdefault(task, {})[pass_] = dict(
                videos=len(pooled), identity_k20_equals_T1=ident,
                instances_k20=sum(len(x) for x in sets[0].values()), instances_pooled=sum(len(x) for x in pooled.values()),
                extras_per_video=dict(mean=float(np.mean(n_extra)), median=float(np.median(n_extra)), max=int(np.max(n_extra)),
                                      total=int(np.sum(n_extra))),
                b_outputs=sum(len(x) for x in b.values()), c_outputs=sum(len(x) for x in c.values()), b_outputs_pushed_out=pushed,
                extras_in_top200=dict(total=int(np.sum(in_top)), per_video_mean=float(np.mean(in_top)),
                                      videos_with_any=int(sum(x > 0 for x in in_top))),
                pooled_by_origin_k=summ(pool_by_k), top200_by_origin_k=summ(top_by_k))
            print(f"  {task} {pass_}: identity {ident}; extras/video {np.mean(n_extra):.1f}; extras in top-200 {int(np.sum(in_top))} "
                  f"of {sum(len(x) for x in c.values())}; b outputs pushed out {pushed}", flush=True)
    res["identity_ok"], res["any_extra_in_top200"] = ok_id, any_extra
    res["stop"] = (not ok_id) or (not any_extra)
    return res, []


def d1_overlong(task, gt_novel):
    """D1's over-long C1 cases (tc_posthoc.py's procedure, verbatim, on tc_missed.json's C1 rows)
    -> {video: [index into that video's filtered novel GT list]}."""
    from inference.video_relation_detection_openvoc import viou
    phi, split = C.load_phi()
    names = json.load(open(os.path.join(C.ROOT, "data", "vidvrd", "data", "id2predicate.json")))
    cols = [j for j, p in enumerate(names) if split[p] == "novel"]
    rows = [r for r in json.load(open(TC_MISSED))["novel"][task]["rows"] if r["bucket"] == "C1"]
    vids = sorted({r["video"] for r in rows})
    cache = T.load_cache(task, set(vids))
    out = collections.defaultdict(list)
    for r in rows:
        v = r["video"]
        cands = [x for x in gt_novel[v] if x["triplet"] == [r["subject"], r["predicate"], r["object"]] and list(x["duration"]) == r["duration"]]
        g = cands[0]
        if task == "predcls":
            ok = [(x, p) for x in cands for p in cache[v]["pairs"] if (p["sub_gt"], p["obj_gt"]) == (x["subject_tid"], x["object_tid"])]
            if not ok:
                continue
            g, pair = ok[0]
        else:
            tracks = cache[v]["tracks"]
            gb, ge = g["duration"]

            def best(k_):
                sc = {}
                for tid, t in tracks.items():
                    lo, hi = max(gb, t["begin_fid"]), min(ge, t["end_fid"])
                    sc[tid] = 0.0 if hi <= lo else viou(t["boxes"][lo - t["begin_fid"]:hi - t["begin_fid"]], [lo, hi], g[k_], [gb, ge])
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
        S = E4.slot_scores(pair["logits"], pair["int_logit"], "all", None).numpy()[:, cols]
        j = cols.index(names.index(r["predicate"]))
        ranks = [int(1 + (S[t] > S[t, j]).sum()) for t in range(len(S))]
        sl = T.pair_slots(len(pair["sbj_traj"]))
        b0 = pair["duration"][0]
        gb, ge = g["duration"]
        best_run, bo_ = None, 0
        for a_, e_ in T.runs([k <= 20 for k in ranks]):
            o = T.overlap(b0 + sl[a_][0], b0 + sl[e_ - 1][1], gb, ge)
            if o > bo_:
                best_run, bo_ = (a_, e_), o
        if best_run is None:
            continue
        f0, f1 = b0 + sl[best_run[0]][0], b0 + sl[best_run[1] - 1][1]
        lo, hi = max(f0, gb), min(f1, ge)
        s = viou(pair["sbj_traj"][lo - b0:hi - b0], [lo, hi], g["sub_traj"], [gb, ge])
        o = viou(pair["obj_traj"][lo - b0:hi - b0], [lo, hi], g["obj_traj"], [gb, ge])
        if min(s, o) >= T.THR:
            out[v].append(next(i for i, x in enumerate(gt_novel[v]) if x is g))
    return out


def step3(a):
    PL = Pipeline()
    t1 = json.load(open(T1_RESULTS))["rules_run"]
    keep = filt("novel")
    gt_novel = {v: [x for x in g if keep(x)] for v, g in json.load(open(C.TEST_GT)).items()}   # test labels: step 3 only
    res = {}
    for task, sub in SOURCES.items():
        cache = load_pairs(sub)
        for pass_ in ("novel", "all"):
            r, cfg, sets, pooled, msets = run_configs(PL, cache, pass_, C.TEST_GT, pass_)
            r["a"]["ref"], r["a"]["diff"] = REF_A[task][pass_], r["a"]["mAP"] - REF_A[task][pass_]
            r["b"]["ref_T1"], r["b"]["diff_T1"] = t1[task][pass_]["mAP"], r["b"]["mAP"] - t1[task][pass_]["mAP"]
            if pass_ == "novel":
                try:
                    ol = d1_overlong(task, gt_novel)
                    r["d1_overlong"] = dict(n=sum(len(x) for x in ol.values()), expected=D1_EXPECTED[task],
                                            **{f"matched_in_{c}": sum(len(set(ol[v]) & set(msets[c].get(v, []))) for v in ol)
                                               for c in ("a", "b", "c", "d")})
                except Exception as e:                    # descriptive only; never stops the run
                    r["d1_overlong"] = dict(error=repr(e))
                print(f"  {task} D1 over-long: {r['d1_overlong']}", flush=True)
            res.setdefault(task, {})[pass_] = r
            print(f"  {task} {pass_}: " + " | ".join(f"{k} {v['mAP']:.2f}" for k, v in r.items() if isinstance(v, dict) and "mAP" in v)
                  + f" | b == T1 path: {r['b_identical_to_T1_path']}", flush=True)
    ok = all(abs(res[t][p]["a"]["diff"]) <= TOL and abs(res[t][p]["b"]["diff_T1"]) < 1e-9 and res[t][p]["b_identical_to_T1_path"]
             for t in res for p in res[t])
    sg = res["sgdet"]
    c_n, c_a = sg["novel"]["c"]["mAP"], sg["all"]["c"]["mAP"]
    res["verdict"] = dict(reproduced=bool(ok), metor=METOR, c_sgdet=dict(novel=c_n, all=c_a),
                          margins=dict(novel=c_n - METOR["novel"], all=c_a - METOR["all"]),
                          metor_inside_c_ci=dict(novel=sg["novel"]["c_ci95"][0] <= METOR["novel"] <= sg["novel"]["c_ci95"][1],
                                                 all=sg["all"]["c_ci95"][0] <= METOR["all"] <= sg["all"]["c_ci95"][1]),
                          primary="PASS" if (c_n > METOR["novel"] and c_a > METOR["all"]) else "FAIL")
    return res, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    assert not (a.step == 3 and a.small), "step 3 runs once, in full"
    t0 = time.time()
    C.verify_frozen()
    os.makedirs(OUT, exist_ok=True)
    phi, split = C.load_phi()
    labels = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))["labels"]
    assert all(set(l) <= {p for p in phi if split[p] == "base"} for l in labels)    # 0 novel labels in training
    inputs = [C.CACHE_SUMS, RULES_FILE]
    if a.step == 1:
        inputs.append(HELD_GT)
    if a.step == 3:
        s2 = json.load(open(os.path.join(OUT, "e1_step2.json")))
        if s2["stop"]:
            sys.exit("step 2 stopped the batch (A21)")
        assert not os.path.exists(os.path.join(OUT, "e1_step3.json")), "step 3 has already run"
        inputs += [os.path.join(OUT, "e1_step2.json"), T1_RESULTS, TC_MISSED, C.TEST_GT]
    res, extra_paths = {1: step1, 2: step2, 3: step3}[a.step](a)
    res["seconds"] = round(time.time() - t0, 1)
    path = os.path.join(OUT, f"e1_step{a.step}{'_small' if a.small else ''}.json")
    json.dump(res, open(path, "w"), indent=1, default=str)
    write_log(f"e1_step{a.step}" + ("_small" if a.small else ""),
              dict(ks=KS, boot=N_BOOT, boot_seed=BOOT_SEED, control_seeds=CTRL_SEEDS, tol=TOL),
              inputs, [path] + extra_paths, extra=dict(seconds=res["seconds"]))
    if a.step == 2 and res["stop"]:
        sys.exit("!! STOP: the identity check failed or no extra instance enters any top-200 (A21)")
    if a.step == 3:
        print(json.dumps(res["verdict"], indent=1))
        if not res["verdict"]["reproduced"]:
            print("!! (a) or (b) did not reproduce")
    print(f"done | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
