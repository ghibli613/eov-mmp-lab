#!/usr/bin/env python3
"""Feasibility batch F2 (A14) -- three cheap checks for a combined final method.

    python pilot_analysis/f2/f2_run.py --part 1 [--small]     # relation-aware object classes
    python pilot_analysis/f2/f2_run.py --part 2 [--small]     # temporal flicker
    python pilot_analysis/f2/f2_run.py --part 3 [--small]     # type rules (needs part 1's fused classes)

Training videos: base labels only. Test videos: MODEL OUTPUTS AND INPUTS ONLY --
session 3's cache_s3/test (PredCls inputs: GT tracks and categories, logits,
embeddings) and cache_s3_sgdet/test (detected tracks, logits, embeddings). No test
annotation is read and no test evaluation is run; these checks only decide which
components enter the final run.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, json, math, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "span"))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "f1"))
import span_common as SC                 # noqa: E402  (slot table)
from span_common import C                # noqa: E402  (stage 2: paths, frozen check, phi, tokeniser)
sys.path.insert(0, C.HERE)
sys.path.insert(0, os.path.join(C.PA, "stage1c"))
import evaluation as EV                  # noqa: E402
from f1_prior import object_embeddings   # noqa: E402  (once.pt object_text, checked against SHA256SUMS)

TEMP = 0.01                               # rc:442
G1_TOP1, G1_DISAGREE, G2_PERSIST, G3_FALSE, G3_REACH = 0.70, 0.10, 0.75, 0.005, 0.03
RULE_SHARE, RULE_MIN = 0.995, 100
OUT = os.path.join(HERE, "data")


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


SUMS = None


def load_cache(rel):
    """torch.load a session-3 cache file after checking it against SHA256SUMS_session3.txt."""
    import torch
    global SUMS
    if SUMS is None:
        SUMS = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    f = os.path.join(C.BOX_COPY, rel)
    assert C.sha256(f) == SUMS[rel], rel
    return torch.load(f, weights_only=False, map_location="cpu")     # slot_inputs were saved as CUDA tensors (A15)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def head_probs(emb, E):
    """modelC's subject/object class head (rc:496-509): normalised embedding . object-class
    text / 0.01, softmax over the 35 categories ('all' split)."""
    e = np.asarray(emb, np.float64).reshape(-1, E.shape[1])
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    return softmax(e @ E.T.astype(np.float64) / TEMP)


def eov(logits, int_logit):
    return 1 / (1 + np.exp(-np.asarray(logits, np.float64))) * (1 / (1 + np.exp(-np.asarray(int_logit, np.float64))))[:, None]


def subject_type(cat):
    return SC.subject_type(cat)


def track_index(tracks):
    idx = collections.defaultdict(list)
    for tid, t in tracks.items():
        for i, b in enumerate(t["boxes"]):
            idx[(t["category"], t["begin_fid"] + i)].append((tid, b))
    return idx


def match(idx, cat, frame, box):
    return [tid for tid, b in idx.get((cat, frame), []) if max(abs(x - y) for x, y in zip(b, box)) < 1e-6]


def fuse(track_cat, track_score, head_p, cats):
    """A14's fusion. The cache keeps only the tracker's top-1 class and its mean score, so the
    tracker's distribution is approximated as q(top-1) = score and (1 - score) / 34 elsewhere
    (floored at 1e-6); fused = normalised geometric mean of q and the head's per-track p."""
    q = np.full(len(cats), max((1 - track_score) / (len(cats) - 1), 1e-6))
    q[cats.index(track_cat)] = max(track_score, 1e-6)
    f = np.sqrt(q * np.maximum(head_p, 1e-12))
    f /= f.sum()
    k = int(np.argmax(f))
    return cats[k], float(f[k])


# ---------------------------------------------------------------------------
def part1(a, E, cats):
    import torch
    res = {}
    # ---- 1-0: reproduce the head against a direct CPU forward of modelC on training videos
    torch.Tensor.cuda = lambda self, *x, **k: self
    torch.nn.Module.cuda = lambda self, *x, **k: self
    torch.set_num_threads(6)
    from cache_wrapper import CKPT, build_modelC
    m, _, load = build_modelC(os.path.join(C.ROOT, CKPT))
    m.eval(); m.tgt_split = "all"
    text_diff = float((m.obj_text_encoder.classifier_weights.float() - torch.from_numpy(E)).abs().max())
    cap = {}
    m.featEmbedding.register_forward_hook(lambda mod, i, o: cap.__setitem__("fe", o))
    d = dict(np.load(os.path.join(SC.DATA2, "slots.npz")))
    videos = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))["videos"]
    fit_v = sorted({int(v) for v in d["video"][d["subset"] == 0]})[:1 if a.small else 5]
    worst_fp32 = worst_cache = 0.0
    agree_cache, n_pairs = 0, 0
    for vi in fit_v:
        c = load_cache(f"cache_s3/train/{videos[vi]}.pt")
        for p in c["pairs"]:
            feats = {k: v.float().unsqueeze(0) for k, v in p["slot_inputs"].items()}
            slen = feats["clip_feat"].shape[1]
            with torch.no_grad():
                _, sbj_s, obj_s = m(feats, torch.tensor([slen]))
            _, sbj, obj, _ = cap["fe"][:4]
            for model_p, emb_fp32, emb_cache in ((sbj_s, sbj, p["sbj_emb"]), (obj_s, obj, p["obj_emb"])):
                mp = model_p.reshape(-1, 35).double().numpy()
                worst_fp32 = max(worst_fp32, float(np.abs(head_probs(emb_fp32.reshape(-1, 768).numpy(), E) - mp).max()))
                pc = head_probs(emb_cache.float().numpy(), E)
                worst_cache = max(worst_cache, float(np.abs(pc - mp).max()))
                agree_cache += int(np.argmax(pc[0]) == np.argmax(mp[0]))
            n_pairs += 1
    res["reproduction"] = dict(videos=[videos[v] for v in fit_v], pairs=n_pairs, object_text_vs_model_max_abs=text_diff,
                               max_abs_prob_diff_head_on_fp32_embedding=worst_fp32,
                               max_abs_prob_diff_from_cached_fp16_embedding=worst_cache,
                               top1_same_from_cache=agree_cache / max(2 * n_pairs, 1), temperature=TEMP,
                               passed=worst_fp32 <= 1e-4, load=load)
    print("1-0:", json.dumps({k: v for k, v in res["reproduction"].items() if k != "load"}), flush=True)
    del m

    # ---- 1a: HELD, GT tracks
    held = sorted({int(v) for v in d["video"][d["subset"] == 2]})[:5 if a.small else None]
    acc1 = acc3 = n = 0
    per = collections.defaultdict(lambda: [0, 0, 0])
    for vi in held:
        c = load_cache(f"cache_s3/train/{videos[vi]}.pt")
        acc = collections.defaultdict(lambda: np.zeros(35)); wsum = collections.Counter()
        for p in c["pairs"]:
            for (wa, wb), se, oe in zip(p["windows"], p["sbj_emb"], p["obj_emb"]):
                k = wb - wa
                acc[p["sub"]] += k * head_probs(se.float().numpy(), E)[0]; wsum[p["sub"]] += k
                acc[p["obj"]] += k * head_probs(oe.float().numpy(), E)[0]; wsum[p["obj"]] += k
        for tid, s in acc.items():
            gtc = c["tracks"][tid]["category"]
            order = np.argsort(-s)
            t1, t3 = cats[order[0]] == gtc, gtc in [cats[i] for i in order[:3]]
            acc1 += t1; acc3 += t3; n += 1
            per[gtc][0] += 1; per[gtc][1] += t1; per[gtc][2] += t3
    res["held"] = dict(videos=len(held), tracks=n, top1=acc1 / max(n, 1), top3=acc3 / max(n, 1),
                       per_category={k: dict(tracks=v[0], top1=v[1] / v[0], top3=v[2] / v[0]) for k, v in sorted(per.items())})
    print(f"1a: HELD tracks {n}, top-1 {acc1 / max(n, 1):.3f}, top-3 {acc3 / max(n, 1):.3f}", flush=True)

    # ---- 1b: test, detected tracks (label-free)
    import eval_from_cache as E4
    sdir = os.path.join(C.BOX_COPY, "cache_s3_sgdet", "test")
    vids = sorted(f[:-3] for f in os.listdir(sdir) if f.endswith(".pt"))[:5 if a.small else None]
    n_tr = n_with = n_dis = n_fdis = 0
    sc_agree, sc_dis = [], []
    n_pred = n_pred_dis = n_unmatched = n_pair_unmatched = 0
    fused_cat = {}
    for v in vids:
        c = load_cache(f"cache_s3_sgdet/test/{v}.pt")
        tracks = c["tracks"]
        idx = track_index(tracks)
        acc = collections.defaultdict(lambda: np.zeros(35)); wsum = collections.Counter()
        for p in c["pairs"]:
            k = len(p["logits"])
            s_hit = match(idx, p["sbj_cls"], p["duration"][0], p["sbj_traj"][0])
            o_hit = match(idx, p["obj_cls"], p["duration"][0], p["obj_traj"][0])
            if len(s_hit) != 1 or len(o_hit) != 1:
                n_pair_unmatched += 1
                continue
            acc[s_hit[0]] += k * head_probs(p["sbj_emb"].float().numpy(), E)[0]; wsum[s_hit[0]] += k
            acc[o_hit[0]] += k * head_probs(p["obj_emb"].float().numpy(), E)[0]; wsum[o_hit[0]] += k
        dis = set()
        for tid, t in tracks.items():
            n_tr += 1
            if tid not in acc:
                continue
            n_with += 1
            hp = acc[tid] / wsum[tid]
            hcat = cats[int(np.argmax(hp))]
            if hcat != t["category"]:
                n_dis += 1; dis.add(tid); sc_dis.append(t["score"])
            else:
                sc_agree.append(t["score"])
            fc, fs = fuse(t["category"], float(t["score"]), hp, cats)
            fused_cat[f"{v}/{tid}"] = dict(tracker=t["category"], tracker_score=float(t["score"]), head=hcat,
                                           head_p=float(hp.max()), fused=fc, fused_score=fs)
            n_fdis += fc != t["category"]
        preds = E4.predict({v: c["pairs"]}, "all")[v]
        for r in preds:
            sh = match(idx, r["triplet"][0], r["duration"][0], r["sub_traj"][0])
            oh = match(idx, r["triplet"][2], r["duration"][0], r["obj_traj"][0])
            n_unmatched += not sh or not oh
            n_pred += 1
            n_pred_dis += bool(set(sh) & dis or set(oh) & dis)
    q = lambda x: dict(n=len(x), **({k: float(np.percentile(x, pp)) for k, pp in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))} if x else {}))
    res["sgdet"] = dict(videos=len(vids), tracks=n_tr, tracks_in_a_pair=n_with, disagreeing=n_dis,
                        disagreement_rate=n_dis / max(n_with, 1), fused_differs_from_tracker=n_fdis,
                        predictions=n_pred, predictions_with_disagreeing_track=n_pred_dis,
                        share_predictions=n_pred_dis / max(n_pred, 1), predictions_unmatched=n_unmatched,
                        pairs_unmatched=n_pair_unmatched, tracker_score_agree=q(sc_agree), tracker_score_disagree=q(sc_dis))
    g = res["held"]["top1"] >= G1_TOP1 and res["sgdet"]["disagreement_rate"] >= G1_DISAGREE
    res["gate1"] = dict(rule=f"1a top-1 >= {G1_TOP1} and 1b disagreement >= {G1_DISAGREE}",
                        checks=dict(top1=res["held"]["top1"] >= G1_TOP1, disagreement=res["sgdet"]["disagreement_rate"] >= G1_DISAGREE),
                        verdict="PASS" if g else "FAIL")
    print(f"1b: tracks {n_tr} (in a pair {n_with}), disagreeing {n_dis} ({n_dis / max(n_with, 1):.3f}); "
          f"predictions with a disagreeing track {n_pred_dis}/{n_pred}; gate 1 {res['gate1']['verdict']}", flush=True)
    return res, fused_cat


# ---------------------------------------------------------------------------
def flicker(pairs_scores):
    """pairs_scores: iterable of (S (L, 61) novel scores). -> persistence, runs, Spearman."""
    keep = kept = trans = 0
    runs, rhos = [], []

    def rank(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)       # average ties
        sums = np.bincount(inv, r); return (sums / cnt)[inv]
    for S in pairs_scores:
        L = len(S)
        top = [set(np.argsort(-S[t])[:20].tolist()) for t in range(L)]
        for t in range(L - 1):
            trans += 1
            keep += len(top[t]); kept += len(top[t] & top[t + 1])
            ra, rb = rank(S[t]), rank(S[t + 1])
            if ra.std() > 0 and rb.std() > 0:
                rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
        for p in set().union(*top):
            run = 0
            for t in range(L):
                if p in top[t]:
                    run += 1
                elif run:
                    runs.append(run); run = 0
            if run:
                runs.append(run)
    R = np.array(runs) if runs else np.zeros(1)
    return dict(transitions=trans, persistence=kept / max(keep, 1), runs=len(runs), run_median=float(np.median(R)),
                run_mean=float(R.mean()), share_single_slot_runs=float((R == 1).mean()),
                spearman_mean=float(np.mean(rhos)) if rhos else None, spearman_median=float(np.median(rhos)) if rhos else None)


def part2(a, names, split):
    nov = np.array([i for i, p in enumerate(names) if split[p] == "novel"])
    res = {}
    d = dict(np.load(os.path.join(SC.DATA2, "slots.npz")))
    hp = json.load(open(os.path.join(SC.DATA2, "held_pairs.json")))
    vids = sorted(hp)[:5 if a.small else None]
    E_ = eov(d["logits"], d["int_logit"])
    res["i_held_gt"] = flicker(E_[p["rows"][0]:p["rows"][1]][:, nov] for v in vids for p in hp[v])
    for key, sub in (("ii_test_gt", "cache_s3/test"), ("iii_test_detected", "cache_s3_sgdet/test")):
        tv = sorted(f[:-3] for f in os.listdir(os.path.join(C.BOX_COPY, sub)) if f.endswith(".pt"))[:5 if a.small else None]

        def gen():
            for v in tv:
                c = load_cache(f"{sub}/{v}.pt")
                for p in c["pairs"]:
                    yield eov(p["logits"].numpy(), p["int_logit"].numpy())[:, nov]
        res[key] = flicker(gen())
        res[key]["videos"] = len(tv)
    res["i_held_gt"]["videos"] = len(vids)
    g = res["ii_test_gt"]["persistence"] <= G2_PERSIST or res["iii_test_detected"]["persistence"] <= G2_PERSIST
    res["gate2"] = dict(rule=f"persistence <= {G2_PERSIST} on (ii) or (iii)", verdict="PASS" if g else "FAIL")
    for k in ("i_held_gt", "ii_test_gt", "iii_test_detected"):
        print(f"2 {k}: {json.dumps(res[k])}", flush=True)
    print("gate 2:", res["gate2"]["verdict"])
    return res


# ---------------------------------------------------------------------------
def part3(a, names, split, base, fused):
    res = {}
    sp = C.splits()
    verb = lambda p: (C.S0.tokenise(p)[0] if C.compositional(p) and len(C.S0.tokenise(p)) == 2 else None)
    vt = collections.defaultdict(collections.Counter)
    for v in sp["fit"]:
        _, _, _, rels, _, ann = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{v}.json"), base)
        cat = {so["tid"]: so["category"] for so in ann["subject/objects"]}
        for r in rels:
            if verb(r["predicate"]):
                vt[verb(r["predicate"])][subject_type(cat[r["subject_tid"]])] += 1
    rules = {}
    for v_, c_ in vt.items():
        n = sum(c_.values())
        for typ in ("vehicle", "animate"):
            if n >= RULE_MIN and c_[typ] / n >= RULE_SHARE:
                rules[v_] = typ
    res["rules"] = dict(rules=rules, verb_subject_types={k: dict(v) for k, v in sorted(vt.items())}, fit_videos=len(sp["fit"]))
    deletes = lambda cat, p: verb(p) in rules and subject_type(cat) != rules[verb(p)]
    gt = json.load(open(os.path.join(SC.DATA2, "held_gt.json")))
    inst = [(r["triplet"][0], r["triplet"][1]) for g in gt.values() for r in g]
    bad = collections.Counter((c_, p) for c_, p in inst if deletes(c_, p))
    res["false_delete"] = dict(held_instances=len(inst), deleted=sum(bad.values()),
                               rate=sum(bad.values()) / max(len(inst), 1),
                               by_subject_predicate={f"{c_}/{p}": k for (c_, p), k in bad.most_common(20)})
    nov = np.array([i for i, p in enumerate(names) if split[p] == "novel"])

    def reach(items):
        n = k = 0
        for S, cats_ in items:
            for t in range(len(S)):
                for i in np.argsort(-S[t])[:20]:
                    n += 1; k += deletes(cats_[t], names[nov[i]])
        return dict(top20_novel=n, deleted=k, share=k / max(n, 1))
    d = dict(np.load(os.path.join(SC.DATA2, "slots.npz")))
    meta = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))
    E_ = eov(d["logits"], d["int_logit"])
    if a.small:
        keepv = sorted({int(v) for v in d["video"]})[:5]
        dm = dict(d); dm["video"] = np.where(np.isin(d["video"], keepv), d["video"], -1)
    else:
        dm = d
    sub_cat, _, _, pairs = SC.slot_table(dm, meta["videos"], base)
    res["reach_train"] = reach((E_[r0:r1][:, nov], sub_cat[r0:r1]) for _, _, r0, r1 in pairs)
    tv = sorted(f[:-3] for f in os.listdir(os.path.join(C.BOX_COPY, "cache_s3/test")) if f.endswith(".pt"))[:5 if a.small else None]

    def gen_gt():
        for v in tv:
            for p in load_cache(f"cache_s3/test/{v}.pt")["pairs"]:
                S = eov(p["logits"].numpy(), p["int_logit"].numpy())[:, nov]
                yield S, [p["sbj_cls"]] * len(S)
    res["reach_test_gt"] = reach(gen_gt())
    sv = sorted(f[:-3] for f in os.listdir(os.path.join(C.BOX_COPY, "cache_s3_sgdet/test")) if f.endswith(".pt"))[:5 if a.small else None]

    def gen_det(use_fused):
        for v in sv:
            c = load_cache(f"cache_s3_sgdet/test/{v}.pt")
            idx = track_index(c["tracks"])
            for p in c["pairs"]:
                S = eov(p["logits"].numpy(), p["int_logit"].numpy())[:, nov]
                cat = p["sbj_cls"]
                if use_fused:
                    hit = match(idx, p["sbj_cls"], p["duration"][0], p["sbj_traj"][0])
                    if len(hit) == 1 and f"{v}/{hit[0]}" in fused:
                        cat = fused[f"{v}/{hit[0]}"]["fused"]
                yield S, [cat] * len(S)
    res["reach_test_detected_tracker"] = reach(gen_det(False))
    res["reach_test_detected_fused"] = reach(gen_det(True))
    best = max(res["reach_test_gt"]["share"], res["reach_test_detected_tracker"]["share"], res["reach_test_detected_fused"]["share"])
    g = res["false_delete"]["rate"] <= G3_FALSE and best >= G3_REACH
    res["gate3"] = dict(rule=f"false-delete <= {G3_FALSE} and reach >= {G3_REACH} on (ii) or (iii)",
                        checks=dict(false_delete=res["false_delete"]["rate"] <= G3_FALSE, reach=best >= G3_REACH),
                        verdict="PASS" if g else "FAIL")
    print(json.dumps({k: v for k, v in res.items() if k != "rules"} | {"rules": rules}, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    C.verify_frozen()
    out = os.path.join(HERE, "data_small" if a.small else "data")
    os.makedirs(out, exist_ok=True)
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    labels = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))["labels"]
    assert all(set(l) <= base for l in labels)                # 0 novel labels used
    names = EV.names()
    E, cats = object_embeddings()
    paths = []
    if a.part == 1:
        res, fused = part1(a, E, cats)
        paths = [os.path.join(out, "f2_part1.json"), os.path.join(out, "f2_fused_classes.json")]
        json.dump(res, open(paths[0], "w"), indent=1, default=str)
        json.dump(fused, open(paths[1], "w"))
    elif a.part == 2:
        res = part2(a, names, split)
        paths = [os.path.join(out, "f2_part2.json")]
        json.dump(res, open(paths[0], "w"), indent=1)
    else:
        fused = json.load(open(os.path.join(out, "f2_fused_classes.json")))
        res = part3(a, names, split, base, fused)
        paths = [os.path.join(out, "f2_part3.json")]
        json.dump(res, open(paths[0], "w"), indent=1, default=str)
    write_log(f"f2_part{a.part}" + ("_small" if a.small else ""),
              dict(part=a.part, small=a.small, temperature=TEMP,
                   gates=dict(top1=G1_TOP1, disagree=G1_DISAGREE, persistence=G2_PERSIST, false_delete=G3_FALSE, reach=G3_REACH),
                   rule=dict(share=RULE_SHARE, min=RULE_MIN)),
              [C.CACHE_SUMS, os.path.join(SC.DATA2, "slots.npz"), os.path.join(SC.DATA2, "held_gt.json")], paths,
              extra=dict(seconds=round(time.time() - t0, 1)))
    print(f"done | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
