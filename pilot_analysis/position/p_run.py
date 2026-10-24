#!/usr/bin/env python3
"""Position verifier (A18), steps 1-4.

    python pilot_analysis/position/p_run.py --step 1 [--small]   # features, targets, instances, symmetry
    python pilot_analysis/position/p_run.py --step 2 [--small]   # final model, 5-fold cross-fitting, tau
    python pilot_analysis/position/p_run.py --step 3 [--small]   # G-P1 / G-P2 on HELD (GT and jittered boxes)
    python pilot_analysis/position/p_run.py --step 4 [--small]   # G-P3 reach on cached test inputs (label-free)

Training: all 800 training videos (boxes only, no ECC needed), base labels only.
Test: cached model inputs only (pair trajectories, per-slot logits); no test relation
label is opened and the evaluator is never run on test.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, json, os, time

import numpy as np

import pos_common as P
from pos_common import C, SC

SUB = dict(fit_train=0, es=1, held=2)


def out_dir(small):
    d = os.path.join(P.HERE, "data_small" if small else "data")
    os.makedirs(d, exist_ok=True)
    return d


def splits800():
    sp = json.load(open(C.SPLIT))
    es = json.load(open(os.path.join(C.OUT, "step0_selection.json")))["early_stop_videos"]
    assert set(es) <= set(sp["fit"]) and len(es) == 63
    sub = {v: "held" for v in sp["held"]}
    sub.update({v: ("es" if v in set(es) else "fit_train") for v in sp["fit"]})
    return sub


def slot_feats(sb, ob, W, H):
    F, ok = P.frame_features(sb, ob, W, H)
    Fs, oks = P.frame_features(ob, sb, W, H)
    return F, Fs, ok


# ---------------------------------------------------------------------------
def step1(a, out):
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    sub = splits800()
    vids = sorted(sub)
    if a.small:
        vids = sorted([v for v in vids if sub[v] == "fit_train"][:5] + [v for v in vids if sub[v] == "es"][:5]
                      + [v for v in vids if sub[v] == "held"][:5])
    X, XS, MASK, UNT, rows, labels, inst = [], [], [], [], [], [], []
    sym = collections.Counter()
    nov = 0
    for vi, v in enumerate(vids):
        _, W, H, rels, n_nov, _ = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{v}.json"), base)
        nov += n_nov
        c = P.load_cache(f"cache_s3/train/{v}.pt")
        assert tuple(c["size"]) == (W, H)
        pair_rows = collections.defaultdict(list)          # (sub_gt, obj_gt) -> [(row, s0, s1)]
        for pi, p in enumerate(c["pairs"]):
            sb, ob = np.asarray(p["sbj_traj"], float), np.asarray(p["obj_traj"], float)
            slots = P.pair_slots(len(sb))
            assert len(slots) == len(p["logits"]), (v, pi)
            d0 = p["duration"][0]
            gap_all = P.gap_over_diag(sb, ob)                       # W3: the whole pair
            gap_all_s = P.gap_over_diag(ob, sb)
            for k, (s0, s1) in enumerate(slots):
                F, Fs, ok = slot_feats(sb[s0:s1], ob[s0:s1], W, H)
                sym["feature_swap_n"] += 1
                sym["feature_swap_exact"] += bool(np.array_equal(P.swap_features(F), Fs))
                w0, w1 = P.windows_for(k, slots, "W1")
                F1, _, ok1 = slot_feats(sb[w0:w1], ob[w0:w1], W, H)
                okw3 = np.ones(len(sb), bool)
                u = P.untrained_scores(F, ok, np.full(len(F), 0.0))
                u.update({k_: P.untrained_scores(F1, ok1, np.zeros(len(F1)))[k_] for k_ in ("front", "behind")})
                u["adjacent"] = P.untrained_scores(np.zeros((len(sb), 15)), okw3, gap_all)["adjacent"]
                # swapped-role untrained scores: must flip exactly
                us = P.untrained_scores(Fs, ok, np.zeros(len(F)))
                F1s = P.frame_features(ob[w0:w1], sb[w0:w1], W, H)[0]
                us.update({k_: P.untrained_scores(F1s, ok1, np.zeros(len(F1s)))[k_] for k_ in ("front", "behind")})
                us["adjacent"] = P.untrained_scores(np.zeros((len(sb), 15)), okw3, gap_all_s)["adjacent"]
                sym["rule_swap_n"] += 1
                sym["rule_swap_exact"] += all(u[x] == us[y] for x, y in (("left", "right"), ("right", "left"), ("above", "beneath"),
                                                                       ("beneath", "above"), ("front", "behind"), ("behind", "front"),
                                                                       ("adjacent", "adjacent")))
                pad = np.zeros((P.CLIP, 15)); m = np.zeros(P.CLIP)
                Xp, Xsp = pad.copy(), pad.copy()
                Xp[:len(F)], Xsp[:len(F)], m[:len(F)] = F, Fs, ok
                X.append(Xp); XS.append(Xsp); MASK.append(m)
                UNT.append([u[val] for _, val in P.COVERED])
                lab = sorted(C.S.slot_labels((d0 + s0, d0 + s1), p["sub_gt"], p["obj_gt"], rels))
                assert set(lab) <= base
                labels.append(lab)
                r = len(rows)
                rows.append((vi, pi, k, SUB[sub[v]], int(ok.sum()), p["sbj_cls"]))
                pair_rows[(p["sub_gt"], p["obj_gt"])].append((r, d0 + s0, d0 + s1))
        cat = None
        for rr in rels:
            val = P.value_of(phi, rr["predicate"])
            if val is None:
                continue
            cov = [r for r, s0, s1 in pair_rows.get((rr["subject_tid"], rr["object_tid"]), [])
                   if min(s1, rr["end_fid"]) - max(s0, rr["begin_fid"]) >= C.S.LABEL_MIN_COVER]
            inst.append(dict(video=vi, subset=sub[v], predicate=rr["predicate"], axis=val[0], value=val[1], rows=cov,
                             subject=c["tracks"][[t for t in c["tracks"] if c["tracks"][t]["gt_tid"] == rr["subject_tid"]][0]]["category"]
                             if any(c["tracks"][t]["gt_tid"] == rr["subject_tid"] for t in c["tracks"]) else None))
        if (vi + 1) % 100 == 0:
            print(f"  [{vi + 1}/{len(vids)}]", flush=True)
    if not a.small:
        assert nov == 3301, nov
    Y = np.zeros((len(rows), 4)); Mk = np.zeros((len(rows), 4))
    for i, lab in enumerate(labels):
        for j, ax in enumerate(P.AXES[:3]):
            vals = {phi[p][ax] for p in lab if ax in phi[p]}
            if len(vals) == 1:
                Mk[i, j] = 1; Y[i, j] = float(vals.pop() == P.POS_VALUE[ax])
        prox = {phi[p]["proximity"] for p in lab if "proximity" in phi[p]}
        other = any(ax in phi[p] for p in lab for ax in P.AXES[:3])
        if "adjacent" in prox or (not prox and other):
            Mk[i, 3] = 1; Y[i, 3] = float("adjacent" in prox)
    rr = np.array([r[:5] for r in rows])
    arr = dict(X=np.array(X, np.float32), Xs=np.array(XS, np.float32), M=np.array(MASK, np.float32),
               U=np.array(UNT, np.float64), Y=Y, Mk=Mk, video=rr[:, 0], pair=rr[:, 1], slot=rr[:, 2], subset=rr[:, 3],
               n_frames=rr[:, 4])
    np.savez(os.path.join(out, "pos_slots.npz"), **arr)
    json.dump(dict(videos=vids, labels=labels, sbj_cls=[r[5] for r in rows], novel_dropped=nov, novel_labels_used=0),
              open(os.path.join(out, "pos_meta.json"), "w"))
    json.dump(inst, open(os.path.join(out, "pos_instances.json"), "w"))
    res = dict(videos=len(vids), slots=len(rows), instances_covered_value=len(inst),
               instances_without_covering_slot=sum(1 for i in inst if not i["rows"]),
               targets={ax: dict(observed=int(Mk[:, j].sum()), positive=int((Y[:, j] * Mk[:, j]).sum())) for j, ax in enumerate(P.AXES)},
               symmetry={k: v for k, v in sym.items()},
               symmetry_exact=sym["feature_swap_exact"] == sym["feature_swap_n"] and sym["rule_swap_exact"] == sym["rule_swap_n"],
               novel_dropped_at_parse=nov, novel_labels_used=0)
    json.dump(res, open(os.path.join(out, "step1.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    if not res["symmetry_exact"]:
        sys.exit("!! swap symmetry not exact -- STOP")
    return [os.path.join(out, f) for f in ("pos_slots.npz", "pos_meta.json", "pos_instances.json", "step1.json")]


def load_step1(out):
    d = dict(np.load(os.path.join(out, "pos_slots.npz")))
    meta = json.load(open(os.path.join(out, "pos_meta.json")))
    inst = json.load(open(os.path.join(out, "pos_instances.json")))
    return d, meta, inst


def instance_scores(inst, Pv_rows):
    """instance score = min over its covering slots (a veto in any slot removes it)."""
    return np.array([min(Pv_rows[r] for r in i["rows"]) if i["rows"] else np.inf for i in inst])


def value_matrix(Pnet):
    return {val: P.value_prob(Pnet, ax, val) for ax, val in P.COVERED}


# ---------------------------------------------------------------------------
def step2(a, out):
    import torch
    torch.set_num_threads(6)
    d, meta, inst = load_step1(out)
    stats = P.norm_stats(d["X"][d["subset"] == 0].reshape(-1, 15), d["M"][d["subset"] == 0].reshape(-1))
    D = dict(X=P.normalise(d["X"], stats) * d["M"][..., None], Xs=P.normalise(d["Xs"], stats) * d["M"][..., None],
             M=d["M"], Y=d["Y"], Mk=d["Mk"])
    obs = d["Mk"].sum(1) > 0
    tr = np.where((d["subset"] == 0) & obs)[0]
    es = np.where((d["subset"] == 1) & obs)[0]
    nets, runs = [], []
    for s in P.SEEDS:
        net, info = P.train(D, tr, es, s, max_ep=5 if a.small else P.MAX_EP)
        nets.append(net); runs.append(info)
        print(f"  final seed {s}: epochs {info['epochs_run']}, best {info['best_epoch']}, es loss {info['best_es_loss']:.4f}", flush=True)
    Pfin = P.predict(nets, D["X"], D["Xs"], D["M"])
    # symmetry of the network on 1,000 random slots
    rng = np.random.default_rng(0)
    idx = rng.choice(len(D["X"]), size=min(1000, len(D["X"])), replace=False)
    with torch.no_grad():
        worst_anti = worst_sym = 0.0
        for net in nets:
            Xa, Xb, Mm = (torch.from_numpy(D[k][idx].astype(np.float32)) for k in ("X", "Xs", "M"))
            z1, z2 = net(Xa, Xb, Mm).numpy(), net(Xb, Xa, Mm).numpy()
            worst_anti = max(worst_anti, float(np.abs(z1[:, :3] + z2[:, :3]).max()))
            worst_sym = max(worst_sym, float(np.abs(z1[:, 3] - z2[:, 3]).max()))
    # 5-fold video-level cross-fitting on FIT (fit-train + early-stop), fixed epochs from the final run
    fit_v = sorted({int(v) for v in d["video"][d["subset"] != 2]})
    perm = np.random.default_rng(P.FOLD_SEED).permutation(len(fit_v))
    fold_of = {fit_v[i]: k % P.FOLDS for k, i in enumerate(perm)}
    Poof = np.full((len(D["X"]), 4), np.nan)
    fold_runs = []
    for f in range(P.FOLDS):
        trf = np.where(obs & (d["subset"] != 2) & np.array([fold_of.get(int(v), -1) != f for v in d["video"]]))[0]
        te = np.where((d["subset"] != 2) & np.array([fold_of.get(int(v), -1) == f for v in d["video"]]))[0]
        fn = []
        for s, info in zip(P.SEEDS, runs):
            net, fi = P.train(D, trf, None, s, fixed_epochs=info["best_epoch"])
            fn.append(net); fold_runs.append(dict(fold=f, seed=s, epochs=fi["epochs_run"], train_slots=len(trf), test_slots=len(te)))
        Poof[te] = P.predict(fn, D["X"][te], D["Xs"][te], D["M"][te])
        print(f"  fold {f}: train {len(trf)} slots, predict {len(te)} | {time.time() - T0:.0f}s", flush=True)
    assert not np.isnan(Poof[d["subset"] != 2]).any()
    Vo, Vf = value_matrix(Poof), value_matrix(Pfin)
    U = {val: d["U"][:, k] for k, (_, val) in enumerate(P.COVERED)}
    rows = {}
    for kind, Vm in (("network", Vo), ("untrained", U)):
        for ax, val in P.COVERED:
            fit_i = [i for i in inst if i["value"] == val and i["subset"] != "held" and i["rows"]]
            held_i = [i for i in inst if i["value"] == val and i["subset"] == "held" and i["rows"]]
            sc = instance_scores(fit_i, Vm[val])
            tau = P.choose_tau(sc)
            r = dict(tau=tau, fit_instances=len(fit_i), held_instances=len(held_i),
                     fit_oof_false_delete=int((sc < tau).sum()), fit_oof_rate=float((sc < tau).mean()) if len(sc) else None)
            if kind == "network":
                j = P.AXES.index(ax)
                h = np.where((d["subset"] == 2) & (d["Mk"][:, j] > 0))[0]
                y = d["Y"][h, j] if P.POS_VALUE[ax] == val else 1 - d["Y"][h, j]
                r["held_slot_auc"] = SC.auc(y, Vf[val][h]) if 0 < y.sum() < len(y) else None
            rows.setdefault(kind, {})[val] = r
    taus = {k: {val: rows[k][val]["tau"] for _, val in P.COVERED} for k in rows}
    res = dict(final=[{k: v for k, v in r.items() if k != "es_curve"} for r in runs], es_curves=[r["es_curve"] for r in runs],
               folds=fold_runs, symmetry=dict(slots=len(idx), max_antisymmetry_error=worst_anti, max_symmetry_error_proximity=worst_sym,
                                             passed=worst_anti < 1e-5 and worst_sym < 1e-5),
               per_value=rows, tau=taus, norm=dict(mean=stats["mean"].tolist(), std=stats["std"].tolist()))
    torch.save(dict(final=[n.state_dict() for n in nets], stats=stats, runs=runs), os.path.join(out, "verifier_final.pt"))
    np.savez(os.path.join(out, "verifier_scores.npz"), P_final=Pfin, P_oof=Poof)
    json.dump(res, open(os.path.join(out, "step2.json"), "w"), indent=1)
    print(json.dumps(dict(symmetry=res["symmetry"], per_value=rows), indent=1))
    return [os.path.join(out, f) for f in ("verifier_final.pt", "verifier_scores.npz", "step2.json")]


def load_verifier(out):
    import torch
    m = torch.load(os.path.join(out, "verifier_final.pt"), weights_only=False)
    nets = []
    for st in m["final"]:
        n = P.make_model(); n.load_state_dict(st); n.eval(); nets.append(n)
    return nets, m["stats"]


def featurise(pairs, W, H, stats):
    """pairs: list of (sbj_traj, obj_traj) -> X, Xs, M (normalised), untrained (N, 7), n_frames."""
    X, XS, MASK, UNT, NF = [], [], [], [], []
    for sbt, obt in pairs:
        sb, ob = np.asarray(sbt, float), np.asarray(obt, float)
        slots = P.pair_slots(len(sb))
        gap_all = P.gap_over_diag(sb, ob)
        for k, (s0, s1) in enumerate(slots):
            F, Fs, ok = slot_feats(sb[s0:s1], ob[s0:s1], W, H)
            w0, w1 = P.windows_for(k, slots, "W1")
            F1, _, ok1 = slot_feats(sb[w0:w1], ob[w0:w1], W, H)
            u = P.untrained_scores(F, ok, np.zeros(len(F)))
            u.update({x: P.untrained_scores(F1, ok1, np.zeros(len(F1)))[x] for x in ("front", "behind")})
            u["adjacent"] = P.untrained_scores(np.zeros((len(sb), 15)), np.ones(len(sb), bool), gap_all)["adjacent"]
            Xp, Xsp, m = np.zeros((P.CLIP, 15)), np.zeros((P.CLIP, 15)), np.zeros(P.CLIP)
            n = min(len(F), P.CLIP)
            Xp[:n], Xsp[:n], m[:n] = F[:n], Fs[:n], ok[:n]
            X.append(Xp); XS.append(Xsp); MASK.append(m); UNT.append([u[val] for _, val in P.COVERED]); NF.append(int(ok.sum()))
    M = np.array(MASK, np.float32)
    return (P.normalise(np.array(X), stats) * M[..., None], P.normalise(np.array(XS), stats) * M[..., None], M,
            np.array(UNT), np.array(NF))


# ---------------------------------------------------------------------------
def gate_rates(inst, V, tau, axes):
    out, pooled = {}, [0, 0]
    for ax in axes:
        vals = [val for a_, val in P.COVERED if a_ == ax]
        ii = [i for i in inst if i["value"] in vals and i["subset"] == "held" and i["rows"]]
        k = sum(min(V[i["value"]][r] for r in i["rows"]) < tau[i["value"]] for i in ii)
        out[ax] = dict(instances=len(ii), deleted=int(k), rate=k / max(len(ii), 1), ci95=P.clopper_pearson(int(k), len(ii)))
        pooled[0] += k; pooled[1] += len(ii)
    return out, dict(instances=pooled[1], deleted=int(pooled[0]), rate=pooled[0] / max(pooled[1], 1),
                     ci95=P.clopper_pearson(int(pooled[0]), pooled[1]))


def survivors(g1, g2):
    """A18: an axis survives if G-P1 <= 1.0% and G-P2 (IoU 0.70) <= 1.0%; then, while the pooled
    G-P1 > 0.5% or pooled G-P2 (0.70) > 1.0% over the survivors, drop the survivor with the
    highest G-P1 rate."""
    s = [ax for ax in P.AXES if g1[ax]["rate"] <= 0.010 and g2[ax]["rate"] <= 0.010 and g1[ax]["instances"]]

    def pooled(g, axes):
        k = sum(g[x]["deleted"] for x in axes); n = sum(g[x]["instances"] for x in axes)
        return k / max(n, 1)
    while s and (pooled(g1, s) > 0.005 or pooled(g2, s) > 0.010):
        s.remove(max(s, key=lambda x: g1[x]["rate"]))
    return s


def step3(a, out):
    import torch
    torch.set_num_threads(6)
    d, meta, inst = load_step1(out)
    s2 = json.load(open(os.path.join(out, "step2.json")))
    nets, stats = load_verifier(out)
    Pf = np.load(os.path.join(out, "verifier_scores.npz"))["P_final"]
    V = dict(network=value_matrix(Pf), untrained={val: d["U"][:, k] for k, (_, val) in enumerate(P.COVERED)})
    res = {}
    # ---- G-P2: jittered HELD boxes, mean IoU 0.85 and 0.70 (seed 0), tau unchanged
    held_v = sorted({int(v) for v in d["video"][d["subset"] == 2]})
    caches = {vi: P.load_cache(f"cache_s3/train/{meta['videos'][vi]}.pt") for vi in held_v}
    tracks = {(vi, tid): t["boxes"] for vi, c in caches.items() for tid, t in c["tracks"].items()}
    draws = P.jitter_draws(tracks, seed=0)

    def mean_iou(s):
        num = den = 0.0
        for key, b in tracks.items():
            j = P.apply_jitter(b, draws[key], s)
            num += P.box_iou(np.asarray(b, float), j).sum(); den += len(b)
        return num / den

    def calibrate(target):
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            (lo, hi) = (mid, hi) if mean_iou(mid) > target else (lo, mid)
        return (lo + hi) / 2
    held_rows = np.where(d["subset"] == 2)[0]
    for tgt in (0.85, 0.70):
        s = calibrate(tgt)
        Vj = {k: {val: v.copy() for val, v in V[k].items()} for k in V}
        for vi in held_v:
            c = caches[vi]
            jt = {tid: P.apply_jitter(t["boxes"], draws[(vi, tid)], s) for tid, t in c["tracks"].items()}
            prs = []
            for p in c["pairs"]:
                st, ot = c["tracks"][p["sub"]], c["tracks"][p["obj"]]
                b0, b1 = p["duration"]
                assert np.allclose(np.asarray(st["boxes"], float)[b0 - st["begin_fid"]:b1 - st["begin_fid"]], np.asarray(p["sbj_traj"], float))
                prs.append((jt[p["sub"]][b0 - st["begin_fid"]:b1 - st["begin_fid"]], jt[p["obj"]][b0 - ot["begin_fid"]:b1 - ot["begin_fid"]]))
            W, H = c["size"]
            X, Xs, M, Unt, NF = featurise(prs, W, H, stats)
            rows = held_rows[d["video"][held_rows] == vi]
            assert len(rows) == len(X)
            Pj = P.predict(nets, X, Xs, M)
            for k, (_, val) in enumerate(P.COVERED):
                Vj["network"][val][rows] = P.value_prob(Pj, P.COVERED[k][0], val)
                Vj["untrained"][val][rows] = Unt[:, k]
        res[f"jitter_{tgt}"] = dict(scale=s, mean_iou=mean_iou(s),
                                    **{k: dict(zip(("per_axis", "pooled"), gate_rates(inst, Vj[k], s2["tau"][k], P.AXES))) for k in V})
        print(f"  jitter IoU {tgt}: scale {s:.4f}", flush=True)
    # ---- G-P1 on GT boxes, survivors, breakdowns
    fit_rows = d["subset"] != 2
    subj_slots = collections.Counter(meta["sbj_cls"][r] for r in np.where(fit_rows & (d["Mk"].sum(1) > 0))[0])
    fam = lambda p: (C.S0.tokenise(p)[0] if len(C.S0.tokenise(p)) == 2 and C.S0.tokenise(p)[0] in C.S0.VERB else "no verb")
    for k in V:
        g1, g1p = gate_rates(inst, V[k], s2["tau"][k], P.AXES)
        g2 = res["jitter_0.7"][k]["per_axis"]
        surv = survivors(g1, g2)
        br = {}
        if surv:
            vals = {val for ax, val in P.COVERED if ax in surv}
            hi = [i for i in inst if i["subset"] == "held" and i["rows"] and i["value"] in vals]
            dele = lambda i: min(V[k][i["value"]][r] for r in i["rows"]) < s2["tau"][k][i["value"]]
            rare = [i for i in hi if subj_slots.get(i["subject"], 0) <= 5]
            br["rare_subjects_le5_fit_slots"] = dict(instances=len(rare), deleted=int(sum(map(dele, rare))))
            br["per_family"] = {f: dict(instances=len(x), deleted=int(sum(map(dele, x))))
                                for f in sorted({fam(i["predicate"]) for i in hi})
                                for x in [[i for i in hi if fam(i["predicate"]) == f]]}
        res[k] = dict(gp1=dict(per_axis=g1, pooled=g1p), gp2_070=res["jitter_0.7"][k], gp2_085=res["jitter_0.85"][k],
                      surviving_axes=surv, breakdowns=br,
                      verdict="PASS: " + ", ".join(surv) if surv else "FAIL: no axis survives")
        print(f"  {k}: G-P1 pooled {g1p['rate']:.4f}; per axis {[(ax, round(g1[ax]['rate'], 4), round(g2[ax]['rate'], 4)) for ax in P.AXES]} "
              f"-> survivors {surv}", flush=True)
    json.dump(res, open(os.path.join(out, "step3.json"), "w"), indent=1, default=str)
    return [os.path.join(out, "step3.json")]


# ---------------------------------------------------------------------------
def step4(a, out):
    import torch
    torch.set_num_threads(6)
    sys.path.insert(0, os.path.join(C.PA, "stage1c"))
    import eval_from_cache as E4
    from inference.post_process import association, format_, process_pred
    from types import SimpleNamespace
    s3 = json.load(open(os.path.join(out, "step3.json")))
    s2 = json.load(open(os.path.join(out, "step2.json")))
    nets, stats = load_verifier(out)
    phi, split = C.load_phi()
    names = json.load(open(os.path.join(C.ROOT, "data", "vidvrd", "data", "id2predicate.json")))
    id2pre, obj2id, base_ids = E4.meta()
    novel_cols = [j for j, p in enumerate(names) if split[p] == "novel"]
    rules = json.load(open(os.path.join(C.PA, "f2", "data", "f2_part3.json")))["rules"]["rules"]
    verb = lambda p: (C.S0.tokenise(p)[0] if C.compositional(p) and len(C.S0.tokenise(p)) == 2 else None)
    ruled = {j: rules[verb(p)] for j, p in enumerate(names) if verb(p) in rules}
    type_del = lambda cat: {j for j, t in ruled.items() if SC.subject_type(cat) != t}       # T1's rule
    val_of = {j: P.value_of(phi, p) for j, p in enumerate(names)}
    args = SimpleNamespace(**C.POST)
    res = {}
    for task, sub in (("sgdet", "cache_s3_sgdet/test"), ("predcls", "cache_s3/test")):
        vids = sorted(f[:-3] for f in os.listdir(os.path.join(C.BOX_COPY, sub)) if f.endswith(".pt"))[:5 if a.small else None]
        for kind in ("network", "untrained"):
            surv = s3[kind]["surviving_axes"]
            tau = s2["tau"][kind]
            cnt = collections.Counter(); per_pred = collections.Counter()
            outs = collections.Counter()
            for v in vids:
                c = P.load_cache(f"{sub}/{v}.pt")
                if not c["pairs"]:                       # a video with no pair has no slot to count (A19)
                    cnt["videos_without_pairs"] += 1
                    continue
                W, H = c["size"]
                X, Xs, M, Unt, NF = featurise([(p["sbj_traj"], p["obj_traj"]) for p in c["pairs"]], W, H, stats)
                Pn = P.predict(nets, X, Xs, M) if kind == "network" else None
                r = 0
                cands, tag = [], {}
                for pi, p in enumerate(c["pairs"]):
                    L = len(p["logits"])
                    td = type_del(p["sbj_cls"])
                    s_nov = E4.slot_scores(p["logits"], p["int_logit"], "novel", base_ids).numpy()
                    veto = np.zeros((L, 132), bool)
                    for t in range(L):
                        if NF[r + t] < P.MIN_FRAMES:
                            continue
                        for j in novel_cols:
                            vv = val_of[j]
                            if vv is None or vv[0] not in surv:
                                continue
                            pv = (P.value_prob(Pn[r + t:r + t + 1], vv[0], vv[1])[0] if kind == "network"
                                  else Unt[r + t, P.COVERED.index(vv)])
                            veto[t, j] = pv < tau[vv[1]]
                    for t in range(L):
                        for j in np.argsort(-s_nov[t])[:20]:
                            cnt["top20_novel"] += 1
                            if veto[t, j]:
                                cnt["deleted"] += 1
                                if j not in td:
                                    cnt["incremental"] += 1; per_pred[names[j]] += 1
                            cnt["type_rule_deleted"] += j in td
                    # final outputs of the type-rules pipeline (T1's mechanism, novel split), tagged by pair
                    Lg = p["logits"].clone().float()
                    if td:
                        Lg[:, sorted(td)] = float("-inf")
                    pd = dict(sbj_cls=p["sbj_cls"], obj_cls=p["obj_cls"], sbj_scr=p["sbj_scr"], obj_scr=p["obj_scr"],
                              sbj_traj=list(p["sbj_traj"]), obj_traj=list(p["obj_traj"]), duration=list(p["duration"]))
                    for inst in association(process_pred(args, id2pre, obj2id, None,
                                                         E4.slot_scores(Lg, p["int_logit"], "novel", base_ids), pd)):
                        cands.append(inst)
                        k0 = (inst["duration"][0] - p["duration"][0]) // P.CLIP
                        k1 = k0 + max(1, -(-(inst["duration"][1] - inst["duration"][0]) // P.CLIP))
                        tag[id(inst["sbj_traj"])] = (pi, min(k0, L - 1), min(k1, L), names.index(inst["pre_cls"]), veto)
                    r += L
                for f in format_(args, cands):
                    pi, k0, k1, j, veto = tag[id(f["sub_traj"])]
                    outs["outputs"] += 1
                    hit = veto[k0:k1, j]
                    outs["affected"] += bool(hit.any()); outs["removed"] += bool(hit.all())
            res.setdefault(task, {})[kind] = dict(
                videos=len(vids), surviving_axes=surv, **dict(cnt),
                share_deleted=cnt["deleted"] / max(cnt["top20_novel"], 1),
                incremental_share=cnt["incremental"] / max(cnt["top20_novel"], 1),
                type_rule_share=cnt["type_rule_deleted"] / max(cnt["top20_novel"], 1),
                final_outputs=dict(outs), top10_incremental=per_pred.most_common(10))
            print(f"  {task} {kind}: deleted {cnt['deleted']}/{cnt['top20_novel']}, incremental "
                  f"{cnt['incremental'] / max(cnt['top20_novel'], 1):.4f}; outputs affected {outs['affected']}/{outs['outputs']}", flush=True)
    inc = res["sgdet"]["network"]["incremental_share"]
    res["gate_p3"] = dict(rule="SGDet incremental reach >= 3%", incremental=inc, verdict="PASS" if inc >= 0.03 else "FAIL",
                          projection_rough_guide=0.36 * inc / 0.076)
    json.dump(res, open(os.path.join(out, "step4.json"), "w"), indent=1, default=str)
    print(json.dumps(res["gate_p3"], indent=1))
    return [os.path.join(out, "step4.json")]


T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=(1, 2, 3, 4), required=True)
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    C.verify_frozen()
    out = out_dir(a.small)
    if a.step == 4:
        s3 = json.load(open(os.path.join(out, "step3.json")))
        if not s3["network"]["surviving_axes"] and not s3["untrained"]["surviving_axes"]:
            sys.exit("no axis survived G-P1 / G-P2: step 4 is not run (A18)")
    paths = {1: step1, 2: step2, 3: step3, 4: step4}[a.step](a, out)
    P.write_log(f"p_step{a.step}" + ("_small" if a.small else ""), dict(step=a.step, small=a.small),
                [C.CACHE_SUMS, C.SPLIT, C.PHI], paths, extra=dict(seconds=round(time.time() - T0, 1)))
    print(f"done | {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
