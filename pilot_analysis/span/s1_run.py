#!/usr/bin/env python3
"""Span idea, Batch S1 (A10) -- learn "relation continues vs changes" from training labels.

    python pilot_analysis/span/s1_run.py --small     # 5 videos each of fit-train, early-stop, HELD
    python pilot_analysis/span/s1_run.py             # all 783 training videos with ECC

CPU only; training videos only; base labels only (stage 2's slot labels). No test
file; no mAP. Targets per adjacent slot pair: continue (identical non-empty base
sets) / change (different non-empty sets) / unknown (either empty: masked).
Two versions: full, and LO2 (A7's 10 predicates removed from every slot's set,
with A7's LO2 learned part as the learned-part input).
Gate G-S1: full HELD AUC >= 0.80 with CI lower bound >= 0.75, and LO2 AUC >= 0.75
on HELD transitions where either slot carries an LO2 label (full-label targets).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, json, os, time

import numpy as np

import span_common as SC
from span_common import C

G_AUC, G_LO, G_LO2 = 0.80, 0.75, 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(6)
    t0 = time.time()
    C.verify_frozen()
    out_dir = os.path.join(SC.HERE, "data_small" if a.small else "data")
    os.makedirs(out_dir, exist_ok=True)
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    names = json.load(open(os.path.join(C.ROOT, "data", "vidvrd", "data", "id2predicate.json")))
    lo2 = set(json.load(open(os.path.join(C.OUT, "a2_step0_selection.json")))["chosen"])
    d = dict(np.load(os.path.join(SC.DATA2, "slots.npz")))
    meta = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))
    labels = [set(x) for x in meta["labels"]]
    assert all(l <= base for l in labels)                      # 0 novel labels used
    novel_labels_used = sum(len(l - base) for l in labels)
    videos = meta["videos"]
    vpe = np.load(os.path.join(SC.DATA2, "emb.npy"), mmap_mode="r")
    L_full = np.load(os.path.join(SC.DATA2, "learned_full.npy"))
    L_lo2 = np.load(os.path.join(SC.DATA2, "learned_lo2.npy"))
    if a.small:                                                 # first 5 videos of each subset
        keep = set()
        for c in (0, 1, 2):
            keep |= set(sorted({int(v) for v in d["video"][d["subset"] == c]})[:5])
        vmask = np.isin(d["video"], sorted(keep))
    else:
        vmask = np.ones(len(d["subset"]), bool)

    # ---- slot table and targets
    dm = dict(d)
    dm["video"] = np.where(vmask, d["video"], -1)          # rows outside the run are skipped
    sub_cat, obj_cat, stats, pairs = SC.slot_table(dm, videos, base)
    lab_lo2 = [l - lo2 for l in labels]
    trans = []                                                  # (pair id, t, row t, subset, video, y_full, y_lo2, lo2_touch)
    for pid, (vi, pi, r0, r1) in enumerate(pairs):
        for t in range(r1 - r0 - 1):
            r = r0 + t
            trans.append((pid, t, r, int(d["subset"][r]), vi, SC.transition_target(labels[r], labels[r + 1]),
                          SC.transition_target(lab_lo2[r], lab_lo2[r + 1]), bool((labels[r] | labels[r + 1]) & lo2)))
    T = {k: np.array([x[i] for x in trans]) for i, k in enumerate(("pid", "t", "row", "subset", "video", "y_full", "y_lo2", "lo2_touch"))}
    counts = {}
    for name, c in (("fit_train", 0), ("early_stop", 1), ("held", 2), ("all", None)):
        sel = np.ones(len(trans), bool) if c is None else T["subset"] == c
        for ver in ("full", "lo2"):
            y = T[f"y_{ver}"][sel]
            counts.setdefault(name, {})[ver] = dict(transitions=int(sel.sum()), known=int((y >= 0).sum()),
                                                    change=int((y == 1).sum()), continue_=int((y == 0).sum()),
                                                    change_share=float((y == 1).sum() / max((y >= 0).sum(), 1)))
    print("targets:", json.dumps({k: v["full"] for k, v in counts.items()}), flush=True)

    # ---- inputs: learned part | measured + supplied | PCA-256 of vpe | box deltas; z-scored on fit-train
    rows_all = np.concatenate([np.arange(r0, r1) for _, _, r0, r1 in pairs])
    fit_rows = rows_all[d["subset"][rows_all] != 2]
    V = np.asarray(vpe[fit_rows, :3072], np.float32)
    mu = V.mean(0)
    _, _, Vt = np.linalg.svd(V - mu, full_matrices=False)
    comps = Vt[:SC.PCA_DIM]
    del V
    pca = np.zeros((len(d["subset"]), SC.PCA_DIM), np.float32)
    for b in range(0, len(rows_all), 4096):
        rr = rows_all[b:b + 4096]
        pca[rr, :len(comps)] = (np.asarray(vpe[rr, :3072], np.float32) - mu) @ comps.T   # < 256 rows only in --small
    deltas = np.zeros((len(d["subset"]), 10))
    for _, _, r0, r1 in pairs:
        deltas[r0:r1] = SC.box_deltas(stats[r0:r1])
    meas = np.nan_to_num(d["meas"], nan=0.0)
    ftr = rows_all[d["subset"][rows_all] == 0]

    def inputs(L):
        X = np.concatenate([L.astype(np.float64), meas, d["vehicle"][:, None], pca.astype(np.float64), deltas], 1)
        m, s = X[ftr].mean(0), X[ftr].std(0)
        s = np.where(s > 1e-8, s, 1.0)
        return ((X - m) / s).astype(np.float32), dict(mean=m, std=s)

    results = dict(small=a.small, novel_labels_used=novel_labels_used, counts=counts, lo2=sorted(lo2),
                   inputs=dict(learned=14, measured=16, supplied=1, pca=SC.PCA_DIM, box_deltas=10))
    probs, models = {}, {}
    for ver, L in (("full", L_full), ("lo2", L_lo2)):
        X, norm = inputs(L)
        y_key = f"y_{ver}"
        seqs = []
        for pid, (vi, pi, r0, r1) in enumerate(pairs):
            sel = T["pid"] == pid
            y = T[y_key][sel]
            seqs.append((X[r0:r1], np.where(y >= 0, y, 0).astype(np.float32), (y >= 0).astype(np.float32)))
        sub_of = np.array([int(d["subset"][r0]) for _, _, r0, _ in pairs])
        tr, es = list(np.where(sub_of == 0)[0]), list(np.where(sub_of == 1)[0])
        ytr = T[y_key][(T["subset"] == 0) & (T[y_key] >= 0)]
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        P, runs, states = [], [], []
        for s in SC.SEEDS:
            net, info = SC.train(seqs, tr, es, s, pw, max_ep=5 if a.small else SC.MAX_EP)
            P.append(SC.predict(net, seqs)); runs.append(info); states.append(net.state_dict())
            print(f"  {ver} seed {s}: epochs {info['epochs_run']}, best {info['best_epoch']}, "
                  f"early-stop loss {info['best_es_loss']:.4f} | {time.time() - t0:.0f}s", flush=True)
        pr = np.zeros(len(trans))
        for pid in range(len(pairs)):
            sel = np.where(T["pid"] == pid)[0]
            if len(sel):
                pr[sel] = np.mean([Pi[pid] for Pi in P], 0)
        probs[ver] = pr
        models[ver] = dict(states=states, norm=norm, pos_weight=pw, runs=runs)
        results[f"train_{ver}"] = dict(pos_weight=pw, seeds=[{k: v for k, v in r.items() if k != "es_curve"} for r in runs])

    # ---- theta on the early-stop videos' known transitions (each version on its own targets)
    for ver in ("full", "lo2"):
        sel = (T["subset"] == 1) & (T[f"y_{ver}"] >= 0)
        th, grid = SC.choose_theta(T[f"y_{ver}"][sel], probs[ver][sel])
        results[f"theta_{ver}"] = dict(theta=th, early_stop_known=int(sel.sum()),
                                       auc_early_stop=SC.auc(T[f"y_{ver}"][sel], probs[ver][sel]),
                                       at_theta=SC.at_threshold(T[f"y_{ver}"][sel], probs[ver][sel], th),
                                       grid=[(t_, round(b, 4)) for t_, b in grid if round(t_ * 10, 6) % 1 == 0])
    th = results["theta_full"]["theta"]

    # ---- evaluation on HELD known transitions
    H = (T["subset"] == 2) & (T["y_full"] >= 0)
    yH, vH = T["y_full"][H], T["video"][H]
    a_full, ci = SC.auc_ci(yH, probs["full"][H], vH)
    ev = dict(full=dict(n=int(H.sum()), change=int(yH.sum()), auc=a_full, auc_ci95=ci,
                        at_theta=SC.at_threshold(yH, probs["full"][H], th)))
    HL = H & T["lo2_touch"]
    yL = T["y_full"][HL]
    a_lo2, ci_lo2 = SC.auc_ci(yL, probs["lo2"][HL], T["video"][HL])
    ev["lo2"] = dict(n=int(HL.sum()), change=int(yL.sum()), auc=a_lo2, auc_ci95=ci_lo2,
                     full_version_auc_same_transitions=SC.auc(yL, probs["full"][HL]))
    r, r1 = T["row"], T["row"] + 1
    meas_jump = np.abs(meas[r1] - meas[r]).sum(1)
    pn = pca / np.maximum(np.linalg.norm(pca, axis=1, keepdims=True), 1e-12)
    cos_dist = 1 - (pn[r] * pn[r1]).sum(1)
    ev["baselines"] = {k: dict(held=SC.auc(yH, s[H]), lo2_transitions=SC.auc(yL, s[HL]))
                       for k, s in (("sum_abs_change_measured_parts", meas_jump), ("cosine_distance_pca", cos_dist))}
    # (b) mechanism: on HELD "continue" transitions, does EOV-MMP's base top 20 keep the continuing predicates?
    eov = 1 / (1 + np.exp(-d["logits"].astype(np.float64))) * (1 / (1 + np.exp(-d["int_logit"].astype(np.float64))))[:, None]
    base_cols = np.array([i for i, p in enumerate(names) if split[p] == "base"])
    top = {}

    def top20(row):
        if row not in top:
            s_ = eov[row, base_cols]
            top[row] = {names[base_cols[i]] for i in np.argsort(-s_)[:20]}
        return top[row]
    n_cont = n_in_t = n_kept = n_pred = 0
    for k in np.where((T["subset"] == 2) & (T["y_full"] == 0))[0]:
        rt = int(T["row"][k])
        for p in labels[rt]:
            n_pred += 1
            if p in top20(rt):
                n_in_t += 1
                n_kept += p in top20(rt + 1)
        n_cont += 1
    ev["mechanism"] = dict(continue_transitions=n_cont, continuing_predicates=n_pred,
                           in_top20_at_t=n_in_t, share_in_top20_at_t=n_in_t / max(n_pred, 1),
                           kept_in_top20_at_t_plus_1=n_kept, share_kept=n_kept / max(n_in_t, 1))
    # (c) segment lengths on HELD at theta, cutting every transition (known or not)
    seg = []
    for pid, (vi, pi, r0, r1_) in enumerate(pairs):
        if d["subset"][r0] != 2:
            continue
        cuts = probs["full"][T["pid"] == pid] >= th
        run = 1
        for c_ in cuts:
            if c_:
                seg.append(run); run = 1
            else:
                run += 1
        seg.append(run)
    seg = np.array(seg)
    ev["segments_held"] = dict(n=int(len(seg)), mean_slots=float(seg.mean()), median=float(np.median(seg)),
                               p90=float(np.percentile(seg, 90)), share_1_slot=float((seg == 1).mean()),
                               hist={int(k): int(v) for k, v in sorted(collections.Counter(np.minimum(seg, 10).tolist()).items())})
    gate = dict(full_auc=a_full >= G_AUC, full_ci_low=ci[0] >= G_LO, lo2_auc=a_lo2 >= G_LO2)
    ev["gate"] = dict(rules=dict(full_auc=G_AUC, full_ci_low=G_LO, lo2_auc=G_LO2), checks=gate,
                      verdict="PASS" if all(gate.values()) else "FAIL")
    results["evaluation"] = ev

    # ---- side count (label-free on the novel side): EOV's top-20 novel predicates per slot vs subject-type rules
    verb = lambda p: (C.S0.tokenise(p)[0] if C.compositional(p) and len(C.S0.tokenise(p)) == 2 else None)
    vt = collections.defaultdict(collections.Counter)
    sel_videos = sorted({int(v) for v in d["video"][vmask]})
    for vi in sel_videos:
        _, _, _, rels, _, ann = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{videos[vi]}.json"), base)
        cat = {so["tid"]: so["category"] for so in ann["subject/objects"]}
        for rr in rels:
            if verb(rr["predicate"]):
                vt[verb(rr["predicate"])][SC.subject_type(cat[rr["subject_tid"]])] += 1
    rules = {}
    for v_, c_ in vt.items():
        n = sum(c_.values())
        for typ in ("vehicle", "animate"):
            if n >= 100 and c_[typ] / n >= 0.995:
                rules[v_] = typ
    nov_cols = np.array([i for i, p in enumerate(names) if split[p] == "novel"])
    per_verb = collections.defaultdict(lambda: [0, 0])
    n_top = 0
    for row in rows_all:
        st = SC.subject_type(sub_cat[row])
        for i in np.argsort(-eov[row, nov_cols])[:20]:
            n_top += 1
            v_ = verb(names[nov_cols[i]])
            if v_ in rules:
                per_verb[v_][0] += 1
                per_verb[v_][1] += st != rules[v_]
    tot_ruled = sum(x[0] for x in per_verb.values())
    tot_bad = sum(x[1] for x in per_verb.values())
    results["side_count"] = dict(videos=len(sel_videos), slots=int(len(rows_all)),
                                 verb_subject_types={k: dict(v) for k, v in sorted(vt.items())}, rules=rules,
                                 top20_novel_predictions=n_top, with_a_rule=tot_ruled, violating=tot_bad,
                                 share_violating_of_ruled=tot_bad / max(tot_ruled, 1),
                                 share_violating_of_all_top20=tot_bad / max(n_top, 1),
                                 per_verb={k: dict(predictions=v[0], violating=v[1]) for k, v in sorted(per_verb.items())})

    # ---- save
    results["seconds"] = round(time.time() - t0, 1)
    paths = dict(results=os.path.join(out_dir, "s1_results.json"), probs=os.path.join(out_dir, "s1_transitions.npz"),
                 model_full=os.path.join(out_dir, "span_full.pt"), model_lo2=os.path.join(out_dir, "span_lo2.pt"),
                 pca=os.path.join(out_dir, "pca_fit.npz"))
    json.dump(results, open(paths["results"], "w"), indent=1)
    np.savez(paths["probs"], **T, p_full=probs["full"], p_lo2=probs["lo2"])
    for ver in ("full", "lo2"):
        torch.save(models[ver], paths[f"model_{ver}"])
    np.savez(paths["pca"], mean=mu, components=comps)
    SC.write_log("s1_run" + ("_small" if a.small else ""),
                dict(small=a.small, pca=SC.PCA_DIM, d_model=SC.D_MODEL, heads=SC.N_HEAD, layers=SC.N_LAYER, ff=SC.FF,
                     dropout=SC.DROPOUT, lr=SC.LR, batch=SC.BATCH, max_epochs=SC.MAX_EP, patience=SC.PATIENCE,
                     seeds=SC.SEEDS, gate=dict(full_auc=G_AUC, full_ci_low=G_LO, lo2_auc=G_LO2)),
                [os.path.join(SC.DATA2, f) for f in ("slots.npz", "slots_meta.json", "emb.npy", "learned_full.npy",
                                                     "learned_lo2.npy")] + [C.TRACKS_TRAIN, os.path.join(C.OUT, "a2_step0_selection.json")],
                list(paths.values()), extra=dict(seconds=results["seconds"]))
    print(json.dumps(dict(theta=th, evaluation={k: v for k, v in ev.items()}, side=dict(
        rules=rules, share=results["side_count"]["share_violating_of_ruled"])), indent=1, default=str))


if __name__ == "__main__":
    main()
