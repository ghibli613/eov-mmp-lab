#!/usr/bin/env python3
"""Feasibility batch F1, Part A (A13) -- a compositional "who does what" prior
P(predicate | subject category, object category).

    python pilot_analysis/f1/f1_prior.py --small     # 5 videos each of fit-train, early-stop, HELD
    python pilot_analysis/f1/f1_prior.py             # all

Training videos only; base labels only. No test file. Each predicate is a verb-type
component (a locomotion/posture/vehicle verb, "none" for a bare spatial term, or the
predicate itself when it does not split) and a spatial component ("none" for an
atomic predicate):
    logit(p | a, b) = <W e(a), V_v> + c_v + <MLP([W e(a), W e(b)]), S_s> + d_s
with e = the frozen CLIP ViT-L/14@336px text embedding modelC uses for object
categories ("An image of {name}.", session 3 once.pt). Three versions: full,
minus-A6, minus-LO2 (the hidden labels removed, and the hidden predicates out of the
training softmax). Components with no carrier in a version's training data are
neutral: the mean of the carried components' vectors and biases.
Gate A1 (each hidden set, its minus-version): HELD, only the hidden 10 compete,
prior-only mAP >= 2 x random and >= the verb-frequency baseline + 2.
Gate A2 (full version): the veto's false-veto rate on HELD base GT instances <= 0.5%.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, copy, json, math, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "span"))
import span_common as SC     # noqa: E402  (slot table, logging)
from span_common import C    # noqa: E402  (stage 2: paths, frozen check, phi, tokeniser)
sys.path.insert(0, C.HERE)
import evaluation as EV      # noqa: E402  (the repo's post-processing and evaluator)

DIM, HID, LR, WD, BATCH, MAX_EP, PATIENCE, SEEDS = 64, 128, 1e-3, 1e-4, 256, 300, 20, (0, 1, 2)
VETO_P, VETO_MIN = 0.005, 100
G_RATIO, G_MARGIN, G_VETO = 2.0, 2.0, 0.005


# ---------------------------------------------------------------------------
# decomposition (A13)
# ---------------------------------------------------------------------------
def write_log(name, config, inputs, outputs, extra=None):
    """f1/logs/<name>.json: command, config, environment, code / input / output hashes."""
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rel = lambda p: os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
    rec = dict(step=name, time=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(sys.argv), config=config,
               env=C.env_info(), code={rel(os.path.join(HERE, f)): C.sha256(os.path.join(HERE, f))
                                       for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
               inputs={rel(p): C.sha256(p) for p in inputs}, outputs={rel(p): C.sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    json.dump(rec, open(os.path.join(HERE, "logs", f"{name}.json"), "w"), indent=1, default=str)


def decompose(p):
    """-> (verb-type component, spatial component)."""
    t = C.S0.tokenise(p)
    if len(t) == 2 and t[0] in C.S0.VERB:
        return t[0], t[1]                       # verb + spatial (incl. "with", "inside", "next_to")
    if len(t) == 1 and t[0] in C.S0.SPATIAL:
        return "none", t[0]                     # bare spatial term
    return p, "none"                            # atomic: its own verb-type component


def object_embeddings():
    """(35, 768) frozen CLIP text embeddings of the category names, by category name."""
    import torch
    rel = "cache_s3/once.pt"
    sums = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    assert C.sha256(os.path.join(C.BOX_COPY, rel)) == sums[rel]
    E = torch.load(os.path.join(C.BOX_COPY, rel), weights_only=False)["object_text"].float().numpy()
    id2cls = json.load(open(os.path.join(C.ROOT, "data", "vidvrd", "data", "openvoc_obj_class_spilt_info.json")))["id2cls"]
    cats = [id2cls[str(i)] for i in range(len(id2cls))]
    assert E.shape == (len(cats), 768)
    return E, cats


# ---------------------------------------------------------------------------
# model (A13)
# ---------------------------------------------------------------------------
def make_model(E, n_verb, n_spat):
    import torch
    nn = torch.nn

    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("E", torch.from_numpy(E))
            self.W = nn.Linear(768, DIM, bias=False)
            self.mlp = nn.Sequential(nn.Linear(2 * DIM, HID), nn.ReLU(), nn.Linear(HID, DIM))
            self.V = nn.Parameter(0.01 * torch.randn(n_verb, DIM))
            self.c = nn.Parameter(torch.zeros(n_verb))
            self.S = nn.Parameter(0.01 * torch.randn(n_spat, DIM))
            self.d = nn.Parameter(torch.zeros(n_spat))

        def forward(self, a, b, vi, si, v_carried, s_carried):
            """a, b: (B,) category ids; vi, si: (P,) component ids of the predicates scored;
            *_carried: bool masks -- a component without a carrier takes the carried mean."""
            V, c, S, d = self.V, self.c, self.S, self.d
            V = torch.where(v_carried[:, None], V, V[v_carried].mean(0, keepdim=True))
            c = torch.where(v_carried, c, c[v_carried].mean())
            S = torch.where(s_carried[:, None], S, S[s_carried].mean(0, keepdim=True))
            d = torch.where(s_carried, d, d[s_carried].mean())
            ea, eb = self.W(self.E[a]), self.W(self.E[b])
            h = self.mlp(torch.cat([ea, eb], -1))
            return ea @ V[vi].T + c[vi] + h @ S[si].T + d[si]
    return Prior()


def train_version(E, comp, train_preds, carried, data, seed, max_ep):
    """Soft-target cross-entropy (uniform over a slot's training labels), softmax over
    `train_preds`; Adam with weight decay; early stop on the early-stop slots."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_model(E, len(comp["verbs"]), len(comp["spats"]))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    vi = torch.tensor([comp["v_of"][p] for p in train_preds])
    si = torch.tensor([comp["s_of"][p] for p in train_preds])
    vc, sc = torch.tensor(carried["v"]), torch.tensor(carried["s"])

    def tensors(rows):
        a = torch.tensor([data["a"][r] for r in rows]); b = torch.tensor([data["b"][r] for r in rows])
        Y = torch.zeros(len(rows), len(train_preds))
        for k, r in enumerate(rows):
            ls = [train_preds.index(p) for p in data["labels"][r] if p in train_preds]
            Y[k, ls] = 1.0 / len(ls)
        return a, b, Y

    def ce(a, b, Y):
        return -(Y * torch.log_softmax(net(a, b, vi, si, vc, sc), -1)).sum(-1)

    tr, es = data["train_rows"], data["es_rows"]
    Ta, Tb, TY = tensors(tr)
    Ea, Eb, EY = tensors(es)
    best, best_ep, state, wait, curve = math.inf, -1, None, 0, []
    for ep in range(max_ep):
        net.train()
        order = rng.permutation(len(tr))
        for k in range(0, len(order), BATCH):
            i = torch.from_numpy(order[k:k + BATCH])
            l = ce(Ta[i], Tb[i], TY[i]).mean()
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(ce(Ea, Eb, EY).mean()) if len(es) else 0.0
        curve.append(v)
        if v < best - 1e-7:
            best, best_ep, state, wait = v, ep, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(state)
    return net, dict(seed=seed, epochs_run=ep + 1, best_epoch=best_ep + 1, best_es_ce=best, es_curve=curve)


def probs(nets, comp, carried, preds, a, b):
    """P(p | a, b), softmax over `preds`, mean of the seeds' probabilities. a, b: arrays of ids."""
    import torch
    vi = torch.tensor([comp["v_of"][p] for p in preds]); si = torch.tensor([comp["s_of"][p] for p in preds])
    vc, sc = torch.tensor(carried["v"]), torch.tensor(carried["s"])
    out = []
    with torch.no_grad():
        for net in nets:
            net.eval()
            out.append(torch.softmax(net(torch.as_tensor(a), torch.as_tensor(b), vi, si, vc, sc), -1).numpy())
    return np.mean(out, 0).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(6)
    t0 = time.time()
    C.verify_frozen()
    out_dir = os.path.join(HERE, "data_small" if args.small else "data")
    os.makedirs(out_dir, exist_ok=True)
    phi, split = C.load_phi()
    base = sorted(p for p in phi if split[p] == "base")
    names = EV.names()
    hidden = dict(A6=json.load(open(os.path.join(C.OUT, "step0_selection.json")))["leave_out"]["chosen"],
                  LO2=json.load(open(os.path.join(C.OUT, "a2_step0_selection.json")))["chosen"])
    rules = json.load(open(os.path.join(SC.HERE, "data", "s1_results.json")))["side_count"]["rules"]
    d = dict(np.load(os.path.join(SC.DATA2, "slots.npz")))
    meta = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))
    labels = [set(x) for x in meta["labels"]]
    assert all(l <= set(base) for l in labels)
    videos = meta["videos"]
    E, cats = object_embeddings()
    cid = {c: i for i, c in enumerate(cats)}
    if args.small:
        keep = set()
        for c_ in (0, 1, 2):
            keep |= set(sorted({int(v) for v in d["video"][d["subset"] == c_]})[:5])
        vmask = np.isin(d["video"], sorted(keep))
    else:
        vmask = np.ones(len(d["subset"]), bool)
    dm = dict(d); dm["video"] = np.where(vmask, d["video"], -1)
    sub_cat, obj_cat, _, pairs = SC.slot_table(dm, videos, set(base))
    rows = np.where(vmask)[0]
    A = np.array([cid[sub_cat[r]] if sub_cat[r] else -1 for r in range(len(d["subset"]))])
    B = np.array([cid[obj_cat[r]] if obj_cat[r] else -1 for r in range(len(d["subset"]))])

    # ---- decomposition
    comp_of = {p: decompose(p) for p in names}
    verbs = sorted({v for v, _ in comp_of.values()}); spats = sorted({s for _, s in comp_of.values()})
    comp = dict(verbs=verbs, spats=spats, v_of={p: verbs.index(comp_of[p][0]) for p in names},
                s_of={p: spats.index(comp_of[p][1]) for p in names})
    fit_rows = rows[d["subset"][rows] != 2]
    fit_inst = collections.Counter()           # FIT base instances, per predicate (annotations)
    fit_vids = sorted({int(v) for v in d["video"][fit_rows]})
    for vi_ in fit_vids:
        _, _, _, rels, _, _ = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{videos[vi_]}.json"), set(base))
        for r in rels:
            fit_inst[r["predicate"]] += 1

    results = dict(small=args.small, decomposition={p: dict(verb=v, spatial=s, split=split[p]) for p, (v, s) in comp_of.items()},
                   hidden=hidden, s1_rules=rules)
    versions = dict(full=set(), minus_A6=set(hidden["A6"]), minus_LO2=set(hidden["LO2"]))
    models, carried_of, train_info = {}, {}, {}
    for ver, H in versions.items():
        train_preds = [p for p in base if p not in H]
        lab = [l - H for l in labels]
        seen = set().union(*(lab[r] for r in fit_rows))          # a carrier has >= 1 FIT labelled slot
        carriers = [p for p in train_preds if p in seen]
        carried = dict(v=[any(comp_of[p][0] == v for p in carriers) for v in verbs],
                       s=[any(comp_of[p][1] == s for p in carriers) for s in spats])
        data = dict(a=A, b=B, labels=lab,
                    train_rows=[int(r) for r in rows if d["subset"][r] == 0 and lab[r]],
                    es_rows=[int(r) for r in rows if d["subset"][r] == 1 and lab[r]])
        nets, runs = [], []
        for s in SEEDS:
            net, info = train_version(E, comp, train_preds, carried, data, s, 10 if args.small else MAX_EP)
            nets.append(net); runs.append(info)
            print(f"  {ver} seed {s}: epochs {info['epochs_run']}, best {info['best_epoch']}, "
                  f"early-stop CE {info['best_es_ce']:.4f} | {time.time() - t0:.0f}s", flush=True)
        models[ver], carried_of[ver] = nets, carried
        train_info[ver] = dict(train_slots=len(data["train_rows"]), early_stop_slots=len(data["es_rows"]),
                               neutral_verbs=[v for v, k in zip(verbs, carried["v"]) if not k],
                               neutral_spatials=[s for s, k in zip(spats, carried["s"]) if not k],
                               seeds=runs)
    results["training"] = train_info

    # ---- Gate A1: HELD, only the hidden 10 compete
    held_pairs = json.load(open(os.path.join(SC.DATA2, "held_pairs.json")))
    gt_path = os.path.join(SC.DATA2, "held_gt.json")
    if args.small:
        keepv = {videos[v] for v in keep}
        held_pairs = {v: ps for v, ps in held_pairs.items() if v in keepv}
        gt = json.load(open(gt_path))
        gt_path = os.path.join(out_dir, "held_gt_small.json")
        json.dump({v: gt[v] for v in held_pairs}, open(gt_path, "w"))
    Hrows = np.concatenate([np.arange(*p["rows"]) for ps in held_pairs.values() for p in ps])
    N = len(d["subset"])
    gate1 = {}
    for hname in ("A6", "LO2"):
        Hs = hidden[hname]
        ver = f"minus_{hname}"
        cols = [names.index(p) for p in Hs]
        sp = EV.split_file(os.path.join(out_dir, f"split_{hname}.json"), set(Hs), "compete")
        run = lambda S_: EV.evaluate(EV.predict(held_pairs, EV.only(S_, cols)), gt_path, "compete", sp)
        P = probs(models[ver], comp, carried_of[ver], Hs, A[Hrows], B[Hrows])
        Sp = np.zeros((N, 132)); Sp[np.ix_(Hrows, cols)] = P
        prior = run(Sp)
        rnd = []
        for s in range(C.G2_RANDOM_SEEDS):
            U = np.zeros((N, 132)); U[Hrows] = np.random.default_rng(s).uniform(size=(len(Hrows), 132))
            rnd.append(run(U)["mAP"])
        vcount = collections.Counter()
        for p, n in fit_inst.items():
            if p not in Hs:
                vcount[comp_of[p][0]] += n
        tot = sum(vcount.values())
        Sv = np.zeros((N, 132))
        for j, p in zip(cols, Hs):
            Sv[Hrows, j] = vcount[comp_of[p][0]] / max(tot, 1)
        vf = run(Sv)
        rm = float(np.mean(rnd))
        ok = prior["mAP"] >= G_RATIO * rm and prior["mAP"] >= vf["mAP"] + G_MARGIN
        aucs = {}
        pos = {int(r): k for k, r in enumerate(Hrows)}
        lab_rows = [int(r) for r in Hrows if labels[r]]
        for k, p in enumerate(Hs):
            y = np.array([p in labels[r] for r in lab_rows])
            sc = P[[pos[r] for r in lab_rows], k]
            aucs[p] = dict(n_pos=int(y.sum()), auc=(SC.auc(y, sc) if 0 < y.sum() < len(y) else None))
        gate1[hname] = dict(hidden=Hs, prior=prior, random_mean_mAP=rm, random_sd=float(np.std(rnd)), verb_frequency=vf,
                            verb_frequency_values={p: vcount[comp_of[p][0]] / max(tot, 1) for p in Hs},
                            ratio=prior["mAP"] / max(rm, 1e-9), margin_over_verb_freq=prior["mAP"] - vf["mAP"],
                            per_predicate_auc=aucs, passed=bool(ok))
        print(f"Gate A1 {hname}: prior {prior['mAP']:.2f} | random {rm:.2f} | verb-freq {vf['mAP']:.2f} -> "
              f"{'pass' if ok else 'fail'}", flush=True)
    results["gate_A1"] = dict(rule=f"prior >= {G_RATIO} x random and >= verb-frequency + {G_MARGIN}, on both sets",
                              sets=gate1, verdict="PASS" if all(g["passed"] for g in gate1.values()) else "FAIL")

    # ---- Gate A2: veto safety (full prior)
    nets, car = models["full"], carried_of["full"]
    base_ids = base
    obj_w = collections.defaultdict(collections.Counter)      # FIT objects per subject, base-labelled slots
    for r in fit_rows:
        if labels[r]:
            obj_w[A[r]][B[r]] += 1
    vfit = collections.Counter()
    for p, n in fit_inst.items():
        vfit[comp_of[p][0]] += n
    Pva = {}
    for ai, a in enumerate(cats):
        bs = obj_w.get(ai)
        bl = list(bs) if bs else list(range(len(cats)))
        w = np.array([bs[x] for x in bl], float) if bs else np.ones(len(bl))
        w /= w.sum()
        Pp = probs(nets, comp, car, base_ids, np.full(len(bl), ai), np.array(bl))   # (len(bl), 71)
        marg = (w[:, None] * Pp).sum(0)
        Pva[a] = {v: float(sum(marg[k] for k, p in enumerate(base_ids) if comp_of[p][0] == v)) for v in verbs}
    vetoable = {v for v in verbs if vfit[v] >= VETO_MIN}
    veto = lambda a, p: comp_of[p][0] in vetoable and Pva[a][comp_of[p][0]] < VETO_P
    gt = json.load(open(gt_path))
    inst = [(r["triplet"][0], r["triplet"][1]) for g in gt.values() for r in g]
    fv = sum(veto(a, p) for a, p in inst)
    nov_cols = np.array([i for i, p in enumerate(names) if split[p] == "novel"])
    eov = 1 / (1 + np.exp(-d["logits"].astype(np.float64))) * (1 / (1 + np.exp(-d["int_logit"].astype(np.float64))))[:, None]
    n_top = n_vet = 0
    for r in rows:
        for i in np.argsort(-eov[r, nov_cols])[:20]:
            n_top += 1
            n_vet += veto(sub_cat[r], names[nov_cols[i]])
    vetoed_pairs = sorted((a, v) for a in cats for v in vetoable if Pva[a][v] < VETO_P)
    rule_agree = {v: dict(rule=t, vetoed_subjects=sorted(a for a, v2 in vetoed_pairs if v2 == v)) for v, t in rules.items()}
    results["gate_A2"] = dict(veto_rule=f"P(v|a) < {VETO_P} (full prior) and v has >= {VETO_MIN} FIT base instances",
                              vetoable_verbs=sorted(vetoable), vetoed_subject_verb_pairs=len(vetoed_pairs),
                              held_instances=len(inst), false_vetoes=int(fv), false_veto_rate=fv / max(len(inst), 1),
                              top20_novel_predictions=n_top, vetoed_top20_novel=n_vet,
                              vetoed_share=n_vet / max(n_top, 1), s1_rule_violation_share_reference=0.0838,
                              s1_rules_vs_veto=rule_agree,
                              verdict="PASS" if fv / max(len(inst), 1) <= G_VETO else "FAIL")
    print(f"Gate A2: false-veto rate {fv}/{len(inst)} = {fv / max(len(inst), 1):.4f}; "
          f"vetoed top-20 novel {n_vet}/{n_top} = {n_vet / max(n_top, 1):.4f}", flush=True)

    # ---- A5 descriptive
    subj_freq = collections.Counter(sub_cat[r] for r in fit_rows if labels[r])
    top_subj = [a for a, _ in subj_freq.most_common(10)]
    results["p_verb_given_subject"] = {a: {v: round(Pva[a][v], 4) for v in verbs if v in vetoable or Pva[a][v] >= 0.01}
                                       for a in top_subj}
    pair_freq = collections.Counter((sub_cat[r], obj_cat[r]) for r in fit_rows if labels[r])
    novel = [p for p in names if split[p] == "novel"]
    top5 = {}
    for (a, b), n in pair_freq.most_common(10):
        Pn = probs(nets, comp, car, novel, np.array([cid[a]]), np.array([cid[b]]))[0]
        top5[f"{a} -> {b}"] = [(novel[i], round(float(Pn[i]), 3)) for i in np.argsort(-Pn)[:5]]
    results["top5_novel_for_frequent_pairs"] = top5

    results["seconds"] = round(time.time() - t0, 1)
    paths = dict(results=os.path.join(out_dir, "f1_prior_results.json"), models=os.path.join(out_dir, "f1_prior_models.pt"))
    json.dump(results, open(paths["results"], "w"), indent=1, default=str)
    torch.save({v: dict(states=[n.state_dict() for n in ns], carried=carried_of[v], comp=comp) for v, ns in models.items()},
               paths["models"])
    write_log("f1_prior" + ("_small" if args.small else ""),
                 dict(small=args.small, dim=DIM, hidden=HID, lr=LR, weight_decay=WD, batch=BATCH, max_epochs=MAX_EP,
                      patience=PATIENCE, seeds=SEEDS, veto=dict(p=VETO_P, min_instances=VETO_MIN),
                      gates=dict(ratio=G_RATIO, margin=G_MARGIN, false_veto=G_VETO)),
                 [os.path.join(SC.DATA2, f) for f in ("slots.npz", "slots_meta.json", "held_pairs.json", "held_gt.json")]
                 + [C.TRACKS_TRAIN, os.path.join(C.BOX_COPY, "cache_s3/once.pt"), os.path.join(SC.HERE, "data", "s1_results.json")],
                 list(paths.values()),
                 extra=dict(seconds=results["seconds"]))
    print(json.dumps(dict(gate_A1=results["gate_A1"]["verdict"], gate_A2=results["gate_A2"]["verdict"],
                          seconds=results["seconds"]), indent=1))


if __name__ == "__main__":
    main()
