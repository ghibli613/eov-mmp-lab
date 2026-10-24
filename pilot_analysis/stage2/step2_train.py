#!/usr/bin/env python3
"""Stage 2 Batch A, step 2 -- train the learned part (A6), two versions, and gate G1.

    python pilot_analysis/stage2/step2_train.py --version full
    python pilot_analysis/stage2/step2_train.py --version leaveout
    python pilot_analysis/stage2/step2_train.py --version lo2         (A7: LO2 labels removed)
    (--data pilot_analysis/stage2/data_small --small   for the plumbing run)

CPU only. Inputs: step 1's slots.npz / emb.npy / slots_meta.json. Targets come
from each slot's BASE labels through phi_frozen.json (common.targets). The
leave-out version removes the 10 G2 predicates' labels (step0_selection.json)
from every slot before targets are built.

Writes into the data directory:
  learned_<version>.npy        (N, 14) float32: mean of the 3 seeds' sigmoid outputs, every slot
  model_<version>.pt           the 3 seeds' weights + the input normalisation
  train_<version>.json         per seed: epochs, best epoch, early-stop loss
  g1.json                      (full only) AUC per output on HELD, the bar, the drops
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, copy, json, os, time

import numpy as np

import common as C

HID, DROP, LR, BATCH, MAX_EP, PATIENCE = 512, 0.3, 1e-3, 256, 200, 10


def build_inputs(emb, extra, ref):
    """A6 input: embeddings | robust-standardised A2/A3 scalars | missing flags,
    then every column z-scored. Statistics from the rows in `ref` (all fit-train
    slots, labelled or not)."""
    med = np.nanmedian(extra[ref], axis=0)
    q75, q25 = np.nanpercentile(extra[ref], 75, axis=0), np.nanpercentile(extra[ref], 25, axis=0)
    iqr = np.where((q75 - q25) > 0, q75 - q25, 1.0)
    flags = np.isnan(extra).astype(np.float32)
    z = np.clip((extra - med) / iqr, -5, 5)
    z = np.where(np.isnan(z), 0.0, z).astype(np.float32)
    X = np.concatenate([emb.astype(np.float32), z, flags], axis=1)
    mu, sd = X[ref].mean(0), X[ref].std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    X = (X - mu) / sd
    return X.astype(np.float32), dict(extra_median=med, extra_iqr=iqr, mean=mu, std=sd)


def train_one(X, Y, Mk, tr, es, seed, small):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HID), torch.nn.ReLU(), torch.nn.Dropout(DROP),
                              torch.nn.Linear(HID, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    Xt, Yt, Mt = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Mk)

    def masked(idx):
        net.eval()
        with torch.no_grad():
            l = bce(net(Xt[idx]), Yt[idx]) * Mt[idx]
        return float(l.sum() / Mt[idx].sum())

    best, best_ep, best_state, wait, hist = float("inf"), -1, None, 0, []
    max_ep = 20 if (small and not len(es)) else MAX_EP
    for ep in range(max_ep):
        net.train()
        order = tr[rng.permutation(len(tr))]
        for b in range(0, len(order), BATCH):
            i = order[b:b + BATCH]
            l = (bce(net(Xt[i]), Yt[i]) * Mt[i]).sum() / Mt[i].sum().clamp(min=1)
            opt.zero_grad()
            l.backward()
            opt.step()
        if len(es):
            v = masked(es)
            hist.append(v)
            if v < best - 1e-6:
                best, best_ep, best_state, wait = v, ep, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        else:                                   # small plumbing run only: no early-stop slots
            best_ep, best_state = ep, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        P = torch.sigmoid(net(Xt)).numpy()
    return P, best_state, dict(seed=seed, epochs_run=ep + 1, best_epoch=best_ep + 1,
                               best_es_loss=best if len(es) else None, es_curve=hist)


def auc_ci(score, y, vids, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, score))
    uv = np.unique(vids)
    by = {u: np.where(vids == u)[0] for u in uv}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(uv, len(uv))])
        if 0 < y[idx].sum() < len(idx):
            bs.append(roc_auc_score(y[idx], score[idx]))
    return auc, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=("full", "leaveout", "lo2"), required=True)
    ap.add_argument("--data", default=os.path.join(C.OUT, "data"))
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(6)
    t0 = time.time()
    C.verify_frozen()
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    if a.version == "lo2":                          # A7: the second leave-out set
        leave = set(json.load(open(os.path.join(C.OUT, "a2_step0_selection.json")))["chosen"])
    else:
        leave = set(json.load(open(os.path.join(C.OUT, "step0_selection.json")))["leave_out"]["chosen"])

    d = np.load(os.path.join(a.data, "slots.npz"))
    E = np.load(os.path.join(a.data, "emb.npy"))
    meta = json.load(open(os.path.join(a.data, "slots_meta.json")))
    labels = [set(x) for x in meta["labels"]]
    assert all(l <= base for l in labels)
    removed = 0
    if a.version in ("leaveout", "lo2"):
        removed = sum(len(l & leave) for l in labels)
        labels = [l - leave for l in labels]
    N = len(labels)
    Y = np.zeros((N, len(C.OUTPUTS)), np.float32)
    Mk = np.zeros((N, len(C.OUTPUTS)), np.float32)
    for i, l in enumerate(labels):
        if l:
            Y[i], Mk[i] = C.targets(l, phi)
    sub = d["subset"]
    ref = np.where(sub == 0)[0]
    X, norm = build_inputs(E, d["extra"], ref)
    obs = Mk.sum(1) > 0
    tr = np.where((sub == 0) & obs)[0]
    es = np.where((sub == 1) & obs)[0]
    print(f"{a.version}: N={N} slots, X {X.shape}, train {len(tr)}, early-stop {len(es)}, "
          f"labels removed {removed}", flush=True)

    Ps, states, runs = [], [], []
    for s in C.SEEDS:
        P, st, info = train_one(X, Y, Mk, tr, es, s, a.small)
        Ps.append(P); states.append(st); runs.append(info)
        print(f"  seed {s}: epochs {info['epochs_run']}, best {info['best_epoch']}, "
              f"early-stop loss {info['best_es_loss']} | {time.time() - t0:.0f}s", flush=True)
    P = np.mean(Ps, 0).astype(np.float32)
    out_p = os.path.join(a.data, f"learned_{a.version}.npy")
    out_m = os.path.join(a.data, f"model_{a.version}.pt")
    out_t = os.path.join(a.data, f"train_{a.version}.json")
    np.save(out_p, P)
    torch.save(dict(states=states, norm=norm, outputs=C.OUTPUTS, arch=dict(hidden=HID, dropout=DROP)), out_m)
    pos = lambda rows: {f"{ax}={v}": int((Y[rows, j] * Mk[rows, j]).sum()) for j, (ax, v) in enumerate(C.OUTPUTS)}
    neg = lambda rows: {f"{ax}={v}": int(((1 - Y[rows, j]) * Mk[rows, j]).sum()) for j, (ax, v) in enumerate(C.OUTPUTS)}
    fit_rows, held_rows = np.where(sub != 2)[0], np.where(sub == 2)[0]
    json.dump(dict(version=a.version, labels_removed=removed, n_train=len(tr), n_early_stop=len(es),
                   hyper=dict(hidden=HID, dropout=DROP, lr=LR, batch=BATCH, max_epochs=MAX_EP, patience=PATIENCE,
                              seeds=C.SEEDS, optimizer="Adam"),
                   positives=dict(fit=pos(fit_rows), held=pos(held_rows)),
                   negatives=dict(fit=neg(fit_rows), held=neg(held_rows)), seeds=runs),
              open(out_t, "w"), indent=1)
    outputs = [out_p, out_m, out_t]

    if a.version == "full":
        vids = d["video"]
        rows, drops = [], []
        for j, (ax, v) in enumerate(C.OUTPUTS):
            k = held_rows[Mk[held_rows, j] > 0]
            y = Y[k, j]
            n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
            r = dict(output=f"{ax}={v}", fit_pos=pos(fit_rows)[f"{ax}={v}"], held_pos=n_pos, held_neg=n_neg,
                     gated=n_pos >= C.G1_MIN_HELD)
            if n_pos and n_neg:
                r["auc"], r["auc_ci95"] = auc_ci(P[k, j], y, vids[k])
                r["seed_auc"] = [float(__import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(y, Pi[k, j])) for Pi in Ps]
            else:
                r["auc"], r["auc_ci95"] = None, None
            r["verdict"] = ("PASS" if r["auc"] is not None and r["auc"] >= C.G1_AUC else "FAIL") if r["gated"] \
                else "untested (< 30 HELD positives): kept"
            if r["verdict"] == "FAIL":
                drops.append([ax, v])
            rows.append(r)
        g1 = dict(bar=dict(min_held_pos=C.G1_MIN_HELD, auc=C.G1_AUC), rows=rows, dropped=drops,
                  untrainable=[list(x) for x in sorted(C.untrainable())])
        out_g = os.path.join(a.data, "g1.json")
        json.dump(g1, open(out_g, "w"), indent=1)
        outputs.append(out_g)
        print("\nG1 (HELD):")
        for r in rows:
            ci = f"({r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f})" if r["auc_ci95"] else ""
            print(f"  {r['output']:24s} FIT+ {r['fit_pos']:5d} HELD+ {r['held_pos']:4d} HELD- {r['held_neg']:5d} "
                  f"AUC {r['auc'] if r['auc'] is None else round(r['auc'], 3)} {ci} {r['verdict']}")
        print("dropped:", drops)
    C.write_log(f"step2_train_{a.version}" + ("_small" if a.small else ""),
                dict(version=a.version, hidden=HID, dropout=DROP, lr=LR, batch=BATCH, max_epochs=MAX_EP,
                     patience=PATIENCE, seeds=C.SEEDS, torch_threads=6),
                [os.path.join(a.data, f) for f in ("slots.npz", "emb.npy", "slots_meta.json")]
                + [os.path.join(C.OUT, "a2_step0_selection.json" if a.version == "lo2" else "step0_selection.json"), C.PHI],
                outputs,
                extra=dict(seconds=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
