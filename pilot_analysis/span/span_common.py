"""Span idea, Batch S1 (A10) -- shared definitions: continuation targets, slot inputs,
the span head, training, metrics.

Training videos only, base labels only. No test file is opened; no mAP is computed
(training GT is 30-frame chunks, pilot_analysis/span/s1_precheck*.md).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import copy, json, math, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STAGE2 = os.path.join(ROOT, "pilot_analysis", "stage2")
sys.path.insert(0, STAGE2)
import common as C          # noqa: E402  (stage 2: paths, frozen check, phi, splits, tokeniser)

DATA2 = os.path.join(STAGE2, "data")
OUT = os.path.join(HERE, "data")
SUBSET = dict(fit_train=0, es=1, held=2)
PCA_DIM, D_MODEL, N_HEAD, N_LAYER, FF, DROPOUT, MAX_SLOTS = 256, 128, 4, 2, 256, 0.2, 40
LR, BATCH, MAX_EP, PATIENCE, SEEDS = 1e-3, 64, 100, 10, (0, 1, 2)
THETAS = tuple(round(0.01 * k, 2) for k in range(1, 100))
N_BOOT, BOOT_SEED = 1000, 0
ANIMATE = {"antelope", "bear", "bird", "cattle", "dog", "domestic_cat", "elephant", "fox", "giant_panda", "hamster",
           "horse", "lion", "lizard", "monkey", "person", "rabbit", "red_panda", "sheep", "snake", "squirrel", "tiger",
           "turtle", "whale", "zebra"}           # scoring_rule.md


# ---------------------------------------------------------------------------
# targets (A10)
# ---------------------------------------------------------------------------
def transition_target(a, b):
    """continue (0): both slots labelled, identical base sets; change (1): both
    labelled, sets differ; unknown (-1): either slot unlabelled."""
    if not a or not b:
        return -1
    return 0 if set(a) == set(b) else 1


# ---------------------------------------------------------------------------
# per-slot box features (A10): subject/object centre and size, and their gap,
# averaged over the slot's frames; the input is the change from the previous slot
# ---------------------------------------------------------------------------
def slot_box_stats(sub, obj, W, H):
    """sub, obj: (n, 4) boxes over the slot's frames -> 9 numbers."""
    sub, obj = np.asarray(sub, float), np.asarray(obj, float)

    def one(b):
        w = np.maximum(b[:, 2] - b[:, 0], 1.0)
        h = np.maximum(b[:, 3] - b[:, 1], 1.0)
        return [float(np.mean((b[:, 0] + b[:, 2]) / 2) / W), float(np.mean((b[:, 1] + b[:, 3]) / 2) / H),
                float(np.log(np.mean(w) / W)), float(np.log(np.mean(h) / H))]
    dx = np.maximum(0.0, np.maximum(obj[:, 0] - sub[:, 2], sub[:, 0] - obj[:, 2]))
    dy = np.maximum(0.0, np.maximum(obj[:, 1] - sub[:, 3], sub[:, 1] - obj[:, 3]))
    diag = (np.hypot(np.maximum(sub[:, 2] - sub[:, 0], 1), np.maximum(sub[:, 3] - sub[:, 1], 1)) +
            np.hypot(np.maximum(obj[:, 2] - obj[:, 0], 1), np.maximum(obj[:, 3] - obj[:, 1], 1))) / 2
    return one(sub) + one(obj) + [float(np.mean(np.hypot(dx, dy) / diag))]


def box_deltas(stats):
    """(L, 9) per-slot stats of one pair -> (L, 10): change from the previous slot
    (0 at the first slot) and a first-slot flag."""
    s = np.asarray(stats, float)
    d = np.zeros_like(s)
    d[1:] = s[1:] - s[:-1]
    first = np.zeros((len(s), 1))
    first[0] = 1.0
    return np.concatenate([d, first], 1)


def subject_type(cat):
    return "vehicle" if cat in C.VEHICLE else "animate" if cat in ANIMATE else "neither"


# ---------------------------------------------------------------------------
# slot table: the stage-2 grid, re-derived from the option-B tracks, row-aligned
# with stage2/data/slots.npz
# ---------------------------------------------------------------------------
def slot_table(d, videos, base):
    """-> per-row subject / object categories and box stats, plus per-pair row ranges.
    Asserts that the re-derived grid matches slots.npz row for row."""
    tracks = json.load(open(C.TRACKS_TRAIN))
    n = len(d["subset"])
    sub_cat, obj_cat = [None] * n, [None] * n
    stats = np.zeros((n, 9))
    pairs = []
    for vi, v in enumerate(videos):
        rows = np.where(d["video"] == vi)[0]
        if not len(rows):
            continue
        _, W, H, _, _, _ = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{v}.json"), base)   # novel dropped at parse
        T = {t["tid"]: t for t in tracks[v]}
        span = {t["tid"]: (t["begin_fid"], t["end_fid"]) for t in tracks[v]}
        grid = C.S.slot_grid(span)
        for pi, (s, o, ov, slots) in enumerate(grid):
            r = rows[d["pair"][rows] == pi]
            assert len(r) == len(slots) and (d["slot"][r] == np.arange(len(slots))).all(), (v, pi)
            assert (np.diff(r) == 1).all()
            for k, (a, b) in enumerate(slots):
                sb = [T[s]["boxes"][f - T[s]["begin_fid"]] for f in range(a, b)]
                ob = [T[o]["boxes"][f - T[o]["begin_fid"]] for f in range(a, b)]
                stats[r[k]] = slot_box_stats(sb, ob, W, H)
                sub_cat[r[k]], obj_cat[r[k]] = T[s]["category"], T[o]["category"]
            pairs.append((vi, pi, int(r[0]), int(r[-1]) + 1))
    return sub_cat, obj_cat, stats, pairs


# ---------------------------------------------------------------------------
# model (A10)
# ---------------------------------------------------------------------------
def make_model(d_in):
    import torch
    nn = torch.nn

    class SpanHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d_in, D_MODEL)
            self.pos = nn.Embedding(MAX_SLOTS, D_MODEL)
            layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, FF, DROPOUT, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, N_LAYER)
            self.cls = nn.Linear(3 * D_MODEL, 1)

        def forward(self, x, pad):
            """x: (B, L, d_in); pad: (B, L) True at padding -> (B, L-1) logits of P(change)."""
            L = x.shape[1]
            h = self.inp(x) + self.pos(torch.arange(L, device=x.device))[None]
            h = self.enc(h, src_key_padding_mask=pad)
            a, b = h[:, :-1], h[:, 1:]
            return self.cls(torch.cat([a, b, (a - b).abs()], -1)).squeeze(-1)
    return SpanHead()


def batches(seqs, idx):
    """seqs: list of (X (L, D), y (L-1,), m (L-1,)) -> padded torch tensors for idx."""
    import torch
    L = max(len(seqs[i][0]) for i in idx)
    D = seqs[idx[0]][0].shape[1]
    X = np.zeros((len(idx), L, D), np.float32)
    pad = np.ones((len(idx), L), bool)
    Y = np.zeros((len(idx), max(L - 1, 1)), np.float32)
    Mk = np.zeros((len(idx), max(L - 1, 1)), np.float32)
    for k, i in enumerate(idx):
        x, y, m = seqs[i]
        X[k, :len(x)] = x
        pad[k, :len(x)] = False
        Y[k, :len(y)] = y
        Mk[k, :len(m)] = m
    return torch.from_numpy(X), torch.from_numpy(pad), torch.from_numpy(Y[:, :L - 1]), torch.from_numpy(Mk[:, :L - 1])


def train(seqs, tr, es, seed, pos_weight, max_ep=MAX_EP):
    """BCE on known transitions (mask), positive-class weight; early stop on `es`."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = [i for i in tr if len(seqs[i][0]) >= 2]
    es = [i for i in es if len(seqs[i][0]) >= 2]
    net = make_model(seqs[tr[0]][0].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))

    def loss_on(idx):
        num = den = 0.0
        net.eval()
        with torch.no_grad():
            for b in range(0, len(idx), BATCH):
                X, pad, Y, Mk = batches(seqs, idx[b:b + BATCH])
                l = bce(net(X, pad), Y) * Mk
                num += float(l.sum()); den += float(Mk.sum())
        return num / max(den, 1.0)

    best, best_ep, state, wait, curve = math.inf, -1, None, 0, []
    for ep in range(max_ep):
        net.train()
        order = [tr[i] for i in rng.permutation(len(tr))]
        for b in range(0, len(order), BATCH):
            X, pad, Y, Mk = batches(seqs, order[b:b + BATCH])
            if Mk.sum() == 0:
                continue
            l = (bce(net(X, pad), Y) * Mk).sum() / Mk.sum()
            opt.zero_grad(); l.backward(); opt.step()
        v = loss_on(es) if es else 0.0
        curve.append(v)
        if v < best - 1e-6:
            best, best_ep, state, wait = v, ep, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(state)
    return net, dict(seed=seed, epochs_run=ep + 1, best_epoch=best_ep + 1, best_es_loss=best, es_curve=curve)


def predict(net, seqs):
    """-> list of (L-1,) P(change) arrays, one per sequence (empty for L < 2)."""
    import torch
    net.eval()
    out = [np.zeros(0)] * len(seqs)
    idx = [i for i in range(len(seqs)) if len(seqs[i][0]) >= 2]
    with torch.no_grad():
        for b in range(0, len(idx), BATCH):
            ib = idx[b:b + BATCH]
            X, pad, _, _ = batches(seqs, ib)
            P = torch.sigmoid(net(X, pad)).numpy()
            for k, i in enumerate(ib):
                out[i] = P[k, :len(seqs[i][0]) - 1].astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def auc_ci(y, s, vids, n_boot=N_BOOT, seed=BOOT_SEED):
    """AUC and a 95% percentile bootstrap CI that resamples whole videos."""
    y, s, vids = np.asarray(y), np.asarray(s), np.asarray(vids)
    uv = np.unique(vids)
    by = {u: np.where(vids == u)[0] for u in uv}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(uv, len(uv))])
        if 0 < y[idx].sum() < len(idx):
            bs.append(auc(y[idx], s[idx]))
    return auc(y, s), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def at_threshold(y, s, th):
    y, p = np.asarray(y).astype(bool), np.asarray(s) >= th
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(balanced_accuracy=float((rec + spec) / 2), precision=float(tp / max(tp + fp, 1)),
                recall=float(rec), specificity=float(spec), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def choose_theta(y, s):
    """A10: maximise balanced accuracy over THETAS; ties -> closest to 0.5."""
    rows = [(th, at_threshold(y, s, th)["balanced_accuracy"]) for th in THETAS]
    best = max(round(b, 12) for _, b in rows)
    return min((th for th, b in rows if round(b, 12) == best), key=lambda t: (abs(t - 0.5), t)), rows


def write_log(name, config, inputs, outputs, extra=None):
    """span/logs/<name>.json: command, config, environment, code / input / output hashes."""
    import time as _t
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rel = lambda p: os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
    rec = dict(step=name, time=_t.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(sys.argv), config=config,
               env=C.env_info(), code={rel(os.path.join(HERE, f)): C.sha256(os.path.join(HERE, f))
                                       for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
               inputs={rel(p): C.sha256(p) for p in inputs}, outputs={rel(p): C.sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    json.dump(rec, open(os.path.join(HERE, "logs", f"{name}.json"), "w"), indent=1, default=str)
