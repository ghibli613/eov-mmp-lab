"""Position verifier (A18) -- shared definitions: covered values, box features, the
untrained rule, targets, the swap-symmetric network, training, thresholds, veto,
box jitter.

Training videos: base labels only (novel dropped at parse). Test videos: model
inputs only (cached pair trajectories and per-slot scores); no test relation label,
no test evaluation.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import copy, json, math, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "span"))
import span_common as SC     # noqa: E402
from span_common import C    # noqa: E402

CLIP = 30
AXES = ("horizontal", "vertical", "depth", "proximity")
POS_VALUE = dict(horizontal="left", vertical="above", depth="front", proximity="adjacent")   # value with P = sigmoid(z)
NEG_VALUE = dict(horizontal="right", vertical="beneath", depth="behind")                     # P = 1 - sigmoid(z)
COVERED = [("horizontal", "left"), ("horizontal", "right"), ("vertical", "above"), ("vertical", "beneath"),
           ("depth", "front"), ("depth", "behind"), ("proximity", "adjacent")]
MIN_FRAMES = 5
N_FEAT = 15
HID, LR, BATCH, MAX_EP, PATIENCE, SEEDS = 64, 1e-3, 256, 100, 10, (0, 1, 2)
FOLDS, FOLD_SEED, TAU_RATE = 5, 20261007, 0.0025
GAP_W3 = 0.19752156642033697          # frozen adjacency gap threshold at W3 (thresholds_slot.json)
SUMS = None


# ---------------------------------------------------------------------------
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


def load_cache(rel):
    import torch
    global SUMS
    if SUMS is None:
        SUMS = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    f = os.path.join(C.BOX_COPY, rel)
    assert C.sha256(f) == SUMS[rel], rel
    return torch.load(f, weights_only=False, map_location="cpu")


def covered_predicates(phi):
    """value -> sorted predicates that set it (phi_frozen.json)."""
    out = {f"{a}={v}": sorted(p for p, d in phi.items() if d.get(a) == v) for a, v in COVERED}
    return out


def value_of(phi, p):
    """the covered (axis, value) predicate p sets, or None (no predicate sets two)."""
    hits = [(a, v) for a, v in COVERED if phi[p].get(a) == v]
    assert len(hits) <= 1, p
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# features (A18): per frame where both boxes exist
# ---------------------------------------------------------------------------
SWAP_NEG = (0, 1, 2, 3, 4, 5, 6, 7)          # negate under subject/object swap
SWAP_SAME = (8, 11, 12)                       # unchanged
SWAP_PAIRS = ((9, 10), (13, 14))              # exchanged


def frame_features(s, o, W, H):
    """s, o: (n, 4) boxes -> (n, 15) features and (n,) mask of frames with both boxes."""
    s, o = np.asarray(s, np.float64).reshape(-1, 4), np.asarray(o, np.float64).reshape(-1, 4)
    ok = np.isfinite(s).all(1) & np.isfinite(o).all(1)
    ws, hs = np.maximum(s[:, 2] - s[:, 0], 1.0), np.maximum(s[:, 3] - s[:, 1], 1.0)
    wo, ho = np.maximum(o[:, 2] - o[:, 0], 1.0), np.maximum(o[:, 3] - o[:, 1], 1.0)
    D = (np.hypot(ws, hs) + np.hypot(wo, ho)) / 2
    ix = np.maximum(0.0, np.minimum(s[:, 2], o[:, 2]) - np.maximum(s[:, 0], o[:, 0]))
    iy = np.maximum(0.0, np.minimum(s[:, 3], o[:, 3]) - np.maximum(s[:, 1], o[:, 1]))
    inter = ix * iy
    a_s, a_o = ws * hs, wo * ho
    ux = np.maximum(np.maximum(s[:, 2], o[:, 2]) - np.minimum(s[:, 0], o[:, 0]), 1.0)
    uy = np.maximum(np.maximum(s[:, 3], o[:, 3]) - np.minimum(s[:, 1], o[:, 1]), 1.0)
    F = np.stack([((s[:, 0] + s[:, 2]) - (o[:, 0] + o[:, 2])) / 2 / D,       # centre x offset
                  ((s[:, 1] + s[:, 3]) - (o[:, 1] + o[:, 3])) / 2 / D,       # centre y offset
                  (s[:, 0] - o[:, 0]) / D, (s[:, 2] - o[:, 2]) / D,          # left, right edge offsets
                  (s[:, 1] - o[:, 1]) / D, (s[:, 3] - o[:, 3]) / D,          # top, bottom edge offsets
                  np.log(ws) - np.log(wo), np.log(hs) - np.log(ho),          # log width / height ratios
                  inter / (a_s + a_o - inter),                              # IoU
                  inter / a_s, inter / a_o,                                 # intersection over each area
                  ix / ux, iy / uy,                                         # horizontal / vertical overlap fractions
                  s[:, 3] / H, o[:, 3] / H], 1)                             # bottom-edge heights
    return np.where(ok[:, None], F, 0.0), ok


def swap_features(F):
    G = F.copy()
    G[:, list(SWAP_NEG)] = -F[:, list(SWAP_NEG)]
    for a, b in SWAP_PAIRS:
        G[:, a], G[:, b] = F[:, b], F[:, a]
    return G


def untrained_scores(F, ok, gap_frac):
    """The frozen measured rules as frame fractions (Batch A): left = subject centre x <
    object's, above = centre y smaller, front = subject bottom edge lower, adjacent =
    rectangle gap / mean diagonal < the frozen W3 threshold. F: features of the rule's
    window. -> dict value -> fraction (NaN if no frame)."""
    n = max(int(ok.sum()), 1)
    f = lambda c: float(np.count_nonzero(c & ok)) / n if ok.any() else float("nan")
    return dict(left=f(F[:, 0] < 0), right=f(F[:, 0] > 0), above=f(F[:, 1] < 0), beneath=f(F[:, 1] > 0),
                front=f(F[:, 5] > 0), behind=f(F[:, 5] < 0), adjacent=f(gap_frac < GAP_W3))


def gap_over_diag(s, o):
    s, o = np.asarray(s, np.float64).reshape(-1, 4), np.asarray(o, np.float64).reshape(-1, 4)
    dx = np.maximum(0.0, np.maximum(o[:, 0] - s[:, 2], s[:, 0] - o[:, 2]))
    dy = np.maximum(0.0, np.maximum(o[:, 1] - s[:, 3], s[:, 1] - o[:, 3]))
    d = (np.hypot(np.maximum(s[:, 2] - s[:, 0], 1), np.maximum(s[:, 3] - s[:, 1], 1)) +
         np.hypot(np.maximum(o[:, 2] - o[:, 0], 1), np.maximum(o[:, 3] - o[:, 1], 1))) / 2
    return np.hypot(dx, dy) / d


def pair_slots(n_frames):
    """slot frame ranges inside a pair's trajectory lists, as process_pred cuts them."""
    n = n_frames // CLIP + (1 if n_frames % CLIP >= 10 else 0)
    return [(k * CLIP, n_frames if k == n - 1 else (k + 1) * CLIP) for k in range(n)]


def windows_for(k, slots, w):
    """Batch A's windows in slot units: W0 the slot, W1 +/- 1 slot (clipped)."""
    if w == "W0":
        return slots[k]
    a, b = max(0, k - 1), min(len(slots) - 1, k + 1)
    return slots[a][0], slots[b][1]


# ---------------------------------------------------------------------------
# the network (A18): shared per-frame MLP, masked mean + max pooling, MLP, one output per axis
# ---------------------------------------------------------------------------
def make_model():
    import torch
    nn = torch.nn

    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame = nn.Sequential(nn.Linear(N_FEAT, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(2 * HID, HID), nn.ReLU(), nn.Linear(HID, len(AXES)))

        def forward(self, X, M):
            """X: (B, T, 15) normalised features; M: (B, T) mask -> (B, 4) g."""
            h = self.frame(X)
            m = M.unsqueeze(-1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = h.masked_fill(m == 0, -1e9).max(1).values
            mx = torch.where(m.sum(1) > 0, mx, torch.zeros_like(mx))
            return self.head(torch.cat([mean, mx], -1))

    class Verifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = G()

        def forward(self, Xso, Xos, M):
            """-> (B, 4) logits: h, v, d antisymmetric g(so) - g(os); proximity g(so) + g(os)."""
            a, b = self.g(Xso, M), self.g(Xos, M)
            return torch.cat([a[:, :3] - b[:, :3], a[:, 3:] + b[:, 3:]], -1)
    return Verifier()


def normalise(X, stats):
    """z-score per feature with fit-train statistics. The swap acts on raw features, so the
    statistics are made swap-symmetric (negated features: mean 0; exchanged features:
    pooled) -- normalise(swap(x)) = swap'(normalise(x)) exactly up to rounding."""
    return (X - stats["mean"]) / stats["std"]


def norm_stats(X, M):
    v = X[M > 0]
    vs = np.concatenate([v, swap_features(v)], 0)            # both orders: symmetric statistics
    mean, std = vs.mean(0), vs.std(0)
    mean[list(SWAP_NEG)] = 0.0
    return dict(mean=mean, std=np.where(std > 1e-8, std, 1.0))


def class_weights(Y, Mk):
    """per-axis balanced weights for positives and negatives."""
    w = {}
    for j in range(len(AXES)):
        y = Y[Mk[:, j] > 0, j]
        n, npos = len(y), y.sum()
        w[j] = (n / (2 * max(npos, 1)), n / (2 * max(n - npos, 1)))
    return w


def train(D, tr, es, seed, max_ep=MAX_EP, fixed_epochs=None):
    """D: dict(X (N,T,15), Xs (swapped), M (N,T), Y (N,4), Mk (N,4)); class-balanced BCE per
    axis on observed entries; early stop on `es` unless fixed_epochs is given."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_model()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    w = class_weights(D["Y"][tr], D["Mk"][tr])
    T = {k: torch.from_numpy(D[k].astype(np.float32)) for k in ("X", "Xs", "M", "Y", "Mk")}
    wpos = torch.tensor([w[j][0] for j in range(len(AXES))], dtype=torch.float32)
    wneg = torch.tensor([w[j][1] for j in range(len(AXES))], dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def loss(idx):
        z = net(T["X"][idx], T["Xs"][idx], T["M"][idx])
        y, mk = T["Y"][idx], T["Mk"][idx]
        wt = (y * wpos + (1 - y) * wneg) * mk
        return (bce(z, y) * wt).sum() / mk.sum().clamp(min=1.0)

    best, best_ep, state, wait, curve = math.inf, -1, None, 0, []
    n_ep = fixed_epochs if fixed_epochs else max_ep
    for ep in range(n_ep):
        net.train()
        order = tr[rng.permutation(len(tr))]
        for b in range(0, len(order), BATCH):
            i = torch.from_numpy(order[b:b + BATCH])
            l = loss(i)
            opt.zero_grad(); l.backward(); opt.step()
        if fixed_epochs:
            continue
        net.eval()
        with torch.no_grad():
            v = float(loss(torch.from_numpy(es)))
        curve.append(v)
        if v < best - 1e-7:
            best, best_ep, state, wait = v, ep, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if not fixed_epochs:
        net.load_state_dict(state)
    net.eval()
    return net, dict(seed=seed, epochs_run=(n_ep if fixed_epochs else ep + 1),
                     best_epoch=(fixed_epochs if fixed_epochs else best_ep + 1), best_es_loss=best, es_curve=curve)


def predict(nets, X, Xs, M):
    """-> (N, 4) mean over nets of sigmoid(z)."""
    import torch
    out = []
    with torch.no_grad():
        for net in nets:
            ps = []
            for b in range(0, len(X), 4096):
                ps.append(torch.sigmoid(net(torch.from_numpy(X[b:b + 4096].astype(np.float32)),
                                            torch.from_numpy(Xs[b:b + 4096].astype(np.float32)),
                                            torch.from_numpy(M[b:b + 4096].astype(np.float32)))).numpy())
            out.append(np.concatenate(ps, 0))
    return np.mean(out, 0).astype(np.float64)


def value_prob(P, axis, value):
    """P: (N, 4) sigmoid outputs -> P(value)."""
    j = AXES.index(axis)
    return P[:, j] if POS_VALUE[axis] == value else 1.0 - P[:, j]


def choose_tau(scores, rate=TAU_RATE):
    """largest tau with at most `rate` of the true-v instances below it (P(v) < tau)."""
    s = np.sort(np.asarray(scores, np.float64))
    k = int(math.floor(rate * len(s)))
    return float(s[k]) if len(s) else 0.0


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# box jitter for G-P2 (A18): smooth per-track offset + per-frame noise, one scale s
# ---------------------------------------------------------------------------
def jitter_draws(tracks, seed=0):
    """standard-normal draws per track: knots every 30 frames (smooth part) and per-frame noise,
    for the components cx, cy, log w, log h."""
    rng = np.random.default_rng(seed)
    out = {}
    for key in sorted(tracks):
        n = len(tracks[key])
        nk = n // CLIP + 2
        out[key] = (rng.standard_normal((nk, 4)), rng.standard_normal((n, 4)))
    return out


def apply_jitter(boxes, draws, s):
    b = np.asarray(boxes, np.float64)
    n = len(b)
    knots, per = draws
    t = np.arange(n) / CLIP
    smooth = np.stack([np.interp(t, np.arange(len(knots)), knots[:, c]) for c in range(4)], 1)
    off = s * smooth + 0.5 * s * per[:n]
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    cx, cy = (b[:, 0] + b[:, 2]) / 2 + off[:, 0] * w, (b[:, 1] + b[:, 3]) / 2 + off[:, 1] * h
    w2, h2 = w * np.exp(off[:, 2]), h * np.exp(off[:, 3])
    return np.stack([cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2], 1)


def box_iou(a, b):
    ix = np.maximum(0.0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
    iy = np.maximum(0.0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
    inter = ix * iy
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)
