"""Stage 2, Batch A -- shared definitions (amendment A6 of stage1b/preregistration.md).

Paths, the frozen-artefact check, recipes, part scores and learned-part targets.
Every rule here is the one A6 fixes; the step scripts only call it.

Ground rules (A6): base labels from the TRAINING annotations only (novel
instances dropped at parse, counted, never stored); no test annotation, test
cache file or test evaluation is opened; the repo is imported, never edited.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib, json, math, os, platform, re, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PA = os.path.join(ROOT, "pilot_analysis")
for sub in ("stage0", "stage1", "stage1b", "stage1c"):
    sys.path.insert(0, os.path.join(PA, sub))
sys.path.insert(0, ROOT)
import measured_axes as M   # noqa: E402  (frozen)
import slot_axes as S       # noqa: E402  (frozen)
import stage0 as S0         # noqa: E402  (tokenise / SPATIAL / VERB: the stage-0e compositional test)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PREREG = os.path.join(PA, "stage1b", "preregistration.md")
PREREG_SHA = os.path.join(PA, "stage1b", "preregistration.sha256")
PHI = os.path.join(PA, "stage0e", "phi_frozen.json")
SCORING_RULE = os.path.join(PA, "stage0e", "scoring_rule.md")
SUBSET_PAIRS = os.path.join(PA, "stage0e", "subset_pairs_frozen.csv")
SPLIT = os.path.join(PA, "stage1", "train_split.json")
TH_SLOT = os.path.join(PA, "stage1b", "thresholds_slot.json")
TRACKS_TRAIN = os.path.join(PA, "stage1c", "gt_tracks_train.json")
ANNO_TRAIN = os.path.join(ROOT, "data", "vidvrd", "anno", "train")
ECC_TRAIN = os.path.join(ROOT, "data", "vidvrd", "data", "VidVRD_ECC_train.json")
PRED_SPLIT_INFO = os.path.join(ROOT, "data", "vidvrd", "data", "openvoc_pred_class_spilt_info.json")
BOX_COPY = os.environ.get("OVV_BOX_COPY", os.path.abspath(os.path.join(ROOT, "..", "session3_box_copy")))
CACHE = os.path.join(BOX_COPY, "cache_s3")
CACHE_SUMS = os.path.join(BOX_COPY, "SHA256SUMS_session3.txt")
PARAPHRASES = os.path.join(PA, "stage1b", "paraphrases_frozen.json")
ECC_TEST = os.path.join(ROOT, "data", "vidvrd", "data", "VidVRD_ECC_test.json")      # shipped file (A8)
TRACKS_TEST = os.path.join(PA, "stage1c", "gt_tracks_test.json")
TEST_GT = os.path.join(ROOT, "data", "vidvrd", "data", "test_relation_gt.json")      # read only by the evaluator
OUT = HERE

# ---------------------------------------------------------------------------
# A6 constants
# ---------------------------------------------------------------------------
EPS = 0.01                                   # floor on every part score
VEHICLE = {"airplane", "bicycle", "bus", "car", "motorcycle", "train", "watercraft"}   # scoring_rule.md
SUPPORT_MIN = 50                             # stage-0 LEARNED support threshold (base instances)
G1_MIN_HELD, G1_AUC = 30, 0.75
G2_RATIO, G2_MARGIN, G2_RANDOM_SEEDS = 3.0, 5.0, 20
MIX_WEIGHTS = (0.25, 0.5, 0.75)              # weight on the recipe score; 0.5 is the fixed mix
SEEDS = (0, 1, 2)
ES_SEED, ES_FRAC = 20261005, 0.10            # early-stop videos drawn from FIT
POST = dict(clip_len=30, clip_top_n=20, use_prior=False, max_per_video=200)

# frozen windows: preregistration table + A1 (D2 at W3)
WIN = dict(horizontal="W0", vertical="W0", depth="W1", contained="W3", subject_state="W2",
           object_state="W0", larger="W3", taller="W3", faster="W1", D2="W3", D1="W0", gap="W3")

LEARNED_AXES = ("locomotion", "posture", "contact", "contact_action")
# learned values (A2, A3): every value of the four learned axes, plus pass and adjacent
LEARNED_EXTRA = {("relative_motion", "pass"), ("proximity", "adjacent")}
MEASURED_VALUES = {
    ("horizontal", "left"), ("horizontal", "right"), ("vertical", "above"), ("vertical", "beneath"),
    ("depth", "front"), ("depth", "behind"), ("proximity", "contained"),
    ("subject_state", "moving"), ("subject_state", "stationary"), ("object_state", "moving"),
    ("relative_motion", "approach"), ("relative_motion", "recede"), ("relative_motion", "co_move"),
    ("comparative", "larger"), ("comparative", "taller"), ("comparative", "faster")}
SUPPLIED_VALUES = {("subject_kind", "vehicle")}
# learned outputs of the network (A6): values with >= SUPPORT_MIN base instances AND a
# negative class. contact has no negative class in base data (nothing asserts its
# absence) and bite/feed/hold/kick have 0 base instances: not outputs, see untrainable().
OUTPUTS = (("locomotion", "creep"), ("locomotion", "fly"), ("locomotion", "jump"), ("locomotion", "none"),
           ("locomotion", "run"), ("locomotion", "swim"), ("locomotion", "walk"),
           ("posture", "lie"), ("posture", "sit"), ("posture", "stand"),
           ("contact_action", "ride"), ("contact_action", "touch"),
           ("relative_motion", "pass"), ("proximity", "adjacent"))
OUT_IDX = {v: i for i, v in enumerate(OUTPUTS)}
LOCO_VALUES = [v for a, v in OUTPUTS if a == "locomotion"]
POSTURE_VALUES = [v for a, v in OUTPUTS if a == "posture"]

# A2/A3 extra inputs to the learned part (raw scalars; transformed in train step)
EXTRA_FEATS = ("d1_first", "d1_last", "d1_midmin", "d1_dd", "d1_sub_motion", "d1_obj_motion", "d1_dir_cos",
               "d2_mean_cos", "d2_first_cos", "d2_last_cos", "d2_sub_motion", "d2_obj_motion", "d2_dir_cos",
               "gap_w3")
# measured part columns stored per slot (value or NaN = cannot be evaluated)
MEAS_COLS = ("horizontal=left", "horizontal=right", "vertical=above", "vertical=beneath",
             "depth=front", "depth=behind", "proximity=contained",
             "subject_state=moving", "subject_state=stationary", "object_state=moving",
             "relative_motion=approach", "relative_motion=recede", "relative_motion=co_move",
             "comparative=larger", "comparative=taller", "comparative=faster")


# ---------------------------------------------------------------------------
# hashing, logging
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_info():
    import importlib
    out = dict(python=sys.version.split()[0], platform=platform.platform(), host=platform.node())
    for m in ("numpy", "torch", "sklearn"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            pass
    try:
        import subprocess
        out["git_head"] = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        out["git_dirty_files"] = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"], text=True).splitlines()
    except Exception:
        pass
    return out


def write_log(name, config, inputs, outputs, extra=None):
    """logs/<name>.json: command, config, seeds, input and output hashes, environment."""
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    rec = dict(step=name, time=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(sys.argv),
               config=config, env=env_info(),
               code={os.path.relpath(p, ROOT): sha256(p) for p in sorted(
                   os.path.join(HERE, f) for f in os.listdir(HERE) if f.endswith(".py"))},
               inputs={os.path.relpath(p, ROOT) if p.startswith(ROOT) else p: sha256(p) for p in inputs},
               outputs={os.path.relpath(p, ROOT): sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    path = os.path.join(OUT, "logs", f"{name}.json")
    json.dump(rec, open(path, "w"), indent=1)
    return path


def verify_frozen():
    """Every artefact pinned in preregistration.md (original table, A5, A6 if present)
    must match; the file itself must match preregistration.sha256. Exits on mismatch."""
    bad, rows = [], []
    want = open(PREREG_SHA).read().split()[0]
    got = sha256(PREREG)
    rows.append(("pilot_analysis/stage1b/preregistration.md", want, got))
    if want != got:
        bad.append("preregistration.md")
    pins = {}                       # a later amendment's pin of a path supersedes an earlier one (A12)
    for line in open(PREREG, encoding="utf-8"):
        m = re.match(r"\|[^|]*\|\s*`([^`]+)`[^|]*\|\s*`([0-9a-f]{64})`\s*\|", line)
        if m:
            pins[m.group(1)] = m.group(2)
    for rel, h in pins.items():
        path = os.path.join(ROOT, rel)
        g = sha256(path) if os.path.exists(path) else "MISSING"
        rows.append((rel, h, g))
        if g != h:
            bad.append(rel)
    if bad:
        for r in rows:
            print(("OK  " if r[1] == r[2] else "BAD ") + r[0])
        sys.exit(f"!! frozen artefact mismatch: {bad} -- STOP")
    return rows


def verify_cache(videos):
    """The cache files used must match the session-3 checksum list."""
    sums = {}
    for line in open(CACHE_SUMS):
        h, p = line.split(None, 1)
        sums[p.strip()] = h
    bad = []
    for v in videos:
        rel = f"cache_s3/train/{v}.pt"
        if sha256(os.path.join(BOX_COPY, rel)) != sums.get(rel):
            bad.append(rel)
    if bad:
        sys.exit(f"!! cache checksum mismatch: {bad[:5]} ({len(bad)}) -- STOP")
    return sha256(CACHE_SUMS)


# ---------------------------------------------------------------------------
# schema, splits, labels
# ---------------------------------------------------------------------------
def load_phi():
    raw = json.load(open(PHI))
    phi = {p: {k: v for k, v in d.items() if not k.startswith("_")} for p, d in raw.items()}
    split = {p: d["_split"] for p, d in raw.items()}
    info = json.load(open(PRED_SPLIT_INFO))["cls2split"]
    assert all(info[p] == split[p] for p in split) and len(split) == 132
    return phi, split


def compositional(p):
    """Stage 0e's test (stage0e.py:218-221): a bare spatial term, or verb + spatial term."""
    t = S0.tokenise(p)
    return (len(t) == 1 and t[0] in S0.SPATIAL) or (len(t) == 2 and t[0] in S0.VERB)


def kind(axis, value):
    if (axis, value) in SUPPLIED_VALUES:
        return "supplied"
    if axis in LEARNED_AXES or (axis, value) in LEARNED_EXTRA:
        return "learned"
    assert (axis, value) in MEASURED_VALUES, (axis, value)
    return "measured"


def untrainable():
    """Learned values that are not network outputs, fixed by A6 before any training:
    contact (no negative class in base data) and contact_action bite/feed/hold/kick
    (0 base instances, below the stage-0 threshold)."""
    return {("contact", "contact"), ("contact_action", "bite"), ("contact_action", "feed"),
            ("contact_action", "hold"), ("contact_action", "kick")}


def effective_recipe(p, phi, dropped=frozenset()):
    """A6 recipe of p: the parts phi sets, minus untrainable / G1-dropped learned
    values, plus the absence factors of the scoring rule (compositional predicates
    only, and only for an axis phi leaves unset). Dropping a set value does NOT
    trigger the absence rule. -> tuple of factors, each ('part', axis, value) or
    ('absent', axis, None)."""
    gone = set(untrainable()) | set(dropped)
    f = [("part", a, v) for a, v in sorted(phi[p].items()) if (a, v) not in gone]
    if compositional(p):
        if "locomotion" not in phi[p] and ("locomotion", "none") not in gone:
            f.append(("absent", "locomotion", None))
        if "posture" not in phi[p] and any(("posture", v) not in gone for v in POSTURE_VALUES):
            f.append(("absent", "posture", None))
    return tuple(f)


def needs_learned_part(recipe):
    return any(t == "part" and kind(a, v) == "learned" for t, a, v in recipe)


def consulted_outputs(recipe):
    """Network outputs a recipe reads (set learned parts, and the outputs behind
    each absence factor)."""
    out = set()
    for t, a, v in recipe:
        if t == "part" and kind(a, v) == "learned":
            out.add((a, v))
        elif t == "absent" and a == "locomotion":
            out.add(("locomotion", "none"))
        elif t == "absent" and a == "posture":
            out |= {("posture", x) for x in POSTURE_VALUES}
    return out


def instance_values(p, phi):
    """Learned outputs one base instance of p makes POSITIVE (explicit, or the
    implied locomotion=none of the absence rule)."""
    out = {(a, v) for a, v in phi[p].items() if (a, v) in OUT_IDX}
    if compositional(p) and "locomotion" not in phi[p]:
        out.add(("locomotion", "none"))
    return out


def load_train_video(path, base):
    """-> (video, width, height, base rels, n novel dropped). Novel instances are
    counted and dropped here; nothing downstream sees them."""
    d = json.load(open(path))
    rels = [r for r in d["relation_instances"] if r["predicate"] in base]
    n_novel = len(d["relation_instances"]) - len(rels)
    del d["relation_instances"]
    return d["video_id"], d["width"], d["height"], rels, n_novel, d


def splits():
    """FIT / HELD with the videos lacking ECC removed (as in stage 1), and the
    fixed early-stop subset of FIT."""
    sp = json.load(open(SPLIT))
    ecc_v = set(json.load(open(ECC_TRAIN)).keys())
    fit = sorted(v for v in sp["fit"] if v in ecc_v)
    held = sorted(v for v in sp["held"] if v in ecc_v)
    perm = np.random.default_rng(ES_SEED).permutation(len(fit))
    es = sorted(fit[i] for i in perm[:round(ES_FRAC * len(fit))])
    return dict(fit=fit, held=held, es=es, fit_train=sorted(set(fit) - set(es)),
                no_ecc=sorted((set(sp["fit"]) | set(sp["held"])) - ecc_v))


# ---------------------------------------------------------------------------
# learned-part targets (A6)
# ---------------------------------------------------------------------------
def targets(labels, phi):
    """-> (y, m): float arrays over OUTPUTS; m = 1 where the slot's labels say
    something about that output's axis, else masked.

    locomotion / posture: explicit values are positive; with none, a compositional
    label implies locomotion=none / no posture. Explicit beats implied. The other
    values of an observed axis are negative.
    contact_action: explicit values positive, the other trainable value negative.
    relative_motion=pass: observed when any label sets relative_motion.
    proximity=adjacent: positive when a label asserts adjacent; negative when no
    label sets proximity but some label sets horizontal, vertical or depth (the
    stage-1 rule that A3's evidence used, axis_check.md A3 / slot_check.md)."""
    y = np.zeros(len(OUTPUTS), np.float32)
    m = np.zeros(len(OUTPUTS), np.float32)
    for axis, values in (("locomotion", LOCO_VALUES), ("posture", POSTURE_VALUES)):
        explicit = {phi[p][axis] for p in labels if axis in phi[p]}
        implied = any(compositional(p) and axis not in phi[p] for p in labels)
        if explicit:
            pos = explicit
        elif implied:
            pos = {"none"} if axis == "locomotion" else set()
        else:
            continue
        for v in values:
            m[OUT_IDX[(axis, v)]] = 1
            y[OUT_IDX[(axis, v)]] = float(v in pos)
    ca = {phi[p]["contact_action"] for p in labels if "contact_action" in phi[p]}
    if ca:
        for v in ("ride", "touch"):
            m[OUT_IDX[("contact_action", v)]] = 1
            y[OUT_IDX[("contact_action", v)]] = float(v in ca)
    rm = {phi[p]["relative_motion"] for p in labels if "relative_motion" in phi[p]}
    if rm:
        m[OUT_IDX[("relative_motion", "pass")]] = 1
        y[OUT_IDX[("relative_motion", "pass")]] = float("pass" in rm)
    prox = {phi[p]["proximity"] for p in labels if "proximity" in phi[p]}
    spatial = any(a in phi[p] for p in labels for a in ("horizontal", "vertical", "depth"))
    if "adjacent" in prox or (not prox and spatial):
        m[OUT_IDX[("proximity", "adjacent")]] = 1
        y[OUT_IDX[("proximity", "adjacent")]] = float("adjacent" in prox)
    return y, m


# ---------------------------------------------------------------------------
# measured and supplied part scores (A6)
# ---------------------------------------------------------------------------
def _frac(cond):
    return float(np.count_nonzero(cond)) / len(cond) if len(cond) else float("nan")


def frame_fracs(p):
    """Fraction of the window's frames where each per-frame condition holds.
    Exact integer counts, so swap and time reversal map them exactly."""
    sx, ox = p.sub[:, 0] + p.sub[:, 2], p.obj[:, 0] + p.obj[:, 2]
    sy, oy = p.sub[:, 1] + p.sub[:, 3], p.obj[:, 1] + p.obj[:, 3]
    sa, oa = M._area(p.sub), M._area(p.obj)
    sh = np.maximum(p.sub[:, 3] - p.sub[:, 1], 1.0)
    oh = np.maximum(p.obj[:, 3] - p.obj[:, 1], 1.0)
    ix = np.maximum(0.0, np.minimum(p.sub[:, 2], p.obj[:, 2]) - np.maximum(p.sub[:, 0], p.obj[:, 0]))
    iy = np.maximum(0.0, np.minimum(p.sub[:, 3], p.obj[:, 3]) - np.maximum(p.sub[:, 1], p.obj[:, 1]))
    inter = ix * iy
    return dict(left=_frac(sx < ox), right=_frac(sx > ox), above=_frac(sy < oy), beneath=_frac(sy > oy),
                front=_frac(p.sub[:, 3] > p.obj[:, 3]), behind=_frac(p.sub[:, 3] < p.obj[:, 3]),
                contained=_frac((inter / sa > inter / oa) & (p.sub[:, 1] >= p.obj[:, 1])),
                larger=_frac(sa > oa), smaller=_frac(sa < oa), taller=_frac(sh > oh), shorter=_frac(sh < oh))


def load_thresholds():
    th = json.load(open(TH_SLOT))["per_window"]
    t = dict(motion_w2=th["W2"]["motion"], motion_w0=th["W0"]["motion"],
             motion_w3=th["W3"]["motion"], c_w3=th["W3"]["c"], dead_w0=th["W0"]["dead_zone"])
    assert abs(t["c_w3"] - 0.77) < 1e-12 and abs(t["motion_w3"] - 0.0041003) < 1e-7   # A1
    assert abs(t["motion_w2"] - 0.004471) < 1e-6 and abs(t["motion_w0"] - 0.004467) < 1e-6
    return t


def state01(x, thr):
    """window-level 0/1 of a motion rule; NaN when it cannot be evaluated."""
    return float("nan") if x != x else float(x > thr)


def window_pairs(frames, s, o, slot, ov, ecc_v, W, H, shared=None):
    """-> pair(w) giving (M.Pair, Heading) for window w of this slot. `shared`
    (one dict per ordered pair) reuses W3, which is the same for every slot."""
    cache = {}

    def pair(w):
        store = shared if (w == "W3" and shared is not None) else cache
        if w not in store:
            lo, hi = S.window(slot, ov, w)
            p = M.build_pair(frames, s, o, lo, hi, ecc_v, W, H)
            h = S.build_heading(frames, s, o, lo, hi, ecc_v) if p is not None else None
            store[w] = (p, h)
        return store[w]
    return pair


def slot_measures(pair, th):
    """pair: from window_pairs. -> (measured part dict over MEAS_COLS, extra-feature
    dict over EXTRA_FEATS). Values are fractions (per-frame rules) or 0/1
    (window-level rules); NaN = the rule cannot be evaluated in this slot (that
    factor is then left out)."""
    nan = float("nan")
    out = {k: nan for k in MEAS_COLS}
    for key, w, names in (("horizontal", WIN["horizontal"], ("left", "right")),
                          ("vertical", WIN["vertical"], ("above", "beneath")),
                          ("depth", WIN["depth"], ("front", "behind"))):
        p, _ = pair(w)
        if p is not None:
            f = frame_fracs(p)
            for n in names:
                out[f"{key}={n}"] = f[n]
    p, _ = pair(WIN["contained"])
    if p is not None:
        out["proximity=contained"] = frame_fracs(p)["contained"]
    for c in ("larger", "taller"):
        p, _ = pair(WIN[c])
        if p is not None:
            out[f"comparative={c}"] = frame_fracs(p)[c]
    p, _ = pair(WIN["subject_state"])
    if p is not None:
        mv = state01(M.subject_motion(p), th["motion_w2"])
        out["subject_state=moving"], out["subject_state=stationary"] = mv, (1.0 - mv if mv == mv else nan)
    p, _ = pair(WIN["object_state"])
    if p is not None:
        out["object_state=moving"] = state01(M.object_motion(p), th["motion_w0"])
    p, _ = pair(WIN["faster"])
    if p is not None:
        x = M.faster_score(p)
        out["comparative=faster"] = nan if (x != x or x == 0) else float(x > 0)
    p, h = pair(WIN["D2"])
    ex = {k: nan for k in EXTRA_FEATS}
    if p is not None:
        st = S.d2_stats(p, h)
        if st["sub_motion"] == st["sub_motion"]:
            r = S.d2_decide(st, th["c_w3"], th["motion_w3"])
            out["relative_motion=approach"] = float(r == "toward")
            out["relative_motion=recede"] = float(r == "away")
            out["relative_motion=co_move"] = float(r == "co_move")
        ex.update(d2_mean_cos=st["mean_cos"], d2_first_cos=st["first_cos"], d2_last_cos=st["last_cos"],
                  d2_sub_motion=st["sub_motion"], d2_obj_motion=st["obj_motion"], d2_dir_cos=st["dir_cos"],
                  gap_w3=M.gap_score(p))
    p, _ = pair(WIN["D1"])
    if p is not None:
        st = S.d1_stats(p)
        ex.update(d1_first=st["first"], d1_last=st["last"], d1_midmin=st["midmin"], d1_dd=st["last"] - st["first"],
                  d1_sub_motion=st["sub_motion"], d1_obj_motion=st["obj_motion"], d1_dir_cos=st["dir_cos"])
    return out, ex


# ---------------------------------------------------------------------------
# recipe scores (A6)
# ---------------------------------------------------------------------------
def recipe_matrix(recipes, names, meas, vehicle, learned, fill=True):
    """Recipe score for every (slot, predicate): geometric mean of the factor
    scores; a predicate with no recipe (or no evaluable factor in the slot) gets
    the slot's mean over predicates that have one (fill=False leaves NaN there).
    meas: (N, len(MEAS_COLS)); vehicle: (N,); learned: (N, len(OUTPUTS)) with NaN
    in columns that are dropped -> (N, len(names))."""
    N = len(vehicle)
    R = np.full((N, len(names)), np.nan)
    mcol = {c: i for i, c in enumerate(MEAS_COLS)}
    for j, p in enumerate(names):
        rec = recipes[p]
        if not rec:
            continue
        logs, cnt = np.zeros(N), np.zeros(N)
        for t, a, v in rec:
            if t == "absent" and a == "locomotion":
                x = learned[:, OUT_IDX[("locomotion", "none")]]
            elif t == "absent":
                ks = [OUT_IDX[("posture", q)] for q in POSTURE_VALUES]
                x = 1.0 - np.nanmax(learned[:, ks], axis=1)
            elif kind(a, v) == "measured":
                x = meas[:, mcol[f"{a}={v}"]]
            elif kind(a, v) == "supplied":
                x = vehicle.astype(float)
            else:
                x = learned[:, OUT_IDX[(a, v)]]
            ok = ~np.isnan(x)
            logs[ok] += np.log(np.clip(x[ok], EPS, 1.0))
            cnt[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            R[:, j] = np.where(cnt > 0, np.exp(logs / np.maximum(cnt, 1)), np.nan)
    return fill_neutral(R) if fill else R


def fill_neutral(R):
    """NaN (no recipe / nothing evaluable) -> the slot's mean over predicates that have one."""
    neutral = np.nanmean(R, axis=1)
    return np.where(np.isnan(R), neutral[:, None], R)


def coverage(R_fit):
    """A7: c(p) = mean recipe score over all FIT slots (unfilled R; NaN = no recipe)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(R_fit, axis=0) if len(R_fit) else np.full(R_fit.shape[1], np.nan)


def corrected(R, c, alpha):
    """A7: r' = r / c^alpha on predicates with a recipe, then the neutral fill."""
    return fill_neutral(R / np.power(c, alpha)[None, :])


def collision_groups(recipes):
    g = {}
    for p, r in recipes.items():
        if r:
            g.setdefault(tuple(sorted(r)), []).append(p)
    return sorted(sorted(v) for v in g.values() if len(v) > 1)
