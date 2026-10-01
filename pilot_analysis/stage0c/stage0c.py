#!/usr/bin/env python3
"""Stage 0c -- schema v3, ride/drive geometry, label co-occurrence, stop-as-halt.

    python3 pilot_analysis/stage0c/stage0c.py

Data only. Reads the repo's training annotations, split file and ECC camera
matrices; writes only into pilot_analysis/stage0c/. No checkpoint, no GPU.

Every empirical number here uses BASE-predicate instances from the TRAINING
annotations only. Novel-predicate instances (present in those files, masked from
the loss) are never read as evidence; test annotations are never opened.

Camera compensation: data/vidvrd/data/VidVRD_ECC_train.json holds one 3x3 warp per
frame, keyed by the 1-based frame index. For 0-based annotation frame t,
ecc[str(t+1)] maps frame t-1 coordinates into frame t (the convention
models/tracking/deep_sort/track.py:141 applies). This was checked against the
alternatives on 63k box steps: it gives the lowest residual; the inverse and the
off-by-one keys are worse. Videos without ECC matrices are excluded from the
speed measurements and counted.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv, glob, json, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage0b"))
import stage0 as s0   # noqa: E402
import stage0b as sb  # noqa: E402

ECC_FILE = "data/vidvrd/data/VidVRD_ECC_train.json"
THIN = sb.THIN
RNG = np.random.default_rng(0)
N_BOOT = 1000
MOV = 0.02          # net camera-compensated displacement, fraction of diagonal

# ---------------------------------------------------------------------------
# Task 1 -- schema v3
# ---------------------------------------------------------------------------
SCHEMA = dict(sb.SCHEMA)
SCHEMA["contact"] = ["contact", "none"]                 # M; unset = absent
AXES = list(SCHEMA)
TAG = dict(sb.TAG)
TAG["state_change"] = "LEARNED"                         # (b)
TAG["contact"] = "LEARNED"                              # (c)

WITH_STATIONARY = {"stand_with", "lie_with", "stop_with"}
RECIPES = {                                             # (d)
    "drive": {"proximity": "overlapping", "object_state": "moving",
              "relative_motion": "co_move", "contact": "contact"},
    "pull":  {"contact": "contact", "object_state": "moving",
              "relative_motion": "co_move"},
}


def phi_v3(p):
    d = sb.phi_b(p)
    if p in WITH_STATIONARY:                            # (a)
        d["proximity"] = "adjacent"
    if p in RECIPES:                                    # (d)
        d.update(RECIPES[p])
    if "contact_action" in d or p.endswith("_inside"):  # (c)
        d["contact"] = "contact"
    return d                                            # (e) fight: unchanged, {}


def fallback(phi, sup):
    """Coarse-channel view: a LEARNED contact_action below threshold is dropped
    when the predicate also carries a supported contact=contact."""
    out = {}
    for p, d in phi.items():
        d = dict(d)
        ca = d.get("contact_action")
        if ca and sup[("contact_action", ca)] < THIN and \
                sup[("contact", d.get("contact"))] >= THIN:
            d.pop("contact_action")
        out[p] = d
    return out


def reach(phi, sup, novel):
    unassigned = sorted(p for p in novel if not phi[p])
    thin = {p: [x for x in phi[p].items() if TAG[x[0]] == "LEARNED" and sup[x] < THIN]
            for p in novel}
    thin = {p: v for p, v in thin.items() if v}
    return sorted(set(unassigned) | set(thin)), unassigned, thin


def subset_pairs(phi):
    sets = {p: set(d.items()) for p, d in phi.items() if d}
    return {(a, b) for a in sets for b in sets if a != b and sets[a] < sets[b]}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def load_train():
    vids = {}
    for f in sorted(glob.glob(os.path.join(ROOT, s0.TRAIN_DIR, "*.json"))):
        d = json.load(open(f))
        d["_boxes"] = [{e["tid"]: e["bbox"] for e in fr} for fr in d["trajectories"]]
        vids[d["video_id"]] = d
    return vids


def ctr(b):
    return np.array([(b["xmin"] + b["xmax"]) / 2, (b["ymin"] + b["ymax"]) / 2, 1.0])


def warp(ecc_v, t):
    m = ecc_v.get(str(t + 1))
    if m is None:
        return None
    m = np.array(m)
    return m if np.linalg.norm(np.eye(3) - m) < 100 else np.eye(3)


def steps(d, ecc_v, tid, lo, hi):
    """Camera-compensated per-frame displacement vectors of `tid`'s box centre
    for transitions (t-1 -> t), t in [lo+1, hi). Returns (list of 2-vectors,
    n transitions possible). Normalised by the frame diagonal."""
    diag = np.hypot(d["width"], d["height"])
    B = d["_boxes"]
    out = []
    lo, hi = max(lo, 0), min(hi, len(B))
    for t in range(lo + 1, hi):
        b0, b1 = B[t - 1].get(tid), B[t].get(tid)
        if b0 is None or b1 is None:
            continue
        m = warp(ecc_v, t)
        if m is None:
            continue
        q = m @ ctr(b0)
        out.append((ctr(b1)[:2] - q[:2]) / diag)
    return out, max(hi - lo - 1, 0)


def mean_speed(st):
    return float(np.mean([np.linalg.norm(s) for s in st])) if st else float("nan")


def rankdata(x):
    x = np.asarray(x, float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(len(x), int)
    inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(x, y, vids):
    x, y, vids = np.asarray(x, float), np.asarray(y, int), np.asarray(vids)
    uv = np.unique(vids)
    idx = {v: np.where(vids == v)[0] for v in uv}
    bs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[v] for v in RNG.choice(uv, len(uv))])
        a = auc(x[pick], y[pick])
        if a == a:
            bs.append(a)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pct(a, b):
    return f"{a}/{b} ({a / b:.1%})" if b else f"{a}/0"


# ---------------------------------------------------------------------------
# Task 1(d) -- ride geometry
# ---------------------------------------------------------------------------
def ride_geometry(vids, ecc, base_set):
    rows, skipped_ecc = [], 0
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] != "ride":
                continue
            assert "ride" in base_set
            s, o = r["subject_tid"], r["object_tid"]
            b, e = r["begin_fid"], min(r["end_fid"], len(d["_boxes"]))
            cats, ios, top_above = Counter(), [], 0
            n = 0
            for t in range(b, e):
                bs, bo = d["_boxes"][t].get(s), d["_boxes"][t].get(o)
                if bs is None or bo is None:
                    continue
                n += 1
                ix = max(0, min(bs["xmax"], bo["xmax"]) - max(bs["xmin"], bo["xmin"]))
                iy = max(0, min(bs["ymax"], bo["ymax"]) - max(bs["ymin"], bo["ymin"]))
                a_s = max(bs["xmax"] - bs["xmin"], 1) * max(bs["ymax"] - bs["ymin"], 1)
                f = ix * iy / a_s                       # fraction of subject inside object
                ios.append(f)
                above = ctr(bs)[1] < ctr(bo)[1]
                top_above += bs["ymin"] < bo["ymin"]
                if f >= 0.9:
                    cats["contained (>=90% of subject box inside object box)"] += 1
                elif f > 0:
                    cats["overlapping, subject centre above" if above else
                         "overlapping, subject centre not above"] += 1
                else:
                    cats["disjoint, subject above" if above else "disjoint, other"] += 1
            if n < 2:
                continue
            ecc_v = ecc.get(v)
            if ecc_v is None:
                skipped_ecc += 1
                mv = None
            else:
                ss, _ = steps(d, ecc_v, s, b, e)
                so, _ = steps(d, ecc_v, o, b, e)
                ds = np.sum(ss, axis=0) if ss else np.zeros(2)
                do = np.sum(so, axis=0) if so else np.zeros(2)
                ns, no = np.linalg.norm(ds), np.linalg.norm(do)
                cos = float(ds @ do / (ns * no)) if ns > 0 and no > 0 else float("nan")
                # relative offset drift, normalised by the object box diagonal
                off = []
                for t in range(b, e):
                    bs, bo = d["_boxes"][t].get(s), d["_boxes"][t].get(o)
                    if bs and bo:
                        od = np.hypot(bo["xmax"] - bo["xmin"], bo["ymax"] - bo["ymin"])
                        off.append((ctr(bs)[:2] - ctr(bo)[:2]) / max(od, 1))
                drift = float(np.linalg.norm(off[-1] - off[0])) if len(off) > 1 else float("nan")
                mv = dict(s_net=ns, o_net=no, cos=cos, drift=drift)
            rows.append(dict(video=v, cat=cats.most_common(1)[0][0],
                             ios_median=float(np.median(ios)),
                             top_above=top_above / n, mv=mv))
    return rows, skipped_ecc


# ---------------------------------------------------------------------------
# Task 2 -- co-occurrence
# ---------------------------------------------------------------------------
def overlaps(a, b):
    return max(a["begin_fid"], b["begin_fid"]) < min(a["end_fid"], b["end_fid"])


def cooccurrence(vids, split):
    base = {p for p, s in split.items() if s == "base"}
    by_pair = defaultdict(list)
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] in base:
                by_pair[(v, r["subject_tid"], r["object_tid"])].append(r)

    def study(anchor, partners):
        res = dict(n=0, hit=0, any_other=0, per=Counter())
        for insts in by_pair.values():
            for r in insts:
                if r["predicate"] != anchor:
                    continue
                res["n"] += 1
                hits = {q["predicate"] for q in insts
                        if q is not r and q["predicate"] in partners and overlaps(r, q)}
                if hits:
                    res["hit"] += 1
                for h in hits:
                    res["per"][h] += 1
                if any(q is not r and q["predicate"] != anchor and overlaps(r, q)
                       for q in insts):
                    res["any_other"] += 1
        return res

    bare_rows, move_rows = [], []
    for bare in ("left", "right", "front", "behind"):
        comps = sorted(p for p in base
                       if s0.tokenise(p)[-1] == bare and len(s0.tokenise(p)) == 2)
        excluded = sorted(p for p in split if split[p] == "novel"
                          and s0.tokenise(p)[-1] == bare and len(s0.tokenise(p)) == 2)
        fwd = study(bare, set(comps))
        rev = Counter()
        rev_n = Counter()
        for c in comps:
            x = study(c, {bare})
            rev[c], rev_n[c] = x["hit"], x["n"]
        bare_rows.append(dict(anchor=bare, partners=comps, excluded=excluded,
                              fwd=fwd, rev=rev, rev_n=rev_n))
    manner = ["walk", "run", "fly", "swim", "creep", "jump"]
    for p in sorted(base):
        if sb.verb_of(p) != "move":
            continue
        sp = s0.tokenise(p)[1]
        partners = sorted(f"{m}_{sp}" for m in manner if f"{m}_{sp}" in base)
        excluded = sorted(f"{m}_{sp}" for m in manner if split.get(f"{m}_{sp}") == "novel")
        fwd = study(p, set(partners))
        rev, rev_n = Counter(), Counter()
        for c in partners:
            x = study(c, {p})
            rev[c], rev_n[c] = x["hit"], x["n"]
        move_rows.append(dict(anchor=p, partners=partners, excluded=excluded,
                              fwd=fwd, rev=rev, rev_n=rev_n))
    return bare_rows, move_rows


# ---------------------------------------------------------------------------
# Task 3 -- stop as halt
# ---------------------------------------------------------------------------
def halt_features(vids, ecc, preds):
    skipped = Counter()
    out = []
    for v, d in vids.items():
        ecc_v = ecc.get(v)
        for r in d["relation_instances"]:
            p = r["predicate"]
            if p not in preds:
                continue
            if ecc_v is None:
                skipped["no ECC for video"] += 1
                continue
            s = r["subject_tid"]
            b, e = r["begin_fid"], min(r["end_fid"], len(d["_boxes"]))
            L = e - b
            ext, _ = steps(d, ecc_v, s, b, e)
            if len(ext) < 6:
                skipped["< 6 tracked steps in extent"] += 1
                continue
            bef, possible = steps(d, ecc_v, s, b - L, b)
            k = len(ext) // 3
            first = mean_speed(ext[:k])
            last = mean_speed(ext[-k:])
            cat = None
            for sd in d["subject/objects"]:
                if sd["tid"] == s:
                    cat = sd["category"]
            out.append(dict(pred=p, video=v, cat=cat, L=L,
                            ext=mean_speed(ext), bef=mean_speed(bef) if len(bef) >= 3 else float("nan"),
                            bef_steps=len(bef), bef_possible=possible,
                            first=first, last=last))
    return out, skipped


# ---------------------------------------------------------------------------
def main():
    split, tr = sb.load()
    preds = sorted(split)
    base = [p for p in preds if split[p] == "base"]
    novel = [p for p in preds if split[p] == "novel"]
    base_set = set(base)

    # ======================= Task 1 =======================
    phi_b = {p: sb.phi_b(p) for p in preds}
    phi = {p: phi_v3(p) for p in preds}
    for p, d in phi.items():
        for a, v in d.items():
            assert v in SCHEMA[a], (p, a, v)
    json.dump({p: dict(phi[p], _split=split[p]) for p in preds},
              open(os.path.join(HERE, "phi_v3.json"), "w"), indent=1)

    sup, nb, masked = Counter(), Counter(), Counter()
    for p in preds:
        for x in phi[p].items():
            if split[p] == "base":
                sup[x] += tr[p]; nb[x] += 1
            else:
                masked[x] += tr[p]
    sup_b = Counter()
    for p in base:
        for x in phi_b[p].items():
            sup_b[x] += tr[p]

    unr_b = reach(phi_b, sup_b, novel)[0]
    unr_strict, unas, thin = reach(phi, sup, novel)
    phi_fb = fallback(phi, sup)
    unr_fb, unas_fb, thin_fb = reach(phi_fb, sup, novel)

    g_b = sb.collision_groups(phi_b, preds)
    g_v3 = sb.collision_groups(phi, preds)
    g_fb = sb.collision_groups(phi_fb, preds)
    key = lambda g: tuple(sorted(g))
    new_coll = [g for g in g_v3 if key(g) not in {key(x) for x in g_b}]
    new_coll_fb = [g for g in g_fb if key(g) not in {key(x) for x in g_v3}]

    sp_b, sp_v3, sp_fb = subset_pairs(phi_b), subset_pairs(phi), subset_pairs(phi_fb)
    nov_sub = lambda S: {(a, b) for a, b in S if split[a] == "novel"}
    new_ns = sorted(nov_sub(sp_v3) - nov_sub(sp_b))
    gone_ns = sorted(nov_sub(sp_b) - nov_sub(sp_v3))
    new_ns_fb = sorted(nov_sub(sp_fb) - nov_sub(sp_v3))
    with open(os.path.join(HERE, "subset_pairs_v3.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "subset_split", "superset", "superset_split",
                    "attributes_only_in_superset", "new_vs_stage0b"])
        sets = {p: set(phi[p].items()) for p in preds}
        for a, b in sorted(sp_v3):
            extra = ";".join(f"{k}={v}" for k, v in sorted(sets[b] - sets[a]))
            w.writerow([a, split[a], b, split[b], extra, (a, b) not in sp_b])

    print("loading training annotations ...", file=sys.stderr)
    vids = load_train()
    ecc = json.load(open(os.path.join(ROOT, ECC_FILE)))
    ride_rows, ride_noecc = ride_geometry(vids, ecc, base_set)

    # ======================= Task 2 =======================
    bare_rows, move_rows = cooccurrence(vids, split)

    # ======================= Task 3 =======================
    stop_p = sorted(p for p in base if sb.verb_of(p) == "stop")
    stand_p = sorted(f"stand_{s0.tokenise(p)[1]}" for p in stop_p
                     if f"stand_{s0.tokenise(p)[1]}" in base_set)
    loco_p = sorted(p for p in base if sb.verb_of(p) in ("walk", "run"))
    feats, skipped = halt_features(vids, ecc, set(stop_p) | set(stand_p) | set(loco_p))
    stop_f = [r for r in feats if r["pred"] in stop_p]
    stand_f = [r for r in feats if r["pred"] in stand_p]
    loco_f = [r for r in feats if r["pred"] in loco_p]
    tau = float(np.percentile([r["ext"] for r in loco_f], 25))   # "clearly moving"

    def halt_stats(R):
        wb = [r for r in R if r["bef"] == r["bef"]]
        before_exceeds = [r for r in wb if r["bef"] >= tau and r["bef"] >= 2 * r["ext"]]
        trans = [r for r in R if r["first"] >= tau and r["last"] <= 0.5 * r["first"]]
        return dict(n=len(R), n_before=len(wb),
                    before_exceeds=len(before_exceeds), transition=len(trans),
                    ext_med=float(np.median([r["ext"] for r in R])),
                    bef_med=float(np.median([r["bef"] for r in wb])) if wb else float("nan"),
                    ext_moving=sum(1 for r in R if r["ext"] >= tau))

    hs = dict(stop=halt_stats(stop_f), stand=halt_stats(stand_f), walk_run=halt_stats(loco_f))

    # category-matched control: stand instances restricted to stop's subject
    # categories, reweighted to stop's category mix by resampling
    stop_cats = Counter(r["cat"] for r in stop_f)
    stand_by_cat = defaultdict(list)
    for r in stand_f:
        stand_by_cat[r["cat"]].append(r)
    matched = []
    for c, n in stop_cats.items():
        pool = stand_by_cat.get(c, [])
        if pool:
            matched += [pool[i] for i in RNG.integers(0, len(pool), n)]
    hs["stand_cat_matched"] = halt_stats(matched) if matched else None
    cat_cov = sum(n for c, n in stop_cats.items() if stand_by_cat.get(c))

    # subject categories: every base stop_*/stand_*/sit_*/lie_* training instance
    vcat = defaultdict(Counter)
    for v, d in vids.items():
        c = {so["tid"]: so["category"] for so in d["subject/objects"]}
        for r in d["relation_instances"]:
            vb = sb.verb_of(r["predicate"])
            if r["predicate"] in base_set and vb in ("stop", "stand", "sit", "lie"):
                vcat[vb][c[r["subject_tid"]]] += 1
    vehicles = sorted(vcat["stop"])
    veh_in = {vb: sum(vcat[vb][c] for c in vehicles) for vb in ("stand", "sit", "lie")}

    # option (C) reversion, kept separate from phi_v3.json
    phi_r = {}
    for p in preds:
        d = dict(phi[p])
        if sb.verb_of(p) == "stop":
            d.pop("state_change", None)
            d.update(subject_state="stationary", locomotion="none")
        phi_r[p] = d
    json.dump({p: dict(phi_r[p], _split=split[p]) for p in preds},
              open(os.path.join(HERE, "phi_v3_stop_reverted.json"), "w"), indent=1)
    sp_r = subset_pairs(phi_r)
    stop_nov = sorted(p for p in novel if sb.verb_of(p) == "stop")
    stop_sub = {p: sorted(b for a, b in sp_r if a == p) for p in stop_nov}
    unr_r = reach(phi_r, sup, novel)[0]
    g_r = sb.collision_groups(phi_r, preds)
    new_coll_r = [g for g in g_r if tuple(sorted(g)) not in {tuple(sorted(x)) for x in g_v3}]

    comp = []
    both = stop_f + stand_f
    y = [1 if r["pred"] in stop_p else 0 for r in both]
    vv = [r["video"] for r in both]
    for name, fn in (
            ("extent speed (i)", lambda r: r["ext"]),
            ("before-window speed (ii)", lambda r: r["bef"]),
            ("log(before / extent)", lambda r: np.log((r["bef"] + 1e-6) / (r["ext"] + 1e-6))),
            ("first-third minus last-third speed", lambda r: r["first"] - r["last"])):
        xs = [fn(r) for r in both]
        keep = [i for i, x in enumerate(xs) if x == x]
        X = [xs[i] for i in keep]; Y = [y[i] for i in keep]; V = [vv[i] for i in keep]
        a = auc(X, Y); lo, hi = auc_ci(X, Y, V)
        comp.append((name, len(X), sum(Y), a, lo, hi))

    # ======================= write =======================
    fmt = lambda x: f"{x[0]}={x[1]}"
    T = ["# Stage 0c — Task 1: schema v3\n",
         "Base support = BASE-predicate instances in the 800 training videos; "
         "masked novel instances excluded.\n",
         "## Changes applied\n",
         "- (a) `stand_with`, `lie_with`, `stop_with`: + proximity=adjacent",
         "- (b) state_change re-tagged LEARNED",
         "- (c) new axis `contact` (LEARNED): contact=contact on every predicate with "
         "a contact_action (`hold`, `ride`, `bite`, `touch`, `feed`, `kick`), on the "
         "three `*_inside` predicates, and on the `drive`/`pull` recipes",
         "- (d) `drive` = " + ", ".join(fmt(x) for x in sorted(RECIPES["drive"].items()))
         + "; `pull` = " + ", ".join(fmt(x) for x in sorted(RECIPES["pull"].items())),
         "- (e) `fight` left unassigned\n",
         "## Base support, changed or new values\n",
         "| axis | tag | value | base train instances | base predicates | masked (not usable) |",
         "|---|---|---|---:|---:|---:|"]
    for x in (("contact", "contact"), ("contact", "none"), ("state_change", "halt"),
              ("proximity", "adjacent"), ("proximity", "overlapping"),
              ("relative_motion", "co_move"), ("object_state", "moving")):
        T.append(f"| {x[0]} | {TAG[x[0]]} | {x[1]} | {sup[x]} | {nb[x]} | {masked[x]} |")
    T.append("\ncontact=contact base support comes only from `ride` "
             f"({tr['ride']}) and `touch` ({tr['touch']}); every other carrier is novel.\n")

    T.append("## Reachability (<50 applied to LEARNED axes only)\n")
    T.append("| | stage 0b | v3, strict | v3, coarse contact fallback |")
    T.append("|---|---:|---:|---:|")
    T.append(f"| unreachable novel predicates | {len(unr_b)} | {len(unr_strict)} | {len(unr_fb)} |")
    T.append(f"\n- **v3 strict** (every LEARNED ingredient must have ≥{THIN}): "
             f"{len(unr_strict)} unreachable — " + ", ".join(
                 f"`{p}` ({'no ingredients' if p in unas else ', '.join(fmt(x) + f' ({sup[x]})' for x in thin[p])})"
                 for p in unr_strict))
    T.append(f"- **v3 fallback** (a contact_action below threshold is dropped in "
             f"favour of the supported coarse contact=contact): {len(unr_fb)} unreachable — "
             + ", ".join(f"`{p}`" for p in unr_fb))
    T.append(f"- `drive` and `pull` are reachable in both views: their only LEARNED "
             f"ingredient is contact=contact ({sup[('contact', 'contact')]}).\n")

    T.append("## Identical-phi collisions\n")
    kinds = lambda gs: {k: sum(1 for g in gs if sb.kind(g, split) == k)
                        for k in ("all-novel", "base-novel", "all-base")}
    kb, kv, kf = kinds(g_b), kinds(g_v3), kinds(g_fb)
    T.append("| | stage 0b | v3 | v3 fallback |")
    T.append("|---|---:|---:|---:|")
    for k in ("all-novel", "base-novel", "all-base"):
        T.append(f"| {k} | {kb[k]} | {kv[k]} | {kf[k]} |")
    T.append(f"\n**⚠ NEW collisions created by v3: {len(new_coll)}**\n")
    for g in new_coll:
        T.append(f"- {sb.kind(g, split)}: " + ", ".join(
            f"`{p}`" + ("*" if split[p] == "novel" else "") for p in g)
            + f" — phi {{{', '.join(fmt(x) for x in sorted(phi[g[0]].items()))}}}")
    if new_coll_fb:
        T.append(f"\n**⚠ Further collisions in the fallback view: {len(new_coll_fb)}**\n")
        for g in new_coll_fb:
            T.append(f"- {sb.kind(g, split)}: " + ", ".join(
                f"`{p}`" + ("*" if split[p] == "novel" else "") for p in g)
                + f" — phi {{{', '.join(fmt(x) for x in sorted(phi_fb[g[0]].items()))}}}")
    T.append("\n`*` = novel.\n")

    T.append("## Strict-subset pairs\n")
    T.append("\"Novel-subset\" = the SUBSET predicate is novel (a novel predicate "
             "dominated by another). This is the stage-0b figure of 87; the 209 "
             "reported there counted any pair involving a novel predicate.\n")
    T.append("| | stage 0b | v3 | v3 fallback |")
    T.append("|---|---:|---:|---:|")
    T.append(f"| all strict-subset pairs | {len(sp_b)} | {len(sp_v3)} | {len(sp_fb)} |")
    T.append(f"| novel-subset pairs | {len(nov_sub(sp_b))} | {len(nov_sub(sp_v3))} | {len(nov_sub(sp_fb))} |")
    T.append(f"\n**⚠ NEW novel-subset pairs in v3: {len(new_ns)}**\n")
    for a, b in new_ns:
        T.append(f"- `{a}`* ⊂ `{b}`" + ("*" if split[b] == "novel" else ""))
    T.append(f"\nNovel-subset pairs REMOVED by v3: {len(gone_ns)}\n")
    for a, b in gone_ns:
        T.append(f"- `{a}`* ⊂ `{b}`" + ("*" if split[b] == "novel" else ""))
    if new_ns_fb:
        T.append(f"\n**⚠ Additional novel-subset pairs in the fallback view: {len(new_ns_fb)}**\n")
        for a, b in new_ns_fb:
            T.append(f"- `{a}`* ⊂ `{b}`" + ("*" if split[b] == "novel" else ""))

    # ride geometry
    nr = len(ride_rows)
    cats = Counter(r["cat"] for r in ride_rows)
    mv = [r["mv"] for r in ride_rows if r["mv"]]
    both_move = [m for m in mv if m["s_net"] > MOV and m["o_net"] > MOV]
    aligned = [m for m in both_move if m["cos"] > 0.7]
    stable = [m for m in mv if m["drift"] == m["drift"] and m["drift"] < 0.25]
    neither = [m for m in mv if m["s_net"] <= MOV and m["o_net"] <= MOV]
    drive_like = [r for r in ride_rows if r["mv"] and r["cat"].startswith(("contained", "overlapping"))
                  and r["mv"]["o_net"] > MOV and r["mv"]["s_net"] > MOV and r["mv"]["cos"] > 0.7]
    T.append("\n## Task 1(d) test — is drive separable from ride on geometry?\n")
    T.append(f"Base `ride` training instances: {tr['ride']}; {nr} with ≥2 frames where "
             f"both boxes are annotated. Motion uses {len(mv)} of them "
             f"({ride_noecc} lack ECC matrices).\n")
    T.append("**Box relation (majority over the instance's frames)**\n")
    T.append("| relation | instances |")
    T.append("|---|---:|")
    for c, n in cats.most_common():
        T.append(f"| {c} | {pct(n, nr)} |")
    T.append(f"\nMedian fraction of the subject box inside the object box: "
             f"{np.median([r['ios_median'] for r in ride_rows]):.2f}. Subject box top "
             f"above object box top in {np.mean([r['top_above'] for r in ride_rows]):.1%} "
             f"of frames (a contained box cannot do this).\n")
    T.append(f"**Co-translation** (net camera-compensated displacement over the extent, "
             f"moving = >{MOV:.0%} of the frame diagonal)\n")
    T.append("| | instances |")
    T.append("|---|---:|")
    T.append(f"| both boxes move | {pct(len(both_move), len(mv))} |")
    T.append(f"| both move AND directions agree (cosine > 0.7) | {pct(len(aligned), len(mv))} |")
    T.append(f"| neither moves | {pct(len(neither), len(mv))} |")
    T.append(f"| subject–object offset stable (drift < 0.25 object-box diagonals) | {pct(len(stable), len(mv))} |")
    T.append(f"| **satisfies the drafted drive recipe's geometry** (overlapping or "
             f"contained, object moving, co-moving) | **{pct(len(drive_like), len(mv))}** |")
    open(os.path.join(HERE, "schema_v3.md"), "w").write("\n".join(T) + "\n")

    # cooccurrence.md
    C = ["# Stage 0c — Task 2: label co-occurrence\n",
         "BASE training annotations only. A hit = the same (video, subject tid, "
         "object tid) carries both labels over temporally overlapping extents. "
         "Novel compounds are masked labels and are not counted as partners; they "
         "are listed as excluded.\n",
         "`any other label` is the control: the fraction of anchor instances that "
         "overlap ANY other base label on the same pair (e.g. `larger`, `watch`). "
         "It shows how multi-labelled pairs are in general.\n",
         "## Bare spatial term vs. verb compounds sharing it\n",
         "| bare | instances | overlap a compound | fraction | any other label (control) | partners (base) | excluded (novel) |",
         "|---|---:|---:|---:|---:|---|---|"]
    for r in bare_rows:
        f = r["fwd"]
        C.append(f"| `{r['anchor']}` | {f['n']} | {f['hit']} | {f['hit']/f['n']:.1%} | "
                 f"{f['any_other']/f['n']:.1%} | {', '.join(r['partners'])} | {', '.join(r['excluded']) or '—'} |")
    C.append("\nReverse direction — of each compound's instances, how many overlap the bare term:\n")
    C.append("| bare | " + "compound: overlapping / instances |")
    C.append("|---|---|")
    for r in bare_rows:
        C.append(f"| `{r['anchor']}` | " + "; ".join(
            f"{c} {r['rev'][c]}/{r['rev_n'][c]}" for c in r["partners"]) + " |")
    C.append("\n## move_X vs. manner_X\n")
    C.append("| move_X | instances | overlap a manner_X | fraction | any other label (control) | partners (base) | excluded (novel) |")
    C.append("|---|---:|---:|---:|---:|---|---|")
    for r in move_rows:
        f = r["fwd"]
        C.append(f"| `{r['anchor']}` | {f['n']} | {f['hit']} | {f['hit']/f['n']:.1%} | "
                 f"{f['any_other']/f['n']:.1%} | {', '.join(r['partners']) or '—'} | {', '.join(r['excluded']) or '—'} |")
    C.append("\nReverse direction:\n")
    C.append("| move_X | manner_X: overlapping / instances |")
    C.append("|---|---|")
    for r in move_rows:
        C.append(f"| `{r['anchor']}` | " + ("; ".join(
            f"{c} {r['rev'][c]}/{r['rev_n'][c]}" for c in r["partners"]) or "—") + " |")
    tb = sum(r["fwd"]["n"] for r in bare_rows); hb = sum(r["fwd"]["hit"] for r in bare_rows)
    tm = sum(r["fwd"]["n"] for r in move_rows); hm = sum(r["fwd"]["hit"] for r in move_rows)
    ab = sum(r["fwd"]["any_other"] for r in bare_rows); am = sum(r["fwd"]["any_other"] for r in move_rows)
    C.append(f"\n**Pooled:** bare terms {pct(hb, tb)} overlap a compound (control: "
             f"{ab/tb:.1%} overlap any other label); move_X {pct(hm, tm)} overlap a "
             f"manner_X (control: {am/tm:.1%}).\n")
    C.append("## Conclusion\n")
    C.append(f"**(A) Largely disjoint.** A bare spatial term shares an overlapping "
             f"extent with a verb compound of the same spatial value on "
             f"{hb/tb:.1%} of its instances, although {ab/tb:.1%} of those same "
             f"instances overlap *some* other label on the pair. The pairs are "
             f"heavily multi-labelled; they are just not labelled bare-plus-"
             f"compound. The reverse direction agrees: no compound overlaps its "
             f"bare term more than a few percent of the time. move_X is disjoint "
             f"from manner_X ({hm/tm:.1%}); the only exceptions are "
             f"{hm} `move_beneath`/`jump_beneath` overlaps.\n")
    C.append("Annotators picked one label from each family, not an underspecified "
             "one alongside a specific one. So bare and move_ labels behave as "
             "asserting the absence of the manner their compounds name. **Unset "
             "manner axes should be scored as 'asserts absence', not "
             "marginalised.**\n")
    C.append("Scope: this licenses 'absent' only for the axes these two families "
             "leave unset (locomotion, and posture for the bare terms). It does not "
             "test whether an unset `contact` or `relative_motion` means absence.\n")
    open(os.path.join(HERE, "cooccurrence.md"), "w").write("\n".join(C) + "\n")

    # stop_halt.md
    S = ["# Stage 0c — Task 3: is `stop` a halt?\n",
         f"BASE training annotations only. Subject speed = mean camera-compensated "
         f"per-frame displacement of the subject box centre, as a fraction of the "
         f"frame diagonal. (i) = the annotated extent; (ii) = the window of equal "
         f"length immediately before it (needs ≥3 tracked steps there).\n",
         f"Instances: {', '.join(stop_p)} (stop); control {', '.join(stand_p)} "
         f"(stand, matched on spatial term). Reference for 'clearly moving': "
         f"τ = {tau:.4f}/frame, the 25th percentile of extent speed over base "
         f"walk_* and run_* instances (so 75% of labelled walking/running exceeds it).\n",
         "Excluded: " + ", ".join(f"{k}: {v}" for k, v in skipped.items()) + ".\n",
         "## Rates\n",
         "| | instances | with a before-window | (ii) clearly exceeds (i) ¹ | transition inside extent ² | extent speed ≥ τ | median speed (i) | median speed (ii) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = [("stop_*", "stop"), ("stand_* (spatial-matched)", "stand"),
              ("stand_* (category-matched)", "stand_cat_matched"),
              ("walk_*/run_* (reference)", "walk_run")]
    for lab, k in labels:
        h = hs[k]
        if h is None:
            continue
        S.append(f"| {lab} | {h['n']} | {h['n_before']} | {pct(h['before_exceeds'], h['n_before'])} | "
                 f"{pct(h['transition'], h['n'])} | {pct(h['ext_moving'], h['n'])} | "
                 f"{h['ext_med']:.4f} | {h['bef_med']:.4f} |")
    S.append(f"\n¹ before-window speed ≥ τ AND ≥ 2× the extent speed.  "
             f"² first third of the extent ≥ τ AND last third ≤ half the first third.\n")
    S.append(f"Category-matched control: stand instances resampled to stop's subject-"
             f"category mix; covers {pct(cat_cov, len(stop_f))} of stop instances "
             f"(the rest have subject categories never seen standing). Stop's top "
             f"subject categories: " + ", ".join(f"{c} {n}" for c, n in stop_cats.most_common(6))
             + ".\n")
    S.append("## Threshold-free: can the speeds tell stop_* from stand_*?\n")
    S.append("AUC > 0.5 = the feature is higher for stop. 95% CI from a video-clustered bootstrap.\n")
    S.append("| feature | n | n stop | AUC | 95% CI |")
    S.append("|---|---:|---:|---:|---|")
    for name, n, n1, a, lo, hi in comp:
        S.append(f"| {name} | {n} | {n1} | {a:.3f} | {lo:.3f}–{hi:.3f} |")
    S.append("## Subject categories — the control the brief assumed does not exist\n")
    S.append("Base training instances, subject category by verb:\n")
    S.append("| verb | instances | vehicle subjects (categories seen with stop) | top categories |")
    S.append("|---|---:|---:|---|")
    for vb in ("stop", "stand", "sit", "lie"):
        n = sum(vcat[vb].values())
        vn = n if vb == "stop" else veh_in[vb]
        S.append(f"| {vb} | {n} | {pct(vn, n)} | " + ", ".join(
            f"{c} {k}" for c, k in vcat[vb].most_common(6)) + " |")
    S.append(f"\nVehicle categories: {', '.join(vehicles)}. Every stop_* subject is a "
             f"vehicle; stand/sit/lie subjects almost never are. A category-matched "
             f"stand_* control is therefore empty (coverage {pct(cat_cov, len(stop_f))} "
             f"above), and the stop-vs-stand comparison is necessarily a vehicles-vs-"
             f"animals comparison.\n")
    S.append("## Conclusion\n")
    h1, h2 = hs["stop"], hs["stand"]
    S.append(f"**(C) stop_* is indistinguishable from stand_* on both windows.** "
             f"The before-window clearly exceeds the extent in {h1['before_exceeds']/h1['n_before']:.1%} "
             f"of stop vs {h2['before_exceeds']/h2['n_before']:.1%} of stand. A within-extent "
             f"moving→stationary transition shows in {h1['transition']/h1['n']:.1%} vs "
             f"{h2['transition']/h2['n']:.1%}. Median before-window speed is lower for stop "
             f"({h1['bef_med']:.4f} vs {h2['bef_med']:.4f}). Every threshold-free AUC has a "
             f"CI that includes 0.5. Neither (A) nor (B) holds: no halt is visible inside "
             f"the extent or before it. Only {h1['n_before']} of {h1['n']} stop instances have "
             f"a usable before-window at all, so (B) is also weakly powered — but the "
             f"direction is wrong for it, not merely noisy.\n")
    S.append("What `stop` actually encodes is **stationary + vehicle subject**. It "
             "is the vehicle counterpart of stand/sit/lie, chosen by subject category, "
             "not by motion history. state_change is empty as an axis. Subject "
             f"category separates stop from stand/sit/lie almost perfectly "
             f"({pct(sum(vcat['stop'].values()), sum(vcat['stop'].values()))} vs "
             f"{sum(veh_in.values())}/{sum(sum(vcat[v].values()) for v in ('stand','sit','lie'))} "
             "vehicle subjects), but phi has no axis for it.\n")
    S.append("### Consequence of reverting stop_* to stationary\n")
    S.append("Written to `phi_v3_stop_reverted.json`; `phi_v3.json` keeps the Task 1 "
             "spec. After reversion, stop_X = {subject_state=stationary, locomotion=none, "
             "spatial}; stand_/sit_/lie_X add posture. Each novel stop_* predicate then "
             "becomes a strict subset of:\n")
    for p in stop_nov:
        S.append(f"- `{p}`* ⊂ " + ", ".join(
            f"`{b}`" + ("*" if split[b] == "novel" else "") for b in stop_sub[p]))
    reach_word = "unreachable" if all(p in unr_r for p in stop_nov) else "reachable"
    S.append(f"\nUnder the LEARNED-threshold rule they are {reach_word}: their only "
             f"LEARNED ingredient is locomotion=none, which has {sup[('locomotion','none')]} "
             f"base instances. They are not reachable in the sense that matters. Nothing "
             f"in phi or in the annotated extent distinguishes them from their posture-"
             f"marked supersets. A scorer can only separate them through subject category, "
             f"which is outside phi. New collisions from the reversion: {len(new_coll_r)}"
             + ("" if not new_coll_r else " — " + "; ".join(", ".join(g) for g in new_coll_r))
             + ". The v3 collision `stop_next_to` ≡ `stop_with` persists.\n")
    open(os.path.join(HERE, "stop_halt.md"), "w").write("\n".join(S) + "\n")

    summary = dict(
        support_contact=sup[("contact", "contact")],
        unreachable_b=unr_b, unreachable_strict=unr_strict, unreachable_fallback=unr_fb,
        collisions=dict(stage0b=kb, v3=kv, fallback=kf),
        new_collisions=new_coll, new_collisions_fallback=new_coll_fb,
        subset_pairs=dict(stage0b=len(sp_b), v3=len(sp_v3), fallback=len(sp_fb)),
        novel_subset_pairs=dict(stage0b=len(nov_sub(sp_b)), v3=len(nov_sub(sp_v3)),
                                fallback=len(nov_sub(sp_fb))),
        new_novel_subset=new_ns, removed_novel_subset=gone_ns, new_novel_subset_fallback=new_ns_fb,
        ride=dict(n=nr, n_motion=len(mv), no_ecc=ride_noecc, cats=dict(cats),
                  both_move=len(both_move), aligned=len(aligned), neither=len(neither),
                  stable=len(stable), drive_like=len(drive_like),
                  ios_median=float(np.median([r['ios_median'] for r in ride_rows])),
                  top_above=float(np.mean([r['top_above'] for r in ride_rows]))),
        cooc_bare=[dict(anchor=r["anchor"], n=r["fwd"]["n"], hit=r["fwd"]["hit"],
                        any_other=r["fwd"]["any_other"], per=dict(r["fwd"]["per"]),
                        rev={c: [r["rev"][c], r["rev_n"][c]] for c in r["partners"]},
                        excluded=r["excluded"]) for r in bare_rows],
        cooc_move=[dict(anchor=r["anchor"], n=r["fwd"]["n"], hit=r["fwd"]["hit"],
                        any_other=r["fwd"]["any_other"], per=dict(r["fwd"]["per"]),
                        rev={c: [r["rev"][c], r["rev_n"][c]] for c in r["partners"]},
                        excluded=r["excluded"]) for r in move_rows],
        stop_reverted=dict(subsets=stop_sub, unreachable=unr_r,
                           new_collisions=new_coll_r),
        subject_categories={vb: dict(c.most_common()) for vb, c in vcat.items()},
        halt=dict(tau=tau, stats=hs, auc=[dict(feature=a, n=b, n_stop=c, auc=d, lo=e, hi=f)
                                           for a, b, c, d, e, f in comp],
                  skipped=dict(skipped), stop_cats=dict(stop_cats.most_common(10)),
                  cat_cov=cat_cov),
    )
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
