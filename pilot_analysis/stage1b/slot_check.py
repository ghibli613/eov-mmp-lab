#!/usr/bin/env python3
"""Stage 1 session 2, Tasks 1-3 -- measured axes at the classifier's time scale.

    python3 pilot_analysis/stage1b/slot_check.py

CPU only. BASE labels from the TRAINING annotations only: novel relation
instances are dropped as each file is parsed and never stored. Test annotations
are never opened. Reuses stage1/train_split.json, stage1/measured_axes.py and
stage1/axis_check.py helpers. Writes only into pilot_analysis/stage1b/.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob, json, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage1"))
import slot_axes as S          # noqa: E402
import measured_axes as M      # noqa: E402
import axis_check as A1        # noqa: E402  (session-1 helpers: auc, auc_ci, best_threshold)

PHI = os.path.join(ROOT, "pilot_analysis", "stage0e", "phi_frozen.json")
SPLIT_INFO = os.path.join(ROOT, "data", "vidvrd", "data", "openvoc_pred_class_spilt_info.json")
TRAIN_DIR = os.path.join(ROOT, "data", "vidvrd", "anno", "train")
ECC_FILE = os.path.join(ROOT, "data", "vidvrd", "data", "VidVRD_ECC_train.json")
SPLIT1 = os.path.join(ROOT, "pilot_analysis", "stage1", "train_split.json")
S1_SUMMARY = os.path.join(ROOT, "pilot_analysis", "stage1", "axis_check_summary.json")
C_GRID = np.round(np.linspace(0.0, 1.0, 201), 3)
COMOVE_FLOOR = 0.60


# ---------------------------------------------------------------------------
def load_videos(base):
    ecc = json.load(open(ECC_FILE))
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.json"))):
        d = json.load(open(f))
        rels = [r for r in d["relation_instances"] if r["predicate"] in base]   # novel dropped here
        del d["relation_instances"]
        v = d["video_id"]
        frames = [{e["tid"]: (e["bbox"]["xmin"], e["bbox"]["ymin"], e["bbox"]["xmax"], e["bbox"]["ymax"])
                   for e in fr} for fr in d["trajectories"]]
        yield v, d, frames, rels, ecc.get(v)


def window_stats(frames, s, o, lo, hi, eccv, W, H):
    p = M.build_pair(frames, s, o, lo, hi, eccv, W, H)
    if p is None:
        return None, None, None
    h = S.build_heading(frames, s, o, lo, hi, eccv)
    st = dict(hs=M.horizontal_score(p), vs=M.vertical_score(p), ds=M.depth_score(p),
              gap=M.gap_score(p), cont=M.contained(p),
              lg=M.larger_score(p), tl=M.taller_score(p), fs=M.faster_score(p),
              d1=S.d1_stats(p), d2=S.d2_stats(p, h))
    return st, p, h


def main():
    raw = json.load(open(PHI))
    phi = {p: {k: v for k, v in d.items() if not k.startswith("_")} for p, d in raw.items()}
    split = {p: d["_split"] for p, d in raw.items()}
    assert all(json.load(open(SPLIT_INFO))["cls2split"][p] == split[p] for p in split)
    base = {p for p, s in split.items() if s == "base"}
    sp = json.load(open(SPLIT1)); fit_v = set(sp["fit"])
    vehicles = set()

    # ---------------- pass 1: grid, labels, per-window statistics ----------------
    recs, grid_stats, no_ecc = [], Counter(), []
    for v, d, frames, rels, eccv in load_videos(base):
        cat = {so["tid"]: so["category"] for so in d["subject/objects"]}
        for r in rels:
            if sb_verb(r["predicate"]) == "stop":
                vehicles.add(cat[r["subject_tid"]])
        if eccv is None:
            no_ecc.append(v)
            continue
        grid = S.slot_grid(S.whole_tracks(d["trajectories"]))
        for s, o, ov, slots in grid:
            grid_stats["pairs"] += 1
            w3 = None
            for slot in slots:
                grid_stats["slots"] += 1
                labels = S.slot_labels(slot, s, o, rels)
                if not labels:
                    continue
                grid_stats["labelled_slots"] += 1
                rec = dict(video=v, subset="FIT" if v in fit_v else "HELD", labels=labels,
                           sub_cat=cat[s])
                for w in S.WINDOWS:
                    if w == "W3":
                        if w3 is None:
                            w3 = window_stats(frames, s, o, ov[0], ov[1], eccv, d["width"], d["height"])[0]
                        rec[w] = w3
                    else:
                        lo, hi = S.window(slot, ov, w)
                        rec[w] = window_stats(frames, s, o, lo, hi, eccv, d["width"], d["height"])[0]
                recs.append(rec)
    FIT = [r for r in recs if r["subset"] == "FIT"]
    HELD = [r for r in recs if r["subset"] == "HELD"]

    def asserts(rec, axis):
        vals = {phi[p].get(axis) for p in rec["labels"]} - {None}
        return None if not vals else (vals.pop() if len(vals) == 1 else "CONFLICT")

    def has(rec, axis, value):
        return any(phi[p].get(axis) == value for p in rec["labels"])

    conflicts = {a: sum(asserts(r, a) == "CONFLICT" for r in recs)
                 for a in ("horizontal", "vertical", "depth", "subject_state", "proximity")}

    # evidence sets for relative motion
    is_cf = lambda r: bool(r["labels"] & {"chase", "follow"})
    is_with = lambda r: has(r, "relative_motion", "co_move")
    is_posture = lambda r: any(phi[p].get("posture") for p in r["labels"])
    is_pass = lambda r: has(r, "relative_motion", "pass")
    is_appr = lambda r: has(r, "relative_motion", "approach")

    # ---------------- fitting per window, on FIT ----------------
    TH = {}
    curve = defaultdict(dict)       # axis -> window -> FIT balanced agreement
    SIGN = dict(horizontal=("hs", "left", "right"), vertical=("vs", "above", "beneath"),
                depth=("ds", "behind", "front"))   # (score, value for score<0, value for score>0)

    def sign_val(x, neg, pos):
        return None if x != x or x == 0 else (neg if x < 0 else pos)

    def ok(rs, w):
        return [r for r in rs if r[w] is not None]

    for w in S.WINDOWS:
        F = ok(FIT, w)
        # subject_state motion threshold
        pos = np.array([r[w]["d1"]["sub_motion"] for r in F if asserts(r, "subject_state") == "moving"])
        neg = np.array([r[w]["d1"]["sub_motion"] for r in F if asserts(r, "subject_state") == "stationary"])
        pos, neg = pos[pos == pos], neg[neg == neg]
        mthr, mba = A1.best_threshold(pos, neg, lambda s_, t_: s_ > t_)
        # adjacency gap
        adj = lambda r: asserts(r, "proximity") == "adjacent"
        nadj = lambda r: asserts(r, "proximity") is None and any(
            asserts(r, a) not in (None, "CONFLICT") for a in ("horizontal", "vertical", "depth"))
        gp = np.array([r[w]["gap"] for r in F if adj(r)])
        gn = np.array([r[w]["gap"] for r in F if nadj(r)])
        gthr, gba = A1.best_threshold(gp, gn, lambda s_, t_: s_ < t_)
        # D1 dead zone
        dz = float(np.percentile([abs(r[w]["d1"]["last"] - r[w]["d1"]["first"]) for r in F if is_with(r)], 90))
        # D2 margin c
        def d2_obj(c):
            a = np.mean([S.d2_decide(r[w]["d2"], c, mthr) == "toward" for r in F if is_cf(r)])
            b = np.mean([S.d2_decide(r[w]["d2"], c, mthr) == "co_move" for r in F if is_with(r)])
            n = np.mean([S.d2_decide(r[w]["d2"], c, mthr) not in ("toward", "away", "pass")
                         for r in F if is_posture(r)])
            return (a + b + n) / 3, (a, b, n)
        scores = [(d2_obj(c)[0], -c, c) for c in C_GRID]
        best = max(scores)
        c_fit = float(best[2]); d2_fit, d2_parts = d2_obj(c_fit)
        d1_parts = (np.mean([S.d1_decide(r[w]["d1"], dz, mthr) in ("approach", "co_move") for r in F if is_cf(r)]),
                    np.mean([S.d1_decide(r[w]["d1"], dz, mthr) == "co_move" for r in F if is_with(r)]),
                    np.mean([S.d1_decide(r[w]["d1"], dz, mthr) not in ("approach", "recede", "pass")
                             for r in F if is_posture(r)]))
        TH[w] = dict(motion=mthr, gap=gthr, dead_zone=dz, c=c_fit,
                     fit_ba=dict(motion=mba, gap=gba), d2_fit=d2_fit, d2_parts=d2_parts,
                     d1_fit=float(np.mean(d1_parts)), d1_parts=d1_parts)

        for axis, (key, neg_v, pos_v) in SIGN.items():
            a_ = [sign_val(r[w][key], neg_v, pos_v) == neg_v for r in F if asserts(r, axis) == neg_v]
            b_ = [sign_val(r[w][key], neg_v, pos_v) == pos_v for r in F if asserts(r, axis) == pos_v]
            curve[axis][w] = (np.mean(a_) + np.mean(b_)) / 2
        curve["proximity"][w] = gba
        curve["subject_state"][w] = mba
        curve["object_state"][w] = np.mean([r[w]["d1"]["obj_motion"] > mthr for r in F if is_with(r)])
        for v_, key in (("larger", "lg"), ("taller", "tl"), ("faster", "fs")):
            curve[f"comparative={v_}"][w] = np.mean([r[w][key] > 0 for r in F
                                                     if has(r, "comparative", v_) and r[w][key] == r[w][key]])
        curve["relative_motion D1"][w] = TH[w]["d1_fit"]
        curve["relative_motion D2"][w] = TH[w]["d2_fit"]

    def pick(axis):
        c = curve[axis]
        return max(S.WINDOWS, key=lambda w: (round(c[w], 12), -S.WINDOWS.index(w)))
    chosen = {axis: pick(axis) for axis in curve}
    # object_state uses the motion threshold of its own chosen window

    # ---------------- HELD, once, at the chosen windows ----------------
    rng = np.random.default_rng(0)
    held = {}

    def aucci(x, y, vids):
        a = A1.auc(x, y); lo, hi = A1.auc_ci(x, y, vids, rng)
        return (a, lo, hi)

    for axis, (key, neg_v, pos_v) in SIGN.items():
        w = chosen[axis]; Hs = ok(HELD, w)
        sel = [r for r in Hs if asserts(r, axis) in (neg_v, pos_v)]
        agree = {val: (sum(sign_val(r[w][key], neg_v, pos_v) == val for r in sel if asserts(r, axis) == val),
                       sum(asserts(r, axis) == val for r in sel)) for val in (neg_v, pos_v)}
        held[axis] = dict(window=w, agree=agree,
                          auc=aucci([r[w][key] for r in sel], [1 if asserts(r, axis) == pos_v else 0 for r in sel],
                                    [r["video"] for r in sel]))
    w = chosen["subject_state"]; Hs = ok(HELD, w); thr = TH[w]["motion"]
    sel = [r for r in Hs if asserts(r, "subject_state") in ("moving", "stationary") and r[w]["d1"]["sub_motion"] == r[w]["d1"]["sub_motion"]]
    held["subject_state"] = dict(window=w, agree={
        val: (sum((r[w]["d1"]["sub_motion"] > thr) == (val == "moving") for r in sel if asserts(r, "subject_state") == val),
              sum(asserts(r, "subject_state") == val for r in sel)) for val in ("moving", "stationary")},
        auc=aucci([r[w]["d1"]["sub_motion"] for r in sel], [asserts(r, "subject_state") == "moving" for r in sel],
                  [r["video"] for r in sel]))
    fam = Counter()
    for r in sel:
        vb = next((p.split("_")[0] for p in sorted(r["labels"]) if phi[p].get("subject_state")), None)
        if vb:
            fam[(vb, "n")] += 1
            fam[(vb, "ok")] += (r[w]["d1"]["sub_motion"] > thr) == (asserts(r, "subject_state") == "moving")
    held["subject_state"]["by_verb"] = {k[0]: (fam[(k[0], "ok")], fam[(k[0], "n")]) for k in fam if k[1] == "n"}

    w = chosen["object_state"]; Hs = ok(HELD, w); thr = TH[w]["motion"]
    sel = [r for r in Hs if is_with(r)]
    held["object_state"] = dict(window=w, agree={"moving": (sum(r[w]["d1"]["obj_motion"] > thr for r in sel), len(sel))}, auc=None)

    w = chosen["proximity"]; Hs = ok(HELD, w); thr = TH[w]["gap"]
    adj = lambda r: asserts(r, "proximity") == "adjacent"
    nadj = lambda r: asserts(r, "proximity") is None and any(
        asserts(r, a) not in (None, "CONFLICT") for a in ("horizontal", "vertical", "depth"))
    sel = [r for r in Hs if adj(r) or nadj(r)]
    held["proximity"] = dict(window=w, agree={
        "adjacent": (sum(r[w]["gap"] < thr and not r[w]["cont"] for r in sel if adj(r)), sum(adj(r) for r in sel)),
        "non-adjacent negatives": (sum(not (r[w]["gap"] < thr and not r[w]["cont"]) for r in sel if nadj(r)), sum(nadj(r) for r in sel))},
        auc=aucci([-r[w]["gap"] for r in sel], [1 if adj(r) else 0 for r in sel], [r["video"] for r in sel]))
    for v_, key in (("larger", "lg"), ("taller", "tl"), ("faster", "fs")):
        w = chosen[f"comparative={v_}"]; Hs = ok(HELD, w)
        s_ = np.array([r[w][key] for r in Hs if has(r, "comparative", v_)]); vv = [r["video"] for r in Hs if has(r, "comparative", v_)]
        k = s_ == s_; s_ = s_[k]; vv = [x for x, kk in zip(vv, k) if kk]
        held[f"comparative={v_}"] = dict(window=w, agree={v_: (int((s_ > 0).sum()), len(s_))},
                                         auc=aucci(np.r_[s_, -s_], np.r_[np.ones(len(s_)), np.zeros(len(s_))], vv * 2))

    # ---------------- Task 2: D1 vs D2 on HELD ----------------
    def rm_eval(defn, rs, w):
        t = TH[w]
        dec = (lambda r: S.d1_decide(r[w]["d1"], t["dead_zone"], t["motion"])) if defn == "D1" else \
              (lambda r: S.d2_decide(r[w]["d2"], t["c"], t["motion"]))
        cf_ok = ("approach", "co_move") if defn == "D1" else ("toward",)
        appr_ok = "approach" if defn == "D1" else "toward"
        subj_driven = ("approach", "recede", "pass") if defn == "D1" else ("toward", "away", "pass")
        R = ok(rs, w)
        def rate(sel, pred):
            sel = [r for r in R if sel(r)]
            return (sum(pred(dec(r)) for r in sel), len(sel))
        return dict(
            window=w,
            pass_=rate(is_pass, lambda x: x == "pass"),
            chase_follow=rate(is_cf, lambda x: x in cf_ok),
            co_move=rate(is_with, lambda x: x == "co_move"),
            approach=rate(is_appr, lambda x: x == appr_ok),
            stationary_fp=rate(is_posture, lambda x: x in subj_driven),
            with_toward_away_fp=rate(is_with, lambda x: x in (("approach", "recede") if defn == "D1" else ("toward", "away"))),
            dist_cf=Counter(dec(r) for r in R if is_cf(r)),
            dist_pass=Counter(dec(r) for r in R if is_pass(r)))
    rm = {dfn: dict(HELD=rm_eval(dfn, HELD, chosen[f"relative_motion {dfn}"]),
                    FIT=rm_eval(dfn, FIT, chosen[f"relative_motion {dfn}"])) for dfn in ("D1", "D2")}
    frac = lambda t: t[0] / t[1] if t[1] else float("nan")
    h1, h2 = rm["D1"]["HELD"], rm["D2"]["HELD"]
    win_pass = "D1" if frac(h1["pass_"]) > frac(h2["pass_"]) else "D2" if frac(h2["pass_"]) > frac(h1["pass_"]) else "tie"
    win_cf = "D1" if frac(h1["chase_follow"]) > frac(h2["chase_follow"]) else "D2" if frac(h2["chase_follow"]) > frac(h1["chase_follow"]) else "tie"
    if win_pass == win_cf and win_pass != "tie":
        cand, why = win_pass, f"{win_pass} is higher on both pass and chase/follow"
    elif {win_pass, win_cf} == {"tie"}:
        cand, why = "D2", "tied on both criteria; the rule's tie-break (subject-centred) applies"
    elif "tie" in (win_pass, win_cf):
        cand = win_pass if win_cf == "tie" else win_cf
        why = f"{cand} wins one criterion and ties the other"
    else:
        cand, why = "D2", (f"the criteria disagree (pass: {win_pass}, chase/follow: {win_cf}); "
                           f"the pre-registered rule chooses D2")
    other = "D1" if cand == "D2" else "D2"
    cm = lambda dfn: frac(rm[dfn]["HELD"]["co_move"])
    if cm(cand) >= COMOVE_FLOOR:
        choice, floor_note = cand, f"co_move {cm(cand):.1%} ≥ 60%"
    elif cm(other) >= COMOVE_FLOOR:
        choice, floor_note = other, (f"{cand} fails the co_move floor ({cm(cand):.1%} < 60%); "
                                     f"{other} passes it ({cm(other):.1%}) and is chosen")
    else:
        choice, floor_note = cand, (f"⚠ neither definition reaches the co_move floor "
                                    f"({cm('D1'):.1%} / {cm('D2'):.1%}); {cand} kept by the "
                                    f"other criteria, floor failure flagged")

    # ---------------- Task 3: stop vs move, vehicle subjects ----------------
    w = chosen["subject_state"]
    def stopmove(rs):
        sel = []
        for r in rs:
            if r[w] is None or r["sub_cat"] not in vehicles:
                continue
            st = any(sb_verb(p) == "stop" for p in r["labels"])
            mv = any(sb_verb(p) == "move" for p in r["labels"])
            if st != mv and r[w]["d1"]["sub_motion"] == r[w]["d1"]["sub_motion"]:
                sel.append((r[w]["d1"]["sub_motion"], 1 if mv else 0, r["video"]))
        return sel
    t3 = {}
    for name, rs in (("all training videos (nothing fitted)", recs), ("HELD only", HELD)):
        sel = stopmove(rs)
        t3[name] = dict(n_move=sum(y for _, y, _ in sel), n_stop=sum(1 - y for _, y, _ in sel),
                        auc=aucci([x for x, _, _ in sel], [y for _, y, _ in sel], [v_ for _, _, v_ in sel]))

    # ---------------- pass 2: exact symmetry at every window ----------------
    sym = Counter()
    MAP_T1 = {"approach": "recede", "recede": "approach"}
    MAP_T2 = {"toward": "away", "away": "toward"}
    INV = {"left": "right", "right": "left", "above": "beneath", "beneath": "above",
           "front": "behind", "behind": "front", None: None}
    for v, d, frames, rels, eccv in load_videos(base):
        if eccv is None:
            continue
        for s, o, ov, slots in S.slot_grid(S.whole_tracks(d["trajectories"])):
            for slot in slots:
                if not S.slot_labels(slot, s, o, rels):
                    continue
                for w in S.WINDOWS:
                    lo, hi = S.window(slot, ov, w)
                    _, p, h = window_stats(frames, s, o, lo, hi, eccv, d["width"], d["height"])
                    if p is None:
                        continue
                    t = TH[w]
                    r1 = M.relative_motion(p, t["dead_zone"], t["motion"])
                    sym[f"{w}_d1consistency_n"] += 1
                    sym[f"{w}_d1consistency_ok"] += r1 == S.d1_decide(S.d1_stats(p), t["dead_zone"], t["motion"])
                    pr, hr = M.time_reverse(p), S.reverse_heading(h)
                    sym[f"{w}_timeD1_n"] += 1
                    sym[f"{w}_timeD1_ok"] += M.relative_motion(pr, t["dead_zone"], t["motion"]) == MAP_T1.get(r1, r1)
                    r2 = S.relative_motion_heading(p, h, t["c"], t["motion"])
                    sym[f"{w}_timeD2_n"] += 1
                    sym[f"{w}_timeD2_ok"] += S.relative_motion_heading(pr, hr, t["c"], t["motion"]) == MAP_T2.get(r2, r2)
                    q = M.swap(p)
                    for fn in (M.horizontal, M.vertical, M.depth):
                        sym[f"{w}_swap_n"] += 1
                        sym[f"{w}_swap_ok"] += fn(q) == INV[fn(p)]
                    c1, c2 = M.comparative(p), M.comparative(q)
                    for k in c1:
                        sym[f"{w}_comp_n"] += 1
                        sym[f"{w}_comp_ok"] += c2[k] == (None if c1[k] is None else (not c1[k]))
    sym_exact = all(sym[k] == sym[k[:-2] + "_ok"] for k in sym if k.endswith("_n"))

    out = dict(grid=dict(grid_stats), no_ecc_videos=len(no_ecc), n=dict(fit=len(FIT), held=len(HELD)),
               conflicts=conflicts, thresholds=TH, curve={a: dict(c) for a, c in curve.items()},
               chosen=chosen, held=held, rm=rm, decision=dict(choice=choice, why=why, floor=floor_note,
                                                             win_pass=win_pass, win_cf=win_cf),
               task3=t3, symmetry=dict(sym), symmetry_exact=sym_exact, vehicles=sorted(vehicles))
    json.dump(out, open(os.path.join(HERE, "slot_check_summary.json"), "w"), indent=1, default=lambda x: (
        list(x) if isinstance(x, (set, tuple)) else dict(x) if isinstance(x, Counter) else float(x)))
    json.dump(dict(chosen_windows=chosen, per_window=TH, relative_motion=choice),
              open(os.path.join(HERE, "thresholds_slot.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(grid=dict(grid_stats), chosen=chosen, decision=out["decision"],
                          symmetry_exact=sym_exact), indent=1))
    return out


def sb_verb(p):
    t = p.split("_")
    return t[0] if len(t) >= 2 and p not in ("next_to", "fall_off") else None


if __name__ == "__main__":
    main()
