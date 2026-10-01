#!/usr/bin/env python3
"""Stage 1 session 3 -- train tracks under option B, and a re-score of the frozen
measured axes on the option-B grid.

    python3 pilot_analysis/stage1c/optionB.py

Option B (amendment A5): each annotated train tid is split wherever more than
30 consecutive frames lack a box; every run becomes its own track; inside a run
the missing frames (gaps of <= 30) are linearly interpolated and rounded and
clipped at 0 like the repo's round_and_positive. Every annotated box is kept,
no box is invented across a gap > 30 frames, and no pre-roll frames are added.

Re-score: HELD agreement for every measured axis with the FROZEN windows and
thresholds (stage1b/thresholds_slot.json, preregistration A1-A3); no refit. The
scorer is first run on session 2's grid and must reproduce session 2's HELD
numbers exactly, so any difference on the option-B grid is the grid's.

BASE labels from the TRAINING annotations only; novel instances dropped at parse.
No test annotation is opened.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob, json, os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage1b"))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage1"))
import slot_axes as S       # noqa: E402
import measured_axes as M   # noqa: E402

GAP = 30
PHI = os.path.join(ROOT, "pilot_analysis", "stage0e", "phi_frozen.json")
TRAIN = os.path.join(ROOT, "data", "vidvrd", "anno", "train")
ECC = os.path.join(ROOT, "data", "vidvrd", "data", "VidVRD_ECC_train.json")
SPLIT = os.path.join(ROOT, "pilot_analysis", "stage1", "train_split.json")
TH = json.load(open(os.path.join(ROOT, "pilot_analysis", "stage1b", "thresholds_slot.json")))["per_window"]
S2 = json.load(open(os.path.join(ROOT, "pilot_analysis", "stage1b", "slot_check_summary.json")))

# frozen windows (preregistration + A1-A3)
WIN = dict(horizontal="W0", vertical="W0", depth="W1", proximity="W3", subject_state="W2",
           object_state="W0", larger="W3", taller="W3", faster="W1", D2="W3", D1="W0")
C_D2 = TH["W3"]["c"]


def runs_option_b(frames_of_tid: dict[int, list]) -> list[tuple[int, dict[int, list]]]:
    """{fid: box} for one tid -> [(start, {fid: box dense over the run})]."""
    fs = sorted(frames_of_tid)
    cuts, start = [], fs[0]
    for a, b in zip(fs, fs[1:]):
        if b - a - 1 > GAP:
            cuts.append((start, a + 1)); start = b
    cuts.append((start, fs[-1] + 1))
    out = []
    for lo, hi in cuts:
        ann = [f for f in fs if lo <= f < hi]
        grid = np.arange(lo, hi)
        cols = [np.interp(grid, ann, [frames_of_tid[f][k] for f in ann]) for k in range(4)]
        dense = {}
        for i, f in enumerate(grid):
            f = int(f)
            dense[f] = list(frames_of_tid[f]) if f in frames_of_tid else [max(0, round(c[i])) for c in cols]
        out.append((lo, dense))
    return out


def load_video(path, base):
    d = json.load(open(path))
    rels = [r for r in d["relation_instances"] if r["predicate"] in base]   # novel dropped here
    del d["relation_instances"]
    raw = {}
    for fid, fr in enumerate(d["trajectories"]):
        for e in fr:
            b = e["bbox"]
            raw.setdefault(e["tid"], {})[fid] = [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
    cats = {so["tid"]: so["category"] for so in d["subject/objects"]}
    return d, rels, raw, cats


def grids(d, raw, kind):
    """-> (frames: list[dict key->box], tracks: {key: (b, e)}, gt_of: {key: gt tid})."""
    n = len(d["trajectories"])
    frames = [dict() for _ in range(n)]
    tracks, gt_of = {}, {}
    if kind == "session2":
        for tid, fb in raw.items():
            for f, b in fb.items():
                frames[f][tid] = tuple(b)
        tracks = S.whole_tracks(d["trajectories"])
        gt_of = {t: t for t in tracks}
    else:
        k = 0
        for tid, fb in sorted(raw.items()):
            for lo, dense in runs_option_b(fb):
                for f, b in dense.items():
                    frames[f][k] = tuple(b)
                tracks[k] = (lo, max(dense) + 1)
                gt_of[k] = tid
                k += 1
    return frames, tracks, gt_of


def score(kind, base, phi, held, ecc):
    asserts = lambda labels, axis: (lambda v: None if not v else (v.pop() if len(v) == 1 else "CONFLICT"))(
        {phi[p].get(axis) for p in labels} - {None})
    has = lambda labels, axis, val: any(phi[p].get(axis) == val for p in labels)
    tally = Counter()
    counts = Counter()
    for f in sorted(glob.glob(os.path.join(TRAIN, "*.json"))):
        d, rels, raw, cats = load_video(f, base)
        v = d["video_id"]
        frames, tracks, gt_of = grids(d, raw, kind)
        grid = S.slot_grid(tracks)
        counts["tracks"] += len(tracks)
        counts["pairs"] += len(grid)
        counts["slots"] += sum(len(sl) for *_, sl in grid)
        counts["pairs_over_40"] += sum(len(sl) > 40 for *_, sl in grid)
        for s, o, ov, slots in grid:
            for slot in slots:
                labels = S.slot_labels(slot, gt_of[s], gt_of[o], rels)
                if labels:
                    counts["labelled_slots"] += 1
                    if v in ecc:
                        counts["labelled_slots_with_ecc"] += 1
                if not labels or v not in held or v not in ecc:
                    continue
                W, H = d["width"], d["height"]
                cache = {}

                def pair(w):
                    if w not in cache:
                        lo, hi = S.window(slot, ov, w)
                        p = M.build_pair(frames, s, o, lo, hi, ecc[v], W, H)
                        h = S.build_heading(frames, s, o, lo, hi, ecc[v]) if p is not None else None
                        cache[w] = (p, h)
                    return cache[w]

                def add(key, ok):
                    tally[key + "_n"] += 1
                    tally[key + "_ok"] += bool(ok)

                for axis, (neg, pos), fn in (("horizontal", ("left", "right"), M.horizontal),
                                             ("vertical", ("above", "beneath"), M.vertical),
                                             ("depth", ("behind", "front"), M.depth)):
                    a = asserts(labels, axis)
                    if a in (neg, pos):
                        p, _ = pair(WIN[axis])
                        if p is not None:
                            add(f"{axis}={a}", fn(p) == a)
                a = asserts(labels, "subject_state")
                if a in ("moving", "stationary"):
                    p, _ = pair(WIN["subject_state"])
                    if p is not None and M.subject_motion(p) == M.subject_motion(p):
                        add(f"subject_state={a}", (M.subject_motion(p) > TH[WIN["subject_state"]]["motion"]) == (a == "moving"))
                if has(labels, "relative_motion", "co_move"):
                    p, _ = pair(WIN["object_state"])
                    if p is not None:
                        add("object_state=moving", M.object_motion(p) > TH[WIN["object_state"]]["motion"])
                a = asserts(labels, "proximity")
                spatial = any(asserts(labels, x) not in (None, "CONFLICT") for x in ("horizontal", "vertical", "depth"))
                if a == "adjacent" or (a is None and spatial):
                    p, _ = pair(WIN["proximity"])
                    if p is not None:
                        adj = M.gap_score(p) < TH[WIN["proximity"]]["gap"] and not M.contained(p)
                        add("proximity=adjacent" if a == "adjacent" else "proximity=non-adjacent negatives",
                            adj if a == "adjacent" else not adj)
                for c, fn in (("larger", M.larger_score), ("taller", M.taller_score), ("faster", M.faster_score)):
                    if has(labels, "comparative", c):
                        p, _ = pair(WIN[c])
                        if p is not None:
                            x = fn(p)
                            if x == x:
                                add(f"comparative={c}", x > 0)
                # relative motion: D2 main (A1), D1 ablation
                p, h = pair(WIN["D2"])
                if p is not None:
                    r2 = S.relative_motion_heading(p, h, C_D2, TH[WIN["D2"]]["motion"])
                    if has(labels, "relative_motion", "co_move"):
                        add("D2 co_move on *_with", r2 == "co_move")
                    if labels & {"chase", "follow"}:
                        add("D2 toward on chase/follow", r2 == "toward")
                    if any(phi[x].get("posture") for x in labels):
                        add("D2 subject-driven on stand/sit/lie (false positive)", r2 in ("toward", "away", "pass"))
                p, _ = pair(WIN["D1"])
                if p is not None:
                    r1 = S.d1_decide(S.d1_stats(p), TH[WIN["D1"]]["dead_zone"], TH[WIN["D1"]]["motion"])
                    if has(labels, "relative_motion", "co_move"):
                        add("D1 co_move on *_with", r1 == "co_move")
                    if labels & {"chase", "follow"}:
                        add("D1 approach-or-co_move on chase/follow", r1 in ("approach", "co_move"))
    return tally, counts


def session2_reference():
    h = S2["held"]; rm = S2["rm"]
    ref = {}
    for axis in ("horizontal", "vertical", "depth", "subject_state", "object_state", "proximity"):
        for val, (a, n) in h[axis]["agree"].items():
            ref[f"{axis}={val}"] = (a, n)
    for c in ("larger", "taller", "faster"):
        ref[f"comparative={c}"] = tuple(h[f"comparative={c}"]["agree"][c])
    ref["D2 co_move on *_with"] = tuple(rm["D2"]["HELD"]["co_move"])
    ref["D2 toward on chase/follow"] = tuple(rm["D2"]["HELD"]["chase_follow"])
    ref["D2 subject-driven on stand/sit/lie (false positive)"] = tuple(rm["D2"]["HELD"]["stationary_fp"])
    ref["D1 co_move on *_with"] = tuple(rm["D1"]["HELD"]["co_move"])
    ref["D1 approach-or-co_move on chase/follow"] = tuple(rm["D1"]["HELD"]["chase_follow"])
    return ref


def main():
    raw = json.load(open(PHI))
    phi = {p: {k: v for k, v in d.items() if not k.startswith("_")} for p, d in raw.items()}
    base = {p for p, d in raw.items() if d["_split"] == "base"}
    held = set(json.load(open(SPLIT))["held"])
    ecc = json.load(open(ECC))
    ref = session2_reference()
    t2, c2 = score("session2", base, phi, held, ecc)
    tb, cb = score("optionB", base, phi, held, ecc)
    rows, reproduced, stop = [], True, []
    for k in ref:
        old = (t2[k + "_ok"], t2[k + "_n"])
        new = (tb[k + "_ok"], tb[k + "_n"])
        same = old == tuple(ref[k])
        reproduced &= same
        fo = old[0] / old[1] if old[1] else float("nan")
        fn = new[0] / new[1] if new[1] else float("nan")
        fp = "false positive" in k
        drop = (fn - fo) if fp else (fo - fn)          # positive = got worse
        learned = k.startswith("proximity=")            # A3: adjacent is LEARNED now
        if drop > 0.05 and not learned:
            stop.append(k)
        rows.append((k, ref[k], old, new, fo, fn, drop, same, learned))
    out = dict(counts_session2=dict(c2), counts_optionB=dict(cb), reproduced=reproduced,
               rows=[dict(metric=k, session2_summary=list(r), rescored_session2_grid=list(o),
                          optionB=list(n), drop_points=round(100 * dr, 2), reproduces=sm, learned_since_A3=le)
                     for k, r, o, n, _, _, dr, sm, le in rows],
               stop=stop)
    json.dump(out, open(os.path.join(HERE, "optionB_rescore.json"), "w"), indent=1)
    pct = lambda t: f"{t[0]}/{t[1]} ({t[0] / t[1]:.1%})" if t[1] else "—"
    L = ["# Option-B grid — re-score of the frozen measured axes (HELD, no refit)\n",
         "Frozen windows and thresholds come from `stage1b/thresholds_slot.json` and "
         "preregistration A1–A3. The scorer is first run on session 2's grid, where it "
         f"must reproduce session 2's HELD numbers: **{'reproduced exactly' if reproduced else '⚠ NOT reproduced'}**.\n",
         "| metric | session 2 grid | option-B grid | change (points; + = worse) |", "|---|---:|---:|---:|"]
    for k, r, o, n, fo, fn, dr, sm, le in rows:
        L.append(f"| {k}{' *(LEARNED since A3, reference only)*' if le else ''} | {pct(o)} | {pct(n)} | {100 * dr:+.1f} |")
    L.append(f"\n**Stop rule** (any measured axis worse by > 5 points): "
             + ("**triggered** by " + ", ".join(stop) if stop else "not triggered") + ".\n")
    L.append("## Train counts\n")
    L.append("| grid | tracks | pairs | slots | labelled slots | (with ECC) | pairs > 40 slots |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, c in (("session 2 (whole spans, gaps skipped)", c2), ("**option B**", cb)):
        L.append(f"| {name} | {c['tracks']} | {c['pairs']} | {c['slots']} | {c['labelled_slots']} | "
                 f"{c['labelled_slots_with_ecc']} | {c['pairs_over_40']} |")
    open(os.path.join(HERE, "optionB_rescore.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
