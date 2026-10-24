#!/usr/bin/env python3
"""D2 Steps 3-4 (A25): stage of each U1 miss, the tables, the reading. Laptop, CPU.

    python -m d2.analyse --rec_dir REC      # REC holds manifest.jsonl and test/<video>.{npz,json}

TEST-LABEL ERROR ANALYSIS, DISCLOSED; NO METHOD IS EVALUATED. The order is enforced here:
every pinned input, the d2 code (A25a), the baseline tracks and every recording are verified,
and each video passes ground rule 3 and the replay check, **before** any test label is opened
(`open_labels`). Any failure stops the batch.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import collections
import json
import os
import time

import numpy as np

from d2.common import (ANNO_TEST, CODE_PINS, DATA, FLOOR, FLOORS, LOGS, META, SCORE_LEVELS, THESIS, THR,
                       TC_MISSED, code_hashes, sha256, use_code_root, verify_pins, write_log)
from d2 import stages as S

TOP = ["O1a", "O1b", "O2", "O3", "O4", "S0"]


# =========================================================================== verification
def verify_code_pins():
    if not os.path.exists(CODE_PINS):
        raise SystemExit("!! STOP: d2/logs/code_pins.json (A25a) is missing")
    from d2.common import CODE_ROOT, FRONT_END_FILES, sha256_or_none
    pins = json.load(open(CODE_PINS))
    now = code_hashes()
    bad = {f: (now.get(f), h) for f, h in pins["d2_code"].items() if now.get(f) != h}
    bad |= {f: "front end" for f in FRONT_END_FILES
            if sha256_or_none(os.path.join(CODE_ROOT, f)) != pins["front_end_eov"][f]}
    if bad:
        raise SystemExit(f"!! STOP: code differs from A25a's pins: {bad}")
    return pins


def verify_recordings(rec_dir: str, videos: list) -> dict:
    if not os.path.exists(os.path.join(rec_dir, "manifest.jsonl")):
        raise SystemExit(f"!! STOP: no recordings at {rec_dir}")
    man = {}
    for line in open(os.path.join(rec_dir, "manifest.jsonl")):
        r = json.loads(line)
        man[r["video"]] = r
    missing = [v for v in videos if v not in man]
    if missing:
        raise SystemExit(f"!! STOP: {len(missing)} videos not recorded, e.g. {missing[:3]}")
    for v in videos:
        for e in (".npz", ".json"):
            p = os.path.join(rec_dir, "test", v + e)
            if not os.path.exists(p) or sha256(p) != man[v][f"sha256{e}"]:
                raise SystemExit(f"!! STOP: {v}{e} missing or not matching the manifest")
    return man


def verify_cache_and_anno():
    """The 204 SGDet cache files against SHA256SUMS_session3.txt, and the 200 test annotation
    files against d2/logs/anno_test_sha256.txt (bytes only; nothing is opened)."""
    from d2.common import CACHE_SGDET, CACHE_SUMS
    sums = dict(reversed(l.strip().split(None, 1)) for l in open(CACHE_SUMS))
    rels = [r for r in sums if r.startswith("cache_s3_sgdet/")]
    for rel in rels:
        if sha256(os.path.join(os.path.dirname(CACHE_SGDET), rel)) != sums[rel]:
            raise SystemExit(f"!! STOP: {rel} does not match SHA256SUMS_session3.txt")
    anno = [l.split() for l in open(os.path.join(LOGS, "anno_test_sha256.txt"))]
    for h, name in anno:
        if sha256(os.path.join(ANNO_TEST, name)) != h:
            raise SystemExit(f"!! STOP: anno/test/{name} does not match its A25 hash")
    return dict(cache_files=len(rels), anno_files=len(anno))


def load_recording(rec_dir: str, video: str):
    stem = os.path.join(rec_dir, "test", video)
    with np.load(stem + ".npz", allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.load(open(stem + ".json"))
    return arr, meta


def check_video(G, arr, meta, base: list) -> list:
    """Ground rule 3 (recorded final == baseline), the replay check (the repo's own filter on the
    recorded rows == recorded final), the post-processor checks, and unique (frame, id) rows."""
    from d2.baseline_tracks import compare
    errs = [f"ground rule 3: {d}" for d in compare(meta["final"], base)]
    min_l = meta["end_fid"] - meta["begin_fid"]
    errs += [f"replay: {d}" for d in compare(S.replay_video(G, arr["temp"], min_l), meta["final"])]
    if meta["post_box_mismatch"] or meta["post_cut_mismatch"]:
        errs.append(f"post-processor mismatches {meta['post_box_mismatch']}/{meta['post_cut_mismatch']}")
    if meta["n_post"] != min_l:
        errs.append(f"{meta['n_post']} post-processor calls for {min_l} frames")
    t = arr["temp"]
    if len(t) and len(np.unique(t[:, :2], axis=0)) != len(t):
        errs.append("temp has repeated (frame, id) rows")
    return errs


# =========================================================================== test labels
def open_labels():
    """Opened only after every check above has passed (A25, Order 3)."""
    tc = json.load(open(TC_MISSED))
    rows = [r for r in tc["novel"]["sgdet"]["rows"] if r["bucket"] == "U1"]
    gt_all = json.load(open(os.path.join(META, "test_relation_gt.json")))
    return rows, gt_all


def load_anno(video):
    return json.load(open(os.path.join(ANNO_TEST, video + ".json")))


def gt_object_tracks(anno) -> list:
    """Every GT object track of a test video: one per tid, over the frames where it has a box;
    extent first..last+1; a missing frame inside is empty (EMPTY) for the GT too."""
    cats = {so["tid"]: so["category"] for so in anno["subject/objects"]}
    fb = collections.defaultdict(dict)
    for f, boxes in enumerate(anno["trajectories"]):
        for b in boxes:
            bb = b["bbox"]
            fb[b["tid"]][f] = [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]
    out = []
    for tid in sorted(fb):
        d = fb[tid]
        f0, f1 = min(d), max(d) + 1
        out.append(dict(tid=tid, category=cats[tid], gb=f0, traj=[d.get(f, S.EMPTY) for f in range(f0, f1)],
                        n_boxes=len(d), n_gap_frames=(f1 - f0) - len(d)))
    return out


def keep_filter(obj_split, pred_split):
    tc = {c for c in obj_split if c != "__background__"}
    novel_p = {p for p, s in pred_split.items() if s == "novel"}
    return lambda g: g["triplet"][0] in tc and g["triplet"][1] in novel_p and g["triplet"][2] in tc


def best_pair_min(final: dict, g, viou) -> float:
    """D1's U1 quantity (tc_run.py): max over ordered track pairs s != o of min(sub vIoU, obj vIoU)."""
    gb = g["duration"][0]
    vs = S.best_viou(final, g["sub_traj"], gb, viou)
    vo = S.best_viou(final, g["obj_traj"], gb, viou)
    cand = [(min(vs[s], vo[o]), -s, -o) for s in final for o in final if s != o]
    return max(cand)[0] if cand else 0.0


def join_u1(rows, gt_all, finals: dict, keep, viou) -> list:
    """Each D1 U1 row -> its GT entry (A25). Within a key (video, classes, predicate, duration), GT
    entries are matched to rows by D1's best_pair_min_viou recomputed on the baseline tracks
    (|Δ| < 1e-9), so every row's value is also checked against D1's. Rows equal in key and value
    carry nothing that tells them apart; they take the equally many matching GT entries in file
    order (A25a). Unequal counts -- a row unmatched, or a GT entry left ambiguous -- stop the batch."""
    idx = collections.defaultdict(list)
    for v, gts in gt_all.items():
        for g in gts:
            if keep(g):
                idx[(v, g["triplet"][0], g["triplet"][1], g["triplet"][2], tuple(g["duration"]))].append(g)
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["video"], r["subject"], r["predicate"], r["object"], tuple(r["duration"]))].append(r)
    out = []
    for key, rs in by_key.items():
        cands = idx.get(key, [])
        vals = [best_pair_min(finals[key[0]], g, viou) for g in cands]
        free = list(range(len(cands)))
        by_val = collections.defaultdict(list)
        for r in rs:
            by_val[r["best_pair_min_viou"]].append(r)
        for v, group in by_val.items():
            m = [i for i in free if abs(vals[i] - v) < 1e-9]
            if len(m) != len(group):
                raise SystemExit(f"!! STOP: U1 join: {len(group)} rows but {len(m)} GT entries for {key} "
                                 f"at best_pair_min_viou {v}")
            for r, i in zip(group, m):
                out.append(dict(row=r, gt=cands[i]))
                free.remove(i)
    return out


# =========================================================================== per side
def analyse_side(ds, pre, final, gt_traj, gb, min_l, G, viou) -> dict:
    fin = S.best_viou(final, gt_traj, gb, viou)
    best_final = max(fin.values(), default=0.0)
    row = dict(best_final=best_final, missed=best_final < THR)
    if not row["missed"]:
        return row
    o_raw = {s: S.oracle_viou(lambda f, s=s: ds.raw(f, s), gt_traj, gb, viou)[0] for s in FLOORS}
    row["o1_scores"] = S.oracle_scores(ds, gt_traj, gb, FLOOR)       # A25c, descriptive (Table 6)
    o_cut = S.oracle_viou(ds.cut, gt_traj, gb, viou)[0]
    o_ti, ti_chosen = S.oracle_viou(ds.ti, gt_traj, gb, viou)
    pv = S.best_viou(pre, gt_traj, gb, viou)
    best_tid = min(pv, key=lambda t: (-pv[t], t)) if pv else None
    best_pre = pv[best_tid] if pv else 0.0
    frag = S.fragmentation(ti_chosen, gb, len(gt_traj), pre, gt_traj)
    replay = None
    if best_pre >= THR:
        replay = S.replay_track(G, pre[best_tid]["rows"], min_l)
        if replay["rule"] is None:
            replay["final_viou"] = S.cut_viou(replay["begin_fid"], replay["boxes"], gt_traj, gb, viou)
            replay["in_final"] = any(t["begin_fid"] == replay["begin_fid"] and t["boxes"] == replay["boxes"]
                                     for t in final.values())
            del replay["boxes"]
    row.update(o_raw={str(s): v for s, v in o_raw.items()}, o_cut=o_cut, o_ti=o_ti, best_pre=best_pre,
               best_pre_tid=best_tid, n_pre_tracks=len(pre), frag=frag, replay=replay,
               labels={str(s): S.classify_side(o_raw[s], o_cut, o_ti, best_pre, frag, replay) for s in FLOORS})
    row["label"] = row["labels"][str(FLOOR)]
    return row


def video_inputs(arr, meta, base):
    ds = S.DetSets(arr, meta["w"], meta["h"])
    pre = S.pre_filter_tracks(arr["temp"])
    final = {t["tid"]: t for t in base}
    return ds, pre, final, meta["end_fid"] - meta["begin_fid"]


def analyse_relations(video, joined, arr, meta, base, obj_split, G, viou) -> list:
    ds, pre, final, min_l = video_inputs(arr, meta, base)
    out = []
    for j in joined:
        g = j["gt"]
        gb = g["duration"][0]
        sides = {}
        for side, key, cls in (("subject", "sub_traj", g["triplet"][0]), ("object", "obj_traj", g["triplet"][2])):
            sides[side] = analyse_side(ds, pre, final, g[key], gb, min_l, G, viou)
            sides[side]["class"] = cls
            sides[side]["split"] = obj_split[cls]
        rel = {}
        for s in FLOORS:
            rel[str(s)] = S.relation_label([sd["labels"][str(s)] for sd in sides.values() if sd["missed"]])
        out.append(dict(video=video, triplet=g["triplet"], duration=g["duration"],
                        subject_tid=g["subject_tid"], object_tid=g["object_tid"],
                        best_pair_min_viou=j["row"]["best_pair_min_viou"], sides=sides, labels=rel,
                        label=rel[str(FLOOR)],
                        rel_split="base-only" if all(sd["split"] == "base" for sd in sides.values()) else "involves novel"))
    return out


def analyse_funnel(video, anno, arr, meta, base, obj_split, G, viou) -> list:
    ds, pre, final, _ = video_inputs(arr, meta, base)
    out = []
    for t in gt_object_tracks(anno):
        gb, traj = t["gb"], t["traj"]
        f = dict(F1=S.oracle_viou(lambda fr: ds.raw(fr, FLOOR), traj, gb, viou)[0] >= THR,
                 F2=S.oracle_viou(ds.cut, traj, gb, viou)[0] >= THR,
                 F3=S.oracle_viou(ds.ti, traj, gb, viou)[0] >= THR,
                 F4=max(S.best_viou(pre, traj, gb, viou).values(), default=0.0) >= THR,
                 F5=max(S.best_viou(final, traj, gb, viou).values(), default=0.0) >= THR)
        cum, ok = {}, True
        for k in ("F1", "F2", "F3", "F4", "F5"):
            ok = ok and f[k]
            cum[k] = ok
        out.append(dict(video=video, tid=t["tid"], category=t["category"], split=obj_split[t["category"]],
                        n_frames=len(traj), n_gap_frames=t["n_gap_frames"], stage=f, cumulative=cum))
    return out


# =========================================================================== tables
def _cell(mask, vids):
    mask = [bool(m) for m in mask]
    n = len(mask)
    c = sum(mask)
    return dict(count=c, share=(c / n) if n else None, ci95=S.boot_share(mask, vids) if n else (None, None))


def stage_table(items, label_of):
    """items: dicts with 'video'; label_of(item) -> (stage, sub)."""
    vids = [it["video"] for it in items]
    labs = [label_of(it) for it in items]
    t = dict(n=len(items), top={}, sub={})
    for st in TOP:
        t["top"][st] = _cell([l[0] == st for l in labs], vids)
    for k in S.ORDER:
        if k[1] is not None:
            t["sub"][f"{k[0]}:{k[1]}"] = _cell([l == k for l in labs], vids)
    return t


def _q(xs):
    """median, p25, p75 of a list (None if empty)."""
    if not xs:
        return dict(median=None, p25=None, p75=None)
    return dict(median=float(np.median(xs)), p25=float(np.percentile(xs, 25)), p75=float(np.percentile(xs, 75)))


def o1b_score_table(sides: list) -> dict:
    """A25c, descriptive only: for each O1b side, the cut scores of the raw(0.05) oracle's boxes.
    Per side: the median score over frames with an oracle box, and the share of the GT extent's
    frames whose oracle box scores >= 0.1 / 0.2 / 0.3 (a frame with no box counts as below), plus
    the share whose box the cut selected. Summarised across sides (median, IQR) and pooled over
    all their GT frames; all / base / novel."""
    out = {}
    for split in ("all", "base", "novel"):
        ss = [s for s in sides if s["label"][0] == "O1b" and (split == "all" or s["split"] == split)]
        per, pooled_scores, pooled_n = [], [], 0
        pooled = {t: 0 for t in SCORE_LEVELS} | {"selected": 0}
        for s in ss:
            sc = s["o1_scores"]
            have = [x for x, _ in sc if x is not None]
            n = len(sc)
            d = dict(median=float(np.median(have)) if have else None,
                     **{f"ge_{t}": sum(1 for x, _ in sc if x is not None and x >= t) / n for t in SCORE_LEVELS},
                     selected=sum(1 for _, sel in sc if sel) / n)
            per.append(d)
            pooled_scores += have
            pooled_n += n
            for t in SCORE_LEVELS:
                pooled[t] += sum(1 for x, _ in sc if x is not None and x >= t)
            pooled["selected"] += sum(1 for _, sel in sc if sel)
        out[split] = dict(
            n_sides=len(ss), n_frames=pooled_n,
            across_sides={k: _q([d[k] for d in per if d[k] is not None])
                          for k in ["median"] + [f"ge_{t}" for t in SCORE_LEVELS] + ["selected"]},
            pooled=dict(median=float(np.median(pooled_scores)) if pooled_scores else None,
                        **{f"ge_{t}": (pooled[t] / pooled_n) if pooled_n else None for t in SCORE_LEVELS},
                        selected=(pooled["selected"] / pooled_n) if pooled_n else None))
    return out


def frame_range_table(metas: list) -> dict:
    """A25c finding, from the recordings' own metadata only (no label file): the frames each
    video's processed range [begin_fid, end_fid) -- the GT object extent range, dataset.py:120-126 --
    leaves out of the video's length."""
    rows = []
    for m in metas:
        lo, hi, n = m["begin_fid"], m["end_fid"], m["video_len"]
        inside = max(0, min(hi, n) - max(lo, 0))
        rows.append(dict(video=m["video"], video_len=n, begin_fid=lo, end_fid=hi, outside=n - inside,
                         share=(n - inside) / n, range_beyond_video=lo < 0 or hi > n))
    shares = [r["share"] for r in rows]
    worst = max(rows, key=lambda r: r["share"]) if rows else None
    tot, out_ = sum(r["video_len"] for r in rows), sum(r["outside"] for r in rows)
    return dict(n_videos=len(rows), videos_with_frames_outside=sum(1 for r in rows if r["outside"] > 0),
                share_median=float(np.median(shares)) if rows else None,
                share_p75=float(np.percentile(shares, 75)) if rows else None,
                share_max=worst["share"] if worst else None, share_max_video=worst["video"] if worst else None,
                frames_total=tot, frames_outside=out_, share_total=(out_ / tot) if tot else None,
                ranges_beyond_video=sum(r["range_beyond_video"] for r in rows), per_video=rows)


def tables(relations: list, funnel: list, metas: list | None = None) -> dict:
    sides = [dict(video=r["video"], **sd) for r in relations for sd in r["sides"].values() if sd["missed"]]
    T = {}
    T["t1"] = dict(relation=stage_table(relations, lambda r: tuple(r["label"])),
                   side=stage_table(sides, lambda s: tuple(s["label"])))
    T["t2"] = dict(relation={k: stage_table([r for r in relations if r["rel_split"] == k], lambda r: tuple(r["label"]))
                             for k in ("base-only", "involves novel")},
                   side={k: stage_table([s for s in sides if s["split"] == k], lambda s: tuple(s["label"]))
                         for k in ("base", "novel")})
    o3 = [s for s in sides if s["label"][0] == "O3"]
    T["t3"] = dict(n_missed_sides=len(sides), n_o3=len(o3),
                   rules={k[1]: dict(count=sum(tuple(s["label"]) == k for s in o3),
                                     **{"share_of_missed_sides": _cell([tuple(s["label"]) == k for s in sides],
                                                                       [s["video"] for s in sides])})
                          for k in S.ORDER if k[0] == "O3"},
                   f_len_after_trim=sum(1 for s in o3 if s["label"][1] == "F-len" and s["replay"]["trimmed"]))
    T["t4"] = {str(f): dict(relation=stage_table(relations, lambda r, f=f: tuple(r["labels"][str(f)]))["top"],
                            side=stage_table(sides, lambda s, f=f: tuple(s["labels"][str(f)]))["top"])
               for f in FLOORS}
    fun = {}
    for k in ("all", "base", "novel"):
        sel = [x for x in funnel if k == "all" or x["split"] == k]
        vids = [x["video"] for x in sel]
        fun[k] = dict(n=len(sel), **{st: _cell([x["cumulative"][st] for x in sel], vids)
                                     for st in ("F1", "F2", "F3", "F4", "F5")})
    fun["non_monotone"] = sum(1 for x in funnel if any(x["stage"][b] and not x["stage"][a]
                                                       for a, b in zip(("F1", "F2", "F3", "F4"), ("F2", "F3", "F4", "F5"))))
    T["t5"] = fun
    T["t6"] = o1b_score_table(sides)
    if metas is not None:
        T["t7"] = frame_range_table(metas)
    T["checks"] = dict(
        o4_missed_sides=sum(1 for s in sides if s["label"][0] == "O4"),
        s0_relations=sum(1 for r in relations if r["label"][0] == "S0"),
        upstream_label_but_tracked=sum(1 for s in sides if s["label"][0] in ("O1a", "O1b") and s["best_pre"] >= THR),
        replay_kept_not_in_final=sum(1 for s in sides if s.get("replay") and s["replay"]["rule"] is None
                                     and not s["replay"]["in_final"]))
    return T


def reading(relations: list) -> dict:
    vids = [r["video"] for r in relations]
    masks = {g: [r["label"][0] in st for r in relations] for g, st in S.GROUPS.items()}
    n = len(relations)
    g = {k: dict(count=sum(m), share=sum(m) / n if n else None, ci95=S.boot_share(m, vids)) for k, m in masks.items()}
    order = sorted(g, key=lambda k: (-g[k]["count"], list(S.GROUPS).index(k)))
    top = [k for k in order if g[k]["count"] == g[order[0]]["count"]]   # a tie in count names all of them
    first, second = order[0], order[1]
    diff = S.boot_diff(masks[first], masks[second], vids)
    separated = not (diff[0] <= 0 <= diff[1])
    pct = lambda k: f"{g[k]['count']} ({100 * g[k]['share']:.1f}%, 95% CI {100 * g[k]['ci95'][0]:.1f}–{100 * g[k]['ci95'][1]:.1f}%)"
    if len(top) > 1:
        line = (f"By relation, pooled: tied largest groups " + " and ".join(f"{k} {pct(k)}" for k in top)
                + " → " + "; ".join(S.MODULE[k] for k in top) + ".")
    else:
        line = (f"By relation, pooled: the largest group is {first}, {pct(first)} → {S.MODULE[first]}. "
                f"Second: {second}, {pct(second)}" + ("" if separated else f"; not separated from {second}") + ".")
    return dict(n=n, groups=g, order=order, largest=top, second=second, diff_ci95=diff,
                separated=separated, line=line)


# =========================================================================== markdown
def _fmt(c):
    if c["share"] is None:
        return "—"
    lo, hi = c["ci95"]
    return f"{c['count']} ({100 * c['share']:.1f}%, {100 * lo:.1f}–{100 * hi:.1f})"


def render_md(T: dict, R: dict) -> str:
    L = []
    for unit in ("relation", "side"):
        t = T["t1"][unit]
        L += [f"**Table 1 — by {unit}** (n = {t['n']}; count (share, 95% video-bootstrap CI))", "",
              "| " + " | ".join(TOP) + " |", "|" + "---|" * len(TOP),
              "| " + " | ".join(_fmt(t["top"][s]) for s in TOP) + " |", "",
              "| sub-label | " + unit + " |", "|---|---|"]
        L += [f"| {k} | {_fmt(c)} |" for k, c in t["sub"].items() if c["count"]]
        L.append("")
    for unit, keys in (("relation", ("base-only", "involves novel")), ("side", ("base", "novel"))):
        L += [f"**Table 2 — by {unit}, by class**", "", "| | n | " + " | ".join(TOP) + " |",
              "|---|---:|" + "---|" * len(TOP)]
        for k in keys:
            t = T["t2"][unit][k]
            L.append(f"| {k} | {t['n']} | " + " | ".join(_fmt(t["top"][s]) for s in TOP) + " |")
        L.append("")
    t3 = T["t3"]
    L += [f"**Table 3 — O3 by rule** ({t3['n_o3']} of {t3['n_missed_sides']} missed sides; "
          f"F-len after a gap trim: {t3['f_len_after_trim']})", "", "| rule | count | share of missed sides |", "|---|---:|---|"]
    L += [f"| {k} | {v['count']} | {_fmt(v['share_of_missed_sides'])} |" for k, v in t3["rules"].items()]
    L += ["", "**Table 4 — O1a floor sensitivity**", "", "| floor | unit | " + " | ".join(TOP) + " |",
          "|---|---|" + "---|" * len(TOP)]
    for f, d in T["t4"].items():
        for unit in ("relation", "side"):
            L.append(f"| {f} | {unit} | " + " | ".join(_fmt(d[unit][s]) for s in TOP) + " |")
    L += ["", "**Table 5 — funnel, all test GT object tracks (cumulative, vIoU ≥ 0.5)**", "",
          "| | n | F1 detected (≥ 0.05) | F2 passes the cut | F3 passes the dedupe | F4 tracked | F5 kept |",
          "|---|---:|---|---|---|---|---|"]
    for k in ("all", "base", "novel"):
        d = T["t5"][k]
        L.append(f"| {k} | {d['n']} | " + " | ".join(_fmt(d[s]) for s in ("F1", "F2", "F3", "F4", "F5")) + " |")
    L += ["", f"Non-monotone funnel tracks (a later stage passes where an earlier one fails): {T['t5']['non_monotone']}"]
    pc = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    nm = lambda x: "—" if x is None else f"{x:.3f}"
    L += ["", "**Table 6 — O1b sides: cut scores of the raw(0.05) oracle's boxes** (A25c, descriptive only)", "",
          "| | O1b sides | GT frames | median score: pooled / across sides (IQR) | ≥ 0.1 | ≥ 0.2 | ≥ 0.3 | selected by the cut |",
          "|---|---:|---:|---|---|---|---|---|"]
    for k in ("all", "base", "novel"):
        d = T["t6"][k]
        a, p = d["across_sides"], d["pooled"]
        cells = [f"{pc(p[f'ge_{t}'])} / {pc(a[f'ge_{t}']['median'])}" for t in SCORE_LEVELS]
        L.append(f"| {k} | {d['n_sides']} | {d['n_frames']} | {nm(p['median'])} / {nm(a['median']['median'])} "
                 f"({nm(a['median']['p25'])}–{nm(a['median']['p75'])}) | " + " | ".join(cells)
                 + f" | {pc(p['selected'])} / {pc(a['selected']['median'])} |")
    L += ["", "Shares: pooled over all GT frames / median across sides; a GT frame with no oracle box counts as below."]
    if "t7" in T:
        t7 = T["t7"]
        L += ["", "**Table 7 — frames outside the GT-extent range** (A25c finding; recordings' metadata only)", "",
              "| videos | with frames outside | share outside: median / p75 / max (video) | frames outside / total |",
              "|---:|---:|---|---|",
              f"| {t7['n_videos']} | {t7['videos_with_frames_outside']} | {pc(t7['share_median'])} / {pc(t7['share_p75'])} / "
              f"{pc(t7['share_max'])} ({t7['share_max_video']}) | {t7['frames_outside']:,} / {t7['frames_total']:,} "
              f"({pc(t7['share_total'])}) |"]
    L += ["", f"Checks: {json.dumps(T['checks'])}", "", "**Reading (A25):** " + R["line"],
          f"Paired 95% CI of (largest − second) share: {R['diff_ci95']}", ""]
    return "\n".join(L)


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_dir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    use_code_root()
    viou, _iou, G = S.repo()

    # ---- 1. verification; no test label is opened before all of it passes
    pins = verify_pins()
    n_checked = verify_cache_and_anno()
    if n_checked != dict(cache_files=204, anno_files=200):
        raise SystemExit(f"!! STOP: checked {n_checked}, expected 204 cache and 200 annotation files")
    cp = verify_code_pins()
    from d2.baseline_tracks import OUT as BASE, load
    if sha256(BASE) != cp["baseline_tracks_json"]:
        raise SystemExit("!! STOP: baseline_tracks.json differs from A25a's pin")
    base = load(BASE)
    videos = sorted(base)
    man = verify_recordings(a.rec_dir, videos)
    metas = []
    for v in videos:
        arr, meta = load_recording(a.rec_dir, v)
        errs = check_video(G, arr, meta, base[v])
        if errs:
            raise SystemExit(f"!! STOP: {v}: {errs[:5]}")
        metas.append({k: meta[k] for k in ("video", "video_len", "begin_fid", "end_fid")})
    print(f"verified: pins, code, baseline, {len(videos)} recordings, ground rule 3 and replay on every video", flush=True)

    # ---- 2. test labels
    rows, gt_all = open_labels()
    obj_split = json.load(open(os.path.join(META, "openvoc_obj_class_spilt_info.json")))["cls2split"]
    pred_split = json.load(open(os.path.join(META, "openvoc_pred_class_spilt_info.json")))["cls2split"]
    finals = {v: {t["tid"]: t for t in base[v]} for v in videos}
    joined = join_u1(rows, gt_all, finals, keep_filter(obj_split, pred_split), viou)
    by_video = collections.defaultdict(list)
    for j in joined:
        by_video[j["row"]["video"]].append(j)
    relations, funnel = [], []
    for v in videos:
        arr, meta = load_recording(a.rec_dir, v)
        if by_video.get(v):
            relations += analyse_relations(v, by_video[v], arr, meta, base[v], obj_split, G, viou)
        funnel += analyse_funnel(v, load_anno(v), arr, meta, base[v], obj_split, G, viou)
    T = tables(relations, funnel, metas)
    R = reading(relations)
    md = render_md(T, R)

    os.makedirs(DATA, exist_ok=True)
    outs = dict(relations=os.path.join(DATA, "d2_relations.json"), funnel=os.path.join(DATA, "d2_funnel.json"),
                tables=os.path.join(DATA, "d2_tables.json"), md=os.path.join(DATA, "d2_tables.md"))
    json.dump(relations, open(outs["relations"], "w"), default=str)
    json.dump(funnel, open(outs["funnel"], "w"))
    json.dump(dict(tables=T, reading=R, u1_rows=len(rows)), open(outs["tables"], "w"), indent=1, default=str)
    open(outs["md"], "w").write(md)
    write_log("d2_analyse", dict(thr=THR, floor=FLOOR, floors=FLOORS, rec_dir=a.rec_dir),
              [os.path.join(THESIS, p) for p in pins] + [BASE, CODE_PINS, os.path.join(a.rec_dir, "manifest.jsonl")],
              list(outs.values()), extra=dict(seconds=round(time.time() - t0, 1), videos=len(videos),
                                              u1_rows=len(rows), relations=len(relations), gt_tracks=len(funnel)))
    print(md)


if __name__ == "__main__":
    main()
