#!/usr/bin/env python3
"""Post-G1 Task 2 (A30): the SAM 3 screen on HELD -- exploration, the last chance for SAM 3 as a detector.
CPU only; the test-label guard is installed first (by importing G1's analyse); HELD training data only.

    python studies/postg1/sam3_screen.py --out studies/postg1/results_screen

Each rule removes whole SAM 3 confident tracks (G1's adapter, 157 HELD videos). Every rule is judged
on the video's full set of confident tracks; a configuration removes the union of what its rules
flag. The kept tracks are then scored with A29's own code (studies/rpt/g1/metrics.py: matching,
D1's best pair, unmatched tracks) against EOV-MMP R0w on the same 157 videos.

Boxes are the confident track's per-frame boxes as G1 used them (the tight box of SAM 3's mask,
gap frames interpolated, rounded by the repo's round_and_positive), integer pixel indices with the
official iou's convention: width x2 - x1 + 1, height y2 - y1 + 1.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import itertools
import json
import os
import statistics
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
G1 = os.path.join(REPO, "studies", "rpt", "g1")
sys.path.insert(0, G1)

import analyse as AN  # noqa: E402  (installs the test-label guard)
import metrics as M  # noqa: E402
from eov_held import held_relations  # noqa: E402

# ---------------------------------------------------------------- A30's grid (fixed)
TINY = (8, 12, 16, 24)                                  # rule 1: t px
EDGE = tuple(itertools.product((0.5, 0.8), (16, 32)))   # rule 2: (f, w)
FRAG = tuple(itertools.product((0.8, 0.9), (0.5, 0.8)))  # rule 3: (c, f)
SCORE = (0.6, 0.7, 0.8, 0.9)                            # rule 4: s (reference only)
PARTNER = (1, 2, 3)                                     # rule 5: d
EDGE_PX = 2
PARTNER_OVERLAP = 10
RECALL_BAR, PAIR_BAR, BURDEN_BAR = 8.0, 8.0, 1.35


# ---------------------------------------------------------------- per-frame geometry
def wh(b):
    return b[2] - b[0] + 1, b[3] - b[1] + 1


def area(b):
    w, h = wh(b)
    return w * h


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1) * max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def diag(b):
    w, h = wh(b)
    return (w * w + h * h) ** 0.5


def box_at(t, f):
    return t["boxes"][f - t["begin_fid"]]


def shared_frames(a, b):
    return range(max(a["begin_fid"], b["begin_fid"]), min(a["end_fid"], b["end_fid"]))


# ---------------------------------------------------------------- the rules (each -> set of removed tids)
def rule_tiny(tracks, t):
    """Rule 1: min(median box width, median box height) < t px."""
    out = set()
    for tid, tr in tracks.items():
        ws, hs = zip(*(wh(b) for b in tr["boxes"]))
        if min(statistics.median(ws), statistics.median(hs)) < t:
            out.add(tid)
    return out


def rule_edge(tracks, width, f, w):
    """Rule 2: the box touches the left or right frame edge (within 2 px) on >= f of the track's
    frames, and its median width < w px. Pixel indices run 0 .. width - 1."""
    out = set()
    for tid, tr in tracks.items():
        touch = [b[0] <= EDGE_PX or b[2] >= width - 1 - EDGE_PX for b in tr["boxes"]]
        if sum(touch) / len(touch) >= f and statistics.median(wh(b)[0] for b in tr["boxes"]) < w:
            out.add(tid)
    return out


def rule_fragment(tracks, c, f):
    """Rule 3: on >= f of the frames it shares with some larger confident track (any class, larger
    by median box area), >= c of its box area lies inside that track's box."""
    med = {tid: statistics.median(area(b) for b in tr["boxes"]) for tid, tr in tracks.items()}
    out = set()
    for tid, tr in tracks.items():
        for oid, ot in tracks.items():
            if oid == tid or med[oid] <= med[tid]:
                continue
            fs = shared_frames(tr, ot)
            if not len(fs):
                continue
            inside = sum(inter(box_at(tr, x), box_at(ot, x)) / area(box_at(tr, x)) >= c for x in fs)
            if inside / len(fs) >= f:
                out.add(tid)
                break
    return out


def rule_score(tracks, s):
    """Rule 4 (reference only): track score < s."""
    return {tid for tid, tr in tracks.items() if tr["score"] < s}


def rule_partner(tracks, d):
    """Rule 5: kept only if some other confident track overlaps it in time by >= 10 frames and, on
    >= half of those frames, their box centres are within d x the larger box's diagonal."""
    out = set()
    for tid, tr in tracks.items():
        ok = False
        for oid, ot in tracks.items():
            if oid == tid:
                continue
            fs = shared_frames(tr, ot)
            if len(fs) < PARTNER_OVERLAP:
                continue
            near = 0
            for x in fs:
                a, b = box_at(tr, x), box_at(ot, x)
                (ax, ay), (bx, by) = centre(a), centre(b)
                near += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= d * max(diag(a), diag(b))
            if near >= len(fs) / 2:
                ok = True
                break
        if not ok:
            out.add(tid)
    return out


def rule_gate(tracks, eov_classes):
    """Rule 7: kept only if EOV-MMP's R0w tracks contain its class in the same video."""
    return {tid for tid, tr in tracks.items() if tr["category"] not in eov_classes}


# ---------------------------------------------------------------- scoring a subset with A29's code
class Video:
    """One video's precomputed A29 inputs; score(kept) equals A29's matching, best pair and unmatched
    count on the kept tracks (tests: test_sam3_screen.py; the run checks it again on real videos)."""

    def __init__(self, tracks, runs, rels):
        self.tracks, self.runs, self.rels = tracks, runs, rels
        self.tids = sorted(tracks)
        cands = []
        for t in tracks.values():
            for r in runs:
                if t["category"] == r["category"]:
                    v = M.cut_viou(t, r)
                    if v >= M.THR:
                        cands.append((-v, t["tid"], r["rid"]))
        self.cands = sorted(cands)
        self.gt_of = {r["rid"]: r["gt_tid"] for r in runs}
        self.vs, self.vo = [], []
        for rel in rels:
            gb = rel["duration"][0]
            s = M.D1.best_viou(tracks, rel["sub_traj"], gb, M.viou)
            o = M.D1.best_viou(tracks, rel["obj_traj"], gb, M.viou)
            self.vs.append(np.array([s[t] for t in self.tids]))
            self.vo.append(np.array([o[t] for t in self.tids]))
        self.cat = [tracks[t]["category"] for t in self.tids]
        self.memo = {}

    def match(self, kept):
        run_to, track_gt = {}, {}
        for nv, tid, rid in self.cands:
            if tid not in kept or rid in run_to:
                continue
            if tid in track_gt and track_gt[tid] != self.gt_of[rid]:
                continue
            run_to[rid] = (tid, -nv)
            track_gt[tid] = self.gt_of[rid]
        return run_to

    def bucket(self, k, idx):
        if len(idx) < 2:
            return "U1"
        A = np.minimum.outer(self.vs[k][idx], self.vo[k][idx])
        np.fill_diagonal(A, -np.inf)
        m = A.max()
        if m < M.THR:
            return "U1"
        i, j = np.argwhere(A == m)[0]                  # row-major: lowest s, then lowest o (tids ascending)
        s, o = idx[i], idx[j]
        rel = self.rels[k]
        return "U2" if (self.cat[s] != rel["triplet"][0] or self.cat[o] != rel["triplet"][2]) else "recalled"

    def score(self, kept: frozenset):
        if kept not in self.memo:
            run_to = self.match(kept)
            used = {tid for tid, _ in run_to.values()}
            idx = np.array([i for i, t in enumerate(self.tids) if t in kept], dtype=int)
            recalled = sum(self.bucket(k, idx) == "recalled" for k in range(len(self.rels)))
            self.memo[kept] = dict(matched_runs=len(run_to), recalled=recalled,
                                   unmatched=sorted(kept - used), matched_tracks=sorted(used))
        return self.memo[kept]


# ---------------------------------------------------------------- configurations
def configs():
    """The 1,000 grid configurations: rules 1, 2, 3, 5 and 7, each off (None) or at a setting."""
    for t, e, fr, d, g in itertools.product((None,) + TINY, (None,) + EDGE, (None,) + FRAG, (None,) + PARTNER,
                                            (False, True)):
        yield dict(tiny=t, edge=e, fragment=fr, partner=d, gate=g)


def n_rules(cfg):
    return sum(v not in (None, False) for v in cfg.values())


def label(cfg):
    parts = []
    if cfg["tiny"] is not None:
        parts.append(f"tiny<{cfg['tiny']}")
    if cfg["edge"] is not None:
        parts.append(f"edge(f>={cfg['edge'][0]},w<{cfg['edge'][1]})")
    if cfg["fragment"] is not None:
        parts.append(f"fragment(c>={cfg['fragment'][0]},f>={cfg['fragment'][1]})")
    if cfg["partner"] is not None:
        parts.append(f"partner(d={cfg['partner']})")
    if cfg["gate"]:
        parts.append("gate")
    return " + ".join(parts) or "none"


def removed_by(cfg, flags):
    r = set()
    if cfg["tiny"] is not None:
        r |= flags["tiny", cfg["tiny"]]
    if cfg["edge"] is not None:
        r |= flags["edge", cfg["edge"]]
    if cfg["fragment"] is not None:
        r |= flags["fragment", cfg["fragment"]]
    if cfg["partner"] is not None:
        r |= flags["partner", cfg["partner"]]
    if cfg["gate"]:
        r |= flags["gate", True]
    return r


def select(rows):
    """A30's selection rule among passing configurations with the class gate on: the highest
    min(recall gain, pair-recall gain); ties by the lower burden, then fewer rules on, then grid order."""
    ok = [r for r in rows if r["pass"] and r["config"]["gate"]]
    if not ok:
        return None
    return min(ok, key=lambda r: (-min(r["recall_gain"], r["pair_gain"]), r["burden"], n_rules(r["config"]), r["index"]))


def pareto(rows):
    """Configurations no other beats on both burden (lower) and min(recall gain, pair gain) (higher)."""
    pts = sorted(rows, key=lambda r: (r["burden"], -min(r["recall_gain"], r["pair_gain"]), r["index"]))
    out, best = [], -np.inf
    for r in pts:
        g = min(r["recall_gain"], r["pair_gain"])
        if g > best:
            out.append(r)
            best = g
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-configs", type=int, default=25, help="configs re-scored with metrics.py itself")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.chdir(REPO)
    from utils import paths
    sam3_dir, eov_dir = os.path.join(G1, "sam3_out"), os.path.join(G1, "eov_out")
    held = sorted(json.load(open("pilot_analysis/stage1/train_split.json"))["held"])
    videos = [v for v in held if os.path.exists(os.path.join(sam3_dir, f"{v}.json.gz"))]
    assert len(videos) == 157, len(videos)
    rels_all = held_relations()
    option_b = json.load(open(AN.GT_TRACKS))
    novel = {c for c, s in json.load(open("data/vidvrd/data/openvoc_obj_class_spilt_info.json"))["cls2split"].items()
             if s == "novel"}
    obs = json.load(open(os.path.join(G1, "results_descriptive", "descriptive.json")))["viz_sample"]["drawn"]
    inspected = {(x["video"], x["tid"]): x["image"][:2] for x in obs}

    V, flags, eov_un, eov_runs, eov_rec, files, n_runs, n_rels = {}, {}, 0, 0, 0, {}, 0, 0
    for i, v in enumerate(videos):
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        anno.pop("relation_instances", None)
        runs = M.runs_of(option_b[v], anno)
        sp, ep = os.path.join(sam3_dir, f"{v}.json.gz"), os.path.join(eov_dir, f"{v}.json.gz")
        files[v] = dict(sam3=AN.sha256(sp), eov=AN.sha256(ep))
        s3 = AN.load_json_gz(sp)
        _, conf, _ = AN.sam3_tracks(s3)
        e = AN.load_json_gz(ep)
        if e["sam3_file_sha256"] != files[v]["sam3"]:
            raise SystemExit(f"!! STOP: {v}: the EOV-side run read another SAM 3 file")
        eov = M.tracks_from_list(e["configs"]["R0w"]["tracks"])
        er = M.video_record(eov, runs, rels_all[v])
        eov_un += len(er["unmatched"])
        eov_runs += len(er["matched"])
        eov_rec += sum(p["bucket"] == "recalled" for p in er["pairs"])
        n_runs += len(runs)
        n_rels += len(rels_all[v])
        V[v] = Video(conf, runs, rels_all[v])
        W = s3["width"]
        f = {("tiny", t): rule_tiny(conf, t) for t in TINY}
        f |= {("edge", x): rule_edge(conf, W, *x) for x in EDGE}
        f |= {("fragment", x): rule_fragment(conf, *x) for x in FRAG}
        f |= {("score", s): rule_score(conf, s) for s in SCORE}
        f |= {("partner", d): rule_partner(conf, d) for d in PARTNER}
        f[("gate", True)] = rule_gate(conf, {t["category"] for t in eov.values()})
        flags[v] = {k: frozenset(x) for k, x in f.items()}
        print(f"[{i + 1}/{len(videos)}] {v}: {len(conf)} confident, {len(runs)} runs, {len(rels_all[v])} relations",
              flush=True)

    n_v = len(videos)
    eov_recall, eov_pair, eov_burden = eov_runs / n_runs, eov_rec / n_rels, eov_un / n_v
    base = {v: V[v].score(frozenset(V[v].tids)) for v in videos}
    base_matched = {(v, t) for v in videos for t in base[v]["matched_tracks"]}
    base_unmatched = {(v, t) for v in videos for t in base[v]["unmatched"]}

    def evaluate(removed):
        """removed: {video: set of tids} -> the three criterion quantities and track counts."""
        tot = Counter()
        for v in videos:
            s = V[v].score(frozenset(V[v].tids) - frozenset(removed.get(v, ())))
            tot["runs"] += s["matched_runs"]
            tot["rec"] += s["recalled"]
            tot["un"] += len(s["unmatched"])
        rm = {(v, t) for v, ts in removed.items() for t in ts}
        recall, pair, burden_v = tot["runs"] / n_runs, tot["rec"] / n_rels, tot["un"] / n_v
        return dict(recall=recall, pair=pair, unmatched_per_video=burden_v,
                    recall_gain=100 * (recall - eov_recall), pair_gain=100 * (pair - eov_pair),
                    burden=burden_v / eov_burden, removed=len(rm), matched_tracks_lost=len(rm & base_matched),
                    unmatched_tracks_removed=len(rm & base_unmatched),
                    inspected_removed=sorted(inspected[x] for x in rm if x in inspected))

    def passes(r):
        return r["recall_gain"] >= RECALL_BAR and r["pair_gain"] >= PAIR_BAR and r["burden"] <= BURDEN_BAR

    rows = []
    for idx, cfg in enumerate(configs()):
        r = evaluate({v: removed_by(cfg, flags[v]) for v in videos})
        r.update(index=idx, config=cfg, label=label(cfg), rules_on=n_rules(cfg))
        r["pass"] = passes(r)
        rows.append(r)
    assert len(rows) == 1000
    none = rows[0]
    singles = []
    for key in ([("tiny", t) for t in TINY] + [("edge", x) for x in EDGE] + [("fragment", x) for x in FRAG]
                + [("score", s) for s in SCORE] + [("partner", d) for d in PARTNER] + [("gate", True)]):
        r = evaluate({v: flags[v][key] for v in videos})
        r.update(rule=key[0], setting=key[1], pass_=passes(r))
        singles.append(r)

    # A29's own code on the kept tracks, for a sample of configurations (the subset scorer's check)
    rng = np.random.default_rng(0)
    checks = []
    for idx in sorted(set([0, 999] + rng.choice(1000, size=a.check_configs, replace=False).tolist())):
        cfg = rows[idx]["config"]
        tot = Counter()
        for v in videos:
            kept = {t: V[v].tracks[t] for t in V[v].tids if t not in removed_by(cfg, flags[v])}
            rec = M.video_record(kept, V[v].runs, V[v].rels)
            tot["runs"] += len(rec["matched"])
            tot["rec"] += sum(p["bucket"] == "recalled" for p in rec["pairs"])
            tot["un"] += len(rec["unmatched"])
        same = (100 * (tot["runs"] / n_runs - eov_recall) == rows[idx]["recall_gain"]
                and 100 * (tot["rec"] / n_rels - eov_pair) == rows[idx]["pair_gain"]
                and tot["un"] / n_v / eov_burden == rows[idx]["burden"])
        checks.append(dict(index=idx, label=rows[idx]["label"], identical=bool(same)))
    if not all(c["identical"] for c in checks):
        raise SystemExit(f"!! STOP: the subset scorer disagrees with metrics.py: {checks}")

    # the class gate's split: tracks it removes, by novel / base class and matched / unmatched (no other rule)
    gate = Counter()
    for v in videos:
        for t in flags[v][("gate", True)]:
            cls = "novel" if V[v].tracks[t]["category"] in novel else "base"
            gate[cls, "matched" if (v, t) in base_matched else "unmatched"] += 1
    gate_by_class = Counter(V[v].tracks[t]["category"] for v in videos for t in flags[v][("gate", True)])

    sel = select(rows)
    pass_gate = [r for r in rows if r["pass"] and r["config"]["gate"]]
    pass_nogate = [r for r in rows if r["pass"] and not r["config"]["gate"]]
    res = dict(
        label="Post-G1 Task 2 (A30): SAM 3 screen on 157 HELD videos; exploration; HELD in-sample; not a test result",
        verdict=dict(passes_with_gate=len(pass_gate), passes_without_gate=len(pass_nogate),
                     fragile=0 < len(pass_gate) <= 2, selected=sel and dict(index=sel["index"], label=sel["label"]),
                     decision=("pass: the selected configuration becomes Option 2 (validated later on fresh FIT)"
                               if sel else "fail: SAM 3 closes as a detector")),
        criterion=dict(recall_gain_min=RECALL_BAR, pair_gain_min=PAIR_BAR, burden_max=BURDEN_BAR, gate_required=True),
        eov_R0w=dict(recall=eov_recall, pair_recall=eov_pair, unmatched_per_video=eov_burden,
                     runs_matched=eov_runs, runs=n_runs, relations_recalled=eov_rec, relations=n_rels),
        no_rules=dict((k, none[k]) for k in ("recall", "pair", "unmatched_per_video", "recall_gain", "pair_gain", "burden")),
        configurations_tried=len(rows),
        selected=sel,
        passing_with_gate=sorted(pass_gate, key=lambda r: (-min(r["recall_gain"], r["pair_gain"]), r["burden"])),
        passing_without_gate=sorted(pass_nogate, key=lambda r: (-min(r["recall_gain"], r["pair_gain"]), r["burden"])),
        pareto_all=pareto(rows), pareto_with_gate=pareto([r for r in rows if r["config"]["gate"]]),
        single_rules=singles,
        rule6="skipped: G1 cached modelC scores only for SAM 3 pairs that recall a GT relation (box sensitivity), "
              "not for every pair of confident tracks; no features computed",
        class_gate_split={f"{c}/{m}": gate[c, m] for c in ("novel", "base") for m in ("matched", "unmatched")},
        class_gate_by_class=dict(gate_by_class.most_common()),
        inspected_20={img: list(k) for k, img in inspected.items()},
        scorer_checks=checks,
        all_configurations=rows,
        files=files)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sam3_screen.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=list)
    print(json.dumps(res["verdict"], indent=1))
    print(json.dumps(res["no_rules"], indent=1))


if __name__ == "__main__":
    main()
