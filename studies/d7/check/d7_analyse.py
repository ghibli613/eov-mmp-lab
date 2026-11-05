#!/usr/bin/env python3
"""The D7 check's analysis on the laptop (A34): class-agnostic measures for four systems on the final list.

    python studies/d7/check/d7_analyse.py --box studies/d7/check/box_out    # the copied-back folder

Systems: arm A ("object"), arm B (person, animal, vehicle, object; merged by A29's duplicate rule), the
35-name reference (G1's files if check 0 was bitwise, else the machine's rerun), EOV-MMP R0w (G1's files).
Measures, all class-agnostic with A29's matching (metrics.py): object recall (overall, base, novel), pair
recall (not U1), burden (unmatched confident tracks per video), recall at an equal track budget (each
system cut by A29's track score to the reference's confident-track count, per video), and per pass:
seconds per frame, masklets, peak GPU memory. The rule: the better arm (higher recall gain over R0w)
continues D7 if its gain is at least half of the reference's gain. The HELD-B guard is installed first.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
D7 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(D7))
for p in (HERE, D7, os.path.join(REPO, "studies", "rpt"), os.path.join(REPO, "studies", "rpt", "g1"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
import held_b_guard  # noqa: E402
held_b_guard.install()

import classfree as CF  # noqa: E402

SAM3_DIR, EOV_DIR = "studies/rpt/g1/sam3_out", "studies/rpt/g1/eov_out"
EXTRACTS = "studies/d7/extracts"
OPT_B = f"{EXTRACTS}/pilot_analysis__stage1c__gt_tracks_train.held_a.json"
HELD_GT = f"{EXTRACTS}/pilot_analysis__stage2__data__held_gt.held_a.json"
HELD_GT3 = f"{EXTRACTS}/studies__rpt__g1__held_gt_missing3.held_a.json"
SYSTEMS = ("armA", "armB", "ref35", "R0w")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gz(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def summary(pooled):
    r = pooled["run_recall_agnostic"]
    return dict(recall=r["all"]["recall"], recall_base=r["base"]["recall"], recall_novel=r["novel"]["recall"],
                counts={g: [r[g]["hit"], r[g]["n"]] for g in ("all", "base", "novel") if g in r},
                pair_recall=pooled["pairs"]["recall_agnostic"], pairs_n=pooled["pairs"]["n"],
                burden=pooled["unmatched_agnostic_per_video"], tracks_per_video=pooled["tracks_per_video"])


def rule(gains: dict) -> dict:
    """gains: {system: recall gain over R0w, points}. -> the binding rule's outcome (pure; tested)."""
    better = "armA" if gains["armA"] >= gains["armB"] else "armB"
    ref = gains["ref35"]
    ok = gains[better] >= ref / 2
    return dict(better_arm=better, better_gain_points=gains[better], reference_gain_points=ref,
                half_of_reference=ref / 2, continue_d7=bool(ok),
                verdict=("PASS: D7 continues" if ok else "FAIL: D7 stops") +
                        f" ({better} {gains[better]:+.2f} vs half of the reference {ref / 2:+.2f})")


def pass_stats(manifest_rows):
    by = {}
    for r in manifest_rows:
        for p in r["passes"]:
            k = (r["stage"], p["name"])
            by.setdefault(k, []).append(p)
    out = {}
    for (stage, name), ps in sorted(by.items()):
        spf = [p["seconds_per_frame"] for p in ps]
        out[f"{stage}/{name}"] = dict(videos=len(ps), seconds=round(sum(p["seconds"] for p in ps), 1),
                                      frames=sum(p["frames"] for p in ps),
                                      seconds_per_frame_mean=sum(p["seconds"] for p in ps) / max(1, sum(p["frames"] for p in ps)),
                                      seconds_per_frame_max=max(spf), masklets_mean=statistics.mean(p["masklets"] for p in ps),
                                      masklets_max=max(p["masklets"] for p in ps),
                                      peak_gpu_allocated_gib_max=max(p["peak_gpu_allocated_gib"] for p in ps))
    return out


def main():
    import adapter as A
    import metrics as M
    from utils import paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True, help="copied-back folder with armA/, armB/, ref35/ (if any), "
                                                 "manifest.jsonl, check0.json")
    a = ap.parse_args()
    os.chdir(REPO)
    vl_path = "studies/d7/check/video_list.json"
    if sha256(vl_path) != open(vl_path + ".sha256").read().split()[0]:
        raise SystemExit("!! STOP: video_list.json differs from its .sha256")
    videos = json.load(open(vl_path))["final"]
    pins = {r["extract"]: r["extract_sha256"] for r in json.load(open(f"{EXTRACTS}/manifest.json"))["extracts"]}
    for p in (OPT_B, HELD_GT, HELD_GT3):
        if sha256(p) != pins[p]:
            raise SystemExit(f"!! STOP: {p} differs from its manifest hash")
    rows = [json.loads(l) for l in open(os.path.join(a.box, "manifest.jsonl"))]
    for r in rows:
        if sha256(os.path.join(a.box, r["stage"], f"{r['video']}.json.gz")) != r["sha256"]:
            raise SystemExit(f"!! STOP: {r['stage']}/{r['video']} differs from the machine's manifest")
    check0 = json.load(open(os.path.join(a.box, "check0.json")))
    ref_from_g1 = bool(check0["identical"])
    opt_b = json.load(open(OPT_B))
    rels = json.load(open(HELD_GT))
    rels.update(json.load(open(HELD_GT3)))
    sp = json.load(open(paths.OBJ_SPLIT_INFO))
    novel = {c for c, s in sp["cls2split"].items() if s == "novel"}
    runs, rec, rec_eq = {}, {k: {} for k in SYSTEMS}, {k: {} for k in SYSTEMS}
    for v in videos:
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        runs[v] = M.runs_of(opt_b[v], anno)
        ref_file = os.path.join(SAM3_DIR, f"{v}.json.gz") if ref_from_g1 else os.path.join(a.box, "ref35", f"{v}.json.gz")
        repo = {"armA": CF.tracks_classfree(load_gz(os.path.join(a.box, "armA", f"{v}.json.gz")))["repo"],
                "armB": CF.tracks_classfree(load_gz(os.path.join(a.box, "armB", f"{v}.json.gz")))["repo"],
                "ref35": A.tracks_from_sam3(load_gz(ref_file))["repo"],
                "R0w": load_gz(os.path.join(EOV_DIR, f"{v}.json.gz"))["configs"]["R0w"]["tracks"]}
        k = len(repo["ref35"])
        for s in SYSTEMS:
            rec[s][v] = M.video_record(M.tracks_from_list(repo[s]), runs[v], rels[v])
            rec_eq[s][v] = M.video_record(M.tracks_from_list(CF.top_k(repo[s], k)), runs[v], rels[v])
    summ = {s: summary(M.pooled(rec[s], runs, rels, novel, videos)) for s in SYSTEMS}
    summ_eq = {s: summary(M.pooled(rec_eq[s], runs, rels, novel, videos)) for s in SYSTEMS}
    gains = {s: 100 * (summ[s]["recall"] - summ["R0w"]["recall"]) for s in SYSTEMS if s != "R0w"}
    detail = {s: dict(recall_points=gains[s],
                      recall_base_points=100 * (summ[s]["recall_base"] - summ["R0w"]["recall_base"]),
                      recall_novel_points=100 * (summ[s]["recall_novel"] - summ["R0w"]["recall_novel"]),
                      pair_recall_points=100 * (summ[s]["pair_recall"] - summ["R0w"]["pair_recall"]),
                      burden_ratio=summ[s]["burden"] / summ["R0w"]["burden"] if summ["R0w"]["burden"] else None,
                      equal_budget_recall_points=100 * (summ_eq[s]["recall"] - summ_eq["R0w"]["recall"]))
              for s in gains}
    res = dict(label="A34: the D7 check (PaD gate 1), class-agnostic, HELD-A", videos=videos,
               reference=("G1's files (check 0 bitwise)" if ref_from_g1 else "the rerun on this machine (check 0 not bitwise)"),
               check0=dict(identical=check0["identical"], difference_size=check0.get("difference_size")),
               rule=rule(gains), systems=summ, equal_budget=summ_eq, gains_over_R0w=detail,
               per_pass=pass_stats(rows))
    with open(os.path.join(D7, "check", "d7_results.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({k: res[k] for k in ("reference", "rule", "gains_over_R0w")}, indent=1))


if __name__ == "__main__":
    main()
