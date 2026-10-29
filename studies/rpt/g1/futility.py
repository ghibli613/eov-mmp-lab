#!/usr/bin/env python3
"""RPT G1, A29d's futility rule (its item 2): every binding bar on the 157 videos with SAM 3 files,
SAM 3 against EOV-MMP (R0w) on the same 157; then each binding bar's best possible value for SAM 3
over all 160, with SAM 3 at its best case on the 3 videos without SAM 3 files and EOV-MMP at its real
values on all 160. CPU only; the test-label guard is installed first (by importing analyse); HELD
training annotations only.

    python studies/rpt/g1/futility.py --sam3 studies/rpt/g1/sam3_out --eov studies/rpt/g1/eov_out \
        --done studies/rpt/g1/logs/box/sam3_done.json --out studies/rpt/g1/results_futility

Writes OUT/futility.json: the outcome first, then the bounds, the 157-video bars and the counts.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyse as AN  # noqa: E402  (installs the test-label guard)
import metrics as M  # noqa: E402
from eov_held import PINS, held_relations  # noqa: E402


def best_case(rec_s: dict, rec_e: dict, runs: dict, rels: dict, style_values: list, with_sam3: list,
              held: list, novel: set):
    """A29d: SAM 3's best case on the videos of `held` not in `with_sam3` (every GT run matched,
    every base relation recalled, no unmatched track, one pair at IoU 1.0 per GT run); EOV-MMP's real
    records on all of `held`. -> (M.bars of the bounds, what was added)."""
    missing = [v for v in held if v not in set(with_sam3)]
    p_s = M.pooled(rec_s, runs, rels, novel, with_sam3)
    p_e = M.pooled(rec_e, runs, rels, novel, held)
    n_runs = sum(len(runs[v]) for v in missing)
    n_rels = sum(len(rels[v]) for v in missing)
    r, p = p_s["run_recall_aware"]["all"], p_s["pairs"]
    best = dict(run_recall_aware=dict(all=dict(recall=(r["hit"] + n_runs) / (r["n"] + n_runs))),
                pairs=dict(recall=(p["buckets"].get("recalled", 0) + n_rels) / (p["n"] + n_rels)),
                unmatched_per_video=sum(len(rec_s[v]["unmatched"]) for v in with_sam3) / len(held))
    vals = list(style_values) + [1.0] * n_runs
    style = dict(median=statistics.median(vals) if vals else None, n_pairs=len(vals))
    added = dict(videos=missing, gt_runs_matched=n_runs, base_relations_recalled=n_rels, unmatched_tracks=0,
                 box_style_pairs_at_1=n_runs)
    return M.bars(best, p_e, style, None), added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--eov", required=True)
    ap.add_argument("--done", required=True, help="the 8x machine's sam3_done.json (excluded, moved_to_l40)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.sam3, a.eov, a.done, a.out = map(os.path.abspath, (a.sam3, a.eov, a.done, a.out))
    os.chdir(AN.REPO)
    from utils import paths
    for p, h in list(PINS.items()) + [(os.path.relpath(AN.GT_TRACKS, AN.REPO), AN.GT_TRACKS_SHA)]:
        if not p.startswith("output/") and AN.sha256(p) != h:
            raise SystemExit(f"!! STOP: {p} differs from its pin")
    held = sorted(json.load(open("pilot_analysis/stage1/train_split.json"))["held"])
    with_sam3 = [v for v in held if os.path.exists(os.path.join(a.sam3, f"{v}.json.gz"))]
    done = json.load(open(a.done))
    missing = sorted(set(held) - set(with_sam3))
    if missing != sorted(done["excluded"] + done["moved_to_l40"]) or len(with_sam3) != 157:
        raise SystemExit(f"!! STOP: the videos without SAM 3 files are {missing}, not the 8x machine's list")
    rels = held_relations()
    option_b = json.load(open(AN.GT_TRACKS))
    split = json.load(open("data/vidvrd/data/openvoc_obj_class_spilt_info.json"))["cls2split"]
    novel = {c for c, s in split.items() if s == "novel"}

    runs, rec_s, rec_e, files, style_values, no_shared = {}, {}, {}, {}, [], 0
    for i, v in enumerate(held):
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        anno.pop("relation_instances", None)                      # the relations come from held_gt (base only)
        runs[v] = M.runs_of(option_b[v], anno)
        ep = os.path.join(a.eov, f"{v}.json.gz")
        e = AN.load_json_gz(ep)
        files[v] = dict(eov=AN.sha256(ep))
        rec_e[v] = M.video_record(M.tracks_from_list(e["configs"]["R0w"]["tracks"]), runs[v], rels[v])
        if v in missing:
            if e["sam3_file_sha256"] is not None or e["sensitivity"] is not None:
                raise SystemExit(f"!! STOP: {v}: an EOV-side record with SAM 3 input for a video without SAM 3 file")
            files[v]["sam3"] = None
            print(f"[{i + 1}/{len(held)}] {v}: no SAM 3 file; EOV R0w only", flush=True)
            continue
        sp = os.path.join(a.sam3, f"{v}.json.gz")
        files[v]["sam3"] = AN.sha256(sp)
        if e["sam3_file_sha256"] != files[v]["sam3"]:
            raise SystemExit(f"!! STOP: {v}: the EOV-side run read another SAM 3 file")
        _, conf, _ = AN.sam3_tracks(AN.load_json_gz(sp))
        rec_s[v] = M.video_record(conf, runs[v], rels[v])
        st = M.box_style({v: rec_s[v]}, {v: conf}, {v: runs[v]})
        style_values += st["values"]
        no_shared += st["pairs_without_annotated_shared_frame"]
        print(f"[{i + 1}/{len(held)}] {v}: SAM 3 {len(conf)} confident, runs {len(runs[v])}, "
              f"relations {len(rels[v])}", flush=True)

    p_s = M.pooled(rec_s, runs, rels, novel, with_sam3)
    p_e157 = M.pooled(rec_e, runs, rels, novel, with_sam3)
    p_e160 = M.pooled(rec_e, runs, rels, novel, held)
    style157 = dict(median=statistics.median(style_values) if style_values else None, n_pairs=len(style_values),
                    pairs_without_annotated_shared_frame=no_shared)
    bars157 = M.bars(p_s, p_e157, style157, None)
    bounds, added = best_case(rec_s, rec_e, runs, rels, style_values, with_sam3, held, novel)
    failing = [k for k, b in bounds.items() if not b["pass_"]]
    counts = lambda p: dict(runs_matched=p["run_recall_aware"]["all"]["hit"], runs=p["run_recall_aware"]["all"]["n"],
                            relations_recalled=p["pairs"]["buckets"].get("recalled", 0), relations=p["pairs"]["n"],
                            unmatched_per_video=p["unmatched_per_video"])
    res = dict(
        label="G1 (A29d's futility rule): HELD tracker comparison; EOV-MMP was trained on HELD (in-sample); "
              "not a test result; the 157-video numbers are not used for any decision",
        outcome=dict(g1_fails_for_certain=bool(failing), failing_at_best_case=failing,
                     rule="A29d item 2: if any binding bar fails even at SAM 3's best case over 160, G1 fails"),
        bounds_over_160={k: dict(b, pass_=bool(b["pass_"])) for k, b in bounds.items()},
        best_case_added=added,
        eov_R0w_over_160=counts(p_e160),
        bars_on_157={k: dict(b, pass_=bool(b["pass_"])) for k, b in bars157.items()},
        counts_on_157=dict(sam3=counts(p_s), eov_R0w=counts(p_e157), box_style=style157),
        videos=dict(held=len(held), with_sam3=len(with_sam3), without_sam3=missing),
        files=files,
        pins={p: h for p, h in PINS.items()} | {os.path.relpath(AN.GT_TRACKS, AN.REPO): AN.GT_TRACKS_SHA})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "futility.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "box_style_values_157.json"), "w") as f:
        json.dump(style_values, f)
    print(json.dumps(dict(outcome=res["outcome"], bounds_over_160=res["bounds_over_160"],
                          bars_on_157=res["bars_on_157"]), indent=1))


if __name__ == "__main__":
    main()
