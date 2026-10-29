#!/usr/bin/env python3
"""RPT G1, descriptive and after the verdict (the plan author's request after G1's FAIL): what the
unmatched confident tracks are, on the 157 videos with SAM 3 files, for SAM 3 and EOV-MMP (R0w). It
decides nothing. CPU only; the test-label guard is installed first (by importing analyse); HELD
training annotations only.

    python studies/rpt/g1/describe_unmatched.py --sam3 studies/rpt/g1/sam3_out --eov studies/rpt/g1/eov_out \
        --done studies/rpt/g1/logs/box/sam3_done.json --out studies/rpt/g1/results_descriptive \
        --viz studies/rpt/g1/viz_unmatched

1. class-agnostic unmatched confident tracks per video (A29 item 5);
2. unmatched confident tracks by class (beside all confident tracks by class);
3. each unmatched track's group, by its best cut-vIoU (A29 item 2) to the video's GT runs: misnamed
   (>= 0.5 to a run of another class), poorly placed (best in [0.1, 0.5)), no GT object nearby
   (best < 0.1). A track whose only cut-vIoU >= 0.5 is to runs of its own class (the run went to
   another track) fits none of the three; it is counted apart, as is a track >= 0.5 to both;
4. SAM 3's unmatched and matched tracks by track score;
5. 20 of SAM 3's "no GT object nearby" tracks, drawn with numpy.random.default_rng(0) from the list
   sorted by (video, track id), each on its middle frame with the GT runs' boxes there.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyse as AN  # noqa: E402  (installs the test-label guard)
import metrics as M  # noqa: E402
from eov_held import PINS  # noqa: E402

GROUPS = ("misnamed", "poorly_placed", "no_gt_nearby", "own_class_only_run_taken")
SCORE_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, None))
NEAR = 0.1


def group(track: dict, runs: list):
    """-> (group, best cut-vIoU to a run of another class, best to a run of its own class)."""
    other = max((M.cut_viou(track, r) for r in runs if r["category"] != track["category"]), default=0.0)
    own = max((M.cut_viou(track, r) for r in runs if r["category"] == track["category"]), default=0.0)
    if other >= M.THR:
        g = "misnamed"
    elif own >= M.THR:
        g = "own_class_only_run_taken"
    elif max(other, own) >= NEAR:
        g = "poorly_placed"
    else:
        g = "no_gt_nearby"
    return g, other, own


def score_bin(s):
    if s is None:
        return "no score"
    for lo, hi in SCORE_BINS:
        if s >= lo and (hi is None or s < hi):
            return f"{lo}-{hi}" if hi else f">={lo}"
    return "below 0.5"


def draw(item, frame_dir, out_dir, k):
    from PIL import Image, ImageDraw
    v, f = item["video"], item["frame"]
    files = sorted(os.listdir(os.path.join(frame_dir, v)))
    img = Image.open(os.path.join(frame_dir, v, files[f])).convert("RGB")
    dr = ImageDraw.Draw(img)
    for cat, b in item["gt"]:
        dr.rectangle(b, outline=(60, 220, 60), width=3)
        dr.text((b[0] + 3, b[3] - 14), f"GT {cat}", fill=(60, 220, 60))
    b = item["box"]
    dr.rectangle(b, outline=(230, 25, 75), width=3)
    dr.text((b[0] + 3, b[1] + 2), f"SAM3 {item['category']} {item['score']:.2f}", fill=(230, 25, 75))
    dr.text((6, 6), f"{v} track {item['tid']} frame {f} (track's middle; frames {item['begin_fid']}-"
                    f"{item['end_fid'] - 1})  best cut-vIoU {item['best']:.3f}  green: GT runs", fill=(255, 255, 255))
    name = f"{k:02d}_{v}_t{item['tid']}_f{f:04d}.jpg"
    img.save(os.path.join(out_dir, name), quality=90)
    return name


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--eov", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz", required=True)
    a = ap.parse_args()
    a.sam3, a.eov, a.done, a.out, a.viz = map(os.path.abspath, (a.sam3, a.eov, a.done, a.out, a.viz))
    os.chdir(AN.REPO)
    from utils import paths
    for p, h in list(PINS.items()) + [(os.path.relpath(AN.GT_TRACKS, AN.REPO), AN.GT_TRACKS_SHA)]:
        if not p.startswith("output/") and AN.sha256(p) != h:
            raise SystemExit(f"!! STOP: {p} differs from its pin")
    held = sorted(json.load(open("pilot_analysis/stage1/train_split.json"))["held"])
    with_sam3 = [v for v in held if os.path.exists(os.path.join(a.sam3, f"{v}.json.gz"))]
    done = json.load(open(a.done))
    if sorted(set(held) - set(with_sam3)) != sorted(done["excluded"] + done["moved_to_l40"]) or len(with_sam3) != 157:
        raise SystemExit("!! STOP: the videos with SAM 3 files are not the 8x machine's 157")
    option_b = json.load(open(AN.GT_TRACKS))

    agn, by_class, all_by_class = Counter(), {}, {}
    groups, both, scores, none_sam3, files = {}, Counter(), {"unmatched": Counter(), "matched": Counter()}, [], {}
    for k in ("sam3", "eov_R0w"):
        by_class[k], all_by_class[k], groups[k] = Counter(), Counter(), Counter()
    for i, v in enumerate(with_sam3):
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        anno.pop("relation_instances", None)
        runs = M.runs_of(option_b[v], anno)
        sp, ep = os.path.join(a.sam3, f"{v}.json.gz"), os.path.join(a.eov, f"{v}.json.gz")
        files[v] = dict(sam3=AN.sha256(sp), eov=AN.sha256(ep))
        e = AN.load_json_gz(ep)
        if e["sam3_file_sha256"] != files[v]["sam3"]:
            raise SystemExit(f"!! STOP: {v}: the EOV-side run read another SAM 3 file")
        _, conf, _ = AN.sam3_tracks(AN.load_json_gz(sp))
        for k, tracks in (("sam3", conf), ("eov_R0w", M.tracks_from_list(e["configs"]["R0w"]["tracks"]))):
            m_aw, m_ag = M.match(tracks, runs, True), M.match(tracks, runs, False)
            un = M.unmatched_tracks(tracks, m_aw)
            agn[k] += len(M.unmatched_tracks(tracks, m_ag))
            agn[k + "_aware"] += len(un)
            all_by_class[k].update(t["category"] for t in tracks.values())
            for tid in un:
                t = tracks[tid]
                by_class[k][t["category"]] += 1
                g, other, own = group(t, runs)
                groups[k][g] += 1
                both[k] += other >= M.THR and own >= M.THR
                if k == "sam3" and g == "no_gt_nearby":
                    f = t["begin_fid"] + (t["end_fid"] - t["begin_fid"]) // 2
                    none_sam3.append(dict(video=v, tid=tid, category=t["category"], score=t["score"],
                                          begin_fid=t["begin_fid"], end_fid=t["end_fid"], frame=f,
                                          box=t["boxes"][f - t["begin_fid"]], best=max(other, own),
                                          gt=[(r["category"], r["boxes"][f - r["begin_fid"]]) for r in runs
                                              if r["begin_fid"] <= f < r["end_fid"]]))
            if k == "sam3":
                matched = {tid for tid, _ in m_aw.values()}
                for tid, t in tracks.items():
                    scores["matched" if tid in matched else "unmatched"][score_bin(t["score"])] += 1
        print(f"[{i + 1}/{len(with_sam3)}] {v}", flush=True)

    n = len(with_sam3)
    rng = np.random.default_rng(0)
    none_sam3.sort(key=lambda x: (x["video"], x["tid"]))
    pick = sorted(rng.choice(len(none_sam3), size=min(20, len(none_sam3)), replace=False).tolist())
    os.makedirs(a.viz, exist_ok=True)
    sample = []
    for j, idx in enumerate(pick):
        it = none_sam3[idx]
        sample.append(dict({x: it[x] for x in ("video", "tid", "category", "score", "frame", "best")},
                           gt_runs_on_frame=len(it["gt"]), image=draw(it, paths.FRAME_DIR, a.viz, j + 1)))
    order = [score_bin(lo) for lo, _ in SCORE_BINS]
    res = dict(
        label="G1, descriptive, computed after the verdict (FAIL, A29d); decides nothing; 157 videos with SAM 3 "
              "files; HELD in-sample; not a test result",
        videos=n,
        unmatched_per_video={k: dict(class_aware=agn[k + "_aware"] / n, class_agnostic=agn[k] / n,
                                     class_aware_total=agn[k + "_aware"], class_agnostic_total=agn[k])
                             for k in ("sam3", "eov_R0w")},
        unmatched_by_class={k: {c: dict(unmatched=by_class[k][c], confident_tracks=all_by_class[k][c])
                                for c in sorted(all_by_class[k], key=lambda c: (-by_class[k][c], c))}
                            for k in by_class},
        groups={k: {g: groups[k][g] for g in GROUPS} | dict(total=sum(groups[k].values()),
                                                            at_least_0_5_to_both_classes=both[k])
                for k in groups},
        group_thresholds=dict(match=M.THR, near=NEAR),
        sam3_tracks_by_score={k: {b: scores[k][b] for b in order + sorted(set(scores[k]) - set(order))}
                              for k in scores},
        viz_sample=dict(rule="numpy.random.default_rng(0).choice(len, 20, replace=False) over SAM 3's "
                             "no-GT-nearby tracks sorted by (video, track id)",
                        population=len(none_sam3), drawn=sample),
        files=files)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "descriptive.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("unmatched_per_video", "groups", "sam3_tracks_by_score")}, indent=1))


if __name__ == "__main__":
    main()
