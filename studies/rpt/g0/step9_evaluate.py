#!/usr/bin/env python3
"""RPT G0 step 9, on the laptop (A28): the repo's post-processing and evaluator on the smoke videos,
the report's tables, the G1 cost extrapolation and the 18 images. CPU only.

The evaluator's numbers are a smoke test, in-sample, not a result: EOV-MMP was trained on HELD.
GT: the HELD GT of the 3 videos from pilot_analysis/stage2/data/held_gt.json (training annotations,
base relations only), so the evaluator runs with target_split_pred="base".

    python studies/rpt/g0/step9_evaluate.py --box studies/rpt/g0/box_out --out studies/rpt/g0
        [--sam3 studies/rpt/g0/sam3_out]
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RPT, "..", ".."))
for p in (RPT, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

from step9_relations import SMOKE  # noqa: E402

HELD_GT = os.path.join(REPO, "pilot_analysis", "stage2", "data", "held_gt.json")
HELD_GT_SHA256 = "78d55c69f9aa2473f4584ae998592dcb67d95ce02a67ab0cd9a8bd8446184dee"
HELD_FRAMES, HELD_VIDEOS = 47038, 160                 # studies/rpt/g0/logs/check3_middle_frames.json
G1_BAR_GPU_HOURS_PER_20_VIDEOS = 1.0                  # proposal §6
LABEL = "smoke test, in-sample, not a result"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def evaluate(preds, out_dir):
    from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
    if sha256(HELD_GT) != HELD_GT_SHA256:
        raise SystemExit("!! STOP: held_gt.json is not the pinned file")
    gt = json.load(open(HELD_GT))
    sub = {v: gt[v] for v in preds}
    gt_path = os.path.join(out_dir, "held_gt_smoke3.json")
    json.dump(sub, open(gt_path, "w"))
    m, rec = eval_relation_detection_openvoc(target_split_pred="base", prediction_results=preds, gt_json=gt_path)
    return dict(label=LABEL, split="base", gt_file=os.path.relpath(gt_path, REPO), gt_sha256=sha256(gt_path),
                gt_relations={v: len(r) for v, r in sub.items()},
                mAP=100 * float(m), R50=100 * float(rec[50]), R100=100 * float(rec[100]))


def video_table(rec, s3):
    by_tier = defaultdict(Counter)
    for t in rec["tracks"]:
        by_tier[t["tier"]][t["category"]] += 1
    prompts = s3["prompts"]
    return dict(
        video=rec["video"], frames=rec["n_frames"],
        sam3_seconds_total=s3["seconds_total"], sam3_session_start_seconds=s3["session_start_seconds"],
        sam3_seconds_per_prompt={p["class"]: p["seconds"] for p in prompts},
        sam3_seconds_per_frame_step=s3["seconds_total"] / (rec["n_frames"] * len(prompts)),
        sam3_peak_gpu_allocated_gib=s3["peak_gpu_allocated_gib"], sam3_peak_gpu_reserved_gib=s3["peak_gpu_reserved_gib"],
        relation_side_seconds=rec["seconds"], relation_side_peak_gpu_allocated_gib=rec["peak_gpu_allocated_gib"],
        masklets=len(rec["tracks"]),
        tracks_by_tier={k: dict(sorted(v.items())) for k, v in sorted(by_tier.items())},
        confident=sum(by_tier["confident"].values()), candidate=sum(by_tier["candidate"].values()),
        dropped=sum(by_tier["dropped"].values()), no_detected_frame=sum(by_tier["no detected frame"].values()),
        removed_by_sam3_hotstart={p["class"]: len(p["removed_by_sam3"]) for p in prompts if p["removed_by_sam3"]},
        raw_low_detections=sum(len(p["raw_low"]) for p in prompts),
        raw_low_detections_by_class={p["class"]: len(p["raw_low"]) for p in prompts if p["raw_low"]},
        raw_low_frames=len({d[0] for p in prompts for d in p["raw_low"]}),
        duplicates=rec["duplicates"],
        ordered_pairs=len(rec["pairs"]), clips=sum(len(p["logits"]) for p in rec["pairs"]),
        pairs_over_40_clips=sum(1 for p in rec["pairs"] if len(p["logits"]) > 40),
        middle_frames_read=len(rec["frames_encoded"]),
        clips_with_filled_middle_frame=sum(sum(p["gap_flags"]) for p in rec["pairs"]),
        frames_with_several_detections=sum(t["frames_with_several_detections"] for t in rec["tracks"]),
        filled_frames=sum(t["filled"] for t in rec["confident"]),
        confident_frames=sum(t["frames"] for t in rec["confident"]),
        scores={f"{t['category']}#{t['obj_id']}": round(t["score"], 4) for t in rec["confident"]})


def g1_cost(tables):
    steps = sum(t["frames"] * len(t["sam3_seconds_per_prompt"]) for t in tables)
    secs = sum(t["sam3_seconds_total"] for t in tables)
    per_step = secs / steps
    rel = sum(t["relation_side_seconds"] for t in tables) / len(tables)
    sam3_h = per_step * HELD_FRAMES * 35 / 3600
    rel_h = rel * HELD_VIDEOS / 3600
    bar_h = G1_BAR_GPU_HOURS_PER_20_VIDEOS * HELD_VIDEOS / 20
    return dict(sam3_seconds_per_frame_step=per_step, held_frames=HELD_FRAMES, prompts=35,
                sam3_gpu_hours_160=sam3_h, relation_side_gpu_hours_160=rel_h, total_gpu_hours_160=sam3_h + rel_h,
                bar_gpu_hours_160=bar_h, ratio_to_bar=(sam3_h + rel_h) / bar_h,
                per_frame_step_needed_for_bar_seconds=(bar_h - rel_h) * 3600 / (HELD_FRAMES * 35))


def draw(rec, out_dir, frame_dir, anno_dir):
    """6 evenly spaced frames: GT boxes (green, the training annotation's boxes and categories only)
    and SAM 3's confident tracks (one colour per class; a filled frame's box drawn thin)."""
    import numpy as np
    from PIL import Image, ImageDraw
    v = rec["video"]
    anno = json.load(open(os.path.join(anno_dir, f"{v}.json")))
    anno.pop("relation_instances", None)
    cat = {so["tid"]: so["category"] for so in anno["subject/objects"]}
    files = sorted(os.listdir(os.path.join(frame_dir, v)))
    palette = [(230, 25, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
               (255, 225, 25), (0, 128, 128), (170, 110, 40), (128, 0, 0)]
    colours = {}
    paths_out = []
    for f in sorted({int(x) for x in np.linspace(0, len(files) - 1, 6).round()}):
        img = Image.open(os.path.join(frame_dir, v, files[f])).convert("RGB")
        dr = ImageDraw.Draw(img)
        for b in anno["trajectories"][f] if f < len(anno["trajectories"]) else []:
            bb = b["bbox"]
            dr.rectangle([bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]], outline=(60, 220, 60), width=3)
            dr.text((bb["xmin"] + 3, bb["ymax"] - 14), f"GT {cat[b['tid']]}", fill=(60, 220, 60))
        for t in rec["confident"]:
            box = t["trajectory"].get(f)
            if box is None:
                continue
            c = colours.setdefault(t["category"], palette[len(colours) % len(palette)])
            filled = t["status"][f] == "filled"
            dr.rectangle(box, outline=c, width=1 if filled else 3)
            dr.text((box[0] + 3, box[1] + 2), f"SAM3 {t['category']} {t['score']:.2f}{' filled' if filled else ''}", fill=c)
        dr.text((6, 6), f"{v}  frame {f}  green: GT  colour: SAM 3 confident tracks", fill=(255, 255, 255))
        p = os.path.join(out_dir, f"{v}_f{f:04d}.jpg")
        img.save(p, quality=90)
        paths_out.append(p)
    return paths_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True, help="the machine's copied-back step 9 outputs (box_out)")
    ap.add_argument("--sam3", default=os.path.join(HERE, "sam3_out"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import no_test_labels
    no_test_labels.install()
    import torch
    from relation import predictions
    from utils import paths
    recs = {v: torch.load(os.path.join(a.box, "step9", f"{v}.pt"), weights_only=False, map_location="cpu") for v in SMOKE}
    s3 = {}
    for v in SMOKE:
        with gzip.open(os.path.join(a.sam3, f"{v}.json.gz"), "rt") as f:
            s3[v] = json.load(f)
    preds = {v: predictions(r["pairs"]) for v, r in recs.items()}
    tables = [video_table(recs[v], s3[v]) for v in SMOKE]
    res = dict(videos=tables, g1_cost=g1_cost(tables), evaluator=evaluate(preds, a.out),
               predictions_per_video={v: len(p) for v, p in preds.items()})
    viz = os.path.join(a.out, "viz")
    os.makedirs(viz, exist_ok=True)
    res["images"] = [os.path.relpath(p, REPO) for v in SMOKE
                     for p in draw(recs[v], viz, paths.FRAME_DIR, paths.ANNO_TRAIN_DIR)]
    path = os.path.join(a.out, "logs", "step9_smoke.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(dict(evaluator=res["evaluator"], g1_cost=res["g1_cost"],
                          per_video=[{k: t[k] for k in ("video", "frames", "sam3_seconds_total", "confident", "candidate",
                                                         "dropped", "raw_low_detections", "ordered_pairs", "clips",
                                                         "clips_with_filled_middle_frame")}
                                     for t in tables]), indent=1))


if __name__ == "__main__":
    main()
