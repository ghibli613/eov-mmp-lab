#!/usr/bin/env python3
"""RPT G1 on the laptop (A29): every A29 quantity from the copied-back SAM 3 and EOV-MMP files.
CPU only; the test-label guard is installed first; HELD training annotations only.

    python studies/rpt/g1/analyse.py --sam3 studies/rpt/g1/sam3_out --eov studies/rpt/g1/eov_out \
        --out studies/rpt/g1/results

Writes OUT/g1_results.json (the bars first, then everything report-only) and OUT/viz/ (GT and
SAM 3 boxes on the middle annotated frame of each of the first 20 videos, A29 item 6).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
STUDIES = os.path.dirname(RPT)
REPO = os.path.dirname(STUDIES)
for p in (HERE, RPT, STUDIES, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import no_test_labels  # noqa: E402
no_test_labels.install()

import metrics as M  # noqa: E402
from eov_held import PINS, held_relations  # noqa: E402

GT_TRACKS = os.path.join(REPO, "pilot_analysis", "stage1c", "gt_tracks_train.json")
GT_TRACKS_SHA = "87b40fff514178f77a57d9e142bbbb9363eef34f6b4c22d8eaa88d4c7612636d"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_gz(path):
    import gzip
    with gzip.open(path, "rt") as f:
        return json.load(f)


def sam3_tracks(s3: dict):
    """Confident tracks (the adapter's repo list, tid = list index) and confident + candidate tracks
    (candidates appended, ordered by prompt id, then SAM 3 ID)."""
    import adapter as A
    out = A.tracks_from_sam3(s3)
    conf = M.tracks_from_list(out["repo"])
    cand = sorted((t for t in out["kept"] if t["tier"] == "candidate"), key=lambda t: (t["prompt_id"], t["obj_id"]))
    rows = list(out["repo"]) + [dict(tid=len(out["repo"]) + i, category=t["category"], score=float(t["score"]),
                                     begin_fid=t["begin_fid"], end_fid=t["end_fid"], trajectory=t["trajectory"])
                                for i, t in enumerate(cand)]
    return out, conf, M.tracks_from_list(rows)


def gap_exposure(video, out, frame_dir):
    """A29: clips of SAM 3's confident pairs whose middle frame is a filled frame of the subject or
    object. Middle frames as gen_feats_test requests them (relation.relation_head), with zero
    features and no modelC: the clip grid does not depend on the features."""
    import torch
    import adapter as A
    from relation import FrameFeatures, relation_head
    if not out["repo"]:
        return 0, 0
    enc = lambda path: (torch.zeros(1, 768), torch.zeros(1, 576, 768))
    stub = lambda m, it: (dict(logits=[], int_logit=[], windows=[], slot_inputs=None), 0.0)
    pairs = relation_head(video, out["repo"], "train", FrameFeatures(video, frame_dir, enc), None, run_pair=stub)
    flags = A.gap_exposure(pairs, [t["status"] for t in out["confident"]])
    return sum(map(sum, flags)), sum(map(len, flags))


def draw(video, anno, conf, out_dir, frame_dir):
    """The middle annotated frame: GT boxes (green) and SAM 3's confident boxes (one colour per class)."""
    from PIL import Image, ImageDraw
    cat = {so["tid"]: so["category"] for so in anno["subject/objects"]}
    annotated = [f for f, fr in enumerate(anno["trajectories"]) if fr]
    f = annotated[len(annotated) // 2]
    files = sorted(os.listdir(os.path.join(frame_dir, video)))
    img = Image.open(os.path.join(frame_dir, video, files[f])).convert("RGB")
    dr = ImageDraw.Draw(img)
    for b in anno["trajectories"][f]:
        bb = b["bbox"]
        dr.rectangle([bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]], outline=(60, 220, 60), width=3)
        dr.text((bb["xmin"] + 3, bb["ymax"] - 14), f"GT {cat[b['tid']]}", fill=(60, 220, 60))
    palette = [(230, 25, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]
    colours = {}
    for t in conf.values():
        if t["begin_fid"] <= f < t["end_fid"]:
            c = colours.setdefault(t["category"], palette[len(colours) % len(palette)])
            box = t["boxes"][f - t["begin_fid"]]
            dr.rectangle(box, outline=c, width=2)
            dr.text((box[0] + 3, box[1] + 2), f"SAM3 {t['category']}", fill=c)
    dr.text((6, 6), f"{video}  frame {f} (middle annotated)  green: GT", fill=(255, 255, 255))
    p = os.path.join(out_dir, f"{video}_f{f:04d}.jpg")
    img.save(p, quality=90)
    return os.path.basename(p)


def sensitivity_hits(eov_recs, rels_all, base_ids, pre2id):
    tot = {s: Counter() for s in ("gt", "sam3", "eov")}
    rows = 0
    for v, rec in eov_recs.items():
        for r in rec["sensitivity"]:
            rows += 1
            gid = pre2id[rels_all[v][r["k"]]["triplet"][1]]
            for src in tot:
                x = r["sources"][src]
                if x is None:
                    tot[src]["relations_without_clips"] += 1
                    continue
                for lg, il in zip(x["logits"], x["int_logit"]):
                    tot[src]["clips"] += 1
                    tot[src]["hits"] += M.top5_base_hit(lg, il, gid, base_ids)
    return {s: dict(clips=c["clips"], hits=c["hits"], hit_rate=c["hits"] / c["clips"] if c["clips"] else None,
                    relations_without_clips=c["relations_without_clips"]) for s, c in tot.items()} | dict(relations=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--eov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", default=None, help="the machine's copied-back log folder (checks, projection)")
    ap.add_argument("--videos", nargs="*", default=None, help="a subset: the laptop's dry run only, never G1's result")
    a = ap.parse_args()
    os.chdir(REPO)
    from utils import paths
    for p, h in list(PINS.items()) + [(os.path.relpath(GT_TRACKS, REPO), GT_TRACKS_SHA)]:
        if not p.startswith("output/") and sha256(p) != h:
            raise SystemExit(f"!! STOP: {p} differs from its pin")
    held = sorted(json.load(open("pilot_analysis/stage1/train_split.json"))["held"])
    if a.videos:
        assert set(a.videos) <= set(held)
        held = sorted(a.videos)
    rels_all = held_relations()
    option_b = json.load(open(GT_TRACKS))
    split = json.load(open("data/vidvrd/data/openvoc_obj_class_spilt_info.json"))["cls2split"]
    novel = {c for c, s in split.items() if s == "novel"}
    psplit = json.load(open(paths.PRED_SPLIT_INFO))
    base_ids = sorted(int(i) for i, c in psplit["id2cls"].items() if c != "__background__" and psplit["cls2split"][c] == "base")
    pre2id = {c: int(i) for i, c in psplit["id2cls"].items()}
    os.makedirs(os.path.join(a.out, "viz"), exist_ok=True)

    runs, tr, rec = {}, defaultdict(dict), defaultdict(dict)
    style_tracks, raw, gaps, files, viz, eov_recs = {}, {"base": [], "novel": []}, Counter(), {}, [], {}
    for i, v in enumerate(held):
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        anno.pop("relation_instances", None)                      # the relations come from held_gt (base only)
        runs[v] = M.runs_of(option_b[v], anno)
        sp, ep = os.path.join(a.sam3, f"{v}.json.gz"), os.path.join(a.eov, f"{v}.json.gz")
        files[v] = dict(sam3=sha256(sp), eov=sha256(ep))
        s3 = load_json_gz(sp)
        out, conf, conf_cand = sam3_tracks(s3)
        e = load_json_gz(ep)
        if e["sam3_file_sha256"] != files[v]["sam3"]:
            raise SystemExit(f"!! STOP: {v}: the EOV-side run read another SAM 3 file")
        eov_recs[v] = e
        tr["sam3"][v], tr["sam3_cand"][v] = conf, conf_cand
        tr["eov_R0w"][v] = M.tracks_from_list(e["configs"]["R0w"]["tracks"])
        tr["eov_R0"][v] = M.tracks_from_list(e["configs"]["R0"]["tracks"])
        for k in tr:
            rec[k][v] = M.video_record(tr[k][v], runs[v], rels_all[v])
        sel = M.sensitivity_relations(rels_all[v], rec["sam3"][v], rec["eov_R0w"][v])
        if [s["k"] for s in sel] != [r["k"] for r in e["sensitivity"]]:
            raise SystemExit(f"!! STOP: {v}: the machine's box-sensitivity relations differ from the laptop's")
        by_class = {p["class"]: p["raw_low"] for p in s3["prompts"]}
        for r in runs[v]:
            if r["rid"] not in rec["sam3"][v]["matched"]:
                raw["novel" if r["category"] in novel else "base"].append(M.raw_fraction(r, by_class[r["category"]]))
        g, n = gap_exposure(v, out, paths.FRAME_DIR)
        gaps["filled_middle"] += g
        gaps["clips"] += n
        if i < 20:
            viz.append(draw(v, anno, conf, os.path.join(a.out, "viz"), paths.FRAME_DIR))
        print(f"[{i + 1}/{len(held)}] {v}: SAM 3 {len(conf)} confident, EOV R0w {len(tr['eov_R0w'][v])}, "
              f"runs {len(runs[v])}, relations {len(rels_all[v])}", flush=True)

    pooled = {k: M.pooled(rec[k], runs, rels_all, novel) for k in rec}
    style = M.box_style(rec["sam3"], tr["sam3"], runs)
    sens = sensitivity_hits(eov_recs, rels_all, base_ids, pre2id)
    bars = M.bars(pooled["sam3"], pooled["eov_R0w"], style, sens)
    bars_r0 = M.bars(pooled["sam3"], pooled["eov_R0"], style, None)

    def stat_recall(vs):
        a_ = M.pooled(rec["sam3"], runs, rels_all, novel, vs)["run_recall_aware"]["all"]["recall"]
        b_ = M.pooled(rec["eov_R0w"], runs, rels_all, novel, vs)["run_recall_aware"]["all"]["recall"]
        return 100 * (a_ - b_)

    def stat_pairs(vs):
        a_, b_ = (M.pooled(rec[k], runs, rels_all, novel, vs)["pairs"]["recall"] for k in ("sam3", "eov_R0w"))
        return None if a_ is None else 100 * (a_ - b_)

    def stat_burden(vs):
        a_, b_ = (M.pooled(rec[k], runs, rels_all, novel, vs)["unmatched_per_video"] for k in ("sam3", "eov_R0w"))
        return a_ / b_ if b_ else None

    ci = dict(overall_recall_diff_points=M.bootstrap(held, stat_recall), pair_recall_diff_points=M.bootstrap(held, stat_pairs),
              false_track_burden_ratio=M.bootstrap(held, stat_burden))
    box = {}
    if a.box:
        for name in ("checks.json", "g1_config.json", "projection.json", "sam3_done.json"):
            p = os.path.join(a.box, name)
            if os.path.exists(p):
                box[name] = json.load(open(p))
    res = dict(
        label="G1 (A29): HELD tracker comparison; EOV-MMP was trained on HELD (in-sample); not a test result"
              + ("" if len(held) == 160 else f"; DRY RUN on {len(held)} videos, not G1's result"),
        verdict={k: dict(v, pass_=bool(v["pass_"])) for k, v in bars.items()},
        verdict_against_R0=bars_r0,
        bootstrap_ci_descriptive=ci,
        report_only=dict(
            novel_object_recall={k: {g: x for g, x in pooled[k]["run_recall_aware"].items()
                                     if g == "novel" or g.startswith("class:") and g[6:] in novel} for k in pooled},
            base_object_recall={k: pooled[k]["run_recall_aware"]["base"] for k in pooled},
            tid_recall={k: pooled[k]["tid_recall_aware"] for k in pooled},
            recall_confident_plus_candidates=pooled["sam3_cand"]["run_recall_aware"]["all"],
            class_agnostic={k: dict(run_recall=pooled[k]["run_recall_agnostic"]["all"],
                                    tid_recall=pooled[k]["tid_recall_agnostic"],
                                    pair_recall_not_U1=pooled[k]["pairs"]["recall_agnostic"],
                                    unmatched_per_video=pooled[k]["unmatched_agnostic_per_video"]) for k in pooled},
            pair_recall_scoreable={k: pooled[k]["pairs_scoreable"] for k in pooled},
            pair_buckets={k: pooled[k]["pairs"] for k in pooled},
            gap_exposure=dict(gaps, share=gaps["filled_middle"] / gaps["clips"] if gaps["clips"] else None),
            track_counts_by_N={k: pooled[k]["track_counts_by_N"] for k in pooled},
            tracks_per_video={k: pooled[k]["tracks_per_video"] for k in pooled},
            raw_detections_for_missed_runs={g: M.fraction_summary(x) for g, x in raw.items()},
            box_style_detail={k: v for k, v in style.items() if k != "values"},
            box_sensitivity=sens),
        counts=dict(videos=len(held), runs=sum(map(len, runs.values())),
                    relations=sum(len(rels_all[v]) for v in held)),
        machine=box, viz=viz, files=files,
        pins={p: h for p, h in PINS.items()} | {os.path.relpath(GT_TRACKS, REPO): GT_TRACKS_SHA})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "g1_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "box_style_values.json"), "w") as f:
        json.dump(style["values"], f)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
