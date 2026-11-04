#!/usr/bin/env python3
"""A33: the naming check. modelB alone names G1's 35-name SAM 3 tracks on HELD-A; class-aware recall with
modelB's names against the prompt names and EOV-MMP R0w. CLIP and modelB on the laptop's GPU; no SAM 3.

    python studies/d7/naming_check.py run        # per video, resumable: studies/d7/naming_check/videos/
    python studies/d7/naming_check.py analyse    # -> studies/d7/naming_check/results.json

Naming (P4), per box, exactly as End2End_Model's test loop (models/end2end_model.py:222-264, without the
fusion): the frame's TagCLIP patch_proj (relation.py's encoder call; the CLIP global pass is skipped, as
naming does not read it); the box clipped to the image, scaled to the 24 x 24 grid, create_mask, masked
mean, L2 norm; modelB (Classifier(args), weights modelB.* of the released end-to-end checkpoint, as R0w
loaded them), softmax; new_order. A track's 35-vector is the mean over its detected frames; its name is
the argmax. The HELD-B guard (and the test-label guard) is installed first.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "studies", "rpt"), os.path.join(REPO, "studies", "rpt", "g1"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
import held_b_guard  # noqa: E402
held_b_guard.install()

import numpy as np  # noqa: E402

OUT = os.path.join(REPO, "studies", "d7", "naming_check")
CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"
CKPT_SHA = "d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70"
NEW_ORDER = [0, 33, 20, 32, 1, 2, 3, 4, 30, 5, 6, 7, 28, 21, 27, 8, 25, 9, 34, 10, 31, 24, 11, 29, 12, 23, 13, 22,
             14, 15, 16, 17, 26, 18, 19]                                   # end2end_model.py:257-258
EXTRACTS = "studies/d7/extracts"
OPT_B = f"{EXTRACTS}/pilot_analysis__stage1c__gt_tracks_train.held_a.json"
HELD_GT = f"{EXTRACTS}/pilot_analysis__stage2__data__held_gt.held_a.json"
HELD_GT3 = f"{EXTRACTS}/studies__rpt__g1__held_gt_missing3.held_a.json"
SAM3_DIR, EOV_DIR = "studies/rpt/g1/sam3_out", "studies/rpt/g1/eov_out"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_extracts():
    pins = {r["extract"]: r["extract_sha256"] for r in json.load(open(f"{EXTRACTS}/manifest.json"))["extracts"]}
    for p in (OPT_B, HELD_GT, HELD_GT3):
        if sha256(p) != pins[p]:
            raise SystemExit(f"!! STOP: {p} differs from its manifest hash")


def held_a_videos():
    split = json.load(open("studies/d7/held_split.json"))
    return sorted(split["held_a"])


# ---------------------------------------------------------------- the pooling, as End2End_Model
def pooled_feature(pp, box, w, h):
    """end2end_model.py:222-253 for one box. pp: (1, 576, 768) tensor; box: [x1, y1, x2, y2] pixels;
    w, h: the frame size. Same number types as the original: a float32 box, int64 sizes."""
    import torch
    from models.end2end_model import create_mask
    w_t, h_t = torch.tensor(int(w)), torch.tensor(int(h))
    b = torch.tensor([float(x) for x in box], dtype=torch.float32)
    new_box = [b[0].item(), b[1].item(), b[2].item(), b[3].item()]
    if b[0] < 0:
        new_box[0] = 0
    if b[1] < 0:
        new_box[1] = 0
    if b[2] > w_t:
        new_box[2] = w_t
    if b[3] > h_t:
        new_box[3] = h_t
    b = torch.tensor(new_box)
    x1, x2 = b[0] / w_t * 24, b[2] / w_t * 24
    y1, y2 = b[1] / h_t * 24, b[3] / h_t * 24
    mask = np.array(create_mask(24, 24, [x1, y1, x2, y2])).flatten()
    mask_tensor = torch.from_numpy(mask).float().to(pp.device).unsqueeze(0)
    m_ = mask_tensor.unsqueeze(-1).expand_as(pp)
    weights_sum = m_.sum(dim=1, keepdim=True).clamp(min=1e-9)
    xx4 = (pp * m_).sum(dim=1) / weights_sum                         # (1, 1, 768), as the original
    xx4 /= xx4.norm(dim=-1, keepdim=True)
    xx4 = xx4.squeeze(0)                                             # end2end_model.py:253
    return xx4                                                       # (1, 768)


def reorder(probs):
    """modelB's class order -> object2id's (end2end_model.py:257-259)."""
    return [probs[i] for i in NEW_ORDER]


# ---------------------------------------------------------------- run: per video
def build_models():
    import torch
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", CKPT, "--frame_stride", "1"]
    try:
        from utils.parser_func import parse_args
        args = parse_args()
    finally:
        sys.argv = argv
    from models.object_classifier import Classifier
    from data_loading.dataset import _shared_encoder, _transform_resize
    m = Classifier(args).cuda().eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = {k[len("modelB."):]: v for k, v in ck["state_dict"].items() if k.startswith("modelB.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    del ck
    if missing or unexpected or len(sd) != 155:
        raise SystemExit(f"!! STOP: modelB weights: {len(sd)} keys, missing {missing}, unexpected {unexpected}")
    tag = _shared_encoder("tagclip", args.clip_backbone)
    return m, tag, _transform_resize(336, 336)


def boxes_of_video(v, opt_b, paths, A, M):
    """-> (tracks, runs, need): G1's confident tracks with their detected frames and boxes; the GT runs;
    {frame: [(kind, index, box)]}."""
    s3 = json.load(gzip.open(os.path.join(SAM3_DIR, f"{v}.json.gz"), "rt"))
    out = A.tracks_from_sam3(s3)
    conf = out["confident"]
    assert [t["category"] for t in conf] == [r["category"] for r in out["repo"]]
    anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
    runs = M.runs_of(opt_b[v], anno)
    need = {}
    tracks = []
    for i, t in enumerate(conf):
        det = sorted(f for f, s in t["status"].items() if s == A.DETECTED)
        tracks.append(dict(tid=i, prompt_name=t["category"], detected=det))
        for f in det:
            need.setdefault(f, []).append(("track", i, t["trajectory"][f]))
    for j, r in enumerate(runs):
        for f in r["annotated"]:
            need.setdefault(f, []).append(("gt", j, r["boxes"][f - r["begin_fid"]]))
    return s3, tracks, runs, need


def run(videos):
    import torch
    from PIL import Image
    import adapter as A
    import metrics as M
    from utils import paths
    os.chdir(REPO)
    check_extracts()
    if sha256(CKPT) != CKPT_SHA:
        raise SystemExit("!! STOP: the end-to-end checkpoint differs from its pin")
    opt_b = json.load(open(OPT_B))
    vdir = os.path.join(OUT, "videos")
    os.makedirs(vdir, exist_ok=True)
    man = os.path.join(OUT, "manifest.jsonl")
    done = set()
    if os.path.exists(man):
        for line in open(man):
            r = json.loads(line)
            p = os.path.join(vdir, f"{r['video']}.json.gz")
            if os.path.exists(p) and sha256(p) == r["sha256"]:
                done.add(r["video"])
    m, tag, pre = build_models()
    for k, v in enumerate(videos):
        if v in done:
            continue
        t0 = time.time()
        s3, tracks, runs, need = boxes_of_video(v, opt_b, paths, A, M)
        w, h = s3["width"], s3["height"]
        files = sorted(os.listdir(os.path.join(paths.FRAME_DIR, v)))
        t_vec = [np.zeros(35) for _ in tracks]
        t_top = [[] for _ in tracks]
        g_vec = [np.zeros(35) for _ in runs]
        g_top = [[] for _ in runs]
        t_enc = t_mb = 0.0
        for f in sorted(need):
            a = time.time()
            with torch.no_grad():
                img = Image.open(os.path.join(paths.FRAME_DIR, v, files[f])).convert("RGB")
                assert img.size == (w, h)
                x = pre(img).unsqueeze(0).cuda()
                p_, pp = tag.encode_image_tagclip(x, 336, 336, attn_mask=1)
                del p_
            t_enc += time.time() - a
            a = time.time()
            for kind, i, box in need[f]:
                with torch.no_grad():
                    probs = reorder(m(pooled_feature(pp, box, w, h))[0].detach().float().cpu().tolist())
                if kind == "track":
                    t_vec[i] += probs
                    t_top[i].append(int(np.argmax(probs)))
                else:
                    g_vec[i] += probs
                    g_top[i].append(int(np.argmax(probs)))
            t_mb += time.time() - a
        rec = dict(video=v, width=w, height=h, sam3_file_sha256=sha256(os.path.join(SAM3_DIR, f"{v}.json.gz")),
                   frames_encoded=len(need), seconds_encode=round(t_enc, 1), seconds_modelB=round(t_mb, 1),
                   tracks=[dict(tid=t["tid"], prompt_name=t["prompt_name"], detected_frames=len(t["detected"]),
                                mean_vector=(t_vec[i] / len(t["detected"])).tolist(), per_box_top1=t_top[i])
                           for i, t in enumerate(tracks)],
                   gt_runs=[dict(rid=r["rid"], category=r["category"], annotated_frames=len(r["annotated"]),
                                 mean_vector=(g_vec[j] / len(r["annotated"])).tolist() if r["annotated"] else None,
                                 per_box_top1=g_top[j]) for j, r in enumerate(runs)])
        p = os.path.join(vdir, f"{v}.json.gz")
        with gzip.open(p + ".part", "wt") as fh:
            json.dump(rec, fh)
        os.replace(p + ".part", p)
        with open(man, "a") as fh:
            fh.write(json.dumps(dict(video=v, sha256=sha256(p), seconds=round(time.time() - t0, 1),
                                     frames=len(need), boxes=sum(len(x) for x in need.values()))) + "\n")
        print(f"[{k + 1}/{len(videos)}] {v}: {len(need)} frames, {sum(len(x) for x in need.values())} boxes, "
              f"{time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- analyse
def name_of(vec, id2cls):
    return id2cls[str(int(np.argmax(vec)))]


def renamed(repo_tracks, names):
    return [dict(t, category=names[i]) for i, t in enumerate(repo_tracks)]


def summarise(pooled_rec):
    r = pooled_rec["run_recall_aware"]
    ra = pooled_rec["run_recall_agnostic"]
    return dict(recall=r["all"]["recall"], recall_base=r["base"]["recall"], recall_novel=r["novel"]["recall"],
                recall_counts={g: [r[g]["hit"], r[g]["n"]] for g in ("all", "base", "novel")},
                pair_recall=pooled_rec["pairs"]["recall"], pairs_n=pooled_rec["pairs"]["n"],
                burden=pooled_rec["unmatched_per_video"],
                agnostic=dict(recall=ra["all"]["recall"], recall_base=ra["base"]["recall"],
                              recall_novel=ra["novel"]["recall"], pair_recall=pooled_rec["pairs"]["recall_agnostic"],
                              burden=pooled_rec["unmatched_agnostic_per_video"]))


def gains(sys_, ref):
    return dict(recall_points=100 * (sys_["recall"] - ref["recall"]),
                recall_base_points=100 * (sys_["recall_base"] - ref["recall_base"]),
                recall_novel_points=100 * (sys_["recall_novel"] - ref["recall_novel"]),
                pair_recall_points=100 * (sys_["pair_recall"] - ref["pair_recall"]),
                burden_ratio=sys_["burden"] / ref["burden"] if ref["burden"] else None)


def stop_rule(gain_prompt, gain_modelb):
    """A33's binding stop-and-report rule, on the class-aware overall object-recall gains (points)."""
    if gain_prompt <= 2.0:
        return "STOP: the prompt-name gain is <= +2 points; the check cannot discriminate"
    if gain_modelb < gain_prompt / 2:
        return "STOP: modelB's gain is less than half of the prompt names' gain; naming is the bottleneck"
    return "CONTINUE: modelB's gain is at least half of the prompt names' gain"


def analyse():
    import adapter as A
    import metrics as M
    from utils import paths
    os.chdir(REPO)
    check_extracts()
    vids = [v for v in held_a_videos() if os.path.exists(os.path.join(SAM3_DIR, f"{v}.json.gz"))]
    opt_b = json.load(open(OPT_B))
    rels_all = json.load(open(HELD_GT))
    rels_all.update(json.load(open(HELD_GT3)))
    sp = json.load(open(paths.OBJ_SPLIT_INFO))
    id2cls = sp["id2cls"]
    novel = {c for c, s in sp["cls2split"].items() if s == "novel"}
    runs, rec = {}, {"prompt": {}, "modelB": {}, "R0w": {}}
    acc = {"modelB": {"base": [0, 0], "novel": [0, 0]}, "prompt": {"base": [0, 0], "novel": [0, 0]}}
    gt_box = {"base": [0, 0], "novel": [0, 0]}
    gt_run = {"base": [0, 0], "novel": [0, 0]}
    seconds = frames = boxes = 0
    man = {json.loads(l)["video"]: json.loads(l) for l in open(os.path.join(OUT, "manifest.jsonl"))}
    for v in vids:
        p = os.path.join(OUT, "videos", f"{v}.json.gz")
        if sha256(p) != man[v]["sha256"]:
            raise SystemExit(f"!! STOP: {p} differs from the manifest")
        nc = json.load(gzip.open(p, "rt"))
        seconds += man[v]["seconds"]
        frames += man[v]["frames"]
        boxes += man[v]["boxes"]
        s3 = json.load(gzip.open(os.path.join(SAM3_DIR, f"{v}.json.gz"), "rt"))
        if sha256(os.path.join(SAM3_DIR, f"{v}.json.gz")) != nc["sam3_file_sha256"]:
            raise SystemExit(f"!! STOP: {v}: another SAM 3 file was named")
        out = A.tracks_from_sam3(s3)
        names = [name_of(t["mean_vector"], id2cls) for t in nc["tracks"]]
        assert [t["prompt_name"] for t in nc["tracks"]] == [r["category"] for r in out["repo"]]
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        runs[v] = M.runs_of(opt_b[v], anno)
        e = json.load(gzip.open(os.path.join(EOV_DIR, f"{v}.json.gz"), "rt"))
        sets = {"prompt": M.tracks_from_list(out["repo"]), "modelB": M.tracks_from_list(renamed(out["repo"], names)),
                "R0w": M.tracks_from_list(e["configs"]["R0w"]["tracks"])}
        for k, tr in sets.items():
            rec[k][v] = M.video_record(tr, runs[v], rels_all[v])
        by_rid = {r["rid"]: r for r in runs[v]}
        for rid, (tid, _) in rec["prompt"][v]["matched_agnostic"].items():
            g = "novel" if by_rid[rid]["category"] in novel else "base"
            acc["modelB"][g][0] += names[tid] == by_rid[rid]["category"]
            acc["modelB"][g][1] += 1
            acc["prompt"][g][0] += out["repo"][tid]["category"] == by_rid[rid]["category"]
            acc["prompt"][g][1] += 1
        for gr in nc["gt_runs"]:
            g = "novel" if gr["category"] in novel else "base"
            gt_box[g][0] += sum(1 for x in gr["per_box_top1"] if id2cls[str(x)] == gr["category"])
            gt_box[g][1] += len(gr["per_box_top1"])
            if gr["mean_vector"] is not None:
                gt_run[g][0] += name_of(gr["mean_vector"], id2cls) == gr["category"]
                gt_run[g][1] += 1
    summ = {k: summarise(M.pooled(rec[k], runs, rels_all, novel)) for k in rec}
    g_prompt, g_mb = gains(summ["prompt"], summ["R0w"]), gains(summ["modelB"], summ["R0w"])
    frac = lambda x: dict(correct=x[0], n=x[1], accuracy=x[0] / x[1] if x[1] else None)  # noqa: E731
    res = dict(
        label="A33: the naming check (HELD-A, in-sample for modelB and R0w on base objects)",
        videos=len(vids), systems=summ,
        gains_over_R0w={"(i) prompt names": g_prompt, "(ii) modelB names": g_mb},
        verdict=stop_rule(g_prompt["recall_points"], g_mb["recall_points"]),
        naming_accuracy_on_agnostic_matches={k: {g: frac(v[g]) for g in v} for k, v in acc.items()},
        modelB_top1_gt_boxes={g: frac(gt_box[g]) for g in gt_box},
        modelB_top1_gt_runs_mean_vector={g: frac(gt_run[g]) for g in gt_run},
        runtime=dict(seconds=seconds, hours=seconds / 3600, frames_encoded=frames, boxes=boxes))
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: res[k] for k in ("verdict", "gains_over_R0w", "naming_accuracy_on_agnostic_matches",
                                           "modelB_top1_gt_boxes", "runtime")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--videos", nargs="*", default=None)
    a = ap.parse_args()
    os.chdir(REPO)
    vids = [v for v in held_a_videos() if os.path.exists(os.path.join(SAM3_DIR, f"{v}.json.gz"))]
    assert len(vids) == 78
    if a.stage == "run":
        run(a.videos or vids)
    else:
        analyse()


if __name__ == "__main__":
    main()
