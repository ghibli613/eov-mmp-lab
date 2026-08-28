#!/usr/bin/env python
"""Phase 1 -- run inference and SAVE both prediction files. NEEDS A >=16GB GPU.

    python pilot_analysis/scripts/dump_predictions.py \
        --ckpt_path output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth \
        --path_AFLink output/ckpt/AFLink_epoch20.pth \
        --frame_stride 30 \
        --limit 5                      # drop --limit for the full test set

Why this exists
---------------
cli/evaluate.py computes mAP and throws the predictions away. Every later phase
of the pilot needs them kept, and Phase 3 needs the PRE-MERGE per-segment
predictions, which nothing currently writes. Re-running inference to get them
would double the GPU bill, so both files are dumped in one pass.

This is a WRAPPER: it re-implements cli/common.py:_predict_split with two
json.dump calls added and changes nothing in the repo. The model, the
post-processing and the evaluator are all the repo's own.

Writes, under pilot_analysis/preds/:
  final_merged_{split}.json   post-merge instances, the evaluator's input format
  segments_raw_{split}.json   pre-merge per-segment predictions:
                              {video_id: [{segment_index, frame_range, triplet,
                                           confidence, sbj_scr, obj_scr}, ...]}
  metrics.json                mAP / R@50 / R@100 per split, plus the config used

Run the sanity gate first: `--limit 5` and check the printed shapes and the
runtime estimate before launching the full 200 videos.

CRASH SAFETY (matters on Colab): results are flushed to disk every
`--flush-every` videos (default 5) via write-to-temp-then-rename, and the run
RESUMES from whatever is already in pilot_analysis/preds/ -- re-running the same
command after a disconnect picks up where it stopped. A disconnect costs at most
5 videos, not the whole run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from tqdm import tqdm

from cli.common import (build_eval_data, build_model_stack, load_end2end,
                        make_logger, require_files, seed_everything)
from inference.post_process import association, format_, process_pred
from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
from utils.parser_func import parse_args

#: Output directory. Override with --out or $PILOT_PREDS_DIR so results can land
#: on a mounted Drive instead of Colab's ephemeral disk.
OUT = os.environ.get("PILOT_PREDS_DIR", "pilot_analysis/preds")
#: If set (--drive-copy), every flush is mirrored here after the atomic local
#: write. Writing atomically DIRECTLY to a Drive FUSE mount is unreliable --
#: os.replace across FUSE is not guaranteed atomic -- so local stays the source
#: of truth and Drive gets a plain copy.
DRIVE_COPY = None


def _plan_batches(shards, budget_bytes):
    """Group shards into batches under a byte budget, never splitting a video --
    the dataset reads every frame of a video in one __getitem__, so a video is
    unusable until all its chunks are present."""
    by_video = {}
    for sh in shards:
        by_video.setdefault(sh["video_id"], []).append(sh)
    out, cur, cur_b = [], [], 0
    for vid in sorted(by_video):
        group = sorted(by_video[vid], key=lambda e: e["chunk_index"])
        sz = sum(e["bytes"] for e in group)
        if cur and cur_b + sz > budget_bytes:
            out.append(cur)
            cur, cur_b = [], 0
        cur.extend(group)
        cur_b += sz
    if cur:
        out.append(cur)
    return out


def _fetch_batch(batch, url_of, staging, frame_dir):
    """Download, verify and untar one batch. Returns the video ids it provides."""
    from tools.hugging_download import download, sha256
    os.makedirs(staging, exist_ok=True)
    for sh in batch:
        dst = os.path.join(staging, sh["tar"])
        if not (os.path.exists(dst) and os.path.getsize(dst) == sh["bytes"]
                and sha256(dst) == sh["sha256"]):
            src = url_of(sh["tar"])
            if "://" in src:
                download(src, dst)
            else:                       # a local bundle -- copy, do not fetch
                shutil.copy2(src, dst)
            if sha256(dst) != sh["sha256"]:
                raise SystemExit(f"sha256 mismatch on {sh['tar']} -- corrupt transfer")
        with tarfile.open(dst) as tf:
            # filter='data' is required from Python 3.14; without it extractall
            # warns now and will reject or rewrite metadata later
            try:
                tf.extractall(frame_dir, filter="data")
            except TypeError:                       # older Python
                tf.extractall(frame_dir)
        os.remove(dst)                              # the tar is dead once unpacked
    return sorted({sh["video_id"] for sh in batch})


def _drop_batch(video_ids, frame_dir):
    for vid in video_ids:
        shutil.rmtree(os.path.join(frame_dir, vid), ignore_errors=True)


def extract_segments(clip_rels):
    """clip_rels (process_pred's output) -> the pre-merge records for
    segments_raw.json. Split out so it can be unit-tested without a GPU; see
    pilot_analysis/scripts/test_dump_logic.py."""
    out = []
    for seg_idx, cands in enumerate(clip_rels):
        for c in cands:
            out.append({
                "segment_index": seg_idx,
                "frame_range": list(c["duration"]),
                "triplet": [c["sbj_cls"], c["pre_cls"], c["obj_cls"]],
                "confidence": float(c["pre_scr"]),
                "sbj_scr": float(c["sbj_scr"]),
                "obj_scr": float(c["obj_scr"]),
            })
    return out


def predict_and_dump(args, model, sort_model, val_loader, val_dataset, split, limit,
                     flush_every=5, evaluate=True):
    """cli/common.py:_predict_split, with incremental dumps added.

    Results are flushed to disk every `flush_every` videos and the run RESUMES
    from whatever is already on disk, so a Colab disconnect costs at most
    `flush_every` videos instead of the whole run. Merging happens per video
    anyway (association() is called per pair, format_ per video), so flushing
    early changes no number -- only when bytes hit the disk.
    """
    model.modelC.tgt_split = split
    # The resume/skip check reads data["video_name"][0], which is only the whole
    # batch when batch_size == 1 (the default, args.batch_size=1). With a larger
    # batch it would skip videos it had not actually finished.
    if args.batch_size != 1:
        raise SystemExit(
            f"--batch_size must be 1 for the resume logic to be correct "
            f"(got {args.batch_size}). cli/common.py builds the eval loader with "
            f"batch_size=args.batch_size.")
    os.makedirs(OUT, exist_ok=True)
    merged_path = f"{OUT}/final_merged_{split}.json"
    segs_path = f"{OUT}/segments_raw_{split}.json"

    # ---- resume: reload whatever a previous session already finished
    pred_rels, segments = defaultdict(list), defaultdict(list)
    done = set()
    if os.path.exists(merged_path):
        try:
            prev = json.load(open(merged_path))
            prevs = json.load(open(segs_path)) if os.path.exists(segs_path) else {}
            for v, r in prev.items():
                pred_rels[v] = r
            for v, r in prevs.items():
                segments[v] = r
            done = set(prev)
            print(f"[{split}] RESUMING: {len(done)} videos already on disk, skipping them")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[{split}] existing dump unreadable ({e}); starting over")
            pred_rels, segments, done = defaultdict(list), defaultdict(list), set()

    def flush():
        # write to a temp file and rename, so a kill mid-write cannot corrupt
        for path, obj in ((merged_path, pred_rels), (segs_path, segments)):
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(obj, fh)
            os.replace(tmp, path)
        if DRIVE_COPY:
            os.makedirs(DRIVE_COPY, exist_ok=True)
            for path in (merged_path, segs_path):
                try:
                    shutil.copy2(path, os.path.join(DRIVE_COPY, os.path.basename(path)))
                except OSError as e:      # a Drive hiccup must not kill the run
                    print(f"  [warn] Drive copy failed ({e}); local copy is intact")

    seen = set(done)
    pending = []                      # videos finished but not yet formatted+flushed
    t0 = time.time()
    n_new = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"eval[{split}]"):
            vid_in = data["video_name"][0]
            if vid_in in done:
                continue              # already have it from a previous session
            if limit and len(seen) - len(done) >= limit and vid_in not in seen:
                break
            for final_result in model(data, sort_model):
                pre_preds = final_result["pre_preds"]
                seq_lens = final_result["seq_lens"]
                vids = final_result["video_name"]
                pair_data = final_result["pair_data"]
                for seq_id, seq_len in enumerate(seq_lens):
                    vid = vids[seq_id]
                    if vid not in seen:
                        seen.add(vid)
                        pending.append(vid)
                    clip_rels = process_pred(
                        args, val_dataset.id2pre, val_dataset.obj2id, val_dataset.prior,
                        pre_preds[seq_id][:seq_len], pair_data[seq_id])
                    # ---- the pre-merge dump, before association() mutates anything
                    segments[vid].extend(extract_segments(clip_rels))
                    pred_rels[vid].extend(association(clip_rels))
            if len(pending) >= flush_every:
                for v in pending:
                    pred_rels[v] = format_(args, pred_rels[v])
                n_new += len(pending)
                pending = []
                flush()
    for v in pending:                 # the tail
        pred_rels[v] = format_(args, pred_rels[v])
    n_new += len(pending)
    flush()
    elapsed = time.time() - t0

    n_seg = sum(len(v) for v in segments.values())
    n_rel = sum(len(v) for v in pred_rels.values())
    print(f"\n[{split}] {n_new} new videos in {elapsed/60:.1f} min "
          f"({elapsed/max(n_new,1):.1f} s/video); {len(pred_rels)} total on disk")
    print(f"[{split}] {n_seg} segment predictions, {n_rel} merged instances")
    if limit and n_new:
        full = elapsed / n_new * 200 / 60
        print(f"[{split}] ESTIMATE for all 200 videos: {full:.0f} min "
              f"({full/60:.1f} h). The protocol says ask before exceeding 2 h.")
    if n_rel:
        ex = next(r for r in pred_rels.values() if r)[0]
        print(f"[{split}] sample merged instance: triplet={ex['triplet']} "
              f"duration={ex['duration']} score={ex['score']:.4f} "
              f"len(sub_traj)={len(ex['sub_traj'])}")

    if not evaluate:
        # Mid-run scoring would average AP over videos that have not been run
        # yet, which reads as a collapse rather than as progress. Only the
        # final call scores.
        return {"videos": len(pred_rels), "new_videos": n_new,
                "segments": n_seg, "instances": n_rel, "minutes": elapsed / 60}
    mean_ap, rec = eval_relation_detection_openvoc(
        target_split_pred=split, prediction_results=dict(pred_rels))
    return {"mAP": mean_ap, "R@50": rec[50], "R@100": rec[100],
            "videos": len(pred_rels), "new_videos": n_new,
            "segments": n_seg, "instances": n_rel, "minutes": elapsed / 60}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--limit", type=int, default=0,
                     help="stop after N NEW videos (sanity gate); 0 = full test set")
    pre.add_argument("--flush-every", type=int, default=5,
                     help="write results to disk every N videos (default 5). Lower "
                          "is safer on a flaky Colab session, slightly slower.")
    pre.add_argument("--out", default=None,
                     help="output directory (default $PILOT_PREDS_DIR or "
                          "pilot_analysis/preds)")
    pre.add_argument("--drive-copy", default=None, metavar="DIR",
                     help="mirror each flush here after the atomic local write, "
                          "e.g. /content/drive/MyDrive/vidvrd/preds")
    pre.add_argument("--shards", default=None, metavar="PATH_OR_URL",
                     help="SHARDS.json. Enables batched mode: fetch a batch of "
                          "per-video frame tars, run them, delete, repeat. The "
                          "model is loaded once and reused across batches.")
    pre.add_argument("--disk-budget", type=float, default=1.0, metavar="GIB",
                     help="batched mode: frames on disk at once (default 1.0 GiB)")
    pre.add_argument("--staging", default="/tmp/vidvrd_shards",
                     help="batched mode: where tars land before untarring")
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    global OUT, DRIVE_COPY
    if known.out:
        OUT = known.out
    DRIVE_COPY = known.drive_copy

    seed_everything(3407)
    args = parse_args()
    if not args.ckpt_path:
        raise SystemExit("--ckpt_path <trained end-to-end .pth> is required")
    require_files([("--ckpt_path", args.ckpt_path),
                   ("--path_AFLink", args.path_AFLink)])

    logger = make_logger(args, "pilot_dump")
    val_dataset, val_loader = build_eval_data(args)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, logger)
    model.eval()

    metrics = {"config": {"ckpt_path": args.ckpt_path,
                          "frame_stride": args.frame_stride,
                          "clip_len": args.clip_len,
                          "clip_top_n": args.clip_top_n,
                          "max_per_video": args.max_per_video,
                          "limit": known.limit,
                          "batched": bool(known.shards),
                          "disk_budget_gib": known.disk_budget if known.shards else None}}

    if not known.shards:
        # ---- all frames already on disk
        for split in ("all", "novel"):
            metrics[split] = predict_and_dump(
                args, model, sort_model, val_loader, val_dataset, split, known.limit,
                flush_every=known.flush_every)
            m = metrics[split]
            print(f"[{split}] mAP {m['mAP']*100:.2f}  R@50 {m['R@50']*100:.2f}  "
                  f"R@100 {m['R@100']*100:.2f}")
    else:
        # ---- batched: fetch a slice of the frames, run it, delete it, repeat.
        # The model stays loaded (it costs ~9 min to build), and val_dataset is
        # reused with its path_list narrowed to the batch -- Dataset_new takes
        # path_list from test_object_trajectories_gt.json, NOT from listing the
        # frame directory, so without narrowing it __getitem__ would raise
        # FileNotFoundError on the first video that is not on disk.
        from torch.utils.data import DataLoader

        from tools.hugging_download import load_manifest
        from utils import paths

        idx = load_manifest(known.shards)
        base = known.shards.rsplit("/", 1)[0] if "://" in known.shards else None
        man = None
        if base:
            man = load_manifest(base + "/MANIFEST.json")
        def url_of(name):
            if man:
                return f"{man['base_url']}/{name}"
            return os.path.join(os.path.dirname(known.shards), name)

        all_videos = set(val_dataset.path_list)
        shards = [sh for sh in idx["shards"] if sh["video_id"] in all_videos]
        skipped = {sh["video_id"] for sh in idx["shards"]} - all_videos
        if skipped:
            print(f"  {len(skipped)} video(s) in SHARDS.json are not in this split; ignored")
        batches = _plan_batches(shards, int(known.disk_budget * 1024 ** 3))
        if known.limit:
            # --limit is a TOTAL, not a per-batch cap. Without this the sanity
            # gate would run `limit` videos in each of the ~29 batches.
            trimmed, seen_v = [], 0
            for b in batches:
                vids_b = {e["video_id"] for e in b}
                if seen_v >= known.limit:
                    break
                trimmed.append(b)
                seen_v += len(vids_b)
            batches = trimmed
            print(f"  --limit {known.limit}: keeping {len(batches)} batch(es)")
        print(f"\nbatched mode: {len(shards)} tars over {len(batches)} batches "
              f"at {known.disk_budget:.2f} GiB\n")

        full_list = list(val_dataset.path_list)
        for split in ("all", "novel"):
            for bi, batch in enumerate(batches):
                vids = _fetch_batch(batch, url_of, known.staging, paths.FRAME_DIR)
                val_dataset.path_list = [v for v in full_list if v in set(vids)]
                loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                print(f"[{split}] batch {bi+1}/{len(batches)}: {len(val_dataset.path_list)} videos")
                metrics[split] = predict_and_dump(
                    args, model, sort_model, loader, val_dataset, split, known.limit,
                    flush_every=known.flush_every,
                    evaluate=(bi == len(batches) - 1))
                _drop_batch(vids, paths.FRAME_DIR)
            val_dataset.path_list = full_list
            m = metrics[split]
            print(f"[{split}] mAP {m['mAP']*100:.2f}  R@50 {m['R@50']*100:.2f}  "
                  f"R@100 {m['R@100']*100:.2f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    print(f"\nwrote {OUT}/metrics.json")
    print("Phase 1 targets: all mAP 26.34 / novel mAP 15.04 (paper), or "
          "26.88 / 15.64 (this checkpoint's reported figures -- see "
          "PILOT-STATUS.md SS C.3).")


if __name__ == "__main__":
    main()
