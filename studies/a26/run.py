#!/usr/bin/env python3
"""A26 on the GPU machine: R0, R0u, R0w and R0uw from one encoding of each test video.

    python -m a26.run --data_root ~/ov-vidvrd/data/vidvrd --weights_root ~/ov-vidvrd --out ~/a26_out --check_only
    python -m a26.run ... --first_last      # the 2-video gate: identity with session 3, timing, projection
    python -m a26.run ...                   # all 200 (resumable)

GPU only, and only after the author approves the time and cost estimate. METOR's code, unmodified,
as A23 sets R0 and R0u: `--arch metor-compat --n_iter 0`, seed 3407, `--frame_stride 1`, the
released end-to-end checkpoint, AFLink_epoch20.pth. Per video, Dataset_new('val') encodes the frames
once; then End2End_Model runs four times, once per configuration (common.CONFIGS):
  - the frame range: the loader's (GT-derived), or the whole video (wrappers.whole_video);
  - the fusion: per-class, as released, or uniform (`args.uniform_fusion`, models/fusion.py).
modelC runs in both predicate splits on the same pass (wrappers.BothSplits). The predictions go
through the repo's own process_pred -> association -> format_, exactly as cli/common.py:_predict_split.

Gates (any failure writes OUT/STOP_*.json and exits non-zero):
  - R0's final tracks equal session 3's (D2's baseline_tracks.json) on every video;        exit 3
  - --first_last: R0's predictions for both videos equal session 3's (r0_reference.json),
    scores within 1e-6;                                                                    exit 5
  - --first_last: the projected full run is at most STOP_HOURS;                            exit 4
Per video and configuration: OUT/<config>/<video>.json.gz (predictions for both splits, final
tracks, frame range), each hashed into OUT/manifest.jsonl. Nothing here reads a label file; only
the pipeline's own Dataset_new does, as on every test run.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
import os
import platform
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.dirname(HERE)
sys.path.insert(0, STUDIES)                          # METOR/studies, which holds `a26`

from a26.common import (AFLINK, CKPT, CONFIGS, FRAMES_GT, FRAMES_WHOLE, LOGS, METOR_ROOT,  # noqa: E402
                        REFERENCE, RUN_ARGV, SCORE_TOL, SEED, STOP_HOURS)

PRICE = (0.35, 0.60)                                 # $/h, the measured rate and a margin


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_info() -> dict:
    import numpy
    import torch
    out = dict(python=sys.version.split()[0], platform=platform.platform(), host=platform.node(),
               numpy=numpy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
               omp_num_threads=os.environ.get("OMP_NUM_THREADS"))
    if torch.cuda.is_available():
        out.update(gpu=torch.cuda.get_device_name(0), cudnn=torch.backends.cudnn.version(),
                   tf32=dict(cudnn=torch.backends.cudnn.allow_tf32, matmul=torch.backends.cuda.matmul.allow_tf32))
    return out


def predictions_for(args, ds, results, other, process_pred, association, format_) -> dict:
    """final_results and the stashed novel outputs -> {split: formatted predictions}, through the
    repo's chain exactly as _predict_split runs it (batch size 1)."""
    if len(other) != len(results):
        raise RuntimeError(f"{len(results)} results but {len(other)} novel outputs")
    rels = {"all": [], "novel": []}
    for fr, nov in zip(results, other):
        for split, pre_preds in (("all", fr["pre_preds"]), ("novel", nov[0])):
            for seq_id, seq_len in enumerate(fr["seq_lens"]):
                clip_rels = process_pred(args, ds.id2pre, ds.obj2id, ds.prior,
                                         pre_preds[seq_id][:seq_len], fr["pair_data"][seq_id])
                rels[split].extend(association(clip_rels))
    # _predict_split formats only the videos that produced results; the evaluator reads a missing
    # video as [] (prediction.get(vid, [])), so [] here is the same thing.
    return {s: (format_(args, r) if results else []) for s, r in rels.items()}


def run_configs(model, sort_model, args, ds, data, configs, fns) -> dict:
    """The four configurations on one encoded video. Returns {name: record}."""
    import torch
    from a26.tracks import serialize_tracks
    from a26.wrappers import config_batch, set_fusion
    process_pred, association, format_ = fns
    vid = data["video_name"][0]
    out = {}
    try:
        for name, frame_range, uniform in configs:
            set_fusion(model, uniform)
            batch = config_batch(data, frame_range)
            model.modelC.other.clear()
            model.traj.pop(vid, None)                # End2End_Model sets it only when tracks exist
            t = time.time()
            with torch.no_grad():
                results = list(model(batch, sort_model) or [])
            preds = predictions_for(args, ds, results, list(model.modelC.other),
                                    process_pred, association, format_)
            tracks = serialize_tracks(model.traj.get(vid, []))
            out[name] = dict(video=vid, config=name, frame_range=frame_range, uniform_fusion=uniform,
                             begin_fid=int(batch["begin_fid"][0]), end_fid=int(batch["end_fid"][0]),
                             video_len=int(batch["video_len"][0]), seconds=round(time.time() - t, 2),
                             predictions=json.loads(json.dumps(preds)), tracks=tracks)
            model.modelC.other.clear()
            model.traj.pop(vid, None)
    finally:
        set_fusion(model, False)
    return out


def projection(gate_rows: list) -> dict:
    """Hours for all 200 videos from the gate videos' per-frame rates (encoding over every frame;
    the GT-range configurations over FRAMES_GT; the whole-video ones over FRAMES_WHOLE)."""
    enc_s = sum(r["encode_seconds"] for r in gate_rows)
    enc_f = sum(r["video_len"] for r in gate_rows)
    hours = enc_s / enc_f * FRAMES_WHOLE / 3600
    per = {}
    for name, frame_range, _ in CONFIGS:
        s = sum(r["configs"][name]["seconds"] for r in gate_rows)
        f = sum(r["configs"][name]["frames"] for r in gate_rows)
        per[name] = s / f * (FRAMES_GT if frame_range == "gt" else FRAMES_WHOLE) / 3600
        hours += per[name]
    return dict(hours=round(hours, 2), encode_hours=round(enc_s / enc_f * FRAMES_WHOLE / 3600, 2),
                config_hours={k: round(v, 2) for k, v in per.items()},
                cost_usd=[round(hours * p, 2) for p in PRICE], stop_hours=STOP_HOURS,
                note="projected from the gate videos only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="the VidVRD data root (sets VIDVRD_DATA_ROOT)")
    ap.add_argument("--weights_root", required=True, help="holds output/ckpt/ (the EOV work dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=os.path.join(STUDIES, "d2", "data", "baseline_tracks.json"))
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--first_last", action="store_true", help="only the first and last test video: the gate")
    ap.add_argument("--check_only", action="store_true", help="verify the pins and stop before loading anything")
    a = ap.parse_args()

    a.data_root, a.weights_root, a.out = (os.path.abspath(os.path.expanduser(p))
                                          for p in (a.data_root, a.weights_root, a.out))
    pins = json.load(open(os.path.join(LOGS, "code_pins.json")))
    bad = [f for f, h in pins["metor_code"].items() if sha256(os.path.join(METOR_ROOT, f)) != h]
    if bad:
        raise SystemExit(f"!! STOP: METOR code differs from A26's pins: {bad}")
    a26_now = {f: sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")}
    if a26_now != pins["a26_code"]:
        raise SystemExit("!! STOP: a26 code differs from A26's pins")
    inputs = {"ckpt": os.path.join(a.weights_root, CKPT), "aflink": os.path.join(a.weights_root, AFLINK),
              "ecc_test": os.path.join(a.data_root, "data", "VidVRD_ECC_test.json"),
              "obj_split": os.path.join(a.data_root, "data", "openvoc_obj_class_spilt_info.json"),
              "pred_split": os.path.join(a.data_root, "data", "openvoc_pred_class_spilt_info.json"),
              "baseline_tracks": a.baseline, "reference": a.reference}
    for k, p in inputs.items():
        if sha256(p) != pins["inputs"][k]:
            raise SystemExit(f"!! STOP: {k} ({p}) differs from its pin")
    print("pins verified: METOR code, a26 code, checkpoint, AFLink, ECC, class splits, baseline tracks, reference",
          flush=True)
    if a.check_only:
        return

    os.environ["VIDVRD_DATA_ROOT"] = a.data_root     # before utils.paths is imported
    sys.path.insert(0, METOR_ROOT)
    os.chdir(METOR_ROOT)
    import torch
    from cli.common import build_model_stack, load_end2end, seed_everything
    from data_loading.feature_dataset import Dataset_new
    from inference.post_process import association, format_, process_pred
    from utils import paths
    from utils.parser_func import parse_args
    from a26.tracks import compare_predictions, compare_tracks
    from a26.wrappers import BothSplits
    if paths.DATA_ROOT != a.data_root:
        raise SystemExit(f"!! STOP: utils.paths.DATA_ROOT is {paths.DATA_ROOT}")

    seed_everything(SEED)
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", inputs["ckpt"], "--path_AFLink", inputs["aflink"],
                                *RUN_ARGV]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    if getattr(args, "uniform_fusion", False) or args.arch != "metor-compat" or args.n_iter != 0:
        raise SystemExit("!! STOP: the run arguments are not A26's")

    os.makedirs(a.out, exist_ok=True)
    for name, _, _ in CONFIGS:
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
    man_path = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line)
            if all(os.path.exists(p := os.path.join(a.out, n, f"{r['video']}.json.gz")) and sha256(p) == r["sha256"][n]
                   for n, _, _ in CONFIGS):
                done.add(r["video"])

    baseline = json.load(open(a.baseline))
    reference = json.load(open(a.reference))
    ds = Dataset_new(args, "val")
    full = list(ds.path_list)
    want = [full[0], full[-1]] if a.first_last else full
    if a.first_last and sorted(want) != sorted(reference["videos"]):
        raise SystemExit(f"!! STOP: the gate videos {want} are not the reference's {reference['videos']}")
    todo = [v for v in want if v not in done]
    ds.path_list = todo
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    class _Log:
        def info(self, *x): print(*x, flush=True)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, _Log())
    model.eval()
    fe = model.modelC.featEmbedding
    if type(fe).__name__ != "METORFeatEmbedding" or fe.iem is not None:
        raise SystemExit("!! STOP: modelC is not EOV-MMP's (A23: --n_iter 0 must leave iem None)")
    model.modelC = BothSplits(model.modelC)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(a.out, f"run_{stamp}.json"), "w") as f:
        json.dump(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), command=" ".join(sys.argv), seed=SEED,
                       args={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in vars(args).items()},
                       env=env_info(), configs=CONFIGS, inputs={k: sha256(p) for k, p in inputs.items()},
                       a26_code=a26_now, todo=todo, already_done=sorted(done)), f, indent=1)

    def stop(kind, code, **info):
        with open(os.path.join(a.out, f"STOP_{kind}.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        print(f"!! STOP ({kind}): {json.dumps(info)[:400]}", flush=True)
        sys.exit(code)

    fns = (process_pred, association, format_)
    gate_rows = []
    t0 = t_prev = time.time()
    for i, data in enumerate(loader, 1):
        vid = data["video_name"][0]
        enc = time.time() - t_prev
        recs = run_configs(model, sort_model, args, ds, data, CONFIGS, fns)
        diffs = compare_tracks(recs["R0"]["tracks"], baseline[vid])
        if diffs:
            stop("R0_tracks", 3, video=vid, n_diffs=len(diffs), diffs=diffs[:200])
        if vid in reference["predictions"]:
            pd = {s: compare_predictions(recs["R0"]["predictions"][s], reference["predictions"][vid][s], SCORE_TOL)
                  for s in ("all", "novel")}
            if any(pd.values()):
                stop("R0_predictions", 5, video=vid, diffs={s: d[:100] for s, d in pd.items()})
        row = dict(video=vid, video_len=recs["R0"]["video_len"], encode_seconds=round(enc, 2),
                   R0_identity=dict(tracks="identical", predictions="identical" if vid in reference["predictions"]
                                    else "no reference"), configs={}, sha256={})
        for name, rec in recs.items():
            p = os.path.join(a.out, name, f"{vid}.json.gz")
            with gzip.open(p, "wt") as fh:
                json.dump(rec, fh)
            row["configs"][name] = dict(seconds=rec["seconds"], frames=rec["end_fid"] - rec["begin_fid"],
                                        tracks=len(rec["tracks"]),
                                        instances={s: len(v) for s, v in rec["predictions"].items()})
            row["sha256"][name] = sha256(p)
        with open(man_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        gate_rows.append(row)
        el = time.time() - t0
        print(f"[A26 {i}/{len(todo)}] {vid} frames={row['video_len']} enc={enc:.0f}s "
              + " ".join(f"{n}={row['configs'][n]['seconds']:.0f}s" for n in row["configs"])
              + f" R0 identical | elapsed {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
        del recs, data
        torch.cuda.empty_cache()
        t_prev = time.time()

    if a.first_last:
        if len(gate_rows) != 2:
            raise SystemExit("!! STOP: the gate needs both videos run in this invocation (delete OUT and rerun)")
        proj = projection(gate_rows)
        with open(os.path.join(a.out, "gate.json"), "w") as fh:
            json.dump(dict(videos=[r["video"] for r in gate_rows], R0="identical to session 3 (tracks and predictions)",
                           projection=proj), fh, indent=1)
        print(f"gate passed: R0 identical to session 3 on {[r['video'] for r in gate_rows]}; projection {proj}",
              flush=True)
        if proj["hours"] > STOP_HOURS:
            stop("projection", 4, **proj)
    print("done", flush=True)


if __name__ == "__main__":
    main()
