#!/usr/bin/env python3
"""D2 Step 2 (A25, "Recording"): run the front end on test videos with the recorders on.

    python -m d2.record_run --code_root /path/to/ov-vidvrd-lab --out OUT --first_last   # the 2-video check
    python -m d2.record_run --code_root /path/to/ov-vidvrd-lab --out OUT                # all 200 (resumable)

GPU only, and only after the time and cost estimate is approved. Built and run as session 3's
`pilot_analysis/stage1c/sgdet_cache.py` built and ran SGDet: seed 3407, `--frame_stride 1`,
the end-to-end checkpoint, AFLink_epoch20.pth, the shipped VidVRD_ECC_test.json, the repo's
`End2End_Model.forward` unchanged (modelC included). The only addition is d2.recorders.

Ground rule 3: after every video the recorded final tracks must equal the baseline's
(d2/data/baseline_tracks.json); any difference writes OUT/STOP_ground_rule_3.json and exits 3.
Per video: OUT/test/<video>.npz (arrays) and OUT/test/<video>.json (meta, filter output, final
tracks), both hashed into OUT/manifest.jsonl.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # METOR/studies, which holds `d2`

CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"
AFLINK = "output/ckpt/AFLink_epoch20.pth"
SEED = 3407


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_root", required=True, help="EOV-MMP working repo at fdeeeb4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=os.path.join(HERE, "data", "baseline_tracks.json"))
    ap.add_argument("--first_last", action="store_true", help="only the first and last video of the test list")
    ap.add_argument("--videos", default="", help="comma-separated video names")
    ap.add_argument("--check_only", action="store_true", help="verify the pins and stop before loading anything")
    a = ap.parse_args()

    from d2.common import CODE_PINS, EOV_COMMIT, FRONT_END_FILES, env_info, git_head, sha256, sha256_or_none
    code_root = os.path.abspath(a.code_root)
    a.out, a.baseline = os.path.abspath(a.out), os.path.abspath(a.baseline)
    gh = git_head(code_root)
    if gh.get("head") not in (None, EOV_COMMIT):
        raise SystemExit(f"!! STOP: code root is at {gh.get('head')}, A25 fixes {EOV_COMMIT}")
    pins = json.load(open(CODE_PINS))
    bad = [f for f in FRONT_END_FILES if sha256_or_none(os.path.join(code_root, f)) != pins["front_end_eov"][f]]
    if bad:
        raise SystemExit(f"!! STOP: front-end files differ from A25a's pins: {bad}")
    if sha256(a.baseline) != pins["baseline_tracks_json"]:
        raise SystemExit("!! STOP: baseline_tracks.json differs from A25a's pin")
    d2_now = {f"d2/{f}": sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")}
    if d2_now != pins["d2_code"]:
        raise SystemExit("!! STOP: d2 code differs from A25a's pins")
    for rel, want in (("output/ckpt/" + os.path.basename(CKPT), pins["inputs"]["end2end_ckpt"]),
                      ("output/ckpt/" + os.path.basename(AFLINK), pins["inputs"]["aflink"]),
                      ("data/vidvrd/data/VidVRD_ECC_test.json", pins["inputs"]["ecc_test"]),
                      ("data/vidvrd/data/openvoc_obj_class_spilt_info.json", pins["inputs"]["obj_split"])):
        if sha256(os.path.join(code_root, rel)) != want:
            raise SystemExit(f"!! STOP: {rel} differs from its pin")
    print("pins verified: front end, d2 code, baseline tracks, checkpoint, AFLink, ECC, class split", flush=True)
    if a.check_only:
        return
    sys.path.insert(0, code_root)
    os.chdir(code_root)                              # session 3 ran from the repo root

    import numpy as np
    import torch
    from cli.common import build_eval_data, build_model_stack, load_end2end, seed_everything
    from utils.parser_func import parse_args
    from d2.baseline_tracks import compare, load
    from d2.recorders import Recorder

    os.makedirs(os.path.join(a.out, "test"), exist_ok=True)
    baseline = load(a.baseline)
    seed_everything(SEED)
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", os.path.join(code_root, CKPT),
                                "--path_AFLink", os.path.join(code_root, AFLINK), "--frame_stride", "1"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    man_path = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line)
            ok = all(os.path.exists(os.path.join(a.out, "test", f"{r['video']}{e}")) and
                     sha256(os.path.join(a.out, "test", f"{r['video']}{e}")) == r[f"sha256{e}"] for e in (".npz", ".json"))
            if ok:
                done.add(r["video"])

    val_dataset, _ = build_eval_data(args)
    full = list(val_dataset.path_list)
    if a.videos:
        want = a.videos.split(",")
    elif a.first_last:
        want = [full[0], full[-1]]
    else:
        want = full
    missing = [v for v in want if v not in full]
    if missing:
        raise SystemExit(f"!! STOP: not in the test list: {missing}")
    todo = [v for v in want if v not in done]
    val_dataset.path_list = todo
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    class _Log:
        def info(self, *x): print(*x, flush=True)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, _Log())
    model.eval()
    model.modelC.tgt_split = "all"                   # as sgdet_cache.py; modelC does not touch the tracks

    run_log = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), command=" ".join(sys.argv), seed=SEED,
                   args={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()},
                   env=env_info(), code_root=dict(path=code_root, **gh),
                   inputs={p: sha256(os.path.join(code_root, p)) for p in (CKPT, AFLINK,
                           "data/vidvrd/data/VidVRD_ECC_test.json", "data/vidvrd/data/openvoc_obj_class_spilt_info.json")}
                   | {"baseline_tracks.json": sha256(a.baseline)},
                   d2_code={f: sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
                   todo=todo, already_done=sorted(done))
    with open(os.path.join(a.out, f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(run_log, f, indent=1)

    rec = Recorder().install(model)
    t0 = time.time()
    try:
        for i, data in enumerate(loader, 1):
            vid = data["video_name"][0]
            tv = time.time()
            rec.start(vid, int(data["begin_fid"][0]), int(data["end_fid"][0]), int(data["video_len"][0]),
                      int(data["image_size"][0]), int(data["image_size"][1]))
            with torch.no_grad():
                model(data, sort_model)
            c = rec.finish(model.traj.get(vid, []))
            diffs = compare(c["final"], baseline[vid])
            stem = os.path.join(a.out, "test", vid)
            np.savez_compressed(stem + ".npz", **rec.arrays())
            with open(stem + ".json", "w") as f:
                json.dump(rec.meta(), f)
            bad_post = c["post_box_mismatch"] or c["post_cut_mismatch"]
            if diffs or bad_post:
                with open(os.path.join(a.out, "STOP_ground_rule_3.json"), "w") as f:
                    json.dump(dict(video=vid, diffs=diffs[:200], n_diffs=len(diffs),
                                   post_box_mismatch=c["post_box_mismatch"],
                                   post_cut_mismatch=c["post_cut_mismatch"]), f, indent=1)
                print(f"!! STOP {vid}: {len(diffs)} track differences, post-processor mismatches "
                      f"{c['post_box_mismatch']}/{c['post_cut_mismatch']}", flush=True)
                sys.exit(3)
            r = dict(video=vid, seconds=round(time.time() - tv, 1), frames=c["end_fid"] - c["begin_fid"],
                     dets=int(sum(len(d["q"]) for d in c["det"])), tracker_input=int(len(c["ti"]["frame"]) if c["ti"] else 0),
                     temp_rows=int(0 if c["temp"] is None else len(c["temp"])), final_tracks=len(c["final"]),
                     ground_rule_3="identical",
                     **{f"sha256{e}": sha256(stem + e) for e in (".npz", ".json")},
                     **{f"bytes{e}": os.path.getsize(stem + e) for e in (".npz", ".json")})
            with open(man_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            el = time.time() - t0
            print(f"[D2 {i}/{len(todo)}] {vid} frames={r['frames']} dets={r['dets']} tracks={r['final_tracks']} "
                  f"{r['seconds']}s identical | elapsed {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.0f} min",
                  flush=True)
            model.traj.pop(vid, None)
            torch.cuda.empty_cache()
    finally:
        rec.uninstall()
    print("done: every recorded video is identical to the baseline", flush=True)


if __name__ == "__main__":
    main()
