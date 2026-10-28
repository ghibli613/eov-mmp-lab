#!/usr/bin/env python3
"""RPT G1 on the GPU (A29): EOV-MMP's tracks on the 160 HELD videos, R0w and R0 (Definitions item 1),
and the box-sensitivity modelC runs (item 7). Runs after all 160 SAM 3 files exist.

    python studies/rpt/g1/eov_held.py --sam3 ~/rpt_g1_out/sam3 --out ~/rpt_g1_out/eov [--videos V ...]

EOV-MMP's front end as D2 ran it (studies/d2/record_run.py: seed 3407, `--frame_stride 1`, the
end-to-end checkpoint, AFLink_epoch20.pth, End2End_Model.forward unchanged, modelC `tgt_split all`),
with two differences A29 fixes, both from outside the code:
- the loader is the repo's own Dataset_new(args, 'train') restricted to HELD, never 'val': HELD
  frames, and the training annotations' object range (dataset.py:120-126) as R0's range; R0w is
  A26's whole_video (studies/a26/wrappers.py) of the same batch;
- `model.test_ecc` is set to the contents of VidVRD_ECC_train.json, which the released test path
  hands the tracker (end2end_model.py:277); the 3 HELD videos absent from it run without
  compensation, as the released code does for a missing video.
The final tracks are model.traj after the 15% rule (end2end_model.py:314-325), serialized as A26 did.

Box sensitivity (item 7), per video, on the same encoded frames: the HELD base GT relations that
both SAM 3's confident tracks and EOV-MMP's R0w tracks recall (studies/rpt/g1/metrics.py), each
built three times over the relation's span (GT boxes; SAM 3's and EOV-MMP's recalled pairs, cut to
the span) and run through gen_feats_test + modelC with session 3's run_pair (studies/rpt/relation.py),
`tgt_split all`. The hits are counted on the laptop. The test-label guard is installed first.
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
RPT = os.path.dirname(HERE)
STUDIES = os.path.dirname(RPT)
REPO = os.path.dirname(STUDIES)
for p in (HERE, RPT, STUDIES, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import no_test_labels  # noqa: E402
no_test_labels.install()

CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"
AFLINK = "output/ckpt/AFLink_epoch20.pth"
SEED = 3407
PINS = {
    CKPT: "d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70",
    AFLINK: "b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4",
    "data/vidvrd/data/VidVRD_ECC_train.json": "202e7ffed90667c4250bf8a0490f20b22b4c7ec620f0cdabb58e0354795d4646",
    "data/vidvrd/data/train_object_trajectories_gt.json": "b66ddb51e9550538bb1b3716c7ea9089a823f3654d4d489f8871065a4f852856",
    "data/vidvrd/data/openvoc_obj_class_spilt_info.json": "9ac86fc36f2c7679f76bc19039057b40263796ea9e8536b6294bfb157b186dec",
    "pilot_analysis/stage2/data/held_gt.json": "78d55c69f9aa2473f4584ae998592dcb67d95ce02a67ab0cd9a8bd8446184dee",
    "studies/rpt/g1/held_gt_missing3.json": "64c0233f9619d023917247078303be062ddddd6156adf39abadad107beb54afd",
    "pilot_analysis/stage1/train_split.json": "095dd89ced7260945e76f020a68dcee5793bdd5bca3a6f375f7537dc7c267f1a",
}
NO_ECC = ["ILSVRC2015_train_00265063", "ILSVRC2015_train_00679000", "ILSVRC2015_val_00123004"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def held_relations():
    gt = json.load(open(os.path.join(REPO, "pilot_analysis", "stage2", "data", "held_gt.json")))
    gt.update(json.load(open(os.path.join(HERE, "held_gt_missing3.json"))))
    return gt


def to_repo(tracks: list) -> list:
    """metrics' two-track lists -> the repo's track format (gen_feats_test)."""
    return [dict(tid=t["tid"], category=t["category"], score=float(t["score"]), begin_fid=t["begin_fid"],
                 end_fid=t["end_fid"], trajectory={t["begin_fid"] + i: list(b) for i, b in enumerate(t["boxes"])})
            for t in tracks]


def sensitivity(video, rels, sam3_tr, eov_tr, rec_s, rec_e, data, m, frame_dir):
    import metrics as M
    from relation import FrameFeatures, relation_head
    gp_all, pp_all = data["global_proj"][0], data["patch_proj"][0]
    sel = M.sensitivity_relations(rels, rec_s, rec_e)
    out = []
    for s in sel:
        rel = rels[s["k"]]
        inp = M.sensitivity_inputs(rel, s, sam3_tr, eov_tr)
        row = dict(k=s["k"], triplet=rel["triplet"], duration=rel["duration"], sam3_pair=list(s["sam3"]),
                   eov_pair=list(s["eov"]), sources={})
        for src in ("gt", "sam3", "eov"):
            if inp[src] is None:
                row["sources"][src] = None
                continue
            ff = FrameFeatures(video, frame_dir, encoder=_batch_encoder(frame_dir, video, gp_all, pp_all))
            pairs = relation_head(video, to_repo(inp[src]), "train", ff, m)
            p = [q for q in pairs if q["sub"] == 0 and q["obj"] == 1]
            row["sources"][src] = None if not p else dict(
                clips=len(p[0]["logits"]), duration=p[0]["duration"],
                logits=[[float(x) for x in c] for c in p[0]["logits"]],
                int_logit=[float(x) for x in p[0]["int_logit"]], mid_frames=p[0]["mid_frames"])
        out.append(row)
    return out


def _batch_encoder(frame_dir, video, gp_all, pp_all):
    """FrameFeatures' encoder answered from the batch's own per-frame features: Dataset_new
    encodes every frame with the same calls as relation.clip_encoder (dataset.py:104-114)."""
    files = sorted(os.listdir(os.path.join(frame_dir, video)))
    index = {os.path.join(frame_dir, video, f): i for i, f in enumerate(files)}

    def encode(path):
        f = index[path]
        return gp_all[f:f + 1], pp_all[f:f + 1]
    return encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True, help="the folder of G1's SAM 3 files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*", default=None, help="a subset (the preflight); default all 160")
    ap.add_argument("--tag", default="", help="this process's manifest name, when several GPUs share --out (A29a)")
    a = ap.parse_args()
    a.sam3, a.out = os.path.abspath(os.path.expanduser(a.sam3)), os.path.abspath(os.path.expanduser(a.out))
    os.chdir(REPO)
    bad = [p for p, h in PINS.items() if sha256(p) != h]
    if bad:
        raise SystemExit(f"!! STOP: inputs differ from their pins: {bad}")
    d2_pins = json.load(open(os.path.join(STUDIES, "d2", "logs", "code_pins.json")))["front_end_eov"]
    bad = [f for f, h in d2_pins.items() if sha256(f) != h]
    if bad:
        raise SystemExit(f"!! STOP: EOV-MMP front-end files differ from A25a's pins: {bad}")
    held = sorted(json.load(open("pilot_analysis/stage1/train_split.json"))["held"])
    assert len(held) == 160
    videos = held if a.videos is None else a.videos
    assert set(videos) <= set(held), set(videos) - set(held)
    rels_all = held_relations()
    assert sorted(rels_all) == held
    print("pins verified: checkpoint, AFLink, train ECC, train GT trajectories, obj split, HELD GT, "
          "EOV-MMP front end", flush=True)

    import torch
    import adapter as A
    import metrics as M
    from a26.tracks import serialize_tracks
    from a26.wrappers import whole_video
    from cache_wrapper import build_modelC
    from cli.common import build_model_stack, load_end2end, seed_everything
    from data_loading.dataset import Dataset_new
    from utils import paths
    from utils.parser_func import parse_args

    os.makedirs(a.out, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""
    man_path = os.path.join(a.out, f"manifest{suffix}.jsonl")
    done = set()
    for name in sorted(os.listdir(a.out)):
        if name.startswith("manifest") and name.endswith(".jsonl"):
            for line in open(os.path.join(a.out, name)):
                r = json.loads(line)
                p = os.path.join(a.out, f"{r['video']}.json.gz")
                if os.path.exists(p) and sha256(p) == r["sha256"]:
                    done.add(r["video"])
    todo = [v for v in videos if v not in done]

    seed_everything(SEED)
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", os.path.join(REPO, CKPT),
                                "--path_AFLink", os.path.join(REPO, AFLINK), "--frame_stride", "1"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    class _Log:
        def info(self, *x): print(*x, flush=True)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, _Log())
    model.eval()
    model.modelC.tgt_split = "all"
    ecc = json.load(open("data/vidvrd/data/VidVRD_ECC_train.json"))
    assert sorted(v for v in held if v not in ecc) == NO_ECC
    model.test_ecc = ecc                              # A29 item 1: the train ECC for HELD
    m, _, load_info = build_modelC(os.path.join(REPO, CKPT))

    ds = Dataset_new(args, "train")
    ds.path_list = todo
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    t0 = time.time()
    for i, data in enumerate(loader, 1):
        vid = data["video_name"][0]
        tv = time.time()
        rec = dict(video=vid, video_len=int(data["video_len"][0]), ecc=vid in ecc, configs={})
        for name, batch in (("R0w", whole_video(data)), ("R0", data)):
            model.traj.pop(vid, None)
            tc = time.time()
            with torch.no_grad():
                model(batch, sort_model)
            rec["configs"][name] = dict(begin_fid=int(batch["begin_fid"][0]), end_fid=int(batch["end_fid"][0]),
                                        tracks=serialize_tracks(model.traj.get(vid, [])),
                                        seconds=round(time.time() - tc, 2))
            model.traj.pop(vid, None)
        s3 = A.tracks_from_sam3(A.load_sam3(os.path.join(a.sam3, f"{vid}.json.gz")))
        sam3_tr = M.tracks_from_list(s3["repo"])
        eov_tr = M.tracks_from_list(rec["configs"]["R0w"]["tracks"])
        rels = rels_all[vid]
        rec_s = dict(pairs=[M.bucket(sam3_tr, r) for r in rels])
        rec_e = dict(pairs=[M.bucket(eov_tr, r) for r in rels])
        ts = time.time()
        rec["sensitivity"] = sensitivity(vid, rels, sam3_tr, eov_tr, rec_s, rec_e, data, m, paths.FRAME_DIR)
        rec["sensitivity_seconds"] = round(time.time() - ts, 2)
        rec["sam3_file_sha256"] = sha256(os.path.join(a.sam3, f"{vid}.json.gz"))
        path = os.path.join(a.out, f"{vid}.json.gz")
        with gzip.open(path + ".part", "wt") as f:
            json.dump(rec, f)
        os.replace(path + ".part", path)
        row = dict(video=vid, frames=rec["video_len"], ecc=rec["ecc"],
                   tracks_R0w=len(rec["configs"]["R0w"]["tracks"]), tracks_R0=len(rec["configs"]["R0"]["tracks"]),
                   sensitivity_relations=len(rec["sensitivity"]), seconds=round(time.time() - tv, 1),
                   sha256=sha256(path))
        with open(man_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        el = time.time() - t0
        print(f"[EOV {i}/{len(todo)}] {json.dumps(row)} | elapsed {el / 60:.1f} min, "
              f"ETA {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
        del data
        torch.cuda.empty_cache()
    json.dump(dict(modelC_load=load_info, torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   omp_num_threads=os.environ.get("OMP_NUM_THREADS"), videos=todo),
              open(os.path.join(a.out, f"run_info{suffix}.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
