#!/usr/bin/env python3
"""Session 3, Task 6 -- the B6 cache on DETECTED trajectories (SGDet), 200 test videos.

    python pilot_analysis/stage1c/sgdet_cache.py --out CACHE_SGDET

The repo's normal test path, unchanged: Dataset_new('val') -> End2End_Model
(detector -> deep_sort -> AFLink -> format_trajectories_test -> gen_feats_test
-> modelC), built and loaded exactly as cli/evaluate.py does (seed 3407,
--frame_stride 1, the shipped VidVRD_ECC_test.json -- NOT the stage1c supplement,
so the pilot's conditions are reproduced). modelC runs in the `all` split; the
novel split is derived from the same logits (base columns zeroed), which is what
modelC's novel branch computes.

Hooks capture every modelC call in order and are aligned with End2End_Model's
final_results; the rebuilt logits must reproduce each returned score
(fidelity residual). Also cached: the detected trajectories (categories, scores,
boxes), per-frame RoI means for them and for pair unions, and the frame
embeddings. Resumable per video.

Fidelity gate: post-process the cached logits (eval_from_cache) and score with
the official evaluator; SGDet must reproduce the pilot's 15.79 novel / 27.18 all
within 0.3 mAP, otherwise exit 6.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"
PILOT_SGDET = dict(novel=15.79, all=27.18)
GATE = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    import torch, torch.nn.functional as F
    from cli.common import build_eval_data, build_model_stack, load_end2end, seed_everything
    from utils.parser_func import parse_args
    from cache_wrapper import roi_means, sha256
    import eval_from_cache as E

    os.makedirs(os.path.join(a.out, "test"), exist_ok=True)
    seed_everything(3407)
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", os.path.join(ROOT, CKPT),
                                "--path_AFLink", os.path.join(ROOT, "output/ckpt/AFLink_epoch20.pth"),
                                "--frame_stride", "1"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    man_path = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line)
            f = os.path.join(a.out, "test", f"{r['video']}.pt")
            if os.path.exists(f) and sha256(f) == r["sha256"]:
                done.add(r["video"])

    val_dataset, _ = build_eval_data(args)
    todo = [v for v in val_dataset.path_list if v not in done]
    if a.limit:
        todo = todo[:a.limit]
    val_dataset.path_list = todo                      # skip finished videos before any frame is encoded
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    class _Log:                                       # load_end2end wants a logger
        def info(self, *x): print(*x, flush=True)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, _Log())
    model.eval()
    mc = model.modelC
    mc.tgt_split = "all"

    calls = []                                        # one dict per modelC call, in order
    def pre_hook(mod, inp):
        calls.append({"inputs": {k: v.detach().clone() for k, v in inp[0].items()}})
    hooks = [mc.register_forward_pre_hook(pre_hook),
             mc.featEmbedding.spatial_decoder.register_forward_hook(lambda m_, i, o: calls[-1].__setitem__("spatial", o)),
             mc.featEmbedding.register_forward_hook(lambda m_, i, o: calls[-1].__setitem__("fe", o)),
             mc.pre_classifier.register_forward_hook(lambda m_, i, o: calls[-1].__setitem__("text", o)),
             mc.pre_classifier.prompt_learner.meta_net.register_forward_hook(lambda m_, i, o: calls[-1].__setitem__("bias", o))]

    t0 = time.time()
    for i, data in enumerate(loader, 1):
        vid = data["video_name"][0]
        tv = time.time()
        calls.clear()
        with torch.no_grad():
            results = model(data, sort_model)
        results = results or []
        if len(results) != len(calls):
            raise SystemExit(f"!! {vid}: {len(results)} results but {len(calls)} modelC calls")
        pairs, worst = [], 0.0
        pp = data["patch_proj"][0]
        w, h = int(data["image_size"][0]), int(data["image_size"][1])
        from utils.eov_utils import gen_union_bbox
        for res, c in zip(results, calls):
            pre_embs, sbj, obj, inter = c["fe"][:4]
            with torch.no_grad():
                vpe = F.normalize(pre_embs, dim=-1)
                text, _ = mc.split_text_embeddings("all", pre_classifier_weights=c["text"].squeeze(0))
                logits = torch.matmul(vpe, text.t()) / mc.temperature
                recon = torch.sigmoid(logits) * torch.sigmoid(inter).unsqueeze(-1)
                worst = max(worst, float((recon - res["pre_preds"]).abs().max()))
            pd = res["pair_data"][0]
            b0, b1 = [int(x) for x in pd["duration"]]
            uf, union = roi_means(pp, list(range(b0, b1)),
                                  [gen_union_bbox(s, o) for s, o in zip(pd["sbj_traj"], pd["obj_traj"])], w, h)
            pairs.append(dict(
                sbj_cls=pd["sbj_cls"], obj_cls=pd["obj_cls"], sbj_scr=float(pd["sbj_scr"]), obj_scr=float(pd["obj_scr"]),
                duration=[b0, b1], sbj_traj=[[float(x) for x in b] for b in pd["sbj_traj"]],
                obj_traj=[[float(x) for x in b] for b in pd["obj_traj"]],
                logits=logits[0].float().cpu(), int_logit=inter[0].float().cpu(), vpe=vpe[0].half().cpu(),
                spatial=c["spatial"][0].half().cpu(),
                slot_inputs={k: v[0].half().cpu() for k, v in c["inputs"].items()},
                cond_mean=vpe[0].mean(0, keepdim=True).half().cpu(), bias=c["bias"].half().cpu(),
                sbj_emb=F.normalize(sbj, dim=-1).half().cpu(), obj_emb=F.normalize(obj, dim=-1).half().cpu(),
                windows=[(0, int(logits.shape[1]))], union_frames=uf, union_roi=union))
        tracks = {}
        for t in model.traj.get(vid, []):
            fs = sorted(int(f) for f in t["trajectory"])
            f_, x = roi_means(pp, fs, [t["trajectory"][f] for f in fs], w, h)
            tracks[t["tid"]] = dict(category=t["category"], score=float(t["score"]), begin_fid=int(t["begin_fid"]),
                                    end_fid=int(t["end_fid"]), frames=f_, roi=x,
                                    boxes=[[float(v) for v in t["trajectory"][f]] for f in fs])
        out = os.path.join(a.out, "test", f"{vid}.pt")
        torch.save(dict(video=vid, split="test", trajectories="detected", size=(w, h),
                        n_frames=int(pp.shape[0]), global_proj=data["global_proj"][0].half(),
                        tracks=tracks, pairs=pairs, fidelity_max_residual=worst), out)
        if worst > 1e-4:
            raise SystemExit(f"!! {vid}: fidelity residual {worst:.2e} > 1e-4")
        rec = dict(video=vid, split="test", seconds=round(time.time() - tv, 1), frames=int(pp.shape[0]),
                   tracks=len(tracks), pairs=len(pairs), slots=int(sum(len(p["logits"]) for p in pairs)),
                   fidelity_max_residual=worst, bytes=os.path.getsize(out), sha256=sha256(out))
        with open(man_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        el = time.time() - t0
        print(f"[SGDET {i}/{len(todo)}] {vid} pairs={rec['pairs']} {rec['seconds']}s residual={worst:.1e} | "
              f"elapsed {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
        model.traj.pop(vid, None)
        torch.cuda.empty_cache()
    for hk in hooks:
        hk.remove()

    # ---- fidelity gate: the cached logits must reproduce the pilot's SGDet numbers
    cache = E.load_cache(a.out)
    res = {}
    for split in ("novel", "all"):
        preds = E.predict(cache, split)
        res[split] = E.score(preds, split)
        res[split]["pilot_SGDet_mAP"] = PILOT_SGDET[split]
        res[split]["diff_vs_pilot"] = res[split]["mAP"] - PILOT_SGDET[split]
        json.dump(preds, open(os.path.join(a.out, f"sgdet_{split}.json"), "w"))
    res["videos"] = len(cache)
    json.dump(res, open(os.path.join(a.out, "task6_sgdet.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    bad = [s for s in ("novel", "all") if abs(res[s]["diff_vs_pilot"]) > GATE]
    if bad and not a.no_gate:
        print(f"!! TASK 6 GATE: {bad} differ from the pilot's SGDet by more than {GATE} -- STOP")
        sys.exit(6)


if __name__ == "__main__":
    main()
