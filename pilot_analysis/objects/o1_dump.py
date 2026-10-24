#!/usr/bin/env python3
"""Object check O1 (A22), step 2 -- raw detector pass on TRAINING videos (GPU box).

    python pilot_analysis/objects/o1_dump.py --out DIR --select-small [--check-repo]   # the 5-video run
    python pilot_analysis/objects/o1_dump.py --out DIR --all                           # all 800 (after approval)

Per video: frames decoded on the box (tools/extract_frames.py's own function, as session 3's
Task 3) -> the repo's Dataset_new('train') item (frame encoding, processed range [begin_fid,
end_fid) from the trajectory extents, exactly as at test time) -> a verbatim copy of
End2End_Model's test-branch front end (models/end2end_model.py:196-317, without modelC), run at
three per-frame cut-offs (OVPostProcess.pruning_threshold = 0.35, 0.2, 0.1; the repo's own
parameter) -> frames deleted.

Changes to the copied code, none of which changes a result:
- recording: each detection's query id, each track line's matched detection;
- the line -> detection association loop (e2e:291-301) visits only detections of the same frame,
  in the same order (only those can satisfy `track[0] == r['frame_id']`), and computes the IoU on
  a CPU copy of the same float32 box;
- modelB similarities are memoised per (frame, query): identical input box -> identical RoI mean.
  At 0.35, modelB is called box by box exactly as the repo does; boxes first needed at 0.2 / 0.1
  are scored in one batched call per frame.
- ECC: the training matrices (VidVRD_ECC_train.json), since the test file has no training video.
`--check-repo` runs the repo's own End2End_Model.forward on the same item (test_ecc swapped for
the training matrices) and requires its trajectories to equal this copy's 0.35 tracks.

Saved per video (`<video>.pt`): every query's box (xyxy, image pixels) and sigmoid class
probabilities over the 35 categories, for every processed frame; the detector's text features
(`clip_query`); per cut-off, the detections (query, clipped box, fused 35-score vector) and the
final tracks (class, score, dense boxes, matched detection per line); the query embeddings
(`pred_embed`, 768-d) of every query that became a detection at any cut-off and of the
best-matching queries of every annotated GT box (A22).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, json, os, shutil, time, traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage1c"))
import o1_common as O          # noqa: E402

CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"


def build():
    import torch
    from cli.common import build_model_stack, load_end2end, seed_everything
    from utils.parser_func import parse_args
    seed_everything(3407)
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", os.path.join(ROOT, CKPT),
                                "--path_AFLink", os.path.join(ROOT, "output/ckpt/AFLink_epoch20.pth"),
                                "--frame_stride", "1"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    class _Log:
        def info(self, *x): print(*x, flush=True)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, _Log())
    model.eval()
    assert len(model.train_ecc) == 783, f"training ECC matrices: {len(model.train_ecc)} videos, expected 783"
    return args, model, sort_model


def select_small():
    """5 training videos with >= 1 novel GT run, spread over the sorted list (annotations only)."""
    runs = json.load(open(O.GT_RUNS))
    vids = sorted(v for v, rs in runs.items() if any(r["category"] in O.NOVEL for r in rs))
    n = len(vids)
    return [vids[i] for i in (0, n // 4, n // 2, 3 * n // 4, n - 1)]


# ---------------------------------------------------------------------------
# the copied front end
# ---------------------------------------------------------------------------
def roi_feat(box, pp, w, h):
    """e2e:224-249 verbatim: clip the box, mask the 24x24 grid, masked mean of patch_proj."""
    import torch
    from models.gen_labels import create_mask
    new_box = [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
    if box[0] < 0:
        new_box[0] = 0
    if box[1] < 0:
        new_box[1] = 0
    if box[2] > w:
        new_box[2] = w
    if box[3] > h:
        new_box[3] = h
    box = torch.tensor(new_box).cuda()
    x1 = box[0] / w * 24
    x2 = box[2] / w * 24
    y1 = box[1] / h * 24
    y2 = box[3] / h * 24
    bbox_resize = [x1, y1, x2, y2]
    mask = create_mask(24, 24, bbox_resize)
    mask = np.array(mask)
    mask = mask.flatten()
    mask_tensor = torch.from_numpy(mask).float().cuda()
    mask_tensor = mask_tensor.unsqueeze(0)
    xx2 = pp
    mask_tensor_ = mask_tensor.unsqueeze(-1)
    mask_tensor_ = mask_tensor_.expand_as(xx2)
    weights_sum = mask_tensor_.sum(dim=1, keepdim=True).clamp(min=1e-9)
    xx3_weighted = xx2 * mask_tensor_
    xx4 = xx3_weighted.sum(dim=1) / weights_sum
    xx4 /= xx4.norm(dim=-1, keepdim=True)
    xx4 = xx4.squeeze(0)
    return box, xx4


NEW_ORDER = [0, 33, 20, 32, 1, 2, 3, 4, 30, 5, 6, 7, 28, 21, 27, 8, 25, 9, 34, 10, 31, 24, 11, 29, 12, 23, 13, 22, 14, 15, 16, 17, 26, 18, 19]


def fuse(scores, scores_classifer):
    """e2e:251-260 verbatim."""
    import torch
    similarity = torch.index_select(scores_classifer, 0, torch.tensor(NEW_ORDER).cuda())
    similarity = similarity.tolist()
    for nb in [1, 3, 8, 12, 14, 23]:
        scores[nb] = similarity[nb] * 0.6 + scores[nb] * 0.4
    for nb in [0, 2, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34]:
        scores[nb] = similarity[nb] * 0.3 + scores[nb] * 0.7
    return torch.tensor(scores)


def entry_qids(set_index, topk_boxes):
    import torch
    si = set_index[0]
    si = torch.stack([torch.as_tensor(x).reshape(()) for x in si]) if isinstance(si, list) else si
    return si.reshape(-1)[topk_boxes[0].reshape(-1)].tolist()


def detections(model, img_id, results, qids, pp, w, h, memo, batch):
    """e2e:209-262 verbatim, per detected box, with the query id recorded. `batch`: boxes whose
    modelB similarity is not memoised are scored in one call (0.2 / 0.1 only)."""
    import torch
    from models.gen_labels import filter
    detected_boxes = filter(results["boxes"])
    rb, rl, rs = results["boxes"].cpu(), results["labels"].cpu(), results["scores"].cpu()   # exact copies
    rows = []
    for box in detected_boxes:
        scores = [0] * 35
        qid = None
        box_c = box.cpu()
        for b in range(len(rb)):
            if torch.equal(box_c, rb[b]):
                label = rl[b]
                score = rs[b].item()
                scores[label] = score
                if qid is None:
                    qid = qids[b]
        rows.append((box, scores, qid))
    if batch:
        need = [(qid, box) for box, _, qid in rows if (img_id, qid) not in memo]
        if need:
            feats = [roi_feat(box, pp, w, h) for _, box in need]
            sims = model.modelB(torch.cat([x for _, x in feats], 0)).detach()
            for (qid, _), (cb, x), s in zip(need, feats, sims):
                memo[(img_id, qid)] = (cb, x, s)
    out = []
    for box, scores, qid in rows:
        if (img_id, qid) not in memo:
            cb, x = roi_feat(box, pp, w, h)
            memo[(img_id, qid)] = (cb, x, model.modelB(x)[0].detach())
        cb, x, s = memo[(img_id, qid)]
        out.append(dict(frame_id=img_id, box=cb, patch_proj=x, score=fuse(scores, s), qid=qid))
    return out


def track(args, model, sort_model, input, object_instances, ecc, begin_fid, end_fid):
    """e2e:263-317 verbatim (without modelC). -> [final trajectory dicts + 'hits']."""
    import torch
    from models.gen_labels import compute_iou, format_trajectories_test
    from models.tracking.deep_sort_app import run
    from models.tracking.aflink.AppFreeLink import AFLink, LinkData
    if not object_instances:
        return [], dict(no_detections=True)
    associate_result = run(args, input["video_name"], object_instances, min_confidence=args.min_confidence,
                           nms_max_overlap=args.nms_max_overlap, min_detection_height=args.min_detection_height,
                           max_cosine_distance=args.max_cosine_distance, nn_budget=args.nn_budget, display=False, ecc=ecc)
    if len(associate_result) == 0:
        return [], dict(no_tracks=True)
    linker = AFLink(results=associate_result, model=sort_model, dataset=LinkData("", ""), thrT=(0, 30), thrS=75, thrP=0.05)
    associate_result = linker.link()
    by_frame = collections.defaultdict(list)
    for ri, r in enumerate(object_instances):
        by_frame[r["frame_id"]].append((ri, r["box"].cpu()))
    temp, hits = [], []
    for tr in associate_result:
        expanded_array = np.zeros(45)
        expanded_array[:10] = tr
        tr = expanded_array
        tr[4] = tr[2] + tr[4]
        tr[5] = tr[3] + tr[5]
        box_t = [tr[2], tr[3], tr[4], tr[5]]
        max_iou = 0.5
        hit = -1
        assert tr[0] == int(tr[0])
        for ri, box_r in by_frame.get(int(tr[0]), []):
            iou = compute_iou(box_t, box_r)
            if iou > max_iou:
                max_iou = iou
                tr[6] = 1
                hit = ri
                for i in range(35):
                    tr[i + 10] = object_instances[ri]["score"][i]
        temp.append(tr)
        hits.append(hit)
    min_l = end_fid - begin_fid
    whole = format_trajectories_test(temp)
    groups = collections.OrderedDict()
    for tr, hit in zip(temp, hits):
        groups.setdefault(tr[1], []).append((tr, hit))
    per = []
    for tid_, lines in groups.items():
        f = format_trajectories_test([tr for tr, _ in lines])
        if f:
            t = f[0]
            t["hits"] = [(int(tr[0]), hit) for tr, hit in lines if max(tr[10:]) > 0]
            per.append(t)
    key = lambda t: (t["category"], float(t["score"]), t["begin_fid"], t["end_fid"], sorted(t["trajectory"].items()))
    assert [key(t) for t in per] == [key(t) for t in whole], "per-track formatting differs from the whole-video call"
    out, tid = [], 0
    for t in per:
        if len(t["trajectory"]) > 0.15 * min_l:
            t["tid"] = tid
            tid += 1
            out.append(t)
    return out, {}


# ---------------------------------------------------------------------------
def dump_video(args, model, sort_model, input, video, check_repo):
    import torch
    from models.util import box_ops
    t = collections.Counter()
    begin_fid, end_fid = int(input["begin_fid"][0]), int(input["end_fid"][0])
    patch_ = input["patch_"].squeeze()
    patch_proj = input["patch_proj"].squeeze()
    w, h = input["image_size"][0].cuda(), input["image_size"][1].cuda()      # as e2e:220-221
    Wi, Hi = int(input["image_size"][0]), int(input["image_size"][1])
    post = model.postprocessorA["bbox"]
    ecc = model.train_ecc
    raw, _ = O.annotated_boxes(video)
    gt_by_frame = collections.defaultdict(list)
    for tid, fb in raw.items():
        for f, b in fb.items():
            gt_by_frame[f].append(b)
    F = end_fid - begin_fid
    boxes_all = np.zeros((F, 300, 4), np.float32)
    prob_all = np.zeros((F, 300, 35), np.float16)
    inst = {thr: [] for thr in O.THRESHOLDS}
    kept = {thr: [] for thr in O.THRESHOLDS}
    emb_keep = {}
    clip_query = None
    memo = {}
    checks = collections.Counter()
    for k, img_id in enumerate(range(begin_fid, end_fid)):
        t0 = time.time()
        p_ = patch_[img_id:img_id + 1, :, :].cuda()
        pp = patch_proj[img_id:img_id + 1, :, :].cuda()
        with torch.no_grad():
            output, seleted_id = model.modelA(p_)
        orig_target_sizes = torch.tensor([[input["image_size"][1], input["image_size"][0]]]).cuda()
        logits = output["pred_logits"][0]
        assert logits.shape == (300, 35), logits.shape
        prob = logits.sigmoid()
        bx = box_ops.box_cxcywh_to_xyxy(output["pred_boxes"][0]) * torch.tensor([Wi, Hi, Wi, Hi], dtype=torch.float32, device="cuda")
        boxes_all[k] = bx.float().cpu().numpy()
        prob_all[k] = prob.half().cpu().numpy()
        if clip_query is None:
            clip_query = output["clip_query"].float().cpu()
        emb = output["pred_embed"][0]
        sel = set()
        t["detector"] += time.time() - t0
        for thr in O.THRESHOLDS:
            t1 = time.time()
            post.pruning_threshold = thr
            with torch.no_grad():
                results, set_index, topk_boxes = post(output, orig_target_sizes, seleted_id, False)
            results = results[0]
            qids = entry_qids(set_index, topk_boxes)
            for b, q in enumerate(qids):                     # the recorded query really is the entry's
                checks["entry"] += 1
                checks["entry_box_ok"] += bool(torch.allclose(results["boxes"][b], bx[q], atol=1e-3))
                checks["entry_prob_ok"] += bool(abs(results["scores"][b].item() - prob[q, results["labels"][b]].item()) < 1e-6)
            kept[thr].append(sorted(set(qids)))
            with torch.no_grad():
                new = detections(model, img_id, results, qids, pp, w, h, memo, batch=thr != 0.35)
            inst[thr].extend(new)
            sel |= {d["qid"] for d in new}
            t[f"post_modelB_{thr}"] += time.time() - t1
        post.pruning_threshold = 0.35
        for g in gt_by_frame.get(img_id, []):
            iou = O.frame_iou(g, boxes_all[k])
            loc = np.where(iou >= O.LOC_IOU)[0]
            if len(loc):
                sel.add(int(loc[np.argmax(iou[loc])]))
                pmax = prob_all[k][loc].astype(np.float32).max(1)
                sel.add(int(loc[np.argmax(pmax)]))
        for q in sel:
            emb_keep[(img_id, q)] = emb[q].half().cpu()
    tracks, flags = {}, {}
    for thr in O.THRESHOLDS:
        t2 = time.time()
        with torch.no_grad():
            tracks[thr], flags[thr] = track(args, model, sort_model, input, inst[thr], ecc, begin_fid, end_fid)
        t[f"track_{thr}"] += time.time() - t2
    repo = None
    if check_repo:
        t3 = time.time()
        saved = model.test_ecc
        model.test_ecc = model.train_ecc
        try:
            with torch.no_grad():
                model(input, sort_model)
        finally:
            model.test_ecc = saved
        theirs = model.traj.pop(video, [])
        mine = tracks[0.35]
        same = len(theirs) == len(mine) and all(
            a["category"] == b["category"] and a["begin_fid"] == b["begin_fid"] and a["end_fid"] == b["end_fid"]
            and a["trajectory"] == b["trajectory"] and abs(float(a["score"]) - float(b["score"])) <= 1e-6
            for a, b in zip(theirs, mine))
        repo = dict(identical=bool(same), repo_tracks=len(theirs), copy_tracks=len(mine))
        t["repo_forward"] += time.time() - t3
    pack = lambda L: dict(frame=np.array([d["frame_id"] for d in L], np.int32), qid=np.array([d["qid"] for d in L], np.int16),
                          box=np.array([d["box"].cpu().numpy() for d in L], np.float32).reshape(-1, 4),
                          score=np.array([d["score"].numpy() for d in L], np.float32).reshape(-1, 35))
    keys = sorted(emb_keep)
    rec = dict(video=video, size=(Wi, Hi), begin_fid=begin_fid, end_fid=end_fid, n_frames=int(patch_.shape[0]),
               boxes=boxes_all, prob=prob_all, clip_query=clip_query.half(),
               kept={thr: kept[thr] for thr in O.THRESHOLDS},
               det={thr: pack(inst[thr]) for thr in O.THRESHOLDS},
               tracks={thr: [dict(tid=x["tid"], category=x["category"], score=float(x["score"]), begin=int(x["begin_fid"]),
                                  end=int(x["end_fid"]), boxes=[x["trajectory"][f] for f in range(x["begin_fid"], x["end_fid"])],
                                  hits=x["hits"]) for x in tracks[thr]] for thr in O.THRESHOLDS},
               track_flags=flags, emb_key=np.array(keys, np.int32).reshape(-1, 2),
               emb=torch.stack([emb_keep[k_] for k_ in keys]) if keys else torch.zeros(0, 768).half(),
               checks=dict(checks), repo_check=repo, seconds={k_: round(v_, 2) for k_, v_ in t.items()})
    assert checks["entry_box_ok"] == checks["entry"] and checks["entry_prob_ok"] == checks["entry"], dict(checks)
    for thr in O.THRESHOLDS:                                 # dense boxes: every track covers [begin, end) exactly
        for x in tracks[thr]:
            assert sorted(x["trajectory"]) == list(range(x["begin_fid"], x["end_fid"]))
    return rec


def verify_box():
    """On the box: the preregistration file and the pins this step uses (A22's scripts, A5's runs).
    Every other pin is verified locally before step 3 (git-ignored data files are not on the box)."""
    import re
    pre = os.path.join(ROOT, "pilot_analysis", "stage1b", "preregistration.md")
    want = open(os.path.join(ROOT, "pilot_analysis", "stage1b", "preregistration.sha256")).read().split()[0]
    bad = [] if O.sha256(pre) == want else ["preregistration.md"]
    pins = {}
    for line in open(pre, encoding="utf-8"):
        m = re.match(r"\|[^|]*\|\s*`([^`]+)`[^|]*\|\s*`([0-9a-f]{64})`\s*\|", line)
        if m:
            pins[m.group(1)] = m.group(2)
    use = [p for p in pins if p.startswith("pilot_analysis/objects/") or p == "pilot_analysis/stage1c/gt_tracks_train.json"]
    for p in use:
        if O.sha256(os.path.join(ROOT, p)) != pins[p]:
            bad.append(p)
    if bad or not any(p.startswith("pilot_analysis/objects/") for p in use):
        sys.exit(f"!! pinned-hash check failed: {bad or 'no A22 pins found'}")
    print(f"  preregistration.md + {len(use)} pins verified: {use}")


def estimate(out, rates=(0.40, 0.60)):
    """Full-run runtime, cost and disk from the small run: decode + encode scale with a video's
    frames, everything else with its processed frames (the repo check is not part of the full run)."""
    from session3_tasks import frame_count
    man = [json.loads(l) for l in open(os.path.join(out, "manifest.jsonl"))]
    frag = json.load(open(O.FRAG))
    allv = sorted(frag)
    tot = {v: frame_count(v) for v in allv}
    proc = {v: max(t["end_fid"] for t in frag[v]) - min(t["begin_fid"] for t in frag[v]) for v in allv}
    per_frame = sum(m["parts"].get("decode", 0) + m["parts"].get("encode", 0) for m in man) / sum(m["frames"] for m in man)
    rest = lambda m: m["seconds"] - m["parts"].get("decode", 0) - m["parts"].get("encode", 0) - m["parts"].get("repo_forward", 0)
    per_proc = sum(rest(m) for m in man) / sum(m["processed"] for m in man)
    bytes_proc = sum(m["bytes"] for m in man) / sum(m["processed"] for m in man)
    T = sum(tot.values()) * per_frame + sum(proc.values()) * per_proc
    res = dict(videos_small=len(man), small_seconds=sum(m["seconds"] for m in man), s_per_frame_decode_encode=per_frame,
               s_per_processed_frame_rest=per_proc, total_frames=sum(tot.values()), processed_frames=sum(proc.values()),
               full_hours=T / 3600, full_disk_gb=bytes_proc * sum(proc.values()) / 1e9,
               cost_usd={f"{r:.2f}/h": round(T / 3600 * r, 2) for r in rates},
               repo_check=[m["repo_check"] for m in man],
               parts_share={k: sum(m["parts"].get(k, 0) for m in man) / sum(m["seconds"] for m in man)
                            for k in sorted({k for m in man for k in m["parts"]})})
    json.dump(res, open(os.path.join(out, "estimate.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select-small", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--videos", nargs="+")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--estimate", action="store_true")
    ap.add_argument("--check-repo", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return verify_box()
    if a.estimate:
        return estimate(a.out)
    assert a.out, "--out is required"
    import torch
    from data_loading.dataset import Dataset_new
    from session3_tasks import frame_count
    import tools.extract_frames as X
    from utils import paths
    os.makedirs(a.out, exist_ok=True)
    vids = select_small() if a.select_small else (sorted(json.load(open(O.GT_RUNS))) if a.all else a.videos)
    json.dump(vids, open(os.path.join(a.out, "videos.json"), "w"))
    man_path = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line)
            f = os.path.join(a.out, f"{r['video']}.pt")
            if os.path.exists(f) and O.sha256(f) == r["sha256"]:
                done.add(r["video"])
    todo = [v for v in vids if v not in done]
    args, model, sort_model = build()
    ds = Dataset_new(args, "train")
    X.FRAME_DIR = paths.FRAME_DIR
    t_all = time.time()
    failures = 0
    for i, v in enumerate(todo, 1):
        tv = time.time()
        try:
            n = frame_count(v)
            d = os.path.join(paths.FRAME_DIR, v)
            if not (os.path.isdir(d) and len(os.listdir(d)) == n):
                got, status = X.extract_one(v, n, quality=95)
                if status not in ("ok", "skip"):
                    raise RuntimeError(f"decode {v}: {status}")
            t_dec = time.time() - tv
            ds.path_list = [v]
            loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
            te = time.time()
            input = next(iter(loader))
            t_enc = time.time() - te
            rec = dump_video(args, model, sort_model, input, v, a.check_repo)
            rec["seconds"].update(decode=round(t_dec, 2), encode=round(t_enc, 2))
            out = os.path.join(a.out, f"{v}.pt")
            torch.save(rec, out)
            m = dict(video=v, seconds=round(time.time() - tv, 1), frames=rec["n_frames"], processed=rec["end_fid"] - rec["begin_fid"],
                     dets={str(k): int(len(x["qid"])) for k, x in rec["det"].items()},
                     tracks={str(k): len(x) for k, x in rec["tracks"].items()}, emb=int(len(rec["emb"])),
                     parts=rec["seconds"], repo_check=rec["repo_check"], bytes=os.path.getsize(out), sha256=O.sha256(out))
            with open(man_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            el = time.time() - t_all
            print(f"[O1 {i}/{len(todo)}] {v} frames={m['frames']} processed={m['processed']} dets={m['dets']} "
                  f"tracks={m['tracks']} {m['seconds']}s repo={m['repo_check']} | ETA {el / i * (len(todo) - i) / 3600:.1f} h", flush=True)
        except Exception:
            failures += 1
            with open(os.path.join(a.out, "failures.jsonl"), "a") as f:
                f.write(json.dumps(dict(video=v, error=traceback.format_exc())) + "\n")
            print(f"!! {v} failed:\n{traceback.format_exc()}", flush=True)
        finally:
            if not a.keep_frames:
                shutil.rmtree(os.path.join(paths.FRAME_DIR, v), ignore_errors=True)
            model.traj.pop(v, None)
            torch.cuda.empty_cache()
    if failures:
        sys.exit(4)


if __name__ == "__main__":
    main()
