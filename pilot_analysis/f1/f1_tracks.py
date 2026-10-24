#!/usr/bin/env python3
"""Feasibility batch F1, Part B1 (A13) -- how fragmented are EOV-MMP's detected test tracks?

    python pilot_analysis/f1/f1_tracks.py --small     # 5 test videos
    python pilot_analysis/f1/f1_tracks.py             # all 200

LABEL-FREE. Reads only EOV-MMP's own outputs on the test videos (session 3 Task 6's
cache_s3_sgdet: detected tracks and per-slot logits), never a test annotation, and
runs no evaluation. It only decides whether a track-stitching component is built.

Stitch candidate: an ordered pair of detected tracks (A, B) of one video with the
same predicted category, B starting 1-30 frames after A's last frame, and B's first
box close to A's last box (IoU >= 0.3, or centre distance <= 0.5 x A's last box
diagonal). Relation predictions: Task 6's own scoring path (eval_from_cache.predict,
"all", top 200 per video), re-derived per video; each is matched to its subject and
object tracks by category and box at its first frame.
Gate B1: >= 10% of tracks are in at least one candidate.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, json, math, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f1_prior import C, write_log      # noqa: E402

MAX_GAP, IOU_MIN, DIST_MAX, G_TRACKS = 30, 0.3, 0.5, 0.10


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def close(a_last, b_first):
    ca = ((a_last[0] + a_last[2]) / 2, (a_last[1] + a_last[3]) / 2)
    cb = ((b_first[0] + b_first[2]) / 2, (b_first[1] + b_first[3]) / 2)
    diag = math.hypot(a_last[2] - a_last[0], a_last[3] - a_last[1])
    return iou(a_last, b_first) >= IOU_MIN or math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= DIST_MAX * diag


def candidates(tracks):
    """tracks: {tid: dict(category, begin_fid, end_fid, boxes)} -> [(A, B, gap)]."""
    out = []
    for ta, A in tracks.items():
        a_last = A["end_fid"] - 1
        for tb, B in tracks.items():
            if ta == tb or A["category"] != B["category"]:
                continue
            gap = B["begin_fid"] - a_last
            if 1 <= gap <= MAX_GAP and close(A["boxes"][-1], B["boxes"][0]):
                out.append((ta, tb, gap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    import torch
    t0 = time.time()
    C.verify_frozen()
    sys.path.insert(0, os.path.join(C.PA, "stage1c"))
    import eval_from_cache as E4
    out_dir = os.path.join(HERE, "data_small" if args.small else "data")
    os.makedirs(out_dir, exist_ok=True)
    sdir = os.path.join(C.BOX_COPY, "cache_s3_sgdet", "test")
    vids = sorted(f[:-3] for f in os.listdir(sdir) if f.endswith(".pt"))[:5 if args.small else None]
    sums = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    lengths, n_cand, involved_n, n_tracks, gaps, nondense = [], 0, 0, 0, [], 0
    n_pred = n_pred_inv = n_unmatched = 0
    per_video = {}
    for v in vids:
        rel = f"cache_s3_sgdet/test/{v}.pt"
        f = os.path.join(C.BOX_COPY, rel)
        assert C.sha256(f) == sums[rel], rel
        dct = torch.load(f, weights_only=False)
        tracks = dct["tracks"]
        for t in tracks.values():
            n = t["end_fid"] - t["begin_fid"]
            nondense += len(t["boxes"]) != n
            lengths.append(n)
        cand = candidates(tracks)
        inv = {a for a, _, _ in cand} | {b for _, b, _ in cand}
        n_tracks += len(tracks); n_cand += len(cand); involved_n += len(inv)
        gaps += [g for _, _, g in cand]
        # relation predictions (Task 6's path), matched to tracks by category and first-frame box
        index = collections.defaultdict(list)
        for tid, t in tracks.items():
            for i, b in enumerate(t["boxes"]):
                index[(t["category"], t["begin_fid"] + i)].append((tid, b))

        def track_of(cat, frame, box):
            hits = [tid for tid, b in index.get((cat, frame), []) if max(abs(x - y) for x, y in zip(b, box)) < 1e-6]
            return hits
        preds = E4.predict({v: dct["pairs"]}, "all")[v]
        pv = pi = 0
        for r in preds:
            s_hits = track_of(r["triplet"][0], r["duration"][0], r["sub_traj"][0])
            o_hits = track_of(r["triplet"][2], r["duration"][0], r["obj_traj"][0])
            if not s_hits or not o_hits:
                n_unmatched += 1
            pv += 1
            pi += bool(set(s_hits) & inv or set(o_hits) & inv)
        n_pred += pv; n_pred_inv += pi
        per_video[v] = dict(tracks=len(tracks), candidates=len(cand), involved=len(inv), predictions=pv,
                            predictions_involving=pi)
        del dct
    L = np.array(lengths)
    share = involved_n / max(n_tracks, 1)
    res = dict(small=args.small, videos=len(vids), tracks=n_tracks, non_dense_tracks=nondense,
               track_length=dict(min=int(L.min()), p25=float(np.percentile(L, 25)), median=float(np.median(L)),
                                 p75=float(np.percentile(L, 75)), p95=float(np.percentile(L, 95)), max=int(L.max()),
                                 share_under_30=float((L < 30).mean()), share_under_90=float((L < 90).mean())),
               candidates=n_cand, gap_hist=dict(sorted(collections.Counter(gaps).items())),
               tracks_involved=involved_n, share_tracks_involved=share,
               predictions=n_pred, predictions_involving_candidate_track=n_pred_inv,
               share_predictions_involving=n_pred_inv / max(n_pred, 1), predictions_unmatched=n_unmatched,
               rule=dict(max_gap=MAX_GAP, iou=IOU_MIN, centre_distance=DIST_MAX, gate=G_TRACKS),
               verdict="PASS: build a track-stitching linker later" if share >= G_TRACKS else "FAIL: Idea B dropped",
               per_video=per_video, seconds=round(time.time() - t0, 1))
    path = os.path.join(out_dir, "f1_tracks_results.json")
    json.dump(res, open(path, "w"), indent=1)
    write_log("f1_tracks" + ("_small" if args.small else ""),
              dict(small=args.small, max_gap=MAX_GAP, iou=IOU_MIN, centre_distance=DIST_MAX, gate=G_TRACKS, post=C.POST),
              [C.CACHE_SUMS], [path], extra=dict(seconds=res["seconds"]))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_video", "gap_hist")}, indent=1))


if __name__ == "__main__":
    main()
