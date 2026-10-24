#!/usr/bin/env python3
"""Stage 2 Batch B, step 1 -- part scores on the 200 test videos (A8).

    python pilot_analysis/stage2/b_step1_test_parts.py                  # test (after A8)
    python pilot_analysis/stage2/b_step1_test_parts.py --plumb-train 5  # same code on 5 training videos

Ground-truth tracks and categories (stage1c/gt_tracks_test.json, pinned in A5), the
SHIPPED VidVRD_ECC_test.json, and the test cache (option: --plumb-train uses the
training tracks, ECC and cache, and must reproduce step 1's stored values exactly).
Reads NO annotation file and NO label: frame count and size come from the cache.

Per (ordered pair, slot), in the cache's own order (checked pair for pair):
measured and supplied parts (common.slot_measures), the A2/A3 scalars, the learned
part from data/model_full.pt (3-seed mean), EOV-MMP's logits and interactiveness.
Videos without ECC: no part is computed (they fall back to EOV-MMP alone, A8).
Symmetry (swap, time reversal) is checked exactly as in step 1.

Writes data/b_test_slots.npz, b_test_emb.npy, b_test_learned_full.npy,
b_test_pairs.json, b_test_parts_summary.json (plumbing: data/b_plumb_*).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time
from collections import Counter

import numpy as np

import common as C
import learned_part as LP
from step1_parts import sym_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plumb-train", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(6)
    t0 = time.time()
    C.verify_frozen()
    data = os.path.join(C.OUT, "data")
    plumb = bool(a.plumb_train)
    split = "train" if plumb else "test"
    pre = "b_plumb" if plumb else "b_test"
    th = C.load_thresholds()
    ecc = json.load(open(C.ECC_TRAIN if plumb else C.ECC_TEST))
    tracks_all = json.load(open(C.TRACKS_TRAIN if plumb else C.TRACKS_TEST))
    if plumb:
        videos = json.load(open(os.path.join(data, "slots_meta.json")))["videos"][:a.plumb_train]
    else:
        videos = sorted(f[:-3] for f in os.listdir(os.path.join(C.CACHE, "test")) if f.endswith(".pt"))
        assert len(videos) == 200
    sums = {}
    for line in open(C.CACHE_SUMS):
        h, p = line.split(None, 1)
        sums[p.strip()] = h
    rows, embs, pairs_out, tally, counts, no_ecc = [], [], {}, Counter(), Counter(), []
    for vi, v in enumerate(videos):
        f = os.path.join(C.CACHE, split, f"{v}.pt")
        assert C.sha256(f) == sums[f"cache_s3/{split}/{v}.pt"], f
        cache = torch.load(f, weights_only=False)
        assert cache["video"] == v
        W, H = cache["size"]
        n = int(cache["n_frames"])
        frames = [dict() for _ in range(n)]
        span = {}
        for t in tracks_all[v]:
            ct = cache["tracks"][t["tid"]]
            assert ct["begin_fid"] == t["begin_fid"] and ct["end_fid"] == t["end_fid"] and ct["boxes"] == t["boxes"]
            for i, b in enumerate(t["boxes"]):
                fid = t["begin_fid"] + i
                if 0 <= fid < n:
                    frames[fid][t["tid"]] = tuple(b)
            span[t["tid"]] = (t["begin_fid"], t["end_fid"])
        grid = C.S.slot_grid(span)
        assert [(s, o) for s, o, _, _ in grid] == [(p["sub"], p["obj"]) for p in cache["pairs"]], v
        ecc_v = ecc.get(v)
        if ecc_v is None:
            no_ecc.append(v)
        pairs_out[v] = []
        for pi, ((s, o, ov, slots), cp) in enumerate(zip(grid, cache["pairs"])):
            assert list(ov) == list(cp["duration"]) and len(slots) == len(cp["logits"]), (v, pi)
            counts["pairs"] += 1
            counts["split_pairs"] += len(cp["windows"]) > 1
            counts["windows"] += len(cp["windows"])
            shared, r0 = {}, len(rows)
            for k, slot in enumerate(slots):
                w = [i for i, (wa, wb) in enumerate(cp["windows"]) if wa <= k < wb]
                assert len(w) == 1
                if ecc_v is not None:
                    pair = C.window_pairs(frames, s, o, slot, ov, ecc_v, W, H, shared)
                    meas, ex = C.slot_measures(pair, th)
                    sym_check(pair, th, tally)
                else:
                    meas, ex = {c: float("nan") for c in C.MEAS_COLS}, {c: float("nan") for c in C.EXTRA_FEATS}
                counts["slots"] += 1
                rows.append(dict(meas=[meas[c] for c in C.MEAS_COLS], extra=[ex[c] for c in C.EXTRA_FEATS],
                                 vehicle=float(cp["sbj_cls"] in C.VEHICLE), logits=cp["logits"][k].numpy(),
                                 int_logit=float(cp["int_logit"][k]), video=vi, pair=pi, slot=k,
                                 has_ecc=ecc_v is not None))
                embs.append(torch.cat([cp["vpe"][k], cp["sbj_emb"][w[0]], cp["obj_emb"][w[0]]]).numpy())
            pairs_out[v].append(dict(sbj_cls=cp["sbj_cls"], obj_cls=cp["obj_cls"], sbj_scr=cp["sbj_scr"],
                                     obj_scr=cp["obj_scr"], duration=list(cp["duration"]),
                                     sbj_traj=[list(map(float, b)) for b in cp["sbj_traj"]],
                                     obj_traj=[list(map(float, b)) for b in cp["obj_traj"]], rows=[r0, len(rows)]))
        del cache
        if (vi + 1) % 25 == 0 or vi + 1 == len(videos):
            el = time.time() - t0
            print(f"[{vi + 1}/{len(videos)}] slots {len(rows)} | {el / 60:.1f} min, "
                  f"ETA {el / (vi + 1) * (len(videos) - vi - 1) / 60:.1f} min", flush=True)
    arr = dict(meas=np.array([r["meas"] for r in rows], np.float64), extra=np.array([r["extra"] for r in rows], np.float64),
               vehicle=np.array([r["vehicle"] for r in rows], np.float64),
               logits=np.stack([r["logits"] for r in rows]).astype(np.float32),
               int_logit=np.array([r["int_logit"] for r in rows], np.float32),
               video=np.array([r["video"] for r in rows], np.int32), pair=np.array([r["pair"] for r in rows], np.int32),
               slot=np.array([r["slot"] for r in rows], np.int32), has_ecc=np.array([r["has_ecc"] for r in rows], bool))
    E = np.stack(embs).astype(np.float16)
    Lf, outs = LP.apply(os.path.join(data, "model_full.pt"), E, arr["extra"])
    assert [tuple(o) for o in outs] == list(C.OUTPUTS)
    paths = dict(slots=os.path.join(data, f"{pre}_slots.npz"), emb=os.path.join(data, f"{pre}_emb.npy"),
                 learned=os.path.join(data, f"{pre}_learned_full.npy"), pairs=os.path.join(data, f"{pre}_pairs.json"),
                 summary=os.path.join(data, f"{pre}_parts_summary.json"))
    np.savez(paths["slots"], **arr)
    np.save(paths["emb"], E)
    np.save(paths["learned"], Lf)
    json.dump(pairs_out, open(paths["pairs"], "w"))
    sym = {k[:-3]: (tally[k], tally[k[:-3] + "_n"]) for k in sorted(tally) if k.endswith("_ok")}
    ecc_rows = arr["has_ecc"]
    summary = dict(split=split, videos=len(videos), counts=dict(counts), videos_without_ecc=no_ecc,
                   slots_without_ecc=int((~ecc_rows).sum()), symmetry=sym,
                   symmetry_exact=all(x == y for x, y in sym.values()),
                   meas_nan_rate_with_ecc={c: float(np.isnan(arr["meas"][ecc_rows, i]).mean()) for i, c in enumerate(C.MEAS_COLS)},
                   shapes={k: list(v.shape) for k, v in arr.items()} | {"emb": list(E.shape), "learned": list(Lf.shape)},
                   seconds=round(time.time() - t0, 1))
    if plumb:                                     # must reproduce step 1 exactly
        d = np.load(os.path.join(data, "slots.npz"))
        meta_v = json.load(open(os.path.join(data, "slots_meta.json")))["videos"]
        ref = np.concatenate([np.where(d["video"] == meta_v.index(v))[0] for v in videos])
        same = lambda x, y: bool(np.array_equal(x, y, equal_nan=True))
        summary["reproduces_step1"] = dict(
            meas=same(arr["meas"], d["meas"][ref]), extra=same(arr["extra"], d["extra"][ref]),
            logits=same(arr["logits"], d["logits"][ref]), vehicle=same(arr["vehicle"], d["vehicle"][ref]),
            emb=same(E, np.load(os.path.join(data, "emb.npy"))[ref]),
            learned_within_1e6=bool(np.abs(Lf - np.load(os.path.join(data, "learned_full.npy"))[ref]).max() <= 1e-6))
        # (float32 matmul rounding depends on batch size: ~2e-7 between a 38-row and a 13,318-row batch)
    json.dump(summary, open(paths["summary"], "w"), indent=1)
    C.write_log(f"{pre}_step1_parts", dict(split=split, plumb=a.plumb_train, thresholds=th, windows=C.WIN, eps=C.EPS),
                [C.TRACKS_TRAIN if plumb else C.TRACKS_TEST, C.ECC_TRAIN if plumb else C.ECC_TEST, C.CACHE_SUMS,
                 os.path.join(data, "model_full.pt")], list(paths.values()))
    print(json.dumps({k: summary[k] for k in summary if k not in ("meas_nan_rate_with_ecc",)}, indent=1))
    if not summary["symmetry_exact"] or (plumb and not all(summary["reproduces_step1"].values())):
        sys.exit("!! check failed -- STOP")


if __name__ == "__main__":
    main()
