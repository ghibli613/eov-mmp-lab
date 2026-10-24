#!/usr/bin/env python3
"""Stage 2 Batch A, step 1 -- measured and supplied part scores, learned-part
inputs, slot labels and HELD ground truth, for the 783 training videos with ECC.

    python pilot_analysis/stage2/step1_parts.py --limit 5     # small first: 5 FIT + 5 HELD videos
    python pilot_analysis/stage2/step1_parts.py               # all

CPU only. Training annotations: base labels only (novel instances dropped at
parse, counted). No test file is opened. The grid is the cached one (option-B GT
tracks, gen_feats_test order); it is re-derived from the tracks and must match.

Also the step-1 symmetry check: swapping subject and object, or reversing time,
must map every measured rule exactly (as in stage1b/slot_check.md).

Writes into OUT (default pilot_analysis/stage2/data, or data_small with --limit):
  slots.npz        per slot: meas (N, 16), extra (N, 14), vehicle, logits (N, 132),
                   int_logit, subset code, video / pair / slot indices
  emb.npy          (N, 4608) float16: visual_pre_embeddings | subject | object embedding
  slots_meta.json  videos, subsets, per-slot base labels
  held_pairs.json  per HELD video: the cached pairs needed for post-processing
  held_gt.json     HELD ground truth in the evaluator's format (base only)
  step1_summary.json
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time
from collections import Counter

import numpy as np

import common as C
from common import M, S

SUBSET_CODE = {"fit_train": 0, "es": 1, "held": 2}


def gt_relation_insts(d, rels):
    """Same construction as third_party/vidvrd_ii_helper/dataset/dataset.py:184-218
    (get_relation_insts), applied to the base relations of one training video."""
    cat = {so["tid"]: so["category"] for so in d["subject/objects"]}
    trajs = [{b["tid"]: (b["bbox"]["xmin"], b["bbox"]["ymin"], b["bbox"]["xmax"], b["bbox"]["ymax"]) for b in fr}
             for fr in d["trajectories"]]
    out = []
    for r in rels:
        b, e = r["begin_fid"], r["end_fid"]
        out.append(dict(triplet=[cat[r["subject_tid"]], r["predicate"], cat[r["object_tid"]]],
                        subject_tid=r["subject_tid"], object_tid=r["object_tid"], duration=[b, e],
                        sub_traj=[list(f[r["subject_tid"]]) for f in trajs[b:e]],
                        obj_traj=[list(f[r["object_tid"]]) for f in trajs[b:e]]))
    return out


def sym_check(pair, th, tally):
    """Exact symmetry of every measured rule under swap and time reversal."""
    def add(k, ok):
        tally[k + "_n"] += 1
        tally[k + "_ok"] += bool(ok)

    for w in sorted({C.WIN[k] for k in ("horizontal", "vertical", "depth", "contained", "larger", "faster",
                                         "subject_state", "object_state")}):
        p, _ = pair(w)
        if p is None:
            continue
        f, fs, fr = C.frame_fracs(p), C.frame_fracs(M.swap(p)), C.frame_fracs(M.time_reverse(p))
        add("swap: left<->right, above<->beneath, front<->behind",
            (fs["left"], fs["right"], fs["above"], fs["beneath"], fs["front"], fs["behind"]) ==
            (f["right"], f["left"], f["beneath"], f["above"], f["behind"], f["front"]))
        add("swap: larger<->smaller, taller<->shorter",
            (fs["larger"], fs["smaller"], fs["taller"], fs["shorter"]) == (f["smaller"], f["larger"], f["shorter"], f["taller"]))
        x, xs = M.faster_score(p), M.faster_score(M.swap(p))
        if x == x and x != 0:
            add("swap: faster flips", (xs > 0) == (x < 0))
        add("time reversal: every per-frame fraction unchanged", fr == f)
        pr = M.time_reverse(p)
        for name, fn, thr in (("subject_state", M.subject_motion, th["motion_w2"]),
                              ("object_state", M.object_motion, th["motion_w0"])):
            a, b = fn(p), fn(pr)
            if a == a:
                add(f"time reversal: {name} unchanged", C.state01(a, thr) == C.state01(b, thr))
        if x == x and x != 0:
            add("time reversal: faster unchanged", (M.faster_score(pr) > 0) == (x > 0))
    p, h = pair(C.WIN["D2"])
    if p is not None:
        st = S.d2_stats(p, h)
        if st["sub_motion"] == st["sub_motion"]:
            r = S.d2_decide(st, th["c_w3"], th["motion_w3"])
            rr = S.d2_decide(S.d2_stats(M.time_reverse(p), S.reverse_heading(h)), th["c_w3"], th["motion_w3"])
            want = {"toward": "away", "away": "toward"}.get(r, r)
            add("time reversal: D2 approach<->recede, co_move/pass/none fixed", rr == want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N videos of FIT and of HELD (small run)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(C.OUT, "data_small" if a.limit else "data")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    C.verify_frozen()
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    sp = C.splits()
    subset_of = {v: "held" for v in sp["held"]}
    subset_of.update({v: "es" for v in sp["es"]})
    subset_of.update({v: "fit_train" for v in sp["fit_train"]})
    videos = sorted(sp["fit"])[:a.limit or None] + sorted(sp["held"])[:a.limit or None]
    videos = sorted(videos)
    cache_sums = C.verify_cache(videos)
    th = C.load_thresholds()
    ecc = json.load(open(C.ECC_TRAIN))
    tracks_all = json.load(open(C.TRACKS_TRAIN))
    import torch

    rows, embs, labels_meta, held_pairs, held_gt = [], [], [], {}, {}
    tally, counts = Counter(), Counter()
    vid_list = []
    for vi, v in enumerate(videos):
        vid_list.append(v)
        _, W, H, rels, n_nov, d = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{v}.json"), base)
        counts["novel_dropped_at_parse"] += n_nov
        counts["base_instances"] += len(rels)
        n_frames = len(d["trajectories"])
        frames = [dict() for _ in range(n_frames)]
        span = {}
        for t in tracks_all[v]:
            for i, b in enumerate(t["boxes"]):
                frames[t["begin_fid"] + i][t["tid"]] = tuple(b)
            span[t["tid"]] = (t["begin_fid"], t["end_fid"])
        cache = torch.load(os.path.join(C.CACHE, "train", f"{v}.pt"), weights_only=False)
        assert cache["video"] == v and tuple(cache["size"]) == (W, H), (v, cache["size"], W, H)
        grid = S.slot_grid(span)
        assert [(s, o) for s, o, _, _ in grid] == [(p["sub"], p["obj"]) for p in cache["pairs"]], v
        if subset_of[v] == "held":
            held_gt[v] = gt_relation_insts(d, rels)
            held_pairs[v] = []
        for pi, ((s, o, ov, slots), cp) in enumerate(zip(grid, cache["pairs"])):
            assert list(ov) == list(cp["duration"]) and len(slots) == len(cp["logits"]), (v, pi)
            assert len(cp["windows"]) == 1          # A5: no train pair exceeds 40 slots
            counts["pairs"] += 1
            shared = {}
            r0 = len(rows)
            for k, slot in enumerate(slots):
                pair = C.window_pairs(frames, s, o, slot, ov, ecc[v], W, H, shared)
                meas, ex = C.slot_measures(pair, th)
                sym_check(pair, th, tally)
                labels = sorted(S.slot_labels(slot, cp["sub_gt"], cp["obj_gt"], rels))
                assert set(labels) <= base
                counts["slots"] += 1
                counts["labelled_slots"] += bool(labels)
                rows.append(dict(meas=[meas[c] for c in C.MEAS_COLS], extra=[ex[c] for c in C.EXTRA_FEATS],
                                 vehicle=float(cp["sbj_cls"] in C.VEHICLE),
                                 logits=cp["logits"][k].numpy(), int_logit=float(cp["int_logit"][k]),
                                 subset=SUBSET_CODE[subset_of[v]], video=vi, pair=pi, slot=k))
                embs.append(torch.cat([cp["vpe"][k], cp["sbj_emb"][0], cp["obj_emb"][0]]).numpy())
                labels_meta.append(labels)
            if subset_of[v] == "held":
                held_pairs[v].append(dict(sbj_cls=cp["sbj_cls"], obj_cls=cp["obj_cls"], sbj_scr=cp["sbj_scr"],
                                          obj_scr=cp["obj_scr"], duration=list(cp["duration"]),
                                          sbj_traj=[list(map(float, b)) for b in cp["sbj_traj"]],
                                          obj_traj=[list(map(float, b)) for b in cp["obj_traj"]],
                                          rows=[r0, len(rows)]))
        del cache
        if (vi + 1) % 50 == 0 or vi + 1 == len(videos):
            el = time.time() - t0
            print(f"[{vi + 1}/{len(videos)}] slots {len(rows)} | {el / 60:.1f} min, "
                  f"ETA {el / (vi + 1) * (len(videos) - vi - 1) / 60:.1f} min", flush=True)

    if not a.limit:      # 3,301 novel instances in all 800 training videos = these 783 + the 17 without ECC
        excl = sum(C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{v}.json"), base)[4] for v in sp["no_ecc"])
        counts["novel_in_excluded_no_ecc_videos"] = excl
        assert counts["novel_dropped_at_parse"] + excl == 3301, (counts["novel_dropped_at_parse"], excl)
    novel_labels_used = sum(1 for ls in labels_meta for p in ls if p not in base)
    assert novel_labels_used == 0
    N = len(rows)
    arr = dict(meas=np.array([r["meas"] for r in rows], np.float64),
               extra=np.array([r["extra"] for r in rows], np.float64),
               vehicle=np.array([r["vehicle"] for r in rows], np.float64),
               logits=np.stack([r["logits"] for r in rows]).astype(np.float32),
               int_logit=np.array([r["int_logit"] for r in rows], np.float32),
               subset=np.array([r["subset"] for r in rows], np.int8),
               video=np.array([r["video"] for r in rows], np.int32),
               pair=np.array([r["pair"] for r in rows], np.int32),
               slot=np.array([r["slot"] for r in rows], np.int32))
    E = np.stack(embs).astype(np.float16)
    assert E.shape == (N, 3072 + 768 + 768) and arr["logits"].shape == (N, 132)
    paths = dict(slots=os.path.join(out_dir, "slots.npz"), emb=os.path.join(out_dir, "emb.npy"),
                 meta=os.path.join(out_dir, "slots_meta.json"), held_pairs=os.path.join(out_dir, "held_pairs.json"),
                 held_gt=os.path.join(out_dir, "held_gt.json"), summary=os.path.join(out_dir, "step1_summary.json"))
    np.savez(paths["slots"], **arr)
    np.save(paths["emb"], E)
    json.dump(dict(videos=vid_list, subset_code=SUBSET_CODE, meas_cols=C.MEAS_COLS, extra_feats=C.EXTRA_FEATS,
                   labels=labels_meta), open(paths["meta"], "w"))
    json.dump(held_pairs, open(paths["held_pairs"], "w"))
    json.dump(held_gt, open(paths["held_gt"], "w"))

    sym = {k[:-3]: (tally[k], tally[k[:-3] + "_n"]) for k in sorted(tally) if k.endswith("_ok")}
    nan_rate = {c: float(np.isnan(arr["meas"][:, i]).mean()) for i, c in enumerate(C.MEAS_COLS)}
    by_subset = {s: dict(slots=int((arr["subset"] == c).sum()),
                         labelled=int(sum(1 for i in np.where(arr["subset"] == c)[0] if labels_meta[i])))
                 for s, c in SUBSET_CODE.items()}
    summary = dict(videos=len(videos), counts=dict(counts), novel_labels_used=novel_labels_used,
                   by_subset=by_subset, symmetry=sym, symmetry_exact=all(a_ == n for a_, n in sym.values()),
                   meas_nan_rate=nan_rate, extra_nan_rate={c: float(np.isnan(arr["extra"][:, i]).mean())
                                                         for i, c in enumerate(C.EXTRA_FEATS)},
                   meas_mean={c: float(np.nanmean(arr["meas"][:, i])) for i, c in enumerate(C.MEAS_COLS)},
                   shapes={k: list(v.shape) for k, v in arr.items()} | {"emb": list(E.shape)},
                   held_gt_relations=sum(len(x) for x in held_gt.values()), seconds=round(time.time() - t0, 1))
    json.dump(summary, open(paths["summary"], "w"), indent=1)
    C.write_log("step1_parts" + ("_small" if a.limit else ""), dict(limit=a.limit, thresholds=th, windows=C.WIN, eps=C.EPS),
                [C.PHI, C.SPLIT, C.TH_SLOT, C.TRACKS_TRAIN, C.ECC_TRAIN, C.CACHE_SUMS],
                list(paths.values()), extra=dict(cache_sums_sha256=cache_sums))
    print(json.dumps({k: summary[k] for k in ("videos", "counts", "novel_labels_used", "by_subset", "symmetry_exact",
                                              "shapes", "held_gt_relations", "seconds")}, indent=1))
    for k, (ok, n) in sym.items():
        print(f"  {k}: {ok}/{n}")
    if not summary["symmetry_exact"]:
        sys.exit("!! symmetry not exact -- STOP")


if __name__ == "__main__":
    main()
