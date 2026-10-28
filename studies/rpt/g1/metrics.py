"""RPT G1 (A29): every quantity of A29's bars, as pure functions. CPU only, no test file.

The GT unit is session 3's option-B run (A29 Definitions item 3); a track is matched to runs by
cut-vIoU (item 2): D1's `cut_viou` convention (studies/d2/stages.py:147-159) with the official `viou`
(inference/video_relation_detection_openvoc.py:73-114). Pair recall is D1's U1/U2 code (item 4,
pilot_analysis/time_check/tc_run.py:232-250, through studies/d2/stages.best_viou). Nothing here is
re-implemented that those files already do; they are imported.

Track format (both trackers): {tid: dict(tid, category, begin_fid, end_fid, boxes)}, dense over
[begin_fid, end_fid), as the repo's format_trajectories_test and the RPT adapter give them.
Run format: dict(rid, gt_tid, category, begin_fid, end_fid, boxes, annotated), `rid` the run's
index in gt_tracks_train.json (its `tid`), `annotated` the frames where the training annotation
has a box for gt_tid (the run's other frames are interpolated).
"""
from __future__ import annotations

import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
STUDIES = os.path.dirname(RPT)
REPO = os.path.dirname(STUDIES)
for p in (STUDIES, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from d2 import stages as D1  # noqa: E402  (cut_viou, best_viou: D1's convention, pinned in A25a)
from inference.video_relation_detection_openvoc import iou, viou  # noqa: E402

THR = 0.5                      # A29 item 2: a match needs cut-vIoU >= 0.5; item 4: below 0.5 is U1
PAIR_OVERLAP = 10              # models/gen_labels.py:659: pairs overlapping < 10 frames are skipped
BOX_STYLE_BAR = 0.70
N_BINS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, None))   # A29: track counts by N


# ---------------------------------------------------------------- inputs
def check_tracks(tracks: dict) -> dict:
    """Raises ValueError unless every track is dense over [begin_fid, end_fid)."""
    for k, t in tracks.items():
        if k != t["tid"] or len(t["boxes"]) != t["end_fid"] - t["begin_fid"] or t["end_fid"] <= t["begin_fid"]:
            raise ValueError(f"track {k}: not dense over [{t['begin_fid']}, {t['end_fid']})")
    return tracks


def tracks_from_list(rows: list) -> dict:
    """[{tid, category, begin_fid, end_fid, boxes | trajectory}] -> {tid: track}."""
    out = {}
    for r in rows:
        boxes = r.get("boxes")
        if boxes is None:
            traj = {int(f): b for f, b in r["trajectory"].items()}
            boxes = [list(traj[f]) for f in sorted(traj)]
        out[int(r["tid"])] = dict(tid=int(r["tid"]), category=r["category"], begin_fid=int(r["begin_fid"]),
                                  end_fid=int(r["end_fid"]), boxes=[list(map(float, b)) for b in boxes],
                                  score=float(r["score"]) if "score" in r else None)
    return check_tracks(out)


def runs_of(option_b: list, anno: dict) -> list:
    """One video's option-B runs (gt_tracks_train.json) with their annotated frames, read from the
    video's training annotation. Each run's box on an annotated frame must be the annotation's."""
    boxes = defaultdict(dict)
    for f, frame in enumerate(anno["trajectories"]):
        for b in frame:
            bb = b["bbox"]
            boxes[b["tid"]][f] = [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]
    out = []
    for r in option_b:
        b, e = int(r["begin_fid"]), int(r["end_fid"])
        if len(r["boxes"]) != e - b:
            raise ValueError(f"run {r['tid']}: not dense over [{b}, {e})")
        ann = sorted(f for f in boxes[r["gt_tid"]] if b <= f < e)
        for f in ann:
            if [float(x) for x in r["boxes"][f - b]] != [float(x) for x in boxes[r["gt_tid"]][f]]:
                raise ValueError(f"run {r['tid']}, frame {f}: box differs from the annotation")
        out.append(dict(rid=int(r["tid"]), gt_tid=int(r["gt_tid"]), category=r["category"], begin_fid=b, end_fid=e,
                        boxes=[list(map(float, x)) for x in r["boxes"]], annotated=ann))
    if len({r["rid"] for r in out}) != len(out):
        raise ValueError("run ids are not unique")
    return out


# ---------------------------------------------------------------- item 2: matching
def cut_viou(track: dict, run: dict) -> float:
    """The track cut to the run's frames, against the run (D1's cut_viou, official viou)."""
    return D1.cut_viou(track["begin_fid"], track["boxes"], run["boxes"], run["begin_fid"], viou)


def match(tracks: dict, runs: list, class_aware: bool = True) -> dict:
    """A29 item 2. Candidate (track, run) pairs with cut-vIoU >= 0.5, greedily in descending
    cut-vIoU (ties: lower track id, then lower run index). Accepted if the run is unmatched and the
    track is unmatched or matched only to runs of the same tid. -> {rid: (tid, cut-vIoU)}."""
    cands = []
    for t in tracks.values():
        for r in runs:
            if class_aware and t["category"] != r["category"]:
                continue
            v = cut_viou(t, r)
            if v >= THR:
                cands.append((-v, t["tid"], r["rid"]))
    cands.sort()
    gt_of = {r["rid"]: r["gt_tid"] for r in runs}
    run_to, track_gt = {}, {}
    for nv, tid, rid in cands:
        if rid in run_to:
            continue
        if tid in track_gt and track_gt[tid] != gt_of[rid]:
            continue
        run_to[rid] = (tid, -nv)
        track_gt[tid] = gt_of[rid]
    return run_to


def unmatched_tracks(tracks: dict, matched: dict) -> list:
    """A29 item 5: tracks that match no run."""
    used = {tid for tid, _ in matched.values()}
    return sorted(t for t in tracks if t not in used)


# ---------------------------------------------------------------- item 6: box style
def box_style_pair(track: dict, run: dict):
    """Mean per-frame official iou over the run's annotated frames where the track has a box;
    None if there is no such frame."""
    fs = [f for f in run["annotated"] if track["begin_fid"] <= f < track["end_fid"]]
    if not fs:
        return None
    return sum(iou(track["boxes"][f - track["begin_fid"]], run["boxes"][f - run["begin_fid"]]) for f in fs) / len(fs)


# ---------------------------------------------------------------- item 4: pair recall
def overlap(a: dict, b: dict) -> int:
    return min(a["end_fid"], b["end_fid"]) - max(a["begin_fid"], b["begin_fid"])


def best_pair(tracks: dict, rel: dict, scoreable: bool = False):
    """D1's best pair (tc_run.py:236-241): max over ordered track pairs s != o of min(subject vIoU,
    object vIoU), ties by the lower s, then the lower o. With `scoreable`, only pairs whose tracks
    overlap >= 10 frames (gen_labels.py:659). -> (min vIoU, s, o); (0.0, None, None) if no pair."""
    gb = rel["duration"][0]
    vs = D1.best_viou(tracks, rel["sub_traj"], gb, viou)
    vo = D1.best_viou(tracks, rel["obj_traj"], gb, viou)
    cand = [(min(vs[s], vo[o]), -s, -o, s, o) for s in tracks for o in tracks
            if s != o and (not scoreable or overlap(tracks[s], tracks[o]) >= PAIR_OVERLAP)]
    if not cand:
        return 0.0, None, None
    m, _, _, s, o = max(cand)
    return m, s, o


def bucket(tracks: dict, rel: dict, scoreable: bool = False) -> dict:
    """U1 (best pair below 0.5), U2 (best pair's names differ from the GT triplet's) or recalled."""
    m, s, o = best_pair(tracks, rel, scoreable)
    if m < THR:
        return dict(bucket="U1", best=m, sub=s, obj=o)
    if tracks[s]["category"] != rel["triplet"][0] or tracks[o]["category"] != rel["triplet"][2]:
        return dict(bucket="U2", best=m, sub=s, obj=o)
    return dict(bucket="recalled", best=m, sub=s, obj=o)


# ---------------------------------------------------------------- item 8: raw detections
def raw_fraction(run: dict, raw_low: list) -> float:
    """Fraction of the run's annotated frames with a raw [0.1, 0.5) detection of the run's class
    (`raw_low` of that class's prompt: [frame, x1, y1, x2, y2, p]) at official iou >= 0.5."""
    if not run["annotated"]:
        return 0.0
    by_f = defaultdict(list)
    for d in raw_low:
        by_f[int(d[0])].append(d[1:5])
    hit = sum(any(iou(b, run["boxes"][f - run["begin_fid"]]) >= THR for b in by_f.get(f, ()))
              for f in run["annotated"])
    return hit / len(run["annotated"])


def fraction_summary(xs: list) -> dict:
    if not xs:
        return dict(n=0)
    s = sorted(xs)
    dec = [s[min(len(s) - 1, int(round(q / 10 * (len(s) - 1))))] for q in range(11)]
    return dict(n=len(s), deciles=dec, share_0=sum(x == 0 for x in s) / len(s),
                share_gt_0=sum(x > 0 for x in s) / len(s),
                **{f"share_ge_{q}": sum(x >= q for x in s) / len(s) for q in (0.25, 0.5, 0.75)})


# ---------------------------------------------------------------- track counts by N
def n_bin(n: int) -> str:
    for lo, hi in N_BINS:
        if n >= lo and (hi is None or n <= hi):
            return f"{lo}" if lo == hi else (f">={lo}" if hi is None else f"{lo}-{hi}")
    raise ValueError(n)


# ---------------------------------------------------------------- one video, one tracker
def video_record(tracks: dict, runs: list, rels: list) -> dict:
    """Every per-video count both trackers share: matches (class-aware and -agnostic), unmatched
    tracks, pair buckets (all pairs and scoreable pairs)."""
    m_aw, m_ag = match(tracks, runs, True), match(tracks, runs, False)
    return dict(n_tracks=len(tracks), n_runs=len(runs),
                matched=m_aw, matched_agnostic=m_ag,
                unmatched=unmatched_tracks(tracks, m_aw), unmatched_agnostic=unmatched_tracks(tracks, m_ag),
                pairs=[bucket(tracks, r) for r in rels], pairs_scoreable=[bucket(tracks, r, True) for r in rels])


# ---------------------------------------------------------------- pooled over videos
def pooled(per_video: dict, runs: dict, rels: dict, novel: set, videos=None) -> dict:
    """per_video: {video: video_record}; runs: {video: runs}; rels: {video: relations}.
    -> run, tid and pair recall, class-aware and -agnostic, overall, base, novel and per class."""
    videos = sorted(per_video) if videos is None else videos
    out = {}
    for key, mk in (("aware", "matched"), ("agnostic", "matched_agnostic")):
        hit = Counter()
        tot = Counter()
        tid_hit, tid_all = set(), set()
        for v in videos:
            for r in runs[v]:
                grp = "novel" if r["category"] in novel else "base"
                rec = r["rid"] in per_video[v][mk]
                for g in ("all", grp, f"class:{r['category']}"):
                    tot[g] += 1
                    hit[g] += rec
                tid_all.add((v, r["gt_tid"], grp))
                if rec:
                    tid_hit.add((v, r["gt_tid"], grp))
        out[f"run_recall_{key}"] = {g: dict(hit=hit[g], n=tot[g], recall=hit[g] / tot[g]) for g in sorted(tot)}
        out[f"tid_recall_{key}"] = {g: dict(hit=sum(1 for t in tid_hit if g == "all" or t[2] == g),
                                            n=sum(1 for t in tid_all if g == "all" or t[2] == g))
                                    for g in ("all", "base", "novel")}
        for g in out[f"tid_recall_{key}"].values():
            g["recall"] = g["hit"] / g["n"] if g["n"] else None
    for key in ("pairs", "pairs_scoreable"):
        b = Counter(p["bucket"] for v in videos for p in per_video[v][key])
        n = sum(b.values())
        out[key] = dict(n=n, buckets=dict(b), recall=b["recalled"] / n if n else None,
                        recall_agnostic=(n - b["U1"]) / n if n else None)
    n_v = len(videos)
    out["unmatched_per_video"] = sum(len(per_video[v]["unmatched"]) for v in videos) / n_v
    out["unmatched_agnostic_per_video"] = sum(len(per_video[v]["unmatched_agnostic"]) for v in videos) / n_v
    out["track_counts_by_N"] = dict(Counter(n_bin(per_video[v]["n_tracks"]) for v in videos))
    out["tracks_per_video"] = sum(per_video[v]["n_tracks"] for v in videos) / n_v
    return out


def box_style(per_video: dict, tracks: dict, runs: dict) -> dict:
    """A29 item 6, SAM 3's class-aware matches: the median over pairs of box_style_pair."""
    vals, none = [], 0
    for v, rec in per_video.items():
        by_rid = {r["rid"]: r for r in runs[v]}
        for rid, (tid, _) in sorted(rec["matched"].items()):
            x = box_style_pair(tracks[v][tid], by_rid[rid])
            if x is None:
                none += 1
            else:
                vals.append(x)
    return dict(n_pairs=len(vals), pairs_without_annotated_shared_frame=none,
                median=statistics.median(vals) if vals else None, values=vals)


# ---------------------------------------------------------------- the bars
def bars(sam3: dict, eov: dict, style: dict, sensitivity: dict | None) -> dict:
    """A29's bars from the pooled records. Differences in points (x100)."""
    r_s, r_e = sam3["run_recall_aware"]["all"]["recall"], eov["run_recall_aware"]["all"]["recall"]
    p_s, p_e = sam3["pairs"]["recall"], eov["pairs"]["recall"]
    u_s, u_e = sam3["unmatched_per_video"], eov["unmatched_per_video"]
    out = {
        "overall_recall": dict(sam3=r_s, eov=r_e, diff_points=100 * (r_s - r_e), threshold=">= 10 points",
                               status="binding", pass_=100 * (r_s - r_e) >= 10),
        "pair_recall": dict(sam3=p_s, eov=p_e, diff_points=100 * (p_s - p_e), threshold=">= 10 points",
                            status="binding", pass_=100 * (p_s - p_e) >= 10),
        "false_track_burden": dict(sam3=u_s, eov=u_e, ratio=(u_s / u_e) if u_e else None, threshold="<= 1.5x",
                                   status="binding", pass_=(u_s / u_e <= 1.5) if u_e else (u_s == 0)),
        "box_style": dict(median=style["median"], n_pairs=style["n_pairs"], threshold=">= 0.70", status="binding",
                          pass_=style["median"] is not None and style["median"] >= BOX_STYLE_BAR),
    }
    if sensitivity is not None:
        d = 100 * (sensitivity["sam3"]["hit_rate"] - sensitivity["eov"]["hit_rate"])
        out["box_sensitivity"] = dict(sam3=sensitivity["sam3"]["hit_rate"], eov=sensitivity["eov"]["hit_rate"],
                                      gt=sensitivity["gt"]["hit_rate"], diff_points=d, threshold=">= -5 points",
                                      status="secondary", pass_=d >= -5)
    return out


# ---------------------------------------------------------------- item 7: box sensitivity
def sensitivity_relations(rels: list, rec_sam3: dict, rec_eov: dict) -> list:
    """The relations both trackers recall (binding sense), with each tracker's best pair."""
    out = []
    for k, (r, a, b) in enumerate(zip(rels, rec_sam3["pairs"], rec_eov["pairs"])):
        if a["bucket"] == "recalled" and b["bucket"] == "recalled":
            out.append(dict(k=k, sam3=(a["sub"], a["obj"]), eov=(b["sub"], b["obj"])))
    return out


def cut_track(t: dict, b: int, e: int, tid: int) -> dict | None:
    """A track cut to [max(b, begin), min(e, end)); None if empty."""
    lo, hi = max(b, t["begin_fid"]), min(e, t["end_fid"])
    if hi <= lo:
        return None
    return dict(tid=tid, category=t["category"], begin_fid=lo, end_fid=hi,
                boxes=t["boxes"][lo - t["begin_fid"]:hi - t["begin_fid"]])


def sensitivity_inputs(rel: dict, sel: dict, sam3: dict, eov: dict) -> dict:
    """Per box source, the two-track list [subject, object] over the relation's span, in the
    repo's track format (score 1.0 for GT, the track's own otherwise; gen_feats_test reads it only
    into the pair data)."""
    gb, ge = rel["duration"]
    out = dict(gt=[dict(tid=0, category=rel["triplet"][0], score=1.0, begin_fid=gb, end_fid=ge,
                        boxes=[list(map(float, x)) for x in rel["sub_traj"]]),
                   dict(tid=1, category=rel["triplet"][2], score=1.0, begin_fid=gb, end_fid=ge,
                        boxes=[list(map(float, x)) for x in rel["obj_traj"]])])
    for name, tr in (("sam3", sam3), ("eov", eov)):
        s, o = sel[name]
        cs, co = cut_track(tr[s], gb, ge, 0), cut_track(tr[o], gb, ge, 1)
        out[name] = None if cs is None or co is None else [dict(cs, score=float(tr[s]["score"])),
                                                           dict(co, score=float(tr[o]["score"]))]
    return out


def top5_base_hit(logits, int_logit, gt_pred_id: int, base_ids: list) -> bool:
    """A29 item 7: the GT predicate among the top 5 base predicates of sigmoid(logit) x
    sigmoid(interactiveness)."""
    import math
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    si = sig(float(int_logit))
    sc = sorted(((sig(float(logits[i])) * si, -i, i) for i in base_ids), reverse=True)[:5]
    return gt_pred_id in {i for _, _, i in sc}


# ---------------------------------------------------------------- bootstrap (descriptive)
def bootstrap(videos: list, stat, n: int = 1000, seed: int = 0) -> dict:
    """1,000 resamples over the videos with numpy.random.default_rng(0); `stat(sample)` gets the
    same list of videos (with repeats) for both trackers. -> the 2.5% and 97.5% percentiles."""
    import numpy as np
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(videos), len(videos))
        x = stat([videos[i] for i in idx])
        if x is not None:
            vals.append(x)
    return dict(n=len(vals), lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)))
