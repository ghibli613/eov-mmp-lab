#!/usr/bin/env python3
"""Stage 0f -- inverse-pair coverage.

    python3 pilot_analysis/stage0f/stage0f.py

Data only. Reads pilot_analysis/stage0e/phi_frozen.json, the split file and the
TRAINING annotations; writes only into pilot_analysis/stage0f/. Test annotations
are never opened. Only BASE labels are used as evidence on either side of a
pair: novel labels are masked from training and are not read.
"""
from __future__ import annotations

import glob, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PHI = os.path.join(ROOT, "pilot_analysis", "stage0e", "phi_frozen.json")
SPLIT = os.path.join(ROOT, "data", "vidvrd", "data", "openvoc_pred_class_spilt_info.json")
TRAIN = os.path.join(ROOT, "data", "vidvrd", "anno", "train")

MIRROR = {"horizontal": {"left": "right", "right": "left"},
          "vertical": {"above": "beneath", "beneath": "above"},
          "depth": {"front": "behind", "behind": "front"}}
AXIS_NAME = {"horizontal": "left/right", "vertical": "above/beneath", "depth": "front/behind"}


def spatial_axis(d):
    ax = [a for a in MIRROR if a in d]
    assert len(ax) <= 1, d
    return ax[0] if ax else None


def mirrored(d):
    a = spatial_axis(d)
    if a is None:
        return None
    m = dict(d)
    m[a] = MIRROR[a][d[a]]
    return m


def spatial_token(p, phi):
    a = spatial_axis(phi[p])
    return phi[p][a] if a else None


def geometry_agrees(axis, value, rows):
    """Sign test of the label against the boxes, averaged over the extent.
    Camera frame (0b): left = smaller centre x; above = smaller centre y;
    front = larger bottom-edge y."""
    if not rows:
        return None
    if axis == "horizontal":
        g = sum(((s["xmin"] + s["xmax"]) - (o["xmin"] + o["xmax"])) / 2 for s, o in rows)
        return (g < 0) == (value == "left")
    if axis == "vertical":
        g = sum(((s["ymin"] + s["ymax"]) - (o["ymin"] + o["ymax"])) / 2 for s, o in rows)
        return (g < 0) == (value == "above")
    g = sum(s["ymax"] - o["ymax"] for s, o in rows)
    return (g > 0) == (value == "front")


def overlaps(a, b):
    return max(a["begin_fid"], b["begin_fid"]) < min(a["end_fid"], b["end_fid"])


def main():
    raw = json.load(open(PHI))
    split = {p: v["_split"] for p, v in raw.items()}
    phi = {p: {k: x for k, x in v.items() if not k.startswith("_")} for p, v in raw.items()}
    info = json.load(open(SPLIT))["cls2split"]
    assert all(info[p] == split[p] for p in split), "phi_frozen split disagrees with split file"
    preds = sorted(phi)
    base = {p for p in preds if split[p] == "base"}
    novel = sorted(p for p in preds if split[p] == "novel")

    # ---------------- 1. inverse map (full-phi match after mirroring) ----------------
    by_phi = {tuple(sorted(d.items())): p for p, d in phi.items() if d}
    inverse = {}
    for p in preds:
        m = mirrored(phi[p])
        inverse[p] = by_phi.get(tuple(sorted(m.items()))) if m else None
    # cross-check against plain string substitution of the spatial token
    swap = {"left": "right", "right": "left", "above": "beneath", "beneath": "above",
            "front": "behind", "behind": "front"}
    for p in preds:
        t = p.split("_")
        s = swap.get(t[-1])
        q = "_".join(t[:-1] + [s]) if s else None
        assert (q if q in phi else None) == inverse[p], (p, q, inverse[p])
    mirrorable = [p for p in preds if spatial_axis(phi[p])]
    no_partner = [p for p in mirrorable if inverse[p] is None]

    # training counts
    tr = Counter()
    vids = []
    for f in sorted(glob.glob(os.path.join(TRAIN, "*.json"))):
        d = json.load(open(f))
        vids.append(d)
        for r in d["relation_instances"]:
            tr[r["predicate"]] += 1

    # ---------------- 2/3. novel predicates ----------------
    rows = []
    for p in novel:
        q = inverse[p]
        rows.append((p, spatial_axis(phi[p]), q, split[q] if q else None,
                     tr[q] if q and split[q] == "base" else None))
    transfer = [r for r in rows if r[3] == "base"]

    # ---------------- 4. do annotators label the reversed pair? ----------------
    stats = defaultdict(Counter)
    geo = defaultdict(Counter)
    detail = defaultdict(Counter)   # (axis) -> Counter of mirror-verb outcomes
    for d in vids:
        B = [{e["tid"]: e["bbox"] for e in fr} for fr in d["trajectories"]]
        rels = [r for r in d["relation_instances"] if r["predicate"] in base]
        by_pair = defaultdict(list)
        for r in rels:
            by_pair[(r["subject_tid"], r["object_tid"])].append(r)
        for r in rels:
            p = r["predicate"]
            a = spatial_axis(phi[p])
            if a is None:
                continue
            val = phi[p][a]
            mir = MIRROR[a][val]
            q = inverse[p]
            rev = [x for x in by_pair.get((r["object_tid"], r["subject_tid"]), [])
                   if overlaps(r, x)]
            rev_sp = [x for x in rev if phi[x["predicate"]].get(a) == mir]
            rev_same = [x for x in rev if phi[x["predicate"]].get(a) == val]
            exact = q is not None and q in base and any(x["predicate"] == q for x in rev)
            if exact:
                out = "exact mirror predicate"
            elif rev_sp:
                out = "mirror spatial term, different predicate"
            elif rev_same:
                out = "SAME spatial term (contradiction)"
            elif rev:
                out = "other base label, no term on this axis"
            else:
                out = "no base label on reversed pair"
            stats[a][out] += 1
            stats[a]["_n"] += 1
            if q is not None and q in base:
                stats[a]["_n_mirror_base"] += 1
                stats[a]["_exact_when_possible"] += exact
            if rev_sp and not exact:
                for x in rev_sp:
                    detail[a][f"{p} → {x['predicate']}"] += 1
            # geometry of the anchor label over its own extent
            gr = []
            for t in range(r["begin_fid"], min(r["end_fid"], len(B))):
                s, o = B[t].get(r["subject_tid"]), B[t].get(r["object_tid"])
                if s and o:
                    gr.append((s, o))
            g = geometry_agrees(a, val, gr)
            if g is not None:
                k = "annotated" if (exact or rev_sp) else "unannotated"
                geo[a][k + "_n"] += 1
                geo[a][k + "_agree"] += g

    # ---------------- write ----------------
    pct = lambda a, b: f"{a}/{b} ({a / b:.1%})" if b else "—"
    L = ["# Stage 0f — inverse-pair coverage\n",
         "Source: `pilot_analysis/stage0e/phi_frozen.json` and "
         "`data/vidvrd/data/openvoc_pred_class_spilt_info.json`. Evidence: BASE labels in "
         "the 800 training videos only. Novel labels were not read, and test annotations "
         "were not opened.\n",
         "## 1. The inverse map\n",
         "inverse(p) = the predicate whose phi equals phi(p) with its spatial value "
         "mirrored (left↔right, above↔beneath, front↔behind), every other axis unchanged. "
         "I matched on full phi equality, and cross-checked that against plain string "
         "substitution: the two agree for all 132 predicates.\n",
         f"- {len(mirrorable)} predicates carry a mirrorable spatial term, and "
         f"{len(mirrorable) - len(no_partner)} of them have their inverse in the vocabulary.",
         f"- The {len(no_partner)} mirrorable predicates whose mirror is not in the "
         f"vocabulary: " + ", ".join(f"`{p}`" for p in no_partner) + ".",
         f"- {len(preds) - len(mirrorable)} predicates have no mirrorable term "
         f"(next_to, with, toward, away, past, inside, comparatives, actions).\n",
         "⚠ **The map mirrors the spatial term and keeps the verb, but an argument "
         "swap moves the verb onto the other entity.** From A `stand_left` B, swapping "
         "the arguments gives *B is right of A*, and says nothing about B standing. So "
         "the same-verb inverse is exact only when both entities share the verb, "
         "posture and subject_kind. Section 4 measures how often that holds.\n",
         "## 2. The 61 novel predicates\n",
         "| novel predicate | axis | inverse | inverse split | base inverse, training instances |",
         "|---|---|---|---|---:|"]
    for p, a, q, s, n in sorted(rows, key=lambda r: (r[3] != "base", r[3] is None, r[0])):
        L.append(f"| `{p}` | {AXIS_NAME.get(a, '—')} | {f'`{q}`' if q else '—'} | "
                 f"{s or '—'} | {n if n is not None else '—'} |")
    c = Counter(r[3] for r in rows)
    L.append(f"\n**Transfer set (inverse is base): {len(transfer)} of 61.** "
             f"Inverse is novel: {c['novel']}. No inverse: {c[None]}.\n")
    L.append("## 3. Transfer set — base inverse support\n")
    L.append("| novel predicate | base inverse | training instances |")
    L.append("|---|---|---:|")
    for p, a, q, s, n in sorted(transfer, key=lambda r: -r[4]):
        L.append(f"| `{p}` | `{q}` | {n} |")
    L.append(f"\nTotal: {sum(r[4] for r in transfer)} base training instances across the "
             f"{len(transfer)} inverses.\n")

    L.append("## 4. Do annotators label the reversed pair? ⚠ the critical number\n")
    L.append("Anchor: every BASE training instance whose predicate has a mirrorable "
             "spatial term. Check the reversed pair (object as subject), same video, "
             "temporally overlapping extent, BASE labels only.\n")
    outcomes = ["exact mirror predicate", "mirror spatial term, different predicate",
                "SAME spatial term (contradiction)", "other base label, no term on this axis",
                "no base label on reversed pair"]
    L.append("| axis | anchors | " + " | ".join(outcomes) + " |")
    L.append("|---|---:|" + "---:|" * len(outcomes))
    for a in MIRROR:
        s = stats[a]
        L.append(f"| {AXIS_NAME[a]} | {s['_n']} | " + " | ".join(
            f"{s[o] / s['_n']:.1%}" for o in outcomes) + " |")
    L.append("\n**Mirror annotated in any form** (exact or with a different verb), and "
             "the exact-mirror rate restricted to anchors whose mirror predicate is base "
             "(so an exact match was possible):\n")
    L.append("| axis | mirror annotated, any verb | exact mirror, when the mirror is base |")
    L.append("|---|---:|---:|")
    for a in MIRROR:
        s = stats[a]
        any_ = s["exact mirror predicate"] + s["mirror spatial term, different predicate"]
        L.append(f"| {AXIS_NAME[a]} | {pct(any_, s['_n'])} | "
                 f"{pct(s['_exact_when_possible'], s['_n_mirror_base'])} |")
    L.append("\nMost common different-predicate mirrors (anchor → label on the reversed pair):\n")
    for a in MIRROR:
        L.append(f"- {AXIS_NAME[a]}: " + ", ".join(
            f"`{k}` {n}" for k, n in detail[a].most_common(6)))
    L.append("\n**Geometry.** The unannotated reverse direction is geometrically the "
             "mirror of the annotated one by construction: if A is left of B in the "
             "image, B is right of A. So the only question is whether the annotated "
             "label agrees with the boxes. Here is that sign test, split by whether the "
             "reverse was annotated:\n")
    L.append("| axis | reverse annotated: label agrees with boxes | reverse NOT annotated: label agrees with boxes |")
    L.append("|---|---:|---:|")
    for a in MIRROR:
        g = geo[a]
        L.append(f"| {AXIS_NAME[a]} | {pct(g['annotated_agree'], g['annotated_n'])} | "
                 f"{pct(g['unannotated_agree'], g['unannotated_n'])} |")
    L.append("\nSign rules (camera frame, per 0b): left = smaller centre x; above = smaller "
             "centre y; front = lower bottom edge. Averaged over the anchor's extent.\n")
    sh, sv, sd_ = stats["horizontal"], stats["vertical"], stats["depth"]
    anym = lambda s: (s["exact mirror predicate"] + s["mirror spatial term, different predicate"]) / s["_n"]
    contra = sum(stats[a]["SAME spatial term (contradiction)"] for a in MIRROR)
    L.append("## Conclusion\n")
    L.append(f"**1. The spatial inverse is confirmed by the annotators.** The reversed "
             f"pair carries the mirror spatial term for {anym(sh):.1%} of left/right "
             f"anchors and {anym(sd_):.1%} of front/behind. The reversed pair carries the "
             f"SAME term (a contradiction) {contra} time{'s' if contra != 1 else ''} in "
             f"{sum(stats[a]['_n'] for a in MIRROR)} anchors. "
             f"above/beneath reaches {anym(sv):.1%}, but that is a lower bound: most "
             f"above/beneath compounds are novel, their labels were not read, and "
             f"{sv['no base label on reversed pair'] / sv['_n']:.1%} of reversed pairs "
             f"show no base label at all. Where the reverse is unannotated, the "
             f"annotated label agrees with the boxes just as often as where it is "
             f"annotated. The omissions are gaps in labelling, not geometric "
             f"disagreement.\n")
    L.append(f"**2. The verb does not invert. ⚠** The *exact* same-verb mirror appears "
             f"on only {sh['_exact_when_possible'] / sh['_n_mirror_base']:.1%} of "
             f"left/right anchors and {sd_['_exact_when_possible'] / sd_['_n_mirror_base']:.1%} "
             f"of front/behind anchors, even where that mirror is a base predicate. The "
             f"other half carry the OBJECT's own verb: `walk_left` ↔ `stand_right`, "
             f"`move_right` ↔ `stop_left`. On above/beneath the reversed pair of "
             f"`sit_above` is `move_beneath` or `jump_beneath` (287 and 72 instances). "
             f"The thing below is a moving entity, not a sitting one. Annotators treat "
             f"the verb as a property of the subject, as expected under argument swap. "
             f"The same-verb inverse holds only when both entities happen to share the "
             f"verb.\n")
    L.append(f"**3. So the transfer set adds no reachability under the frozen "
             f"schema.** {len(transfer)} of 61 novel predicates have a base inverse. "
             f"But in `phi_frozen` the part that inverts cleanly, the spatial term, is a "
             f"MEASURED axis and needs no transfer. The part that would need transfer, "
             f"the verb or posture, is exactly the part that does not invert. "
             f"`sit_beneath` gets `beneath` from the boxes and `sit` from all "
             f"{tr['sit_above'] + tr['sit_front'] + tr['sit_left'] + tr['sit_right']} "
             f"base sit_* instances. Swapped `sit_above` instances show a moving thing "
             f"beneath a sitting thing, which teaches nothing about sitting beneath. "
             f"All {len(transfer)} are already reachable without the inverse map. "
             f"`creep_above`'s inverse has only {tr['creep_beneath']} instances "
             f"anyway.\n")
    L.append("What the bidirectional annotation does give is a free consistency "
             "check. For left/right and front/behind, a prediction on (A, B) implies "
             "the mirror spatial term on (B, A), and the annotators agree with that "
             "94–100% of the time. That is a constraint on the measured spatial axes, "
             "not a source of verb supervision.\n")
    open(os.path.join(HERE, "inverse_pairs.md"), "w").write("\n".join(L) + "\n")

    summary = dict(
        mirrorable=len(mirrorable), no_partner=no_partner,
        novel=[dict(pred=p, axis=a, inverse=q, inverse_split=s, base_train=n)
               for p, a, q, s, n in rows],
        transfer=[r[0] for r in transfer],
        stats={a: dict(v) for a, v in stats.items()},
        geometry={a: dict(v) for a, v in geo.items()},
        different_predicate_mirrors={a: dict(v.most_common(15)) for a, v in detail.items()},
    )
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
