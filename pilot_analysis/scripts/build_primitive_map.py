#!/usr/bin/env python
"""Decompose the 132 VidVRD predicates into (action, spatial) primitives.

    python pilot_analysis/scripts/build_primitive_map.py

Writes pilot_analysis/primitive_map.json and primitive_map_table.md.

Three cases, per the C1 spec:
  lexical      `walk_behind` -> (walk, behind).  105 of 132 split this way.
  degenerate   one slot is genuinely empty: `beneath` -> (None, beneath),
               `feed` -> (feed, None).
  entailed     the predicate is one morpheme but denotes both a manner and a
               configuration: `chase` -> (chase, behind).  String-splitting
               cannot reach these; they are hand-authored below and every one
               carries a confidence flag.

`confidence` is about the SPATIAL slot of the entailed rows -- the action slot
of a single-morpheme predicate is just the predicate, which is never in doubt.
"""
from __future__ import annotations

import json, os, sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

SPLIT_INFO = os.path.join(ROOT, "configs", "VidVRD_pred_class_spilt_info_v2.json")
OUT_JSON = os.path.join(ROOT, "pilot_analysis", "primitive_map.json")
OUT_MD = os.path.join(ROOT, "pilot_analysis", "primitive_map_table.md")

# The 12 spatial morphemes that appear as compound suffixes.
SPATIAL_SUFFIXES = ["next_to", "beneath", "behind", "toward", "inside",
                    "front", "right", "above", "past", "away", "left", "with"]

# ---------------------------------------------------------------------------
# Hand-authored rows for the 27 predicates that do not split lexically.
# (action, spatial, basis, confidence, note)
# ---------------------------------------------------------------------------
MANUAL = {
    # -- bare spatial: no manner is named at all ----------------------------
    "left":     (None, "left",     "degenerate", "high", "bare spatial term"),
    "right":    (None, "right",    "degenerate", "high", "bare spatial term"),
    "front":    (None, "front",    "degenerate", "high", "bare spatial term"),
    "behind":   (None, "behind",   "degenerate", "high", "bare spatial term"),
    "next_to":  (None, "next_to",  "degenerate", "high", "bare spatial term"),
    "above":    (None, "above",    "degenerate", "high", "bare spatial term"),
    "beneath":  (None, "beneath",  "degenerate", "high", "bare spatial term"),
    "away":     (None, "away",     "degenerate", "high", "bare spatial term"),
    "past":     (None, "past",     "degenerate", "high", "bare spatial term"),
    "toward":   (None, "toward",   "degenerate", "high", "bare spatial term"),

    # -- comparatives: box-geometry predicates that are not in the spatial
    #    vocabulary and share no morpheme with anything else. They decompose
    #    into nothing; each is its own singleton primitive. Recorded so the
    #    multiplier for them comes out at 1.0x rather than being silently
    #    dropped.
    "larger":   (None, "larger",   "degenerate", "high", "size comparative; singleton primitive"),
    "taller":   (None, "taller",   "degenerate", "high", "size comparative; singleton primitive"),
    "faster":   (None, "faster",   "degenerate", "high", "speed comparative; singleton primitive"),

    # -- action only: no configuration is entailed --------------------------
    "play":     ("play",  None, "degenerate", "high",
                 "no configuration entailed; the two agents may be anywhere"),
    "watch":    ("watch", None, "degenerate", "medium",
                 "arguably entails `toward` via gaze, but gaze is not box geometry"),
    "feed":     ("feed",  None, "degenerate", "high",
                 "proximity is arguable; left empty per the C1 spec's own example"),

    # -- entailed: one morpheme, two primitives -----------------------------
    "chase":    ("chase",  "behind",  "entailed", "high",
                 "the chaser trails the chased; `behind` is definitional"),
    "follow":   ("follow", "behind",  "entailed", "high",
                 "same geometry as chase, slower manner"),
    "ride":     ("ride",   "above",   "entailed", "medium",
                 "person-on-bicycle/horse: subject box sits above the object's"),
    "touch":    ("touch",  "next_to", "entailed", "medium",
                 "contact entails adjacency"),
    "hold":     ("hold",   "next_to", "entailed", "medium", "contact entails adjacency"),
    "bite":     ("bite",   "next_to", "entailed", "medium", "contact entails adjacency"),
    "kick":     ("kick",   "next_to", "entailed", "medium", "momentary contact entails adjacency"),
    "fight":    ("fight",  "next_to", "entailed", "medium", "engagement entails proximity"),
    "pull":     ("pull",   "toward",  "entailed", "low",
                 "FLAGGED: the object moves toward the subject, but `with` "
                 "(moving together) is an equally defensible reading"),
    "drive":    ("drive",  "inside",  "entailed", "low",
                 "FLAGGED: person-inside-vehicle, but the annotation may intend "
                 "an external view where no containment is visible"),
    "fall_off": ("fall",   "beneath", "entailed", "low",
                 "FLAGGED: `off` is not in the spatial vocabulary. Read as "
                 "ending below the object; `away` (separation) is the "
                 "alternative and would move 4 base instances into `away`"),
}


def decompose(pred):
    if pred in MANUAL:
        a, s, basis, conf, note = MANUAL[pred]
        return dict(action=a, spatial=s, basis=basis, confidence=conf, note=note)
    for suf in SPATIAL_SUFFIXES:
        if pred.endswith("_" + suf):
            return dict(action=pred[: -len(suf) - 1], spatial=suf,
                        basis="lexical", confidence="high", note="")
    raise KeyError(f"no rule for predicate {pred!r}")


def main():
    info = json.load(open(SPLIT_INFO))
    preds = [p for p in info["cls2split"] if p != "__background__"]
    split = info["cls2split"]

    out = {}
    for p in sorted(preds):
        d = decompose(p)
        d["ov_split"] = split[p]
        out[p] = d

    # collisions: two predicates with identical (action, spatial) would be
    # indistinguishable to any purely compositional scorer.
    key = defaultdict(list)
    for p, d in out.items():
        key[(d["action"], d["spatial"])].append(p)
    collisions = {f"{a}+{s}": v for (a, s), v in key.items() if len(v) > 1}

    json.dump(out, open(OUT_JSON, "w"), indent=1, sort_keys=True)

    actions = sorted({d["action"] for d in out.values() if d["action"]})
    spatials = sorted({d["spatial"] for d in out.values() if d["spatial"]})
    basis_c = Counter(d["basis"] for d in out.values())
    conf_c = Counter(d["confidence"] for d in out.values() if d["basis"] == "entailed")

    with open(OUT_MD, "w") as f:
        f.write("# Predicate -> (action, spatial) primitive map\n\n")
        f.write(f"Generated by `scripts/build_primitive_map.py`. "
                f"{len(out)} predicates.\n\n")
        f.write(f"- **{len(actions)} action primitives**: `"
                + "`, `".join(actions) + "`\n")
        f.write(f"- **{len(spatials)} spatial primitives**: `"
                + "`, `".join(spatials) + "`\n")
        f.write(f"- basis: {dict(basis_c)}\n")
        f.write(f"- entailed-row confidence: {dict(conf_c)}\n")
        f.write(f"- (action, spatial) collisions: "
                f"{collisions if collisions else 'none — every predicate has a unique pair'}\n\n")
        f.write("Rows are sorted with the hand-authored ones first; `basis=lexical`\n"
                "rows are mechanical suffix splits and need no review.\n\n")
        f.write("| predicate | split | action | spatial | basis | conf | note |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        order = sorted(out, key=lambda p: (out[p]["basis"] == "lexical", p))
        for p in order:
            d = out[p]
            f.write(f"| `{p}` | {d['ov_split']} | {d['action'] or '—'} | "
                    f"{d['spatial'] or '—'} | {d['basis']} | {d['confidence']} | "
                    f"{d['note']} |\n")

    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    print(f"{len(out)} predicates -> {len(actions)} action primitives, "
          f"{len(spatials)} spatial primitives")
    print(f"basis: {dict(basis_c)}")
    print(f"entailed confidence: {dict(conf_c)}")
    print(f"collisions: {collisions or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
