#!/usr/bin/env python
"""Phase 2.1 -- partition all 132 VidVRD predicates for the H1 test. No GPU.

    python pilot_analysis/scripts/build_predicate_partition.py

Rationale for the axes (see PILOT-STATUS.md SS C.1):

VidVRD predicates are compositional -- 107 of 132 are `{verb}_{spatial}`. What
matters for H1 is not "static vs dynamic" alone but TWO independent questions:

  evidence axis  geometric  -- decidable from box coordinates alone
                 appearance -- needs pixels; CLIP must ground the concept
  time axis      static     -- decidable from one frame (or one frame + boxes)
                 dynamic    -- needs change over time

EOV-MMP is *given* box trajectories, so `geometric_dynamic` predicates need no
temporal visual modelling at all -- their motion is in the input. Collapsing
them with `appearance_dynamic` (the protocol's `dynamic_kinematic`) is what
makes a binary partition uninformative: it mixes predicates the model has the
evidence for with predicates it does not.

H1 -- "CLIP is weak on verbs" -- predicts failure concentrated in the
*appearance* column, and worst in `appearance_dynamic`.

Emits, into pilot_analysis/:
  predicate_partition.json          full record, one entry per predicate
  splits/pred_split_group.json      evaluator-native: predicate -> group
  splits/pred_split_group_x_ov.json evaluator-native: predicate -> group__base|novel
and prints the reviewable table. Every `review: true` row is a judgement call
that needs a human before any Phase 2 number is quoted.
"""
import json
import os
from collections import Counter, defaultdict

OUT = "pilot_analysis"
SPLIT_INFO = "configs/VidVRD_pred_class_spilt_info_v2.json"
ANNO = "data/vidvrd/anno/test"

# ---------------------------------------------------------------- component tables
# (group, rationale) for the spatial component of a compound predicate.
SPATIAL = {
    "above":   ("static",  "relative box position in one frame"),
    "beneath": ("static",  "relative box position in one frame"),
    "behind":  ("static",  "depth ordering, read from box overlap/size in one frame"),
    "front":   ("static",  "depth ordering, read from box overlap/size in one frame"),
    "left":    ("static",  "relative box position in one frame"),
    "right":   ("static",  "relative box position in one frame"),
    "next_to": ("static",  "box proximity in one frame"),
    "to":      ("static",  "box proximity in one frame (only in `next_to`)"),
    "inside":  ("static",  "box containment in one frame"),
    "with":    ("inherit", "accompaniment; temporality comes from the verb"),
    "past":    ("dynamic", "requires relative displacement across frames"),
    "toward":  ("dynamic", "requires the gap to close across frames"),
    "away":    ("dynamic", "requires the gap to widen across frames"),
    "off":     ("dynamic", "requires separation across frames (only in `fall_off`)"),
}

# (time, evidence, rationale) for the verb component of a compound predicate.
VERB = {
    "stand": ("static", "geometric", "posture, readable from box aspect ratio in one frame"),
    "sit":   ("static", "geometric", "posture, readable from box aspect ratio in one frame"),
    "lie":   ("static", "geometric", "posture, readable from box aspect ratio in one frame"),
    "stop":  ("static", "geometric", "absence of box displacement"),
    "walk":  ("dynamic", "geometric", "locomotion, visible as box displacement"),
    "run":   ("dynamic", "geometric", "locomotion, visible as box displacement (faster)"),
    "fly":   ("dynamic", "geometric", "locomotion, visible as box displacement"),
    "swim":  ("dynamic", "geometric", "locomotion, visible as box displacement"),
    "move":  ("dynamic", "geometric", "locomotion, visible as box displacement"),
    "jump":  ("dynamic", "geometric", "vertical box displacement"),
    "creep": ("dynamic", "geometric", "slow locomotion, visible as box displacement"),
    "fall":  ("dynamic", "geometric", "downward box displacement"),
    "next":  ("static",  "geometric", "only in `next_to`; proximity in one frame"),
}

# Single-token predicates, classified individually.
# name: (time, evidence, review, rationale)
SINGLE = {
    # --- pure geometry, one frame
    "above":   ("static", "geometric", False, "relative box position in one frame"),
    "beneath": ("static", "geometric", False, "relative box position in one frame"),
    "behind":  ("static", "geometric", False, "depth ordering from box overlap in one frame"),
    "front":   ("static", "geometric", False, "depth ordering from box overlap in one frame"),
    "left":    ("static", "geometric", False, "relative box position in one frame"),
    "right":   ("static", "geometric", False, "relative box position in one frame"),
    "larger":  ("static", "geometric", False, "box area comparison; no appearance needed"),
    "taller":  ("static", "geometric", False, "box height comparison; no appearance needed"),
    # --- pure geometry, over time
    "past":    ("dynamic", "geometric", False, "relative displacement across frames"),
    "toward":  ("dynamic", "geometric", False, "gap closes across frames"),
    "away":    ("dynamic", "geometric", False, "gap widens across frames"),
    "chase":   ("dynamic", "geometric", False, "both boxes move, one following the other's path"),
    "follow":  ("dynamic", "geometric", False, "both boxes move along the same path, offset in time"),
    "faster":  ("dynamic", "geometric", False, "comparison of box displacement rates"),
    # --- appearance, one frame (state, visually stable once established)
    "hold":    ("static", "appearance", False, "contact state; a single frame suffices once established"),
    "ride":    ("static", "appearance", False, "support state; a single frame suffices once established"),
    "watch":   ("static", "appearance", False, "gaze direction, read from one frame"),
    "drive":   ("static", "geometric", False, "person-inside-vehicle is box containment given the categories; REVIEWED 2026-08-27, moved from appearance_static"),
    # --- appearance, over time
    "bite":    ("dynamic", "appearance", False, "momentary contact event; needs appearance change"),
    "kick":    ("dynamic", "appearance", False, "momentary limb action; needs appearance change"),
    "fight":   ("dynamic", "appearance", False, "sustained reciprocal action; no single frame decides it"),
    "play":    ("dynamic", "appearance", False, "sustained reciprocal activity; REVIEWED 2026-08-27, kept -- note it is 100% of appearance_dynamic on the base split, and its boundary with `fight` remains a caveat"),
    "touch":   ("static", "appearance", False, "a single frame shows the contact; REVIEWED 2026-08-27, moved from appearance_dynamic (38 base instances -- the only reassignment that moves a number)"),
    "pull":    ("dynamic", "appearance", False, "boxes moving together is `move_with`; telling pulling from being dragged alongside needs pixels to see which entity initiates; REVIEWED 2026-08-27, kept"),
    "feed":    ("dynamic", "appearance", False, "transfer action; needs appearance change; REVIEWED 2026-08-27, kept"),
}


def classify(p):
    """-> (time, evidence, review, rationale)"""
    if p in SINGLE:
        return SINGLE[p]
    if "_" not in p:
        raise KeyError(f"unclassified single-token predicate: {p}")
    verb, spatial = p.split("_", 1)
    if verb not in VERB:
        raise KeyError(f"unknown verb component: {verb!r} in {p!r}")
    if spatial not in SPATIAL:
        raise KeyError(f"unknown spatial component: {spatial!r} in {p!r}")
    v_time, v_evid, v_why = VERB[verb]
    s_time, s_why = SPATIAL[spatial]
    if s_time == "inherit":
        s_time = v_time
    # dynamic if EITHER component needs time
    time = "dynamic" if "dynamic" in (v_time, s_time) else "static"
    return time, v_evid, False, f"verb `{verb}`: {v_why}; spatial `{spatial}`: {s_why}"


def main():
    ov = json.load(open(SPLIT_INFO))["cls2split"]
    preds = [p for p in ov if p != "__background__"]

    counts = Counter()
    for fn in os.listdir(ANNO):
        for r in json.load(open(os.path.join(ANNO, fn)))["relation_instances"]:
            counts[r["predicate"]] += 1

    record = {}
    for p in sorted(preds):
        time, evid, review, why = classify(p)
        record[p] = {
            "group": f"{evid}_{time}",
            "evidence": evid,
            "time": time,
            "ov_split": ov[p],
            "test_instances": counts[p],
            "review": review,
            "rationale": why,
        }

    os.makedirs(f"{OUT}/splits", exist_ok=True)
    json.dump(record, open(f"{OUT}/predicate_partition.json", "w"), indent=1, sort_keys=True)

    # Evaluator-native split files. eval_relation_detection_openvoc reads
    # cls2split and keeps categories whose value == target_split_pred, so these
    # make Phase 2 a config change rather than new AP code.
    g = {p: r["group"] for p, r in record.items()}
    g["__background__"] = "__background__"
    json.dump({"cls2split": g}, open(f"{OUT}/splits/pred_split_group.json", "w"), indent=1)

    gx = {p: f"{r['group']}__{r['ov_split']}" for p, r in record.items()}
    gx["__background__"] = "__background__"
    json.dump({"cls2split": gx}, open(f"{OUT}/splits/pred_split_group_x_ov.json", "w"), indent=1)

    # ------------------------------------------------------------------ report
    print(f"{len(record)} predicates partitioned on two axes\n")
    print("PREDICATE COUNTS  (instances = GT relation instances in the 200-video test set)")
    print(f"{'group':22} {'preds':>6} {'base':>5} {'novel':>6} {'instances':>10} {'inst share':>11}")
    tot_i = sum(r["test_instances"] for r in record.values())
    rows = defaultdict(list)
    for p, r in record.items():
        rows[r["group"]].append(r)
    for grp in ("geometric_static", "geometric_dynamic", "appearance_static", "appearance_dynamic"):
        rs = rows[grp]
        i = sum(r["test_instances"] for r in rs)
        print(f"{grp:22} {len(rs):6d} {sum(r['ov_split']=='base' for r in rs):5d} "
              f"{sum(r['ov_split']=='novel' for r in rs):6d} {i:10d} {i/tot_i:10.1%}")
    print(f"{'TOTAL':22} {len(record):6d} "
          f"{sum(r['ov_split']=='base' for r in record.values()):5d} "
          f"{sum(r['ov_split']=='novel' for r in record.values()):6d} {tot_i:10d} {1.0:10.1%}")

    print("\nNOVEL SPLIT ONLY  (what H1 is actually about)")
    print(f"{'group':22} {'preds':>6} {'instances':>10} {'inst share':>11}")
    nov_i = sum(r["test_instances"] for r in record.values() if r["ov_split"] == "novel")
    for grp in ("geometric_static", "geometric_dynamic", "appearance_static", "appearance_dynamic"):
        rs = [r for r in rows[grp] if r["ov_split"] == "novel"]
        i = sum(r["test_instances"] for r in rs)
        print(f"{grp:22} {len(rs):6d} {i:10d} {i/nov_i:10.1%}")

    rev = sorted(p for p, r in record.items() if r["review"])
    print(f"\nNEEDS HUMAN REVIEW ({len(rev)} predicates) -- these are judgement calls:")
    for p in rev:
        r = record[p]
        print(f"  {p:10} -> {r['group']:19} [{r['ov_split']:5}, {r['test_instances']:4d} inst]  {r['rationale']}")

    print(f"\nwrote {OUT}/predicate_partition.json")
    print(f"wrote {OUT}/splits/pred_split_group.json  (target_split_pred=<group>)")
    print(f"wrote {OUT}/splits/pred_split_group_x_ov.json  (target_split_pred=<group>__novel)")


if __name__ == "__main__":
    main()
