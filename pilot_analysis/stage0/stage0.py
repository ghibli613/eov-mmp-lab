#!/usr/bin/env python3
"""Stage 0 -- predicate decomposition and ingredient coverage. Data only.

    PYTHONDONTWRITEBYTECODE=1 python3 pilot_analysis/stage0/stage0.py

Reads the repo's annotation and split files; writes only into
pilot_analysis/stage0/. Imports nothing from the repo (so no __pycache__ lands
in it), loads no checkpoint, touches no GPU.

Support-count protocol: an (axis, value) is "taught" only by BASE-predicate
relation instances in the 800 TRAINING videos. Novel-predicate instances are
present in those annotations but masked from the loss
(models/relation_classifier.py trains on pre_label[:, :, base_pids]); they are
counted separately and labelled not-available-for-training.
"""
from __future__ import annotations

import csv, glob, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# The file the code actually loads (utils/paths.py:56 PRED_SPLIT_INFO).
SPLIT_FILE = "data/vidvrd/data/openvoc_pred_class_spilt_info.json"
# A second copy shipped in configs/; cross-checked, not used.
SPLIT_FILE_ALT = "configs/VidVRD_pred_class_spilt_info_v2.json"
TRAIN_DIR = "data/vidvrd/anno/train"
TEST_DIR = "data/vidvrd/anno/test"

THIN = 50

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = {
    "horizontal":      ["left", "right", "none"],
    "vertical":        ["above", "beneath", "none"],
    "depth":           ["front", "behind", "none"],
    "proximity":       ["overlapping", "adjacent", "separated"],
    "relative_motion": ["approach", "recede", "co_move", "pass", "none"],
    "subject_state":   ["moving", "stationary"],
    "object_state":    ["moving", "stationary"],
    "comparative":     ["larger", "taller", "faster", "none"],
    "locomotion":      ["walk", "run", "fly", "swim", "creep", "jump", "none"],
    "contact_action":  ["hold", "ride", "bite", "touch", "feed", "kick", "none"],
}
AXES = list(SCHEMA)
BOX_AXES = AXES[:8]          # A-H
PIXEL_AXES = AXES[8:]        # I-J

# ---------------------------------------------------------------------------
# Draft phi. Convention: an axis the predicate is SILENT about is left unset.
# `none` is written only where the predicate ASSERTS absence (e.g. `stand`
# asserts no locomotion). Only lexically asserted or definitional values are
# set; everything else is a proposal in UNRESOLVED below.
# ---------------------------------------------------------------------------
VERB = {
    "walk":  {"subject_state": "moving", "locomotion": "walk"},
    "run":   {"subject_state": "moving", "locomotion": "run"},
    "fly":   {"subject_state": "moving", "locomotion": "fly"},
    "swim":  {"subject_state": "moving", "locomotion": "swim"},
    "creep": {"subject_state": "moving", "locomotion": "creep"},
    "jump":  {"subject_state": "moving", "locomotion": "jump"},
    "move":  {"subject_state": "moving"},                 # manner unspecified
    "stand": {"subject_state": "stationary", "locomotion": "none"},
    "sit":   {"subject_state": "stationary", "locomotion": "none"},
    "lie":   {"subject_state": "stationary", "locomotion": "none"},
    "stop":  {"subject_state": "stationary", "locomotion": "none"},
}
SPATIAL = {
    "left":    {"horizontal": "left"},
    "right":   {"horizontal": "right"},
    "above":   {"vertical": "above"},
    "beneath": {"vertical": "beneath"},
    "front":   {"depth": "front"},
    "behind":  {"depth": "behind"},
    "next_to": {"proximity": "adjacent"},
    "inside":  {"proximity": "overlapping"},
    "toward":  {"relative_motion": "approach"},
    "away":    {"relative_motion": "recede"},
    "past":    {"relative_motion": "pass"},
    # `with` is handled per verb below: co_move needs a moving subject.
}
SINGLE = {
    "larger":   {"comparative": "larger"},
    "taller":   {"comparative": "taller"},
    "faster":   {"comparative": "faster", "subject_state": "moving"},
    "chase":    {"subject_state": "moving"},
    "follow":   {"subject_state": "moving"},
    "ride":     {"contact_action": "ride"},
    "touch":    {"contact_action": "touch"},
    "hold":     {"contact_action": "hold"},
    "bite":     {"contact_action": "bite"},
    "feed":     {"contact_action": "feed"},
    "kick":     {"contact_action": "kick"},
    "fall_off": {"subject_state": "moving"},
    "watch":    {},
    "play":     {},
    "fight":    {},
    "pull":     {},
    "drive":    {},
}
MULTIWORD = {"next_to"}   # kept as one lexical component


def tokenise(p):
    if p in MULTIWORD:
        return [p]
    for mw in MULTIWORD:
        if p.endswith("_" + mw):
            return tokenise(p[: -len(mw) - 1]) + [mw]
    return p.split("_")


def phi_of(p):
    if p in SINGLE:
        return dict(SINGLE[p])
    toks = tokenise(p)
    if len(toks) == 1 and toks[0] in SPATIAL:
        return dict(SPATIAL[toks[0]])
    if len(toks) == 2 and toks[0] in VERB:
        v, s = toks
        out = dict(VERB[v])
        if s == "with":
            if out["subject_state"] == "moving":
                out.update(relative_motion="co_move", object_state="moving")
            # stationary + with: no schema value fits -> UNRESOLVED
        else:
            out.update(SPATIAL[s])
        return out
    raise KeyError(f"no phi rule for {p!r}")


# (predicate(s), best guess, reason for doubt). None of these guesses is in
# phi_draft.json; they are proposals for review.
UNRESOLVED = [
    ("stand_* / sit_* / lie_* / stop_*",
     "schema needs a posture axis (stand | sit | lie), pixel-readable",
     "The schema has no axis separating stand, sit, lie and stop: all four map "
     "to subject_state=stationary, locomotion=none. Every stand_X / sit_X / "
     "lie_X / stop_X quartet therefore gets an IDENTICAL phi -- see the "
     "collision table below. Several of these collisions pair a novel "
     "predicate with a base one (e.g. sit_behind vs stand_behind), so a "
     "phi-only scorer cannot tell them apart. Posture is also a pixel property, "
     "so these predicates are marked box-only here while really needing both."),
    ("stop_*",
     "subject_state=stationary (as drafted)",
     "`stop` may denote the transition moving -> stationary, not a state. If "
     "so it needs a temporal value the schema lacks."),
    ("move_*",
     "locomotion left unset (as drafted)",
     "`move` is manner-unspecified, so phi(move_X) is a strict subset of "
     "phi(walk_X), phi(run_X), ... A scorer will rank move_X at least as high "
     "as each of them whenever they fire."),
    ("left / right / front / behind (and every *_left etc.)",
     "camera-frame (image-plane) relations",
     "Reference frame unconfirmed. If VidVRD annotates front/behind relative "
     "to the object's heading (walk_behind = following), `depth` is not a "
     "depth axis and is not readable from boxes as the schema assumes. "
     "Camera-frame front/behind is only indirectly box-readable (occlusion, "
     "box size, y-position)."),
    ("stand_with, lie_with, stop_with",
     "relative_motion=none, proximity=adjacent",
     "`with` for a stationary pair means 'together'; co_move needs motion, and "
     "the schema has no 'accompany/together' value. relative_motion left unset."),
    ("*_with (moving verbs)",
     "proximity=adjacent",
     "co_move and object_state=moving are drafted as definitional; adjacency is "
     "likely but not entailed (a flock flying 'with' each other can be "
     "separated), so proximity is left unset."),
    ("inside, sit_inside, lie_inside, stand_inside",
     "proximity=overlapping (as drafted)",
     "Containment is stronger than overlap; the schema has no 'contained' value "
     "so `inside` shares proximity=overlapping with any overlapping pair."),
    ("toward, away, past (bare)",
     "subject_state=moving",
     "Bare forms do not say which entity moves; the relative motion could come "
     "from the object. subject_state left unset."),
    ("chase",
     "object_state=moving, relative_motion=approach, depth=behind",
     "subject_state=moving is definitional. relative_motion could equally be "
     "co_move (pursuit at a constant gap); depth=behind depends on the "
     "reference-frame question above."),
    ("follow",
     "object_state=moving, relative_motion=co_move, depth=behind",
     "As chase, with a constant gap -> co_move rather than approach."),
    ("faster",
     "object_state=moving",
     "'faster than' a stationary object is degenerate but not impossible in "
     "the annotations; left unset."),
    ("ride",
     "vertical=above, proximity=overlapping, relative_motion=co_move, "
     "subject_state=moving, object_state=moving",
     "Only contact_action=ride is drafted. Rider-above-mount is typical but a "
     "parked bicycle can be 'ridden' in a static frame."),
    ("touch, hold, bite, feed, kick",
     "proximity=overlapping (hold, bite, touch, kick); adjacent (feed)",
     "Contact implies the boxes meet, but 2-D boxes of touching objects can "
     "overlap or merely abut. Left unset."),
    ("watch",
     "no axis fits; nearest is none",
     "Gaze direction is not in the schema. No ingredient assigned -> the "
     "predicate is unscorable under phi as drafted. Base predicate."),
    ("play",
     "no axis fits",
     "Too broad for any schema value. No ingredient assigned. Base predicate."),
    ("fight",
     "contact_action=? (no 'fight' value); proximity=adjacent",
     "The contact_action list has no value for fight. No ingredient assigned. "
     "NOVEL predicate -> uncoverable under the current schema."),
    ("pull",
     "contact_action=hold?, relative_motion=co_move or approach",
     "No 'pull' value; pulling usually involves holding, but that is an "
     "inference. No ingredient assigned. NOVEL predicate."),
    ("drive",
     "contact_action=ride?, proximity=overlapping",
     "No 'drive' value; ride is the nearest but conflates rider and driver. "
     "No ingredient assigned. NOVEL predicate."),
    ("fall_off",
     "vertical=beneath (end state) or relative_motion=recede",
     "Tokenises to [fall, off]; neither is a schema value. Only "
     "subject_state=moving drafted. `fall` is not a locomotion value either. "
     "Base predicate, 4 training instances."),
    ("next_to (tokenisation)",
     "treated as ONE lexical component",
     "Splitting on '_' would give [next, to]; kept whole in "
     "component_frequency.csv. `fall_off` IS split into [fall, off]."),
]


# ---------------------------------------------------------------------------
def load_split():
    info = json.load(open(os.path.join(ROOT, SPLIT_FILE)))
    alt = json.load(open(os.path.join(ROOT, SPLIT_FILE_ALT)))
    split = {p: s for p, s in info["cls2split"].items() if p != "__background__"}
    alt_split = {p: s for p, s in alt["cls2split"].items() if p != "__background__"}
    return split, split == alt_split


def count(dirpath):
    inst, vids = Counter(), defaultdict(set)
    files = sorted(glob.glob(os.path.join(ROOT, dirpath, "*.json")))
    for f in files:
        d = json.load(open(f))
        for r in d["relation_instances"]:
            inst[r["predicate"]] += 1
            vids[r["predicate"]].add(d["video_id"])
    return len(files), inst, {k: len(v) for k, v in vids.items()}


def main():
    split, alt_agrees = load_split()
    n_train_v, tr_i, tr_v = count(TRAIN_DIR)
    n_test_v, te_i, te_v = count(TEST_DIR)
    preds = sorted(split)
    base = [p for p in preds if split[p] == "base"]
    novel = [p for p in preds if split[p] == "novel"]
    unknown = (set(tr_i) | set(te_i)) - set(preds)

    # ---------------- Task 1: inventory ----------------
    print(f"split file: {SPLIT_FILE}  (configs copy agrees: {alt_agrees})")
    print(f"predicates {len(preds)}  base {len(base)}  novel {len(novel)}")
    print(f"videos: train {n_train_v}, test {n_test_v}")
    if len(preds) != 132 or len(novel) != 61:
        print("STOP: predicate or novel count differs from 132 / 61")
        return 1
    if unknown:
        print(f"STOP: annotation predicates not in the split file: {unknown}")
        return 1

    with open(os.path.join(HERE, "predicate_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predicate", "split", "train_instances", "test_instances",
                    "train_videos", "test_videos", "train_instances_usable",
                    "train_note"])
        for p in preds:
            usable = tr_i[p] if split[p] == "base" else 0
            note = "" if split[p] == "base" else "not-available-for-training (masked)"
            w.writerow([p, split[p], tr_i[p], te_i[p], tr_v.get(p, 0),
                        te_v.get(p, 0), usable, note])

    tot = dict(
        base_train=sum(tr_i[p] for p in base), novel_train=sum(tr_i[p] for p in novel),
        base_test=sum(te_i[p] for p in base), novel_test=sum(te_i[p] for p in novel))

    # ---------------- Task 2: tokenisation ----------------
    comp = defaultdict(lambda: dict(base=0, novel=0, base_train=0))
    for p in preds:
        for t in tokenise(p):
            comp[t][split[p]] += 1
            if split[p] == "base":
                comp[t]["base_train"] += tr_i[p]
    with open(os.path.join(HERE, "component_frequency.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "n_base_predicates", "n_novel_predicates",
                    "base_train_instances"])
        for t in sorted(comp, key=lambda t: (-comp[t]["base_train"], t)):
            w.writerow([t, comp[t]["base"], comp[t]["novel"], comp[t]["base_train"]])

    # ---------------- Task 3: phi ----------------
    phi = {p: phi_of(p) for p in preds}
    for p, d in phi.items():
        for a, v in d.items():
            assert v in SCHEMA[a], (p, a, v)

    def group(d):
        box = any(a in BOX_AXES for a in d)
        pix = any(a in PIXEL_AXES and v != "none" for a, v in d.items())
        return "both" if box and pix else "box" if box else "pixel" if pix else "unassigned"

    out = {}
    for p in preds:
        out[p] = dict(phi[p])
        out[p]["_split"] = split[p]
        out[p]["_evidence"] = group(phi[p])
    json.dump(out, open(os.path.join(HERE, "phi_draft.json"), "w"), indent=1)

    # collisions: identical phi
    by_phi = defaultdict(list)
    for p in preds:
        if phi[p]:
            by_phi[tuple(sorted(phi[p].items()))].append(p)
    collisions = [v for v in by_phi.values() if len(v) > 1]

    # ---------------- Task 4: coverage ----------------
    sup, nb, masked = Counter(), Counter(), Counter()
    for p in preds:
        for a, v in phi[p].items():
            if split[p] == "base":
                sup[(a, v)] += tr_i[p]
                nb[(a, v)] += 1
            else:
                masked[(a, v)] += tr_i[p]

    novel_rows = []
    for p in novel:
        ing = sorted(phi[p].items())
        mn = min((sup[x] for x in ing), default=None)
        novel_rows.append((p, ing, mn))
    scored = sorted([r for r in novel_rows if r[2] is not None], key=lambda r: (r[2], r[0]))
    unscored = [r[0] for r in novel_rows if r[2] is None]

    used = {x for p in preds for x in phi[p].items()}
    thin = [(a, v) for a in AXES for v in SCHEMA[a]
            if sup[(a, v)] < THIN and (a, v) in used]
    unused = [(a, v) for a in AXES for v in SCHEMA[a] if (a, v) not in used]
    dep = {x: sorted(p for p in novel if x in phi[p].items()) for x in thin}

    above50 = [p for p, _, m in scored if m > 50]
    above200 = [p for p, _, m in scored if m > 200]
    zero = [p for p, _, m in scored if m == 0]

    base_sets = [set(phi[b].items()) for b in base]
    exact_new = [p for p, ing, _ in scored if set(ing) not in base_sets]
    contain_new = [p for p, ing, _ in scored
                   if not any(set(ing) <= bs for bs in base_sets)]
    same_as_base = {p: sorted(b for b in base if phi[b] == phi[p])
                    for p, _, _ in scored}
    same_as_base = {p: v for p, v in same_as_base.items() if v}

    ev = Counter(out[p]["_evidence"] for p in preds)
    ev_nov = Counter(out[p]["_evidence"] for p in novel)

    fmt = lambda x: f"{x[0]}={x[1]}"
    L = []
    L.append("# Stage 0 — coverage report\n")
    L.append(f"Support = BASE-predicate relation instances in the {n_train_v} "
             f"training videos. `masked` = novel-predicate instances present in "
             f"those same videos but excluded from the loss — "
             f"**not available for training**, shown for reference only.\n")
    L.append("## (a) Support per (axis, value)\n")
    L.append("| axis | value | base train instances | base predicates | masked (novel, not usable) |")
    L.append("|---|---|---:|---:|---:|")
    for a in AXES:
        for v in SCHEMA[a]:
            flag = (" (never assigned)" if (a, v) not in used
                    else " ⚠" if sup[(a, v)] < THIN else "")
            L.append(f"| {a} | {v}{flag} | {sup[(a, v)]} | {nb[(a, v)]} | {masked[(a, v)]} |")
    L.append(f"\n⚠ = assigned to at least one predicate but fewer than {THIN} base "
             f"training instances. (never assigned) = no predicate sets this value "
             f"in the draft, so its zero reflects the drafting convention, not thin "
             f"data.\n")

    L.append("## (b) Novel predicates, sorted by weakest ingredient\n")
    L.append("| novel predicate | min support | ingredients (base support) |")
    L.append("|---|---:|---|")
    for p, ing, m in scored:
        L.append(f"| `{p}` | {m} | " + ", ".join(f"{fmt(x)} ({sup[x]})" for x in ing) + " |")
    L.append(f"\n**{len(unscored)} novel predicates have no ingredient assigned** "
             f"(no schema value fits — see unresolved.md) and cannot be scored: "
             + ", ".join(f"`{p}`" for p in unscored) + ".\n")

    L.append(f"## (c) Thin ingredients (< {THIN} base training instances)\n")
    L.append("| axis=value | base support | novel predicates depending on it |")
    L.append("|---|---:|---|")
    for x in sorted(thin, key=lambda x: (sup[x], x)):
        d = ", ".join(f"`{p}`" for p in dep[x]) or "— (no novel predicate uses it)"
        L.append(f"| {fmt(x)} | {sup[x]} | {d} |")
    L.append(f"\nNear the line: " + ", ".join(
        f"{fmt(x)} ({sup[x]})" for a in AXES for v in SCHEMA[a]
        for x in [(a, v)] if x in used and THIN <= sup[x] < 2 * THIN) + ".\n")
    L.append(f"**Never assigned by the draft** (zero support because no predicate "
             f"sets them, not because data is thin): "
             + ", ".join(fmt(x) for x in unused) + ".\n")

    L.append("## (d) Counts over the 61 novel predicates\n")
    L.append(f"- all ingredients > 50 base instances: **{len(above50)}**")
    L.append(f"- all ingredients > 200 base instances: **{len(above200)}**")
    L.append(f"- at least one ingredient with ZERO base support: **{len(zero)}** — "
             + ", ".join(f"`{p}`" for p in zero))
    L.append(f"- no ingredient assigned at all (not counted above): **{len(unscored)}** — "
             + ", ".join(f"`{p}`" for p in unscored))
    L.append(f"- remainder (min support 1–50): "
             f"**{len(scored) - len(above50) - len(zero)}**\n")

    L.append("## (e) Ingredient combinations no base predicate instantiates (descriptive)\n")
    L.append(f"Over the {len(scored)} novel predicates with ≥1 ingredient:\n")
    L.append(f"- **exact**: no base predicate has an identical phi — "
             f"**{len(exact_new)}**")
    L.append(f"- **containment**: no single base predicate's phi contains all of "
             f"the novel predicate's ingredients — **{len(contain_new)}**: "
             + ", ".join(f"`{p}`" for p in contain_new))
    L.append(f"- the converse — **{len(same_as_base)} novel predicates have a phi "
             f"IDENTICAL to a base predicate**, so phi alone cannot separate them:\n")
    for p, bs in sorted(same_as_base.items()):
        L.append(f"  - `{p}` ≡ " + ", ".join(f"`{b}`" for b in bs))
    L.append("")
    open(os.path.join(HERE, "coverage_report.md"), "w").write("\n".join(L) + "\n")

    # ---------------- unresolved.md ----------------
    U = ["# Stage 0 — unresolved judgement calls\n",
         "Nothing below is in `phi_draft.json`. Each row is a proposal for "
         "review: the best guess, and why it was not drafted.\n",
         "**Drafting convention.** An axis the predicate is silent about is left "
         "*unset*. `none` is written only where the predicate asserts absence "
         "(`stand` → locomotion=none). Evidence group: a predicate needs pixels "
         "if it sets locomotion or contact_action to a non-`none` value; "
         "`locomotion=none` on stationary verbs is implied by "
         "subject_state=stationary and does not count.\n",
         "| predicate(s) | best guess | reason for doubt |", "|---|---|---|"]
    for who, guess, why in UNRESOLVED:
        U.append(f"| {who} | {guess} | {why} |")
    U.append(f"\n## Identical-phi collisions ({len(collisions)} groups, "
             f"{sum(len(c) for c in collisions)} predicates)\n")
    U.append("Predicates in one row are indistinguishable under the drafted phi. "
             "`*` = novel.\n")
    for c in sorted(collisions, key=lambda c: (-len(c), c)):
        U.append("- " + ", ".join(f"`{p}`" + ("*" if split[p] == "novel" else "")
                                  for p in c))
    open(os.path.join(HERE, "unresolved.md"), "w").write("\n".join(U) + "\n")

    # ---------------- machine-readable summary for README ----------------
    summary = dict(
        split_file=SPLIT_FILE, alt_split_file=SPLIT_FILE_ALT, alt_agrees=alt_agrees,
        n_pred=len(preds), n_base=len(base), n_novel=len(novel),
        n_train_videos=n_train_v, n_test_videos=n_test_v, totals=tot,
        n_components=len(comp),
        evidence_all=dict(ev), evidence_novel=dict(ev_nov),
        d=dict(above50=len(above50), above200=len(above200), zero=len(zero),
               zero_list=zero, unassigned=unscored,
               mid=len(scored) - len(above50) - len(zero)),
        thin=[dict(ingredient=fmt(x), support=sup[x], novel=dep[x])
              for x in sorted(thin, key=lambda x: (sup[x], x))],
        unused=[fmt(x) for x in unused],
        e=dict(exact=len(exact_new), containment=len(contain_new),
               containment_list=contain_new, identical_to_base=len(same_as_base)),
        collisions=dict(groups=len(collisions),
                        predicates=sum(len(c) for c in collisions)),
    )
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
