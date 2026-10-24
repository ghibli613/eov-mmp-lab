#!/usr/bin/env python3
"""Q7 (A25e): of D1's 262 SGDet U1 relations (novel pass), how many does the released baseline
match in PredCls (GT tracks and GT classes)? An upper bound on what fixing the tracks could recover.
DESCRIPTIVE ONLY: no decision rule, no tuning.

    python studies/design_review/q7_headroom.py

Reads ONE file: D1's tc_missed.json (SHA256-checked). No label file is opened, and no D2 output
is used (A25d). A U1 relation counts as matched in PredCls when D1 did not list it among the
PredCls misses. D1's rows carry no track ids, so relations are identified by (video, subject,
predicate, object, duration). Where one key holds several rows, the count is bounded, not exact.
"""
import collections
import hashlib
import json
import os
import sys

TC = os.path.expanduser("~/Documents/thesis/source/ov-vidvrd-lab/pilot_analysis/time_check/data/tc_missed.json")
PIN = "2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2"
NOVEL_OBJ = {"antelope", "bear", "cattle", "fox", "giant_panda", "horse", "lizard", "motorcycle", "red_panda", "watercraft"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q7_headroom.json")


def key(r):
    return (r["video"], r["subject"], r["predicate"], r["object"], tuple(r["duration"]))


def main():
    if hashlib.sha256(open(TC, "rb").read()).hexdigest() != PIN:
        sys.exit("!! STOP: tc_missed.json does not match its pin")
    d = json.load(open(TC))["novel"]
    sg, pc = d["sgdet"], d["predcls"]
    u1 = [r for r in sg["rows"] if r["bucket"] == "U1"]
    u = collections.Counter(key(r) for r in u1)
    sg_other = collections.Counter(key(r) for r in sg["rows"] if r["bucket"] != "U1")
    m = collections.Counter(key(r) for r in pc["rows"])
    pc_bucket = collections.defaultdict(list)
    for r in pc["rows"]:
        pc_bucket[key(r)].append(r["bucket"])

    exact_keys = [k for k in u if u[k] + sg_other[k] == 1 and m[k] <= 1]   # one known GT entry for this key
    ambiguous = [k for k in u if k not in exact_keys]
    matched_exact = sum(1 for k in exact_keys if m[k] == 0)
    missed_exact = len(exact_keys) - matched_exact
    lo = hi = 0                                       # bounds on the ambiguous keys
    for k in ambiguous:
        lo += u[k] - min(u[k], m[k])                  # PredCls misses fall on U1 rows first
        hi += u[k] - max(0, m[k] - sg_other[k])       # ... or on the other known rows first (SGDet hits unknown)
    out = dict(
        u1_relations=len(u1), predcls_missed_rows=len(pc["rows"]), novel_gt=d["sgdet"]["gt_total"],
        sgdet_hits=sg["gt_total"] - sg["missed"], predcls_hits=pc["gt_total"] - pc["missed"],
        exact_keys=len(exact_keys), matched_in_predcls_exact=matched_exact, missed_in_predcls_exact=missed_exact,
        ambiguous_keys=len(ambiguous), ambiguous_rows=sum(u[k] for k in ambiguous),
        matched_ambiguous_bounds=[lo, hi],
        matched_total_bounds=[matched_exact + lo, matched_exact + hi])
    # descriptive splits over the exact keys
    rows_by_key = {key(r): r for r in u1}
    split = collections.defaultdict(lambda: [0, 0])
    for k in exact_keys:
        r = rows_by_key[k]
        g = "involves novel" if (r["subject"] in NOVEL_OBJ or r["object"] in NOVEL_OBJ) else "base-only"
        split[g][0 if m[k] == 0 else 1] += 1
        split["long" if r["long"] else "short"][0 if m[k] == 0 else 1] += 1
    out["exact_by_group_matched_missed"] = {k: v for k, v in split.items()}
    out["predcls_bucket_of_missed_exact"] = dict(collections.Counter(pc_bucket[k][0] for k in exact_keys if m[k] == 1))
    lo_t, hi_t = out["matched_total_bounds"]
    out["sgdet_novel_recall_if_all_recovered"] = [(out["sgdet_hits"] + lo_t) / out["novel_gt"],
                                                  (out["sgdet_hits"] + hi_t) / out["novel_gt"]]
    out["sgdet_novel_recall_now"] = out["sgdet_hits"] / out["novel_gt"]
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
