#!/usr/bin/env python3
"""Post-pin correction to q7_headroom.py (A25e), disclosed in review_1.md. DESCRIPTIVE ONLY.

The pinned script's upper bound for ambiguous keys, u - max(0, m - s), lacks a floor at 0. It
also assumes that no SGDet hit shares a key with a U1 relation. That assumption fails for one
key (m > u + s), which made its interval come out inverted. Here: the same bounds on the keys
where the assumption is consistent (m <= u + s), and the assumption-free interval [u - min(u, m), u]
on the keys where it is not. Same single input: D1's tc_missed.json, SHA256-checked.
"""
import collections
import hashlib
import json
import os
import sys

TC = os.path.expanduser("~/Documents/thesis/source/ov-vidvrd-lab/pilot_analysis/time_check/data/tc_missed.json")
PIN = "2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q7_bounds_note.json")


def main():
    if hashlib.sha256(open(TC, "rb").read()).hexdigest() != PIN:
        sys.exit("!! STOP: tc_missed.json does not match its pin")
    d = json.load(open(TC))["novel"]
    key = lambda r: (r["video"], r["subject"], r["predicate"], r["object"], tuple(r["duration"]))
    u = collections.Counter(key(r) for r in d["sgdet"]["rows"] if r["bucket"] == "U1")
    s = collections.Counter(key(r) for r in d["sgdet"]["rows"] if r["bucket"] != "U1")
    m = collections.Counter(key(r) for r in d["predcls"]["rows"])
    ok = [k for k in u if m[k] <= u[k] + s[k]]
    bad = [k for k in u if m[k] > u[k] + s[k]]
    lo = sum(u[k] - min(u[k], m[k]) for k in ok) + sum(u[k] - min(u[k], m[k]) for k in bad)
    hi = sum(u[k] - max(0, m[k] - s[k]) for k in ok) + sum(u[k] for k in bad)
    out = dict(keys=len(u), consistent_keys=len(ok), inconsistent_keys=[list(k[1:4]) + [list(k[4])] for k in bad],
               inconsistent_rows=sum(u[k] for k in bad), matched_in_predcls_bounds=[lo, hi], of=sum(u.values()),
               sgdet_novel_recall_now=121 / 605, sgdet_novel_recall_if_all_recovered=[(121 + lo) / 605, (121 + hi) / 605],
               predcls_novel_recall=244 / 605)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
