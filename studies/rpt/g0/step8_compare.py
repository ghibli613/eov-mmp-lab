#!/usr/bin/env python3
"""RPT G0 step 8, on the laptop (A28): the machine's logits against session 3's cache_s3, pair by pair.

Bar: max |logit - cached logit| <= 1e-4 over every pair, clip and predicate column of the 5 videos.
Reported beside it, not bars: interactiveness-logit and slot-input residuals, the windows, and
whether each requested frame's global CLIP embedding equals the cached one (frame provenance).

    python studies/rpt/g0/step8_compare.py --machine studies/rpt/g0/box_out/step8 \
        --cache ../session3_box_copy/cache_s3/train --out studies/rpt/g0/logs/step8_compare.json
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
sys.path.insert(0, RPT)
from step8_reproduce import VIDEOS  # noqa: E402

BAR = 1e-4


def compare_video(new, old):
    import torch
    assert new["video"] == old["video"]
    keys = lambda ps, s, o: [(p[s], p[o], tuple(p["duration"])) for p in ps]
    order_new, order_old = keys(new["pairs"], "sub_tid", "obj_tid"), keys(old["pairs"], "sub", "obj")
    row = dict(video=new["video"], pairs=len(new["pairs"]), pairs_cached=len(old["pairs"]),
               same_pairs_in_same_order=order_new == order_old)
    if not row["same_pairs_in_same_order"]:
        row["first_difference"] = next(((i, a, b) for i, (a, b) in enumerate(zip(order_new, order_old)) if a != b), None)
        return row
    d = lambda a, b: float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    row["clips"] = sum(len(p["logits"]) for p in new["pairs"])
    row["max_abs_logit"] = max((d(p["logits"], q["logits"]) for p, q in zip(new["pairs"], old["pairs"])), default=0.0)
    row["max_abs_int_logit"] = max((d(p["int_logit"], q["int_logit"]) for p, q in zip(new["pairs"], old["pairs"])), default=0.0)
    row["max_abs_slot_input"] = {k: max(d(p["slot_inputs"][k], q["slot_inputs"][k]) for p, q in zip(new["pairs"], old["pairs"]))
                                 for k in old["pairs"][0]["slot_inputs"]} if old["pairs"] else {}
    row["windows_equal"] = all([tuple(w) for w in p["windows"]] == [tuple(w) for w in q["windows"]]
                               for p, q in zip(new["pairs"], old["pairs"]))
    gp = old["global_proj"]
    diffs = {int(f): float((g.reshape(-1).float() - gp[f].float()).abs().max()) for f, g in new["frames"].items()}
    row["frames_requested"] = len(diffs)
    row["frames_global_proj_identical"] = sum(1 for x in diffs.values() if x == 0.0)
    row["max_abs_global_proj"] = max(diffs.values(), default=0.0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import no_test_labels
    no_test_labels.install()
    import torch
    rows = []
    for v in VIDEOS:
        new = torch.load(os.path.join(a.machine, f"{v}.pt"), weights_only=False, map_location="cpu")
        old = torch.load(os.path.join(a.cache, f"{v}.pt"), weights_only=False, map_location="cpu")
        rows.append(compare_video(new, old))
        print(json.dumps(rows[-1]), flush=True)
    ok = all(r["same_pairs_in_same_order"] for r in rows)
    worst = max((r.get("max_abs_logit", float("inf")) for r in rows), default=float("inf"))
    res = dict(bar=BAR, max_abs_logit=worst if ok else None, verdict="PASS" if ok and worst <= BAR else "FAIL",
               videos=rows)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "videos"}))


if __name__ == "__main__":
    main()
