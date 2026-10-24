#!/usr/bin/env python3
"""Stage 2 Batch B, step 0 -- the training-side pieces A8 freezes. Reads no test
annotation, test cache file or test part score.

    python pilot_analysis/stage2/b_step0.py

a. c(p) for all 132 predicates with the full learned part (data/learned_full.npy,
   = model_full.pt's 3-seed mean, checked) over all FIT slots; must equal A2's
   a2_coverage_full.json.
b. Run 7's inputs: whether session 3 saved per-slot reworded scores; the CPU
   timing measured on training videos (data/b_run7_timing_cpu.json).
c. Test videos without ECC, from file names and ECC keys only.
Writes b_step0.json and data/b_coverage_full.json.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json, os

import numpy as np

import common as C
import evaluation as EV
from a2_step1_alpha import raw_recipes, recipes_after_g1

TEST_PAIRS_TASK4 = 1392          # session 3 Task 4 (task4_predcls.json "pairs")


def main():
    C.verify_frozen()
    data = os.path.join(C.OUT, "data")
    phi, split = C.load_phi()
    names = EV.names()
    rec, dropped = recipes_after_g1(phi, data)
    saved = json.load(open(os.path.join(data, "recipes_after_g1.json")))["recipes"]
    assert all([list(x) for x in rec[p]] == saved[p] for p in names)
    d = dict(np.load(os.path.join(data, "slots.npz")))
    P = np.load(os.path.join(data, "learned_full.npy"))
    fit = np.where(d["subset"] != 2)[0]
    c = C.coverage(raw_recipes(d, fit, P, rec, names, dropped))
    cov = {p: (None if np.isnan(c[j]) else float(c[j])) for j, p in enumerate(names)}
    assert cov == json.load(open(os.path.join(data, "a2_coverage_full.json")))
    cov_path = os.path.join(data, "b_coverage_full.json")
    json.dump(dict(learned_part="data/model_full.pt (3-seed mean)", fit_slots=len(fit), alpha=0.25, coverage=cov),
              open(cov_path, "w"), indent=1)

    # c. test videos without ECC: file names only
    anno_ids = sorted(f[:-5] for f in os.listdir(os.path.join(C.ROOT, "data", "vidvrd", "anno", "test")) if f.endswith(".json"))
    cache_ids = sorted(f[:-3] for f in os.listdir(os.path.join(C.CACHE, "test")) if f.endswith(".pt"))
    assert anno_ids == cache_ids and len(cache_ids) == 200
    ecc_keys = set(json.load(open(C.ECC_TEST)).keys())
    no_ecc = sorted(v for v in cache_ids if v not in ecc_keys)

    # b. run 7's inputs
    t5 = sorted(os.listdir(os.path.join(C.CACHE, "task5")))
    tim = json.load(open(os.path.join(data, "b_run7_timing_cpu.json")))
    est_h = (TEST_PAIRS_TASK4 * tim["seconds_per_window"] + tim["seconds_set_strings"]) * 5 / 3600
    out = dict(coverage_file=os.path.relpath(cov_path, C.ROOT), no_recipe=sorted(p for p in names if cov[p] is None),
               test_videos=len(cache_ids), test_videos_without_ecc=no_ecc,
               run7=dict(session3_task5_files=t5, reworded_per_slot_scores_saved=False,
                         cpu_timing=tim, test_pairs_task4=TEST_PAIRS_TASK4,
                         cpu_estimate_hours=round(est_h, 1), deferred=est_h > 2))
    path = os.path.join(C.OUT, "b_step0.json")
    json.dump(out, open(path, "w"), indent=1)
    C.write_log("b_step0", dict(alpha=0.25), [os.path.join(data, f) for f in ("slots.npz", "learned_full.npy",
                                                                               "recipes_after_g1.json", "a2_coverage_full.json",
                                                                               "b_run7_timing_cpu.json")] + [C.ECC_TEST],
                [path, cov_path])
    print(json.dumps({k: v for k, v in out.items() if k != "run7"} | {"run7_cpu_hours": out["run7"]["cpu_estimate_hours"]}, indent=1))


if __name__ == "__main__":
    main()
