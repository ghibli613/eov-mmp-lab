#!/usr/bin/env python3
"""Stage 2 Batch B, step 2 -- THE single test run (A8): runs 1-6, in order.

    python pilot_analysis/stage2/b_run.py              # the test run (after b_step1_test_parts.py)
    python pilot_analysis/stage2/b_run.py --dry-run    # identical code on HELD; reads no test file

All on the 200 test videos, PredCls on GT tracks and categories:
  1. baseline: EOV-MMP alone, novel and all; must reproduce 21.70 / 37.20 within
     +-0.05 (and the session-3 Task-4 scoring path exactly), else STOP (exit 3)
  2. method, novel split: sqrt(EOV x r') -- THE GATE: mAP >= 24.70
  3. method, all split
  4. control: shuffled recipes (A6 permutation), novel and all
  5. recipe alone (r'), novel
  6. mix EOV^(1-w) r'^w at w = 0.25, 0.75, novel and all
  + per-predicate novel AP, baseline vs method (the official function, one predicate at a time)
Run 7 (rewording) is b_run7_rewording.py.

r' = r / c^0.25 with the full learned part (model_full.pt) and c from
data/b_coverage_full.json; recipes = data/recipes_after_g1.json. A predicate with
no recipe (or nothing evaluable in the slot) gets the slot's mean r' over the
predicates COMPETING IN THAT SPLIT that have one. Videos without ECC take EOV-MMP
alone in every run. Nothing here is chosen from test numbers.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV
from a2_step1_alpha import recipes_after_g1

ALPHA, GATE, BASE_REF, TOL, CONTROL_MARGIN = 0.25, 24.70, dict(novel=21.70, all=37.20), 0.05, 1.0
GROUPS = dict(a=["above", "away", "beneath", "past", "toward"],
              b=["creep_next_to", "move_next_to", "run_next_to", "sit_next_to", "swim_next_to", "stop_next_to"],
              c=["bite", "feed", "fight", "hold", "kick"])


def load_inputs(dry, data):
    if not dry:
        t = dict(np.load(os.path.join(data, "b_test_slots.npz")))
        t["learned"] = np.load(os.path.join(data, "b_test_learned_full.npy"))
        pairs = json.load(open(os.path.join(data, "b_test_pairs.json")))
        return t, pairs, C.TEST_GT
    d = dict(np.load(os.path.join(data, "slots.npz")))
    H = np.where(d["subset"] == 2)[0]
    pos = -np.ones(len(d["subset"]), int)
    pos[H] = np.arange(len(H))
    t = {k: d[k][H] for k in ("meas", "vehicle", "logits", "int_logit")}
    t["has_ecc"] = np.ones(len(H), bool)
    t["learned"] = np.load(os.path.join(data, "learned_full.npy"))[H]
    hp = json.load(open(os.path.join(data, "held_pairs.json")))
    for ps in hp.values():
        for p in ps:
            p["rows"] = [int(pos[p["rows"][0]]), int(pos[p["rows"][1] - 1]) + 1]
    return t, hp, os.path.join(data, "held_gt.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    C.verify_frozen()
    data = os.path.join(C.OUT, "data")
    pre = "b_dry" if a.dry_run else "b"
    NOV = "base" if a.dry_run else "novel"          # the dry run stands HELD's base split in for novel
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    labels = json.load(open(os.path.join(data, "slots_meta.json")))["labels"]
    assert all(set(l) <= base for l in labels)       # 0 novel labels used in training
    names = EV.names()
    rec, dropped = recipes_after_g1(phi, data)
    saved = json.load(open(os.path.join(data, "recipes_after_g1.json")))
    assert all([list(x) for x in rec[p]] == saved["recipes"][p] for p in names) and not saved["dropped"]
    cov = json.load(open(os.path.join(data, "b_coverage_full.json")))
    assert cov["alpha"] == ALPHA
    c = np.array([np.nan if cov["coverage"][p] is None else cov["coverage"][p] for p in names])
    perm = json.load(open(os.path.join(C.OUT, "step0_selection.json")))["shuffled_recipes"]["receives_recipe_of"]

    t, pairs, gt = load_inputs(a.dry_run, data)
    N = len(t["int_logit"])
    fb = ~t["has_ecc"].astype(bool)
    L = t["learned"].astype(np.float64).copy()
    for ax, v in dropped:
        L[:, C.OUT_IDX[(ax, v)]] = np.nan
    R_raw = C.recipe_matrix(rec, names, t["meas"], t["vehicle"], L, fill=False)
    Rp_raw = R_raw / np.power(c, ALPHA)[None, :]
    Rc_raw = Rp_raw.copy()                            # control: each novel predicate takes its donor's r'
    for p, donor in perm.items():
        Rc_raw[:, names.index(p)] = Rp_raw[:, names.index(donor)]
    eov = EV.eov_task4(t, pairs)                       # = Task 4's slot_scores, bit for bit
    cols = {NOV: [i for i, p in enumerate(names) if split[p] == NOV], "all": list(range(132))}

    def filled(Rraw, s):
        """r' on the split's competing columns; NaN -> the slot's mean over competing predicates with a recipe."""
        X = np.zeros((N, 132))
        sub = Rraw[:, cols[s]]
        with np.errstate(invalid="ignore"):
            neutral = np.nanmean(sub, axis=1)
        X[:, cols[s]] = np.where(np.isnan(sub), neutral[:, None], sub)
        return X

    def scores(kind, s, w=0.5, Rraw=None):
        Rraw = Rp_raw if Rraw is None else Rraw
        r = filled(Rraw, s)
        S = eov if kind == "eov" else r if kind == "recipe" else eov ** (1 - w) * r ** w
        S = EV.only(S, cols[s])
        S[fb] = EV.only(eov, cols[s])[fb]             # videos without ECC: EOV-MMP alone
        assert not np.isnan(S).any()
        return S

    res = dict(dry_run=a.dry_run, novel_split_name=NOV, alpha=ALPHA, gate=GATE, fallback_slots=int(fb.sum()),
               fallback_videos=sorted({v for v, ps in pairs.items() for p in ps if fb[p["rows"][0]]}))
    out_path = os.path.join(data, f"{pre}_results.json")

    def ev(S, s, label):
        r = EV.evaluate(EV.predict(pairs, S), gt, s)
        print(f"  {label:34s} {s:6s} mAP {r['mAP']:6.2f}  R@50 {r['R50']:6.2f}  R@100 {r['R100']:6.2f}  "
              f"| {time.time() - t0:.0f}s", flush=True)
        return r

    # ---- run 1: baseline check
    print("run 1: baseline (EOV-MMP alone)")
    r1 = {s: ev(scores("eov", s), s, "EOV-MMP alone") for s in (NOV, "all")}
    if a.dry_run:
        ref = json.load(open(os.path.join(data, "a2_step3.json")))["rows"]["EOV-MMP alone"]["mAP"]
        r1["check"] = {NOV: dict(ref=ref, diff=r1[NOV]["mAP"] - ref)}
        ok = abs(r1[NOV]["mAP"] - ref) <= TOL
    else:
        import eval_from_cache as E4                 # session 3 Task 4's own scoring path
        cache = E4.load_cache(C.CACHE)
        from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
        r1["check"] = {}
        for s in ("novel", "all"):
            m4, _ = eval_relation_detection_openvoc(target_split_pred=s, prediction_results=E4.predict(cache, s))
            r1["check"][s] = dict(ref=BASE_REF[s], diff=r1[s]["mAP"] - BASE_REF[s], task4_path_mAP=100 * float(m4),
                                  same_as_task4_path=abs(100 * float(m4) - r1[s]["mAP"]) < 1e-9)
        del cache
        ok = all(abs(r1["check"][s]["diff"]) <= TOL and r1["check"][s]["same_as_task4_path"] for s in ("novel", "all"))
    r1["check_ok"] = bool(ok)
    res["run1"] = r1
    json.dump(res, open(out_path, "w"), indent=1)
    if not ok:
        print("!! run 1 does not reproduce the baseline -- STOP")
        sys.exit(3)

    # ---- run 2: THE GATE
    print("run 2: method, novel split -- the gate")
    r2 = {NOV: ev(scores("mix", NOV), NOV, "method sqrt(EOV x r')")}
    r2["gate"] = dict(threshold=GATE, mAP=r2[NOV]["mAP"], verdict="PASS" if r2[NOV]["mAP"] >= GATE else "FAIL")
    res["run2"] = r2
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"  GATE: {r2[NOV]['mAP']:.2f} vs {GATE} -> {r2['gate']['verdict']}", flush=True)

    print("run 3: method, all split")
    res["run3"] = {"all": ev(scores("mix", "all"), "all", "method sqrt(EOV x r')")}
    print("run 4: control (shuffled recipes)")
    r4 = {s: ev(scores("mix", s, Rraw=Rc_raw), s, "control sqrt(EOV x r'_shuffled)") for s in (NOV, "all")}
    gap = r2[NOV]["mAP"] - r4[NOV]["mAP"]
    r4["method_minus_control_novel"] = gap
    r4["verdict"] = ("within 1.0: the gain cannot be attributed to the recipes" if abs(gap) < CONTROL_MARGIN
                     else "the method beats the control by >= 1.0" if gap >= CONTROL_MARGIN else "the control beats the method by >= 1.0")
    res["run4"] = r4
    print("run 5: recipe alone")
    res["run5"] = {NOV: ev(scores("recipe", NOV), NOV, "r' alone")}
    print("run 6: mix weights")
    res["run6"] = {f"w={w}": {s: ev(scores("mix", s, w=w), s, f"EOV^{1 - w:g} r'^{w:g}") for s in (NOV, "all")}
                   for w in (0.25, 0.75)}
    json.dump(res, open(out_path, "w"), indent=1)

    # ---- per-predicate novel AP, baseline vs method (official function, one predicate per call)
    print("per-predicate AP")
    gtj = json.load(open(gt))
    count = {}
    for rels in gtj.values():
        for r in rels:
            count[r["triplet"][1]] = count.get(r["triplet"][1], 0) + 1
    del gtj
    one_dir = os.path.join(data, f"{pre}_split_one")
    os.makedirs(one_dir, exist_ok=True)
    preds = dict(baseline=EV.predict(pairs, scores("eov", NOV)), method=EV.predict(pairs, scores("mix", NOV)))
    per = {}
    for p in sorted(names[i] for i in cols[NOV]):
        n = count.get(p, 0)
        row = dict(n=n)
        if n:
            sp = EV.split_file(os.path.join(one_dir, f"{p}.json"), {p}, "one")
            for k, pr in preds.items():
                try:
                    row[k] = EV.evaluate(pr, gt, "one", sp)["mAP"]
                except (IndexError, ValueError):
                    row[k] = 0.0                       # no prediction of p in any video with its GT
                    row[k + "_no_prediction"] = True
        per[p] = row
    res["per_predicate"] = per
    grp = {}
    for g, members in list(GROUPS.items()) + [("d", [p for p in per if not any(p in m for m in GROUPS.values())])]:
        rows = [per[p] for p in members if p in per and per[p]["n"]]
        grp[g] = dict(predicates=[p for p in members if p in per], with_gt=len(rows),
                      **{f"{k}_{f}": (float(fn([r[k] for r in rows])) if rows else None)
                         for k in ("baseline", "method") for f, fn in (("mean", np.mean), ("median", np.median))})
    res["groups"] = grp
    np.save(os.path.join(data, f"{pre}_rprime_novel_test.npy" if not a.dry_run else f"{pre}_rprime_novel.npy"),
            filled(Rp_raw, NOV).astype(np.float32))
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out_path, "w"), indent=1)
    C.write_log(f"{pre}_run", dict(alpha=ALPHA, gate=GATE, base_ref=BASE_REF, tol=TOL, control_margin=CONTROL_MARGIN,
                                   post=C.POST, dry_run=a.dry_run),
                [os.path.join(data, f) for f in (["slots.npz", "learned_full.npy", "held_pairs.json", "held_gt.json"] if a.dry_run
                                                  else ["b_test_slots.npz", "b_test_learned_full.npy", "b_test_pairs.json"])]
                + [os.path.join(data, "b_coverage_full.json"), os.path.join(data, "recipes_after_g1.json"),
                   os.path.join(C.OUT, "step0_selection.json")],
                [out_path], extra=dict(seconds=res["seconds"]))
    print(f"done | {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
