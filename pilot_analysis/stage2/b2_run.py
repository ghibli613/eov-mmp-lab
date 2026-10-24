#!/usr/bin/env python3
"""Stage 2 Batch B2 (A9) -- recipes reorder within each predicate; EOV-MMP's score
values are kept. A SECOND ATTEMPT, designed after Batch B's test result.

    python pilot_analysis/stage2/b2_run.py --part1                    # Part 1: HELD, base predicates
    python pilot_analysis/stage2/b2_run.py --test --dry-run           # test path, on HELD (no test file)
    python pilot_analysis/stage2/b2_run.py --test --go B2-TEST        # Part 2: the one test run

Method (nothing fitted): per video, per predicate p with a recipe, take EOV(p, .)
over all the video's (pair, slot) units; order the units by the key
m = sqrt(EOV x r') (ties: higher EOV first, then lower unit index) and hand out
EOV(p, .)'s values, highest first, in that order. Predicates without a recipe, and
every predicate in a video without ECC, keep EOV-MMP's scores. Invariant (asserted
for every video and predicate): the multiset of final values equals EOV's exactly.
r' = r / c^0.25 as frozen in A8 (full learned part, recipes_after_g1.json,
b_coverage_full.json), NaN units filled with the slot's mean r' over the
predicates competing in the split that have one (A8's rule).

Control: the same reordering keyed by sqrt(EOV x r'_shuffled), A6's permutation
(novel predicates only, so on HELD's base split it is identical to B2 by
construction). Variant (test only, descriptive): keyed by r' alone.

EOV is session 3 Task 4's own per-pair slot_scores (bit-identical to Batch B run 1);
scores go through the repo's post-processing and eval_relation_detection_openvoc.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV
from a2_step1_alpha import recipes_after_g1

ALPHA, GATE, BASE_REF, TOL, CONTROL_MARGIN, PART1_MARGIN = 0.25, 24.70, dict(novel=21.70, all=37.20), 0.05, 1.0, 2.0
# Batch B's test-side outputs (batch_b_report.md section 11); verified only in --test mode,
# after GO B2-TEST -- Part 1 opens no test file.
TEST_FILES = {"b_test_slots.npz": "787f18f0376484978b934250e30fe37287b12c9ba77ad41dadc03bd47aa3b073",
              "b_test_learned_full.npy": "d08b75a960c45f656274da112d49a878992b12784e95d69f971c0acf6c1aa0a9",
              "b_test_pairs.json": "b4fd47899a502b27b9e2df30ad066e1c8cd18bd17042defeddf488f981c2a7e6",
              "b_rprime_novel_test.npy": "64fc3a8add4f55ed3f1828c53be3872e46b6898d5928ec0d92a33d9c734e5187"}
GROUPS = dict(a=["above", "away", "beneath", "past", "toward"],
              b=["creep_next_to", "move_next_to", "run_next_to", "sit_next_to", "swim_next_to", "stop_next_to"],
              c=["bite", "feed", "fight", "hold", "kick"])


def reorder(S, key, video_rows, cols, has_recipe, skip_videos=frozenset()):
    """B2. S: (N, 132) EOV on competing columns; key: (N, 132). Returns the reassigned
    scores and the invariant tally."""
    out = S.copy()
    checks = ok = 0
    for v, rows in video_rows.items():
        for j in cols:
            e = S[rows, j]
            if has_recipe[j] and v not in skip_videos:
                order = np.lexsort((np.arange(len(rows)), -e, -key[rows, j]))    # key desc, EOV desc, index asc
                out[rows[order], j] = np.sort(e)[::-1]
            checks += 1
            ok += bool(np.array_equal(np.sort(out[rows, j]), np.sort(e)))
        if v in skip_videos:
            ok_fb = np.array_equal(out[rows], S[rows])
            checks += 1
            ok += bool(ok_fb)
    return out, dict(checks=checks, held=ok, exact=checks == ok)


def load_inputs(mode, data):
    """-> (slots dict, pairs, gt path). HELD for part1 / dry run; Batch B's test files for test."""
    if mode == "test":
        for f, h in TEST_FILES.items():
            assert C.sha256(os.path.join(data, f)) == h, f"{f} does not match batch_b_report.md -- STOP"
        t = dict(np.load(os.path.join(data, "b_test_slots.npz")))
        t["learned"] = np.load(os.path.join(data, "b_test_learned_full.npy"))
        return t, json.load(open(os.path.join(data, "b_test_pairs.json"))), C.TEST_GT
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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--part1", action="store_true")
    g.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go", default="")
    a = ap.parse_args()
    data = os.path.join(C.OUT, "data")
    real_test = a.test and not a.dry_run
    if real_test:
        interim = os.path.join(C.OUT, "batch_b2_interim.md")
        if a.go != "B2-TEST" or not os.path.exists(interim) or "Part 1: PASS" not in open(interim).read():
            sys.exit("The test run needs Part 1 to have passed and the user's GO B2-TEST (--go B2-TEST).")
    t0 = time.time()
    C.verify_frozen()
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    labels = json.load(open(os.path.join(data, "slots_meta.json")))["labels"]
    assert all(set(l) <= base for l in labels)       # 0 novel labels used in training
    names = EV.names()
    rec, dropped = recipes_after_g1(phi, data)
    saved = json.load(open(os.path.join(data, "recipes_after_g1.json")))
    assert all([list(x) for x in rec[p]] == saved["recipes"][p] for p in names) and not saved["dropped"]
    has_recipe = np.array([bool(rec[p]) for p in names])
    cov = json.load(open(os.path.join(data, "b_coverage_full.json")))
    assert cov["alpha"] == ALPHA
    c = np.array([np.nan if cov["coverage"][p] is None else cov["coverage"][p] for p in names])
    perm = json.load(open(os.path.join(C.OUT, "step0_selection.json")))["shuffled_recipes"]["receives_recipe_of"]

    mode = "test" if real_test else "held"
    t, pairs, gt = load_inputs(mode, data)
    N = len(t["int_logit"])
    video_rows = {v: np.concatenate([np.arange(*p["rows"]) for p in ps]) for v, ps in pairs.items()}
    fb = ~t["has_ecc"].astype(bool)
    skip = {v for v, rows in video_rows.items() if fb[rows].any()}
    assert all(fb[video_rows[v]].all() for v in skip)
    L = t["learned"].astype(np.float64).copy()
    for ax, v in dropped:
        L[:, C.OUT_IDX[(ax, v)]] = np.nan
    Rp_raw = C.recipe_matrix(rec, names, t["meas"], t["vehicle"], L, fill=False) / np.power(c, ALPHA)[None, :]
    Rc_raw = Rp_raw.copy()
    for p, donor in perm.items():
        Rc_raw[:, names.index(p)] = Rp_raw[:, names.index(donor)]
    has_recipe_ctrl = has_recipe.copy()
    for p, donor in perm.items():
        has_recipe_ctrl[names.index(p)] = has_recipe[names.index(donor)]
    eov = EV.eov_task4(t, pairs)

    if a.part1:
        splits = ["base"]
    else:
        splits = ["base" if a.dry_run else "novel", "all"]
    cols = {s: ([i for i, p in enumerate(names) if split[p] == s] if s != "all" else list(range(132))) for s in splits}

    def filled(Rraw, s):
        X = np.zeros((N, 132))
        sub = Rraw[:, cols[s]]
        with np.errstate(invalid="ignore"):
            neutral = np.nanmean(sub, axis=1)
        X[:, cols[s]] = np.where(np.isnan(sub), neutral[:, None], sub)
        return X

    if real_test:     # Batch B's stored novel r' must be what this code computes
        assert np.array_equal(filled(Rp_raw, "novel").astype(np.float32),
                              np.load(os.path.join(data, "b_rprime_novel_test.npy")))

    pre = "b2_part1" if a.part1 else ("b2_dry" if a.dry_run else "b2_test")
    out_path = os.path.join(data, f"{pre}_results.json")
    res = dict(mode=pre, alpha=ALPHA, fallback_videos=sorted(skip))

    def ev(S, s, label):
        r = EV.evaluate(EV.predict(pairs, S), gt, s)
        print(f"  {label:36s} {s:6s} mAP {r['mAP']:6.2f}  R@50 {r['R50']:6.2f}  R@100 {r['R100']:6.2f}  "
              f"| {time.time() - t0:.0f}s", flush=True)
        return r

    def b2(s, key_kind="m", ctrl=False):
        S = EV.only(eov, cols[s])
        R = filled(Rc_raw if ctrl else Rp_raw, s)
        key = np.sqrt(S * R) if key_kind == "m" else R
        return reorder(S, key, video_rows, cols[s], has_recipe_ctrl if ctrl else has_recipe, skip)

    inv = {}
    if a.part1:
        print("Part 1 (HELD, base predicates compete)")
        S0 = EV.only(eov, cols["base"])
        res["eov"] = ev(S0, "base", "EOV-MMP alone")
        Sb, inv["b2"] = b2("base")
        res["b2"] = ev(Sb, "base", "B2 (key sqrt(EOV x r'))")
        Sc, inv["control"] = b2("base", ctrl=True)
        res["control"] = ev(Sc, "base", "control (shuffled recipes)")
        res["control_identical_to_b2"] = bool(np.array_equal(Sc, Sb))
        res["invariant"] = inv
        res["moved_values"] = int((Sb != S0).sum())
        ok = res["b2"]["mAP"] >= res["eov"]["mAP"] - PART1_MARGIN and all(x["exact"] for x in inv.values())
        res["pass_rule"] = f"B2 mAP >= EOV-alone mAP - {PART1_MARGIN} and the invariant everywhere"
        res["part1_verdict"] = "PASS" if ok else "FAIL"
        print(f"Part 1: {res['part1_verdict']} (B2 {res['b2']['mAP']:.2f} vs EOV {res['eov']['mAP']:.2f}; "
              f"invariant {[(k, v['held'], v['checks']) for k, v in inv.items()]})")
    else:
        NOV = splits[0]
        print("run 1: baseline (EOV-MMP alone)")
        r1 = {s: ev(EV.only(eov, cols[s]), s, "EOV-MMP alone") for s in splits}
        if a.dry_run:
            ref = json.load(open(os.path.join(data, "a2_step3.json")))["rows"]["EOV-MMP alone"]["mAP"]
            r1["check"] = {NOV: dict(ref=ref, diff=r1[NOV]["mAP"] - ref)}
            ok = abs(r1[NOV]["mAP"] - ref) <= TOL
        else:
            import eval_from_cache as E4
            from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
            cache = E4.load_cache(C.CACHE)
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
        print("run 2: B2, novel split -- the gate")
        Sb, inv["b2_" + NOV] = b2(NOV)
        x = ev(Sb, NOV, "B2 (key sqrt(EOV x r'))")
        base_m = r1[NOV]["mAP"]
        verdict = ("PASS: gate reached" if x["mAP"] >= GATE else
                   "helps, below the gate: method work ends" if x["mAP"] > BASE_REF["novel"] else
                   "negative result: method work ends")
        res["run2"] = {NOV: x, "gate": dict(threshold=GATE, baseline=base_m, mAP=x["mAP"], verdict=verdict)}
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"  GATE: {x['mAP']:.2f} vs {GATE} (baseline {BASE_REF['novel']}) -> {verdict}", flush=True)
        print("run 3: B2, all split")
        Sa, inv["b2_all"] = b2("all")
        res["run3"] = {"all": ev(Sa, "all", "B2 (key sqrt(EOV x r'))")}
        print("run 4: control (shuffled recipes, same reordering)")
        r4 = {}
        for s in splits:
            Sc, inv[f"control_{s}"] = b2(s, ctrl=True)
            r4[s] = ev(Sc, s, "control")
        gap = x["mAP"] - r4[NOV]["mAP"]
        r4["b2_minus_control_novel"] = gap
        r4["verdict"] = ("within 1.0: the effect cannot be attributed to the recipes" if abs(gap) < CONTROL_MARGIN
                         else "B2 beats the control by >= 1.0" if gap >= CONTROL_MARGIN else "the control beats B2 by >= 1.0")
        res["run4"] = r4
        print("run 5: reordering keyed by r' alone (descriptive)")
        r5 = {}
        for s in splits:
            Sr, inv[f"rprime_key_{s}"] = b2(s, key_kind="r")
            r5[s] = ev(Sr, s, "key r' alone")
        res["run5"] = r5
        res["invariant"] = inv
        json.dump(res, open(out_path, "w"), indent=1)

        print("per-predicate AP")
        gtj = json.load(open(gt))
        count = {}
        for rels in gtj.values():
            for r in rels:
                count[r["triplet"][1]] = count.get(r["triplet"][1], 0) + 1
        del gtj
        one_dir = os.path.join(data, f"{pre}_split_one")
        os.makedirs(one_dir, exist_ok=True)
        preds = dict(baseline=EV.predict(pairs, EV.only(eov, cols[NOV])), b2=EV.predict(pairs, Sb))
        per = {}
        for p in sorted(names[i] for i in cols[NOV]):
            row = dict(n=count.get(p, 0), has_recipe=bool(rec[p]))
            if row["n"]:
                sp = EV.split_file(os.path.join(one_dir, f"{p}.json"), {p}, "one")
                for k, pr in preds.items():
                    try:
                        row[k] = EV.evaluate(pr, gt, "one", sp)["mAP"]
                    except (IndexError, ValueError):
                        row[k] = 0.0
                        row[k + "_no_prediction"] = True
            per[p] = row
        res["per_predicate"] = per
        grp = {}
        for gname, members in list(GROUPS.items()) + [("d", [p for p in per if not any(p in m for m in GROUPS.values())])]:
            rows = [per[p] for p in members if p in per and per[p]["n"]]
            grp[gname] = dict(predicates=[p for p in members if p in per], with_gt=len(rows),
                              **{f"{k}_{f}": (float(fn([r[k] for r in rows])) if rows else None)
                                 for k in ("baseline", "b2") for f, fn in (("mean", np.mean), ("median", np.median))})
        res["groups"] = grp
    res["invariant_exact_everywhere"] = all(v["exact"] for v in inv.values())
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out_path, "w"), indent=1)
    ins = [os.path.join(data, f) for f in (list(TEST_FILES) if real_test else ["slots.npz", "learned_full.npy", "held_pairs.json", "held_gt.json"])]
    C.write_log(f"{pre}_run", dict(alpha=ALPHA, gate=GATE, base_ref=BASE_REF, tol=TOL, control_margin=CONTROL_MARGIN,
                                   part1_margin=PART1_MARGIN, post=C.POST, mode=pre),
                ins + [os.path.join(data, f) for f in ("b_coverage_full.json", "recipes_after_g1.json")]
                + [os.path.join(C.OUT, "step0_selection.json")], [out_path], extra=dict(seconds=res["seconds"]))
    if not res["invariant_exact_everywhere"]:
        sys.exit("!! the invariant failed -- STOP")
    print(f"done | {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
