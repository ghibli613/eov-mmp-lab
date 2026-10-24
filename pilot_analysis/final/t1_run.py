#!/usr/bin/env python3
"""Final test T1 (A17) -- type-consistency rules on the test set, PredCls and SGDet, one run.

    python pilot_analysis/final/t1_run.py --dry-run     # plumbing: PredCls path on HELD ('all' split),
                                                        # SGDet path label-free on 5 test videos (no evaluation)
    python pilot_analysis/final/t1_run.py --go          # THE run (after A17)

Rules (F2 3a, FIT only): stand, sit, lie, walk, run, creep, swim -> the subject must be
animate; stop -> a vehicle (subject types: scoring_rule.md's lists). Deletion: for
every slot of every pair, the logit of each predicate whose verb has a rule is set to
-inf (score exactly 0 through session 3's own slot_scores) when the pair's subject
type contradicts the rule -- both splits, base and novel predicates -- BEFORE the
repo's post-processing. Nothing else changes. PredCls: GT subject categories (the
cache, Task 4's path); SGDet: the tracker's categories (Task 6's cache and path).
All mAP from eval_relation_detection_openvoc with the shipped test GT.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, collections, json, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "span"))
import span_common as SC              # noqa: E402
from span_common import C             # noqa: E402
sys.path.insert(0, C.HERE)
sys.path.insert(0, os.path.join(C.PA, "stage1c"))
import evaluation as EV               # noqa: E402
import eval_from_cache as E4          # noqa: E402  (Task 4 / Task 6 scoring path)

REF = dict(predcls=dict(novel=21.70, all=37.20), sgdet=dict(novel=15.79, all=27.18))
TOL, METOR, CONTROL_MARGIN, CONTROL_SEEDS = 0.05, dict(novel=16.75, all=27.53), 0.3, (0, 1, 2, 3, 4)
RULES_FILE = os.path.join(C.PA, "f2", "data", "f2_part3.json")
KEEP = ("sbj_cls", "obj_cls", "sbj_scr", "obj_scr", "sbj_traj", "obj_traj", "duration", "logits", "int_logit")
SUMS = None


def load_pairs(sub, videos):
    """{video: [pair dicts]} from session-3 cache files, each checked against SHA256SUMS."""
    import torch
    global SUMS
    if SUMS is None:
        SUMS = dict(reversed(l.strip().split(None, 1)) for l in open(C.CACHE_SUMS))
    out = {}
    for v in videos:
        rel = f"{sub}/{v}.pt"
        f = os.path.join(C.BOX_COPY, rel)
        assert C.sha256(f) == SUMS[rel], rel
        d = torch.load(f, weights_only=False, map_location="cpu")
        out[v] = [{k: p[k] for k in KEEP} for p in d["pairs"]]
    return out


def write_log(name, config, inputs, outputs, extra=None):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rel = lambda p: os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
    rec = dict(step=name, time=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(sys.argv), config=config,
               env=C.env_info(), code={rel(os.path.join(HERE, f)): C.sha256(os.path.join(HERE, f))
                                       for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
               inputs={rel(p): C.sha256(p) for p in inputs}, outputs={rel(p): C.sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    json.dump(rec, open(os.path.join(HERE, "logs", f"{name}.json"), "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--go", action="store_true")
    a = ap.parse_args()
    import torch
    t0 = time.time()
    C.verify_frozen()
    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    phi, split = C.load_phi()
    base = {p for p in phi if split[p] == "base"}
    labels = json.load(open(os.path.join(SC.DATA2, "slots_meta.json")))["labels"]
    assert all(set(l) <= base for l in labels)                         # 0 novel labels used in training
    names = EV.names()
    rules = json.load(open(RULES_FILE))["rules"]["rules"]
    assert rules == {"stand": "animate", "run": "animate", "walk": "animate", "sit": "animate", "stop": "vehicle",
                     "lie": "animate", "creep": "animate", "swim": "animate"}
    verb = lambda p: (C.S0.tokenise(p)[0] if C.compositional(p) and len(C.S0.tokenise(p)) == 2 else None)
    ruled = {j: rules[verb(p)] for j, p in enumerate(names) if verb(p) in rules}
    novel_cols = [j for j, p in enumerate(names) if split[p] == "novel"]
    base_ids = [j for j, p in enumerate(names) if split[p] == "base"]

    def deleted_cols(cat):
        t = SC.subject_type(cat)
        return [j for j, typ in ruled.items() if t != typ]

    def with_rules(v, k, p):
        L = p["logits"].clone().float()
        cols = deleted_cols(p["sbj_cls"])
        if cols:
            L[:, cols] = float("-inf")
        return L

    def top20_novel(p):
        s = E4.slot_scores(p["logits"], p["int_logit"], "novel", base_ids).numpy()
        return [set(np.argsort(-s[t])[:20].tolist()) for t in range(len(s))]

    def evaluate(preds, s, gt):
        from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
        m, rec = eval_relation_detection_openvoc(target_split_pred=s, prediction_results=preds, gt_json=gt)
        return dict(mAP=100 * float(m), R50=100 * float(rec[50]), R100=100 * float(rec[100]))

    res = dict(rules=rules, ruled_predicates=sorted(names[j] for j in ruled), dry_run=a.dry_run)
    path = os.path.join(out_dir, "t1_dry_results.json" if a.dry_run else "t1_results.json")
    tests = {}
    if a.dry_run:
        held = sorted(json.load(open(os.path.join(SC.DATA2, "held_pairs.json"))))
        tests["predcls"] = (load_pairs("cache_s3/train", held), os.path.join(SC.DATA2, "held_gt.json"), ["all"])
        sv = sorted(f[:-3] for f in os.listdir(os.path.join(C.BOX_COPY, "cache_s3_sgdet/test")) if f.endswith(".pt"))[:5]
        sg = load_pairs("cache_s3_sgdet/test", sv)
        n_del = sum(len(set().union(*top20_novel(p)) & set(deleted_cols(p["sbj_cls"]))) > 0 for ps in sg.values() for p in ps)
        res["sgdet_plumbing"] = dict(videos=len(sv), pairs_with_a_deleted_top20_novel=int(n_del),
                                     predictions_with_rules=sum(len(x) for x in E4.predict(sg, "novel", logits_of=with_rules).values()))
    else:
        tv = sorted(f[:-3] for f in os.listdir(os.path.join(C.CACHE, "test")) if f.endswith(".pt"))
        sv = sorted(f[:-3] for f in os.listdir(os.path.join(C.BOX_COPY, "cache_s3_sgdet/test")) if f.endswith(".pt"))
        assert len(tv) == len(sv) == 200
        tests["predcls"] = (load_pairs("cache_s3/test", tv), C.TEST_GT, ["novel", "all"])
        tests["sgdet"] = (load_pairs("cache_s3_sgdet/test", sv), C.TEST_GT, ["novel", "all"])

    # ---- run 1: baseline check
    print("run 1: baseline check", flush=True)
    base_preds, ok = {}, True
    for task, (cache, gt, splits) in tests.items():
        for s in splits:
            base_preds[(task, s)] = E4.predict(cache, s)
            r = evaluate(base_preds[(task, s)], s, gt)
            ref = REF[task][s] if not a.dry_run else 45.66
            r["ref"], r["diff"] = ref, r["mAP"] - ref
            ok &= abs(r["diff"]) <= TOL
            res.setdefault("run1", {}).setdefault(task, {})[s] = r
            print(f"  {task} {s}: mAP {r['mAP']:.3f} (ref {ref}, diff {r['diff']:+.3f}) R@50 {r['R50']:.2f} R@100 {r['R100']:.2f}", flush=True)
    res["run1"]["check_ok"] = bool(ok)
    json.dump(res, open(path, "w"), indent=1, default=str)
    if not ok:
        print("!! baseline not reproduced -- STOP")
        sys.exit(3)

    # ---- runs 2-3: + rules
    rule_preds = {}
    for task, (cache, gt, splits) in tests.items():
        print(f"run {2 if task == 'predcls' else 3}: {task} + rules", flush=True)
        for s in splits:
            rule_preds[(task, s)] = E4.predict(cache, s, logits_of=with_rules)
            r = evaluate(rule_preds[(task, s)], s, gt)
            r["gain"] = r["mAP"] - res["run1"][task][s]["mAP"]
            res.setdefault("rules_run", {}).setdefault(task, {})[s] = r
            print(f"  {task} {s} + rules: mAP {r['mAP']:.3f} (gain {r['gain']:+.3f}) R@50 {r['R50']:.2f} R@100 {r['R100']:.2f}", flush=True)
    json.dump(res, open(path, "w"), indent=1, default=str)

    # ---- run 4: control -- per video, as many random top-20 novel deletions as the rules made
    # (the dry run exercises runs 4-5 on HELD with the 'all' split standing in for 'novel')
    NOV = "all" if a.dry_run else "novel"
    if True:
        print("run 4: control", flush=True)
        for task, (cache, gt, _) in tests.items():
            entries, n_rule, per_verb = {}, {}, collections.Counter()
            for v, ps in cache.items():
                ent, n = [], 0
                for k, p in enumerate(ps):
                    dc = set(deleted_cols(p["sbj_cls"]))
                    for t, top in enumerate(top20_novel(p)):
                        for j in sorted(top):
                            ent.append((k, t, j))
                            if j in dc:
                                n += 1; per_verb[verb(names[j])] += 1
                entries[v], n_rule[v] = ent, n
            vals = []
            for seed in CONTROL_SEEDS:
                rng = np.random.default_rng(seed)
                kill = {}
                for v in sorted(cache):
                    pick = rng.choice(len(entries[v]), size=n_rule[v], replace=False) if n_rule[v] else []
                    for i in pick:
                        k, t, j = entries[v][i]
                        kill.setdefault((v, k), []).append((t, j))

                def ctrl(v, k, p, kill=kill):
                    L = p["logits"].clone().float()
                    for t, j in kill.get((v, k), []):
                        L[t, j] = float("-inf")
                    return L
                vals.append(evaluate(E4.predict(cache, NOV, logits_of=ctrl), NOV, gt))
            m = {x: float(np.mean([r[x] for r in vals])) for x in ("mAP", "R50", "R100")}
            rules_m = res["rules_run"][task][NOV]["mAP"]
            res.setdefault("control", {})[task] = dict(per_seed=[r["mAP"] for r in vals], mean=m,
                                                      sd_mAP=float(np.std([r["mAP"] for r in vals])),
                                                      deletions_per_video_total=int(sum(n_rule.values())),
                                                      rules_minus_control=rules_m - m["mAP"],
                                                      attributable=abs(rules_m - m["mAP"]) >= CONTROL_MARGIN)
            res.setdefault("diagnostics", {}).setdefault(task, {})["deleted_top20_novel_per_verb"] = dict(per_verb)
            print(f"  {task} control novel: {m['mAP']:.3f} (sd {res['control'][task]['sd_mAP']:.3f}); "
                  f"rules - control {rules_m - m['mAP']:+.3f}", flush=True)
        json.dump(res, open(path, "w"), indent=1, default=str)

        # ---- run 5: diagnostics (test labels for reporting only)
        print("run 5: diagnostics", flush=True)
        from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
        gtj = json.load(open(tests["predcls"][1]))
        count = collections.Counter(r["triplet"][1] for g in gtj.values() for r in g)
        del gtj
        one_dir = os.path.join(out_dir, "split_one")
        os.makedirs(one_dir, exist_ok=True)
        for task, (cache, gt, _) in tests.items():
            bp = base_preds[(task, NOV)]
            eval_relation_detection_openvoc(target_split_pred=NOV, prediction_results=bp, gt_json=gt, rt_hit_infos=True)
            viol = [r for rs in bp.values() for r in rs
                    if r["triplet"][1] in {names[j] for j in ruled}
                    and SC.subject_type(r["triplet"][0]) != ruled[names.index(r["triplet"][1])]]
            res["diagnostics"][task]["baseline_output_predictions_breaking_a_rule"] = len(viol)
            res["diagnostics"][task]["of_which_true_positives_lost"] = int(sum(bool(r.get("hit_gt")) for r in viol))
            res["diagnostics"][task]["true_positives_lost_by_predicate"] = dict(collections.Counter(
                r["triplet"][1] for r in viol if r.get("hit_gt")))
            per = {}
            for p in sorted(names[j] for j in ruled if split[names[j]] == "novel"):
                row = dict(n=count.get(p, 0))
                if row["n"]:
                    sp = EV.split_file(os.path.join(one_dir, f"{p}.json"), {p}, "one")
                    for key, pr in (("baseline", bp), ("rules", rule_preds[(task, NOV)])):
                        try:
                            row[key] = EV.evaluate(pr, gt, "one", sp)["mAP"]
                        except (IndexError, ValueError):
                            row[key] = 0.0
                            row[key + "_no_prediction"] = True
                per[p] = row
            res["diagnostics"][task]["per_predicate_novel_ap"] = per
            print(f"  {task}: baseline outputs breaking a rule {len(viol)}, true positives among them "
                  f"{res['diagnostics'][task]['of_which_true_positives_lost']}", flush=True)

    if not a.dry_run:
        # ---- verdict
        sg, pc = res["rules_run"]["sgdet"], res["rules_run"]["predcls"]
        beats = sg["novel"]["mAP"] >= METOR["novel"] and sg["all"]["mAP"] >= METOR["all"]
        res["verdict"] = dict(
            metor=METOR, margins=dict(novel=sg["novel"]["mAP"] - METOR["novel"], all=sg["all"]["mAP"] - METOR["all"]),
            sgdet_gain=dict(novel=sg["novel"]["gain"], all=sg["all"]["gain"]),
            predcls_gain=dict(novel=pc["novel"]["gain"], all=pc["all"]["gain"]),
            line=("beats METOR" if beats else
                  "improves the baseline but not METOR" if sg["novel"]["gain"] > 0 or sg["all"]["gain"] > 0 else
                  "no gain, or worse"),
            control_attributable={t: res["control"][t]["attributable"] for t in ("predcls", "sgdet")})
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(path, "w"), indent=1, default=str)
    write_log("t1_dry" if a.dry_run else "t1_run",
              dict(ref=REF, tol=TOL, metor=METOR, control_margin=CONTROL_MARGIN, control_seeds=CONTROL_SEEDS, post=C.POST),
              [RULES_FILE, C.CACHE_SUMS], [path], extra=dict(seconds=res["seconds"]))
    print(f"done | {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
