#!/usr/bin/env python3
"""Stage 2 Batch B, run 7 -- rewording robustness (A8).

    python pilot_analysis/stage2/b_run7_rewording.py --device cuda          # the run (after b_run.py)
    python pilot_analysis/stage2/b_run7_rewording.py --device cpu --timing-train 5   # CPU estimate, training videos

For the original strings (recomputed) and each of the 4 frozen rewording sets
(stage1b/paraphrases_frozen.json, SHA256 checked), EOV-MMP's test logits are
recomputed from the cached visual_pre_embeddings exactly as session 3 Task 5 did:
paraphrase_test.set_strings + paraphrase_test.recompute, unchanged. Then, in the
novel split: baseline = EOV' (sigmoid(logit') * sigmoid(interactiveness)); method =
sqrt(EOV' x r'), with r' the run's frozen novel-split recipe score (it does not
depend on wording); videos without ECC take EOV'. Novel mAP through the repo's
post-processing and eval_relation_detection_openvoc. Spread = best - worst over
the 5 conditions, per method.

--device cpu replaces modelC's hard-coded .cuda() calls with no-ops (CPU only).
--timing-train N: original strings only, on the first N training videos; reports
seconds per pair-window and the check against the cached logits. Reads no test file.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, glob, json, os, time

import numpy as np

import common as C
import evaluation as EV

STAGE1C = os.path.join(C.PA, "stage1c")
sys.path.insert(0, STAGE1C)


def cpu_patch():
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self


def load_model(device):
    import torch
    if device == "cpu":
        cpu_patch()
        torch.set_num_threads(6)
    from cache_wrapper import CKPT, build_modelC
    from vlm.backbones.clip import clip
    m, _, load = build_modelC(os.path.join(C.ROOT, CKPT))
    assert not load["missing"] and not load["unexpected"], load
    clip_model, _ = clip.load(name="ViT-L/14@336px", device="cpu")
    assert clip_model.dtype == torch.float32     # clip.load on cpu returns float32, as in session 3
    stub = torch.nn.Module()                     # only token_embedding and .dtype are used (set_strings);
    stub.conv1 = torch.nn.Conv2d(1, 1, 1)        # the visual tower is dropped to save memory
    clip_model.visual = stub
    clip_model = clip_model.cuda().eval()
    return m, clip_model


def recompute_video(m, pairs, recompute):
    return [recompute(m, p) for p in pairs]


def timing(n, device):
    import torch
    import paraphrase_test as PT
    m, clip_model = load_model(device)
    names = [m.pre_text_encoder.id2cls[str(i)] for i in range(132)]
    t = time.time()
    PT.set_strings(m, clip_model, names)
    t_set = time.time() - t
    files = sorted(glob.glob(os.path.join(C.CACHE, "train", "*.pt")))[:n]
    n_win, worst, t_rec = 0, 0.0, 0.0
    for f in files:
        d = torch.load(f, weights_only=False)
        for p in d["pairs"]:
            t = time.time()
            lg = PT.recompute(m, p)
            t_rec += time.time() - t
            n_win += len(p["windows"])
            worst = max(worst, float((lg - p["logits"]).abs().max()))
    per = t_rec / max(n_win, 1)
    out = dict(device=device, videos=n, windows=n_win, seconds_recompute=t_rec, seconds_per_window=per,
               seconds_set_strings=t_set, max_abs_logit_diff_vs_cache=worst)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--timing-train", type=int, default=0)
    a = ap.parse_args()
    if a.timing_train:
        r = timing(a.timing_train, a.device)
        json.dump(r, open(os.path.join(C.OUT, "data", f"b_run7_timing_{a.device}.json"), "w"), indent=1)
        return
    import torch
    import paraphrase_test as PT
    t0 = time.time()
    C.verify_frozen()
    sha = PT.precondition()                      # paraphrases_frozen.json must match its SHA256
    data = os.path.join(C.OUT, "data")
    res_main = json.load(open(os.path.join(data, "b_results.json")))
    assert res_main["run1"]["check_ok"], "run 1 failed its check: run 7 is not run"
    t = dict(np.load(os.path.join(data, "b_test_slots.npz")))
    pairs_meta = json.load(open(os.path.join(data, "b_test_pairs.json")))
    rprime = np.load(os.path.join(data, "b_rprime_novel_test.npy")).astype(np.float64)
    phi, split = C.load_phi()
    names = EV.names()
    novel_cols = [i for i, p in enumerate(names) if split[p] == "novel"]
    paras = json.load(open(C.PARAPHRASES))
    m, clip_model = load_model(a.device)
    assert [m.pre_text_encoder.id2cls[str(i)] for i in range(132)] == names
    conds = [("original (recomputed)", names)] + [(f"set {k + 1}", [paras[n][k] for n in names]) for k in range(4)]
    vids = sorted(pairs_meta)
    N = len(t["subset"]) if "subset" in t else len(t["int_logit"])
    rows = {}
    for label, strings in conds:
        tc = time.time()
        PT.set_strings(m, clip_model, strings)
        L = np.zeros((N, 132))
        for v in vids:
            d = torch.load(os.path.join(C.CACHE, "test", f"{v}.pt"), weights_only=False)
            for p, pm in zip(d["pairs"], pairs_meta[v]):
                r0, r1 = pm["rows"]
                L[r0:r1] = PT.recompute(m, p).double().numpy()
        eov = EV.eov_task4(dict(logits=L.astype(np.float32), int_logit=t["int_logit"]), pairs_meta)
        meth = np.sqrt(eov * rprime)
        meth[~t["has_ecc"].astype(bool)] = eov[~t["has_ecc"].astype(bool)]
        r = dict(baseline=EV.evaluate(EV.predict(pairs_meta, EV.only(eov, novel_cols)), C.TEST_GT, "novel"),
                 method=EV.evaluate(EV.predict(pairs_meta, EV.only(meth, novel_cols)), C.TEST_GT, "novel"))
        if label.startswith("original"):
            r["max_abs_logit_diff_vs_cache"] = float(np.abs(L - t["logits"]).max())
        rows[label] = r
        print(f"{label}: baseline novel {r['baseline']['mAP']:.2f} | method novel {r['method']['mAP']:.2f} "
              f"| {time.time() - tc:.0f}s", flush=True)
    spread = {k: max(rows[l][k]["mAP"] for l in rows) - min(rows[l][k]["mAP"] for l in rows) for k in ("baseline", "method")}
    out = dict(paraphrase_sha256=sha, device=a.device, conditions=rows, spread_novel_mAP=spread,
               cached_logits_reference=dict(baseline=res_main["run1"]["novel"]["mAP"], method=res_main["run2"]["novel"]["mAP"]))
    path = os.path.join(data, "b_run7.json")
    json.dump(out, open(path, "w"), indent=1)
    C.write_log("b_run7_rewording", dict(device=a.device, post=C.POST),
                [os.path.join(data, f) for f in ("b_results.json", "b_test_slots.npz", "b_test_pairs.json",
                                                 "b_rprime_novel_test.npy")] + [C.PARAPHRASES, C.CACHE_SUMS],
                [path], extra=dict(seconds=round(time.time() - t0, 1)))
    print(json.dumps(spread, indent=1))


if __name__ == "__main__":
    main()
