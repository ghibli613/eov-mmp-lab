#!/usr/bin/env python3
"""Session 3, Task 5 -- paraphrase stability (decision gate). GPU box.

    python pilot_analysis/stage1c/paraphrase_test.py --cache CACHE --out CACHE/task5

Precondition: stage1b/paraphrases_frozen.json matches paraphrases_frozen.sha256,
otherwise exit 5.

For each of the 4 frozen sets, every predicate string enters modelC's text side
where it entered originally (interface_audit.md B4(c)):
  * the learned prompt: "X x16 {name}." -> clip.tokenize -> token_embedding ->
    prompt_learner.token_prefix / token_suffix / tokenized_prompts;
  * the three fixed templates, via PredicateTextEncoder.build_clip_fixed_prompts.
Test logits are recomputed from the cached per-slot visual_pre_embeddings (the
per-pair prompt conditioning is recomputed from them by the model's own
meta_net), then post-processed and scored exactly as Task 4.

Built-in checks, both before any paraphrase is scored:
  1. rebuilding the buffers from the ORIGINAL strings reproduces the
     checkpoint's token_prefix / token_suffix and the init-time template
     embeddings exactly;
  2. logits recomputed with the original strings stay close to the cached ones
     (the cache stores visual_pre_embeddings in fp16, so not bit-exact).
The spread baseline is that RECOMPUTED original, so every condition goes
through the identical path; the cached-logit Task 4 numbers are reported too.

Reported (amendment A4) twice: over all predicates, and excluding the 19
predicates with known convention shifts (12 *_front for "ahead of", 8 stop_* for
"halts"; stop_front is in both). Excluding = removed from GT and predictions
before scoring.

Verdict on the novel-mAP spread (max - min over original + 4 sets):
>= 2.0 string-sensitive; < 0.5 robust; otherwise inconclusive.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import eval_from_cache as E  # noqa: E402

PARA = os.path.join(ROOT, "pilot_analysis", "stage1b", "paraphrases_frozen.json")
PARA_SHA = os.path.join(ROOT, "pilot_analysis", "stage1b", "paraphrases_frozen.sha256")
SENSITIVE, ROBUST, MOVE = 2.0, 0.5, 5.0


def precondition():
    want = open(PARA_SHA).read().split()[0]
    got = hashlib.sha256(open(PARA, "rb").read()).hexdigest()
    if want != got:
        sys.exit(5)
    return got


def a4_exclusions(names):
    ex = {n for n in names if n == "front" or n.endswith("_front") or n.startswith("stop_")}
    assert len(ex) == 19, sorted(ex)
    return sorted(ex)


def set_strings(m, clip_model, names):
    """Put 132 strings (id order) into every place the predicate name enters."""
    import torch
    from vlm.backbones.clip import clip
    pl = m.pre_classifier.prompt_learner
    cls = [n.replace("_", " ") for n in names]
    prefix = " ".join(["X"] * pl.n_ctx)
    tok = torch.cat([clip.tokenize(f"{prefix} {n}.") for n in cls]).cuda()
    with torch.no_grad():
        emb = clip_model.token_embedding(tok).type(clip_model.dtype)
        pl.token_prefix.copy_(emb[:, :1, :])
        pl.token_suffix.copy_(emb[:, 1 + pl.n_ctx:, :])
    pl.tokenized_prompts = tok
    m.pre_classifier.tokenized_prompts = tok
    te = m.pre_text_encoder
    with torch.no_grad():
        te.sbj_classifier_weights.copy_(te.build_clip_fixed_prompts("subject", names))
        te.obj_classifier_weights.copy_(te.build_clip_fixed_prompts("object", names))
        te.pre_classifier_weights.copy_(te.build_clip_fixed_prompts("predicate", names))


def recompute(m, pair):
    """Logits from the cached visual_pre_embeddings, window by window (each
    window was one modelC call, with its own prompt conditioning)."""
    import torch
    out = []
    with torch.no_grad():
        for a, b in pair["windows"]:
            v = pair["vpe"][a:b].float().cuda().unsqueeze(0)
            text = m.pre_classifier(v).squeeze(0)
            T, _ = m.split_text_embeddings("all", pre_classifier_weights=text)
            out.append((v[0] @ T.t() / m.temperature).float().cpu())
    return torch.cat(out, 0)


_PREDS = None


def _ap_one(args):
    split, c = args
    from inference.video_relation_detection_openvoc import evaluate
    g, p = E._filtered(_PREDS, split, only=c)
    n = sum(len(r) for r in g.values())
    if not n:
        return c, None
    if not any(p.get(v) for v in g):
        return c, (0.0, n)
    try:
        mm, _, _ = evaluate(g, p, viou_threshold=0.5)
        return c, (100 * float(mm), n)
    except IndexError:
        return c, (0.0, n)


def per_predicate(preds, split, names, workers):
    global _PREDS
    _PREDS = preds
    import json as _j
    from utils import paths
    sp = _j.load(open(paths.PRED_SPLIT_INFO))["cls2split"]
    want = [(split, c) for c in names if split == "all" or sp[c] == split]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return {c: r for c, r in ex.map(_ap_one, want) if r is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sha = precondition()
    import torch
    from cache_wrapper import CKPT, build_modelC
    from vlm.backbones.clip import clip

    paras = json.load(open(PARA))
    m, _, _ = build_modelC(os.path.join(ROOT, CKPT))
    names = [m.pre_text_encoder.id2cls[str(i)] for i in range(132)]
    exclude = a4_exclusions(names)
    clip_model, _ = clip.load(name="ViT-L/14@336px", device="cpu")
    clip_model = clip_model.cuda().eval()

    # ---- check 1: the original strings rebuild the original buffers exactly
    pl, te = m.pre_classifier.prompt_learner, m.pre_text_encoder
    ref = {k: getattr(pl, k).clone() for k in ("token_prefix", "token_suffix")}
    ref.update({k: getattr(te, k).clone() for k in ("sbj_classifier_weights", "obj_classifier_weights",
                                                     "pre_classifier_weights")})
    set_strings(m, clip_model, names)
    check1 = {k: float((getattr(pl if k.startswith("token") else te, k) - v).abs().max()) for k, v in ref.items()}
    print("check 1 (rebuild == original buffers):", check1, flush=True)
    if max(check1.values()) > 1e-5:
        sys.exit("!! check 1 failed: the string-rebuild path does not reproduce the original buffers")

    cache = E.load_cache(a.cache)
    conditions = [("original (recomputed)", names)] + [
        (f"set {k + 1}", [paras[n][k] for n in names]) for k in range(4)]
    results, ppa = {}, {}
    for label, strings in conditions:
        t0 = time.time()
        set_strings(m, clip_model, strings)
        lg = {(v, k): recompute(m, p) for v, ps in cache.items() for k, p in enumerate(ps)}
        if label.startswith("original"):
            diff = max(float((lg[(v, k)] - p["logits"]).abs().max()) for v, ps in cache.items() for k, p in enumerate(ps))
            results["check2_max_abs_logit_diff_vs_cache"] = diff
            print(f"check 2 (recomputed original vs cached logits): max |diff| {diff:.4f}", flush=True)
        r = {}
        for split in ("novel", "all"):
            preds = E.predict(cache, split, logits_of=lambda v, k, p: lg[(v, k)])
            r[split] = E.score(preds, split)
            r[split + "_excl19"] = E.score(preds, split, exclude=exclude)
            ppa[(label, split)] = per_predicate(preds, split, names, a.workers)
        results[label] = r
        print(f"{label}: novel {r['novel']['mAP']:.2f} (excl19 {r['novel_excl19']['mAP']:.2f}) | "
              f"all {r['all']['mAP']:.2f} (excl19 {r['all_excl19']['mAP']:.2f}) | {time.time() - t0:.0f}s", flush=True)

    # cached-logit reference (= Task 4)
    for split in ("novel", "all"):
        preds = E.predict(cache, split)
        results.setdefault("cached logits (Task 4)", {})[split] = E.score(preds, split)
        results["cached logits (Task 4)"][split + "_excl19"] = E.score(preds, split, exclude=exclude)

    labels = [c for c, _ in conditions]
    spread = {}
    for key in ("novel", "novel_excl19", "all", "all_excl19"):
        vals = [results[l][key]["mAP"] for l in labels]
        sets = vals[1:]
        spread[key] = dict(with_original=max(vals) - min(vals), sets_only=max(sets) - min(sets))

    def verdict(s):
        return "string-sensitive" if s >= SENSITIVE else "robust" if s < ROBUST else "inconclusive"
    verdicts = {k: verdict(spread[k]["with_original"]) for k in ("novel", "novel_excl19")}

    rows = []
    for split in ("all", "novel"):
        for c in names:
            vals = [ppa[(l, split)].get(c) for l in labels]
            if any(x is None for x in vals):
                continue
            aps = [x[0] for x in vals]
            rows.append(dict(split=split, predicate=c, gt_instances=vals[0][1], ap=aps,
                             spread=max(aps) - min(aps), moved_over_5=max(aps) - min(aps) > MOVE,
                             a4_excluded=c in exclude))
    out = dict(paraphrase_sha256=sha, conditions=labels, results=results, spread=spread,
               verdict=verdicts, thresholds=dict(sensitive=SENSITIVE, robust=ROBUST, move=MOVE),
               a4_excluded=exclude, per_predicate=rows)
    json.dump(out, open(os.path.join(a.out, "task5_paraphrase.json"), "w"), indent=1)
    print(json.dumps(dict(spread=spread, verdict=verdicts,
                          moved_over_5=sorted({r["predicate"] for r in rows if r["moved_over_5"]})), indent=1))


if __name__ == "__main__":
    main()
