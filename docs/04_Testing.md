# The test suite

```bash
python -m pytest tests/ -q          # 55 tests, ~70 s
```

12 tests construct a model and need a GPU; they skip cleanly without one, and the
checkpoint tests skip if `output/ckpt/` is empty. So the suite runs on a laptop,
on the 4 GB development card, and on Colab.

---

## What each file covers

| File | Covers |
|---|---|
| `test_checkpoint_compat.py` | the released checkpoints load into the current model with **0 missing / 0 unexpected keys** (196 tensors for the relation classifier, 155 for the object classifier) |
| `test_feature_streams.py` | disabling an input stream is exactly equivalent to zeroing it, and does change the output; the geometry-only configuration constructs and runs |
| `test_inference_postprocess.py` | `process_pred` / `association` / `format_` — top-N selection, clip chaining, score products, `--max_per_video` truncation |
| `test_config.py` | the published splits (132 predicates 71/61, 35 objects 25/10), the paper's hyperparameters, and that every flag added during cleanup defaults to upstream behaviour |
| `test_imports.py` | every live module imports; nothing anywhere imports a module removed in the cleanup |

---

## Why these, specifically

The failures that matter in this codebase are **silent**.
`load_state_dict(strict=False)` returns a list of missing keys that nobody reads,
leaves those tensors randomly initialised, and costs a few mAP with no error
anywhere. A feature flag that stops gating anything does not raise either — it
makes an ablation meaningless while still producing a plausible-looking number.

None of these tests assert that the model is *good*. They assert that it is the
model you think it is.

`test_inference_postprocess.py` is deliberately a **characterisation** test: it
pins current behaviour, including behaviour worth changing, such as a one-clip
gap splitting a relation instance in two. Those are exactly the things
[91_Extension-guide.md](91_Extension-guide.md) §4 proposes changing — the tests
are there so the change arrives as a failing test you update on purpose, rather
than as an unexplained shift in the numbers three weeks later.

---

## Not covered

- **Reproducing the published mAP.** Needs 16 GB and hours; that belongs in a run
  log. See [10_Known-issues.md](10_Known-issues.md).
- **The vendored trees** (`third_party/`, `ops/`, `vlm/backbones/`) — testing
  them tests someone else's repository.
- **The detector**, which needs more VRAM to construct than the development
  machine has.

If you change `models/relation_classifier.py` or `inference/post_process.py` and
nothing fails, the missing test is the one worth writing *before* you make the
change — that is while you still know what the old behaviour was.
