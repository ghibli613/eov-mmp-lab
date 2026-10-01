# Refactor notes

Changes made to this repository that are not fixes, listed so the boundary
between "the authors' method" and "our scaffolding" stays legible. The rule
throughout: **nothing edits `forward()` or any model-semantic code.** Every
addition is input-side or observational.

See [docs/11_Port-status.md](docs/11_Port-status.md) for the original port's
file-by-file provenance and the fixes applied on the way in.

## `pilot_analysis/scripts/dump_predictions.py`

A wrapper around `cli/common.py:_predict_split`. Nothing in the repo changes.

| addition | what it does | model-semantic? |
|---|---|---|
| prediction dumps | writes post-merge and pre-merge predictions that the repo computes and discards | no |
| `--fuse-splits` | `_BothSplits` wraps `modelC` so one pipeline pass scores both predicate splits. Calls the real `modelC` twice with different `tgt_split`; returns the `all` triple so `end2end_model`'s unpacking is untouched. **Validated numerically identical** to two passes (§B.21) | no — same calls, fewer repeats |
| resume / flush / guards | crash-safety and a refusal to blend runs with different settings | no |

## `pilot_analysis/scripts/eval_at_threshold.py`

`eval_relation_detection_openvoc` hardcodes `viou_threshold=0.5` at both call
sites and exposes no parameter. This replicates its category filtering with the
threshold plumbed through. **The AP computation is still the repo's own
`evaluate()`** — required by the protocol's ground rule 2.

## `tools/`

`shard_frames.py` and `setup_rented_box.sh` are new and touch no model code.
`hugging_download.py` gained parallel HuggingFace transfers and destination
mapping through `utils.paths`; both are fixes, recorded in git rather than here.
