# The authors' data-generation scripts

Part of the [knowledge base](README.md). The scripts themselves live in
[`tools/upstream_data_scripts/`](../tools/upstream_data_scripts).

Supplied by the EOV-MMP author alongside the checkpoints, **unmodified**. They
are vendored as *provenance*, not as part of the pipeline — nothing in this
repository imports or runs them.

> **These do not run as-is.** They expect the author's working directory
> (relative `./` paths into their `data/` folder), and some reference files or
> absolute paths that were never shipped. Read them to understand how the data
> was built; do not expect to execute them without edits.

Everything they produce is already downloaded by
[`tools/prepare_data.py`](../tools/prepare_data.py), so you should not need to.

| Script | Produces | Notes |
|---|---|---|
| `gen_prior.py` | `prior.pkl` | The predicate prior is a **35 × 35 × 132** tensor — subject class × object class × predicate — counted from the training annotations. Reads `object2id.json`, `predicate2id.json`. |
| `gen_trajs.py` | `test_object_trajectories_*.json` | Builds the alternative trajectory sets from raw detector output, scored with `video_object_evaluation.py`. Contains an absolute path (`/media/sda1/jixf/...`) that will not exist on your machine. |
| `video_object_evaluation.py` | — | Trajectory-level (not relation-level) evaluation, used by `gen_trajs.py`. |
| `gen_avg_feature.py` | averaged features | No file paths at module level; the shortest of the set. |
| `gen_tid.py` | — | Assigns sequential `tid` fields to a trajectory file in place. Thirteen lines. |
| `gen_vectors.py` | `object_vectors.pkl` | GoogleNews word2vec embeddings of the 35 object names, 300-d. Requires `GoogleNews-vectors-negative300.bin` (~3.4 GB, not shipped) and `gensim`. |

## What is *not* here

**No script builds `clip_L14_feat_vidvrd.pkl`.** The CLIP object bank's recipe
remains undocumented — we have the file itself (189,345 exemplars, float16,
unnormalised) but not the code that made it.
[`tools/build_clip_object_bank.py`](../tools/build_clip_object_bank.py) is our own
approximation, kept for datasets where no published bank exists. See
[`03_Data.md`](03_Data.md).

## Dead ends worth knowing

`object_vectors.pkl` — the output of `gen_vectors.py` — is **not read anywhere**
in this codebase. It is a leftover from the RePro-era approach, where word2vec
embeddings stood in for the semantic space that CLIP's text encoder now
provides. Included for completeness only.

Likewise `predicate_split.json`, `id2predicate_test.json` and
`id2predicate_testall.json` are unreferenced by any live code path.
