#!/usr/bin/env python
"""Upload a folder of large artefacts to HuggingFace, and emit a fetch manifest.

The counterpart to tools/hugging_download.py: this pushes, that pulls. Use it for
anything too big for git — pretrained weights, preprocessed frames, cached
features.

    hf auth login                               # once ("hf", not the retired huggingface-cli)

    # weights (the built-in preset knows where each file belongs)
    python tools/upload_to_hf.py --repo <user>/ov-vidvrd-weights

    # anything else: say where the files should land in a clone
    python tools/upload_to_hf.py --repo <user>/ov-vidvrd-frames \\
        --repo-type dataset --bundle ../_frames_bundle \\
        --dest data/vidvrd/frames --title "VidVRD frames, 336x336"

It hashes every file, writes MANIFEST.json carrying the repo's resolve URLs and
a sha256 per entry, generates a card, and uploads with upload_large_folder
(chunked and resumable — the detector checkpoint alone is 3.5 GB).

PRIVATE BY DEFAULT. The EOV-MMP weights were supplied privately by their author,
whose own Drive folder is not publicly shared. Publishing them is his call.
--public exists and warns first. A private repo works identically for you:
hugging_download.py reads a token from HF_TOKEN or the `hf auth login` cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_BUNDLE = os.path.join(os.path.dirname(REPO), "_upload_bundle")

#: Built-in layout for the EOV-MMP weights: where each file belongs in a clone,
#: and whether it is needed for training, evaluation, or both. Anything not
#: listed here falls back to --dest / --needed-for, or a --layout file.
WEIGHTS_LAYOUT = {
    "checkpoint_vidvrd0059_new_1e-5.pth": ("output/ckpt", "train"),
    "vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth": ("output/ckpt", "train"),
    "baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth":
        ("output/ckpt", "train"),
    "AFLink_epoch20.pth": ("output/ckpt", "always"),
    "baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth":
        ("output/ckpt", "eval"),
    "clip_L14_feat_vidvrd.pkl": ("data/vidvrd/data", "always"),
    "VidVRD_ECC_train.json": ("data/vidvrd/data", "always"),
    "VidVRD_ECC_test.json": ("data/vidvrd/data", "always"),
}

#: files that describe the upload rather than being part of it
SIDECARS = {"MANIFEST.json", "README.md", ".gitattributes"}

CARD = """---
license: other
license_name: see-original-authors
tags:
  - video-visual-relationship-detection
  - open-vocabulary
---

# {title}

For [ov-vidvrd-lab](https://github.com/{gh}) — open-vocabulary video visual
relationship detection.

{provenance}

## Contents

| File | Size | Needed for |
|---|---|---|
{table}

## Use

```bash
python tools/hugging_download.py --manifest {base}/MANIFEST.json
python tools/hugging_download.py --manifest ... --only eval   # skip train-only files
```

Every file is sha256-verified on download, and `hugging_download.py` places each
one at the path recorded in the manifest.
"""

WEIGHTS_PROVENANCE = """**These weights are not mine.** They were produced by the authors of *End-to-end
Open-vocabulary Video Visual Relationship Detection using Multi-modal Prompting*
(Wang, Wu, Yang & Luo, TPAMI 2025) and supplied by them directly. Cite that paper
for anything built on them. Original repository:
<https://github.com/wangyongqi558/EOV-MMP-VidVRD>"""

DERIVED_PROVENANCE = """Derived from **ImageNet-VidVRD** (Shang et al.). The processing code is in
[ov-vidvrd-lab](https://github.com/{gh}); cite the original dataset for the
underlying data."""


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_layout(args) -> dict:
    """filename -> (dest, needed_for), from --layout, the preset, or --dest."""
    if args.layout:
        with open(args.layout) as f:
            raw = json.load(f)
        return {k: (v["dest"], v.get("needed_for", "always")) for k, v in raw.items()}
    return dict(WEIGHTS_LAYOUT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="HuggingFace repo id, e.g. user/ov-vidvrd-weights")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE,
                    help=f"folder of files to upload (default: {DEFAULT_BUNDLE})")
    ap.add_argument("--repo-type", choices=["model", "dataset"], default="model",
                    help="'model' for weights (default), 'dataset' for data")
    ap.add_argument("--layout", default=None,
                    help='JSON mapping filename -> {"dest": ..., "needed_for": ...}, '
                         "overriding the built-in weights preset")
    ap.add_argument("--dest", default=None,
                    help="where files not covered by the layout should land in a "
                         "clone, e.g. data/vidvrd/frames")
    ap.add_argument("--needed-for", choices=["always", "train", "eval"], default="always",
                    help="tag for files not covered by the layout (default: always)")
    ap.add_argument("--title", default="EOV-MMP pretrained weights (VidVRD)",
                    help="heading for the generated card")
    ap.add_argument("--public", action="store_true",
                    help="create a PUBLIC repo. Confirm with the original author first.")
    ap.add_argument("--github", default="<you>/ov-vidvrd-lab",
                    help="GitHub slug for the card")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the manifest and card, upload nothing")
    args = ap.parse_args()

    if not os.path.isdir(args.bundle):
        raise SystemExit(f"bundle directory not found: {args.bundle}")

    layout = resolve_layout(args)
    names = [f for f in sorted(os.listdir(args.bundle))
             if f not in SIDECARS and os.path.isfile(os.path.join(args.bundle, f))]
    if not names:
        raise SystemExit(f"no files to upload in {args.bundle}")

    # Anything the layout does not name needs --dest, rather than being dropped
    # silently -- that would upload an empty repo and look like success.
    unmapped = [f for f in names if f not in layout]
    if unmapped and not args.dest:
        raise SystemExit(
            f"{len(unmapped)} file(s) are not in the layout and --dest was not given:\n"
            + "\n".join(f"  {f}" for f in unmapped[:8])
            + (f"\n  ... and {len(unmapped) - 8} more" if len(unmapped) > 8 else "")
            + "\n\nPass --dest <path in the clone>, or --layout <mapping.json>.")

    files = []
    total = sum(os.path.getsize(os.path.join(args.bundle, f)) for f in names)
    print(f"  {len(names)} file(s), {total/1e9:.2f} GB   -> {args.repo} ({args.repo_type})\n")
    for f in names:
        p = os.path.join(args.bundle, f)
        dest, needed = layout.get(f, (args.dest, args.needed_for))
        print(f"    hashing {f[:60]:60s}", flush=True)
        files.append({"name": f, "dest": dest, "needed_for": needed,
                      "bytes": os.path.getsize(p), "sha256": sha256(p)})

    # dataset repos resolve under /datasets/<id>; model repos have no prefix
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    base = f"https://huggingface.co/{prefix}{args.repo}/resolve/main"
    manifest = {"description": args.title,
                "source": "see the repository card",
                "base_url": base,
                "files": files}
    with open(os.path.join(args.bundle, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  wrote {os.path.join(args.bundle, 'MANIFEST.json')}")

    prov = (WEIGHTS_PROVENANCE if layout is not None and any(f in WEIGHTS_LAYOUT for f in names)
            else DERIVED_PROVENANCE.format(gh=args.github))
    table = "\n".join(
        f"| `{f['name'][:60]}` | {f['bytes']/1e9:.2f} GB | {f['needed_for']} |" for f in files)
    with open(os.path.join(args.bundle, "README.md"), "w") as fh:
        fh.write(CARD.format(title=args.title, gh=args.github, base=base,
                             provenance=prov, table=table))
    print(f"  wrote {os.path.join(args.bundle, 'README.md')} (card)")

    if args.dry_run:
        print("\n  --dry-run: nothing uploaded")
        print(f"  fetch with:\n    python tools/hugging_download.py --manifest {base}/MANIFEST.json")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    private = not args.public
    if args.public:
        print("\n  WARNING: creating a PUBLIC repo. If this includes the EOV-MMP\n"
              "  weights, they were shared privately -- confirm with the author first.")
    api.create_repo(args.repo, repo_type=args.repo_type, private=private, exist_ok=True)
    print(f"\n  uploading ({args.repo_type}, {'private' if private else 'PUBLIC'}) ...")
    # chunked and resumable -- the detector checkpoint alone is 3.5 GB
    api.upload_large_folder(folder_path=args.bundle, repo_id=args.repo,
                            repo_type=args.repo_type)

    print(f"\n  done: https://huggingface.co/{prefix}{args.repo}")
    print(f"  fetch with:\n    python tools/hugging_download.py --manifest {base}/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
