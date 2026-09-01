#!/usr/bin/env python
"""Download large artefacts listed in a MANIFEST.json — weights, frames, caches.

The counterpart to tools/hugging_upload.py. Usually reached through
`tools/prepare_data.py --manifest <url>`, which runs it as its last step, but it
works standalone too:

    python tools/hugging_download.py --manifest https://huggingface.co/<repo>/resolve/main/MANIFEST.json
    python tools/hugging_download.py --manifest ... --only eval    # skip 4.5 GB
    python tools/hugging_download.py --manifest ... --check        # verify, download nothing

Private HuggingFace repos work: a token is read from --token, HF_TOKEN,
HUGGINGFACE_HUB_TOKEN, or the `hf auth login` cache, and attached only to
huggingface.co requests.

HuggingFace URLs are pulled with hf_hub_download (parallel range requests,
resumable, and hf_transfer-aware if that package is installed) rather than the
single urllib stream used for other hosts -- on a 2.58 GB checkpoint that is the
difference between minutes and most of an hour. Install `hf_transfer` and set
HF_HUB_ENABLE_HF_TRANSFER=1 to go faster still.

Each entry carries a sha256, so a truncated or corrupted download is caught
rather than surfacing later as an unexplained accuracy drop. Files already
present with the right hash are skipped, making this resumable.

`--only eval` fetches just what `--eval_only` needs: the trained end-to-end
checkpoint, the tracker, the CLIP bank and the ECC matrices. The three
step-1/2/3 component checkpoints (4.5 GB) are for training and are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import shutil
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _hf_token() -> str | None:
    """A HuggingFace token, if one is available.

    Private repos need it. Checked in order: HF_TOKEN, HUGGINGFACE_HUB_TOKEN,
    then the cache written by `hf auth login`.
    """
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    for path in (os.path.expanduser("~/.cache/huggingface/token"),
                 os.path.expanduser("~/.huggingface/token")):
        if os.path.exists(path):
            with open(path) as fh:
                tok = fh.read().strip()
            if tok:
                return tok
    return None


def _open(url: str):
    """Open a URL, attaching a HuggingFace token when the host wants one."""
    req = urllib.request.Request(url)
    if "huggingface.co" in url:
        tok = _hf_token()
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req)


def hf_file(repo_id: str, filename: str, repo_type: str = "dataset",
            local_dir: str | None = None) -> str:
    """One file from a HuggingFace repo. Returns the local path.

    Wraps hf_hub_download, which caches, resumes and handles auth. Imported
    lazily so the module stays usable without huggingface_hub installed.
    """
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, repo_type=repo_type, local_dir=local_dir)


def url_file(url: str, dst: str) -> str:
    """One file from a plain URL, skipped if already present. Returns dst."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    download(url, dst)
    return dst


#: Manifest `dest` values are repo-relative, but utils.paths honours
#: VIDVRD_DATA_ROOT / VIDVRD_OUTPUT_ROOT -- which Colab sets so data lands on the
#: session disk and output on Drive. Writing to REPO/<dest> then puts files where
#: nothing looks for them: the run dies on a missing clip_L14_feat_vidvrd.pkl even
#: though it downloaded fine. Route the known destinations through paths instead.
def _resolve_dest(dest: str) -> str:
    from utils import paths
    known = {
        "output/ckpt": paths.CKPT_DIR,
        "data/vidvrd/data": paths.META_DIR,
        "data/vidvrd/frames": paths.FRAME_DIR,
        "data/vidvrd/videos": paths.VIDEO_DIR,
        "data/gt_jsons": paths.GT_JSON_DIR,
    }
    d = known.get(dest.strip("/").replace("\\", "/"))
    return d if d else os.path.join(REPO, dest)


def _parse_hf_url(base: str):
    """A HuggingFace resolve URL -> (repo_id, repo_type), or None if not one.

    Layouts:
      https://huggingface.co/<owner>/<name>/resolve/<rev>            model
      https://huggingface.co/datasets/<owner>/<name>/resolve/<rev>   dataset
    """
    if "huggingface.co/" not in base or "/resolve/" not in base:
        return None
    tail = base.split("huggingface.co/", 1)[1].split("/resolve/", 1)[0]
    parts = tail.split("/")
    if parts and parts[0] in ("datasets", "spaces"):
        kind = {"datasets": "dataset", "spaces": "space"}[parts[0]]
        parts = parts[1:]
    else:
        kind = "model"
    if len(parts) != 2:
        return None
    return "/".join(parts), kind


def hf_fast_download(base: str, name: str, dst: str) -> bool:
    """Pull one file via hf_hub_download instead of urllib. Returns True if used.

    urllib.request.urlopen is a SINGLE stream with no parallelism, which on a
    2.58 GB checkpoint is the difference between minutes and the better part of an
    hour. hf_hub_download issues parallel range requests, resumes, and honours
    HF_HUB_ENABLE_HF_TRANSFER for the Rust downloader when it is installed.
    Falls back to urllib for anything that is not a HuggingFace resolve URL.
    """
    parsed = _parse_hf_url(base)
    if parsed is None:
        return False
    repo_id, repo_type = parsed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        cached = hf_hub_download(repo_id, name, repo_type=repo_type)
    except Exception as exc:            # auth, 404, offline -- let urllib try
        print(f"       (hf_hub_download failed: {type(exc).__name__}; "
              f"falling back to a single stream)")
        return False
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    shutil.copyfile(cached, tmp)        # the cache may be on another filesystem
    os.replace(tmp, dst)
    return True


def load_manifest(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        with _open(src) as r:
            return json.load(r)
    with open(src) as f:
        return json.load(f)


def wanted(entry: dict, only: str) -> bool:
    if only == "all":
        return True
    return entry["needed_for"] in ("always", only)


def download(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"          # never leave a half file at the real name
    with _open(url) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r      {pct:5.1f}%  {done/1e6:8.1f} / {total/1e6:.1f} MB",
                      end="", flush=True)
    print()
    os.replace(tmp, dst)


def fetch(manifest: str, only: str = "all", check: bool = False,
          base_url: str | None = None, token: str | None = None) -> int:
    """Download (or verify) everything in a manifest. Returns an exit code."""
    if token:
        os.environ["HF_TOKEN"] = token
    man = load_manifest(manifest)
    base = (base_url or man.get("base_url", "")).rstrip("/")
    entries = [e for e in man["files"] if wanted(e, only)]
    args = argparse.Namespace(check=check)

    if not check and not base:
        raise SystemExit("manifest has no base_url; pass --base-url")

    print(f"  {len(entries)} file(s), {sum(e['bytes'] for e in entries)/1e9:.2f} GB\n")
    missing = 0
    for e in entries:
        dst = os.path.join(_resolve_dest(e["dest"]), e["name"])
        label = f"{e['name'][:54]:54s}"
        if os.path.exists(dst) and os.path.getsize(dst) == e["bytes"]:
            if sha256(dst) == e["sha256"]:
                print(f"  OK   {label}")
                continue
            print(f"  BAD  {label}  sha256 mismatch")
            if args.check:
                missing += 1
                continue
        elif args.check:
            print(f"  MISS {label}")
            missing += 1
            continue

        print(f"  GET  {label}")
        if not hf_fast_download(base, e["name"], dst):
            download(f"{base}/{e['name']}", dst)
        got = sha256(dst)
        if got != e["sha256"]:
            os.remove(dst)
            raise SystemExit(
                f"  sha256 mismatch for {e['name']}\n"
                f"    expected {e['sha256']}\n    got      {got}\n"
                "  The file was deleted. Re-run to try again.")

    if check:
        print(f"\n  {missing} file(s) missing or corrupt")
        return 1 if missing else 0
    print("\n  All files present and verified.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="URL or path to MANIFEST.json")
    ap.add_argument("--only", choices=["all", "train", "eval"], default="all",
                    help="'eval' skips the 4.5 GB of step-1/2/3 training weights")
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    ap.add_argument("--base-url", default=None, help="override the manifest's base_url")
    ap.add_argument("--token", default=None,
                    help="HuggingFace token for a private repo "
                         "(else HF_TOKEN, or the `hf auth login` cache)")
    a = ap.parse_args()
    return fetch(a.manifest, a.only, a.check, a.base_url, a.token)


if __name__ == "__main__":
    sys.exit(main())
