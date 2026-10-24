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
        dst = os.path.join(REPO, e["dest"], e["name"])
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
