#!/usr/bin/env python3
"""Session 3, Task 3 -- the full B6 cache over all 1,000 videos, streaming frames.

    python pilot_analysis/stage1c/run_task3.py --out CACHE --shards SHARDS_URL
    python pilot_analysis/stage1c/run_task3.py --out SCRATCH --shards LOCAL/SHARDS.json \
        --frame-root SCRATCH/frames --dry-run --limit 2          # local plumbing test

One model load, then per video: get its frames -> cache it -> (train) delete
its frames. Test videos first (Tasks 4-6 need them; their frames are KEPT for
Task 6's detector pass), then train. Disk stays small: one train video's frames
at a time (the longest is 5,492 frames / ~1.8 GB).

Frames: test from the private bundle (the pilot's exact inputs); train decoded
here with tools/extract_frames.py's own function (cv2, JPEG q95) -- byte-identical
to the bundle on this box (Task 2 provenance check).

Resumable: a video whose manifest entry exists and whose file's SHA256 matches
is skipped. A failing video is logged to failures.jsonl and skipped; the run
exits 4 at the end if any video failed, so the orchestrator stops before Task 4.
`--dry-run` replaces the model with a stub that only checks the frames.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, shutil, tarfile, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


class Frames:
    def __init__(self, root, shards):
        from tools.hugging_download import load_manifest
        self.root = root
        idx = load_manifest(shards)
        if "://" in shards:
            man = load_manifest(shards.rsplit("/", 1)[0] + "/MANIFEST.json")
            self.url = lambda name: f"{man['base_url']}/{name}"
        else:
            self.url = lambda name: os.path.join(os.path.dirname(shards), name)
        self.by_video = {}
        for s in idx["shards"]:
            self.by_video.setdefault(s["video_id"], []).append(s)
        import tools.extract_frames as X
        X.FRAME_DIR = root                      # extract_one writes under FRAME_DIR
        self.X = X

    def complete(self, v, n):
        d = os.path.join(self.root, v)
        return os.path.isdir(d) and len(os.listdir(d)) == n

    def get(self, v, split, n):
        if self.complete(v, n):
            return
        if split == "test":
            from tools.hugging_download import download, sha256
            staging = os.path.join(self.root, "_staging")
            os.makedirs(staging, exist_ok=True)
            for sh in self.by_video[v]:
                dst = os.path.join(staging, sh["tar"])
                src = self.url(sh["tar"])
                if "://" in src:
                    download(src, dst)
                else:
                    shutil.copy2(src, dst)
                if sha256(dst) != sh["sha256"]:
                    raise RuntimeError(f"sha256 mismatch on {sh['tar']}")
                with tarfile.open(dst) as tf:
                    try:
                        tf.extractall(self.root, filter="data")
                    except TypeError:
                        tf.extractall(self.root)
                os.remove(dst)
        else:
            got, status = self.X.extract_one(v, n, quality=95)
            if status not in ("ok", "skip"):
                raise RuntimeError(f"decode {v}: {status}")
        if not self.complete(v, n):
            raise RuntimeError(f"frames incomplete for {v}")

    def drop(self, v):
        shutil.rmtree(os.path.join(self.root, v), ignore_errors=True)


def stub_cache(video, split, tracks, frame_dir, out_path):
    """--dry-run: no model; check frames and write a small placeholder."""
    from cache_wrapper import sha256
    n = len(os.listdir(os.path.join(frame_dir, video)))
    json.dump(dict(video=video, frames=n, tracks=len(tracks)), open(out_path, "w"))
    return dict(video=video, split=split, seconds=0.0, encode_seconds=0.0, frames=n,
                tracks=len(tracks), pairs=0, slots=0, split_pairs=[], fidelity_max_residual=0.0,
                bytes=os.path.getsize(out_path), sha256=sha256(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--frame-root", default=None)
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--limit", type=int, default=0, help="per split (testing only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from cache_wrapper import CKPT, build_modelC, cache_video, save_once, sha256
    from gt_tracks import load_tracks
    from session3_tasks import frame_count
    from utils import paths

    root = a.frame_root or paths.FRAME_DIR
    os.makedirs(root, exist_ok=True)
    frames = Frames(root, a.shards)
    m = None
    if not a.dry_run:
        m, _, load = build_modelC(os.path.join(ROOT, CKPT))
        json.dump(load, open(os.path.join(a.out, "modelC_load.json"), "w"), indent=1)
        if load["missing"] or load["unexpected"]:
            sys.exit(f"!! modelC load not clean: {load}")
        if not os.path.exists(os.path.join(a.out, "once.pt")):
            save_once(m, os.path.join(a.out, "once.pt"))

    man_path = os.path.join(a.out, "manifest.jsonl")
    done = {}
    if os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line)
            done[(r["split"], r["video"])] = r
    todo = []
    for split in a.splits:
        vids = sorted(load_tracks(split).items())
        todo += [(split, v, t) for v, t in (vids[:a.limit] if a.limit else vids)]
    failures, t0, n_new = [], time.time(), 0
    for i, (split, v, tracks) in enumerate(todo, 1):
        out = os.path.join(a.out, split, f"{v}.pt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        prev = done.get((split, v))
        if prev and os.path.exists(out) and sha256(out) == prev["sha256"]:
            continue
        try:
            frames.get(v, split, frame_count(v))
            rec = (stub_cache(v, split, tracks, root, out) if a.dry_run
                   else cache_video(m, v, split, tracks, root, out))
            if rec["fidelity_max_residual"] > 1e-4:
                raise RuntimeError(f"fidelity residual {rec['fidelity_max_residual']:.2e} > 1e-4")
            with open(man_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_new += 1
            el = time.time() - t0
            left = len(todo) - i
            print(f"[{i}/{len(todo)}] {split} {v} frames={rec['frames']} pairs={rec['pairs']} "
                  f"{rec['seconds']:.1f}s residual={rec['fidelity_max_residual']:.1e} | "
                  f"elapsed {el / 60:.1f} min, ETA {el / n_new * left / 60:.0f} min", flush=True)
        except Exception as e:
            failures.append(dict(split=split, video=v, error=repr(e), trace=traceback.format_exc()))
            with open(os.path.join(a.out, "failures.jsonl"), "a") as f:
                f.write(json.dumps(failures[-1]) + "\n")
            print(f"[{i}/{len(todo)}] !! FAILED {split} {v}: {e!r}", flush=True)
        finally:
            if split == "train":
                frames.drop(v)
            if m is not None:
                import torch
                torch.cuda.empty_cache()
    print(f"TASK3 DONE: {n_new} cached now, {len(failures)} failed, {(time.time() - t0) / 3600:.2f} h")
    if failures:
        sys.exit(4)


if __name__ == "__main__":
    main()
