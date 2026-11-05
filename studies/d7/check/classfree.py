"""The adapter's class-free path (the D7 check). studies/rpt/adapter.py is pinned and unchanged; this module
reuses its rules for prompts that are not VidVRD class names ("object", "animal", "vehicle"; "person" is
treated the same way here):
- one masklet -> one track: adapter.build_track (gap filling, detected/propagated/filled status, the
  mean detected-frame score, the 0.5 / 0.1 tiers);
- duplicates across prompts: adapter.deduplicate (A29: mean mask IoU > 0.7, the higher score kept;
  same-prompt masklets never compared);
- the confident tracks in the repo's track format, but with the prompt's name as the category and no
  check that it is one of the 35 classes. The D7 check matches class-agnostically, so the name is
  never compared.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "rpt")
if RPT not in sys.path:
    sys.path.insert(0, RPT)
import adapter as A  # noqa: E402


def tracks_classfree(s3: dict) -> dict:
    """All of one video's masklets, any prompts -> dict(all, kept, duplicates, confident, repo)."""
    W, H = s3["width"], s3["height"]
    tracks = []
    for pid, p in enumerate(s3["prompts"]):
        for m in p["masklets"]:
            tracks.append(A.build_track(m, p["class"], pid, W, H))
    kept, removed = A.deduplicate(tracks)
    confident = sorted((t for t in kept if t["tier"] == "confident"), key=lambda t: (t["prompt_id"], t["obj_id"]))
    repo = [dict(tid=i, category=t["category"], score=float(t["score"]), begin_fid=int(t["begin_fid"]),
                 end_fid=int(t["end_fid"]), trajectory={int(f): list(b) for f, b in t["trajectory"].items()})
            for i, t in enumerate(confident)]
    return dict(all=tracks, kept=kept, duplicates=removed, confident=confident, repo=repo)


def top_k(repo: list, k: int) -> list:
    """The equal-budget cut: the k highest-scoring tracks (A29's track score; ties: lower tid), re-numbered."""
    keep = sorted(repo, key=lambda t: (-t["score"], t["tid"]))[:k]
    keep = sorted(keep, key=lambda t: t["tid"])
    return [dict(t, tid=i) for i, t in enumerate(keep)]
