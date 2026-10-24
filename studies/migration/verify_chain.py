#!/usr/bin/env python3
"""Recompute the pre-registration chain from its new place in ov-vidvrd-lab (A26m).

Read-only. Prints one line per check and exits non-zero if any check fails.

    cd ov-vidvrd-lab/studies && python3 migration/verify_chain.py

1. Hash chains. The pilot file (A1-A22, pilot_analysis/stage1b/) and this chain's file (A23 on,
   studies/prereg/). Every line of each preregistration.sha256 is recomputed from the file
   itself: the current hash, and each earlier state as the bytes before the named amendment.
   Every preregistration.after-*.bak snapshot must equal its state, and every
   preregistration.sha256.before-*.bak must equal the hash list as it stood then.
2. Links. This file's header cites the pilot file's final hash. A25 cites the closed ov-pairtime
   file (M1, A24-A24i; hashed here, nothing else) and its base, this file after A23a. The state
   after A26 must be the head METOR handed over.
3. Moved files. Every row of migration/manifest.tsv has its SHA256 at its new path.
4. Pins. The structured pin files are checked at the new paths, and the EOV-MMP front-end pins
   (d2/logs/code_pins.json) against the repo root. Then the transitive closure of every SHA256 the
   chain cites (in full, or abbreviated as a…b) is recomputed over studies/. The moved files it
   reaches must be exactly those it reached in METOR (the manifest's "chain" column).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.dirname(HERE)
REPO = os.path.dirname(STUDIES)                              # ov-vidvrd-lab
THESIS = os.path.dirname(REPO)
METOR_CODE = os.path.join(STUDIES, "metor_code")
PILOT = os.path.join(REPO, "pilot_analysis", "stage1b")
PREREG = os.path.join(STUDIES, "prereg")
PAIRTIME = os.path.join(THESIS, "ov-pairtime", "prereg", "preregistration.md")
MANIFEST = os.path.join(HERE, "manifest.tsv")

PILOT_FINAL = "442623b82cf5392c480e889ab160d5e0606300acc9d588d338e0a806036211f0"     # after A22
AFTER_A23A = "bdfe25e62b20d3d82c743c848197d3448aff9d40fe6e98efdac3fd6134af268b"      # ov-pairtime's base
PAIRTIME_FINAL = "981bf067da43c41df8759a260c9bde61014b411809e13bf9838f62f6906b5ce5"  # after A24i (cited by A25)
A26_FINAL = "9485e1a8f72971db328b7c8a71a1c3ff96c301b3701a42ab8853b11e8559c494"       # METOR's head, after A26

FULL = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
ABBR = re.compile(r"(?<![0-9a-f])([0-9a-f]{6,20})(?:…|\.\.\.)([0-9a-f]{3,20})(?![0-9a-f])")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
LOCAL_LINKS = (os.path.join(METOR_CODE, "data"), os.path.join(METOR_CODE, "output"))

failures: list[str] = []


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fsha(p: str) -> str:
    with open(p, "rb") as f:
        return sha(f.read())


def rel(p: str) -> str:
    return os.path.relpath(p, REPO)


def check(ok: bool, msg: str) -> bool:
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)
    return ok


def short(h: str) -> str:
    return f"{h[:8]}…{h[-4:]}"


# ---------------------------------------------------------------- 1. hash chains
def chain(folder: str) -> tuple[bytes, dict[str, str], list[str]]:
    md = open(os.path.join(folder, "preregistration.md"), "rb").read()
    lines = open(os.path.join(folder, "preregistration.sha256"), encoding="utf-8").read().splitlines()
    states = {}
    for ln in lines:
        parts = ln.split()
        if len(parts[0]) == 64:
            label, want, prefix = "current", parts[0], md
        else:
            label, want = parts[0], parts[1]
            m = re.search(r"first (\d+) bytes", ln)
            if m:
                prefix = md[:int(m.group(1))]
            else:
                nxt = re.search(r"amendment (A\d+[a-z]?)\)", ln).group(1)
                i = md.find(f"\n#### {nxt} (".encode())
                prefix = md[:i] if i >= 0 else None
        check(prefix is not None and sha(prefix) == want, f"{rel(folder)}: {label:11s} {short(want)}")
        states[label] = want
    return md, states, lines


def snapshots(states: dict[str, str], lines: list[str], pilot: dict[str, str], pilot_lines: list[str]):
    for p in sorted(glob.glob(os.path.join(STUDIES, "*", "logs", "preregistration.after-*.bak"))):
        label = re.match(r"preregistration\.(after-[^.]+)(\.md)?\.bak$", os.path.basename(p)).group(1)
        want = PILOT_FINAL if label == "after-A22" else states.get(label)
        check(want is not None and fsha(p) == want, f"snapshot {rel(p)} = state {label}")
    for p in sorted(glob.glob(os.path.join(STUDIES, "*", "logs", "preregistration.sha256.before-*.bak"))):
        bak = open(p, encoding="utf-8").read().splitlines()
        head = bak[0].split()[0]
        ok = False
        for ls in (lines, pilot_lines):
            if ls[0].split()[0] == head:                     # unchanged since (the pilot file after A22)
                ok = bak == ls
            idx = [i for i, ln in enumerate(ls) if len(ln.split()) > 1 and ln.split()[1] == head]
            if idx:
                ok = bak[1:] == ls[idx[0] + 1:]
            if ok:
                break
        check(ok, f"hash list {rel(p)} = the list as it stood at {short(head)}")


# ---------------------------------------------------------------- 3. moved files
def manifest() -> list[dict]:
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        next(f)
        for ln in f:
            old, new, h, size, chain_, git = ln.rstrip("\n").split("\t")
            rows.append(dict(old=old, new=new, sha=h, size=int(size), hashed=chain_ == "hashed", git=git))
    return rows


# ---------------------------------------------------------------- 4. pins
def pin_files():
    a26 = json.load(open(os.path.join(STUDIES, "a26", "logs", "code_pins.json")))
    d2 = json.load(open(os.path.join(STUDIES, "d2", "logs", "code_pins.json")))
    groups = [
        ("a26 code_pins: metor_code", {os.path.join(METOR_CODE, k): v for k, v in a26["metor_code"].items()}),
        ("a26 code_pins: a26_code", {os.path.join(STUDIES, "a26", k): v for k, v in a26["a26_code"].items()}),
        ("a26 code_pins: in-repo inputs", {
            os.path.join(STUDIES, "d2", "data", "baseline_tracks.json"): a26["inputs"]["baseline_tracks"],
            os.path.join(STUDIES, "a26", "data", "r0_reference.json"): a26["inputs"]["reference"]}),
        ("d2 code_pins: d2_code, d2_tests, d2_box", {os.path.join(STUDIES, k): v for g in ("d2_code", "d2_tests", "d2_box")
                                                     for k, v in d2[g].items()}),
        ("d2 code_pins: baseline tracks", {os.path.join(STUDIES, "d2", "data", "baseline_tracks.json"): d2["baseline_tracks_json"]}),
        ("d2 code_pins: EOV-MMP front end (repo root, fdeeeb4)", {os.path.join(REPO, k): v for k, v in d2["front_end_eov"].items()}),
        ("rb1 changed_files_sha256.txt", sums(os.path.join(STUDIES, "rb1", "logs", "changed_files_sha256.txt"), METOR_CODE)),
        ("d2 rec_sha256.txt (the test recordings)", sums(os.path.join(STUDIES, "d2", "logs", "rec_sha256.txt"),
                                                         os.path.join(STUDIES, "d2", "data", "rec"))),
    ]
    for name, pins in groups:
        bad = [p for p, h in pins.items() if not os.path.isfile(p) or fsha(p) != h]
        check(not bad, f"{name}: {len(pins) - len(bad)}/{len(pins)} match" + (f"; first bad: {rel(bad[0])}" if bad else ""))


def sums(path: str, base: str) -> dict[str, str]:
    out = {}
    for ln in open(path, encoding="utf-8"):
        h, p = ln.rstrip("\n").split(None, 1)
        out[os.path.normpath(os.path.join(base, p))] = h
    return out


def closure(rows: list[dict]):
    files = {}
    for d, sub, names in os.walk(STUDIES):
        sub[:] = [s for s in sub if s not in SKIP_DIRS and os.path.join(d, s) not in LOCAL_LINKS]
        for n in names:
            p = os.path.join(d, n)
            if not os.path.islink(p):
                files[rel(p)] = fsha(p)
    heads = [rel(os.path.join(PREREG, "preregistration.md")), rel(os.path.join(PREREG, "preregistration.sha256"))]
    full, abbr, reached, frontier = set(), set(), set(heads), list(heads)
    while frontier:
        nxt = []
        for p in frontier:
            b = open(os.path.join(REPO, p), "rb").read()
            f = {m.decode() for m in FULL.findall(b)}
            try:
                a = set(ABBR.findall(b.decode("utf-8")))
            except UnicodeDecodeError:
                a = set()
            new_f, new_a = f - full, a - abbr
            full |= f
            abbr |= a
            for q, h in files.items():
                if q not in reached and (h in new_f or any(h.startswith(x) and h.endswith(y) for x, y in new_a)):
                    reached.add(q)
                    nxt.append(q)
        frontier = nxt
    moved = {r["new"] for r in rows}
    want = {r["new"] for r in rows if r["hashed"]}
    got = reached & moved
    check(got == want, f"closure over studies/: {len(got)} moved files hashed by the chain (METOR: {len(want)})"
          + ("" if got == want else f"; missing {sorted(want - got)[:3]}, extra {sorted(got - want)[:3]}"))
    extra = sorted(reached - moved)
    print(f"      reached outside the manifest: {extra if extra else 'none'}")


def main():
    print("== 1. hash chains")
    _, pilot, pilot_lines = chain(PILOT)
    md, states, lines = chain(PREREG)
    snapshots(states, lines, pilot, pilot_lines)

    print("== 2. links")
    header = md[:md.find(b"\n## Amendments")]
    check(pilot["current"] == PILOT_FINAL and PILOT_FINAL.encode() in header,
          f"A23 header cites the pilot file's final hash {short(PILOT_FINAL)}, and the pilot file has it")
    a25 = md[md.find(b"\n#### A25 ("):md.find(b"\n#### A25a (")]
    check(states.get("after-A23a") == AFTER_A23A and AFTER_A23A.encode() in a25 and PAIRTIME_FINAL.encode() in a25,
          f"A25 cites ov-pairtime's final {short(PAIRTIME_FINAL)} and its base {short(AFTER_A23A)} (= after-A23a)")
    if os.path.isfile(PAIRTIME):
        check(fsha(PAIRTIME) == PAIRTIME_FINAL, f"ov-pairtime prereg (M1, A24-A24i) hashes to {short(PAIRTIME_FINAL)} (read only)")
    else:
        print(f"skip  ov-pairtime prereg not found at {PAIRTIME}")
    through_a26 = states.get("after-A26", states["current"])
    check(through_a26 == A26_FINAL, f"the chain through A26 recomputes to {short(A26_FINAL)} (METOR's head)")
    print(f"      current head: {states['current']}")

    print("== 3. moved files")
    rows = manifest()
    bad = [r for r in rows if not os.path.isfile(os.path.join(REPO, r["new"])) or fsha(os.path.join(REPO, r["new"])) != r["sha"]]
    check(not bad, f"manifest: {len(rows) - len(bad)}/{len(rows)} files have their SHA256 at the new path"
          + (f"; first bad: {bad[0]['new']}" if bad else ""))

    print("== 4. pins")
    pin_files()
    closure(rows)

    print(f"== {'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
