#!/usr/bin/env python3
"""verify_chain.py's checks, for the chain after A26m. Read-only; exits non-zero on any failure.

verify_chain.py checks the state at the move. Once A26m is appended, two of its checks no longer
apply as written:
- the two prereg files have grown (by A26m, and by any later amendment);
- A26m cites MIGRATION.md, which lists every moved file's SHA256, so the closure now reaches all
  775 moved files, not only the 677 METOR's chain reached.
This runs the same checks with those two adjusted:
- the two prereg files are checked through the backups taken just before A26m
  (migration/logs/), which the hash-chain check shows are the chain's state after A26;
- the closure is followed through the moved files only, as METOR's chain was.
It adds one check: A26m cites MIGRATION.md's SHA256.

The ov-pairtime folder is gone since A26m. Its prereg file (M1, A24-A24i) is read from the archive
bundle instead (~/Documents/thesis/archive/ov-pairtime-2026-10-24-a820a0e.bundle, commit a820a0e)
and must hash to 981bf067…5ce5. A missing bundle, one git cannot read, or a different hash fails;
the check is never skipped (RPT G0, step 3).

    cd ov-vidvrd-lab/studies && python3 migration/verify_after_A26m.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_chain as V  # noqa: E402

BACKUPS = {"studies/prereg/preregistration.md": "studies/migration/logs/preregistration.after-A26.bak",
           "studies/prereg/preregistration.sha256": "studies/migration/logs/preregistration.sha256.before-A26m.bak"}
PAIRTIME_BUNDLE = os.path.join(os.path.dirname(V.THESIS), "archive", "ov-pairtime-2026-10-24-a820a0e.bundle")
PAIRTIME_COMMIT = "a820a0eb2a3767c3b3b62bc94f82e768db158576"       # the bundle's master, ov-pairtime's last commit


def pairtime_from_bundle(bundle: str, commit: str) -> bytes:
    """prereg/preregistration.md at `commit`, read from a git bundle through a throwaway bare repo."""
    with tempfile.TemporaryDirectory() as tmp:
        def git(*a):
            return subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True).stdout
        git("init", "-q", "--bare")
        git("fetch", "-q", bundle, "+refs/*:refs/*")
        return git("show", f"{commit}:prereg/preregistration.md")


def check_pairtime(bundle: str | None = None) -> bool:
    """A25 cites ov-pairtime's final hash; the file now lives only in the archive bundle."""
    bundle = bundle or PAIRTIME_BUNDLE
    msg = (f"ov-pairtime prereg (M1, A24-A24i) at {PAIRTIME_COMMIT[:7]} hashes to {V.short(V.PAIRTIME_FINAL)} "
           f"(read only, from {bundle})")
    if not os.path.isfile(bundle):
        return V.check(False, f"{msg}: the bundle is missing")
    try:
        got = V.sha(pairtime_from_bundle(bundle, PAIRTIME_COMMIT))
    except subprocess.CalledProcessError as e:
        return V.check(False, f"{msg}: git cannot read it ({e.stderr.decode(errors='replace').strip()})")
    return V.check(got == V.PAIRTIME_FINAL, msg + ("" if got == V.PAIRTIME_FINAL else f": it hashes to {got}"))


def closure_through_moved(rows: list[dict]) -> None:
    moved = {r["new"]: r["sha"] for r in rows}
    heads = list(BACKUPS)
    full, abbr, reached, frontier = set(), set(), set(heads), list(heads)
    while frontier:
        nxt = []
        for p in frontier:
            b = open(os.path.join(V.REPO, p), "rb").read()
            f = {m.decode() for m in V.FULL.findall(b)}
            try:
                a = set(V.ABBR.findall(b.decode("utf-8")))
            except UnicodeDecodeError:
                a = set()
            new_f, new_a = f - full, a - abbr
            full |= f
            abbr |= a
            for q, h in moved.items():
                if q not in reached and (h in new_f or any(h.startswith(x) and h.endswith(y) for x, y in new_a)):
                    reached.add(q)
                    nxt.append(q)
        frontier = nxt
    want = {r["new"] for r in rows if r["hashed"]}
    V.check(reached == want, f"closure through the moved files: {len(reached)} hashed by the chain (METOR: {len(want)})")


def main():
    print("== 1. hash chains")
    _, pilot, pilot_lines = V.chain(V.PILOT)
    md, states, lines = V.chain(V.PREREG)
    V.snapshots(states, lines, pilot, pilot_lines)

    print("== 2. links")
    V.check(pilot["current"] == V.PILOT_FINAL and V.PILOT_FINAL.encode() in md[:md.find(b"\n## Amendments")],
            f"A23 header cites the pilot file's final hash {V.short(V.PILOT_FINAL)}, and the pilot file has it")
    a25 = md[md.find(b"\n#### A25 ("):md.find(b"\n#### A25a (")]
    V.check(states.get("after-A23a") == V.AFTER_A23A and V.AFTER_A23A.encode() in a25 and V.PAIRTIME_FINAL.encode() in a25,
            f"A25 cites ov-pairtime's final {V.short(V.PAIRTIME_FINAL)} and its base {V.short(V.AFTER_A23A)} (= after-A23a)")
    check_pairtime()
    V.check(states.get("after-A26") == V.A26_FINAL, f"the chain through A26 recomputes to {V.short(V.A26_FINAL)} (METOR's head)")
    a26m = md[md.find(b"\n#### A26m ("):]
    mig = V.fsha(os.path.join(V.STUDIES, "MIGRATION.md"))
    V.check(len(a26m) > 1 and mig.encode() in a26m, f"A26m cites MIGRATION.md's SHA256 {V.short(mig)}")
    print(f"      current head: {states['current']}")

    print("== 3. moved files")
    rows = V.manifest()
    bad = [r for r in rows if V.fsha(os.path.join(V.REPO, BACKUPS.get(r["new"], r["new"]))) != r["sha"]]
    V.check(not bad, f"manifest: {len(rows) - len(bad)}/{len(rows)} files have their SHA256 at the new path "
            f"(the 2 prereg files through their pre-A26m backups)" + (f"; first bad: {bad[0]['new']}" if bad else ""))

    print("== 4. pins")
    V.pin_files()
    closure_through_moved(rows)

    print(f"== {'ALL CHECKS PASSED' if not V.failures else f'{len(V.failures)} FAILED'}")
    sys.exit(1 if V.failures else 0)


if __name__ == "__main__":
    main()
