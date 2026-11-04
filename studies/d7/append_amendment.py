#!/usr/bin/env python3
"""Append one amendment to studies/prereg/preregistration.md, as A28-A30 were appended by hand.

    python3 studies/d7/append_amendment.py A31 2026-11-04 path/to/A31.md

The steps, each checked:
1. the prereg file's SHA256 equals the head line of preregistration.sha256 (else stop);
2. backups in studies/rpt/logs/: preregistration.after-<previous label>.bak and
   preregistration.sha256.before-<label>.bak;
3. the appended bytes are "\\n" + the amendment text (the text ends with one newline), so a blank line
   separates it from the previous amendment, as before;
4. the first N bytes are unchanged and the appended bytes equal the text; both files go back to 444;
5. preregistration.sha256 gets the new head line, then "after-<previous> <hash>  preregistration.md
   (before the <date> amendment <label>)", then its old lines after the first;
6. two lines in studies/rpt/logs/prereg_log.txt, in the earlier format;
7. studies/migration/verify_after_A26m.py runs from studies/; its output goes to
   studies/rpt/logs/verify_after_<label>.txt and must end in "ALL CHECKS PASSED".
Stdlib only. Run from the repo root.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys

PREREG = "studies/prereg/preregistration.md"
SHAFILE = "studies/prereg/preregistration.sha256"
LOGS = "studies/rpt/logs"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def main(label: str, date: str, text_path: str) -> None:
    if not re.fullmatch(r"A\d+[a-z]?", label) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise SystemExit("!! label must look like A31, date like 2026-11-04")
    text = open(text_path, "rb").read()
    if not text.endswith(b"\n") or text.endswith(b"\n\n"):
        raise SystemExit("!! the amendment text must end with exactly one newline")
    if not text.startswith(f"#### {label} ({date}): ".encode()):
        raise SystemExit(f"!! the text must start with '#### {label} ({date}): '")
    before = open(PREREG, "rb").read()
    lines = open(SHAFILE).read().splitlines()
    head = lines[0].split()[0]
    if sha(before) != head:
        raise SystemExit(f"!! STOP: {PREREG} hashes to {sha(before)}, not the recorded head {head}")
    prev = re.findall(rb"^#### (A\d+[a-z]?) \(", before, flags=re.M)[-1].decode()
    if f"#### {label} (".encode() in before:
        raise SystemExit(f"!! {label} is already in the file")
    log = open(os.path.join(LOGS, "prereg_log.txt"), "a")
    log.write(f"{now()} {label} append starting: before sha256={head} bytes={len(before)}\n")
    log.flush()
    shutil.copyfile(PREREG, os.path.join(LOGS, f"preregistration.after-{prev}.bak"))
    shutil.copyfile(SHAFILE, os.path.join(LOGS, f"preregistration.sha256.before-{label}.bak"))
    appended = b"\n" + text
    for p in (PREREG, SHAFILE):
        os.chmod(p, 0o644)
    with open(PREREG, "ab") as f:
        f.write(appended)
    after = open(PREREG, "rb").read()
    unchanged = after[:len(before)] == before
    equal = after[len(before):] == appended
    new_lines = [f"{sha(after)}  preregistration.md",
                 f"after-{prev} {head}  preregistration.md (before the {date} amendment {label})",
                 *lines[1:]]
    with open(SHAFILE, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    for p in (PREREG, SHAFILE):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    perms = "/".join(oct(os.stat(p).st_mode & 0o777)[2:] for p in (PREREG, SHAFILE))
    log.write(f"{now()} {label} appended: before={head} after={sha(after)} {label}-text-alone={sha(appended)} "
              f"first-{len(before)}-bytes-unchanged={'yes' if unchanged else 'NO'} "
              f"appended-bytes-equal-text={'yes' if equal else 'NO'} perms={perms}\n")
    log.close()
    if not (unchanged and equal and perms == "444/444"):
        raise SystemExit("!! STOP: the append check failed; see prereg_log.txt")
    out = subprocess.run([sys.executable, "migration/verify_after_A26m.py"], cwd="studies",
                         capture_output=True, text=True)
    report = (f"# {now()} cd ov-vidvrd-lab/studies; python3 migration/verify_after_A26m.py   (after {label})\n"
              + out.stdout + out.stderr + f"exit={out.returncode}\n")
    open(os.path.join(LOGS, f"verify_after_{label}.txt"), "w").write(report)
    print(f"{label}: head {sha(after)}; text alone {sha(appended)}; verify exit {out.returncode}")
    if out.returncode != 0 or "ALL CHECKS PASSED" not in out.stdout:
        raise SystemExit("!! STOP: the chain check failed; see " + os.path.join(LOGS, f"verify_after_{label}.txt"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
