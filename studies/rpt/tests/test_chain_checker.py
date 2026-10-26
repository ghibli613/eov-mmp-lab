"""migration/verify_after_A26m.py's ov-pairtime check (RPT G0 step 3): the prereg file is read from
the archive bundle, and a missing bundle or a changed file fails loudly; it is never skipped.

Run from studies/:  python -m pytest rpt/tests/test_chain_checker.py
"""
import hashlib
import os
import subprocess
import sys

import pytest

STUDIES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(STUDIES, "migration"))
import verify_after_A26m as M  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_failures(monkeypatch):
    monkeypatch.setattr(M.V, "failures", [])


def make_bundle(tmp_path, content: bytes, name="pairtime"):
    """A one-commit repo holding prereg/preregistration.md, bundled with every ref, as the archive was."""
    repo = tmp_path / name
    os.makedirs(repo / "prereg")
    (repo / "prereg" / "preregistration.md").write_bytes(content)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                              check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("add", "prereg/preregistration.md")
    git("commit", "-q", "-m", "prereg")
    bundle = tmp_path / f"{name}.bundle"
    git("bundle", "create", str(bundle), "--all")
    return str(bundle), git("rev-parse", "HEAD")


def test_present_and_matching(tmp_path, monkeypatch):
    content = b"# ov-pairtime prereg\nA24 ... A24i\n"
    bundle, commit = make_bundle(tmp_path, content)
    monkeypatch.setattr(M, "PAIRTIME_COMMIT", commit)
    monkeypatch.setattr(M.V, "PAIRTIME_FINAL", hashlib.sha256(content).hexdigest())
    assert M.check_pairtime(bundle) is True
    assert M.V.failures == []


def test_the_real_bundle_matches_the_chain():
    """The archive on this machine: a820a0e's prereg file hashes to the value A25 cites."""
    assert os.path.isfile(M.PAIRTIME_BUNDLE)
    assert M.V.PAIRTIME_FINAL == "981bf067da43c41df8759a260c9bde61014b411809e13bf9838f62f6906b5ce5"
    assert M.check_pairtime() is True
    assert M.V.failures == []


def test_bundle_missing(tmp_path, capsys):
    assert M.check_pairtime(str(tmp_path / "gone.bundle")) is False
    assert len(M.V.failures) == 1 and "the bundle is missing" in M.V.failures[0]
    assert capsys.readouterr().out.startswith("FAIL  ov-pairtime prereg")


def test_content_altered(tmp_path, monkeypatch):
    original = b"# ov-pairtime prereg\nA24 ... A24i\n"
    altered = original.replace(b"A24i", b"A24j")
    bundle, commit = make_bundle(tmp_path, altered)
    monkeypatch.setattr(M, "PAIRTIME_COMMIT", commit)
    monkeypatch.setattr(M.V, "PAIRTIME_FINAL", hashlib.sha256(original).hexdigest())
    assert M.check_pairtime(bundle) is False
    assert len(M.V.failures) == 1 and f"it hashes to {hashlib.sha256(altered).hexdigest()}" in M.V.failures[0]


@pytest.mark.parametrize("kind", ["not a bundle", "a bundle without a820a0e"])
def test_unreadable_bundle_fails(tmp_path, kind):
    """Anything git cannot resolve to a820a0e's file fails, with git's own message."""
    if kind == "not a bundle":
        bundle = tmp_path / "junk.bundle"
        bundle.write_bytes(b"not a git bundle\n")
        bundle = str(bundle)
    else:
        bundle, _ = make_bundle(tmp_path, b"some other repo\n")
    assert M.check_pairtime(bundle) is False
    assert len(M.V.failures) == 1 and "git cannot read it" in M.V.failures[0]


def test_main_exits_nonzero_without_the_bundle(tmp_path, monkeypatch, capsys):
    """The whole checker, on the real chain, with the bundle moved away: one FAIL and exit code 1."""
    monkeypatch.setattr(M, "PAIRTIME_BUNDLE", str(tmp_path / "moved-away.bundle"))
    with pytest.raises(SystemExit) as e:
        M.main()
    out = capsys.readouterr().out
    assert e.value.code == 1
    assert "FAIL  ov-pairtime prereg" in out and "the bundle is missing" in out
    assert "== 1 FAILED" in out
