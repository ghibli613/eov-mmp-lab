"""A31 §5: the HELD-B guard. Each case runs in a fresh subprocess, since audit hooks cannot be removed.
Opens are attempted on real paths and on paths that need not exist: the audit event fires before the
open, so a sealed path raises PermissionError whether or not it exists."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D7 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(D7))
sys.path.insert(0, D7)
import held_b_guard as G  # noqa: E402

SPLIT = json.load(open(os.path.join(D7, "held_split.json")))
A, B = sorted(SPLIT["held_a"]), sorted(SPLIT["held_b"])
FIT = sorted(json.load(open(os.path.join(REPO, "pilot_analysis", "stage1", "train_split.json")))["fit"])
B_VAL = [v for v in B if "_val_" in v]


def outcome(paths):
    """Install the guard in a subprocess, try to open each path; -> {path: 'blocked' | 'allowed'}."""
    code = (
        "import sys, json, gzip\n"
        f"sys.path.insert(0, {D7!r})\n"
        "import held_b_guard; held_b_guard.install()\n"
        "out = {}\n"
        f"for p in {paths!r}:\n"
        "    try:\n"
        "        (gzip.open if p.endswith('.gz') else open)(p, 'rb').close(); out[p] = 'allowed'\n"
        "    except PermissionError: out[p] = 'blocked'\n"
        "    except (FileNotFoundError, IsADirectoryError, OSError): out[p] = 'allowed'\n"
        "print(json.dumps(out))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def rp(*parts):
    return os.path.join(REPO, *parts)


def test_held_b_per_video_files_are_blocked():
    paths = [rp("data", "vidvrd", "anno", "train", f"{B[0]}.json"),
             rp("studies", "rpt", "g1", "sam3_out", f"{B[1]}.json.gz"),
             rp("studies", "rpt", "g1", "eov_out", f"{B[2]}.json.gz"),
             rp("studies", "somewhere_new", f"{B[3]}_anything.json")]
    if B_VAL:
        paths.append(rp("data", "vidvrd", "anno", "train", f"{B_VAL[0]}.json"))
    assert set(outcome(paths).values()) == {"blocked"}
    assert B_VAL, "HELD-B has val-form IDs; the test above covers that form"


def test_every_aggregate_is_blocked():
    files, dirs = G.sealed_paths()
    paths = sorted(files) + [os.path.join(d, "anything.json") for d in dirs]
    paths += [rp("pilot_analysis", "stage2", "data", "held_gt.json"), rp("pilot_analysis", "stage2", "data", "slots.npz"),
              rp("studies", "rpt", "g1", "logs", "box", "sam3_manifest.jsonl"),
              rp("studies", "rpt", "g1", "viz_unmatched", os.listdir(rp("studies", "rpt", "g1", "viz_unmatched"))[0])]
    res = outcome(paths)
    assert [p for p, v in res.items() if v != "blocked"] == []
    assert len(files) >= 18 and len(dirs) >= 10


def test_allowed_paths_stay_open():
    paths = [rp("data", "vidvrd", "anno", "train", f"{A[0]}.json"),
             rp("data", "vidvrd", "anno", "train", f"{FIT[0]}.json"),
             rp("studies", "rpt", "g1", "sam3_out", f"{A[0]}.json.gz"),
             rp("data", "vidvrd", "frames", B[0], "000001.jpg"),                  # frames are inputs
             rp("studies", "d7", "held_split.json"),
             rp("studies", "d7", "extracts", "pilot_analysis__stage1c__gt_tracks_train.held_a.json"),
             rp("pilot_analysis", "stage1", "train_split.json"),
             rp("studies", "rpt", "g1", "metrics.py")]
    assert set(outcome(paths).values()) == {"allowed"}


def test_test_label_guard_is_installed_too():
    assert outcome([rp("data", "vidvrd", "anno", "test", "x.json"),
                    rp("data", "vidvrd", "data", "test_relation_gt.json")]) == {
        rp("data", "vidvrd", "anno", "test", "x.json"): "blocked",
        rp("data", "vidvrd", "data", "test_relation_gt.json"): "blocked"}


def test_split_hash_is_checked(tmp_path):
    bad = tmp_path / "held_split.json"
    bad.write_text(json.dumps(SPLIT) + " ")
    try:
        G.held_b(str(bad))
    except RuntimeError as e:
        assert "pinned" in str(e)
    else:
        raise AssertionError("a changed held_split.json must be refused")


def test_check_function_unit():
    b = frozenset(B[:2])
    chk = G.make_check(b, frozenset({"/r/agg.json"}), ("/r/sealed/",), ("/r/frames/",))
    assert chk("/r/agg.json") and chk("/r/sealed/x/y.npy") and chk(f"/r/out/{B[0]}.json.gz")
    assert chk(f"/r/frames/{B[0]}/000001.jpg") is None and chk(f"/r/out/{A[0]}.json") is None and chk(3) is None
