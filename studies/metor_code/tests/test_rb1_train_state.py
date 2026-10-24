"""RB1 (A23): resumable training, no test inside the loop, identical data order.

A toy model stands in for the EOV stack. It copies the one habit that matters to
the loop: `model(data, sort_model)` runs its own backward inside forward, and the
loop only steps the optimizer.
"""
from __future__ import annotations

import copy
import json
import logging
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from cli import common
from cli.train_state import (
    epoch_ckpt_path, load_epoch_checkpoint, mirror_files, run_training, sha256_file,
    startup_checks,
)
from metor.modules.iterative_enhancement import IterativeEnhancementModule

LOG = logging.getLogger("rb1-test")


class _Videos(torch.utils.data.Dataset):
    def __init__(self, n=7):
        self.names = [f"vid{i:02d}" for i in range(n)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"video_name": self.names[i], "x": torch.randn(3, generator=g)}


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(3, 3)
        self.drop = nn.Dropout(0.3)

    def forward(self, data, sort_model):
        out = self.drop(self.lin(data["x"].reshape(1, 3)))   # consumes the CPU RNG
        out.pow(2).sum().backward()


def _setup(seed=3407, n_videos=7):
    ds = _Videos(n_videos)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, generator=gen)
    model = _Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 2], gamma=0.5)
    torch.manual_seed(3407)
    return model, loader, opt, sched


def _args(max_epoch):
    return types.SimpleNamespace(max_epoch=max_epoch, exclude_novel_train=False)


def _weights(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_two_epochs_equal_one_then_resume_then_one(tmp_path):
    m, loader, opt, sch = _setup()
    run_training(m, None, loader, opt, sch, _args(2), LOG, ckpt_dir=str(tmp_path / "straight"))

    m1, loader1, opt1, sch1 = _setup()
    run_training(m1, None, loader1, opt1, sch1, _args(1), LOG, ckpt_dir=str(tmp_path / "a"))
    # a fresh process: new objects, then restore from the epoch-1 checkpoint
    m2, loader2, opt2, sch2 = _setup(seed=12345)             # deliberately different seed
    done = load_epoch_checkpoint(epoch_ckpt_path(str(tmp_path / "a"), 1), m2, opt2, sch2,
                                 loader2.generator)
    assert done == 1
    run_training(m2, None, loader2, opt2, sch2, _args(2), LOG, start_epoch=done,
                 ckpt_dir=str(tmp_path / "b"))

    w, w2 = _weights(m), _weights(m2)
    assert all(torch.equal(w[k], w2[k]) for k in w)          # bit-identical, not just close
    o1 = json.load(open(tmp_path / "straight" / "video_order_epoch_002.json"))["order"]
    o2 = json.load(open(tmp_path / "b" / "video_order_epoch_002.json"))["order"]
    assert o1 == o2


def test_checkpoint_carries_everything_resume_needs(tmp_path):
    m, loader, opt, sch = _setup()
    run_training(m, None, loader, opt, sch, _args(1), LOG, ckpt_dir=str(tmp_path))
    ck = torch.load(epoch_ckpt_path(str(tmp_path), 1), weights_only=False)
    assert {"epoch", "state_dict", "optimizer", "scheduler", "rng", "video_order"} <= set(ck)
    assert {"python", "numpy", "torch_cpu", "loader"} <= set(ck["rng"])
    assert ck["epoch"] == 1 and len(ck["video_order"]) == 7


def test_video_order_does_not_depend_on_what_was_built_before(tmp_path):
    """R2 and R3 build different models (R3 adds the IEM, which draws random
    numbers). With --loader_seed the order must not care."""
    def order(burn):
        torch.rand(burn)                                      # model construction draws
        m, loader, opt, sch = _setup()
        run_training(m, None, loader, opt, sch, _args(1), LOG, ckpt_dir=str(tmp_path / f"o{burn}"))
        return json.load(open(tmp_path / f"o{burn}" / "video_order_epoch_001.json"))["order"]
    assert order(0) == order(10_000)


def test_loop_without_flags_trains_and_calls_the_hook(tmp_path):
    seen = []
    m, loader, opt, sch = _setup()
    before = _weights(m)
    run_training(m, None, loader, opt, sch, _args(2), LOG, after_epoch=seen.append)
    assert seen == [1, 2]
    assert not list(tmp_path.iterdir())                       # nothing written without flags
    assert any(not torch.equal(before[k], v) for k, v in _weights(m).items())


def test_mirror_copies_and_verifies(tmp_path):
    src = tmp_path / "ck.pth"
    src.write_bytes(b"x" * 1000)
    out = mirror_files([str(src), str(tmp_path / "missing.json")], str(tmp_path / "mirror"))
    assert out == {"ck.pth": sha256_file(str(src))}
    assert (tmp_path / "mirror" / "ck.pth.sha256").read_text().startswith(out["ck.pth"])


# ---------------------------------------------------------------- startup checks
class _Rel(nn.Module):
    def __init__(self, n_iter):
        super().__init__()
        self.featEmbedding = nn.Module()
        self.featEmbedding.base = nn.Linear(4, 4)
        self.featEmbedding.iem = (None if n_iter == 0 else
                                  IterativeEnhancementModule(16, n_iter=n_iter, n_heads=2))


class _Stack(nn.Module):
    def __init__(self, n_iter):
        super().__init__()
        self.modelC = _Rel(n_iter)
        self.modelA = nn.Linear(2, 2)


def _a(**kw):
    return types.SimpleNamespace(arch="metor-compat", n_iter=kw.get("n_iter", 2),
                                 expect_n_iter=kw.get("expect", -1))


def test_startup_asserts_no_iem_for_n_iter_0_and_two_layers_for_2():
    c0 = startup_checks(_a(n_iter=0, expect=0), _Stack(0), LOG)
    c2 = startup_checks(_a(n_iter=2, expect=2), _Stack(2), LOG)
    assert c0["n_layers"] == 0 and c0["iem_params"] == 0
    assert c2["n_layers"] == 2 and c2["iem_params"] > 0


def test_startup_aborts_when_the_compat_default_would_attach_an_iem():
    with pytest.raises(SystemExit, match="expect_n_iter 0"):
        startup_checks(_a(n_iter=2, expect=0), _Stack(2), LOG)
    with pytest.raises(SystemExit, match="expect_n_iter 2"):
        startup_checks(_a(n_iter=0, expect=2), _Stack(0), LOG)


def test_trainable_counts_differ_by_exactly_the_iem_size():
    m0, m2 = _Stack(0), _Stack(2)
    c0 = startup_checks(_a(n_iter=0, expect=0), m0, LOG)
    c2 = startup_checks(_a(n_iter=2, expect=2), m2, LOG)
    assert c2["trainable_total"] - c0["trainable_total"] == c2["iem_params"]
    assert c2["trainable_without_iem"] == c0["trainable_without_iem"]


def test_startup_aborts_if_the_iem_is_frozen():
    m = _Stack(2)
    for p in m.modelC.featEmbedding.iem.parameters():
        p.requires_grad_(False)
    with pytest.raises(SystemExit, match="frozen"):
        startup_checks(_a(n_iter=2, expect=2), m, LOG)


def test_set_trainable_leaves_the_iem_trainable():
    """The real `set_trainable`, on parameter names shaped like the real model's."""
    m = _Stack(2)
    m.modelC.pre_classifier = nn.Linear(2, 2)
    m.modelA.backbone = nn.Linear(2, 2)
    common.set_trainable(m)
    assert all(p.requires_grad for p in m.modelC.featEmbedding.iem.parameters())
    assert not any(p.requires_grad for p in m.modelC.pre_classifier.parameters())
    assert not any(p.requires_grad for p in m.modelA.backbone.parameters())


# ---------------------------------------------------------------------- parser
def test_every_rb1_flag_defaults_to_the_released_behaviour(args):
    assert args.exclude_novel_train is False
    assert args.uniform_fusion is False
    assert args.no_epoch_eval is False
    assert args.epoch_ckpt_dir == "" and args.mirror_dir == ""
    assert args.loader_seed == -1 and args.expect_n_iter == -1
    assert args.small_train_videos == 0
    assert args.resume == ""            # falsy, as the old store_true default was


def test_evenly_spaced_subset_includes_first_and_last_and_is_deterministic():
    items = [f"v{i:03d}" for i in range(800)]
    pick = common._evenly_spaced(items, 5)
    assert pick == ["v000", "v200", "v400", "v599", "v799"] or (pick[0], pick[-1]) == ("v000", "v799")
    assert pick == common._evenly_spaced(list(reversed(items)), 5)
    assert common._evenly_spaced(items, 1) == ["v000"]
