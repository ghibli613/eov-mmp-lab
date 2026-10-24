"""Epoch checkpoints, RNG state and the epoch loop for the EOV-MMP step-4 path.

Pre-registration A23 needs three things the released loop did not have:

* a full-state checkpoint at the end of **every** epoch (model, optimizer,
  scheduler, epoch, every RNG stream including the DataLoader's own generator),
  so a run can die with a rented machine and resume exactly;
* a copy of that checkpoint on storage outside the machine, with a verified hash;
* no test evaluation inside the loop (that was model selection on test).

All of it is opt-in (`--epoch_ckpt_dir`, `--mirror_dir`, `--no_epoch_eval`,
`--resume`). With the flags off, `run_training` does what the loop in
`cli/train.py` always did: train every video, `scheduler.step()`, call the
optional `after_epoch` hook.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil

import numpy as np
import torch
from tqdm import tqdm

from models import novel_filter

FROZEN_SUBSTRINGS = ("text_encoder", "backbone", "pre_classifier")


# ---------------------------------------------------------------- RNG state
def capture_rng(loader_gen):
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader": loader_gen.get_state() if loader_gen is not None else None,
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st, loader_gen):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"].cpu())
    if st.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["torch_cuda"]])
    if loader_gen is not None:
        if st["loader"] is None:
            raise RuntimeError("checkpoint has no loader-generator state but this "
                               "run uses --loader_seed")
        loader_gen.set_state(st["loader"].cpu())


# ------------------------------------------------------------- checkpoints
def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def epoch_ckpt_path(ckpt_dir, epoch):
    return os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")


def save_epoch_checkpoint(path, model, optimizer, scheduler, epoch, loader_gen,
                          args, video_order, novel_stats):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),     # `evaluate.py --ckpt_path` loads this key
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng(loader_gen),
        "config": args,
        "video_order": video_order,
        "novel_stats": novel_stats,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)                     # never leave a half-written checkpoint
    return path


def load_epoch_checkpoint(path, model, optimizer, scheduler, loader_gen):
    """Restore everything `save_epoch_checkpoint` wrote. Returns the epoch done."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    restore_rng(ck["rng"], loader_gen)
    return ck["epoch"]


def mirror_files(paths, mirror_dir, logger=None):
    """Copy files to persistent storage and verify each copy by SHA256.
    Writes `<name>.sha256` next to each copy. Raises on a mismatch."""
    os.makedirs(mirror_dir, exist_ok=True)
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        dst = os.path.join(mirror_dir, os.path.basename(p))
        shutil.copy2(p, dst)
        a, b = sha256_file(p), sha256_file(dst)
        if a != b:
            raise RuntimeError(f"mirror copy of {p} does not match its source")
        with open(dst + ".sha256", "w") as f:
            f.write(f"{a}  {os.path.basename(p)}\n")
        out[os.path.basename(p)] = a
        if logger:
            logger.info(f"mirrored {p} -> {dst} sha256={a}")
    return out


# ---------------------------------------------------------- startup checks
def startup_checks(args, model, logger):
    """Log and assert what A23 requires before the first step.

    * the iterative enhancement module is absent for --n_iter 0 and has exactly
      --n_iter layers otherwise (and `--expect_n_iter`, when given, must agree --
      this guards the compat default of 2);
    * its parameters are trainable;
    * trainable parameter counts, so R3 - R2 can be checked against the module's
      size afterwards.
    Returns the counts as a dict."""
    fe = getattr(model.modelC, "featEmbedding", None)
    iem = getattr(fe, "iem", None)
    n_layers = 0 if iem is None else len(iem.layers)
    logger.info(f"featEmbedding.iem is {'None' if iem is None else 'present'}; layers={n_layers}")
    expect = getattr(args, "expect_n_iter", -1)
    if expect >= 0 and n_layers != expect:
        raise SystemExit(f"--expect_n_iter {expect} but the model has {n_layers} "
                         f"iterative enhancement layers (iem is "
                         f"{'None' if iem is None else 'present'}). "
                         "Pass --n_iter explicitly: compat defaults to 2.")
    if iem is None and getattr(args, "n_iter", 2) != 0 and getattr(args, "arch", "") == "metor-compat":
        raise SystemExit("no iterative enhancement module although --n_iter != 0")

    iem_params = 0 if iem is None else sum(p.numel() for p in iem.parameters())
    iem_trainable = 0 if iem is None else sum(p.numel() for p in iem.parameters() if p.requires_grad)
    if iem is not None:
        if iem_trainable != iem_params:
            raise SystemExit(f"IEM parameters frozen: {iem_trainable}/{iem_params} trainable")
        bad = [n for n, _ in model.named_parameters()
               if ".iem." in n and any(s in n for s in FROZEN_SUBSTRINGS)]
        if bad:
            raise SystemExit(f"IEM parameter names match a frozen pattern: {bad[:3]}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts = {"n_layers": n_layers, "iem_params": iem_params,
              "trainable_total": trainable, "trainable_without_iem": trainable - iem_trainable,
              "all_params": total}
    logger.info("param counts: " + json.dumps(counts))
    return counts


# ---------------------------------------------------------------- the loop
def run_training(model, sort_model, train_loader, optimizer, scheduler, args, logger, *,
                 start_epoch=0, ckpt_dir="", mirror_dir="", after_epoch=None,
                 mirror_extra=()):
    """Train epochs start_epoch+1 .. args.max_epoch.

    Per epoch: train every video once (the model does its own backward inside
    its forward, so only `optimizer.step()` and `zero_grad()` happen here),
    `scheduler.step()`, then -- if `ckpt_dir` -- write the full-state checkpoint
    and the video order, mirror them if `mirror_dir`, then call
    `after_epoch(epoch)`.
    """
    loader_gen = getattr(train_loader, "generator", None)
    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        model.train()
        novel_filter.STATS.reset()
        order = []
        for data in tqdm(train_loader, desc=f"train[{epoch}/{args.max_epoch}]"):
            order.append(data["video_name"][0])
            model(data, sort_model)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        stats = novel_filter.STATS.snapshot()
        if getattr(args, "exclude_novel_train", False):
            logger.info(f"epoch {epoch} novel-filter counts: {json.dumps(stats)}")
            for k in ("novel_object_ids_in_targets", "novel_predicate_entries_in_pre_label",
                      "novel_gt_tids_in_label_matching"):
                if stats.get(k, 0) != 0:
                    raise RuntimeError(f"{k} = {stats[k]} (must be 0)")

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = save_epoch_checkpoint(epoch_ckpt_path(ckpt_dir, epoch), model, optimizer,
                                         scheduler, epoch, loader_gen, args, order, stats)
            order_path = os.path.join(ckpt_dir, f"video_order_epoch_{epoch:03d}.json")
            with open(order_path, "w") as f:
                json.dump({"epoch": epoch, "order": order, "novel_stats": stats}, f)
            logger.info(f"saved {path} sha256={sha256_file(path)}")
            if mirror_dir:
                mirror_files([path, order_path, *mirror_extra], mirror_dir, logger)
        if after_epoch is not None:
            after_epoch(epoch)
