#!/usr/bin/env python
"""Training entry point. Dispatches on `--arch`.

    python -m cli.train --config configs/metor_defaults.yaml \
                        --config configs/metor_vidvrd.yaml     # --arch metor
    python -m cli.train --arch metor-compat                    # EOV-MMP step 4

There is one entry point but two loop bodies, and that is deliberate. METOR
trains with one forward, one loss and one backward per video (`metor/engine.py`);
EOV-MMP's step 4 fine-tunes a stack whose `End2End_Model` calls `.backward()`
inside its own forward, per frame and per object pair. Those procedures share
nothing but the word "epoch", so folding them into a single body would produce a
function with two disjoint halves. They share this file instead.

Everything below is the EOV-MMP path.

--- EOV-MMP step 4: joint end-to-end fine-tuning.

    python -m cli.train                      # the paper's VidVRD config
    python -m cli.train --max_epoch 5 --lr 5e-6

Run from the repository root: `python -m cli.train`, not `python cli/train.py`.

Starts from the three step-1/2/3 checkpoints, which must exist -- this is
fine-tuning, not training from scratch. See docs/01_Architecture.md.

By default the test set is evaluated after every epoch and the best-mAP checkpoint
kept, so model selection uses the test set. That is the convention across this
benchmark (RePro and MMP do the same), but state it in any write-up.

`--no_epoch_eval` turns that off: nothing touches the test set, and the last
epoch is the final model. With `--epoch_ckpt_dir` every epoch writes a full-state
checkpoint and `--resume <ckpt>` continues from one exactly; `--mirror_dir`
copies each to storage outside the machine. See pre-registration A23 (prereg/).

To evaluate an already-trained checkpoint, use cli/evaluate.py instead.
"""
from __future__ import annotations

import sys
from os.path import join

import torch
from tqdm import tqdm

from cli.common import (
    build_eval_data,
    build_model_stack,
    build_train_data,
    evaluate_model,
    experiment_name,
    make_logger,
    require_files,
    seed_everything,
    set_trainable,
)
from utils import paths
from cli.config import parse_args_with_config as parse_args
from cli.train_state import (
    epoch_ckpt_path,
    load_epoch_checkpoint,
    run_training,
    startup_checks,
)


def main() -> int:
    seed_everything(3407)
    args = parse_args()

    if getattr(args, "arch", "metor") == "metor":
        from metor.engine import train_metor
        return train_metor(args)

    # ---------------------------- everything below is --arch metor-compat

    # Validate before building anything: dataset construction takes minutes.
    require_files([
        ("--detector_ckpt", args.detector_ckpt),
        ("--obj_classifier_ckpt", args.obj_classifier_ckpt),
        ("--relation_ckpt", args.relation_ckpt),
        ("--path_AFLink", args.path_AFLink),
    ])

    env_config = experiment_name(args)
    logger = make_logger(args, "train")

    _, train_loader = build_train_data(args)
    no_eval = getattr(args, "no_epoch_eval", False)
    val_dataset = val_loader = None
    if not no_eval:
        val_dataset, val_loader = build_eval_data(args)

    model, sort_model = build_model_stack(args, load_components=True)
    set_trainable(model)
    startup_checks(args, model, logger)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 20, 25], gamma=0.1)

    # Building the model consumed random numbers (more of them when the
    # iterative enhancement module is attached). With --loader_seed, re-seed here
    # so the training stream is the same whatever was built. Off = as released.
    if getattr(args, "loader_seed", -1) >= 0:
        seed_everything(3407)

    start_epoch = args.start_epoch
    if args.resume:
        start_epoch = load_epoch_checkpoint(
            args.resume, model, optimizer, scheduler,
            getattr(train_loader, "generator", None))
        logger.info(f"resumed from {args.resume} (epoch {start_epoch} done)")

    state = {"best_mmap": 0.0, "best_state": None}

    def after_epoch(epoch):
        if no_eval:
            return
        map_list, mmap = evaluate_model(
            args, model, sort_model, val_loader, val_dataset, logger)
        logger.info(f"Mean mAP: {mmap}, Best Mean mAP: {state['best_mmap']}")
        if mmap > state["best_mmap"]:
            state["best_mmap"] = mmap
            state["best_state"] = {"map": map_list, "epoch": epoch,
                                   "config": args, "state_dict": model.state_dict()}
            torch.save(state["best_state"], join(paths.CKPT_DIR, env_config + ".pth"))

    log_path = join(paths.LOG_DIR, f"{env_config}_train.log")
    run_training(model, sort_model, train_loader, optimizer, scheduler, args, logger,
                 start_epoch=start_epoch, ckpt_dir=args.epoch_ckpt_dir,
                 mirror_dir=args.mirror_dir, after_epoch=after_epoch,
                 mirror_extra=[log_path])

    if no_eval:
        final = epoch_ckpt_path(args.epoch_ckpt_dir, args.max_epoch) if args.epoch_ckpt_dir else "<not saved>"
        logger.info(f"No test evaluation was run (--no_epoch_eval). Final model = epoch {args.max_epoch}: {final}")
        print(f"  final model = epoch {args.max_epoch}: {final}")
        return 0
    best_mmap, best_state = state["best_mmap"], state["best_state"]

    print("================================ Final Results ================================")
    if best_state is None:
        # Only reachable if the loop never ran (start_epoch >= max_epoch).
        logger.info("No epoch completed; nothing was saved.")
        return 1
    logger.info("Best Epoch: {}, mAP List: {}".format(
        best_state["epoch"], str(best_state["map"])))
    print(f"  best epoch {best_state['epoch']}   mean mAP {best_mmap:.2f}")
    print(f"  saved to   {join(paths.CKPT_DIR, env_config + '.pth')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
