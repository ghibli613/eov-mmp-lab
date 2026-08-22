#!/usr/bin/env python
"""Step 4 of the paper's training scheme: joint end-to-end fine-tuning.

    python -m cli.train                      # the paper's VidVRD config
    python -m cli.train --max_epoch 5 --lr 5e-6

Run from the repository root: `python -m cli.train`, not `python cli/train.py`.

Starts from the three step-1/2/3 checkpoints, which must exist -- this is
fine-tuning, not training from scratch. See docs/01_Architecture.md.

The test set is evaluated after every epoch and the best-mAP checkpoint kept, so
model selection uses the test set. That is the convention across this benchmark
(RePro and MMP do the same), but state it in any write-up.

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
from utils.parser_func import parse_args


def main() -> int:
    seed_everything(3407)
    args = parse_args()

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
    val_dataset, val_loader = build_eval_data(args)

    model, sort_model = build_model_stack(args, load_components=True)
    set_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 20, 25], gamma=0.1)

    best_mmap = 0.0
    best_state = None
    for epoch in range(args.start_epoch + 1, args.max_epoch + 1):
        model.train()
        for data in tqdm(train_loader, desc=f"train[{epoch}/{args.max_epoch}]"):
            model(data, sort_model)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        map_list, mmap = evaluate_model(
            args, model, sort_model, val_loader, val_dataset, logger)
        logger.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")

        if mmap > best_mmap:
            best_mmap = mmap
            best_state = {"map": map_list, "epoch": epoch,
                          "config": args, "state_dict": model.state_dict()}
            torch.save(best_state, join(paths.CKPT_DIR, env_config + ".pth"))

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
