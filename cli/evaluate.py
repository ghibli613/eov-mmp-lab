#!/usr/bin/env python
"""Evaluate a trained checkpoint on the VidVRD test set.

    python -m cli.evaluate --ckpt_path output/ckpt/<trained>.pth

Run from the repository root.

Reports mAP and Recall@50/100 for both the `all` and `novel` predicate splits.

Only the test dataset is built, and the three step-1/2/3 component checkpoints
are not loaded -- a trained end-to-end checkpoint already contains those
weights. Both of those save memory that cli/train.py necessarily spends: the
train dataset alone holds two CLIP ViT-L/14@336px models (~2.7 GB).

So evaluation needs materially less VRAM than training, and needs only the
end-to-end checkpoint plus the AFLink tracker.
"""
from __future__ import annotations

import sys

from cli.common import (
    build_eval_data,
    build_model_stack,
    evaluate_model,
    load_end2end,
    make_logger,
    require_files,
    seed_everything,
)
from utils.parser_func import parse_args


def main() -> int:
    seed_everything(3407)
    args = parse_args()

    if not args.ckpt_path:
        raise SystemExit(
            "cli/evaluate.py requires --ckpt_path <trained .pth>\n"
            "  This is a checkpoint written by cli/train.py, or one of the\n"
            "  authors' trained end-to-end models. To fine-tune from the\n"
            "  step-1/2/3 components instead, use cli/train.py.")
    require_files([("--ckpt_path", args.ckpt_path),
                   ("--path_AFLink", args.path_AFLink)])

    logger = make_logger(args, "eval")

    val_dataset, val_loader = build_eval_data(args)
    # load_components=False: the end-to-end checkpoint supersedes all three.
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, logger)

    map_list, mmap = evaluate_model(
        args, model, sort_model, val_loader, val_dataset, logger)

    logger.info("Mean mAP: {:.2f}  (all {:.2f}, novel {:.2f})".format(mmap, *map_list))
    print("================================ Eval Results ================================")
    print(f"  all    mAP  {map_list[0]:.2f}")
    print(f"  novel  mAP  {map_list[1]:.2f}")
    print(f"  mean   mAP  {mmap:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
