#!/usr/bin/env python
"""Count unscored-track fallbacks at INFERENCE on N training videos (GPU).

Runs the released checkpoint's settings in eval mode over N evenly spaced
TRAINING videos (`Dataset_new(args, "train")`, no GT used by the eval branch).
It touches no test video and no test annotation, so it is not a test evaluation.
Changes no behaviour; see fallback_probe.py.

    python studies/rb1/scripts/fallback_count_gpu.py --n_videos 2 --out studies/rb1/logs/fallback_counts_eval.json \
        --ckpt_path output/ckpt/baseline_fbce_..._end2end_base-001.pth --path_AFLink output/ckpt/AFLink_epoch20.pth

`--uniform_fusion` may be added; extra flags go to the repo's own parser.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fallback_probe as probe          # also puts the repo root on sys.path


def main():
    argv = sys.argv[1:]
    n, out = 2, os.path.join(probe.ROOT, "studies", "rb1", "logs", "fallback_counts_eval.json")
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--n_videos":
            n = int(argv[i + 1]); i += 2
        elif argv[i] == "--out":
            out = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1
    sys.argv = [sys.argv[0], "--arch", "metor-compat", "--n_iter", "0", *rest]

    import torch
    from cli.common import (_evenly_spaced, build_model_stack, load_end2end, make_logger,
                            require_files, seed_everything)
    from cli.config import parse_args_with_config as parse_args
    from data_loading.feature_dataset import Dataset_new
    seed_everything(3407)
    args = parse_args()
    require_files([("--ckpt_path", args.ckpt_path), ("--path_AFLink", args.path_AFLink)])
    logger = make_logger(args, "fallback_probe")

    ds = Dataset_new(args, "train")
    ds.path_list = _evenly_spaced(ds.path_list, n)
    model, sort_model = build_model_stack(args, load_components=False)
    load_end2end(model, args.ckpt_path, logger)
    model.eval()
    probe.install()
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)   # as build_eval_data
    with torch.no_grad():
        for data in loader:
            model(data, sort_model)
    probe.dump(out)
    print(json.dumps(probe.COUNTS, indent=2))


if __name__ == "__main__":
    main()
