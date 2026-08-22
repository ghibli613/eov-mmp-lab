"""Shared setup for the train and evaluate entry points.

Both scripts need the same model stack, the same experiment naming and the same
evaluation pass; only what they do afterwards differs. Keeping that here means
the two entry points stay short enough to read in one screen.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading.dataset import Dataset_new
from inference.post_process import association, format_, process_pred
from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
from models.end2end_model import End2End_Model
from models.methods import build_model
from models.object_classifier import Classifier
from models.tracking.aflink.AppFreeLink import PostLinker
from utils import paths
from utils.eov_utils import AverageMeter, get_feat_types, get_logger

__all__ = [
    "AverageMeter", "build_eval_data", "build_model_stack", "build_train_data",
    "evaluate_model", "experiment_name", "load_end2end", "make_logger",
    "require_files", "seed_everything", "set_trainable",
]


def seed_everything(seed: int = 3407) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment_name(args) -> str:
    """The config string used for log and checkpoint filenames."""
    feat_config = "_"
    for type_ in get_feat_types(args):
        feat_config += type_.split("_")[0] + "_"
    return ("baseline_fbce_" + args.dataset
            + "_bs" + str(args.batch_size)
            + "_lr" + str(args.lr)
            + "_dim" + str(args.clip_emb_dim)
            + "_" + str(args.temp_model)
            + feat_config + args.ps)


def make_logger(args, kind: str):
    paths.ensure_output_dirs()
    logger = get_logger(join(paths.LOG_DIR, f"{experiment_name(args)}_{kind}.log"))
    logger.info(f"Experiment Config: {args}")
    return logger


def require_files(pairs) -> None:
    """Fail before anything expensive is built. `pairs` is [(flag, path), ...]."""
    missing = [(flag, p) for flag, p in pairs if not p or not os.path.exists(p)]
    if missing:
        lines = "\n".join(f"  {flag:22s} {p or '<not set>'}" for flag, p in missing)
        raise SystemExit("Missing required file(s):\n" + lines
                         + "\nSee README section 4 for how to obtain them.")


# --------------------------------------------------------------------- data
def build_eval_data(args):
    ds = Dataset_new(args, "val")
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=False)


def build_train_data(args):
    ds = Dataset_new(args, "train")
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=True)


# -------------------------------------------------------------------- model
def build_model_stack(args, load_components: bool):
    """Assemble the end-to-end model and the AFLink tracker.

    `load_components` loads the three step-1/2/3 checkpoints that step 4
    fine-tunes (paper Sec. III-D). Evaluation passes False: one trained
    end-to-end checkpoint supersedes all three.

    torch>=2.6 defaults weights_only=True, which refuses checkpoints holding
    anything but tensors -- these carry an argparse.Namespace under 'config'.
    They come from the paper's authors or from this script, so loading them
    fully is intended.
    """
    from models.relation_classifier import Model

    detector, criterion, postprocessors = build_model(args)
    relation = Model(args).cuda()
    classifier = Classifier(args).cuda()

    if load_components:
        ckpt = torch.load(args.detector_ckpt, weights_only=False)
        detector.load_state_dict(ckpt["model"])

        ckpt = torch.load(args.relation_ckpt, weights_only=False)
        merged = relation.state_dict()
        merged.update(ckpt["state_dict"])
        relation.load_state_dict(merged)

        ckpt = torch.load(args.obj_classifier_ckpt, weights_only=False)
        classifier.load_state_dict(ckpt["state_dict"])

    model = End2End_Model(args, detector, classifier, relation,
                          criterion, postprocessors).cuda()

    sort_model = PostLinker()
    sort_model.load_state_dict(torch.load(args.path_AFLink, weights_only=False))
    return model, sort_model


def load_end2end(model, ckpt_path: str, logger):
    """Load a checkpoint written by cli/train.py into the assembled model."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    epoch = f" (epoch {ckpt['epoch']})" if isinstance(ckpt, dict) and "epoch" in ckpt else ""
    logger.info(f"Loaded {ckpt_path}{epoch}")
    if missing or unexpected:
        logger.info(f"  {len(missing)} missing / {len(unexpected)} unexpected keys")
    return ckpt


def set_trainable(model) -> None:
    """Freeze the frozen parts; step 4 tunes everything else."""
    for name, param in model.named_parameters():
        frozen = ("text_encoder" in name or "backbone" in name
                  or "pre_classifier" in name)
        param.requires_grad_(not frozen)


# --------------------------------------------------------------- evaluation
def _predict_split(args, model, sort_model, val_loader, val_dataset, split):
    """Inference over the test set for one predicate split ('all'|'novel').

    The two evaluation passes in the original training loop were identical apart
    from this one string, so they live here once.
    """
    model.modelC.tgt_split = split
    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"eval[{split}]"):
            for final_result in model(data, sort_model):
                pre_preds = final_result["pre_preds"]
                seq_lens = final_result["seq_lens"]
                vids = final_result["video_name"]
                pair_data = final_result["pair_data"]
                for seq_id, seq_len in enumerate(seq_lens):
                    clip_rels = process_pred(
                        args, val_dataset.id2pre, val_dataset.obj2id, val_dataset.prior,
                        pre_preds[seq_id][:seq_len], pair_data[seq_id])
                    pred_rels[vids[seq_id]].extend(association(clip_rels))
    for vid in pred_rels:
        pred_rels[vid] = format_(args, pred_rels[vid])
    return eval_relation_detection_openvoc(
        target_split_pred=split, prediction_results=pred_rels, rt_hit_infos=True)


def evaluate_model(args, model, sort_model, val_loader, val_dataset, logger):
    """Evaluate on both predicate splits. Returns (map_list, mean-of-both)."""
    model.eval()
    map_list = []
    for split, label in (("all", "All  "), ("novel", "Novel")):
        mean_ap, rec_at_n = _predict_split(
            args, model, sort_model, val_loader, val_dataset, split)
        logger.info(f"SGDet and {label} split | mAP:{mean_ap * 100:.2f}, Recall@50:{rec_at_n[50] * 100:.2f}, Recall@100:{rec_at_n[100] * 100:.2f}")
        map_list.append(mean_ap * 100)
    return map_list, sum(map_list) / len(map_list)
