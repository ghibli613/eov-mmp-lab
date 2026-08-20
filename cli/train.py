import numpy as np
from collections import defaultdict
from os.path import join
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import visdom
import json

# Imports are package-absolute; only the repo root needs to be importable so
# `python cli/train.py` works as well as `python -m cli.train`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_loading.dataset import Dataset_new
from utils.parser_func import parse_args
from utils import paths
from utils.arguments import get_args_parser
from inference.video_relation_detection import evaluate
from utils.eov_utils import get_feat_types, AverageMeter, get_logger, print_results
from inference.post_process import process_pred, association, format_
from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
import multiprocessing as mp
from models.methods import build_model
from models.end2end_model import End2End_Model
from models.model import Classifier
import warnings
warnings.filterwarnings("ignore")
# from models.tracking.deep_sort_app import run
from models.tracking.aflink.AppFreeLink import *

paths.ensure_output_dirs()          # output/{ckpt,log} are gitignored, create on demand
OUTPUT_LOG_DIR = paths.LOG_DIR
OUTPUT_CKPT_DIR = paths.CKPT_DIR
MODEL_CKPT_DIR = paths.CKPT_DIR


def seed_everything(seed = 3407):
    import random, os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict_split(args, model, sort_model, val_loader, val_dataset, split):
    """Run inference over the test set for one predicate split ('all'|'novel').

    The two evaluation passes in the original training loop were identical apart
    from this one string, so they live here once.
    """
    model.modelC.tgt_split = split
    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"eval[{split}]"):
            for final_result in model(data, sort_model):
                pre_preds = final_result['pre_preds']
                seq_lens = final_result['seq_lens']
                vids = final_result['video_name']
                pair_data = final_result['pair_data']
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
    for split, label in (('all', 'All  '), ('novel', 'Novel')):
        mean_ap, rec_at_n = _predict_split(
            args, model, sort_model, val_loader, val_dataset, split)
        logger.info("SGDet and {} split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(
            label, mean_ap * 100, rec_at_n[50] * 100, rec_at_n[100] * 100))
        map_list.append(mean_ap * 100)
    return map_list, sum(map_list) / len(map_list)


if __name__ == '__main__':
    seed_everything(3407)
    args = parse_args()

    # validate before building anything -- dataset construction takes minutes
    if args.eval_only:
        if not args.ckpt_path:
            raise SystemExit("--eval_only requires --ckpt_path <trained .pth>")
        if not os.path.exists(args.ckpt_path):
            raise SystemExit(f"--eval_only: checkpoint not found: {args.ckpt_path}")
    if not os.path.exists(args.path_AFLink):
        raise SystemExit(
            f"AFLink tracker weights not found: {args.path_AFLink}\n"
            "Pass --path_AFLink <file>. Tracking cannot run without it.")

    from models.model_zoo.model_tuing_plus_repro_copy_new_cross_dataset import Model

    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = 'baseline_fbce_' +\
        args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+str(args.temp_model)+ \
        feat_config+args.ps
    # vis = visdom.Visdom(env=env_config)

    logger = get_logger(join(OUTPUT_LOG_DIR, env_config + '_train.log'))
    logger.info('Experiment Config: {}'.format(args))

    train_dataset = Dataset_new(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_det = Dataset_new(args, "val")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size, shuffle=False)

# torch>=2.6 defaults weights_only=True, which refuses checkpoints containing
# anything but tensors -- these hold an argparse.Namespace under 'config'. The
# files are produced locally or by the paper's authors, so loading them fully
# is intended.
    object_detection_model, object_detection_criterion, object_detection_postprocessors = build_model(args)
    relationship_classification_model = Model(args).cuda()
    object_classifer = Classifier(args).cuda()

    # The three step-1/2/3 checkpoints (paper Sec. III-D) initialise the parts
    # that step 4 fine-tunes. In --eval_only we load one *trained* end-to-end
    # checkpoint instead, which supersedes all three -- so testing a model you
    # already trained does not require them.
    if not args.eval_only:
        checkpoint_path = join(MODEL_CKPT_DIR, 'checkpoint_vidvrd0059_new_1e-5.pth')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        object_detection_model.load_state_dict(checkpoint['model'])

        checkpoint_path = join(MODEL_CKPT_DIR, 'baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth')
        ckpt = torch.load(checkpoint_path, weights_only=False)
        pretrained_dict = ckpt['state_dict']
        model_dict = relationship_classification_model.state_dict()
        model_dict.update(pretrained_dict)
        relationship_classification_model.load_state_dict(model_dict)

        checkpoint_path = join(MODEL_CKPT_DIR, 'vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        object_classifer.load_state_dict(checkpoint['state_dict'])

    model = End2End_Model(args, object_detection_model, object_classifer, relationship_classification_model,
                          object_detection_criterion, object_detection_postprocessors).cuda()

    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad_(False)
        elif 'backbone' in name:
            param.requires_grad_(False)
        elif 'pre_classifier' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25], gamma=0.1)
    
    sort_model = PostLinker()
    sort_model.load_state_dict(torch.load(args.path_AFLink, weights_only=False))
    epoch_loss = AverageMeter()

    if args.eval_only:
        ckpt = torch.load(args.ckpt_path, weights_only=False)
        # checkpoints written by this script are {'map','epoch','config','state_dict'}
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded {args.ckpt_path}"
                    + (f" (epoch {ckpt['epoch']})" if isinstance(ckpt, dict) and 'epoch' in ckpt else ""))
        if missing or unexpected:
            logger.info(f"  {len(missing)} missing / {len(unexpected)} unexpected keys")
        map_list, mmap = evaluate_model(
            args, model, sort_model, val_loader_det, val_dataset_det, logger)
        logger.info("Mean mAP: {:.2f}  (all {:.2f}, novel {:.2f})".format(mmap, *map_list))
        print("================================ Eval Results ================================")
        print("  all   mAP {:.2f}\n  novel mAP {:.2f}\n  mean  mAP {:.2f}".format(
            map_list[0], map_list[1], mmap))
        sys.exit(0)

    best_mmap = 0
    for epoch in range(args.start_epoch+1, args.max_epoch+1):      
 
        model.train()
        batch_loss = AverageMeter()
        for idx, data in enumerate(tqdm(train_loader)):
            model(data, sort_model)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        map_list, mmap = evaluate_model(
            args, model, sort_model, val_loader_det, val_dataset_det, logger)

        logger.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")
        if mmap > best_mmap:
            best_mmap = mmap
            state = {
                'map': map_list,
                'epoch': epoch,
                'config': args,
                'state_dict': model.state_dict()}
            ckpt_path = join(OUTPUT_CKPT_DIR, env_config + ".pth")
            torch.save(state, ckpt_path)
    
    print("================================Final Results======================================")
    logger.info('Best Epoch: {}, mAP List: {}'.format(state['epoch'], str(state['map'])))



