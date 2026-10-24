"""The `--arch metor` training loop -- the paper's Sec. 3.5 objective.

Lives here rather than in `cli/` because it is the training *procedure* for this
architecture, and `cli/train.py` is just the entry point that chooses between
this and the EOV-MMP one.

Why there are two loops at all, when there is only one `cli/train.py`:

  `--arch metor`        one forward over the whole video, one loss (Eq. 12),
                        one backward. That is the paper's "end-to-end manner",
                        and it is what makes the framework "unified".

  `--arch metor-compat` EOV-MMP's step-4 fine-tuning, where `End2End_Model`
                        calls `.backward()` *inside its own forward*, per frame
                        and per object pair, because the detector, the tracker
                        and the relation classifier are trained as separate
                        stages joined at the end.

Those are genuinely different procedures, not one procedure with a flag. Merging
them into a single loop body would mean a function whose two halves share
nothing but the word "epoch". They share an entry point instead.
"""
from __future__ import annotations

import sys
import time
from os.path import join

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading.frame_dataset import METORDataset, metor_collate
from metor.matcher import TrajectoryHungarianMatcher, build_pair_targets
from metor.model import build_metor
from utils import paths
from utils.eov_utils import AverageMeter, get_logger


def _device(args):
    if getattr(args, "cpu", False) or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _slice_to_base(outputs, dataset, device):
    """Restrict the scored vocabulary to base categories for training.

    Open-vocabulary means the novel categories are *never* supervised. The model
    scores against the full text bank because that is what evaluation needs, so
    training slices the columns down here and remaps the labels to match. Get
    this wrong in either direction and the novel-split number is meaningless:
    too many columns and you have trained on the test vocabulary, too few and
    the label ids no longer line up with the columns.
    """
    base_obj = torch.tensor(dataset.base_obj_ids, device=device)
    outputs = dict(outputs)
    outputs["sbj_logits"] = outputs["sbj_logits"].index_select(-1, base_obj)
    outputs["obj_logits"] = outputs["obj_logits"].index_select(-1, base_obj)
    base_rel = torch.tensor(dataset.base_rel_ids, device=device)
    outputs["rel_scores"] = outputs["rel_scores"].index_select(-1, base_rel)
    outputs["text_obj"] = outputs["text_obj"].index_select(0, base_obj)
    outputs["text_rel"] = outputs["text_rel"].index_select(0, base_rel)
    return outputs, base_obj, base_rel


def _remap_labels(targets, base_obj, device):
    """Global object ids -> positions in the base-only slice; -1 stays -1."""
    lookup = torch.full((int(base_obj.max()) + 1,), -1, dtype=torch.long, device=device)
    lookup[base_obj] = torch.arange(len(base_obj), device=device)
    out = dict(targets)
    for key in ("sbj_labels", "obj_labels"):
        lab = targets[key]
        safe = lab.clamp(min=0)
        mapped = lookup[safe]
        out[key] = torch.where(lab >= 0, mapped, torch.full_like(lab, -1))
    return out


def train_one_epoch(args, model, criterion, dataset, loader, optimizer,
                    matcher, device, logger, epoch):
    model.train()
    meters = {}
    t0 = time.time()

    for item in tqdm(loader, desc=f"train[{epoch}/{args.max_epoch}]"):
        images = item["images"].to(device)
        outputs = model(images=images)

        # --- assign queries to ground-truth trajectories
        query_rows, gt_rows = matcher(
            outputs["pred_sbj_boxes"], item["gt_boxes"].to(device),
            item["gt_box_mask"].to(device))
        # The matcher ran over pair-level subject boxes; map those back onto the
        # kept-query indices so both sides of a pair share one assignment.
        pair_index = outputs["pair_index"]
        q_rows = [int(pair_index[r, 0]) for r in query_rows]

        targets = build_pair_targets(dataset, item, q_rows, gt_rows,
                                     pair_index, device)

        sliced, base_obj, _ = _slice_to_base(outputs, dataset, device)
        targets = _remap_labels(targets, base_obj, device)
        # rel_labels were already sliced to base by build_pair_targets.
        loss, parts = criterion(sliced, targets)

        if not torch.isfinite(loss):
            logger.warning(f"non-finite loss on {item['video_name']}, skipped")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(args, "clip_grad", 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        for k, v in parts.items():
            meters.setdefault(k, AverageMeter()).update(float(v))

    summary = "  ".join(f"{k}={m.avg:.4f}" for k, m in sorted(meters.items()))
    logger.info(f"epoch {epoch} ({time.time() - t0:.0f}s)  {summary}")
    return meters.get("total", AverageMeter()).avg



def train_metor(args, logger=None):
    """Train `--arch metor` end to end. Returns 0 on success."""
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from os.path import join as _join

    _torch.manual_seed(getattr(args, "seed", 3407))
    device = _device(args)
    paths.ensure_output_dirs()
    if logger is None:
        logger = get_logger(_join(paths.LOG_DIR, f"metor_{args.ps}_train.log"))
    logger.info(f"config files: {getattr(args, 'config_files', [])}")
    logger.info(f"args: {args}")

    train_ds = METORDataset(args, "train")
    train_loader = _DataLoader(train_ds, batch_size=1, shuffle=True,
                               collate_fn=metor_collate,
                               num_workers=getattr(args, "num_workers", 0))
    logger.info(f"train videos: {len(train_ds)}")

    model, criterion = build_metor(args)
    model = model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"parameters: {n_train:,} trainable / {n_total:,} total")

    optimizer = _torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr), weight_decay=getattr(args, "weight_decay", 0.01))
    scheduler = _torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, "lr_milestones", [15, 20, 25]),
        gamma=getattr(args, "lr_gamma", 0.1))
    matcher = TrajectoryHungarianMatcher()

    for epoch in range(args.start_epoch + 1, args.max_epoch + 1):
        train_one_epoch(args, model, criterion, train_ds, train_loader,
                        optimizer, matcher, device, logger, epoch)
        scheduler.step()
        _torch.save({"epoch": epoch, "config": args,
                     "state_dict": model.state_dict()},
                    _join(paths.CKPT_DIR, f"metor_{args.ps}_e{epoch}.pth"))

    logger.info("done")
    return 0
