"""Dataset for `--arch metor`: pixels and trajectory targets.

Narrative walkthrough: docs/22_Data-pipeline.md

Why a second dataset
--------------------
EOV-MMP's `Dataset_new` is an *encoder*, not a loader: `__getitem__` returns
CLIP features because the detector's input is patch tokens. METOR cannot use it
for two reasons.

1. `--cre_mode inject` (Eq. 7) appends learnable tokens to the CLIP ViT's
   **input**, so the ViT has to run inside the training step with gradients
   reaching those tokens. Pre-encoded features are already past that point.
2. METOR regresses trajectories directly (Eq. 5), so the loss needs per-frame
   ground-truth boxes for both entities of every pair. EOV got trajectories from
   a tracker and never needed them as targets.

So this returns frames and annotations, and the model does its own encoding.

Sampling
--------
"For all experiments, key frames are sampled every 30 video frames to form
30-frame video segments" (Sec. 4.2). One timestep `t` of the model is one key
frame; a video of 160 frames at `--frame_stride 30` gives T = 6. Every target
here is built on that same key-frame grid, so `t` means the same thing in the
model, the losses and the evaluator.

Coordinates
-----------
VidVRD's frames are 640x360 and its annotations are in that same pixel space
(verified, not assumed). Boxes are normalised by the frame size and converted to
`cxcywh`, matching `TrajectoryRegressionHead`'s sigmoid output and Deformable-
DETR's convention.
"""
from __future__ import annotations

import json
import os
from os.path import join
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode

from utils import paths

BICUBIC = InterpolationMode.BICUBIC

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_transform(size: int = 336):
    return Compose([
        Resize((size, size), interpolation=BICUBIC),
        lambda im: im.convert("RGB"),
        ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])


def xyxy_to_cxcywh_normalised(box, w: float, h: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = x1 / w, x2 / w
    y1, y2 = y1 / h, y2 / h
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


class METORDataset(BaseDataset):
    """One item = one video: key frames, trajectories, and relation triplets.

    Returns plain Python containers alongside tensors, so use `metor_collate`
    with `batch_size=1` -- the paper trains on one video at a time (Sec. 4.2),
    and videos have different lengths and different numbers of trajectories,
    which no default collate can stack.
    """

    def __init__(self, args, split: str, image_size: int = 336):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.split = split
        self.args = args
        self.frame_stride = int(getattr(args, "frame_stride", 30))
        self.transform = clip_transform(image_size)
        self.frame_root = paths.FRAME_DIR

        meta = paths.META_DIR
        traj_file = ("train_object_trajectories_gt.json" if split == "train"
                     else "test_object_trajectories_gt.json")
        self.gt_traj = json.load(open(join(meta, traj_file)))
        self.video_names = sorted(self.gt_traj.keys())

        self.anno_dir = paths.ANNO_TRAIN_DIR if split == "train" else paths.ANNO_TEST_DIR

        obj_split = json.load(open(paths.OBJ_SPLIT_INFO))
        rel_split = json.load(open(paths.PRED_SPLIT_INFO))
        self.obj2id: Dict[str, int] = obj_split["cls2id"]
        self.rel2id: Dict[str, int] = rel_split["cls2id"]
        self.obj_cls2split = obj_split["cls2split"]
        self.rel_cls2split = rel_split["cls2split"]
        self.n_obj = len(obj_split["id2cls"])
        self.n_rel = len(rel_split["id2cls"])

        # Training sees base categories only -- that is what "open-vocabulary"
        # means here, and leaking a novel category into training invalidates
        # every novel-split number the model later produces.
        self.base_rel_ids = sorted(self.rel2id[c] for c in self.rel2id
                                   if self.rel_cls2split[c] == "base")
        self.base_obj_ids = sorted(self.obj2id[c] for c in self.obj2id
                                   if self.obj_cls2split[c] == "base")
        self.rel_id_to_base = {g: i for i, g in enumerate(self.base_rel_ids)}

    def __len__(self):
        return len(self.video_names)

    # ------------------------------------------------------------------ frames
    def _key_frames(self, video_name: str):
        """Frame filenames on the key-frame grid, and their 1-indexed numbers.

        Frame files are 1-indexed (`000001.jpg`), so the grid is
        `(frame_no - 1) % stride == 0` -- frame 1 is always a key frame. This
        matches the convention `data_loading/feature_dataset.py` uses, deliberately: a
        half-frame offset between the two would make any comparison meaningless.
        """
        d = join(self.frame_root, video_name)
        names = sorted(os.listdir(d))
        keys = [(n, int(n.split(".")[0])) for n in names]
        return [(n, fno) for n, fno in keys if (fno - 1) % self.frame_stride == 0]

    # ------------------------------------------------------------------- items
    def __getitem__(self, index):
        video_name = self.video_names[index]
        key_frames = self._key_frames(video_name)
        if not key_frames:
            raise RuntimeError(f"no frames found for {video_name}")

        images, frame_ids = [], []
        width = height = None
        for name, fno in key_frames:
            img = Image.open(join(self.frame_root, video_name, name))
            if width is None:
                width, height = img.size
            images.append(self.transform(img))
            # Trajectory keys are 0-indexed while filenames are 1-indexed.
            frame_ids.append(fno - 1)

        images = torch.stack(images)                       # (T, 3, H, W)
        t = images.shape[0]

        trajs = self.gt_traj[video_name]
        n_traj = len(trajs)

        # --- per-trajectory boxes on the key-frame grid
        boxes = torch.zeros(n_traj, t, 4)
        box_mask = torch.zeros(n_traj, t, dtype=torch.bool)
        traj_labels, tid_to_row = [], {}
        for row, traj in enumerate(trajs):
            tid_to_row[traj["tid"]] = row
            category = traj["category"]
            if isinstance(category, list):                 # some copies wrap it
                category = category[0]
            traj_labels.append(self.obj2id.get(category, -1))
            table = traj["trajectory"]
            for ti, fid in enumerate(frame_ids):
                box = table.get(str(fid))
                if box is None:
                    continue
                if isinstance(box[0], list):
                    box = [box[0][0], box[1][0], box[2][0], box[3][0]]
                boxes[row, ti] = torch.tensor(
                    xyxy_to_cxcywh_normalised(box, width, height))
                box_mask[row, ti] = True

        item = {
            "video_name": video_name,
            "images": images,
            "frame_ids": frame_ids,
            "image_size": (width, height),
            "gt_boxes": boxes,
            "gt_box_mask": box_mask,
            "gt_labels": torch.tensor(traj_labels, dtype=torch.long),
            "tid_to_row": tid_to_row,
            "num_frames": t,
        }
        item.update(self._relation_targets(video_name, frame_ids, tid_to_row, t))
        return item

    def _relation_targets(self, video_name, frame_ids, tid_to_row, t):
        """Per-pair, per-timestep predicate labels plus the frame category sets.

        `pair_labels` is indexed by `(subject_row, object_row)` so the training
        loop can look up whatever pairing the matcher produced, rather than
        committing here to a pair ordering the model has not chosen yet.

        `present_obj` / `present_rel` are the multi-hot sets Eq. 16 supervises
        the context tokens against. The paper defines them as "the sets of
        relationship and object categories present in each frame"; they are
        pooled to the video here because `C_o` and `C_r` reach the losses
        video-pooled (see RefinedTextEncoder.forward for that reading).
        """
        anno_path = join(self.anno_dir, video_name + ".json")
        pair_labels: Dict[tuple, torch.Tensor] = {}
        present_rel = torch.zeros(self.n_rel)
        present_obj = torch.zeros(self.n_obj)

        if os.path.exists(anno_path):
            anno = json.load(open(anno_path))
            for so in anno.get("subject/objects", []):
                oid = self.obj2id.get(so["category"], -1)
                if oid >= 0:
                    present_obj[oid] = 1.0

            for inst in anno.get("relation_instances", []):
                pid = self.rel2id.get(inst["predicate"], -1)
                if pid < 0:
                    continue
                present_rel[pid] = 1.0
                s_row = tid_to_row.get(inst["subject_tid"])
                o_row = tid_to_row.get(inst["object_tid"])
                if s_row is None or o_row is None:
                    continue
                key = (s_row, o_row)
                if key not in pair_labels:
                    pair_labels[key] = torch.zeros(t, self.n_rel)
                begin, end = inst["begin_fid"], inst["end_fid"]
                for ti, fid in enumerate(frame_ids):
                    if begin <= fid < end:
                        pair_labels[key][ti, pid] = 1.0

        return {
            "pair_labels": pair_labels,
            "present_obj": present_obj,
            "present_rel": present_rel,
        }


def metor_collate(batch):
    """Batch size 1 only -- unwrap rather than stack.

    The paper trains one video at a time (Sec. 4.2, "batch size is set to 1,
    which means that only one video is processed at a time"), and the shapes
    here are genuinely ragged: different T, different trajectory counts,
    different pair sets. Failing loudly beats silently collating a batch of 2
    into something that trains and means nothing.
    """
    if len(batch) != 1:
        raise ValueError(
            f"METORDataset requires batch_size=1, got {len(batch)}. "
            "The paper trains on one video at a time (Sec. 4.2).")
    return batch[0]
