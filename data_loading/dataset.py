import os
import numpy as np
import pickle
import _pickle as cPickle
import json
from collections import defaultdict
from os.path import join

from utils import paths
import random
from PIL import Image

from vlm.backbones import clip
from vlm.backbones import clip_tagclip
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import copy

_ENCODER_CACHE = {}


def _shared_encoder(kind: str, backbone: str, device: str = "cuda"):
    """One copy of each frozen encoder per process.

    cli/train.py builds a train and a val dataset, and each used to load its own
    CLIP-L *and* TagCLIP-L onto the GPU -- four copies of two frozen models, the
    5.32 GB measured in docs/10_Known-issues.md. They are frozen and stateless, so
    one copy each is enough. Loading is deferred to first use so that tooling
    which only needs the annotations never pays for it.
    """
    key = (kind, backbone, device)
    if key not in _ENCODER_CACHE:
        loader = clip if kind == "clip" else clip_tagclip
        _ENCODER_CACHE[key] = loader.load(backbone, device=device)[0]
    return _ENCODER_CACHE[key]

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        #Resize(n_px, interpolation=BICUBIC),
        Resize((h,w), interpolation=BICUBIC),
        # CenterCrop(224),
        #RandomHorizontalFlip(1.0),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class Dataset_new(BaseDataset):

    def __init__(self, args, split):
        super().__init__()
        self.dataset = args.dataset
        self.split = split
        data_dir = paths.META_DIR
        self.anno_train_dir = paths.ANNO_TRAIN_DIR
        if split=='train':
            feat_path = join(data_dir, 'train_object_trajectories_gt.json')
        elif split=='val':
            self.gt_rels = json.load(open(join(data_dir, "test_relation_gt.json"), "r"))
            feat_path = join(data_dir, 'test_object_trajectories_gt.json')

        self.FEAT_ROOT = paths.FRAME_DIR
        self.gt_traj = json.load(open(feat_path,"r"))
        self.path_list = list(self.gt_traj.keys())
        self.pre_split = json.load(open(
            join(data_dir, 'openvoc_pred_class_spilt_info.json'), "r"))
        self.obj_split = json.load(open(
            join(data_dir, 'openvoc_obj_class_spilt_info.json'), "r"))
        self.pre_num = len(self.pre_split['id2cls'])
        self.prior = pickle.load(open(join(data_dir, 'prior.pkl'), 'rb'))
        self.preprocess = _transform_resize(336, 336)
        backbone = getattr(args, "clip_backbone", "ViT-L/14@336px")
        self.clip = _shared_encoder("clip", backbone)
        self.tagclip = _shared_encoder("tagclip", backbone)
        self.frame_stride = getattr(args, "frame_stride", 1)
        self.id2pre = json.load(open(join(data_dir, 'id2predicate.json'), "r"))
        self.pre2id = json.load(open(join(data_dir, 'predicate2id.json'), "r"))
        self.id2obj = json.load(open(join(data_dir, 'id2object.json'), "r"))
        self.obj2id = json.load(open(join(data_dir, 'object2id.json'), "r"))

    def __getitem__(self, index):
        video_name = self.path_list[index]
        data_path = self.FEAT_ROOT + '/' + video_name
        frame_list = sorted(os.listdir(data_path))
        video_len = len(frame_list)
        item = {}
        patch_ = []
        patch_proj = []
        global_proj = []
        gp = p = pp = None
        for i in frame_list:
            # Frame files are 1-indexed (docs/03_Data.md), so frame 1 is always
            # encoded and later frames reuse it until the next stride boundary.
            frame_no = int(i.split('.')[0])
            encode_this_frame = (frame_no - 1) % self.frame_stride == 0 or gp is None
            if encode_this_frame:
                with torch.no_grad():
                    frame_path = data_path + '/' + i
                    image = Image.open(frame_path).convert("RGB")
                    image_w, image_h = image.size
                    image_resize = self.preprocess(image).unsqueeze(0).cuda()
                    _, gp = self.clip.encode_image(image_resize)
                    gp = gp.cpu()
                    p, pp = self.tagclip.encode_image_tagclip(
                        image_resize, 336, 336, attn_mask=1)
                    p = p.cpu()
                    pp = pp.cpu()
            # Otherwise reuse the last encoded frame's features. image_w/image_h
            # are constant within a video, so they carry over unchanged.
            global_proj.append(gp)
            patch_.append(p)
            patch_proj.append(pp)
        begin_fid = video_len
        end_fid = 0
        for t in self.gt_traj[video_name]:
            if t['begin_fid'] < begin_fid:
                begin_fid = t['begin_fid']
            if t['end_fid'] > end_fid:
                end_fid = t['end_fid']
        item['video_name'] = video_name
        item['video_len'] = video_len
        item['patch_'] = torch.cat(patch_,dim=0)
        item['patch_proj'] = torch.cat(patch_proj,dim=0)
        item['global_proj'] = torch.cat(global_proj,dim=0)
        item['image_size'] = [image_w,image_h]
        item['gt_traj'] = self.gt_traj[video_name]
        item['begin_fid'] = begin_fid
        item['end_fid'] = end_fid

        if self.split == 'train':
            anno_path = join(self.anno_train_dir, video_name + '.json')
            anno = json.load(open(anno_path,"r"))
            gt_rel = anno['relation_instances']
            item['gt_rel'] = gt_rel
        
        return item

    def __len__(self):
        return len(self.path_list)
