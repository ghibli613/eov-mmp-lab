import numpy as np 
import os
from os.path import join
import pickle
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, val=0):
        self.reset(val)

    def reset(self, val=0):
        self.avg = val
        self.count = 0

    def update(self, val, n=1):
        assert n > 0
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n

feat_set = "train_gt_30"
feat_path = join("..", "feature", feat_set)
feat_pairs = os.listdir(feat_path)

avg_feat_meter = {}
sample = pickle.load(open(join("..", "feature", feat_set, feat_pairs[0]), "rb"))[0]
for type_ in sample:
    *feat_shape, = sample[type_][0].shape
    print(type_, feat_shape)
    avg_feat_meter[type_] = AverageMeter(np.zeros(feat_shape,))

for pair_path in tqdm(feat_pairs):
    pair_feat = pickle.load(open(join("..","feature", feat_set, pair_path), "rb"))[0]
    for type_ in pair_feat:
        clip_num = len(pair_feat[type_])
        pair_feat_avg = np.array(pair_feat[type_]).mean(axis=0)
        avg_feat_meter[type_].update(pair_feat_avg, clip_num)

avg_feat = {}
for type_ in avg_feat_meter:
    avg_feat[type_] = avg_feat_meter[type_].avg
with open(join("..", "feature", "%s_avg.pkl"%feat_set),'wb') as f:
    pickle.dump(avg_feat, f)


