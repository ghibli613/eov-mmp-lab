import pickle
import os
from os.path import join
import json

import numpy as np


prior = np.zeros((35,35,132))
object2id = json.load(open('./object2id.json','r'))
predicate2id = json.load(open('./predicate2id.json','r'))

vid_list = os.listdir(join("..", 'anno', 'train'))
for vid_name in vid_list:
    vid_anno = json.load(open(join("..", 'anno', 'train', vid_name),'r'))
    tid2object = {traj["tid"]:traj["category"] for traj in vid_anno["subject/objects"]}
    for rel in vid_anno["relation_instances"]:
        sbj_id = object2id[tid2object[rel["subject_tid"]]]
        obj_id = object2id[tid2object[rel["object_tid"]]]
        rel_id = predicate2id[rel["predicate"]]
        prior[sbj_id, obj_id, rel_id] += 1

for i in range(35):
    for j in range(35):
        prior[i,j] = prior[i,j]/(np.sum(prior[i,j]) + 10e-8)

with open('./prior.pkl','wb') as f:
    pickle.dump(prior,f)
