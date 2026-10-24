import numpy as np 
import os
from os.path import join
import json
from collections import defaultdict

from tqdm import tqdm

from video_object_evaluation import evaluate

def prepare_gt(split='test'):
    gt_results = {}
    vid_list = os.listdir(join("..", 'anno', split))
    for vid_name in vid_list:
        video_anno = json.load(open(join("..", 'anno', split, vid_name),'r'))    
        tid2cat = {traj["tid"]:traj["category"] for traj in video_anno["subject/objects"]} # we need this
        trajs = defaultdict(dict)
        
        for fid, frame in enumerate(video_anno["trajectories"]):
            for bbox_anno in frame:
                tid = bbox_anno["tid"]
                bbox = bbox_anno["bbox"]
                bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                trajs[tid][int(fid)] = bbox

        result_per_video = []
        for tid in trajs.keys():

            # ### discontinuous traj
            fids = sorted(trajs[tid].keys())
            bboxes = [trajs[tid][fid] for fid in fids]
            head = 0
            tail = -1
            for fno, fid in enumerate(fids):
                if fno > 0 and fid != fids[fno-1] + 1:
                    tail = fno
                    result_per_video.append({
                            "category":tid2cat[tid],
                            "trajectory":{fid:box for fid,box in zip(fids[head:tail],bboxes[head:tail])}
                        })
                    head = fno
            result_per_video.append({
                    "category":tid2cat[tid],
                    "trajectory":{fid:box for fid,box in zip(fids[head:],bboxes[head:])}
                })

            ### continuous traj
            # result_per_video.append({
            #         "category":tid2cat[tid],
            #         "trajectory":trajs[tid]
            #     })

        vid_name = vid_name.split(".")[0]
        gt_results[vid_name] = result_per_video
    return gt_results

def gen_gt_trajs(split='test'):
    gt_results = {}
    vid_list = os.listdir(join("..", 'anno', split))
    for vid_name in vid_list:
        video_anno = json.load(open(join("..", 'anno', split, vid_name), 'r'))    
        tid2cat = {traj["tid"]:traj["category"] for traj in video_anno["subject/objects"]} # we need this
        trajs = defaultdict(dict)
        
        for fid,frame in enumerate(video_anno["trajectories"]):
            for bbox_anno in frame:
                tid = bbox_anno["tid"]
                bbox = bbox_anno["bbox"]
                bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                trajs[tid][fid] = bbox

        result_per_video = []
        for tid in trajs.keys():
            fids = sorted(trajs[tid].keys())
            bboxes = [trajs[tid][fid] for fid in fids]
            head = 0
            tail = -1
            for fno, fid in enumerate(fids):
                if fno > 0 and fid != fids[fno-1] + 1:
                    tail = fno
                    result_per_video.append({
                            "category":tid2cat[tid],
                            "tid":tid,
                            "trajectory":{fid:box for fid,box in zip(fids[head:tail],bboxes[head:tail])},
                            "begin_fid":fids[head],
                            "end_fid":fids[tail-1] + 1,
                            "score":1.0
                    })
                    head = fno
            result_per_video.append({
                    "category":tid2cat[tid],
                    "tid":tid,
                    "trajectory":{fid:box for fid,box in zip(fids[head:],bboxes[head:])},
                    "begin_fid":fids[head],
                    "end_fid":fids[-1] + 1,
                    "score":1.0
            })
        
        for r in result_per_video:
            fids = sorted(r["trajectory"].keys())
            assert len(fids) == r["end_fid"] - r["begin_fid"]

        vid_name = vid_name.split(".")[0]
        gt_results[vid_name] = result_per_video
    return gt_results

   
def interp(fids, bboxes):
    track = {fid:box for fid,box in zip(fids,bboxes)}
    for i, fid in enumerate(fids):
        if i > 0 and (fid != fids[i-1]+1):
            head = fids[i-1]
            tail = fid
            for k in range(head + 1, tail):
                interpolated = list(map(lambda c: np.interp(k, [head, tail], [track[head][c], track[tail][c]]), range(4)))
                track[k] = interpolated
    fids = sorted(track.keys())
    bboxes = [track[fid] for fid in fids]
    return fids, bboxes
    
def gen_vidvrd_ii_trajs(raw_trajs):
    for vid in raw_trajs:
        for traj in raw_trajs[vid]:
            tem_traj = {}
            for fid in traj['trajectory']:
                tem_traj[int(fid)] = [round(traj['trajectory'][fid][i]) for i in range(len(traj['trajectory'][fid]))]
            traj['trajectory'] = tem_traj
            fids = sorted(tem_traj.keys())
            begin_fid = fids[0]
            end_fid = fids[-1] + 1
            assert len(fids) == (end_fid - begin_fid)
    return raw_trajs

def gen_openvoc_trajs(raw_trajs):
    trajs = defaultdict(list)
    cats = set()
    for vid in raw_trajs:
        for traj in raw_trajs[vid]:
            score = traj['det_score']
            category = traj['det_cls'].replace(' ', '_')
            begin_fid = traj['fstart']
            end_fid = traj['fend']
            trajectory = {}
            for fid in traj['bboxes']:
                trajectory[int(fid)] = [round(traj['bboxes'][fid][i]) for i in range(len(traj['bboxes'][fid]))]
            fids = sorted(trajectory.keys())
            begin_fid = fids[0]
            end_fid = fids[-1] + 1
            assert len(fids) == (end_fid - begin_fid)
            if score > 0.6:
                cats.add(category)
                trajs[vid].append({
                    "category":category,
                    "score":score,
                    "trajectory":trajectory,
                    'begin_fid':begin_fid,
                    'end_fid':end_fid
                })

    print(len(list(cats)), len(trajs), cats)
    return trajs

# gt_trajs_train = gen_gt_trajs("train")
# with open("train_object_trajectories_gt.json", "w") as f:
#     json.dump(gt_trajs_train, f)

gt_trajs_test = gen_gt_trajs("test")
with open("test_object_trajectories_gt.json", "w") as f:
    json.dump(gt_trajs_test, f)

# vidvrd_ii_results = json.load(open('./raw_trajs/tracking_results_nms_0.7_score_0.0_frames_aligned_21.9.json','r'))
# vidvrd_ii_results = gen_vidvrd_ii_trajs(vidvrd_ii_results)
# with open("test_object_trajectories_vidvrd-ii-2.json", "w") as f:
#     json.dump(vidvrd_ii_results, f

# openvoc_raw_trajs = json.load(open('/media/sda1/jixf/OpenVoc-VidVRD/data0/vidvrd_test_traj.json', 'r')) 
# trajs = gen_openvoc_trajs(openvoc_raw_trajs)
# with open("./test_object_trajectories_openvoc3.json", "w") as f:
#     json.dump(trajs, f)

# trajs = json.load(open('./test_object_trajectories_openvoc3.json', 'r'))
# print(sum([len(trajs[vid]) for vid in trajs]))
# gt = json.load(open('./test_object_trajectories_gt.json', 'r'))
# print(sum([len(gt[vid]) for vid in gt]))
# evaluate(gt, trajs)

# openvoc 0.2
# openvoc2 0.3
# openvoc3 0.6