import json

d = json.load(open('./test_object_trajectories_meta_d5.json', 'r'))

for vid in d:
    data = d[vid]
    tid = 0
    for obj_ in data:
        obj_['tid'] = tid
        tid += 1

with open('./test_object_trajectories_meta_d5.json', 'w') as f:
    json.dump(d, f)