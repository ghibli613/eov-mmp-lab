import gensim
import numpy as np
import json
import pickle

w2v_path = '/media/sda1/jixf/data/GoogleNew/GoogleNews-vectors-negative300.bin' 
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
obj_ind2name = json.load(open("./id2object.json", "r"))
obj_vecs = np.zeros((len(obj_ind2name), 300))
for i in range(len(obj_ind2name)):
    obj_label = obj_ind2name[i]
    obj_label = obj_label.split('/')[0]

    if obj_label == 'domestic_cat':
        obj_label = 'cat'
    if obj_label == 'red_panda':
        obj_label = 'panda' # no red panda in GoogleNews

    vec = w2v_model[obj_label]
    if vec is None or len(vec) == 0 or np.sum(vec) == 0:
        print('[WARNING] %s' % obj_label)
    obj_vecs[i] = vec
with open("./object_vectors.pkl", 'wb') as f:
    pickle.dump(obj_vecs, f)