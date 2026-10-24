"""Detector / object-classifier score fusion for EOV-MMP inference.

Released behaviour (`uniform=False`): per-class weights, as hard-coded in
`End2End_Model.forward` before RB1 --

    classes 1, 3, 8, 12, 14, 23      0.6 * classifier + 0.4 * detector
    classes 0, 2, 4, ..., 34 (below)  0.3 * classifier + 0.7 * detector

Two quirks of the released lists are preserved exactly: class 23 (red_panda)
is in both, so it is fused twice in sequence, and class 31 (turtle) is in
neither, so it keeps the raw detector score. The six 0.6/0.4 classes are all
novel categories (antelope, bear, cattle, fox, giant_panda, red_panda).

`uniform=True` (`--uniform_fusion`): every one of the 35 classes uses
0.3 * classifier + 0.7 * detector, once. Nothing is tuned.

`scores` is mutated in place, as the released code did, and returned.
"""
from __future__ import annotations

N_CLASSES = 35
HIGH_CLASSIFIER = [1, 3, 8, 12, 14, 23]
LOW_CLASSIFIER = [0, 2, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34]


def fuse_scores(similarity, scores, uniform: bool = False):
    if uniform:
        for nb in range(N_CLASSES):
            scores[nb] = similarity[nb] * 0.3 + scores[nb] * 0.7
        return scores
    for nb in HIGH_CLASSIFIER:
        scores[nb] = similarity[nb] * 0.6 + scores[nb] * 0.4
    for nb in LOW_CLASSIFIER:
        scores[nb] = similarity[nb] * 0.3 + scores[nb] * 0.7
    return scores
