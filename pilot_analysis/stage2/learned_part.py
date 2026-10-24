"""Stage 2 -- apply a trained learned part (step2_train.py's model_*.pt) to new slots:
the stored input normalisation (fit-train statistics), each seed's network, the
mean of the seeds' sigmoid outputs. Reproduces step 2's own outputs exactly."""
from __future__ import annotations

import numpy as np


def inputs(emb, extra, norm):
    """step2_train.build_inputs with the stored statistics (no refit)."""
    flags = np.isnan(extra).astype(np.float32)
    z = np.clip((extra - norm["extra_median"]) / norm["extra_iqr"], -5, 5)
    z = np.where(np.isnan(z), 0.0, z).astype(np.float32)
    X = np.concatenate([emb.astype(np.float32), z, flags], axis=1)
    return ((X - norm["mean"]) / norm["std"]).astype(np.float32)


def apply(model_path, emb, extra):
    import torch
    m = torch.load(model_path, weights_only=False)
    X = torch.from_numpy(inputs(emb, extra, m["norm"]))
    outs = []
    for st in m["states"]:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], m["arch"]["hidden"]), torch.nn.ReLU(),
                                  torch.nn.Dropout(m["arch"]["dropout"]), torch.nn.Linear(m["arch"]["hidden"], len(m["outputs"])))
        net.load_state_dict(st)
        net.eval()
        with torch.no_grad():
            outs.append(torch.sigmoid(net(X)).numpy())
    return np.mean(outs, 0).astype(np.float32), list(m["outputs"])
