#!/usr/bin/env python3
"""RPT G0 step 6, before the checkpoint is fetched: can the laptop GPU hold SAM 3's video model at
all? The model is built exactly as build_sam3_video_model() builds it (default settings), but with
random weights (load_from_HF=False): parameter memory does not depend on the weight values. No
video, no prompt, no data. Run in the SAM 3 environment:

    ~/miniconda3/envs/sam3/bin/python studies/rpt/g0/laptop_memory_probe.py
"""
import json
import time

import torch
from sam3.model_builder import build_sam3_video_model

GiB = 2 ** 30
free0, total = torch.cuda.mem_get_info()
out = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
           torch=torch.__version__, gpu_total_gib=total / GiB, gpu_free_before_gib=free0 / GiB)
t0 = time.time()
model = build_sam3_video_model(load_from_HF=False, checkpoint_path=None, device="cpu")
n = sum(p.numel() for p in model.parameters())
nbytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for b in model.buffers())
out.update(build_seconds=round(time.time() - t0, 1), parameters=n,
           param_dtypes=sorted({str(p.dtype) for p in model.parameters()}), weights_and_buffers_gib=nbytes / GiB)
try:
    model.cuda()
    torch.cuda.synchronize()
    out.update(to_cuda="ok", allocated_gib=torch.cuda.memory_allocated() / GiB,
               free_after_gib=torch.cuda.mem_get_info()[0] / GiB)
except torch.OutOfMemoryError as e:
    out.update(to_cuda="CUDA out of memory", error=str(e).splitlines()[0],
               max_allocated_gib=torch.cuda.max_memory_allocated() / GiB)
print(json.dumps(out, indent=1))
