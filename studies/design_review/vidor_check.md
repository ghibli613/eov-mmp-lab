# VidOR check — what would a VidOR evaluation cost?

For the plan's author: the decision on an A* paper with VidOR results, for the agreed
plan (relation-aware tracklet keeping on EOV-MMP).

**Rules followed:** reading and web lookups only. No download beyond web pages, no GPU,
no test-label use.

**Short answer:** affordable **only if the authors share their VidOR checkpoints**. Then
it's about 60–70 GPU-hours (≈ $25–45) and 3–5 weeks of engineering. Without them, no:
step-4 training alone on VidOR runs to thousands of dollars.

---

## 1. Released VidOR checkpoints — **none found**

| where | what is there | VidOR? |
|---|---|---|
| `github.com/wangyongqi558/EOV-MMP-VidVRD`, live README (and our clone, last commit 2025-07-25) | a Baidu link with "the datasets and models obtained from the first three training steps", and a Google Drive folder of "model weights and data files". All four to-do items unchecked (inference, detection training, relation training, end-to-end training). | **No mention of VidOR.** No issues filed. |
| the same Google Drive folder | two folders, `data` and `output` | no VidOR-named item |
| `github.com/wangyongqi558/METOR` | a README only: code "will follow afterwards", after the TPAMI paper; no checkpoints | no |
| `github.com/wangyongqi558/MMP_OV_VidVRD` (our clone) | `dataset/vidor/data/`: VidOR id maps, base/novel split files, a predicate split and `sample.json`. Trained models (Baidu) go to `dataset/vidvrd/model`. The README's VidOR section is "…". | metadata only; **no VidOR weights** |
| our local weights (`ov-vidvrd-lab/tools/weights_manifest.json`, `output/ckpt/`) | 8 files, all VidVRD | no |
| our RePro clone (`repro/`) | VidOR configs (`experiments/TrajCls_VidOR/*/cfg_.py`), VidOR text embeddings (84 KB) | no weights |
| our own docs | `docs/90_Research-ideas.md:354`: "VidOR checkpoints arriving from the author → both directions get much stronger" | VidOR weights were **hoped for, never received** |

**The one lead:** the EOV-MMP README invites code requests by email
(`1285441164yq@gmail.com`). The authors have already lent the VidVRD stage 1–3
checkpoints (`pilot_analysis/trainability-brief.md` §1). Asking for the VidOR set is the
only route to case (a).

## 2. Code support in the released code — **partial; VidOR does not run as shipped**

**Present:**
- **detector:** `dataset_file = "open_vidor"` builds VidOR text embeddings and
  seen/unseen lists (`ov_prompt/model.py:543-586`; `post_process.py:147-149`;
  `models/util/vidor_categories.py`);
- **relation classifier:** `ds2dim = {"vidor": 50, …}` (`relation_classifier.py:424`);
- **RePro's open-vocabulary split for VidOR:**
  `configs/VidOR_OjbectClass_spilt_info_v2.json` and `VidOR_PredClass_spilt_info_v2.json`
  (in EOV and METOR). That's 50 base / 30 novel objects and 30 base / 20 novel
  predicates (RePro, Appendix A.8);
- **a VidOR annotation reader** and `prepare_gts_for_vidor`
  (`third_party/vidvrd_ii_helper/`);
- **METOR's paper values for VidOR:** `METOR/configs/metor_vidor.yaml` ("nothing here
  has been run against VidOR", `docs/32_Not-implemented.md` §3).

**Missing, or hard-coded to VidVRD on the test path:**
1. **35 object classes, hard-coded:** `end2end_model.py:216`, `:311`;
   `gen_labels.py:612`.
2. **A VidVRD-specific class permutation and the per-class fusion lists:**
   `end2end_model.py:257-264`.
3. **132 predicates and 35 classes, hard-coded** in the relation classifier's novel test
   path: `relation_classifier.py:512-524`.
4. **The evaluator.** Its docstring says "only support for VidVRD currently"
   (`inference/video_relation_detection_openvoc.py:350`). It needs VidOR's split files
   and a VidOR-format GT built with `prepare_gts_for_vidor`.
5. **The data layer is VidVRD-only** (`utils/paths.py`; `Dataset_new`). VidOR needs, in
   the same formats:
   - object-trajectory GT;
   - id maps;
   - `prior.pkl`;
   - a **CLIP crop bank**, which the detector loads at build time (`model.py:274`);
   - **ECC camera matrices**, not shipped; without them the tracker runs with no camera
     compensation.
6. **Training:** no stage 1–3 code at all (as for VidVRD). Step 4 has the same
   hard-coding.

**Engineering, case (a):** about **3–5 weeks** to generalise the eight hard-coded places
as a VidOR fork, build the data files, and gate on reproducing EOV-MMP's VidOR numbers.

## 3. Data

**Source:** the official page (`xdshang.github.io/docs/vidor.html`):

| split | videos | download |
|---|---:|---|
| train | 7,000 | 8 parts, ≈ 24.5 GB |
| val | 835 | ≈ 2.9 GB |
| test | 2,165 | only "after participating the grand challenges"; **annotations not public** |

- **Contents:** 98.6 h in all, averaging ≈ 35 s per video. 80 object categories, 50
  predicates, ≈ 50k objects and ≈ 380k relation instances. Annotations are one JSON per
  video, for train and val.
- **Hosts:** Lark, Baidu and Hugging Face, with MD5s. The videos are also on AWS S3
  (Multimedia Commons).
- **Terms:** the page states **no licence**. The videos come from YFCC100M, where each
  video carries its uploader's Creative Commons licence, under Yahoo's Webscope research
  terms. VidOR's own annotation terms are **unconfirmed**.
- **Split evaluated:** **every paper evaluates on VidOR val.** EOV-MMP: "the
  annotations of the VidOR test set are not available". METOR, OpenVidVRD and RePro
  ("we only evaluated models on validation set") say the same.

## 4. Published VidOR numbers (VidOR val, SGDet unless stated; protocol A)

| method | novel mAP / R@50 / R@100 / mAP_o | all mAP / R@50 / R@100 / mAP_o | source |
|---|---|---|---|
| OV-MMP (AAAI'24) | 0.84 / 1.44 / 1.44 / 1.11 | 7.15 / 6.54 / 8.29 / 3.38 | METOR Table 1 |
| **EOV-MMP** (TPAMI'25) | **2.45** / 4.79 / 4.79 / 2.33 | **11.08** / 8.43 / 9.82 / 12.99 | METOR Table 1; EOV-MMP Table III |
| **METOR** (IJCAI'25) | **3.75** / 4.86 / 5.32 / 3.37 | **12.32** / 8.54 / 9.72 / 14.02 | METOR Table 1 |
| OpenVidVRD (arXiv'25) | 1.45 / 5.32 / 4.68 / — | 10.18 / 7.65 / 9.82 / — | its Table II; compares only to MMP |
| RePro (ICLR'23) | **no SGDet on VidOR.** PredCls R@50 / R@100 7.20 / 8.35 (novel), 27.11 / 35.76 (all); SGCls recall only | | RePro Tables 5 and 7 |

**Reading:**
- On VidOR, **METOR is the bar** (3.75 / 12.32).
- The numbers are tiny. EOV-MMP's novel *object* mAP_o is 2.33. The paper says SGDet on
  VidOR "is limited by the trajectory association due to more blurs and occlusions". A
  front-end keeping step is therefore **more relevant** on VidOR than on VidVRD, but
  differences of a few tenths will be hard to separate from noise, even over 835
  videos.

## 5. Cost estimates

**Basis:**
- D2's measured speed: 3.5 h for 200 VidVRD test videos, 52,104 encoded frames, so
  ≈ **0.24 s per frame**;
- the measured price, ≈ **$0.35/h** (pilot run); I also show $0.60/h;
- VidOR val is 835 videos × ≈ 35 s × ≈ 30 fps ≈ **870k frames, 16.7× VidVRD test.**
  VidOR's frame rates vary, so ±15%.
- **Hardware note:** the loader holds a whole video's per-frame features twice during
  `torch.cat`. That's about 5 GB transient for VidVRD's longest test video (1,234
  frames, `tools/setup_rented_box.sh`), and the pilot recommends ≥ 32 GB RAM. VidOR
  videos run to several minutes (a 3-minute video is about 20+ GB transient), so rent
  ≥ 64 GB RAM.

| case | GPU-h | $ at 0.35 / 0.60 per h | engineering | feasible? |
|---|---:|---:|---|---|
| **(a) inference only, with the authors' VidOR checkpoints, plus our keeping step** | val pass ≈ 58; keeping-step refit pass on an outside set ≈ 10; 2-video checks ≈ 1 → **≈ 65–75** | **≈ $23–26 / $39–45** | **3–5 weeks** (§2) | **yes, if the checkpoints exist and are shared** |
| (b) step 4 trained on VidOR from released stage 1–3 VidOR checkpoints | train ≈ 7,000 × 1,040 ≈ 7.3M frames per epoch. At VidVRD's estimated 0.37–0.7 s per training frame (review 3 §1.5), **≈ 750–1,420 h per epoch.** The released VidVRD model is epoch 6, so ≈ 4,500–8,500 h. | **≈ $1,600–3,000 / $2,700–5,100** for one model; a clean variant doubles it | case (a) plus 2–4 weeks | **no.** Moot anyway: no stage 1–3 VidOR checkpoints are released. |
| (c) nothing released: the whole pipeline on VidOR | (b), plus a detector (COCO, then VidOR), the object classifier and relation classifier stages, with **no training code** for stages 1–3 | **several thousand $**, as a floor | **3–6 months** | **no** |

The planned frame-stride sampling can't rescue (b) or (c) here: stride 30 destroys the
released VidVRD model (`PILOT-STATUS.md` B.20). A model *trained* at stride 30 is
untested here.

## 6. The conflict: VidOR as an evaluation set and as a check set

**If VidOR becomes an evaluation dataset, its val split must stay untouched** until a
pre-registered VidOR test session. It can't also be the keeping step's check set.

**The cleanest replacement is a VidOR *train* part.**
- One of the 8 train parts is ≈ 3 GB, roughly 875 videos, and comes from the official
  hosts with MD5s.
- **For the VidVRD stack, every VidOR video is outside data.** Every VidVRD component's
  saved config names VidVRD only (review 2, Q2).
- It carries **object and relation labels.** That gives an outside object-level check,
  and a partial relation-level one: the vocabularies overlap only in part.
- It's **disjoint from VidOR val** by construction.
- It **fixes ImageNet-VID's access problem** (review 2, Q2: the official host is gone).
- Its VidVRD-novel classes (e.g. horse, cattle, motorcycle) are used for suppression
  counts only, as before.

**For a VidOR evaluation with the authors' VidOR models,** the keeping step's refit set
must be outside *their* training data. VidVRD's training videos are a natural choice,
**if** the authors confirm their VidOR models never saw VidVRD. Ask that together with
the checkpoints.

**Caveat:** CLIP's web pretraining may include Flickr or YFCC content. That's the
standard, unavoidable caveat for every outside set.

---

## Bottom line

A VidOR evaluation is **affordable only if the authors share their trained VidOR
checkpoints.** That means the end-to-end model, ideally with the stage 1–3 checkpoints,
the VidOR CLIP crop bank and the ECC files.
- **With them:** about 65–75 GPU-hours (≈ $25–45) and 3–5 weeks of engineering. The
  work is generalising the eight VidVRD-hard-coded places, building VidOR data files,
  and a reproduction gate against EOV-MMP's published 2.45 / 11.08.
- **Without them:** no. Step 4 alone on VidOR's 7.3M training frames costs about
  $1,600–3,000 per model, and the full pipeline has no training code.
- **The cheapest credible route:**
  1. **Email the EOV-MMP author now** (the README's address). Ask for the VidOR
     checkpoints, the bank, the ECC files, and confirmation of the VidOR models'
     training data.
  2. Meanwhile, **switch the outside check set to one VidOR train part.** That removes
     the conflict and the ImageNet-VID problem.
  3. Keep the VidOR val split untouched for one pre-registered session.
- **Expectations:** the VidOR bar is METOR's 3.75 / 12.32, on very small numbers.

## Sources

- EOV-MMP repo and README: [github.com/wangyongqi558/EOV-MMP-VidVRD](https://github.com/wangyongqi558/EOV-MMP-VidVRD) (no issues listed); its [Google Drive folder](https://drive.google.com/drive/folders/1IH5fyfZN7on_DMsv55385scd0YrlueTO?usp=drive_link)
- METOR repo: [github.com/wangyongqi558/METOR](https://github.com/wangyongqi558/METOR)
- VidOR official page: [xdshang.github.io/docs/vidor.html](https://xdshang.github.io/docs/vidor.html)
- YFCC100M terms: [arXiv 1503.01817](https://arxiv.org/pdf/1503.01817); VidOR summary: [paperswithcode.com/dataset/vidor](https://paperswithcode.com/dataset/vidor)
- Papers (local PDFs, text extracted with PyMuPDF): EOV-MMP (arXiv 2409.12499v2, Table III and §IV-A), METOR (arXiv 2505.06663v1, Table 1), OpenVidVRD (arXiv 2503.09416, Table II), RePro (arXiv 2302.00268v1, Tables 5 and 7, Appendix A.8)
- Local code: EOV `fdeeeb4` and METOR `HEAD`, with the line references above

**Stop.** Reading and lookups only.
