# Equation index

Every equation in the paper → where it lives. The `# ---- Eq. N` comments in the
source are the durable anchors if a line number drifts:

```bash
grep -rn "Eq\. 11" metor/
```

---

## Section 3.2 — the framework

| Eq. | Statement | Implemented | Doc |
|---|---|---|---|
| **1** | `(H_CLS, H_patch, T_o, T_r, Q̂, C_o, C_r) = Φ(V, Q)` | called [model.py:197](../metor/model.py#L197) · class `ContextualRefinementEncoder` [contextual_encoding.py:409](../metor/modules/contextual_encoding.py#L409) | [11](11_Contextual-refinement-encoding.md) |
| **2** | `O = Decoder(Q̂, H_patch)` | called [model.py:211](../metor/model.py#L211) · class `VanillaQueryDecoder` [model.py:52](../metor/model.py#L52) | [12](12_Transformer-decoder-and-queries.md) |
| **3** | `(Ô_s, Ô_o, R̂) = Ψ(O, H_CLS)` | called [model.py:221](../metor/model.py#L221) · class `IterativeEnhancementModule` [iterative_enhancement.py:181](../metor/modules/iterative_enhancement.py#L181) | [13](13_Iterative-enhancement.md) |
| **4** | `S = σ(cos(·, T))` | called [model.py:226](../metor/model.py#L226) · class `CosineScoreHead` [heads.py:43](../metor/modules/heads.py#L43) | [14](14_Scoring-and-trajectory-heads.md) |
| **5** | `τ = M_b(Ô)` | called [model.py:239](../metor/model.py#L239) · class `TrajectoryRegressionHead` [heads.py:12](../metor/modules/heads.py#L12) | [14](14_Scoring-and-trajectory-heads.md) |

## Section 3.3 — contextual refinement encoding

| Eq. | Statement | Implemented | Doc |
|---|---|---|---|
| **6** | `h_patch = L(P(V))` | `conv1` (patchify + project), [contextual_encoding.py:111](../metor/modules/contextual_encoding.py#L111) | [11](11_Contextual-refinement-encoding.md) |
| **7** | `(H_CLS, H_patch, C_o, C_r) = V([h_CLS; h_patch; c_o; c_r])` | `ContextTokenInjector.forward` [contextual_encoding.py:102](../metor/modules/contextual_encoding.py#L102) | [11 §3.1](11_Contextual-refinement-encoding.md) |
| **8** | `Q̂ = MHA(Q, C_o, C_o) + Q` | `ObjectQueryRefiner` [contextual_encoding.py:176](../metor/modules/contextual_encoding.py#L176) | [11 §3.2](11_Contextual-refinement-encoding.md) |
| **9** | `T_o = G([v_o; M_o(C_o); OBJ])` | `RefinedTextEncoder.forward` [contextual_encoding.py:359](../metor/modules/contextual_encoding.py#L359) | [11 §3.3](11_Contextual-refinement-encoding.md) |
| **10** | `T_r = G([v_r; M_r(C_r); REL])` | same; `token_position` branch at [line 371](../metor/modules/contextual_encoding.py#L371) | [11 §3.3](11_Contextual-refinement-encoding.md) |

`M_o` / `M_r` are `ContextMappingLayer`,
[contextual_encoding.py:218](../metor/modules/contextual_encoding.py#L218).

## Section 3.4 — iterative enhancement

| Eq. | Statement | Implemented | Doc |
|---|---|---|---|
| **11** | the three-line update | [iterative_enhancement.py:171](../metor/modules/iterative_enhancement.py#L171) | [13](13_Iterative-enhancement.md) |

Sub-parts: `STTrans⁽ᵏ⁾` = `SpatioTemporalTransformer`
[line 48](../metor/modules/iterative_enhancement.py#L48); `M_f⁽ᵏ⁾` = `.mapping`
[line 147](../metor/modules/iterative_enhancement.py#L147); one layer =
`IterativeEnhancementLayer` [line 115](../metor/modules/iterative_enhancement.py#L115).

## Section 3.5 — training objective

| Eq. | Statement | Implemented | Doc |
|---|---|---|---|
| **12** | `L = L_rel-ctr + L_obj-ctr + θ_traj L_traj + θ_ctx(…)` | [criterion.py:160](../metor/modules/criterion.py#L160) | [15 §1](15_Training-objective.md) |
| **13** | `L_rel-ctr = (1/|C^b_r|) BCE(S_r, r̃)` | [criterion.py:53](../metor/modules/criterion.py#L53) | [15 §2](15_Training-objective.md) |
| **14** | `L_obj-ctr = CE(S_s, s̃) + CE(S_o, õ)` | [criterion.py:89](../metor/modules/criterion.py#L89) | [15 §3](15_Training-objective.md) |
| **15** | `L_traj = L_box + θ_cst L_cst` | [criterion.py:106](../metor/modules/criterion.py#L106) | [15 §4](15_Training-objective.md) |
| **16** | `L_ctx = BCE(cos(C, T), Ỹ)` | [criterion.py:139](../metor/modules/criterion.py#L139) | [15 §5](15_Training-objective.md) |

---

## Not in the paper, but required to run

| Piece | Where | Why it exists | Doc |
|---|---|---|---|
| query ↔ GT trajectory matching | `TrajectoryHungarianMatcher` [matcher.py:58](../metor/matcher.py#L58) | nothing says which query is the dog | [16 §1](16_Matching-and-pair-formation.md) |
| subject–object pair selection | `METOR.select_pairs` [model.py:173](../metor/model.py#L173) | paper says pairs are "derived", never how | [16 §2](16_Matching-and-pair-formation.md) |
| loss target assembly | `build_pair_targets` [matcher.py:126](../metor/matcher.py#L126) | turns an assignment into tensors | [16 §3](16_Matching-and-pair-formation.md) |
| base-vocabulary slicing | `_slice_to_base` [engine.py:46](../metor/engine.py#L46) | novel categories must never be supervised | [14 §2](14_Scoring-and-trajectory-heads.md) |
| label remapping | `_remap_labels` [engine.py:67](../metor/engine.py#L67) | ids must line up with the sliced columns | [14 §2](14_Scoring-and-trajectory-heads.md) |
| the training loop | `train_one_epoch` [engine.py:80](../metor/engine.py#L80) | | [15 §6](15_Training-objective.md) |

---

## Paper section → document

| Paper | Document |
|---|---|
| Sec. 3.1 problem definition | [00_Start-here](00_Start-here.md) |
| Sec. 3.2 overview | [10_Pipeline-overview](10_Pipeline-overview.md) |
| Sec. 3.3 contextual refinement | [11](11_Contextual-refinement-encoding.md) |
| Sec. 3.4 iterative enhancement | [13](13_Iterative-enhancement.md) |
| Sec. 3.5 training objective | [15](15_Training-objective.md) |
| Sec. 4.1 dataset and evaluation | [22_Data-pipeline](22_Data-pipeline.md) |
| Sec. 4.2 implementation details | [23_Configuration-and-flags](23_Configuration-and-flags.md) |
| Sec. 4.3–4.4 results and ablations | [33_Verification-status](33_Verification-status.md) |
