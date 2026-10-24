# Documentation index

A reconstruction of METOR (arXiv 2505.06663, IJCAI 2025) built from the paper,
because the authors never released code.

Documents are grouped by what you are trying to do. Nothing here repeats
anything else — if two documents touch the same subject, one owns it and the
other links.

---

## Reading paths

**"I am new to this project."**
[00_Start-here](00_Start-here.md) → [10_Pipeline-overview](10_Pipeline-overview.md) → [21_Repository-layout](21_Repository-layout.md)

**"I want to understand how the method works."**
[10_Pipeline-overview](10_Pipeline-overview.md), then the component documents in
execution order: [11](11_Contextual-refinement-encoding.md) → [12](12_Transformer-decoder-and-queries.md) → [13](13_Iterative-enhancement.md) → [14](14_Scoring-and-trajectory-heads.md) → [15](15_Training-objective.md)

**"I want to follow the paper's diagrams and find each box in the code."**
The figures are embedded in the component documents, and every drawn element is
traced to its variable, shape, equation and line:
[Fig. 2 → doc 10](10_Pipeline-overview.md) ·
[Fig. 3 → doc 11](11_Contextual-refinement-encoding.md) ·
[Fig. 4 → doc 13](13_Iterative-enhancement.md)

**"I am looking for a specific equation or class."**
[20_Equation-index](20_Equation-index.md)

**"Can I trust a number this produces?"**
[33_Verification-status](33_Verification-status.md) → [30_Paper-gaps](30_Paper-gaps.md) → [31_Lineage-evidence](31_Lineage-evidence.md) → [32_Not-implemented](32_Not-implemented.md)

**"I want to run or train it."**
[40_Hardware](40_Hardware.md) → [23_Configuration-and-flags](23_Configuration-and-flags.md) → [22_Data-pipeline](22_Data-pipeline.md)

**"I want to change something."**
[42_Extending](42_Extending.md) → [41_Testing](41_Testing.md)

---

## All documents

### 00 — Orientation
| | |
|---|---|
| [00_Start-here](00_Start-here.md) | What this is, what it can and cannot do, the two architectures |

### 10 — Architecture, in execution order
| | |
|---|---|
| [10_Pipeline-overview](10_Pipeline-overview.md) | **Figure 2 traced box by box**, plus tensor shapes at every stage |
| [11_Contextual-refinement-encoding](11_Contextual-refinement-encoding.md) | Φ — Sec. 3.3, Eq. 6–10. **Figure 3 traced box by box** |
| [12_Transformer-decoder-and-queries](12_Transformer-decoder-and-queries.md) | Eq. 2, object queries, and the missing COCO initialisation |
| [13_Iterative-enhancement](13_Iterative-enhancement.md) | Ψ — Sec. 3.4, Eq. 3 and 11. **Figure 4 traced box by box** |
| [14_Scoring-and-trajectory-heads](14_Scoring-and-trajectory-heads.md) | Eq. 4 and 5 |
| [15_Training-objective](15_Training-objective.md) | Sec. 3.5, Eq. 12–16. The five losses |
| [16_Matching-and-pair-formation](16_Matching-and-pair-formation.md) | The two things the paper needs but never describes |

### 20 — Reference
| | |
|---|---|
| [20_Equation-index](20_Equation-index.md) | Every equation → file and line |
| [21_Repository-layout](21_Repository-layout.md) | What each folder is for, and whether you need it |
| [22_Data-pipeline](22_Data-pipeline.md) | VidVRD formats, the two datasets, the indexing traps |
| [23_Configuration-and-flags](23_Configuration-and-flags.md) | Config layering, every METOR flag, ablation recipes |

### 30 — Provenance: what to trust
| | |
|---|---|
| [30_Paper-gaps](30_Paper-gaps.md) | Every point where the paper is silent, and what was chosen |
| [31_Lineage-evidence](31_Lineage-evidence.md) | What MMP and EOV-MMP state that METOR omits, quoted |
| [32_Not-implemented](32_Not-implemented.md) | What is missing and what it would take |
| [33_Verification-status](33_Verification-status.md) | Verified vs assumed, and the paper's own confound |

### 40 — Practical
| | |
|---|---|
| [40_Hardware](40_Hardware.md) | What runs on what. Read before training |
| [41_Testing](41_Testing.md) | The suite, and which test catches which failure |
| [42_Extending](42_Extending.md) | Where each likely change hooks in |

---

## The one-paragraph version

METOR replaces the *objects → relations* cascade with a single model in which
the two refine each other. This repository implements that method faithfully
enough to read and to test, but **it has never been trained**, and the paper's
COCO decoder initialisation is **not wired up** — so it should not be expected
to reproduce the published numbers as it stands. Every choice the paper left
open is recorded in [30_Paper-gaps](30_Paper-gaps.md); roughly half turned out to
be answerable from the authors' two earlier papers
([31_Lineage-evidence](31_Lineage-evidence.md)), and several more from METOR's
own figures, which are traced element by element in docs 10, 11 and 13.
