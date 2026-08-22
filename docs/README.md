# Knowledge base

Everything known about this project that isn't obvious from the code. The
top-level [README](../README.md) tells you how to *run* it; these documents tell
you how it *works*, what state it's in, and where the research is going.

Files are numbered by kind, not by reading order: **0x** the system, **1x** its
current state, **2x–3x** what is known about the problem, **9x** where to take it.

---

## Start here

| If you want to… | Read |
|---|---|
| Understand the method | [01_Architecture.md](01_Architecture.md) |
| Understand the code | [02_Code-walkthrough.md](02_Code-walkthrough.md) |
| Know what every data file is | [03_Data.md](03_Data.md) |
| Verify a change | [04_Testing.md](04_Testing.md) |
| Know what's broken or blocked | [10_Known-issues.md](10_Known-issues.md) |
| Diff this against upstream EOV | [11_Port-status.md](11_Port-status.md) |
| See how the authors built the data | [12_Upstream-scripts.md](12_Upstream-scripts.md) |
| Know what the benchmark measures | [20_Benchmark-analysis.md](20_Benchmark-analysis.md) |
| Compare the methods, or choose a baseline | [30_Landscape.md](30_Landscape.md) |
| Pick a research direction, and see the case for the current one | [90_Research-ideas.md](90_Research-ideas.md) |
| Find where a change hooks in | [91_Extension-guide.md](91_Extension-guide.md) |

---

## The one-paragraph version

`ov-vidvrd-lab` is a research codebase for open-vocabulary video visual
relationship detection — detecting `<subject, predicate, object>` triplets in
video, including categories never seen during training.

Its baseline is **EOV-MMP** (TPAMI 2025), restructured for extension rather than
reproduction. Data preparation is automated and verified; all four pretrained
components — plus two fully trained end-to-end models — came from the EOV-MMP
author and load with zero key mismatches. The remaining constraint is hardware:
~16 GB of VRAM, so Colab is the supported environment.

---

## The documents

### 0x — the system

**[01_Architecture.md](01_Architecture.md)** — the method as the paper describes
it: four models answering four questions, the four-step training scheme and its
hyperparameters, and where the code and the paper disagree.

**[02_Code-walkthrough.md](02_Code-walkthrough.md)** — the implementation. The
path a video takes, the four input streams and exactly what numbers are in each,
how the predicate classifier decomposes into four text-to-region matches, and how
per-clip scores become a ranked list. **Start here before changing code.**

**[03_Data.md](03_Data.md)** — every file under `data/`: what it is, where it came
from, what reads it. Includes the two things easiest to get wrong — frames are
1-indexed, and the CLIP object bank is the authors', not a reconstruction.

**[04_Testing.md](04_Testing.md)** — the 55 tests, what each protects, and what is
deliberately not covered.

### 1x — current state

**[10_Known-issues.md](10_Known-issues.md)** — the blocker list, ordered by what
stops you first, with measurements rather than guesses.

**[11_Port-status.md](11_Port-status.md)** — provenance: which upstream file
became which file here, every fix applied on the way in, and every way this
repository now diverges from upstream EOV.

**[12_Upstream-scripts.md](12_Upstream-scripts.md)** — the six data-generation
scripts the EOV-MMP author supplied, and what each produces.

### 2x–3x — the problem

**[20_Benchmark-analysis.md](20_Benchmark-analysis.md)** — what VidVRD's test set
actually asks for, measured from the annotations: which predicates need which
kind of evidence, and how much of the "open vocabulary" split is recombination of
seen parts.

**[30_Landscape.md](30_Landscape.md)** — the seven frameworks: what each step
fixed, the scoreboard and the two incompatible protocols hiding inside it, which
methods have usable code and checkpoints, and why the baseline here is EOV rather
than the peer-reviewed state of the art.

### 9x — where to take it

**[90_Research-ideas.md](90_Research-ideas.md)** — the directions, ranked, with
the argument for the current recommendation: that the benchmark measures neither
video understanding nor vocabulary generalisation, why that shape of finding can
carry a paper, the experiment that would kill it, and the risks to plan around.
Keeps the directions considered and set aside, with reasons.

**[91_Extension-guide.md](91_Extension-guide.md)** — for each direction, the file
and function it hooks into, and what running it would establish.

---

## Conventions

- **Nothing here is fabricated.** Where data is missing it is called missing.
  Where something is reconstructed it is labelled a reconstruction.
- **Measurements carry their method.** Numbers like "5.32 GB" or "296,204
  frames" were produced by running something, and the document says what.
- **Dates are absolute.** "Recently" rots; "2026-08-22" doesn't.
