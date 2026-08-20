# Knowledge base

Everything known about this project that isn't obvious from the code. The
top-level [README](../README.md) tells you how to *run* it; these documents tell
you how it *works*, what state it's in, and where the research is going.

---

## Start here

| If you want to… | Read |
|---|---|
| Set up and run the thing | [`../README.md`](../README.md) |
| Understand the model and its four training steps | [architecture.md](architecture.md) |
| Know what every data file is and where it came from | [data.md](data.md) |
| Know what's broken or blocked, and what it costs | [known-issues.md](known-issues.md) |
| See how this code differs from upstream EOV-MMP | [port-status.md](port-status.md) |
| Understand the research landscape | [landscape.md](landscape.md) |
| See candidate directions for a paper | [research-ideas.md](research-ideas.md) |

---

## The one-paragraph version

This is **EOV-MMP** (End-to-end Open-vocabulary Video Visual Relationship
Detection using Multi-modal Prompting, TPAMI 2025) re-housed in a structured,
machine-independent layout. The method detects `<subject, predicate, object>`
triplets in video, including object and predicate categories never seen in
training. Data preparation is fully automated and verified. **Training is
blocked** on four pretrained checkpoints that the authors have not made
downloadable and whose training code was never released; evaluation of an
already-trained checkpoint is not.

---

## Document map

### [architecture.md](architecture.md)
What each of the four components does, how the open-vocabulary trick works
(CLIP text embeddings as classifier weights), and the paper's four-step training
scheme with its hyperparameters. Also records where the code and the paper
disagree — worth reading before you report any numbers.

### [data.md](data.md)
Every file under `data/`: what it is, where it came from, and what reads it.
Includes the two things that are easy to get wrong — frames are **1-indexed**,
and the CLIP object bank is a **reconstruction** whose results are not directly
comparable to the paper.

### [known-issues.md](known-issues.md)
The blocker list, ordered by what stops you first, with measurements rather than
guesses: 5.32 GB of dataloader VRAM, 11.8 M redundant CLIP forwards per run, and
the checkpoint situation. Ends with what has already been resolved.

### [port-status.md](port-status.md)
The provenance record: which upstream file became which file here, every fix
applied on the way in, and — importantly — **every place this repository
diverges from upstream EOV**, including an edit to vendored CUDA source. Read
this before diffing against upstream.

### [landscape.md](landscape.md)
RePro → UASAN → MMP → EOV-MMP → METOR: what each step actually fixed, the
scoreboard, and where the remaining headroom is.

### [research-ideas.md](research-ideas.md)
Candidate directions, including which ones were investigated and **killed**, and
why. The compositional-gap measurement is the one currently recommended.

---

## Conventions

- **Nothing here is fabricated.** Where data is missing it is called missing.
  Where something is reconstructed it is labelled a reconstruction.
- **Measurements carry their method.** Numbers like "5.32 GB" or "296,204
  frames" were produced by running something, and the document says what.
- **Dates are absolute.** "Recently" rots; "2026-08-20" doesn't.
