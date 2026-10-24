"""Detector factory.

Only one detector is implemented: `ov_prompt`, the CLIP-distilled Deformable
DETR described in docs/01_Architecture.md §2.1. Upstream EOV also shipped a `solq`
variant, but its import was already commented out there and it was never
reachable; it has been removed.
"""
from models.methods.detectors.ov_prompt.model import build as ov_prompt_build

AVAILABLE_METHODS = ("ov_prompt",)


def build_model(args):
    if args.method not in AVAILABLE_METHODS:
        raise ValueError(
            f"method [{args.method}] is not supported; "
            f"available: {', '.join(AVAILABLE_METHODS)}")
    return ov_prompt_build(args)
