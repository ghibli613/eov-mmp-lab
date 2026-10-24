"""METOR -- Mutual Enhancement of Objects and Relationships.

A reconstruction of

    Wang, Wu and Yang, "METOR: A Unified Framework for Mutual Enhancement of
    Objects and Relationships in Open-vocabulary Video Visual Relationship
    Detection", IJCAI 2025.  arXiv:2505.06663v1

built from the paper text, because the authors' repository
(github.com/wangyongqi558/METOR) has never published code.

Read docs/33_Verification-status.md before trusting any number this produces.
Every place the paper underspecifies something is marked `PAPER-GAP` in the
source and listed in that document.
"""

__version__ = "0.1.0"

from metor.model import METOR, build_metor  # noqa: E402,F401

__all__ = ["METOR", "build_metor", "__version__"]
