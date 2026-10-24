"""A miniature CLIP-shaped model, for testing METOR without loading ViT-L.

Why this exists
---------------
`--arch metor` with `--cre_mode inject` runs the frozen CLIP ViT **inside** the
training step (that is the whole point of Eq. 7), and CLIP ViT-L/14@336px is
~400M parameters over 577 tokens per frame. Constructing it and running a
forward-backward costs several GB of RAM. On a machine with 7 GB that is enough
to take the session down, and it tells you nothing a 60k-parameter stand-in
would not.

So the tests build METOR against this instead. It exposes exactly the attributes
`ContextTokenInjector` and `RefinedTextEncoder` reach for, at 1/2000th the size,
which means the shape algebra of Sec. 3.2-3.3 is verifiable on any machine in
under a second.

What it does NOT do is validate anything about accuracy -- the weights are
random. It checks that the wiring is right. See docs/33_Verification-status.md
for what "right" can and cannot mean here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: CLIP's BPE vocabulary size and context length -- these must be real, because
#: the tests tokenise with the genuine CLIP tokenizer.
VOCAB_SIZE = 49408
CONTEXT_LENGTH = 77


class _MiniTransformer(nn.Module):
    """Stands in for CLIP's residual-attention stack. LND in, LND out."""

    def __init__(self, width: int, layers: int = 2, heads: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            dropout=0.0, batch_first=False)
        self.stack = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        return self.stack(x)


class _MiniVisual(nn.Module):
    """CLIP's `VisionTransformer` surface, at toy scale."""

    def __init__(self, width: int, resolution: int, patch_size: int,
                 output_dim: int, layers: int = 2, heads: int = 2):
        super().__init__()
        self.input_resolution = resolution
        self.output_dim = output_dim
        grid = resolution // patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size,
                               stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(grid * grid + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _MiniTransformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


class StubCLIP(nn.Module):
    """Everything METOR touches on a CLIP model, and nothing else.

    Defaults give a 32x32 image in 8x8 patches (16 patch tokens) and width 64 --
    small enough that a full forward-backward is instant, large enough that a
    genuine shape bug still shows up.
    """

    def __init__(self, width: int = 64, text_width: int = 64,
                 output_dim: int = 32, resolution: int = 32,
                 patch_size: int = 8, layers: int = 2, heads: int = 2):
        super().__init__()
        self.visual = _MiniVisual(width, resolution, patch_size, output_dim,
                                  layers, heads)
        self.transformer = _MiniTransformer(text_width, layers, heads)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, text_width)
        self.positional_embedding = nn.Parameter(
            torch.randn(CONTEXT_LENGTH, text_width) * 0.01)
        self.ln_final = nn.LayerNorm(text_width)
        self.text_projection = nn.Parameter(
            torch.randn(text_width, output_dim) * (text_width ** -0.5))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


def stub_clip(**kw) -> StubCLIP:
    """A frozen StubCLIP, matching how the real backbone is used."""
    model = StubCLIP(**kw).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
