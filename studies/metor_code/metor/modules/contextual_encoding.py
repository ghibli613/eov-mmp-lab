"""Contextual refinement encoding -- METOR Sec. 3.3, Eq. 6-10.

The idea in one paragraph
-------------------------
CLIP is frozen, so its text embeddings for "zebra" and for "walk past" are the
same no matter what video you show it. METOR makes them video-specific. It
appends two small sets of *learnable* tokens to the CLIP ViT's input -- object
context tokens `c_o` and relationship context tokens `c_r` -- and lets them
attend over the frame alongside the patches. What comes out (`C_o`, `C_r`) is a
summary of "what kind of objects / what kind of interactions are in this frame",
and it is used twice:

    Eq. 8   Q_hat = MHA(Q, C_o, C_o) + Q         refine the object queries
    Eq. 9   T_o   = G([v_o; M_o(C_o); OBJ])      refine the object text features
    Eq. 10  T_r   = G([v_r; M_r(C_r); REL])      refine the predicate text features

So both ends of the eventual `cos(visual, text)` comparison are conditioned on
this video rather than fixed. That is the paper's stated mechanism for
generalising to novel categories, and the ablation that removes it (`w/o CRE`)
is the single largest drop in their Table 3: 16.43 -> 13.49 novel mAP.

Two ways to get the context embeddings
--------------------------------------
`--cre_mode inject` (faithful, default for `--arch metor`)
    Really appends the tokens to the ViT input and runs the frozen ViT, in the
    style of VPT. Gradients reach `c_o` / `c_r` through the frozen weights. This
    is what Eq. 7 describes. It costs a ViT-L forward per frame per step.

`--cre_mode pool` (cheap, default for `--arch metor-compat`)
    Attention-pools the *already computed* patch embeddings with learnable
    queries. Mathematically this is a one-layer approximation of the above: the
    context tokens see the patches, but the patches never see the context
    tokens, and the interaction is not iterated through the ViT's 24 layers.
    It exists so the module can be trained and ablated on top of cached
    features, which is the only regime this workspace can afford.

    It is *not* equivalent, and any number produced with it must say so.

PAPER-GAP notes are inline. Summary in docs/30_Paper-gaps.md;
narrative walkthrough in docs/11_Contextual-refinement-encoding.md.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clip_token_ids(text: str):
    """BPE token ids for `text`, without SOS/EOS -- i.e. its length in tokens.

    Needed only for `token_position='three_quarter'`, where the trailing
    learnable tokens are spliced in after the class name and the class name's
    token length therefore has to be known per class.
    """
    from vlm.backbones.clip.simple_tokenizer import SimpleTokenizer
    global _TOKENIZER
    try:
        _TOKENIZER
    except NameError:
        _TOKENIZER = SimpleTokenizer()
    return _TOKENIZER.encode(text)


# --------------------------------------------------------------------------
# Eq. 7 -- context tokens through the frozen CLIP ViT
# --------------------------------------------------------------------------
class ContextTokenInjector(nn.Module):
    """Run a frozen CLIP `VisionTransformer` with extra learnable tokens.

    Reproduces Eq. 7:

        (H_CLS, H_patch, C_o, C_r) = V([h_CLS; h_patch; c_o; c_r])

    The wrapped ViT is used but never trained -- its parameters are left
    exactly as `requires_grad_(False)` set them. Only `c_o` and `c_r` learn.

    The appended tokens get **no positional embedding**, which is the usual VPT
    choice: they are not at a place in the image, so a position for them would
    be meaningless. PAPER-GAP: the paper does not say either way.
    """

    def __init__(self, visual: nn.Module, n_ctx_obj: int = 8, n_ctx_rel: int = 8):
        super().__init__()
        self.visual = visual
        for p in self.visual.parameters():
            p.requires_grad_(False)

        width = visual.class_embedding.shape[0]
        self.width = width
        self.n_ctx_obj = n_ctx_obj
        self.n_ctx_rel = n_ctx_rel

        # Same init scale CLIP uses for its own class embedding, so the added
        # tokens start at the magnitude the frozen layers expect to see.
        scale = width ** -0.5
        self.ctx_obj = nn.Parameter(scale * torch.randn(n_ctx_obj, width))
        self.ctx_rel = nn.Parameter(scale * torch.randn(n_ctx_rel, width))

    def forward(self, images: torch.Tensor):
        """images: (B, 3, H, W) -> H_CLS (B,d), H_patch (B,N,d), C_o, C_r.

        `H_CLS` and `H_patch` are returned in the ViT's *width* space (1024 for
        ViT-L/14), pre-projection -- that is what the detector consumes, exactly
        as `patch_` does in the EOV path.
        """
        v = self.visual
        dtype = v.conv1.weight.dtype

        x = v.conv1(images.type(dtype))                    # (B, width, g, g)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)   # (B, N, width)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                     # (B, 1+N, width)
        x = x + v.positional_embedding.to(x.dtype)

        # Context tokens are appended AFTER the positional embedding is added,
        # so they carry none -- see the class docstring.
        b = x.shape[0]
        ctx = torch.cat([self.ctx_obj, self.ctx_rel], dim=0).to(x.dtype)
        x = torch.cat([x, ctx.unsqueeze(0).expand(b, -1, -1)], dim=1)

        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND
        x = v.transformer(x)
        # TagCLIP's Transformer returns a tuple; CLIP's returns a tensor.
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.permute(1, 0, 2)          # LND -> NLD
        x = v.ln_post(x)

        n_patch = x.shape[1] - 1 - self.n_ctx_obj - self.n_ctx_rel
        h_cls = x[:, 0]                                     # (B, width)
        h_patch = x[:, 1:1 + n_patch]                       # (B, N, width)
        c_o = x[:, 1 + n_patch:1 + n_patch + self.n_ctx_obj]
        c_r = x[:, 1 + n_patch + self.n_ctx_obj:]
        return h_cls, h_patch, c_o, c_r


class PooledContextEncoder(nn.Module):
    """The `--cre_mode pool` approximation: learnable queries over cached patches.

    Takes patch embeddings that a frozen encoder already produced and pools them
    into the same `(B, n_ctx, width)` shape `ContextTokenInjector` returns, so
    every downstream module is identical. See the module docstring for why this
    is an approximation and not a shortcut to the same result.
    """

    def __init__(self, width: int, n_ctx_obj: int = 8, n_ctx_rel: int = 8,
                 n_heads: int = 8):
        super().__init__()
        self.n_ctx_obj = n_ctx_obj
        self.n_ctx_rel = n_ctx_rel
        scale = width ** -0.5
        self.ctx_obj = nn.Parameter(scale * torch.randn(n_ctx_obj, width))
        self.ctx_rel = nn.Parameter(scale * torch.randn(n_ctx_rel, width))
        self.attn = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(width)

    def forward(self, h_patch: torch.Tensor):
        """h_patch: (B, N, width) -> C_o (B, n_ctx_obj, width), C_r (...)."""
        b = h_patch.shape[0]
        q = torch.cat([self.ctx_obj, self.ctx_rel], dim=0)
        q = q.unsqueeze(0).expand(b, -1, -1).to(h_patch.dtype)
        kv = self.ln(h_patch)
        ctx, _ = self.attn(q, kv, kv, need_weights=False)
        ctx = ctx + q                       # residual, as in Eq. 8's spirit
        return ctx[:, :self.n_ctx_obj], ctx[:, self.n_ctx_obj:]


# --------------------------------------------------------------------------
# Eq. 8 -- refining the object queries
# --------------------------------------------------------------------------
class ObjectQueryRefiner(nn.Module):
    """Eq. 8:  `Q_hat = MHA(Q, C_o, C_o) + Q`.

    The residual is the whole point and the paper says so explicitly: it
    "preserves information in the original object queries while incorporating
    contextual knowledge". Drop the residual and the detector's learned query
    initialisation is destroyed, which is what `--no_crq` measures (their
    Table 3: 16.43 -> 15.32 novel mAP when this refinement is removed).

    Queries live in the detector's hidden dim (256 for Deformable-DETR, and the
    query embedding is `hidden_dim * 2` because DETR splits it into a query and
    a reference point). Context lives in the ViT width (1024). The projections
    bridge the two.
    """

    def __init__(self, query_dim: int, ctx_dim: int, n_heads: int = 8):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, query_dim)
        self.attn = nn.MultiheadAttention(query_dim, n_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(query_dim)
        self.ln_kv = nn.LayerNorm(query_dim)
        # Zero-init the output projection so the module starts as exactly the
        # identity Q_hat = Q. A randomly-initialised refiner would corrupt the
        # COCO-pretrained queries on step 0 and the detector would have to
        # relearn them. Standard adapter practice; PAPER-GAP: unspecified.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, queries: torch.Tensor, ctx_obj: torch.Tensor):
        """queries: (B, Nq, query_dim) or (Nq, query_dim); ctx: (B, n_ctx, ctx_dim)."""
        squeeze = queries.dim() == 2
        if squeeze:
            queries = queries.unsqueeze(0).expand(ctx_obj.shape[0], -1, -1)
        kv = self.ln_kv(self.ctx_proj(ctx_obj.to(queries.dtype)))
        out, _ = self.attn(self.ln_q(queries), kv, kv, need_weights=False)
        refined = out + queries                          # the residual of Eq. 8
        return refined


# --------------------------------------------------------------------------
# Eq. 9 / 10 -- refining the text features
# --------------------------------------------------------------------------
class ContextMappingLayer(nn.Module):
    """`M_o` / `M_r` -- "implemented as a multi-layer perceptron" (Sec. 3.3).

    Maps context embeddings from the ViT width into the text transformer's
    token space so they can sit inside a prompt.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RefinedTextEncoder(nn.Module):
    """Eq. 9 and Eq. 10 -- context-conditioned class prompts through frozen CLIP.

        T_o = G([v_o; M_o(C_o); OBJ])
        T_r = G([v_r; M_r(C_r); REL])

    Read the bracket as a token sequence assembled in CLIP's *embedding* space:

        [SOS] v_1..v_n  M(C)_1..M(C)_k  <class name tokens> [EOS] <pad...>

    `v` is a set of free learnable vectors (CoOp's continuous context), `M(C)`
    are the mapped context tokens, and the class name is the "object category
    embedding OBJ" / "relationship category embedding REL" of the paper -- one
    prompt per category, so the output is (n_classes, d), a classifier whose
    rows are computed from words and from this video.

    Sizes. METOR states neither |v| nor the number of context tokens, but both
    predecessors state both, identically:

        MMP     "we set the number of tokens for both learnable continuous
                 prompts and conditional prompts to 8"
        EOV-MMP "we use eight tokens each for learnable continuous prompts and
                 learnable conditional prompts"

    `v` is the continuous half and `M(C)` is the conditional half, so both
    default to **8** here on that evidence rather than on a guess.

    Category token position. `token_position` selects where the class name sits:

        "end"            what Eq. 10 literally prints, `[...; REL]`
        "three_quarter"  what BOTH predecessors do -- MMP: "the [CLS] token is
                         positioned at 75% of the token length"; EOV-MMP:
                         "positioning the relationship token [REL] at 75% of
                         the token length" (while [OBJ] stays at the end)

    Genuinely ambiguous: METOR's equation says one thing and the authors' own
    two prior systems say another. Default is the paper-literal "end".
    """

    def __init__(self, clip_model, classnames: Sequence[str], ctx_dim: int,
                 n_prompt_vec: int = 8, n_ctx: int = 8,
                 tokenizer=None, prompt_tail: str = ".",
                 token_position: str = "end"):
        super().__init__()
        from vlm.backbones.clip import clip as clip_pkg

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        for p in self.parameters():
            p.requires_grad_(False)

        embed_dim = clip_model.ln_final.weight.shape[0]
        self.embed_dim = embed_dim
        self.n_prompt_vec = n_prompt_vec
        self.n_ctx = n_ctx
        self.n_cls = len(classnames)

        # v -- the free learnable vectors of Eq. 9/10.
        self.prompt_vectors = nn.Parameter(torch.empty(n_prompt_vec, embed_dim))
        nn.init.normal_(self.prompt_vectors, std=0.02)

        # M -- context mapping layer, ViT width -> text embedding space.
        self.mapping = ContextMappingLayer(ctx_dim, embed_dim)

        if token_position not in ("end", "three_quarter"):
            raise ValueError(f"token_position must be 'end' or 'three_quarter', "
                             f"got {token_position!r}")
        self.token_position = token_position

        # How many of the learnable slots sit BEFORE the class name.
        n_placeholder = n_prompt_vec + n_ctx
        if token_position == "end":
            n_before = n_placeholder
        else:
            # 75% of the learnable prompt before the category, the rest after.
            n_before = int(round(0.75 * n_placeholder))
        self.n_before = n_before
        self.n_after = n_placeholder - n_before

        # The class-name half of the prompt, tokenised once and frozen. The
        # placeholder "X"s reserve exactly the slots that v and M(C) will fill,
        # so the EOS position -- which is where CLIP reads the sentence
        # embedding from -- is correct for every class.
        names = [n.replace("_", " ") for n in classnames]
        before = " ".join(["X"] * n_before)
        after = (" " + " ".join(["X"] * self.n_after)) if self.n_after else ""
        prompts = [f"{before} {n}{after}{prompt_tail}" for n in names]
        tokenized = torch.cat([clip_pkg.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)
        self.register_buffer("tokenized_prompts", tokenized, persistent=False)
        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)   # SOS

        if token_position == "end":
            # [SOS] <learnable...> <name> [EOS] <pad>
            self.register_buffer("token_suffix",
                                 embedding[:, 1 + n_placeholder:, :], persistent=False)
        else:
            # [SOS] <learnable x n_before> <name> <learnable x n_after> [EOS] <pad>
            # The class name has a different token length per class, so the
            # split point moves; store the whole tail and slice per class.
            self.register_buffer("token_tail", embedding[:, 1 + n_before:, :],
                                 persistent=False)
            name_lens = [len(_clip_token_ids(n + "")) for n in names]
            self.register_buffer("name_lens", torch.tensor(name_lens),
                                 persistent=False)

    def _encode(self, prompts, tokenized):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, ctx: torch.Tensor):
        """ctx: (n_ctx, ctx_dim) -- one video's context -> (n_cls, d), L2-normalised.

        The context is video-level here. PAPER-GAP: Eq. 9/10 do not say whether
        `C_o` is per-frame or pooled, while Eq. 8 is explicitly per-frame ("to
        refine the object queries in each frame"). Pooling for text and keeping
        per-frame for queries is the reading that makes both equations
        type-check: a text classifier that changed every frame could not be
        compared against a video-level trajectory score in Eq. 4.
        """
        if ctx.dim() == 3:                      # (T, n_ctx, d) -> pool over time
            ctx = ctx.mean(dim=0)
        mapped = self.mapping(ctx.to(self.mapping.net[0].weight.dtype)).type(self.dtype)

        n = self.n_cls
        # [v_o ; M_o(C_o)] -- the continuous half then the conditional half,
        # which is the order Eq. 9/10 print them in.
        learnable = torch.cat([
            self.prompt_vectors.type(self.dtype),
            mapped,
        ], dim=0).unsqueeze(0).expand(n, -1, -1)                    # (n, n_ph, d)

        if self.token_position == "end":
            prompts = torch.cat([
                self.token_prefix,                                  # (n, 1, d)
                learnable,
                self.token_suffix,                                  # name + EOS + pad
            ], dim=1)
        else:
            # Splice the trailing learnable tokens in AFTER each class name.
            # Name length varies per class, so the insertion point does too.
            head = torch.cat([self.token_prefix, learnable[:, :self.n_before]], dim=1)
            rows = []
            for i in range(n):
                ln = int(self.name_lens[i])
                tail = self.token_tail[i]                           # (L, d)
                rows.append(torch.cat([
                    tail[:ln],                                      # the class name
                    learnable[i, self.n_before:],                   # trailing tokens
                    tail[ln + self.n_after:],                       # EOS + padding
                ], dim=0))
            prompts = torch.cat([head, torch.stack(rows)], dim=1)

        feats = self._encode(prompts, self.tokenized_prompts)
        return feats / feats.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# Phi(.) of Eq. 1 -- the whole module
# --------------------------------------------------------------------------
class ContextualRefinementEncoder(nn.Module):
    """Phi(V, Q) of Eq. 1, returning all seven of its outputs.

        (H_CLS, H_patch, T_o, T_r, Q_hat, C_o, C_r) = Phi(V, Q)

    Ablation switches map to the paper's Table 3 exactly:

        no_crq -> `Q_hat = Q`, the object-query refinement is skipped
        no_crt -> `T_o`/`T_r` fall back to fixed CLIP prompts
        both   -> the `w/o CRE` row
    """

    def __init__(self, clip_model, obj_classnames, rel_classnames, *,
                 query_dim: int, mode: str = "inject",
                 n_ctx_obj: int = 8, n_ctx_rel: int = 8,
                 n_prompt_vec: int = 8, n_heads: int = 8,
                 rel_token_position: str = "end",
                 refine_queries: bool = True, refine_text: bool = True):
        super().__init__()
        if mode not in ("inject", "pool"):
            raise ValueError(f"cre_mode must be 'inject' or 'pool', got {mode!r}")
        self.mode = mode
        self.refine_queries = refine_queries
        self.refine_text = refine_text

        width = clip_model.visual.class_embedding.shape[0]
        self.ctx_dim = width

        if mode == "inject":
            self.context = ContextTokenInjector(clip_model.visual, n_ctx_obj, n_ctx_rel)
        else:
            self.context = PooledContextEncoder(width, n_ctx_obj, n_ctx_rel)

        if refine_queries:
            self.query_refiner = ObjectQueryRefiner(query_dim, width, n_heads)
        if refine_text:
            # [OBJ] stays at the end in both predecessors; only [REL] moves.
            self.obj_text = RefinedTextEncoder(
                clip_model, obj_classnames, width, n_prompt_vec, n_ctx_obj,
                token_position="end")
            self.rel_text = RefinedTextEncoder(
                clip_model, rel_classnames, width, n_prompt_vec, n_ctx_rel,
                token_position=rel_token_position)

    def encode(self, images=None, h_patch=None):
        """Whichever half of Eq. 7 this mode needs. Returns H_CLS, H_patch, C_o, C_r."""
        if self.mode == "inject":
            if images is None:
                raise ValueError("cre_mode='inject' needs pixel input `images`")
            return self.context(images)
        if h_patch is None:
            raise ValueError("cre_mode='pool' needs cached `h_patch`")
        c_o, c_r = self.context(h_patch)
        return None, h_patch, c_o, c_r

    def forward(self, queries, images=None, h_patch=None,
                fixed_obj_text=None, fixed_rel_text=None):
        h_cls, h_patch, c_o, c_r = self.encode(images=images, h_patch=h_patch)

        q_hat = self.query_refiner(queries, c_o) if self.refine_queries else queries

        if self.refine_text:
            t_o = self.obj_text(c_o)
            t_r = self.rel_text(c_r)
        else:
            # `--no_crt`: the fixed CLIP prompts EOV-MMP uses, unconditioned.
            if fixed_obj_text is None or fixed_rel_text is None:
                raise ValueError("refine_text=False requires the fixed text banks")
            t_o, t_r = fixed_obj_text, fixed_rel_text

        return h_cls, h_patch, t_o, t_r, q_hat, c_o, c_r
