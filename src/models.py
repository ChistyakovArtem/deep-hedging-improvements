"""
models.py — hedge policy network architectures.

Entry point: build_policy(cfg)  where cfg is a ModelConfig dataclass.

Architecture choices (cfg.arch):
    "mlp"       — plain MLP, no embedding
    "paf"       — global Periodic Adaptive Fourier embedding
    "paf_fw"    — feature-wise PAF (one freq set per input dimension)

Skip-connection variants (cfg.skip_conn):
    False       — embedding output = [sin, cos]
    True        — embedding output = [proj, sin, cos]  (preserves linear info)

Initialisation (cfg.zero_last_layer):
    True        — zero-init last linear layer so net starts at δ ≡ 0
                  (useful for deviation / pretrain modes)

All networks:
    forward(x) → (M, 1)
    input_dim  = state_dim + 2          (use_pnl=False)
               = state_dim + 3          (use_pnl=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    arch:             str         = "mlp"       # "mlp" | "paf" | "paf_fw"
    hidden_dims:      Tuple[int]  = (64, 32)
    n_frequencies:    int         = 16
    sigma:            float       = 1.0          # PAF init std
    skip_conn:        bool        = False
    zero_last_layer:  bool        = False        # zero-init last layer
    output_dim:       int         = 1


# ── Embeddings ────────────────────────────────────────────────────────────────

class PAFEmbedding(nn.Module):
    """
    Global Periodic Adaptive Fourier embedding.

        proj = x @ B
        out  = [sin(proj), cos(proj)]          (skip_conn=False)
             = [proj, sin(proj), cos(proj)]    (skip_conn=True)

    B ∈ R^{input_dim × n_freq} is learned.
    """

    def __init__(self, input_dim: int, n_freq: int = 16,
                 sigma: float = 1.0, skip_conn: bool = False):
        super().__init__()
        self.skip_conn = skip_conn
        self.B = nn.Parameter(sigma * torch.randn(input_dim, n_freq))

    @property
    def out_dim(self) -> int:
        n = self.B.shape[1]
        return 3 * n if self.skip_conn else 2 * n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj  = x @ self.B
        parts = [torch.sin(proj), torch.cos(proj)]
        if self.skip_conn:
            parts.insert(0, proj)
        return torch.cat(parts, dim=-1)


class FeatureWisePAFEmbedding(nn.Module):
    """
    Feature-wise PAF: each input dimension j gets its own frequencies C[j, :].

        v_{j,k} = 2π · x_j · C_{j,k}
        emb_j   = [sin(v_j), cos(v_j)]          (skip_conn=False)
                = [v_j, sin(v_j), cos(v_j)]      (skip_conn=True)
        output  = concat_j(emb_j)

    Reference: Gorishniy et al. (2022) "On Embeddings for Numerical Features".
    """

    def __init__(self, input_dim: int, n_freq: int = 16,
                 sigma: float = 1.0, skip_conn: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.n_freq    = n_freq
        self.skip_conn = skip_conn
        self.C = nn.Parameter(sigma * torch.randn(input_dim, n_freq))

    @property
    def out_dim(self) -> int:
        k = 3 if self.skip_conn else 2
        return k * self.input_dim * self.n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, input_dim, 1) * (input_dim, n_freq) → (B, input_dim, n_freq)
        v     = 2 * math.pi * x.unsqueeze(-1) * self.C
        parts = [torch.sin(v), torch.cos(v)]
        if self.skip_conn:
            parts.insert(0, v)
        return torch.cat(parts, dim=-1).view(x.shape[0], -1)


# ── MLP ───────────────────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, hidden_dims: Tuple[int],
              out_dim: int = 1) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LeakyReLU(0.01)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── Policy network ────────────────────────────────────────────────────────────

class HedgePolicy(nn.Module):
    """
    Parametric hedge policy: embedding (optional) → MLP → scalar delta.

    Constructed by build_policy(cfg, input_dim).
    Do not instantiate directly.
    """

    def __init__(self, embedding: nn.Module | None, mlp: nn.Sequential):
        super().__init__()
        self.embedding = embedding
        self.mlp       = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x) if self.embedding is not None else x
        return self.mlp(h)


def build_policy(cfg: ModelConfig, input_dim: int) -> HedgePolicy:
    """
    Build a HedgePolicy from a ModelConfig.

    Parameters
    ----------
    cfg       : ModelConfig
    input_dim : state_dim + 2  (or +3 if use_pnl)
    """
    if cfg.arch == "mlp":
        embedding = None
        mlp_in    = input_dim

    elif cfg.arch == "paf":
        embedding = PAFEmbedding(input_dim, cfg.n_frequencies,
                                 cfg.sigma, cfg.skip_conn)
        mlp_in    = embedding.out_dim

    elif cfg.arch == "paf_fw":
        embedding = FeatureWisePAFEmbedding(input_dim, cfg.n_frequencies,
                                            cfg.sigma, cfg.skip_conn)
        mlp_in    = embedding.out_dim

    else:
        raise ValueError(f"Unknown arch: {cfg.arch!r}. "
                         f"Choose from 'mlp', 'paf', 'paf_fw'.")

    mlp = _make_mlp(mlp_in, cfg.hidden_dims, cfg.output_dim)

    if cfg.zero_last_layer:
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)

    return HedgePolicy(embedding, mlp)
