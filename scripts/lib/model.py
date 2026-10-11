"""OS-NicheGNN and the matched deep baselines.

OS-NicheGNN (dual-scale, centre / near / outer separated):
    e_i        = Enc(x_i)                                  shared expression encoder (MLP)
    h_i^near   = GAT_near(e, edge_near)_i                  ring-1 message passing
    h_i^outer  = GAT_outer(e, edge_outer)_i                ring-2–3 message passing (ring-1 excluded)
    z_i        = MLP([e_i, h_i^near, h_i^outer, h_i^near − h_i^outer])
    x̂_i        = Dec(z_i)                                  masked-expression reconstruction
    ĉ_i        = AuxDec(z_i)  → target: ring-2–3 mean expression   (λ_ctx-weighted auxiliary head)
Ablations are switches: use_outer=False (near only), use_diff=False (no difference term),
use_aux_context=False (no auxiliary head). Coordinates, patient identity and region labels are never node features; coordinates build the graph and patient identity defines the anchors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


def mlp(i: int, h: int, o: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class Branch(nn.Module):
    def __init__(self, in_dim: int, dim: int, heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.layers.append(GATv2Conv(d, dim // heads, heads=heads, dropout=dropout, add_self_loops=False))
            d = dim
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False):
        h = e
        att = None
        for i, layer in enumerate(self.layers):
            if return_attention and i == len(self.layers) - 1:
                h, att = layer(h, edge_index, return_attention_weights=True)
            else:
                h = layer(h, edge_index)
            h = F.gelu(h)
        # nodes without edges keep a zero context (GAT with no incoming edges yields 0)
        return (self.norm(h), att) if return_attention else self.norm(h)


class OSNicheGNN(nn.Module):
    def __init__(self, n_genes: int, expr_hidden=128, branch_dim=64, fuse_dim=64, n_layers=1, heads=2, dropout=0.15,
                 use_outer=True, use_diff=True, use_aux_context=True, in_dim: int | None = None):
        super().__init__()
        self.use_outer, self.use_diff, self.use_aux = use_outer, use_diff, use_aux_context
        self.enc = nn.Sequential(nn.Linear(in_dim or n_genes, expr_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(expr_hidden, branch_dim), nn.GELU())
        self.near = Branch(branch_dim, branch_dim, heads, n_layers, dropout)
        self.outer = Branch(branch_dim, branch_dim, heads, n_layers, dropout) if use_outer else None
        fuse_in = branch_dim * (2 + (1 if use_outer else 0) + (1 if (use_outer and use_diff) else 0))
        self.fuse = mlp(fuse_in, expr_hidden, fuse_dim, dropout)
        self.dec = mlp(fuse_dim, expr_hidden, n_genes, dropout)
        self.aux = mlp(fuse_dim, expr_hidden, n_genes, dropout) if use_aux_context else None

    def embed(self, x, edge_near, edge_outer, return_attention: bool = False):
        e = self.enc(x)
        parts = [e]
        if return_attention:
            hn, att_n = self.near(e, edge_near, return_attention=True)
        else:
            hn, att_n = self.near(e, edge_near), None
        parts.append(hn)
        att_o = None
        if self.use_outer:
            if return_attention:
                ho, att_o = self.outer(e, edge_outer, return_attention=True)
            else:
                ho = self.outer(e, edge_outer)
            parts.append(ho)
            if self.use_diff:
                parts.append(hn - ho)
        z = self.fuse(torch.cat(parts, dim=1))
        return (z, att_n, att_o) if return_attention else z

    def forward(self, x, edge_near, edge_outer):
        z = self.embed(x, edge_near, edge_outer)
        xhat = self.dec(z)
        chat = self.aux(z) if self.use_aux else None
        return z, xhat, chat


class MLPAE(nn.Module):
    """Non-spatial autoencoder with capacity matched to OS-NicheGNN (same hidden / latent widths)."""

    def __init__(self, n_genes: int, hidden=128, latent=64, dropout=0.15):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.dec = mlp(latent, hidden, n_genes, dropout)

    def embed(self, x, *_):
        return self.enc(x)

    def forward(self, x, *_):
        z = self.enc(x)
        return z, self.dec(z), None


class NeighMeanAE(nn.Module):
    """Self expression + ring-1 mean + ring-2–3 mean concatenated, then the same MLP autoencoder.
    Answers: does a fixed neighbourhood average already explain the gain of learned message passing?"""

    def __init__(self, n_genes: int, hidden=128, latent=64, dropout=0.15):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(3 * n_genes, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.dec = mlp(latent, hidden, n_genes, dropout)

    def embed(self, x, near_mean, outer_mean):
        return self.enc(torch.cat([x, near_mean, outer_mean], 1))

    def forward(self, x, near_mean, outer_mean):
        z = self.embed(x, near_mean, outer_mean)
        return z, self.dec(z), None


def masked_mse(xhat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (xhat - x) ** 2
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
