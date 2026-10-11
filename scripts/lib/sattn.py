"""self-versus-context attention encoder.

    p_i      PCA-50 profile of spot i (fitted on the training patients)
    s_i    = L p_i                                   linear self path, identity-initialised (no information lost)
    e_i    = GELU(W p_i)                             64-d projection fed to the neighbourhood branches
    h_i^n  = GATv2_ring1(e)_i, h_i^o = GATv2_ring2–3(e)_i      neighbour attention, spot itself excluded
    g_i    = sigmoid(MLP([p_i, h_i^n, h_i^o])) ∈ (0,1)^2         per-spot gate on each neighbourhood summary
    z_i    = [s_i, g_i^n h_i^n, g_i^o h_i^o]
Trained with the GraphMAE objective of A17 (objective = "mae": mask token at the input, re-mask, ring-1 attention decoder).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import Branch, mlp
from .remedies import MAEDecoder
from .train import prepare


class SelfContextAttention(nn.Module):
    objective = "mae"

    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        m = cfg["model"]; d = m["branch_dim"]
        self.self_lin = nn.Linear(in_dim, in_dim)
        with torch.no_grad():
            self.self_lin.weight.copy_(torch.eye(in_dim)); self.self_lin.bias.zero_()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.GELU())
        self.near = Branch(d, d, m["heads"], m["n_layers"], m["dropout"])
        self.outer = Branch(d, d, m["heads"], m["n_layers"], m["dropout"])
        self.gate = mlp(in_dim + 2 * d, m["expr_hidden"], 2, m["dropout"])
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.decoder = MAEDecoder(in_dim + 2 * d, in_dim)
        self.z_dim = in_dim + 2 * d

    def parts(self, x, e_near, e_outer):
        e = self.proj(x); hn = self.near(e, e_near); ho = self.outer(e, e_outer)
        g = torch.sigmoid(self.gate(torch.cat([x, hn, ho], 1)))
        return self.self_lin(x), hn, ho, g

    def embed(self, x, e_near, e_outer):
        s, hn, ho, g = self.parts(x, e_near, e_outer)
        return torch.cat([s, g[:, :1] * hn, g[:, 1:] * ho], 1)


@torch.no_grad()
def gate_profile(model: SelfContextAttention, graphs, device: str) -> dict:
    """Mean ring-1 / ring-2–3 gate on the held-out patient's tumour-boundary spots and on all other evaluable spots."""
    model.eval(); gb, gi = [], []
    for g in graphs:
        p = prepare(g, device); gt = model.parts(p.x, p.e_near, p.e_outer)[3].cpu().numpy()
        b = g.boundary if g.boundary is not None else np.zeros(g.n, bool); ev = np.isin(g.labels, ["tumor_cellular", "lung_parenchyma", "paucicellular"])
        gb.append(gt[b & ev]); gi.append(gt[~b & ev])
    gb, gi = np.vstack(gb), np.vstack(gi)
    out = {"gate_near_boundary": float(gb[:, 0].mean()) if len(gb) else np.nan, "gate_near_interior": float(gi[:, 0].mean()),
           "gate_outer_boundary": float(gb[:, 1].mean()) if len(gb) else np.nan, "gate_outer_interior": float(gi[:, 1].mean())}
    return out
