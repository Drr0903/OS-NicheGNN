"""optional modules of the OS-NicheGNN encoder.

MoEFusion   the latent as a mixture of four experts (expression, morphology, attention context, neighbourhood skip), each
            mapped to the latent space and weighted per spot by a softmax gate; a load-balancing term keeps all experts in use
CLIPAlign   cross-modal contrastive alignment of the latent with the morphology of the same spot (symmetric InfoNCE over
            the spots of a batch)
GraphInfomax  deep graph infomax: latents of the true graph against latents computed from permuted spot profiles, scored
            against a batch summary by a bilinear discriminator
`attach(model, ac, e_dim)` adds the modules named in the architecture settings; the encoder uses them when present.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MoEFusion(nn.Module):
    def __init__(self, in_dim: int, e_dim: int, d: int, z_dim: int):
        super().__init__()
        self.e_dim = e_dim
        self.ex_e = mlp(e_dim, d, d); self.ex_m = mlp(in_dim - e_dim, d, d); self.ex_s = nn.Linear(3 * in_dim, d)
        self.out = nn.ModuleList([nn.Linear(d, z_dim) for _ in range(4)]); self.gate = nn.Linear(4 * d, 4)

    def forward(self, x_rows, c, skip_in):
        E = [self.ex_e(x_rows[:, :self.e_dim]), self.ex_m(x_rows[:, self.e_dim:]), c, self.ex_s(skip_in)]
        g = torch.softmax(self.gate(torch.cat(E, 1)), dim=1)                       # (n, 4) gate per spot
        z = sum(g[:, k:k + 1] * self.out[k](E[k]) for k in range(4))
        return z, g

    @staticmethod
    def balance(g):
        """K · Σ_k ḡ_k² − 1: zero when the mean gate is uniform over the K experts."""
        m = g.mean(0); return g.shape[1] * (m ** 2).sum() - 1.0


class CLIPAlign(nn.Module):
    """The latent of a spot (u, through a projection head) and the morphology of the same spot (w) are pulled together
    and away from the other spots of the batch, so morphology guides the latent without being copied into it."""

    def __init__(self, in_dim: int, e_dim: int, d: int, z_dim: int, p: int = 128, tau: float = 0.1):
        super().__init__()
        self.e_dim, self.tau = e_dim, tau; self.pz = mlp(z_dim, d, p); self.pm = mlp(in_dim - e_dim, d, p)

    def loss(self, z_rows, x_rows, n_max: int = 1024):
        x = x_rows[:n_max]; u = F.normalize(self.pz(z_rows[:n_max]), dim=1); w = F.normalize(self.pm(x[:, self.e_dim:]), dim=1)
        logits = u @ w.T / self.tau; y = torch.arange(len(x), device=x.device)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


class GraphInfomax(nn.Module):
    def loss_local(self, z_pos, z_neg, summary_z):
        """Local readout in latent space: s_i = sigmoid(mean latent of the ring-1 neighbours of i)."""
        s = torch.sigmoid(summary_z)
        lp = self.W(z_pos, s).squeeze(1); ln = self.W(z_neg, s).squeeze(1)
        return F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))

    def __init__(self, z_dim: int, d: int):
        super().__init__()
        self.W = nn.Bilinear(z_dim, z_dim, 1); self.proj = nn.Linear(d, z_dim)

    def loss(self, z_pos, z_neg, summary=None):
        """summary=None: global batch summary (DGI); otherwise a per-spot local summary (GraphST-style)."""
        s = torch.sigmoid(z_pos.mean(0, keepdim=True)).expand_as(z_pos) if summary is None else torch.sigmoid(self.proj(summary))
        lp = self.W(z_pos, s).squeeze(1); ln = self.W(z_neg, s).squeeze(1)
        return F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))


def spatial_nce(z, z_nb, tau: float = 0.5):
    """Each spot against a random ring-1 neighbour (positive) and the other spots of the batch (negatives)."""
    u = F.normalize(z, dim=1); v = F.normalize(z_nb, dim=1); logits = u @ v.T / tau; y = torch.arange(len(u), device=u.device)
    return F.cross_entropy(logits, y)


def attach(model, ac: dict, in_dim: int, e_dim: int = 50):
    d, z = ac["d"], ac["z_dim"]
    model.moe = MoEFusion(in_dim, e_dim, d, z) if ac.get("moe") else None
    model.clip = CLIPAlign(in_dim, e_dim, d, z) if ac.get("clip", 0) > 0 else None
    model.dgi = GraphInfomax(z, d) if ac.get("dgi", 0) > 0 else None
    if ac.get("img_scale", 1.0) != 1.0:  # input weight of the morphology block; CLIP still sees the complete profile
        sc = torch.ones(1, in_dim); sc[:, e_dim:] = ac["img_scale"]; model.register_buffer("in_scale", sc)
    return model
