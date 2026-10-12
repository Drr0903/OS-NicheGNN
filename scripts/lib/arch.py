"""spot encoders that choose among their neighbours.

Every model maps a spot's PCA profile x_i and its neighbourhood tokens {x_j, pos_ij} (lib.nbhd) to
    h_i = OwnMLP(x_i)                       residual MLP on the spot itself (the MLP component)
    c_i = Context(h_i, tokens_i)            "none" | "xattn" | "mamba" | "xmamba"
    z_i = Fuse([h_i, c_i])                  64-d latent;  x̂_i = Dec(z_i)
The neighbour tokens never contain the spot's own profile except in "mamba", where the spot is the last element of the scan.
The selective scan (Gu & Dao 2023, Mamba-1) is written in plain PyTorch: sequences are ≤ 37 tokens, so a sequential scan is
cheap; padded positions get step size 0 and leave the state unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .nbhd import Nbhd


class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, h):
        return h + self.ff(self.norm(h))


class OwnMLP(nn.Module):
    def __init__(self, in_dim: int, d: int, n_blocks: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(in_dim, d); self.blocks = nn.Sequential(*[ResBlock(d, dropout) for _ in range(n_blocks)]); self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.blocks(self.inp(x)))


class Tokens(nn.Module):
    """Neighbour tokens: shared linear embedding of the neighbour's profile + MLP embedding of its relative position."""

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.val = nn.Linear(in_dim, d); self.pos = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x_full, nb: Nbhd):
        xv = x_full[nb.idx.clamp_min(0)] * nb.valid[..., None]
        return self.val(xv) + self.pos(nb.pos)


class CrossAttn(nn.Module):
    def __init__(self, d: int, heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.att = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(n_layers)])
        self.nq = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)]); self.nk = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ff = nn.ModuleList([ResBlock(d, dropout) for _ in range(n_layers)]); self.out = nn.LayerNorm(d)

    def forward(self, q, kv, valid, return_attn: bool = False):
        pad = ~valid; empty = pad.all(1)
        pad = pad.clone(); pad[empty, 0] = False  # spots without neighbours attend to one zero token
        q = q[:, None]; ws = []
        for att, nq, nk, ff in zip(self.att, self.nq, self.nk, self.ff):
            k = nk(kv); o, w = att(nq(q), k, k, key_padding_mask=pad, need_weights=return_attn); q = q + o; q = ff(q)
            if return_attn:
                ws.append(w[:, 0])  # head-averaged weights over the neighbour tokens
        out = self.out(q[:, 0])
        return (out, torch.stack(ws, 1)) if return_attn else out


class Mamba(nn.Module):
    """One Mamba-1 block (in-projection, causal depthwise conv, selective SSM, gated out-projection) with a pre-LN residual."""

    def __init__(self, d: int, d_state: int = 16, expand: int = 1, conv: int = 4):
        super().__init__()
        e = expand * d; self.e, self.n = e, d_state; self.dt_rank = max(1, d // 16)
        self.norm = nn.LayerNorm(d); self.inp = nn.Linear(d, 2 * e); self.conv = nn.Conv1d(e, e, conv, groups=e, padding=conv - 1)
        self.xproj = nn.Linear(e, self.dt_rank + 2 * d_state, bias=False); self.dt = nn.Linear(self.dt_rank, e)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(e, 1)); self.D = nn.Parameter(torch.ones(e))
        self.out = nn.Linear(e, d)

    def forward(self, u, valid):
        L = u.shape[1]; xz = self.inp(self.norm(u)); x, z = xz.chunk(2, -1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)) * valid[..., None]
        dbc = self.xproj(x); dt, B, C = dbc.split([self.dt_rank, self.n, self.n], -1)
        dt = F.softplus(self.dt(dt)) * valid[..., None]; A = -torch.exp(self.A_log)
        h = u.new_zeros(u.shape[0], self.e, self.n); ys = []
        for t in range(L):
            h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t, :, None] * B[:, t, None, :]) * x[:, t, :, None]
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = (torch.stack(ys, 1) + self.D * x) * F.silu(z)
        return u + self.out(y)


def niche_target_values(x: torch.Tensor, nb: Nbhd) -> dict[str, torch.Tensor]:
    """Per spot, from the complete (unmasked) profiles: mean of the ring-1 neighbours, mean of the ring-2–3 neighbours and
    the per-component variance over all ring-1–3 neighbours (a spot without such neighbours gets its own profile / zero)."""
    xv = x[nb.idx.clamp_min(0)]; r1 = nb.pos[..., 2] * nb.valid; ro = (nb.pos[..., 3] + nb.pos[..., 4]) * nb.valid; al = nb.valid.float()

    def wmean(w, v):
        n = w.sum(1, keepdim=True); return torch.where(n > 0, (w[..., None] * v).sum(1) / n.clamp_min(1), x)
    m_all = wmean(al, xv)
    disp = (al[..., None] * (xv - m_all[:, None]) ** 2).sum(1) / al.sum(1, keepdim=True).clamp_min(1)
    return {"ring1": wmean(r1, xv), "outer": wmean(ro, xv), "disp": disp}


class NicheEncoder(nn.Module):
    def __init__(self, in_dim: int, context: str, d: int = 128, z_dim: int = 64, heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 niche_targets: tuple[str, ...] = (), linear_heads: bool = False, prop_skip: bool = False, grad_skip: bool = False):
        super().__init__()
        self.context = context; self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # ringmlp: no neighbour tokens; the residual MLP reads [own profile, ring-1 mean, ring-2–3 mean] (a learned, non-linear
        # function of the same neighbourhood averages that the linear skip receives)
        self.own = OwnMLP(in_dim * (3 if context == "ringmlp" else 1), d, 2, dropout)
        if context not in ("none", "ringmlp"):
            self.tok = Tokens(in_dim, d)
        if context in ("xattn", "xmamba"):
            self.xattn = CrossAttn(d, heads, n_layers, dropout)
        if context in ("mamba", "xmamba"):
            self.mamba = nn.ModuleList([Mamba(d) for _ in range(n_layers)]); self.mnorm = nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(d * (1 if context in ("none", "ringmlp") else 2), d), nn.GELU(), nn.Linear(d, z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, d), nn.GELU(), nn.Linear(d, in_dim))
        # niche targets (A20): z also decodes the spot's surroundings — ring-1 mean, ring-2–3 mean, dispersion of rings 1–3
        # linear_heads: the surroundings must be linearly readable from z, not only through a non-linear decoder
        self.ndec = nn.ModuleDict({t: (nn.Linear(z_dim, in_dim) if linear_heads else nn.Sequential(nn.Linear(z_dim, d), nn.GELU(), nn.Linear(d, in_dim)))
                                   for t in niche_targets})
        # prop_skip: z also receives a linear map of [own profile, ring-1 mean, ring-2–3 mean], so the explicit neighbourhood
        # average is never lost in the bottleneck; attention and the anchor term then reshape what is added on top of it
        # grad_skip: also the azimuthal gradient of the ring-1–2 neighbours (|Σ_j x_j e^{iθ_j}| per feature, as in BANKSY's AGF),
        # which separates a spot on a tissue edge from one inside a mixed region with the same neighbourhood mean
        self.grad = grad_skip
        self.pskip = nn.Linear((4 if grad_skip else 3) * in_dim, z_dim) if prop_skip else None

    def _mamba(self, h, tok, valid):
        seq = torch.cat([tok, h[:, None]], 1); v = torch.cat([valid, valid.new_ones(valid.shape[0], 1)], 1)
        for blk in self.mamba:  # activation checkpointing: the sequential scan is recomputed in backward (memory, not maths)
            seq = torch.utils.checkpoint.checkpoint(blk, seq, v, use_reentrant=False) if self.training else blk(seq, v)
        return self.mnorm(seq[:, -1])

    def scale_in(self, x_full):
        """A23: the morphology block of the input can be down-weighted (img_scale); 1 keeps the profile as it is.
        With a morphology gate the block is multiplied by a per-spot weight in [0, 1] read from the spot's own expression and
        morphology, before the spot is used as centre, as neighbour token or in the ring means, so every path sees the same."""
        sc = getattr(self, "in_scale", None); x_full = x_full if sc is None else x_full * sc
        mg = getattr(self, "mgate", None)
        return x_full if mg is None else mg(x_full)

    @staticmethod
    def ring_means(x_full, nb: Nbhd, rows: torch.Tensor):
        xv = x_full[nb.idx[rows].clamp_min(0)]; v = nb.valid[rows].float(); p = nb.pos[rows]
        def wm(w):
            return (w[..., None] * xv).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
        return wm(p[..., 2] * v), wm((p[..., 3] + p[..., 4]) * v)

    def parts_rows(self, x_full, nb: Nbhd, rows: torch.Tensor):
        """Own and context encodings of the centre spots `rows`; neighbour tokens are looked up in the whole section x_full."""
        x_full = self.scale_in(x_full)
        if self.context == "ringmlp":
            return self.own(torch.cat([x_full[rows], *self.ring_means(x_full, nb, rows)], 1)), None
        h = self.own(x_full[rows])
        if self.context == "none":
            return h, None
        nb = Nbhd(nb.idx[rows], nb.valid[rows], nb.pos[rows]); tok = self.tok(x_full, nb)
        if self.context == "xattn":
            c = self.xattn(h, tok, nb.valid)
        elif self.context == "mamba":
            c = self._mamba(h, tok, nb.valid)
        else:  # xmamba: the scanned neighbourhood state joins the neighbour tokens as one more key/value
            m = self._mamba(h, tok, nb.valid)
            c = self.xattn(h, torch.cat([tok, m[:, None]], 1), torch.cat([nb.valid, nb.valid.new_ones(nb.valid.shape[0], 1)], 1))
        return h, c

    def attention_rows(self, x_full, nb: Nbhd, rows: torch.Tensor):
        """Attention weights (layers × neighbour tokens, heads averaged) of the centre spots `rows` (xattn context only)."""
        h = self.own(x_full[rows]); nbr = Nbhd(nb.idx[rows], nb.valid[rows], nb.pos[rows])
        return self.xattn(h, self.tok(x_full, nbr), nbr.valid, return_attn=True)[1]

    def z_parts(self, x_full, nb: Nbhd, rows: torch.Tensor):
        """(learned part fuse([h, c]), linear-skip part) of z for the centre spots `rows`."""
        h, c = self.parts_rows(x_full, nb, rows); learned = self.fuse(h if c is None else torch.cat([h, c], 1))
        return learned, self.z_from(x_full, nb, rows, h, c) - learned

    def z_from(self, x_full, nb: Nbhd, rows: torch.Tensor, h, c):
        x_full = self.scale_in(x_full); moe = getattr(self, "moe", None)
        if moe is None:
            z = self.fuse(h if c is None else torch.cat([h, c], 1))
        if self.pskip is not None or moe is not None:
            xv = x_full[nb.idx[rows].clamp_min(0)]; v = nb.valid[rows].float(); p = nb.pos[rows]
            def wm(w):
                return (w[..., None] * xv).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
            parts = [x_full[rows], wm(p[..., 2] * v), wm((p[..., 3] + p[..., 4]) * v)]
            if moe is not None:  # the latent as a gated mixture of expression, morphology, context and skip experts
                z, self.last_gates = moe(x_full[rows], c, torch.cat(parts, 1)); return z
            if self.grad:
                w = (p[..., 2] + p[..., 3]) * v; ang = torch.atan2(p[..., 1], p[..., 0]); n = w.sum(1, keepdim=True).clamp_min(1)
                cx = ((w * torch.cos(ang))[..., None] * xv).sum(1) / n; sy = ((w * torch.sin(ang))[..., None] * xv).sum(1) / n
                parts.append(torch.sqrt(cx ** 2 + sy ** 2 + 1e-8))
            z = z + self.pskip(torch.cat(parts, 1))
        return z

    def embed_rows(self, x_full, nb: Nbhd, rows: torch.Tensor):
        h, c = self.parts_rows(x_full, nb, rows)
        return self.z_from(x_full, nb, rows, h, c)
