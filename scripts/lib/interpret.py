""" (stage 28) — what the atlas encoder uses: attention, the two parts of the latent, gene attribution,
and the contribution of morphology. Everything is computed with the delivered encoder (seed 1) on the delivered atlas.

Niche reference for perturbations: centroids of the delivered niches in the raw latent; a perturbed latent is re-assigned
to the nearest centroid and compared with the same assignment of the unperturbed latent (so only the perturbation counts).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch

from .arch_train import prep


@torch.no_grad()
def latent(model, g, device: str, x: np.ndarray | None = None, chunk: int = 4096, parts: bool = False):
    s = prep(g if x is None else dataclasses.replace(g, x=x.astype(np.float32)), device); out = []
    for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
        if parts:
            a, b = model.z_parts(s.x, s.nb, ch); out.append(torch.stack([a, b], 0).cpu())
        else:
            h, c = model.parts_rows(s.x, s.nb, ch); out.append(model.z_from(s.x, s.nb, ch, h, c).cpu())
    return torch.cat(out, 1 if parts else 0).numpy()


@torch.no_grad()
def latent_without_context(model, g, device: str, chunk: int = 4096) -> np.ndarray:
    """z with the attention context replaced by zeros (the linear skip and the spot's own branch kept)."""
    s = prep(g, device); out = []
    for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
        h, c = model.parts_rows(s.x, s.nb, ch); out.append(model.z_from(s.x, s.nb, ch, h, torch.zeros_like(c)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def latent_without_skip(model, g, device: str, chunk: int = 4096) -> np.ndarray:
    """z with the linear neighbourhood skip removed (the learned part only)."""
    s = prep(g, device); out = []
    for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
        out.append(model.z_parts(s.x, s.nb, ch)[0].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def latent_rewired(model, g, device: str, seed: int, chunk: int = 4096) -> np.ndarray:
    """z on a graph whose neighbours are random spots of the same section (positions kept)."""
    s = prep(g, device, rewire_seed=seed); out = []
    for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
        h, c = model.parts_rows(s.x, s.nb, ch); out.append(model.z_from(s.x, s.nb, ch, h, c).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def attention(model, g, device: str, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last-layer attention (n × 36), neighbour indices (n × 36, −1 = none) and rings (n × 36)."""
    s = prep(g, device); W = []
    for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
        W.append(model.attention_rows(s.x, s.nb, ch)[:, -1].cpu())
    ring = (s.nb.pos[..., 2:5].argmax(-1) + 1) * s.nb.valid
    return torch.cat(W).numpy(), s.nb.idx.cpu().numpy(), ring.cpu().numpy()


def attention_summary(W: np.ndarray, idx: np.ndarray, ring: np.ndarray, label: np.ndarray, boundary: np.ndarray, niche: np.ndarray) -> list[dict]:
    """Per spot: attention on ring 1/2/3 and on same-label neighbours, each against its share of the valid neighbours."""
    valid = idx >= 0; n = valid.sum(1); rows = []
    same = np.zeros_like(valid); same[valid] = label[idx[valid]] == np.repeat(label, valid.sum(1))
    for r in (1, 2, 3):
        m = ring == r
        rows.append(("ring", r, (W * m).sum(1), m.sum(1) / np.maximum(n, 1)))
    rows.append(("same_label", 0, (W * same).sum(1), same.sum(1) / np.maximum(n, 1)))
    out = []
    for kind, r, att, share in rows:
        for scope, sel in [("all", np.ones(len(W), bool)), ("tumour boundary", boundary)] + [(f"niche {k}", niche == k) for k in np.unique(niche[niche >= 0])]:
            ok = sel & (n > 0)
            if ok.sum() >= 20:
                out.append({"what": kind, "ring": r, "scope": scope, "n_spots": int(ok.sum()), "attention": float(att[ok].mean()), "share_of_neighbours": float(share[ok].mean()),
                            "enrichment": float(att[ok].mean() / max(share[ok].mean(), 1e-9))})
    return out


def reassign_changed(Z0: np.ndarray, Z1: np.ndarray, C: np.ndarray, niche: np.ndarray) -> pd.DataFrame:
    a0 = np.linalg.norm(Z0[:, None] - C[None], axis=2).argmin(1); a1 = np.linalg.norm(Z1[:, None] - C[None], axis=2).argmin(1)
    return pd.DataFrame({"niche": niche, "changed": a0 != a1}).groupby("niche")["changed"].mean()


def gene_attribution(model, g, device: str, direction: torch.Tensor, rows: np.ndarray, back: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Gradient × input on genes of the score z·direction summed over `rows`; gradients reach every spot that enters those
    latents (centres and their neighbours) and are mapped from the expression PCs to genes through the PCA loadings `back`
    (n_pc × n_genes, already divided by the PC scale)."""
    s = prep(g, device); x = s.x.clone().requires_grad_(True); r = torch.tensor(rows, device=device)
    h, c = model.parts_rows(x, s.nb, r); z = model.z_from(x, s.nb, r, h, c)
    (z @ direction).sum().backward()
    gpc = x.grad[:, :back.shape[0]].cpu().numpy()  # expression PCs only
    ggene = gpc @ back
    return (ggene * x_std).sum(0) / len(rows)
