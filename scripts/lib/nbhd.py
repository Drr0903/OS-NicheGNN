"""per-spot neighbourhood token tables on the Visium hexagonal array.

For every spot: up to 36 neighbours (rings 1–3), ordered ring 3 → ring 2 → ring 1 and by angle within a ring, so the last
position of a sequence is the ring-1 neighbour nearest in angle to 0 and the spot itself can be appended after it.
Padded slots carry index −1 and a False validity flag. Relative positions are in pitch units; nothing but coordinates is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .spatial import RING_BOUNDS

MAX_TOKENS = 36


@dataclass
class Nbhd:
    idx: torch.Tensor     # (n, 36) long, −1 = padding
    valid: torch.Tensor   # (n, 36) bool
    pos: torch.Tensor     # (n, 36, 5) float: Δx, Δy, ring-1/2/3 one-hot (zeros on padding)


def ring_of(d: np.ndarray) -> np.ndarray:
    r = np.zeros(len(d), int)
    for k, (a, b) in RING_BOUNDS.items():
        r[(d >= a) & (d < b)] = k
    return r


def build_nbhd(xy: np.ndarray, pitch: float, device: str, rewire_seed: int | None = None) -> Nbhd:
    n = len(xy); tree = cKDTree(xy); pairs = tree.query_pairs(RING_BOUNDS[3][1] * pitch, output_type="ndarray")
    idx = np.full((n, MAX_TOKENS), -1, np.int64); pos = np.zeros((n, MAX_TOKENS, 5), np.float32)
    if len(pairs):
        src = np.concatenate([pairs[:, 0], pairs[:, 1]]); dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        rel = (xy[dst] - xy[src]) / pitch; ring = ring_of(np.linalg.norm(rel, axis=1)); keep = ring > 0
        src, dst, rel, ring = src[keep], dst[keep], rel[keep], ring[keep]
        ang = np.arctan2(rel[:, 1], rel[:, 0]); order = np.lexsort((ang, -ring, src))
        src, dst, rel, ring = src[order], dst[order], rel[order], ring[order]
        start = np.searchsorted(src, np.arange(n)); slot = np.arange(len(src)) - start[src]; ok = slot < MAX_TOKENS
        src, dst, rel, ring, slot = src[ok], dst[ok], rel[ok], ring[ok], slot[ok]
        if rewire_seed is not None:  # diagnostic: neighbours replaced by random spots of the same section, positions kept
            dst = np.random.default_rng(rewire_seed).permutation(n)[dst]
        idx[src, slot] = dst; pos[src, slot, 0:2] = rel; pos[src, slot, 1 + ring] = 1.0
    t = torch.tensor(idx, device=device)
    return Nbhd(t, t >= 0, torch.tensor(pos, device=device))
