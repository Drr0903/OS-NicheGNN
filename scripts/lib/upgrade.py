"""post hoc upgrade heads for OS-NicheGNN (patient-adversarial, view-contrastive, spatial-neighbour and
cross-patient anchor positives).

Neither head changes the encoder's inputs: the adversarial head only receives z (through a gradient-reversal layer) and the
training patient's index as *target*; the contrastive head compares two independently masked views of the same section.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from .cluster import leiden_fixed_k


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    return _GradReverse.apply(x)


class UpgradeHeads(nn.Module):
    """Patient classifier on reversed gradients + projection head for NT-Xent."""

    def __init__(self, z_dim: int, n_patients: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.patient = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_patients))
        self.proj = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def adversarial_loss(self, z: torch.Tensor, patient_idx: int) -> torch.Tensor:
        logits = self.patient(grad_reverse(z))
        target = torch.full((z.shape[0],), patient_idx, dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, target)

    def pair_nt_xent(self, za: torch.Tensor, zb: torch.Tensor, tau: float) -> torch.Tensor:
        """Symmetric NT-Xent between paired rows of za and zb (positives on the diagonal, all other rows negatives)."""
        a = F.normalize(self.proj(za), dim=1); b = F.normalize(self.proj(zb), dim=1)
        logits = a @ b.t() / tau; target = torch.arange(a.shape[0], device=za.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, tau: float, n_sub: int, gen: torch.Generator) -> torch.Tensor:
        """View-contrastive term (A4): two masked views of the same spots; negatives = other spots of the same section."""
        n = z1.shape[0]; idx = torch.randperm(n, generator=gen, device=z1.device)[: min(n_sub, n)]
        return self.pair_nt_xent(z1[idx], z2[idx], tau)

    def neighbour_loss(self, z: torch.Tensor, nb_flat: torch.Tensor, nb_ptr: torch.Tensor, nb_deg: torch.Tensor, tau: float, n_sub: int, gen: torch.Generator) -> torch.Tensor:
        """Spatial-neighbour positives (A7): each spot with a random ring-1 neighbour; negatives = other sampled spots."""
        has = torch.nonzero(nb_deg > 0).squeeze(1); idx = has[torch.randperm(len(has), generator=gen, device=z.device)[: min(n_sub, len(has))]]
        r = (torch.rand(len(idx), generator=gen, device=z.device) * nb_deg[idx]).long(); j = nb_flat[nb_ptr[idx] + r]
        return self.pair_nt_xent(z[idx], z[j], tau)

    def anchor_loss(self, za: torch.Tensor, zb: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor, tau: float, n_sub: int, gen: torch.Generator) -> torch.Tensor:
        """Cross-patient anchor positives (A7): mutual-nearest-neighbour pairs (ia in section a, ib in section b)."""
        sel = torch.randperm(len(ia), generator=gen, device=za.device)[: min(n_sub, len(ia))]
        return self.pair_nt_xent(za[ia[sel]], zb[ib[sel]], tau)


def patient_ari(emb: dict[str, np.ndarray], patient_of: dict[str, str], K: int, seed: int, per_patient: int = 500) -> float:
    """ARI between Leiden clusters (fixed K, A5) of a patient-balanced subsample and patient identity (diagnostic used for selection)."""
    rng = np.random.default_rng(seed); Z, P = [], []
    for pat in sorted(set(patient_of[s] for s in emb)):
        zs = np.vstack([emb[s] for s in emb if patient_of[s] == pat]); idx = rng.choice(len(zs), min(per_patient, len(zs)), replace=False)
        Z.append(zs[idx]); P += [pat] * len(idx)
    Z = np.vstack(Z); lab, _ = leiden_fixed_k(Z, K, seed, batch=np.array(P), k_within=3)
    return float(adjusted_rand_score(P, lab))


def mnn_anchors(graphs, n_pcs: int = 50, k: int = 10, seed: int = 0, max_spots: int = 1500, gate: bool = False) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Mutual-nearest-neighbour pairs between spots of different patients in PCA space of the standardised expression of the
    given (training) sections, on a random subsample of ≤ max_spots spots per section (indices refer to the full section).
    Returns {(section_a, section_b): (idx_a, idx_b)} for both orders. No labels, no held-out data."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed); X = np.vstack([g.x for g in graphs]); pca = PCA(n_pcs, random_state=seed).fit(X)
    sub = [rng.choice(g.n, min(max_spots, g.n), replace=False) for g in graphs]; Z = [pca.transform(g.x[i]) for g, i in zip(graphs, sub)]
    nn = [NearestNeighbors(n_neighbors=min(k, len(z))).fit(z) for z in Z]; out = {}
    radius = [float(np.median(nn_.kneighbors(z)[0][:, -1])) for nn_, z in zip(nn, Z)] if gate else None  # median within-section k-NN radius (v2-A9 gate)
    for a in range(len(graphs)):
        for b in range(a + 1, len(graphs)):
            if graphs[a].patient == graphs[b].patient:
                continue
            dab, nab = nn[b].kneighbors(Z[a]); _, nba = nn[a].kneighbors(Z[b])
            back = np.zeros((len(Z[b]), len(Z[a])), dtype=bool); back[np.repeat(np.arange(len(Z[b])), nba.shape[1]), nba.ravel()] = True
            ia_l = np.repeat(np.arange(len(Z[a])), nab.shape[1]); ib_l = nab.ravel(); mutual = back[ib_l, ia_l]
            if gate:
                mutual &= dab.ravel() <= max(radius[a], radius[b])
            if mutual.any():
                ia = sub[a][ia_l[mutual]]; ib = sub[b][ib_l[mutual]]
                out[(graphs[a].section, graphs[b].section)] = (ia, ib); out[(graphs[b].section, graphs[a].section)] = (ib, ia)
    return out
