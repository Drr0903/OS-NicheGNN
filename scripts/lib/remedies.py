"""objectives that make the OS-NicheGNN encoder depend on its neighbours.

The encoder is the unchanged dual-scale OS-NicheGNN (centre / ring-1 / ring-2–3); only the training signal changes.
    mae   GraphMAE (Hou et al., KDD 2022): a fraction of whole spots is replaced by a learnable mask token at the input, the
          embedding of those spots is re-masked, a ring-1 graph-attention decoder reconstructs them, and the loss is the
          scaled cosine error on the masked spots only. A masked spot's own expression is never visible, so the reconstruction
          must come from its neighbours.
    dgi   deep graph infomax with a local summary (as in GraphST, Long et al., Nat Commun 2023): positives are (z_i, s_i) with
          s_i the ring-1 mean embedding, negatives use the same graph on row-shuffled expression; added to the feature-masked
          reconstruction of the original model.
Optional anchor-only term (A10) reuses UpgradeHeads.anchor_loss unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .data import SectionGraph
from .model import OSNicheGNN, count_parameters, masked_mse
from .train import Prepared, prepare
from .upgrade import UpgradeHeads


class MAEDecoder(nn.Module):
    def __init__(self, z_dim: int, out_dim: int):
        super().__init__()
        self.dec_token = nn.Parameter(torch.zeros(1, z_dim))
        self.gat = GATv2Conv(z_dim, out_dim, heads=1, add_self_loops=True)

    def forward(self, z: torch.Tensor, node_mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = torch.where(node_mask[:, None], self.dec_token.expand_as(z), z)  # re-mask
        return self.gat(z, edge_index)


class RemedyModel(nn.Module):
    """OS-NicheGNN encoder + objective-specific heads. `embed` is the frozen representation used everywhere downstream."""

    def __init__(self, in_dim: int, cfg: dict, objective: str):
        super().__init__()
        m = cfg["model"]; self.objective = objective
        self.gnn = OSNicheGNN(in_dim, m["expr_hidden"], m["branch_dim"], m["fuse_dim"], m["n_layers"], m["heads"], m["dropout"],
                              m["use_outer"], m["use_diff"], use_aux_context=(objective == "dgi"))
        if objective == "mae":
            self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
            self.decoder = MAEDecoder(m["fuse_dim"], in_dim)
        if objective == "dgi":
            self.W = nn.Parameter(torch.empty(m["fuse_dim"], m["fuse_dim"])); nn.init.xavier_uniform_(self.W)

    def embed(self, x, e_near, e_outer):
        return self.gnn.embed(x, e_near, e_outer)


def sce_loss(xhat: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1 - F.cosine_similarity(xhat, x, dim=1)).pow(alpha).mean()


def _mae_step(model: RemedyModel, p: Prepared, rc: dict, gen: torch.Generator):
    n = p.x.shape[0]; node_mask = torch.rand(n, generator=gen, device=p.x.device) < rc["mae_mask_rate"]
    x_in = torch.where(node_mask[:, None], model.mask_token.expand_as(p.x), p.x)
    z = model.embed(x_in, p.e_near, p.e_outer)
    xhat = model.decoder(z, node_mask, p.e_near)
    return z, sce_loss(xhat[node_mask], p.x[node_mask], rc["mae_alpha"])


def _dgi_step(model: RemedyModel, p: Prepared, cfg: dict, rc: dict, gen: torch.Generator):
    tr = cfg["train"]; mask = (torch.rand(p.x.shape, generator=gen, device=p.x.device) < tr["mask_frac"]).float()
    z, xhat, chat = model.gnn(p.x * (1 - mask), p.e_near, p.e_outer)
    l = masked_mse(xhat, p.x, mask) + tr["lambda_ctx"] * torch.mean((chat - p.outer_mean) ** 2)
    perm = torch.randperm(p.x.shape[0], generator=gen, device=p.x.device)
    z_neg = model.embed(p.x[perm], p.e_near, p.e_outer)
    s = torch.sigmoid(torch.sparse.mm(p.A1, z))
    pos = ((z @ model.W) * s).sum(1); neg = ((z_neg @ model.W) * s).sum(1)
    dgi = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
    return z, l + rc["dgi_weight"] * dgi


def objective_loss(model, p, cfg, gen):
    rc = cfg["remedies"]
    return _mae_step(model, p, rc, gen) if model.objective == "mae" else _dgi_step(model, p, cfg, rc, gen)


@torch.no_grad()
def _val_loss(model, P_va, cfg, seed):
    model.eval(); gen = torch.Generator(device=P_va[0].x.device); gen.manual_seed(seed + 999); tot = n = 0
    for p in P_va:
        tot += float(objective_loss(model, p, cfg, gen)[1]) * p.sg.n; n += p.sg.n
    return tot / max(n, 1)


def train_remedy(objective: str, tr_g: list[SectionGraph], va_g: list[SectionGraph], cfg: dict, seed: int, device: str,
                 anchors: dict | None = None, lambda_anchor: float = 0.0, factory=None) -> tuple[RemedyModel, dict]:
    """`factory(in_dim, cfg)` builds an alternative model with the same interface (A18 self-versus-context attention)."""
    torch.manual_seed(seed); np.random.seed(seed); tr = cfg["train"]
    model = (factory(tr_g[0].x.shape[1], cfg) if factory else RemedyModel(tr_g[0].x.shape[1], cfg, objective)).to(device)
    heads = UpgradeHeads(getattr(model, "z_dim", cfg["model"]["fuse_dim"]), 1, cfg["upgrade"]["adv_hidden"], cfg["model"]["dropout"]).to(device) if lambda_anchor > 0 else None
    params = list(model.parameters()) + (list(heads.parameters()) if heads is not None else [])
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    anc = {k: (torch.tensor(a, device=device), torch.tensor(b, device=device)) for k, (a, b) in (anchors or {}).items()}
    P_tr = [prepare(g, device) for g in tr_g]; P_va = [prepare(g, device) for g in va_g]
    patients = sorted({g.patient for g in tr_g}); by_pat = {pt: [p for p in P_tr if p.sg.patient == pt] for pt in patients}
    rng = np.random.default_rng(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
    spe = max(1, int(np.ceil(len(P_tr) / tr["batch_sections_per_step"]))); best, best_state, bad, hist = np.inf, None, 0, []; t0 = time.time()
    for epoch in range(tr["max_epochs"]):
        model.train(); tot = 0.0
        for _ in range(spe):
            pats = rng.choice(patients, size=min(tr["batch_sections_per_step"], len(patients)), replace=False); loss = 0.0; zs = {}
            for pt in pats:
                p = by_pat[pt][rng.integers(len(by_pat[pt]))]; z, l = objective_loss(model, p, cfg, gen); zs[p.sg.section] = z; loss = loss + l
            loss = loss / len(pats)
            if heads is not None and len(zs) == 2:
                (sa, za), (sb, zb) = list(zs.items())
                if (sa, sb) in anc:
                    ia, ib = anc[(sa, sb)]; loss = loss + lambda_anchor * heads.anchor_loss(za, zb, ia, ib, cfg["upgrade"]["tau"], cfg["upgrade"]["con_subsample"], gen)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); tot += float(loss)
        val = _val_loss(model, P_va, cfg, seed) if P_va else tot / spe; hist.append(val)
        if val < best - 1e-5:
            best, bad, best_state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tr["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"objective": objective, "seed": seed, "epochs_run": len(hist), "best_val": float(best), "n_params": count_parameters(model), "train_time_s": round(time.time() - t0, 1)}


@torch.no_grad()
def embed_remedy(model: RemedyModel, graphs: list[SectionGraph], device: str, rewire_seed: int | None = None) -> dict[str, np.ndarray]:
    model.eval(); out = {}
    for g in graphs:
        p = prepare(g, device, rewire_seed); out[g.section] = model.embed(p.x, p.e_near, p.e_outer).cpu().numpy()
    return out


@torch.no_grad()
def embed_skip(gnn: OSNicheGNN, graphs_pca: list[SectionGraph], device: str) -> dict[str, np.ndarray]:
    """A17 third set: [own PCA profile, ring-1 branch output, ring-2–3 branch output]; the branches exclude the spot itself."""
    gnn.eval(); out = {}
    for g in graphs_pca:
        p = prepare(g, device); e = gnn.enc(p.x); parts = [p.x, gnn.near(e, p.e_near)]
        if gnn.use_outer:
            parts.append(gnn.outer(e, p.e_outer))
        out[g.section] = torch.cat(parts, 1).cpu().numpy()
    return out
