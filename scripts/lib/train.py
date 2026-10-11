"""Self-supervised training of OS-NicheGNN / MLP-AE / NeighMean-AE on a list of SectionGraphs.

Masked-expression reconstruction: a random `mask_frac` of (spot, gene) entries is zeroed in the input;
the loss is the MSE on exactly those entries. The masked tensor is the only expression the network sees —
for NeighMean-AE the ring means are recomputed from the masked input each step, so no branch can copy
the target through a side door. The auxiliary context target (ring-2–3 mean of the *unmasked* expression)
is a target, never an input.
Patient-balanced sampling: each step draws `batch_sections_per_step` patients uniformly, then one section
per patient. Early stopping on the masked loss of the inner validation patient(s).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .data import SectionGraph
from .model import MLPAE, NeighMeanAE, OSNicheGNN, count_parameters, masked_mse
from .upgrade import UpgradeHeads
from .spatial import ring_neighbors


@dataclass
class Prepared:
    sg: SectionGraph
    x: torch.Tensor
    e_near: torch.Tensor
    e_outer: torch.Tensor
    A1: torch.Tensor      # sparse row-normalised ring-1 averaging matrix
    A23: torch.Tensor     # sparse row-normalised ring-2–3 averaging matrix
    outer_mean: torch.Tensor
    nb_flat: torch.Tensor | None = None   # ring-1 neighbour lists (CSR-style) for neighbour-positive sampling (A7)
    nb_ptr: torch.Tensor | None = None
    nb_deg: torch.Tensor | None = None


def _avg_matrix(nb: list[np.ndarray], n: int) -> sp.csr_matrix:
    rows = np.concatenate([np.full(len(v), i) for i, v in enumerate(nb)] + [np.where([len(v) == 0 for v in nb])[0]])
    cols = np.concatenate(nb + [np.where([len(v) == 0 for v in nb])[0]])
    w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in nb if len(v)] + [np.ones(sum(len(v) == 0 for v in nb))])
    return sp.csr_matrix((w, (rows, cols)), shape=(n, n))


def _to_torch_sparse(A: sp.csr_matrix, device: str) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data, dtype=torch.float32), A.shape).coalesce().to(device)


def prepare(sg: SectionGraph, device: str, rewire_seed: int | None = None) -> Prepared:
    e_near, e_outer = sg.edge_near, sg.edge_outer
    nb1 = ring_neighbors(sg.xy, sg.pitch, rings=(1,)); nb23 = ring_neighbors(sg.xy, sg.pitch, rings=(2, 3))
    if rewire_seed is not None:  # diagnostic: same degree sequence, random partners
        rng = np.random.default_rng(rewire_seed)
        perm = rng.permutation(sg.n)
        e_near = perm[e_near]; e_outer = perm[e_outer]
        nb1 = [perm[v] for v in nb1]; nb23 = [perm[v] for v in nb23]
    A1 = _to_torch_sparse(_avg_matrix(nb1, sg.n), device); A23 = _to_torch_sparse(_avg_matrix(nb23, sg.n), device)
    x = torch.tensor(sg.x, device=device)
    deg = np.array([len(v) for v in nb1]); ptr = np.concatenate([[0], np.cumsum(deg)[:-1]]); flat = np.concatenate([v for v in nb1 if len(v)] + [np.zeros(0, int)])
    return Prepared(sg, x, torch.tensor(e_near, dtype=torch.long, device=device), torch.tensor(e_outer, dtype=torch.long, device=device),
                    A1, A23, torch.sparse.mm(A23, x), torch.tensor(flat, dtype=torch.long, device=device), torch.tensor(ptr, dtype=torch.long, device=device), torch.tensor(deg, dtype=torch.long, device=device))


def make_model(kind: str, n_genes: int, cfg: dict, ablation: str | None = None) -> torch.nn.Module:
    m = cfg["model"]
    if kind == "nichegnn":
        kw = dict(expr_hidden=m["expr_hidden"], branch_dim=m["branch_dim"], fuse_dim=m["fuse_dim"], n_layers=m["n_layers"], heads=m["heads"], dropout=m["dropout"],
                  use_outer=m["use_outer"], use_diff=m["use_diff"], use_aux_context=m["use_aux_context"])
        if ablation == "no_outer":
            kw.update(use_outer=False, use_diff=False)
        elif ablation == "no_diff":
            kw.update(use_diff=False)
        elif ablation == "no_aux":
            kw.update(use_aux_context=False)
        return OSNicheGNN(n_genes, **kw)
    if kind == "nichegnn_hybrid":  # plan v2 amendment A8: neighbourhood-mean features appended to the encoder input
        kw = dict(expr_hidden=m["expr_hidden"], branch_dim=m["branch_dim"], fuse_dim=m["fuse_dim"], n_layers=m["n_layers"], heads=m["heads"], dropout=m["dropout"],
                  use_outer=m["use_outer"], use_diff=m["use_diff"], use_aux_context=m["use_aux_context"], in_dim=3 * n_genes)
        return OSNicheGNN(n_genes, **kw)
    if kind == "mlp_ae":
        b = cfg["baselines"]["mlp_ae"]; return MLPAE(n_genes, b["hidden"], b["latent"], m["dropout"])
    if kind == "neigh_mean_ae":
        b = cfg["baselines"]["neigh_mean_ae"]; return NeighMeanAE(n_genes, b["hidden"], b["latent"], m["dropout"])
    raise ValueError(kind)


def _forward(model, p: Prepared, x_in: torch.Tensor, kind: str, no_edges: bool):
    if kind == "neigh_mean_ae":
        return model(x_in, torch.sparse.mm(p.A1, x_in), torch.sparse.mm(p.A23, x_in))
    if kind == "nichegnn_hybrid":
        return model(torch.cat([x_in, torch.sparse.mm(p.A1, x_in), torch.sparse.mm(p.A23, x_in)], 1), p.e_near, p.e_outer)
    if no_edges:
        empty = torch.zeros((2, 0), dtype=torch.long, device=x_in.device)
        return model(x_in, empty, empty)
    return model(x_in, p.e_near, p.e_outer)


def train_model(kind: str, train_graphs: list[SectionGraph], val_graphs: list[SectionGraph], cfg: dict, seed: int,
                ablation: str | None = None, device: str = "cuda", verbose: bool = False, upgrade: dict | None = None) -> tuple[torch.nn.Module, dict]:
    """`upgrade` (post hoc): {"lambda_adv": float, "lambda_con": float} adds a patient-adversarial and/or a
    view-contrastive term for kind == "nichegnn"; the encoder and its inputs are unchanged."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr = cfg["train"]
    n_genes = train_graphs[0].x.shape[1]
    model = make_model(kind, n_genes, cfg, ablation).to(device)
    up = upgrade or {}; lam_adv, lam_con = float(up.get("lambda_adv", 0.0)), float(up.get("lambda_con", 0.0))
    lam_nbr, lam_anc, anchors = float(up.get("lambda_nbr", 0.0)), float(up.get("lambda_anchor", 0.0)), up.get("anchors") or {}
    if anchors:
        anchors = {k: (torch.tensor(v[0], dtype=torch.long, device=device), torch.tensor(v[1], dtype=torch.long, device=device)) for k, v in anchors.items()}
    patients = sorted({g.patient for g in train_graphs}); pidx = {p: i for i, p in enumerate(patients)}
    heads = UpgradeHeads(cfg["model"]["fuse_dim"], len(patients), cfg["upgrade"]["adv_hidden"], cfg["model"]["dropout"]).to(device) if (lam_adv > 0 or lam_con > 0 or lam_nbr > 0 or lam_anc > 0) else None
    params = list(model.parameters()) + (list(heads.parameters()) if heads is not None else [])
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    rewire = seed if ablation == "rewired" else None
    no_edges = ablation == "no_edges"
    P_tr = [prepare(g, device, rewire) for g in train_graphs]
    P_va = [prepare(g, device, rewire) for g in val_graphs]
    by_pat = {pt: [p for p in P_tr if p.sg.patient == pt] for pt in patients}
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    steps_per_epoch = max(1, int(np.ceil(len(P_tr) / tr["batch_sections_per_step"])))
    best, best_state, bad, hist = np.inf, None, 0, []
    lam = tr["lambda_ctx"] if (kind in ("nichegnn", "nichegnn_hybrid") and ablation != "no_aux") else 0.0
    t0 = time.time()
    for epoch in range(tr["max_epochs"]):
        model.train(); tot = 0.0
        for _ in range(steps_per_epoch):
            pats = rng.choice(patients, size=min(tr["batch_sections_per_step"], len(patients)), replace=False)
            loss = 0.0; zs = {}
            for pt in pats:
                p = by_pat[pt][rng.integers(len(by_pat[pt]))]
                mask = (torch.rand(p.x.shape, generator=gen, device=device) < tr["mask_frac"]).float()
                x_in = p.x * (1 - mask) / ((1 - tr["mask_frac"]) if tr.get("inverted_mask") else 1.0)  # inverted masking keeps the input scale of inference
                z, xhat, chat = _forward(model, p, x_in, kind, no_edges); zs[p.sg.section] = z
                l = masked_mse(xhat, p.x, mask)
                if lam > 0 and chat is not None:
                    l = l + lam * torch.mean((chat - p.outer_mean) ** 2)
                if heads is not None:
                    if lam_adv > 0:
                        l = l + lam_adv * heads.adversarial_loss(z, pidx[p.sg.patient])
                    if lam_con > 0:
                        mask2 = (torch.rand(p.x.shape, generator=gen, device=device) < tr["mask_frac"]).float()
                        z2, _, _ = _forward(model, p, p.x * (1 - mask2), kind, no_edges)
                        l = l + lam_con * heads.contrastive_loss(z, z2, cfg["upgrade"]["tau"], cfg["upgrade"]["con_subsample"], gen)
                    if lam_nbr > 0:
                        l = l + lam_nbr * heads.neighbour_loss(z, p.nb_flat, p.nb_ptr, p.nb_deg, cfg["upgrade"]["tau"], cfg["upgrade"]["con_subsample"], gen)
                loss = loss + l
            loss = loss / len(pats)
            if lam_anc > 0 and len(zs) == 2:
                (sa, za_), (sb, zb_) = list(zs.items())
                if (sa, sb) in anchors:
                    ia, ib = anchors[(sa, sb)]; loss = loss + lam_anc * heads.anchor_loss(za_, zb_, ia, ib, cfg["upgrade"]["tau"], cfg["upgrade"]["con_subsample"], gen)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            tot += float(loss)
        val = evaluate_masked(model, P_va, kind, no_edges, cfg, seed) if P_va else tot / steps_per_epoch
        hist.append({"epoch": epoch, "train_loss": tot / steps_per_epoch, "val_loss": val})
        if verbose and epoch % 10 == 0:
            print(f"  ep{epoch:3d} train {tot / steps_per_epoch:.4f} val {val:.4f}", flush=True)
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tr["patience"]:
                break
    model.load_state_dict(best_state)
    info = {"kind": kind, "ablation": ablation, "seed": seed, "epochs_run": len(hist), "best_val_masked_mse": float(best), "upgrade": {k: v for k, v in up.items() if k != "anchors"} or None,
            "n_params": count_parameters(model), "train_time_s": round(time.time() - t0, 1), "history": hist}
    return model, info


@torch.no_grad()
def evaluate_masked(model, P: list[Prepared], kind: str, no_edges: bool, cfg: dict, seed: int) -> float:
    model.eval()
    gen = torch.Generator(device=P[0].x.device); gen.manual_seed(seed + 999)
    tot, n = 0.0, 0
    for p in P:
        mask = (torch.rand(p.x.shape, generator=gen, device=p.x.device) < cfg["train"]["mask_frac"]).float()
        x_in = p.x * (1 - mask) / ((1 - cfg["train"]["mask_frac"]) if cfg["train"].get("inverted_mask") else 1.0)
        _, xhat, _ = _forward(model, p, x_in, kind, no_edges)
        tot += float(masked_mse(xhat, p.x, mask)) * p.sg.n; n += p.sg.n
    return tot / max(n, 1)


@torch.no_grad()
def embed_sections(model, graphs: list[SectionGraph], kind: str, cfg: dict, device: str = "cuda", ablation: str | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Frozen forward pass on complete (unmasked) sections → embedding per section."""
    model.eval()
    out = {}
    rewire = seed if ablation == "rewired" else None
    no_edges = ablation == "no_edges"
    for g in graphs:
        p = prepare(g, device, rewire)
        if kind == "neigh_mean_ae":
            z = model.embed(p.x, torch.sparse.mm(p.A1, p.x), torch.sparse.mm(p.A23, p.x))
        elif kind == "nichegnn_hybrid":
            z = model.embed(torch.cat([p.x, torch.sparse.mm(p.A1, p.x), torch.sparse.mm(p.A23, p.x)], 1), p.e_near, p.e_outer)
        elif no_edges:
            empty = torch.zeros((2, 0), dtype=torch.long, device=device); z = model.embed(p.x, empty, empty)
        else:
            z = model.embed(p.x, p.e_near, p.e_outer)
        out[g.section] = z.detach().cpu().numpy()
    return out
