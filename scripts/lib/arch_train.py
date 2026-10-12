"""training and embedding for lib.arch.NicheEncoder.

Steps are patient-balanced as in stage 14 (two patients per step, one section each). In every step a fraction of the
section's spots is masked (replaced by the learnable mask token wherever it appears, as centre or as neighbour); the loss is
computed on a random subset of ≤ `centre_batch` centre spots: MSE on masked spots + w_vis × MSE on visible spots.
The optional anchor-only term (A10) reuses UpgradeHeads.anchor_loss on the latent z of anchor spots.
Early stopping on the inner-validation patient's loss with a fixed generator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .arch import NicheEncoder, niche_target_values
from .arch_ext import MoEFusion, attach
from .data import SectionGraph
from .model import count_parameters
from .nbhd import Nbhd, build_nbhd
from .upgrade import UpgradeHeads


@dataclass
class Sec:
    sg: SectionGraph
    x: torch.Tensor
    nb: Nbhd
    tgt: dict | None = None


def prep(g: SectionGraph, device: str, rewire_seed: int | None = None, targets: bool = False, max_ring: int = 3) -> Sec:
    x = torch.tensor(g.x, device=device); nb = build_nbhd(g.xy, g.pitch, device, rewire_seed)
    if max_ring < 3:  # single-section mode: neighbourhood restricted to rings 1..max_ring
        keep = nb.valid & (nb.pos[..., 2:2 + max_ring].sum(-1) > 0); nb = Nbhd(torch.where(keep, nb.idx, -1), keep, nb.pos * keep[..., None])
    return Sec(g, x, nb, niche_target_values(x, nb) if targets else None)


def _loss(model: NicheEncoder, s: Sec, ac: dict, gen: torch.Generator, rows: torch.Tensor | None = None):
    n = s.x.shape[0]; dev = s.x.device
    mask = torch.rand(n, generator=gen, device=dev) < ac["spot_mask_rate"]
    if rows is None:
        rows = torch.randperm(n, generator=gen, device=dev)[: ac["centre_batch"]]
    xm = torch.where(mask[:, None], model.mask_token.expand_as(s.x), s.x)
    pd_ = ac.get("morph_drop", 0.0)
    if pd_ > 0 and model.training and float(torch.rand(1, generator=gen, device=dev)) < pd_:  # the section is seen without its image
        xm = xm.clone(); xm[:, 50:100] = 0.0  # morphology PCA block (columns 50–99); 0 = training mean; targets keep the true values
    h, c = model.parts_rows(xm, s.nb, rows); z = model.z_from(xm, s.nb, rows, h, c); xhat = model.dec(z)
    err = ((xhat - s.x[rows]) ** 2).mean(1); m = mask[rows].float()
    l = (err * m).sum() / m.sum().clamp_min(1) + ac["w_visible"] * (err * (1 - m)).sum() / (1 - m).sum().clamp_min(1)
    if ac.get("nt_masked"):  # surroundings of the masked neighbours only: values that are absent from the visible input
        from .nbhd import Nbhd as _N
        nbm = _N(s.nb.idx, s.nb.valid & mask[s.nb.idx.clamp_min(0)], s.nb.pos); tg = niche_target_values(s.x, nbm)
        ok = {"ring1": (nbm.valid & (nbm.pos[..., 2] > 0)).any(1), "outer": (nbm.valid & ((nbm.pos[..., 3] + nbm.pos[..., 4]) > 0)).any(1), "disp": nbm.valid.sum(1) > 1}
        for t, w in (ac.get("niche_targets") or {}).items():
            o = ok[t][rows].float(); l = l + w * (((model.ndec[t](z) - tg[t][rows]) ** 2).mean(1) * o).sum() / o.sum().clamp_min(1)
    else:
        for t, w in (ac.get("niche_targets") or {}).items():  # the latent must also describe the spot's surroundings
            l = l + w * ((model.ndec[t](z) - s.tgt[t][rows]) ** 2).mean()
    if getattr(model, "moe", None) is not None:  # keep every expert in use
        l = l + ac.get("moe_balance", 0.01) * MoEFusion.balance(model.last_gates)
    if getattr(model.fuse, "last_gates", None) is not None:  # niche-adaptive experts: keep every expert in use
        l = l + ac.get("moe_balance", 0.01) * MoEFusion.balance(model.fuse.last_gates)
    if getattr(model, "clip", None) is not None:  # align the latent with the morphology of the same spots
        l = l + ac["clip"] * model.clip.loss(z, s.x[rows])
    if getattr(model, "dgi", None) is not None:  # true graph against permuted profiles (deep graph infomax)
        perm = torch.randperm(n, generator=gen, device=dev); z_neg = model.embed_rows(xm[perm], s.nb, rows)
        if ac.get("dgi_local") == "nbz" and len(rows) == n:  # GraphST-style readout: mean latent of each spot's ring-1 neighbours
            zf = torch.zeros_like(z); zf[rows] = z; v = (s.nb.valid & (s.nb.pos[..., 2] > 0)).float()
            summ = (v[..., None] * zf[s.nb.idx.clamp_min(0)]).sum(1) / v.sum(1, keepdim=True).clamp_min(1)
            l = l + ac["dgi"] * model.dgi.loss_local(z, z_neg, summ[rows])
        else:
            l = l + ac["dgi"] * model.dgi.loss(z, z_neg, c if ac.get("dgi_local") else None)
    if getattr(model, "site_adv", None) is not None:  # the latent must not tell the anatomical site
        l = l + ac["site_adv"] * model.site_adv.loss(z, ac["site_code"][s.sg.section])
    if ac.get("nbr_nce", 0) > 0:  # spatial contrast with a random ring-1 neighbour of each centre spot
        v = s.nb.valid[rows] & (s.nb.pos[rows][..., 2] > 0); k = torch.multinomial(v.float() + 1e-6, 1, generator=gen).squeeze(1)
        j = s.nb.idx[rows].gather(1, k[:, None]).squeeze(1); ok = v.any(1) & (j >= 0)
        if ok.sum() > 8:
            from .arch_ext import spatial_nce
            l = l + ac["nbr_nce"] * spatial_nce(z[ok], model.embed_rows(xm, s.nb, j[ok]))
    return z, rows, l


def train_arch(context: str, tr_g: list[SectionGraph], va_g: list[SectionGraph], cfg: dict, seed: int, device: str,
               anchors: dict | None = None, lambda_anchor: float = 0.0) -> tuple[NicheEncoder, dict]:
    torch.manual_seed(seed); np.random.seed(seed); tr, ac = cfg["train"], cfg["arch"]
    nt = {t: w for t, w in (ac.get("niche_targets") or {}).items() if w > 0}; ac = {**ac, "niche_targets": nt}
    model = NicheEncoder(tr_g[0].x.shape[1], context, ac["d"], ac["z_dim"], ac["heads"], ac["n_layers"], ac["dropout"], tuple(nt),
                         ac.get("linear_heads", False), ac.get("prop_skip", False), ac.get("grad_skip", False))
    model = attach(model, ac, tr_g[0].x.shape[1]).to(device); model.max_ring = int(ac.get("max_ring", 3))
    heads = UpgradeHeads(ac["z_dim"], 1, cfg["upgrade"]["adv_hidden"], ac["dropout"]).to(device) if lambda_anchor > 0 else None
    params = list(model.parameters()) + (list(heads.parameters()) if heads is not None else [])
    opt = torch.optim.AdamW(params, lr=ac["lr"], weight_decay=tr["weight_decay"])
    anc = {k: (torch.tensor(a, device=device), torch.tensor(b, device=device)) for k, (a, b) in (anchors or {}).items()}
    S_tr = [prep(g, device, targets=bool(nt), max_ring=model.max_ring) for g in tr_g]; S_va = [prep(g, device, targets=bool(nt), max_ring=model.max_ring) for g in va_g]
    patients = sorted({g.patient for g in tr_g}); by_pat = {p: [s for s in S_tr if s.sg.patient == p] for p in patients}
    rng = np.random.default_rng(seed); gen = torch.Generator(device=device); gen.manual_seed(seed)
    spe = max(1, int(np.ceil(len(S_tr) / tr["batch_sections_per_step"]))); best, best_state, bad, hist = np.inf, None, 0, []; t0 = time.time()
    for _ in range(ac["max_epochs"]):
        model.train(); tot = 0.0
        for _ in range(spe):
            pats = rng.choice(patients, size=min(tr["batch_sections_per_step"], len(patients)), replace=False); loss = 0.0; zs = []
            for p in pats:
                s = by_pat[p][rng.integers(len(by_pat[p]))]
                z, rows, l = _loss(model, s, ac, gen); zs.append((s, z, rows)); loss = loss + l
            loss = loss / len(pats)
            if heads is not None and len(zs) == 2:
                (sa, _, _), (sb, _, _) = zs
                key = (sa.sg.section, sb.sg.section)
                if key in anc:
                    ia, ib = anc[key]; sel = torch.randperm(len(ia), generator=gen, device=device)[: cfg["upgrade"]["con_subsample"]]
                    za = model.embed_rows(sa.x, sa.nb, ia[sel]); zb = model.embed_rows(sb.x, sb.nb, ib[sel])
                    loss = loss + lambda_anchor * heads.pair_nt_xent(za, zb, cfg["upgrade"]["tau"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); tot += float(loss)
        val = _val(model, S_va, ac, seed) if S_va else tot / spe; hist.append(val)
        if val < best - 1e-5:
            best, bad, best_state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ac["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"context": context, "seed": seed, "epochs_run": len(hist), "best_val": float(best), "n_params": count_parameters(model),
                   "train_time_s": round(time.time() - t0, 1)}


@torch.no_grad()
def _val(model, S, ac, seed):
    model.eval(); gen = torch.Generator(device=S[0].x.device); gen.manual_seed(seed + 999); tot = n = 0
    for s in S:
        rows = torch.arange(s.x.shape[0], device=s.x.device)
        for ch in rows.split(ac["centre_batch"]):
            _, _, l = _loss(model, s, ac, gen, ch); tot += float(l) * len(ch); n += len(ch)
    return tot / max(n, 1)


@torch.no_grad()
def embed_arch(model: NicheEncoder, graphs: list[SectionGraph], device: str, chunk: int = 4096, rewire_seed: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Per section: {"z": fused latent, "c": context output (None for the MLP-only model)} on complete, unmasked inputs."""
    model.eval(); out = {}
    for g in graphs:
        s = prep(g, device, rewire_seed, max_ring=getattr(model, "max_ring", 3)); zs, cs = [], []
        for ch in torch.arange(s.x.shape[0], device=device).split(chunk):
            h, c = model.parts_rows(s.x, s.nb, ch)
            zs.append(model.z_from(s.x, s.nb, ch, h, c).cpu()); cs.append(None if c is None else c.cpu())
        out[g.section] = {"z": torch.cat(zs).numpy(), "c": None if cs[0] is None else torch.cat(cs).numpy()}
    return out
