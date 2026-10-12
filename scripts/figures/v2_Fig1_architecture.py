"""v2 Figure 1a — the OS-NicheGNN network and its objective, drawn as vector shapes (deep-learning architecture style).

Inputs (expression PCA, Phikon morphology PCA, ring 1–3 neighbourhood), masking, the ring-mean branch, neighbour tokens and two
cross-attention layers (graph attention branch), the mixture-of-experts fusion, the latent with its linear neighbourhood skip, the decoder and the three linear neighbourhood heads with
the two loss terms, and the use of the latent (atlas, mapping of a new patient, applications). Wording follows Methods.
The axes are in inches (W × H) so that shapes keep their proportions.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

W, H = 7.2, 3.25
C = {"ink": "#2f3d48", "edge": "#8a9aa5", "in": "#eef3f7", "enc": "#eceffa", "lat": "#e3f1f2", "head": "#f8eee4", "loss": "#fbe3df",
     "down": "#eaf3ec", "expr": "#2a7f8a", "img": "#c27b4f", "att": "#5b6bb5", "z": "#1f6f7a", "skip": "#8f7fbf",
     "centre": "#1f6f7a", "moe": "#9a4f8c", "r1": "#74b6bf", "r23": "#cfe6e9", "mask": "#9aa4ab"}
NICHE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]  # = lib.plotting.PALETTE, as in Figure 4
FS, FS_T, FS_S = 6.0, 6.6, 5.6


def box(ax, x, y, w, h, fc, ec=None, lw=0.6, r=0.06, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or C["edge"], lw=lw, zorder=z))


def text(ax, x, y, s, fs=FS, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", C["ink"]); kw.setdefault("zorder", 6)
    return ax.text(x, y, s, fontsize=fs, **kw)


def arrow(ax, x0, y0, x1, y1, col=None, lw=0.8, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=6, lw=lw, color=col or C["edge"],
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=4))


def hexagon(ax, cx, cy, r, **kw):
    a = np.deg2rad(np.arange(6) * 60 + 30)
    ax.add_patch(Polygon(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)], closed=True, zorder=3, **kw))


def lattice(ax, cx, cy, r):
    """centre spot with rings 1–3 on a hexagonal Visium lattice (one masked spot shown in grey)"""
    dx, dy = np.sqrt(3) * r, 1.5 * r
    for q in range(-3, 4):
        for s_ in range(-3, 4):
            ring = max(abs(q), abs(s_), abs(-q - s_))
            if ring > 3:
                continue
            x = cx + dx * (q + s_ / 2); y = cy + dy * s_
            fc = C["centre"] if ring == 0 else C["r1"] if ring == 1 else C["r23"]
            if (q, s_) in ((2, -1), (-1, 2), (1, 1)):
                fc = C["mask"]
            hexagon(ax, x, y, r * 0.97, fc=fc, ec="white", lw=0.4)


def vec(ax, x, y, n, w, h, col, horizontal=True):
    """a latent / feature vector as n small cells"""
    for k in range(n):
        if horizontal:
            ax.add_patch(Rectangle((x + k * w / n, y), w / n * 0.9, h, fc=col, ec="white", lw=0.3, zorder=3))
        else:
            ax.add_patch(Rectangle((x, y + k * h / n), w, h / n * 0.9, fc=col, ec="white", lw=0.3, zorder=3))


def column_title(ax, x, w, s):
    text(ax, x + w / 2, H - 0.11, s, fs=FS_T, fontweight="bold")


def draw(ax) -> None:
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    # ---- 1 input -----------------------------------------------------------------------------------------------
    x0, w0 = 0.04, 1.52; column_title(ax, x0, w0, "Input: spot and rings 1–3")
    box(ax, x0, 0.08, w0, H - 0.3, C["in"])
    lattice(ax, x0 + 0.40, 2.30, 0.068)
    for yy, fc, lab in [(2.62, C["centre"], "centre spot $i$"), (2.44, C["r1"], "ring 1"), (2.26, C["r23"], "rings 2–3"), (2.08, C["mask"], "masked, 40 %")]:
        hexagon(ax, x0 + 0.86, yy, 0.045, fc=fc, ec="white", lw=0.3); text(ax, x0 + 0.93, yy, lab, fs=FS_S - 0.2, ha="left")
    box(ax, x0 + 0.08, 1.30, w0 - 0.16, 0.42, "white", ec=C["expr"])
    text(ax, x0 + w0 / 2, 1.60, "expression, 2,000 genes", fs=FS_S); text(ax, x0 + w0 / 2, 1.43, r"PCA-50  $\rightarrow$  $\mathbf{e}_i$", fs=FS, color=C["expr"])
    box(ax, x0 + 0.08, 0.78, w0 - 0.16, 0.44, "white", ec=C["img"])
    text(ax, x0 + w0 / 2, 1.10, "H&E patch (3 spot diameters)", fs=FS_S)
    text(ax, x0 + w0 / 2, 0.92, r"Phikon, PCA-50  $\rightarrow$  $\mathbf{m}_i$", fs=FS, color=C["img"])
    vec(ax, x0 + 0.20, 0.30, 10, 0.55, 0.13, C["expr"]); vec(ax, x0 + 0.77, 0.30, 10, 0.55, 0.13, C["img"])
    text(ax, x0 + w0 / 2, 0.57, r"$\mathbf{x}_i=[\mathbf{e}_i;\mathbf{m}_i]\in\mathbb{R}^{100}$", fs=FS)
    text(ax, x0 + w0 / 2, 0.18, r"masked spots $\rightarrow$ learnable $\mathbf{x}_{\mathrm{mask}}$", fs=FS_S)
    # ---- 2 encoder ---------------------------------------------------------------------------------------------
    x1, w1 = 1.72, 2.30; column_title(ax, x1, w1, "Encoder")
    box(ax, x1, 0.08, w1, H - 0.3, C["enc"])
    box(ax, x1 + 0.08, 2.18, 1.02, 0.56, "white", ec=C["expr"])
    text(ax, x1 + 0.59, 2.60, "ring-mean branch", fs=FS, fontweight="bold"); text(ax, x1 + 0.59, 2.44, "residual MLP", fs=FS_S)
    text(ax, x1 + 0.59, 2.28, r"$[\mathbf{x}_i;\bar{\mathbf{x}}^{(1)}_i;\bar{\mathbf{x}}^{(2,3)}_i] \rightarrow \mathbf{h}_i$", fs=FS_S - 0.3)
    box(ax, x1 + 1.20, 2.18, 1.02, 0.56, "white", ec=C["att"])
    text(ax, x1 + 1.71, 2.58, "neighbour tokens", fs=FS, fontweight="bold"); text(ax, x1 + 1.71, 2.42, r"$\mathbf{t}_{ij}=\mathbf{W}_t\mathbf{x}_j+\phi(\boldsymbol{\delta}_{ij})$", fs=FS_S)
    text(ax, x1 + 1.71, 2.27, r"$j\in\mathcal{N}(i)$, $\leq 36$ spots", fs=FS_S)
    box(ax, x1 + 0.08, 1.30, w1 - 0.16, 0.68, "white", ec=C["att"], lw=0.9)
    text(ax, x1 + w1 / 2, 1.84, "graph attention branch  × 2 layers, 4 heads", fs=FS, fontweight="bold", color=C["att"])
    text(ax, x1 + w1 / 2, 1.66, r"query $\mathbf{h}_i$, keys and values $\mathbf{t}_{ij}$", fs=FS_S)
    text(ax, x1 + w1 / 2, 1.48, r"$\alpha_{ij}=\mathrm{softmax}_j(\mathbf{q}_i^{\top}\mathbf{k}_{ij}/\sqrt{d/H})$  $\rightarrow$  context $\mathbf{c}_i$", fs=FS_S)
    arrow(ax, x1 + 0.59, 2.18, x1 + 0.59, 1.98, C["expr"]); arrow(ax, x1 + 1.71, 2.18, x1 + 1.71, 1.98, C["att"])
    box(ax, x1 + 0.08, 0.62, 1.02, 0.54, "white", ec=C["moe"], lw=0.9)
    text(ax, x1 + 0.59, 1.04, "mixture of experts", fs=FS, fontweight="bold", color=C["moe"])
    text(ax, x1 + 0.59, 0.88, r"$\pi_{i1}g_1+\pi_{i2}g_2$ on $[\mathbf{h}_i;\mathbf{c}_i]$", fs=FS_S)
    text(ax, x1 + 0.59, 0.72, r"router: softmax of $[\mathbf{h}_i;\mathbf{c}_i]$", fs=FS_S - 0.3)
    box(ax, x1 + 1.20, 0.66, 1.02, 0.46, "white", ec=C["skip"])
    text(ax, x1 + 1.71, 0.96, r"$\mathbf{W}_s[\mathbf{x}_i;\bar{\mathbf{x}}^{(1)}_i;\bar{\mathbf{x}}^{(2,3)}_i]$", fs=5.0, color=C["skip"])
    text(ax, x1 + 1.71, 0.78, "linear skip of ring means", fs=FS_S - 0.2)
    arrow(ax, x1 + 0.59, 1.30, x1 + 0.59, 1.16, C["att"])
    ax.add_patch(Circle((x1 + w1 / 2, 0.36), 0.075, fc="white", ec=C["z"], lw=0.8, zorder=5)); text(ax, x1 + w1 / 2, 0.36, "+", fs=7, fontweight="bold", color=C["z"])
    arrow(ax, x1 + 0.59, 0.62, x1 + w1 / 2 - 0.08, 0.38, C["moe"]); arrow(ax, x1 + 1.71, 0.66, x1 + w1 / 2 + 0.08, 0.38, C["skip"])
    # input → encoder arrows
    ax.plot([x0 + 1.36, x0 + w0 + 0.08, x0 + w0 + 0.08], [0.365, 0.365, 2.40], color=C["expr"], lw=0.8, zorder=4); arrow(ax, x0 + w0 + 0.08, 2.40, x1 + 0.08, 2.40, C["expr"]); arrow(ax, x0 + 0.40, 2.62, x1 + 1.71, 2.74, C["att"], rad=-0.22)
    # ---- 3 latent and objective ---------------------------------------------------------------------------------
    x2, w2 = 4.18, 1.46; column_title(ax, x2, w2, "Latent and objective")
    box(ax, x2, 0.08, w2, H - 0.3, C["head"])
    vec(ax, x2 + 0.12, 2.48, 16, 1.22, 0.16, C["z"]); text(ax, x2 + w2 / 2, 2.76, r"latent $\mathbf{z}_i\in\mathbb{R}^{256}$", fs=FS, fontweight="bold", color=C["z"])
    ax.plot([x1 + w1 / 2 + 0.075, x1 + w1 + 0.08, x1 + w1 + 0.08], [0.36, 0.36, 2.56], color=C["z"], lw=1.0, zorder=4, solid_capstyle="round")
    arrow(ax, x1 + w1 + 0.08, 2.56, x2 + 0.12, 2.56, C["z"], lw=1.0)
    box(ax, x2 + 0.10, 1.70, w2 - 0.18, 0.56, "white", ec=C["z"])
    text(ax, x2 + w2 / 2, 2.12, r"decoder $\rightarrow\hat{\mathbf{x}}_i$", fs=FS)
    text(ax, x2 + w2 / 2, 1.93, r"$\mathcal{L}_{\mathrm{rec}}$: MSE on masked spots", fs=FS_S); text(ax, x2 + w2 / 2, 1.79, r"$+\,0.5\times$ visible spots", fs=FS_S)
    box(ax, x2 + 0.10, 0.80, w2 - 0.18, 0.82, "white", ec=C["skip"])
    text(ax, x2 + w2 / 2 + 0.01, 1.50, r"linear heads $\mathbf{A}_1,\mathbf{A}_2,\mathbf{A}_3$", fs=FS)
    text(ax, x2 + w2 / 2 + 0.01, 1.355, r"ring-1 mean $\bar{\mathbf{x}}^{(1)}_i$", fs=FS_S); text(ax, x2 + w2 / 2 + 0.01, 1.195, r"ring-2–3 mean $\bar{\mathbf{x}}^{(2,3)}_i$", fs=FS_S)
    text(ax, x2 + w2 / 2 + 0.01, 1.035, r"spread $\mathbf{v}_i$ (rings 1–3)", fs=FS_S); text(ax, x2 + w2 / 2 + 0.01, 0.88, r"$\mathcal{L}_{\mathrm{nb}}$, weights 0.5, 0.5, 0.25", fs=FS_S)
    arrow(ax, x2 + w2 / 2, 2.48, x2 + w2 / 2, 2.26, C["z"]); ax.plot([x2 + 0.12, x2 + 0.05, x2 + 0.05], [2.50, 2.50, 1.24], color=C["z"], lw=0.8, zorder=4); arrow(ax, x2 + 0.05, 1.24, x2 + 0.10, 1.24, C["z"])
    box(ax, x2 + 0.08, 0.20, w2 - 0.16, 0.50, C["loss"], ec="#c0564f", lw=0.9)
    text(ax, x2 + w2 / 2, 0.58, r"$\mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\mathcal{L}_{\mathrm{nb}}+\mathcal{L}_{\mathrm{route}}$", fs=6.6, fontweight="bold", color="#9b3a33")
    text(ax, x2 + w2 / 2, 0.43, r"$\mathcal{L}_{\mathrm{route}}$: balance + routing entropy", fs=FS_S - 0.4)
    text(ax, x2 + w2 / 2, 0.29, "two patients per step, AdamW", fs=FS_S - 0.2)
    # ---- 4 use of the latent ------------------------------------------------------------------------------------
    x3, w3 = 5.80, 1.36; column_title(ax, x3, w3, "Niche atlas and use")
    box(ax, x3, 0.08, w3, H - 0.3, C["down"])
    arrow(ax, x2 + w2, 2.56, x3, 2.56, C["z"], lw=1.0)
    text(ax, x3 + w3 / 2, 2.62, "PCA-30, Harmony, Leiden K = 8", fs=FS_S); text(ax, x3 + w3 / 2, 2.47, "consensus of five networks", fs=FS_S)
    for k, c in enumerate(NICHE):
        ax.add_patch(Circle((x3 + 0.20 + (k % 4) * 0.32, 2.20 - (k // 4) * 0.22), 0.075, fc=c, ec="white", lw=0.4, zorder=4))
        text(ax, x3 + 0.20 + (k % 4) * 0.32, 2.20 - (k // 4) * 0.22, str(k), fs=5.2, color="white", fontweight="bold")
    text(ax, x3 + w3 / 2, 1.78, "eight niches, 19 patients", fs=FS_S)
    box(ax, x3 + 0.08, 1.06, w3 - 0.16, 0.54, "white", ec=C["z"])
    text(ax, x3 + w3 / 2, 1.44, "new patient", fs=FS, fontweight="bold"); text(ax, x3 + w3 / 2, 1.28, "mapped onto the fixed", fs=FS_S)
    text(ax, x3 + w3 / 2, 1.15, "reference, nearest niche", fs=FS_S)
    box(ax, x3 + 0.08, 0.20, w3 - 0.16, 0.74, "white", ec=C["edge"])
    text(ax, x3 + w3 / 2, 0.80, "applications", fs=FS, fontweight="bold")
    text(ax, x3 + w3 / 2, 0.62, "tumour–lung interface", fs=FS_S); text(ax, x3 + w3 / 2, 0.47, "diagnosis → relapse", fs=FS_S)
    text(ax, x3 + w3 / 2, 0.32, "five tissue sites", fs=FS_S)


HL = "#c0564f"  # colour of the contribution badges


def badge(ax, x, y, n, r=0.075):
    ax.add_patch(Circle((x, y), r, fc=HL, ec="white", lw=0.6, zorder=8)); text(ax, x, y - 0.004, str(n), fs=5.6, color="white", fontweight="bold", zorder=9)


def mark(ax) -> None:
    """numbered badges on the components that carry the contributions listed under the panel"""
    for x, y, n in [(1.50, 1.20, 1), (3.94, 1.96, 2), (5.58, 1.58, 3), (7.10, 1.58, 4), (7.10, 0.92, 5)]:
        badge(ax, x, y, n)


CONTRIB = [("Multimodal input", "expression with foundation-model\nH&E features in every spot"),
           ("Two-branch encoder", "ring means and attention to 36\nneighbours, fused by two experts"),
           ("Readable latent", "beats BANKSY on unseen patients'\nimmune context (18 of 19)"),
           ("Transferable atlas", "97 % of a new patient's spots\nmapped onto the fixed atlas"),
           ("Osteosarcoma biology", "band marks macrophage signal;\nrelapse shifts tumour state")]
SH = 0.62  # height of the contribution strip (inches)


def contributions(ax) -> None:
    ax.set_xlim(0, W); ax.set_ylim(0, SH); ax.set_aspect("equal"); ax.axis("off")
    n = len(CONTRIB); gap = 0.08; w = (W - 0.08 - gap * (n - 1)) / n
    for k, (t, d) in enumerate(CONTRIB):
        x = 0.04 + k * (w + gap); box(ax, x, 0.03, w, SH - 0.06, "#fdf3f1", ec=HL, lw=0.7)
        badge(ax, x + 0.13, SH - 0.17, k + 1); text(ax, x + 0.25, SH - 0.17, t, fs=FS, fontweight="bold", ha="left")
        text(ax, x + w / 2, 0.20, d, fs=FS_S - 0.2, linespacing=1.25)
