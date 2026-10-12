"""v2 Figure 1 panels a, b and d, drawn in millimetre coordinates with illustrated icons (fig1_draw.py).

a  the cohort: diagnostic bone biopsies and relapsed lesions at four sites, the matched pairs and patient-level hold-out,
   and the independent annotated datasets
b  the network: spot input and its hexagonal neighbourhood, ring-mean and graph attention branches, router and two MLP
   experts, latent with the linear skip, decoder and linear heads, and the training objective
d  from the atlas to biology: niche and router-weight maps of section MR2 (stage 35) and the three biological readouts
Panel c (region reference labels of MR11B) is drawn by v2_Fig1_cohort_framework.py.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

import fig1_draw as D
from fig1_draw import GREY, INK, arrow, box, icon, txt

C = {"ring": "#2a7f8a", "att": "#5b6bb5", "host": "#3b6fb6", "tumour": "#c8553d", "z": "#1f5f70", "router": "#7d4a8c",
     "expr": "#2a7f8a", "r0": "#1f5f70", "r1": "#7fb9c2", "r2": "#d3e8eb", "mask": "#a7b0b7", "skip": "#6b5fa8",
     "fa": "#edf2f8", "fa2": "#f8efe6", "fb": "#f1f2f6", "fd": "#f6f3ea", "hold": "#e39a6b", "train": "#7b97c4"}
FS, FT = 6.0, 7.0


def frame(ax, x0, y0, x1, y1, fc, title):
    p = box(ax, x0, y0, x1, y1, fc, "none", lw=0, r=2.0, z=1)
    txt(ax, x0 + 3, y1 - 3.8, title, p, fs=FT, fw="bold", ha="left")
    return p


def panel_a(ax, y0, y1):
    p = frame(ax, 6, y0, 122, y1, C["fa"], "Osteosarcoma spatial atlas: 19 patients, 26 Visium sections")
    yi, h = y1 - 19.5, 12.5
    txt(ax, 17, y1 - 10.0, "diagnosis", p, fs=FS, col=GREY, style="italic")
    ax.plot([50, 120], [y1 - 9.3] * 2, color=GREY, lw=0.6, zorder=4)
    for x in (50, 120):
        ax.plot([x, x], [y1 - 9.3, y1 - 10.6], color=GREY, lw=0.6, zorder=4)
    txt(ax, 85, y1 - 11.2, "relapse after chemotherapy", p, fs=FS, col=GREY, style="italic")
    sites = [("femur", 17, "Primary bone", "9 sections"), ("lungs", 58, "Lung", "12 sections"), ("lymph_node", 77.5, "Lymph node", "1 section"),
             ("heart", 95, "Heart", "1 section"), ("soft_tissue", 112, "Soft tissue", "3 sections")]
    for name, x, lab, n in sites:
        icon(ax, name, x, yi, h=h, w=14); txt(ax, x, yi - 9.0, lab, p, fs=FS, fw="bold"); txt(ax, x, yi - 12.0, n, p, fs=FS - 0.3, col=GREY)
    icon(ax, "chemo", 37, yi + 0.5, h=9); arrow(ax, (26, yi - 5.6), (48, yi - 5.6), col=INK, lw=1.0)
    yb = y0 + 5.0
    icon(ax, "visium", 14.5, yb + 0.8, h=7, w=12)
    txt(ax, 22.5, yb + 0.8, "Visium FFPE + H&E;\n4 matched diagnosis–relapse pairs", p, fs=FS - 0.3, ha="left")
    for k in range(19):
        ax.add_patch(Rectangle((75 + k * 2.2, yb - 1.6), 1.8, 3.2, fc=C["hold"] if k == 6 else C["train"], ec="none", zorder=5))
    txt(ax, 75, yb + 3.4, "one patient held out at a time", p, fs=FS - 0.3, ha="left", col=GREY)
    q = frame(ax, 126, y0, 178, y1, C["fa2"], "Independent annotated datasets")
    icon(ax, "slides_stack", 134.5, (y0 + y1) / 2 - 3.5, h=13, w=12)
    rows = [("Colorectal cancer", "7 patients, pathologist"), ("Breast cancer", "6 patients, pathologist"),
            ("Prefrontal cortex", "3 donors, cortical layers"), ("Osteosarcoma OS1–OS9", "network applied unchanged")]
    for k, (a, b) in enumerate(rows):
        yy = y1 - 11.5 - k * 8.4; txt(ax, 143, yy, a, q, fs=FS, fw="bold", ha="left"); txt(ax, 143, yy - 3.1, b, q, fs=FS - 0.4, col=GREY, ha="left")


def _inputs(ax, p, yt, ya):
    icon(ax, "he_patch", 15, yt + 1.5, h=10); txt(ax, 15, yt - 6.2, "H&E patch\n(Phikon, 50 PCs)", p, fs=FS - 0.6)
    D.vector(ax, 25, yt - 3.5, 3.0, 10, [C["expr"], "#5fa3ad", "#a9d0d5"], n=10, vertical=True)
    txt(ax, 26.5, yt + 9.0, "expression\n(50 PCs)", p, fs=FS - 0.6)
    D.lattice(ax, 19.5, ya + 1.5, 1.4, [C["r0"], C["r1"], C["r2"], C["mask"]], masked=((2, -1), (-1, 2), (1, 1)))
    txt(ax, 17.5, ya - 12.0, "spot i, rings 1–3;\n40 % masked", p, fs=FS - 0.6)


def _branches(ax, p, yt, ya):
    txt(ax, 50.5, yt + 10.4, "ring-mean branch", p, fs=FS, fw="bold", col=C["ring"])
    D.trapezoid(ax, 42, 59, yt, 15, 8, C["ring"], "residual\nMLP", fs=FS - 0.4)
    for k, (c, lab) in enumerate([(C["r0"], r"$\mathbf{x}_i$"), (C["r1"], r"$\bar{\mathbf{x}}^{(1)}$"), (C["r2"], r"$\bar{\mathbf{x}}^{(2,3)}$")]):
        D.hexagon(ax, 33.8, yt + 4.5 - k * 4.5, 1.15, fc=c, ec=INK if k == 2 else "white", lw=0.3); txt(ax, 35.6, yt + 4.5 - k * 4.5, lab, None, fs=FS - 0.8, ha="left")
    txt(ax, 50.5, ya - 10.8, "graph attention branch", p, fs=FS, fw="bold", col=C["att"])
    D.trapezoid(ax, 42, 59, ya, 15, 8, C["att"], "cross-\nattention", fs=FS - 0.4)
    arrow(ax, (29.6, ya + 1.5), (41.6, ya + 1.5), lw=0.7); txt(ax, 35.8, ya + 4.4, "≤ 36 tokens", None, fs=FS - 1.2, col=GREY)
    for yc, col, lab, dy in ((yt, C["ring"], r"$\mathbf{h}_i$", 6.4), (ya, C["att"], r"$\mathbf{c}_i$", -6.4)):
        D.vector(ax, 61.5, yc - 4, 2.4, 8, [col], n=6, vertical=True); txt(ax, 62.7, yc + dy, lab, p, fs=FS + 0.4)
        arrow(ax, (59.2, yc), (61.2, yc), lw=0.7)
    arrow(ax, (62.7, yt - 4.6), (55, ya + 4.6), col=C["ring"], lw=0.7, ls=(0, (2.5, 1.5)), rad=-0.3)
    txt(ax, 63.2, (yt + ya) / 2 + 0.5, "query", None, fs=FS - 0.8, col=C["ring"], style="italic", ha="left")


def _fusion(ax, p, yt, ya, ys):
    ym = (yt + ya) / 2
    arrow(ax, (64.3, yt), (72.6, ym + 2), lw=0.7); arrow(ax, (64.3, ya), (72.6, ym - 2), lw=0.7)
    D.vector(ax, 73, ym - 6, 2.4, 12, [C["ring"]] * 3 + [C["att"]] * 3, n=6, vertical=True)
    txt(ax, 74.2, ym + 8.4, r"$[\mathbf{h}_i;\mathbf{c}_i]$", p, fs=FS)
    rb = box(ax, 81, yt - 5.5, 96, yt + 7.5, "white", C["router"], lw=0.8, r=1.2, z=4); txt(ax, 88.5, yt + 4.7, "router", rb, fs=FS, fw="bold", col=C["router"])
    for k, (v, c) in enumerate([(0.3, C["host"]), (0.85, C["tumour"])]):
        ax.add_patch(Rectangle((84.3 + k * 5.0, yt - 3.8), 3.4, 6.6 * v, fc=c, ec="none", zorder=6))
    ax.plot([83.3, 93.7], [yt - 3.9] * 2, color=INK, lw=0.5, zorder=6)
    txt(ax, 88.5, yt - 8.0, r"weights $\pi_i$", p, fs=FS - 0.5)
    arrow(ax, (75.8, ym + 4), (80.7, yt - 1.5), lw=0.7)
    box(ax, 100.5, ym - 10.5, 115.5, ym + 9.0, "white", INK, lw=0.7, r=1.2, z=4); txt(ax, 108, ym + 11.6, "MLP experts", p, fs=FS, fw="bold")
    for k, (c, lab) in enumerate([(C["host"], "expert 1"), (C["tumour"], "expert 2")]):
        e = box(ax, 102, ym + 1.0 - k * 10.0, 114, ym + 7.5 - k * 10.0, c, c, r=1.0, z=5); txt(ax, 108, ym + 4.25 - k * 10.0, lab, e, fs=FS - 0.3, col="white", fw="bold")
    arrow(ax, (75.8, ym - 2), (100.2, ym - 2), lw=0.7)
    arrow(ax, (96.3, yt - 2.0), (100.2, ym + 6.0), col=C["router"], lw=0.7, rad=-0.25)
    D.sigma(ax, 120.5, ym, 2.6, C["z"]); arrow(ax, (115.8, ym), (117.8, ym), lw=0.7)
    txt(ax, 120.5, ym + 5.4, r"$\sum_k \pi_{ik}\,f_k$", None, fs=FS - 0.6, col=GREY)
    D.vector(ax, 125.8, ym - 9, 2.8, 18, [C["z"], "#3f8597", "#7fb0bd"], n=14, vertical=True); arrow(ax, (123.2, ym), (125.5, ym), lw=0.7)
    txt(ax, 127.2, ym + 12.2, r"$\mathbf{z}_i\in\mathbb{R}^{256}$", p, fs=FS, fw="bold", col=C["z"])
    ax.plot([26.5, 26.5, 120.5], [yt - 4.0, ys, ys], color=C["skip"], lw=0.7, ls=(0, (3, 1.5)), zorder=3)
    arrow(ax, (120.5, ys), (120.5, ym - 2.7), col=C["skip"], lw=0.7, ls=(0, (3, 1.5)))
    txt(ax, 88, ys + 2.0, r"linear skip $\mathbf{W}_s[\mathbf{x}_i;\bar{\mathbf{x}}^{(1)}_i;\bar{\mathbf{x}}^{(2,3)}_i]$", p, fs=FS - 0.6, col=C["skip"])
    D.trapezoid(ax, 132, 141.5, ym, 8, 16, "#4b5a68"); txt(ax, 136.75, ym, "decoder\n+ heads", None, fs=FS - 1.0, col="white", fw="bold")
    arrow(ax, (129, ym), (131.8, ym), lw=0.7); arrow(ax, (141.8, ym), (144.7, ym), lw=0.7)


def _cards(ax, p, y0, y1):
    """the equations of each module, under the module they describe (as in Methods)"""
    cards = [(9, 63, C["ring"], [r"$\bar{\mathbf{x}}^{(r)}_i=\frac{1}{|\mathcal{N}_r(i)|}\sum_{j\in\mathcal{N}_r(i)}\mathbf{x}_j$",
                                   r"$\mathbf{h}_i=\mathrm{MLP}([\mathbf{x}_i;\bar{\mathbf{x}}^{(1)}_i;\bar{\mathbf{x}}^{(2,3)}_i])$",
                                   r"$\mathbf{t}_{ij}=\mathbf{W}_t\mathbf{x}_j+\phi(\delta_{ij})$"]),
             (65.5, 100.5, C["att"], [r"$\alpha_{ij}=\mathrm{softmax}_j\left(\frac{\mathbf{q}_i^{\top}\mathbf{k}_{ij}}{\sqrt{d/H}}\right)$",
                                      r"$\mathbf{q}_i^{(0)}=\mathbf{h}_i,\ \ \mathbf{c}_i=\sum_j\alpha_{ij}\mathbf{v}_{ij}$",
                                      r"2 layers, $H=4$ heads"]),
             (103, 142.5, C["router"], [r"$\pi_i=\mathrm{softmax}(\mathbf{W}_r[\mathbf{h}_i;\mathbf{c}_i])$",
                                        r"$\mathbf{z}_i=\sum_{k=1}^{2}\pi_{ik}\,g_k([\mathbf{h}_i;\mathbf{c}_i])$",
                                        r"$\qquad+\,\mathbf{W}_s[\mathbf{x}_i;\bar{\mathbf{x}}^{(1)}_i;\bar{\mathbf{x}}^{(2,3)}_i]$"])]
    for x0, x1, col, lines in cards:
        cb = box(ax, x0, y0, x1, y1, "white", col, lw=0.6, r=1.2, z=3)
        for k, ln in enumerate(lines):
            txt(ax, x0 + 1.6, y1 - 4.0 - k * 6.2, ln, cb, fs=FS - 0.5, ha="left", col=INK if "$" in ln else GREY)


def panel_b(ax, y0, y1):
    p = frame(ax, 6, y0, 178, y1, C["fb"], "OS-NicheGNN: two neighbourhood branches fused by a mixture of experts")
    yt, ya = y1 - 20, y1 - 41  # centre lines of the ring-mean and graph attention branches
    _inputs(ax, p, yt, ya); _branches(ax, p, yt, ya); _fusion(ax, p, yt, ya, y0 + 24.5); _cards(ax, p, y0 + 3, y0 + 21.5)
    ob = box(ax, 145.5, y0 + 3, 176.5, y1 - 8, "white", "#3f4f5c", lw=0.7, r=1.5, z=3)
    txt(ax, 161, y1 - 12.0, "Training objective", ob, fs=FS + 0.4, fw="bold")
    rows = [(r"$\mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\mathcal{L}_{\mathrm{nb}}+\mathcal{L}_{\mathrm{route}}$", None, INK),
            (r"$\mathcal{L}_{\mathrm{rec}}=\Vert\hat{\mathbf{x}}_i-\mathbf{x}_i\Vert_2^2$", "masked and visible spots", INK),
            (r"$\mathcal{L}_{\mathrm{nb}}=\sum_{r}\Vert\mathbf{A}_r\mathbf{z}_i-\bar{\mathbf{x}}^{(r)}_i\Vert_2^2$", "+ spread of the neighbourhood", C["ring"]),
            (r"$\mathcal{L}_{\mathrm{route}}=\lambda_{\mathrm{bal}}(K\sum_k\bar{\pi}_k^{2}-1)$", None, C["router"]),
            (r"$\qquad\quad+\,\lambda_{\mathrm{ent}}\,H(\pi_i)$", "both experts used,\neach spot committed", C["router"])]
    yy = y1 - 21.0
    for a, b, c in rows:
        txt(ax, 147.3, yy, a, ob, fs=FS - 0.2, col=c, ha="left"); yy -= 6.6
        if b:
            txt(ax, 147.3, yy + 2.6, b, ob, fs=FS - 1.0, ha="left", col=GREY, va="top"); yy -= 4.6 + 2.6 * b.count("\n")


def panel_d(ax, x0, y0, x1, y1, img, d, niche_cols):
    p = frame(ax, x0, y0, x1, y1, C["fd"], "From the niche atlas to biology")
    tw = 21.0; th = tw * img.shape[0] / img.shape[1]; yt = y1 - 10.5 - th
    for k, (col, lab) in enumerate([("niche_c", "8 consensus niches"), ("host_expert_weight", "router weights")]):
        xx = x0 + 4 + k * (tw + 3)
        ax.add_patch(Rectangle((xx, yt), tw, th, fc="white", ec="#c9d1d8", lw=0.5, zorder=4)); ax.imshow(img, extent=[xx, xx + tw, yt, yt + th], alpha=0.2, zorder=5)
        px = xx + d["x"] / img.shape[1] * tw; py = yt + th - d["y"] / img.shape[0] * th
        c = d[col].map(niche_cols) if col == "niche_c" else d[col]
        ax.scatter(px, py, s=0.3, c=c, cmap="coolwarm_r", vmin=0, vmax=1, linewidths=0, zorder=6)
        txt(ax, xx + tw / 2, yt + th + 2.0, lab, p, fs=FS - 0.4)
    gx = x0 + 4 + tw + 3; ax.imshow(np.linspace(0, 1, 64)[None, :], extent=[gx, gx + tw, yt - 2.6, yt - 1.6], cmap="coolwarm_r", aspect="auto", zorder=6)
    txt(ax, gx, yt - 4.4, "tumour", p, fs=FS - 1.2, ha="left", col=GREY); txt(ax, gx + tw, yt - 4.4, "host tissue", p, fs=FS - 1.2, ha="right", col=GREY)
    txt(ax, x0 + 4 + tw / 2, yt - 3.0, "section MR2 (lung)", p, fs=FS - 1.2, col=GREY)
    mx = x0 + 2 * tw + 10; mb = box(ax, mx, yt - 3.0, x1 - 3, yt + th + 4.0, "white", C["z"], lw=0.7, r=1.2, z=3)
    txt(ax, (mx + x1 - 3) / 2, yt + th, "new patient", mb, fs=FS, fw="bold")
    txt(ax, (mx + x1 - 3) / 2, yt + th / 2 + 2.6, r"$\kappa(i)=\arg\min_k\Vert\mathbf{y}_i-\mu_k\Vert$", mb, fs=FS - 0.8)
    txt(ax, (mx + x1 - 3) / 2, yt + th / 2 - 4.0, "nearest atlas niche;\n97 % of spots\nassigned", mb, fs=FS - 0.9, col=GREY)
    cards = [(("lungs", "femur"), "Niches shared\nacross five sites"), (("macrophage", "t_cell"), "Peritumoural band:\nmacrophage signal"),
             (("tumour_cell", "osteoclast"), "Relapse: tumour\nniches remodelled")]
    cw = (x1 - x0 - 6 - 2 * 2.5) / 3; ctop = yt - 7.5
    for k, (icons, lab) in enumerate(cards):
        cx0 = x0 + 3 + k * (cw + 2.5); cb = box(ax, cx0, y0 + 3, cx0 + cw, ctop, "white", "#c8b88f", lw=0.7, r=1.5, z=3)
        iy = (ctop + y0 + 12) / 2
        icon(ax, icons[0], cx0 + cw * 0.28, iy, h=10, w=cw * 0.4); icon(ax, icons[1], cx0 + cw * 0.74, iy, h=10, w=cw * 0.4)
        if k == 2:
            arrow(ax, (cx0 + cw * 0.47, iy), (cx0 + cw * 0.55, iy), lw=0.7)
        txt(ax, cx0 + cw / 2, y0 + 8.0, lab, cb, fs=FS - 0.6)


def draw(ax, img, d, niche_cols, W, H):
    """Panels a, b and d on a millimetre canvas of W x H; returns the y-ranges used, for panel c and the letters."""
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    a, b, cd = (H - 46, H - 2), (H - 130, H - 50), (2, H - 134)
    panel_a(ax, *a); panel_b(ax, *b); panel_d(ax, 94, cd[0], 178, cd[1], img, d, niche_cols)
    return {"a": a, "b": b, "cd": cd}
