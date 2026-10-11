"""v2 Figure 1a — study framework, drawn as vector shapes in a flat scientific-illustration style.

The stage list, palette and wording follow `figures/source_data/Fig1_a_spec.json`, a layout specification
produced by reading the manuscript; this module renders that specification deterministically so that the
panel stays editable vector art (PDF) and reproducible from the repository alone.
"""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

PAL = {"ink": "#33444f", "edge": "#93a3ad", "cohort": "#eef2f5", "spatial": "#e8f3f4", "encoder": "#edf0fa",
       "training": "#f8efe6", "proto": "#f1eef7", "atlas": "#eaf3ec", "remod": "#fcefe9",
       "centre": "#1f6f7a", "near": "#74b6bf", "outer": "#cfe6e9", "pa": "#4f7fa8", "pb": "#c1845d",
       "bone": "#a1866f", "lung": "#4f7fa8", "ln": "#8f7fbf", "heart": "#c0564f", "soft": "#57a473"}
NICHE = ["#4f7fa8", "#e08a4a", "#57a473", "#c0564f", "#8f7fbf", "#a1866f", "#dd94c0", "#8a8a8a"]
ASPECT = 7.03 / 2.07  # panel width / height, so circles and hexagons stay round


DATA: dict = {}  # section -> DataFrame(x, y, niche, site); filled by the caller so the glyphs can show real tissue


def _minimap(ax, x0, y0, w, h, df, colour, s=0.45):
    """draw one real section inside the box (x0, y0, w, h), aspect preserved"""
    px = df["x"].to_numpy(); py = -df["y"].to_numpy()
    sx = (px - px.min()) / max(px.ptp(), 1e-9); sy = (py - py.min()) / max(py.ptp(), 1e-9)
    ar = max(px.ptp(), 1e-9) / max(py.ptp(), 1e-9) / ASPECT  # width/height in axes units for a square-pixel section
    hh = min(h, w / ar); ww = hh * ar
    ax.scatter(x0 + (w - ww) / 2 + sx * ww, y0 + (h - hh) / 2 + sy * hh, s=s, c=colour, linewidths=0)


def _hex(ax, cx, cy, r, **kw):
    a = np.deg2rad(np.arange(6) * 60 + 30)
    ax.add_patch(Polygon(np.c_[cx + r * np.cos(a) / ASPECT, cy + r * np.sin(a)], closed=True, **kw))


def _circ(ax, cx, cy, r, **kw):
    ax.add_patch(Polygon(np.c_[cx + r * np.cos(np.linspace(0, 2 * np.pi, 41)) / ASPECT,
                               cy + r * np.sin(np.linspace(0, 2 * np.pi, 41))], closed=True, **kw))


def _panel(ax, x, y, w, h, fill, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=PAL[fill], ec=PAL["edge"], lw=0.6, mutation_aspect=ASPECT))
    ax.text(x + w / 2, y + h - 0.040, "\n".join(textwrap.wrap(title, max(12, int(w * 140)))), ha="center", va="top", fontsize=5.8,
            fontweight="bold", color=PAL["ink"], linespacing=1.15)


def _items(ax, x, w, y0, lines, fs=4.6, dy=0.072):
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y0 - i * dy, t, ha="center", va="top", fontsize=fs, color=PAL["ink"])


def _arrow(ax, x0, y0, x1, y1, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=PAL["edge"], shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, y0 + 0.055, label, ha="center", va="bottom", fontsize=4.4, color=PAL["ink"])


def _g_sections(ax, x, w, yc):
    """five real sections, one per anatomical site, coloured by site"""
    keys = [("bone", "bone 9", "PT3"), ("lung", "lung 12", "MR3"), ("ln", "LN 1", "MR14"), ("heart", "heart 1", "MR16"), ("soft", "soft 3", "LR17")]
    hh = 0.088
    for i, (k, lab, sec) in enumerate(keys):
        y = yc + 0.178 - i * (hh + 0.022)
        ax.add_patch(FancyBboxPatch((x + 0.012, y), w - 0.024, hh, boxstyle="round,pad=0,rounding_size=0.006",
                                    fc="white", ec=PAL["edge"], lw=0.5, mutation_aspect=ASPECT))
        if sec in DATA:
            _minimap(ax, x + 0.050, y + 0.008, w - 0.066, hh - 0.016, DATA[sec], PAL[k], s=0.30)
        ax.text(x + 0.020, y + hh / 2, lab, ha="left", va="center", fontsize=4.2, color=PAL["ink"])


def _g_lattice(ax, x, w, yc):
    """hexagonal spot lattice with the centre, ring 1 and rings 2-3 shaded apart"""
    r = 0.031; cx = x + w / 2; cy = yc + 0.098
    for q in range(-3, 4):
        for s in range(-3, 4):
            if abs(q + s) > 3:
                continue
            d = max(abs(q), abs(s), abs(q + s))
            px = cx + 1.5 * r * q / ASPECT; py = cy + np.sqrt(3) * r * (s + q / 2)
            fc = PAL["centre"] if d == 0 else (PAL["near"] if d == 1 else PAL["outer"])
            _hex(ax, px, py, r * 0.92, fc=fc, ec="white", lw=0.35)
    for lab, col, dy in [("centre spot", PAL["centre"], 0.0), ("ring 1", PAL["near"], 0.058), ("rings 2–3", PAL["outer"], 0.116)]:
        yy = yc - 0.155 - dy
        _hex(ax, x + 0.028, yy, 0.020, fc=col, ec=PAL["edge"], lw=0.35)
        ax.text(x + 0.046, yy, lab, ha="left", va="center", fontsize=4.4, color=PAL["ink"])


def _g_encoder(ax, x, w, yc):
    """spot profile (expression + H&E) → spot MLP; the spot attends to its ring-1–3 neighbour tokens; outputs latent and context"""
    bx = x + 0.012; bw = w * 0.30
    for i, (lab, col, fg) in enumerate([("expression", PAL["centre"], "white"), ("H&E patch", PAL["pb"], "white")]):
        y = yc + 0.205 - i * 0.090
        ax.add_patch(FancyBboxPatch((bx, y), bw, 0.062, boxstyle="round,pad=0,rounding_size=0.006",
                                    fc=col, ec=PAL["edge"], lw=0.5, mutation_aspect=ASPECT))
        ax.text(bx + bw / 2, y + 0.031, lab, ha="center", va="center", fontsize=4.0, color=fg)
    qx = x + w * 0.60; qy = yc + 0.170
    _hex(ax, qx, qy, 0.030, fc=PAL["centre"], ec="white", lw=0.4)
    for i in range(2):
        _arrow(ax, bx + bw + 0.003, yc + 0.236 - i * 0.090, qx - 0.012, qy)
    wts = [1.6, 0.4, 1.0, 0.3, 1.3]
    for k, ang in enumerate([-80, -40, 0, 40, 80]):
        a = np.deg2rad(ang); tx = qx + 0.085 * np.cos(a) / ASPECT * 2.2; ty = qy + 0.085 * np.sin(a)
        tx = min(tx, x + w - 0.018)
        ax.plot([qx, tx], [qy, ty], color=PAL["ink"], lw=wts[k] * 0.6, alpha=0.75, zorder=1)
        _hex(ax, tx, ty, 0.020, fc=PAL["near"] if k % 2 else PAL["outer"], ec="white", lw=0.3)
    ax.text(x + w / 2, yc + 0.048, "attention over\nring 1–3 neighbours", ha="center", va="top", fontsize=4.0, color=PAL["ink"])
    for i in range(8):
        ax.add_patch(Rectangle((x + w / 2 - 0.052 + i * 0.013, yc - 0.075), 0.011, 0.030,
                               fc=PAL["near"], ec=PAL["edge"], lw=0.3))
    ax.text(x + w / 2, yc - 0.10, "256-d latent\n(+ linear neighbourhood skip)", ha="center", va="top", fontsize=4.0, color=PAL["ink"])


def _g_contrastive(ax, x, w, yc):
    """two patients' spots, mutual nearest neighbours pulled together"""
    rng = np.random.default_rng(3)
    ca = (x + 0.055, yc + 0.11); cb = (x + w - 0.055, yc + 0.11)
    for c, col in [(ca, PAL["pa"]), (cb, PAL["pb"])]:
        p = rng.normal(0, 1, (22, 2)) * [0.016, 0.055]; p[:, 1] = np.clip(p[:, 1], -0.09, 0.085)  # keep the cloud clear of the caption
        ax.scatter(c[0] + p[:, 0], c[1] + p[:, 1], s=3.0, c=col, linewidths=0)
    for k in range(3):
        y = yc + 0.155 - k * 0.045
        ax.plot([ca[0] + 0.022, cb[0] - 0.022], [y, y - 0.012], ls=(0, (1.4, 1.2)), lw=0.6, color=PAL["ink"])
    ax.add_patch(FancyArrowPatch((ca[0] + 0.004, yc - 0.050), (cb[0] - 0.004, yc - 0.050), arrowstyle="<|-|>",
                                 mutation_scale=6, lw=0.8, color=PAL["ink"], connectionstyle="arc3,rad=0.55",
                                 shrinkA=0, shrinkB=0, joinstyle="miter"))
    ax.text(x + w / 2, yc + 0.232, "cross-patient mutual\nnearest neighbours", ha="center", va="bottom",
            fontsize=4.4, color=PAL["ink"])
    ax.text(ca[0], yc - 0.105, "patient A", ha="center", va="top", fontsize=4.2, color=PAL["pa"])
    ax.text(cb[0], yc - 0.105, "patient B", ha="center", va="top", fontsize=4.2, color=PAL["pb"])


def _g_targets(ax, x, w, yc):
    """the latent decodes the spot (masked) and its surroundings: ring-1 mean, rings 2-3 mean, dispersion"""
    lx = x + 0.016
    for i in range(6):
        ax.add_patch(Rectangle((lx, yc + 0.215 - i * 0.034), 0.016, 0.028, fc=PAL["near"], ec=PAL["edge"], lw=0.3))
    ax.text(lx + 0.008, yc + 0.255, "latent", ha="center", va="bottom", fontsize=4.0, color=PAL["ink"])
    rows = [("masked spot", PAL["centre"]), ("ring-1 mean", PAL["near"]), ("rings 2–3 mean", PAL["outer"]), ("spread, rings 1–3", PAL["outer"])]
    for i, (lab, col) in enumerate(rows):
        y = yc + 0.215 - i * 0.062
        _arrow(ax, lx + 0.020, yc + 0.130, x + w * 0.42, y + 0.012)
        _hex(ax, x + w * 0.47, y + 0.012, 0.016, fc=col, ec="white", lw=0.3)
        ax.text(x + w * 0.53, y + 0.012, lab, ha="left", va="center", fontsize=4.0, color=PAL["ink"])
    ax.text(x + w / 2, yc - 0.050, "linear read-out heads", ha="center", va="top", fontsize=4.2, color=PAL["ink"])


def _g_prototypes(ax, x, w, yc):
    """eight consensus prototypes and one section painted with them"""
    for i in range(8):
        cx = x + 0.026 + (i % 4) * (w - 0.052) / 3; cy = yc + 0.215 - (i // 4) * 0.090
        _circ(ax, cx, cy, 0.026, fc=NICHE[i], ec="white", lw=0.5)
        ax.text(cx, cy, str(i), ha="center", va="center", fontsize=3.8, color="white", fontweight="bold")
    if "MR3" in DATA:
        d = DATA["MR3"]; _minimap(ax, x + 0.014, yc - 0.165, w - 0.028, 0.225, d, [NICHE[int(k) % 8] for k in d["niche"]], s=0.55)


def _g_atlas(ax, x, w, yc):
    for i, sec in enumerate(["PT3", "MR3", "MR14", "MR16", "LR17"]):
        if sec not in DATA:
            continue
        d = DATA[sec]; _minimap(ax, x + 0.016 + i * (w - 0.032) / 5, yc - 0.010, (w - 0.032) / 5, 0.105,
                                d, [NICHE[int(k) % 8] for k in d["niche"]], s=0.22)
    ax.text(x + w / 2, yc - 0.022, "five anatomical sites", ha="center", va="top", fontsize=4.4, color=PAL["ink"])


def _g_paired(ax, x, w, yc):
    for i, (sec, lab) in enumerate([("PT3", "primary"), ("MR3", "metastasis")]):
        bx = x + 0.014 + i * (w - 0.028) * 0.60
        if sec in DATA:
            d = DATA[sec]; _minimap(ax, bx, yc + 0.010, (w - 0.028) * 0.40, 0.105, d,
                                    [NICHE[int(k) % 8] for k in d["niche"]], s=0.28)
        ax.text(bx + (w - 0.028) * 0.20, yc - 0.002, lab, ha="center", va="top", fontsize=4.2, color=PAL["ink"])
    _arrow(ax, x + 0.014 + (w - 0.028) * 0.42, yc + 0.062, x + 0.014 + (w - 0.028) * 0.58, yc + 0.062)


GLYPH = {"tissue_sections": _g_sections, "hex_lattice": _g_lattice, "dual_ring": _g_lattice, "encoder": _g_encoder,
         "contrastive": _g_contrastive, "niche_targets": _g_targets, "prototypes": _g_prototypes, "atlas_map": _g_atlas, "paired_arrows": _g_paired}


def draw(ax, spec: dict, data: dict | None = None) -> None:
    DATA.clear(); DATA.update(data or {})
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for st in spec["stages"]:
        _panel(ax, st["x"], st["y"], st["w"], st["h"], st["fill"], st["title"])
        b = st.get("badge")
        if b:
            _circ(ax, st["x"] + 0.016, st["y"] + st["h"] + 0.036, 0.030, fc="white", ec=PAL["edge"], lw=0.6)
            ax.text(st["x"] + 0.016, st["y"] + st["h"] + 0.036, str(b), ha="center", va="center", fontsize=4.6,
                    fontweight="bold", color=PAL["ink"])
        g = GLYPH.get(st.get("glyph", "none"))
        if g:
            g(ax, st["x"], st["w"], st["y"] + st["h"] * st.get("glyph_frac", 0.48))
        _items(ax, st["x"], st["w"], st["y"] + st.get("items_y", 0.22), st["items"], dy=st.get("items_dy", 0.072))
    pos = {s["id"]: s for s in spec["stages"]}
    for a in spec["arrows"]:
        s0, s1 = pos[a["from"]], pos[a["to"]]
        y0 = s0["y"] + s0["h"] / 2; y1 = s1["y"] + s1["h"] / 2
        _arrow(ax, s0["x"] + s0["w"] + 0.004, y0, s1["x"] - 0.004, y1, a.get("label", ""))
