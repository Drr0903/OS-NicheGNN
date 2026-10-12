"""Drawing helpers for Figure 1 in millimetre coordinates: boxes, text, arrows, illustrated icons and network blocks.

Icons are the transparent PNGs in figures/source_data/Fig1_icons/ (fig1_icons_generate.py); they are cropped to their
visible extent and placed by height, keeping their aspect ratio. Text placed in a box is registered so that the figure
script can check that every label stays inside its box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

ICON_DIR = Path(__file__).resolve().parents[2] / "figures" / "source_data" / "Fig1_icons"
INK, GREY = "#24313b", "#5f6b75"
PAIRS: list = []
_cache: dict[str, np.ndarray] = {}


def box(ax, x0, y0, x1, y1, fc, ec, lw=0.6, r=1.5, z=2, alpha=1.0, ls="-"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, alpha=alpha, ls=ls)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, patch=None, fs=6.0, fw="normal", col=None, ha="center", va="center", z=8, style="normal"):
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=col or INK, ha=ha, va=va, zorder=z, linespacing=1.15,
                fontstyle=style)
    if patch is not None:
        PAIRS.append((t, patch))
    return t


def arrow(ax, p0, p1, col="#4a4a4a", lw=0.9, ls="-", rad=0.0, ms=6, z=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, linestyle=ls, zorder=z,
                        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a


def _load(name: str) -> np.ndarray:
    if name not in _cache:
        im = Image.open(ICON_DIR / f"{name}.png").convert("RGBA")
        a = np.asarray(im)[:, :, 3]; ys, xs = np.where(a > 8)
        _cache[name] = np.asarray(im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return _cache[name]


def icon(ax, name, cx, cy, h=None, w=None, z=5):
    """Place an icon centred at (cx, cy) with height h (or width w, whichever binds) in millimetres; returns its extent."""
    im = _load(name); asp = im.shape[1] / im.shape[0]
    if h is None or (w is not None and h * asp > w):
        h = w / asp
    w = h * asp
    ext = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.imshow(im, extent=ext, zorder=z, interpolation="lanczos")
    return ext


def image(ax, img, x0, y0, w, z=5):
    """A raster image placed with its aspect ratio kept, width w; returns its height."""
    h = w * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=[x0, x0 + w, y0, y0 + h], zorder=z, interpolation="lanczos")
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec="#c9d1d8", lw=0.5, zorder=z + 1))
    return h


def trapezoid(ax, x0, x1, yc, h0, h1, fc, label=None, fs=5.8, z=4):
    """Encoder (h0 > h1) or decoder (h0 < h1) block drawn as a trapezoid."""
    p = Polygon([(x0, yc - h0 / 2), (x1, yc - h1 / 2), (x1, yc + h1 / 2), (x0, yc + h0 / 2)], closed=True, fc=fc,
                ec=fc, lw=0.7, zorder=z, joinstyle="round")
    ax.add_patch(p)
    if label:
        txt(ax, (x0 + x1) / 2, yc, label, p, fs=fs, col="white", fw="bold")
    return p


def vector(ax, x, y, w, h, cols, n=None, vertical=False, z=5):
    """A feature vector drawn as a strip of cells (colours cycle through cols)."""
    n = n or len(cols)
    for k in range(n):
        c = cols[k % len(cols)]
        if vertical:
            ax.add_patch(Rectangle((x, y + (n - 1 - k) * h / n), w, h / n * 0.88, fc=c, ec="none", zorder=z))
        else:
            ax.add_patch(Rectangle((x + k * w / n, y), w / n * 0.88, h, fc=c, ec="none", zorder=z))


def hexagon(ax, cx, cy, r, z=6, **kw):
    a = np.deg2rad(np.arange(6) * 60 + 30)
    ax.add_patch(Polygon(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)], closed=True, zorder=z, **kw))


def lattice(ax, cx, cy, r, cols, masked=()):
    """Hexagonal Visium neighbourhood: centre, ring 1 and rings 2-3 coloured by cols[0..2]; masked spots grey."""
    dx, dy = np.sqrt(3) * r, 1.5 * r
    for q in range(-3, 4):
        for s in range(-3, 4):
            ring = max(abs(q), abs(s), abs(-q - s))
            if ring <= 3:
                fc = cols[3] if (q, s) in masked else cols[min(ring, 2)]
                hexagon(ax, cx + dx * (q + s / 2), cy + dy * s, r * 0.94, fc=fc, ec="white", lw=0.3)


def sigma(ax, cx, cy, r, col):
    c = Circle((cx, cy), r, fc="white", ec=col, lw=0.8, zorder=7); ax.add_patch(c)
    txt(ax, cx, cy, "Σ", c, fs=6.6, fw="bold", col=col)
    return c
