"""Figure layout checks used by lib.plotting.save_fig.

align_panel_labels  places every registered panel letter after drawing: panels whose axes overlap vertically form a row and
                    their letters share one baseline just above the highest drawn content of the row; letters of panels that
                    overlap horizontally, or whose drawn left edges are close, share the leftmost left edge
overlap_report      text boxes that collide with other text, or sit on plotted points, line vertices, bars or error bars, and
                    legends that cover data; heat-map cell values and panel letters are exempt
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np


def _overlap_frac(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def align_panel_labels(fig, pad_pt: float = 4.0, col_tol: float = 0.07, fontsize: float = 11) -> None:
    items = getattr(fig, "_panel_labels", [])
    if not items:
        return
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); rec = []
    for ax, letter in items:
        tb = inv.transform_bbox(ax.get_tightbbox(r)); pos = ax.get_position()
        rec.append({"letter": letter, "x0": tb.x0, "top": tb.y1, "y0": pos.y0, "y1": pos.y1, "px0": pos.x0, "px1": pos.x1})
    rows: list[list[dict]] = []
    for it in sorted(rec, key=lambda d: -d["y1"]):
        for row in rows:
            if any(_overlap_frac(it["y0"], it["y1"], o["y0"], o["y1"]) > 0.5 * min(it["y1"] - it["y0"], o["y1"] - o["y0"]) for o in row):
                row.append(it); break
        else:
            rows.append([it])
    pad = pad_pt / 72 / fig.get_figheight()
    for row in rows:
        y = max(o["top"] for o in row) + pad
        for o in row:
            o["y"] = y
    parent = list(range(len(rec)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            a, b = rec[i], rec[j]; wide = max(a["px1"] - a["px0"], b["px1"] - b["px0"])
            if _overlap_frac(a["px0"], a["px1"], b["px0"], b["px1"]) > 0.75 * wide or abs(a["x0"] - b["x0"]) < col_tol:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for k, o in enumerate(rec):
        groups.setdefault(find(k), []).append(o)
    for col in groups.values():
        x = min(o["x0"] for o in col)
        for o in col:
            o["x"] = x
    for o in rec:
        t = fig.text(o["x"], o["y"], o["letter"], fontsize=fontsize, fontweight="bold", ha="left", va="bottom"); t._panel_letter = True
    fig._panel_labels = []


def _drawn_texts(fig, r):
    all_ticks, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {id(tk.label1), id(tk.label2)}
            if ax.axison and axis.get_visible():  # an axis switched off draws no tick labels
                for tk in axis._update_ticks():
                    drawn |= {id(tk.label1), id(tk.label2)}
    out = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip() or getattr(t, "_overlap_exempt", False):  # panel letters are checked against text
            continue
        if id(t) in all_ticks and id(t) not in drawn:
            continue
        out.append((t, mpl.text.Text.get_window_extent(t, r), id(t) in drawn))
    return out


def overlap_report(fig, pad: float = 1.0) -> list[str]:
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig.canvas.draw(); r = fig.canvas.get_renderer(); boxes = _drawn_texts(fig, r); issues = []
    for i in range(len(boxes)):
        ti, bi, _ = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj, _ = boxes[j]
            if ti.get_rotation() or tj.get_rotation() or bi.width == 0 or bj.width == 0:
                continue
            if bi.overlaps(bj) and not (ti.get_text() == tj.get_text() and np.allclose(bi.bounds, bj.bounds, atol=0.5)):  # twin axes repeat ticks
                issues.append(f"text '{ti.get_text()[:30]}' overlaps text '{tj.get_text()[:30]}'")
    # data marks per axes: scatter points, line vertices, error-bar segments, bars
    in_legend = {id(a) for lg in fig.findobj(Legend) for a in lg.findobj()}
    for ax in fig.axes:
        pts = []
        for c in ax.collections:
            if not c.get_visible() or id(c) in in_legend:
                continue
            if isinstance(c, PathCollection) and c.get_offsets().size:
                pts.append(c.get_offset_transform().transform(np.asarray(c.get_offsets())))
            elif isinstance(c, LineCollection):
                for seg in c.get_segments():
                    if len(seg):
                        s = c.get_transform().transform(np.asarray(seg)); pts.append(np.linspace(s[0], s[-1], max(2, int(np.hypot(*(s[-1] - s[0])) / 3))))
        for ln in ax.lines:
            if ln.get_visible() and id(ln) not in in_legend and len(ln.get_xydata()):
                xy = ln.get_transform().transform(ln.get_xydata())
                if len(xy) > 1:  # sample along the segments so that a label sitting on a line is caught
                    xy = np.vstack([np.linspace(xy[k], xy[k + 1], max(2, int(np.hypot(*(xy[k + 1] - xy[k])) / 3))) for k in range(len(xy) - 1)
                                    if np.isfinite(xy[k]).all() and np.isfinite(xy[k + 1]).all()] or [xy])
                pts.append(xy)
        bars = [p.get_window_extent(r) for p in ax.patches if isinstance(p, Rectangle) and p.get_visible() and p.get_width() and p.get_height() and id(p) not in in_legend]
        P = np.vstack(pts) if pts else np.zeros((0, 2))
        for t, b, is_tick in boxes:
            if is_tick or t.axes is not ax or t is ax.title or t is ax.xaxis.label or t is ax.yaxis.label:
                continue
            if P.size:
                inside = (P[:, 0] > b.x0 + pad) & (P[:, 0] < b.x1 - pad) & (P[:, 1] > b.y0 + pad) & (P[:, 1] < b.y1 - pad)
                if inside.any():
                    issues.append(f"text '{t.get_text()[:30]}' sits on {int(inside.sum())} data mark(s)")
            for bb in bars:
                if bb.overlaps(b) and _overlap_frac(b.x0, b.x1, bb.x0, bb.x1) > pad and _overlap_frac(b.y0, b.y1, bb.y0, bb.y1) > 2 * pad:
                    issues.append(f"text '{t.get_text()[:30]}' sits on a bar"); break
        for lg in [c for c in ax.get_children() if isinstance(c, Legend)]:
            lb = lg.get_window_extent(r)
            if P.size and ((P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) & (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1)).any():
                issues.append("legend covers data")
    return sorted(set(issues))
