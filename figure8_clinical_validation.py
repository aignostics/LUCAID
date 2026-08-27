#!/usr/bin/env python3
"""Figure 8 — prospective clinical validation against the expert-panel reference.

Reproduces the quantitative panels of Figure 8 from the tidy per-task CSVs
(LUCAID + five pathologists P1-P5, each vs the adjudicated consensus reference),
in the paper's panel geometry and style:

    8a  rater-vs-reference calibration scatter per task, coloured by clinical-action
        category distance, with a per-rater Spearman rho header and the
        representative case highlighted                          (one per task)
    8c  mean absolute error vs the reference, ranked low->high   (one per task)
    8d  per-task clinical-action concordance track: concordant / off-by-1 /
        off-by->=2                                               (one per task)
    8e  case-level concordance tile grid for LUCAID across the prospective
        cohort (cases x five tasks), ordered by overall concordance

Clinical-action categories use the paper thresholds: cellularity <10 / >=10;
PD-L1 <1 / 1-49 / >=50; H-scores <100 / 100-199 / >=200.

Usage:
    python figure8_clinical_validation.py --output-dir figures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import ImageFont
from scipy.stats import spearmanr

import style as ps
import common as C

ps.apply_house_style()

# Representative case highlighted in every calibration panel (the LUAD case shown
# in Fig 8b); '' disables the highlight.
HIGHLIGHT_DEFAULT = "case_030"

# (display marker, tidy task key, model column name). Cellularity uses the
# cell-count model; the IHC markers use the single model column.
MARKERS = [
    ("Cellularity", "cellularity", "MODEL (count)"),
    ("PD-L1", "pdl1", "MODEL"),
    ("c-MET", "cmet", "MODEL"),
    ("TROP-2 mem.", "trop2_membrane", "MODEL"),
    ("TROP-2 cyt.", "trop2_cytoplasm", "MODEL"),
]
CUTS = {"Cellularity": [10], "PD-L1": [1, 50], "c-MET": [100, 200],
        "TROP-2 mem.": [100, 200], "TROP-2 cyt.": [100, 200]}
MAXV = {"Cellularity": 100, "PD-L1": 100, "c-MET": 300,
        "TROP-2 mem.": 300, "TROP-2 cyt.": 300}
PATHS = ["P1", "P2", "P3", "P4", "P5"]

# Panel geometry / marker + highlight sizes (paper panel proportions).
FIGSIZE = (6.66, 7.6)
AX_RECT = [0.17, 0.11, 0.79, 0.61]
S_D, S_O = 70, 95                       # pathologist diamond / LUCAID circle
HL_S_D, HL_S_O = 150, 240               # highlighted diamond / circle
HL_RING, HL_HALOF = 2.2, 2.0            # black ring width / white-casing factor


class Panel:
    """Thin adapter exposing the columns the plotting code expects."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_pathologist_cols(self):
        return [c for c in self.df.columns if c.startswith("Pathologist")]


def load_panel(task_key: str, model_name: str) -> Panel:
    """Load a tidy task CSV as a Panel: consensus + model + Pathologist 1..5."""
    df = C.load_task(task_key)
    ren = {"reference": "consensus", "LUCAID": model_name}
    ren.update({f"P{i}": f"Pathologist {i}" for i in range(1, 6)})
    return Panel(df.rename(columns=ren))


def category(v, cuts):
    return sum(1 for c in cuts if v >= c)


def dist_color(d):
    """Calibration colour by category distance to the reference: concordant (blue) /
    off-by-1 (orange) / off-by->=2 (dark brown). Same ramp for LUCAID and pathologists
    (pathologist diamonds are drawn translucent, so their orange reads lighter)."""
    return ps.CONCORDANT if d == 0 else (ps.ACCENT if d == 1 else ps.OFF_BY_MANY)


def rater_label(col):
    return ps.MODEL_LABEL if "MODEL" in col else col.replace("Pathologist ", "P")


def san(mk):
    return mk.replace(" ", "_").replace(".", "").replace("-", "")


# --- portable font resolution for the PIL-measured rho header -----------------
def _ttf(bold):
    for fam in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        try:
            path = fm.findfont(fm.FontProperties(family=fam,
                                                 weight="bold" if bold else "normal"),
                               fallback_to_default=False)
            if path:
                return path
        except Exception:
            continue
    return fm.findfont(fm.FontProperties(family="DejaVu Sans",
                                         weight="bold" if bold else "normal"))


_FR, _FB = _ttf(False), _ttf(True)


# =============================================================================
# 8a — calibration panels
# =============================================================================
def _scatter_background(ax, bio, patho_cols, model_col, cuts):
    """Pathologist diamonds then model circles, de-duplicated by (consensus, value)."""
    seen = set()
    for pcol in patho_cols:
        d = bio.df[[pcol, "consensus"]].dropna()
        for cv, rv in zip(d["consensus"], d[pcol]):
            dist = abs(category(rv, cuts) - category(cv, cuts))
            key = (cv, rv, dist)
            if key in seen:
                continue
            seen.add(key)
            ax.scatter(cv, rv, marker=ps.PATHOLOGIST_MARKER, s=S_D,
                       c=dist_color(dist), alpha=0.42, linewidth=0, zorder=2)
    seen = set()
    d = bio.df[[model_col, "consensus"]].dropna()
    for cv, mv in zip(d["consensus"], d[model_col]):
        dist = abs(category(mv, cuts) - category(cv, cuts))
        key = (cv, round(mv, 3), dist)
        if key in seen:
            continue
        seen.add(key)
        ax.scatter(cv, mv, marker=ps.MODEL_MARKER, s=S_O,
                   c=dist_color(dist), alpha=0.95, linewidth=0, zorder=3)


def _highlight_case(ax, bio, patho_cols, model_col, cuts, mx, case_id):
    """Representative case: white-cased spine, black-ringed haloed markers, a
    triangular pointer below the x-axis."""
    if case_id not in bio.df.index:
        return
    row = bio.df.loc[case_id]
    cv = row["consensus"]
    if pd.isna(cv):
        return
    pts = [(row[c], "MODEL" in c) for c in patho_cols + [model_col] if pd.notna(row[c])]
    if not pts:
        return
    ys = [y for y, _ in pts]
    ax.plot([cv, cv], [min(ys), max(ys)], color="white", lw=6.0,
            solid_capstyle="round", zorder=4)
    ax.plot([cv, cv], [min(ys), max(ys)], color="black", lw=1.4, alpha=0.6, zorder=4.1)
    for y, is_model in pts:
        mrk, size, z = (ps.MODEL_MARKER, HL_S_O, 6) if is_model \
            else (ps.PATHOLOGIST_MARKER, HL_S_D, 5)
        ax.scatter(cv, y, marker=mrk, s=size * HL_HALOF, c="white", lw=0, zorder=z - 0.5)
        dist = abs(category(y, cuts) - category(cv, cuts))
        ax.scatter(cv, y, marker=mrk, s=size, c=dist_color(dist), alpha=1.0,
                   edgecolors="black", linewidths=HL_RING, zorder=z)
    ax.plot([cv], [-mx * 0.06], marker="^", markersize=15, color="black",
            clip_on=False, zorder=7)


def _line_entries(ax, y, entries, bold_idx, width_pts, fs, sep=" · "):
    """Draw entries + separators as one centred line (bold winner); PIL-measured."""
    fr = ImageFont.truetype(_FR, int(round(fs)))
    fb = ImageFont.truetype(_FB, int(round(fs)))
    pieces = []
    for i, e in enumerate(entries):
        pieces.append((e, i == bold_idx))
        if i < len(entries) - 1:
            pieces.append((sep, False))
    total = sum((fb if b else fr).getlength(t) for t, b in pieces)
    x = 0.5 - (total / width_pts) / 2
    for t, b in pieces:
        ax.text(x, y, t, transform=ax.transAxes, ha="left", va="center",
                fontsize=fs, fontweight="bold" if b else "normal")
        x += (fb if b else fr).getlength(t) / width_pts


def _rho_header(ax, bio, patho_cols, model_col):
    """Spearman rho of all six raters over three centred lines, best rho bold,
    font auto-shrunk until the widest line fits."""
    entries = []
    for col in patho_cols + [model_col]:
        d = bio.df[[col, "consensus"]].dropna()
        rho, p = spearmanr(d[col], d["consensus"])
        entries.append((rater_label(col), rho, p))
    entries.sort(key=lambda e: e[1], reverse=True)
    e = [f"{lbl} ρ={rho:.2f}" for lbl, rho, _ in entries]
    lines = [e[0:2], e[2:4], e[4:6]]
    if all(p < 0.001 for _, _, p in entries):
        lines[-1][-1] += "  (all p<0.001)"
    bold = (0, -1, -1)
    SEP = " · "
    axw = AX_RECT[2] * FIGSIZE[0] * 72

    def width(es, bi, fs):
        fr = ImageFont.truetype(_FR, int(round(fs)))
        fb = ImageFont.truetype(_FB, int(round(fs)))
        return (sum((fb if i == bi else fr).getlength(x) for i, x in enumerate(es))
                + fr.getlength(SEP) * (len(es) - 1))

    fs = 27.0
    while fs > 14 and max(width(l, b, fs) for l, b in zip(lines, bold)) > 1.18 * axw:
        fs -= 0.5
    for ln, yy, bi in zip(lines, (1.35, 1.25, 1.15), bold):
        _line_entries(ax, yy, ln, bi, axw, fs, sep=SEP)


def calibration_figure(marker, bio, model_col, highlight, outdir):
    cuts, mx = CUTS[marker], MAXV[marker]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)
    ax.plot([0, mx], [0, mx], color=ps.IDENTITY_LINE, lw=1.3, zorder=1)
    for c in cuts:
        ax.axvline(c, color=ps.THRESHOLD_LINE, lw=1.0, ls=(0, (3, 3)), zorder=1)
        ax.axhline(c, color=ps.THRESHOLD_LINE, lw=1.0, ls=(0, (3, 3)), zorder=1)

    patho_cols = sorted(bio.get_pathologist_cols())
    _scatter_background(ax, bio, patho_cols, model_col, cuts)
    if highlight:
        _highlight_case(ax, bio, patho_cols, model_col, cuts, mx, highlight)

    ticks = sorted(set(([0] if min(cuts) >= 0.06 * mx else []) + cuts + [mx]))
    ax.set_xlim(-mx * 0.02, mx * 1.02)
    ax.set_ylim(-mx * 0.02, mx * 1.02)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=26, length=5)
    ax.set_xlabel(f"{marker} — Reference standard", fontsize=27, labelpad=6)
    ax.set_ylabel(f"{marker} — Rater value", fontsize=27, labelpad=6)
    ps.style_axes(ax)
    _rho_header(ax, bio, patho_cols, model_col)
    ps.save_figure(fig, Path(outdir) / f"figure8a_calibration_{san(marker)}")


# =============================================================================
# 8c / 8d — MAE bars and per-task agreement tracks
# =============================================================================
def compute(bio, model_col, cuts):
    """Per rater label: MAE vs consensus and the 3-way severity split."""
    cols = {rater_label(model_col): model_col}
    for pc in sorted(bio.get_pathologist_cols()):
        cols[rater_label(pc)] = pc
    mae, sev = {}, {}
    for lbl, col in cols.items():
        d = bio.df[[col, "consensus"]].dropna()
        mae[lbl] = float(np.mean(np.abs(d[col] - d["consensus"])))
        dist = [abs(category(r, cuts) - category(c, cuts))
                for r, c in zip(d[col], d["consensus"])]
        n = len(dist)
        sev[lbl] = (100 * sum(x == 0 for x in dist) / n,
                    100 * sum(x == 1 for x in dist) / n,
                    100 * sum(x >= 2 for x in dist) / n)
    return mae, sev


def mae_bars(marker, mae, outdir):
    order = ["LUCAID"] + sorted(PATHS, key=lambda r: mae[r])   # best model, then asc
    vals = [mae[r] for r in order]
    if MAXV[marker] == 300:
        vmax, yticks = 80, [0, 40, 80]
    else:
        raw = max(vals) / 0.875
        step = 10 if raw > 25 else 5
        vmax = math.ceil(raw / step) * step
        yticks = [0, vmax / 2, vmax]
    best = min(mae, key=mae.get)
    fig = plt.figure(figsize=(7.92, 6.9))
    ax = fig.add_axes([0.135, 0.12, 0.83, 0.78])
    for i, r in enumerate(order):
        ax.bar(i, mae[r], width=0.64,
               color=ps.CONCORDANT if r == "LUCAID" else ps.PATHOLOGIST)
        ax.text(i, mae[r] + vmax * 0.015, f"{mae[r]:.1f}", ha="center", va="bottom",
                fontsize=24, fontweight="bold" if r == best else "normal")
    ax.axhline(mae["LUCAID"], color=ps.CONCORDANT, lw=1.0, ls=(0, (3, 2)), alpha=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=24)
    for t, r in zip(ax.get_xticklabels(), order):
        t.set_fontweight("bold" if r == best else "normal")
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=24, length=5)
    ax.set_ylim(0, vmax)
    ax.set_title(f"{marker} — MAE", fontsize=26, fontweight="bold", pad=8)
    ps.style_axes(ax)
    ps.save_figure(fig, Path(outdir) / f"figure8c_mae_{san(marker)}")


def severity_track(marker, sev, outdir):
    order = ["LUCAID"] + sorted(PATHS, key=lambda r: sev[r][0], reverse=True)
    n = len(order)
    step, h, fs = 1.0, 0.6, 24
    def ypos(ri):
        return (n - 1) * step if ri == 0 else (n - 1 - ri) * step
    fig, ax = plt.subplots(figsize=(6.03, 5.6))
    for ri, r in enumerate(order):
        y = ypos(ri)
        x = 0.0
        for val, col in zip(sev[r], (ps.CONCORDANT, ps.OFF_BY_ONE, ps.OFF_BY_MANY)):
            if val > 0.3:
                ax.barh(y, val, left=x, height=h, color=col)
            x += val
        bold = "bold" if r == "LUCAID" else "normal"
        ax.text(-1.5, y, r, ha="right", va="center", fontsize=fs,
                fontweight=bold, clip_on=False)
        ax.text(101.5, y, f"{sev[r][0]:.1f}%", ha="left", va="center", fontsize=fs,
                fontweight=bold, clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-h, (n - 1) * step + h)
    ax.axis("off")
    ax.set_title(f"{marker} — Per-task agreement", fontsize=26, fontweight="bold",
                 pad=14, loc="center")
    ps.save_figure(fig, Path(outdir) / f"figure8d_concordance_{san(marker)}")


# =============================================================================
# 8e — case-level concordance tile grid (LUCAID)
# =============================================================================
def panel_8e(outdir):
    """Case-level agreement grid: one imshow row per rater with white tile
    gridlines and thicker case separators, rows best->worst, cases
    easiest->hardest, overall % at row end."""
    task_keys = list(C.TASKS)                                   # 5 markers, paper order
    frames = {tk: C.load_task(tk) for tk in task_keys}
    ROWS = ["Model"] + PATHS
    rater_col = {"Model": "LUCAID", **{p: p for p in PATHS}}
    disp = lambda r: "LUCAID" if r == "Model" else r

    def ch(tk, r, k):
        """0 concordant / 1 off-by-1 / 2 off-by->=2 / 3 not scored."""
        d = frames[tk]
        if k not in d.index:
            return 3
        rv, cv = d.at[k, rater_col[r]], d.at[k, "reference"]
        if pd.isna(rv) or pd.isna(cv):
            return 3
        dd = abs(category(rv, C.TASKS[tk]["cuts"]) - category(cv, C.TASKS[tk]["cuts"]))
        return 0 if dd == 0 else (1 if dd == 1 else 2)

    # Prospective cohort = cases scored for any IHC task (-> 70), first-appearance order.
    cases, seen = [], set()
    for tk in ["pdl1", "cmet", "trop2_membrane", "trop2_cytoplasm"]:
        d = frames[tk]
        for k in d.index[d["LUCAID"].notna() & d["reference"].notna()]:
            if k not in seen:
                seen.add(k); cases.append(k)

    # cases easiest -> hardest: ascending count of discordant (off-by-1 / >=2) tiles.
    oc = sorted(cases, key=lambda k: sum(ch(tk, r, k) in (1, 2)
                                         for r in ROWS for tk in task_keys))

    def ov(r):  # overall concordance % over scored (case, marker) pairs
        v = [x for k in cases for tk in task_keys if (x := ch(tk, r, k)) != 3]
        return 100 * sum(x == 0 for x in v) / len(v) if v else 0.0
    OV = {r: ov(r) for r in ROWS}
    orr = ["Model"] + sorted(PATHS, key=lambda r: OV[r], reverse=True)

    cmap = ListedColormap([ps.CONCORDANT, ps.OFF_BY_ONE, ps.OFF_BY_MANY, ps.NOT_SCORED])
    NC, NB = len(oc), len(task_keys)
    rh, rg = 1.15, 0.42
    total = len(orr) * rh + (len(orr) - 1) * rg

    fig = plt.figure(figsize=(23.4, 6.2))
    ax = fig.add_axes([0.055, 0.32, 0.85, 0.55])
    for ri, r in enumerate(orr):
        row = np.array([[ch(tk, r, k) for k in oc for tk in task_keys]])
        yt = ri * (rh + rg)
        ax.imshow(row, aspect="auto", cmap=cmap, vmin=0, vmax=3,
                  interpolation="nearest", extent=[0, NC * NB, yt + rh, yt])
    for k in range(1, NC * NB):
        ax.axvline(k, color="white", lw=2.2 if k % NB == 0 else 0.4)
    ax.set_xlim(0, NC * NB)
    ax.set_ylim(total, 0)
    ax.set_yticks([ri * (rh + rg) + rh / 2 for ri in range(len(orr))])
    ax.set_yticklabels([disp(r) for r in orr], fontsize=20)
    for t, r in zip(ax.get_yticklabels(), orr):
        t.set_fontweight("bold" if r == "Model" else "normal")
    ax.set_xticks([(kk - 1) * NB + (NB - 1) / 2 for kk in range(10, NC + 1, 10)])
    ax.set_xticklabels(list(range(10, NC + 1, 10)), fontsize=20)
    ax.tick_params(length=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for ri, r in enumerate(orr):
        ax.text(NC * NB * 1.006, ri * (rh + rg) + rh / 2, f"{OV[r]:.0f}%",
                va="center", ha="left", fontsize=20,
                fontweight="bold" if r == "Model" else "normal", clip_on=False)
    ax.set_title("Case-level concordance", fontsize=27, fontweight="bold", pad=16)
    ax.annotate("", xy=(1.0, -0.16), xytext=(0.0, -0.16), xycoords="axes fraction",
                annotation_clip=False, arrowprops=dict(arrowstyle="-|>", color="black", lw=3.0))
    ax.text(0.0, -0.22, "Easiest cases", transform=ax.transAxes, fontsize=20, va="top")
    ax.text(1.0, -0.22, "Hardest cases", transform=ax.transAxes, fontsize=20, ha="right", va="top")
    leg = [Patch(color=ps.CONCORDANT, label="concordant"), Patch(color=ps.OFF_BY_ONE, label="off by 1"),
           Patch(color=ps.OFF_BY_MANY, label="off by ≥2"), Patch(color=ps.NOT_SCORED, label="not scored")]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(1.0, -0.29), ncol=4,
              frameon=False, fontsize=20)
    ps.save_figure(fig, Path(outdir) / "figure8e_case_concordance_grid")
    print("  8e: " + ", ".join(f"{disp(r)} {OV[r]:.0f}%" for r in orr))


def legends(outdir):
    """Stand-alone shared legends for the calibration and severity panels."""
    fig, ax = plt.subplots(figsize=(9, 0.9))
    ax.axis("off")
    ax.legend(handles=[Patch(color=ps.CONCORDANT, label="concordant"),
                       Patch(color=ps.OFF_BY_ONE, label="off by 1"),
                       Patch(color=ps.OFF_BY_MANY, label="off by ≥2")],
              loc="center", ncol=3, frameon=False, fontsize=22)
    ps.save_figure(fig, Path(outdir) / "figure8_legend_severity")

    fig, ax = plt.subplots(figsize=(19, 0.9))
    ax.axis("off")
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ps.CONCORDANT, markersize=14, label="LUCAID correct"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor=ps.ACCENT, markersize=14, label="LUCAID wrong"),
         Line2D([0], [0], marker="D", color="w", markerfacecolor=ps.CONCORDANT, markersize=12, label="Pathologist correct"),
         Line2D([0], [0], marker="D", color="w", markerfacecolor=ps.ACCENT, markersize=12, label="Pathologist wrong by 1"),
         Line2D([0], [0], marker="D", color="w", markerfacecolor=ps.OFF_BY_MANY, markersize=12, label="Pathologist wrong by ≥2"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markeredgecolor="black",
                markeredgewidth=HL_RING, markersize=14, label="representative case (images)")]
    ax.legend(handles=h, loc="center", ncol=6, frameon=False, fontsize=20,
              handletextpad=0.5, columnspacing=1.6)
    ps.save_figure(fig, Path(outdir) / "figure8_legend_calibration")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="figures")
    ap.add_argument("--highlight-case", default=HIGHLIGHT_DEFAULT,
                    help="surrogate case_id to highlight in 8a ('' to disable).")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for marker, task_key, model_col in MARKERS:
        bio = load_panel(task_key, model_col)
        calibration_figure(marker, bio, model_col, args.highlight_case, outdir)
        mae, sev = compute(bio, model_col, CUTS[marker])
        mae_bars(marker, mae, outdir)
        severity_track(marker, sev, outdir)
        print(f"  {marker:14s} MAE(LUCAID)={mae['LUCAID']:.1f}  "
              f"concordance(LUCAID)={sev['LUCAID'][0]:.1f}%")
    panel_8e(outdir)
    legends(outdir)
    print(f"Saved Figure 8 panels to {outdir}/")


if __name__ == "__main__":
    main()
