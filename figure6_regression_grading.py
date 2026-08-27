#!/usr/bin/env python3
"""Figure 6e/6f — tissue-compartment regression grading (LUCAID vs pathologist).

Compares LUCAID-derived tissue-compartment proportions (viable carcinoma /
stroma / necrosis) against the joint pathologist assessment across 140
neoadjuvant-treated resection cases, as a proxy for pathological regression
grading.

Produces the combined scatter (Fig 6e: LUCAID vs pathologist %, all three
compartments, with the identity line, Pearson r and Spearman rho) and the MAE
panel (Fig 6f: half-violin + per-case dots + the mean-absolute-error line), plus
the three per-compartment scatters for reference.

Input: data/regression_grading.csv — per case, the LUCAID and pathologist
fraction for each compartment (each rater's three compartments sum to ~1).

Usage:
    python figure6_regression_grading.py --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

import style as ps

ps.apply_house_style()

SRC = Path(__file__).resolve().parent / "data" / "regression_grading.csv"

# Compartments in legend order; colours match the Fig-6 tissue-compartment
# stacked-bar figure so the panels read as the same palette.
COMPARTMENTS = ["Carcinoma", "Stroma", "Necrosis"]
COL = {"Carcinoma": "#b57075", "Stroma": "#e0bd98", "Necrosis": "#9d9fd6"}


def _pfmt(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def load() -> dict:
    """{compartment: (pathologist_%, lucaid_%)} with values scaled to 0-100."""
    df = pd.read_csv(SRC)
    return {c: (df[f"{c}_pathologist_%"].to_numpy() * 100.0,
                df[f"{c}_LUCAID_%"].to_numpy() * 100.0) for c in COMPARTMENTS}


def _scatter(ax, xs, ys, cols):
    """Identity line + points; return (Pearson r, Spearman rho, MAE) over all points."""
    ax.plot([0, 100], [0, 100], color=ps.IDENTITY_LINE, lw=1.4, zorder=1)
    for x, y, c in zip(xs, ys, cols):
        ax.scatter(x, y, s=90, c=c, alpha=0.85, marker=ps.MODEL_MARKER,
                   edgecolors="black", linewidths=0.4, zorder=3)
    X, Y = np.concatenate(xs), np.concatenate(ys)
    r, pr = pearsonr(X, Y)
    rho, psp = spearmanr(X, Y)
    mae = float(np.mean(np.abs(Y - X)))
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ps.style_axes(ax)
    ax.text(0.0, 1.02, f"r = {r:.2f}, {_pfmt(pr)}", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold")
    ax.text(1.0, 1.02, f"ρ = {rho:.2f}, {_pfmt(psp)}", transform=ax.transAxes,
            ha="right", va="bottom", fontweight="bold")
    return r, rho, mae


def scatter_panel(name, xs, ys, cols, xlab, ylab, outdir, legend=False):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    stats = _scatter(ax, xs, ys, cols)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if legend:
        handles = [Line2D([0], [0], marker=ps.MODEL_MARKER, color="none",
                          markerfacecolor=COL[c], markeredgecolor="black",
                          markeredgewidth=0.4, markersize=10, label=c)
                   for c in COMPARTMENTS]
        ax.legend(handles=handles, loc="lower right", frameon=False)
    ps.save_figure(fig, Path(outdir) / name)
    return stats


def mae_panel(data, outdir):
    """Half-violin + per-case dots + MAE line per compartment, ordered best->worst."""
    err = {c: np.abs(data[c][1] - data[c][0]) for c in COMPARTMENTS}
    mae = {c: float(err[c].mean()) for c in COMPARTMENTS}
    order = sorted(COMPARTMENTS, key=lambda c: mae[c])
    rng = np.random.default_rng(7)  # deterministic jitter

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for i, c in enumerate(order):
        e = err[c]
        v = ax.violinplot([e], positions=[i], widths=0.80, showextrema=False)
        for b in v["bodies"]:
            vt = b.get_paths()[0].vertices
            vt[:, 0] = np.clip(vt[:, 0], i, np.inf)          # right half only
            b.set_facecolor(COL[c])
            b.set_alpha(0.50)
            b.set_edgecolor("black")
            b.set_linewidth(0.4)
        ax.scatter(i - 0.22 + rng.uniform(-0.14, 0.14, len(e)), e, s=16,
                   c=COL[c], alpha=0.85, edgecolors="black", linewidths=0.25, zorder=3)
        ax.hlines(mae[c], i - 0.42, i + 0.40, color="black", lw=2.2, zorder=5)
        ax.text(i + 0.40, mae[c] + 0.4, f"{mae[c]:.1f}", ha="right", va="bottom",
                fontweight="bold", zorder=6)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylim(-0.5, None)
    ax.set_ylabel("Absolute error (percentage points)")
    ps.style_axes(ax)
    ps.save_figure(fig, Path(outdir) / "figure6f_regression_MAE")
    return mae


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load()
    n = len(data[COMPARTMENTS[0]][0])

    print(f"Regression grading — LUCAID vs pathologist, n = {n} cases")
    print(f"{'compartment':11s} {'Pearson r':>9s} {'Spearman':>9s} {'MAE %':>7s}")
    for c in COMPARTMENTS:
        x, y = data[c]
        r, rho, mae = scatter_panel(
            f"figure6e_regression_{c}", [x], [y], [COL[c]],
            f"{c} — Pathologist (%)", f"{c} — LUCAID (%)", outdir)
        print(f"{c:11s} {r:9.3f} {rho:9.3f} {mae:7.2f}")

    r, rho, mae = scatter_panel(
        "figure6e_regression_combined",
        [data[c][0] for c in COMPARTMENTS], [data[c][1] for c in COMPARTMENTS],
        [COL[c] for c in COMPARTMENTS], "Pathologist (%)", "LUCAID (%)", outdir,
        legend=True)
    print(f"{'combined':11s} {r:9.3f} {rho:9.3f} {mae:7.2f}   (pooled {3*n} points)")

    mae_panel(data, outdir)
    print(f"Saved scatter panels + MAE to {outdir}/")


if __name__ == "__main__":
    main()
