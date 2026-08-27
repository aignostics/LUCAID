#!/usr/bin/env python3
"""Supplementary Figure 4a/4b — extended prospective clinical validation.

4a  Pairwise inter-rater Spearman correlation matrices (LUCAID + P1-P5) for each
    of the five scoring tasks, one per panel in a row, with a right-hand 'Avg'
    column giving each rater's mean pairwise correlation with the others.

4b  Table of mean absolute error (with 95% bootstrap CI) and the correlation
    (Spearman ρ vs the adjudicated consensus, Pearson r vs the molecular KRAS
    reference) for LUCAID and each pathologist across the tasks, plus tumor
    cellularity vs KRAS (2 x VAF). LUCAID (area) is the area-based cellularity estimate.
    Full numbers are also written to figures/supplementary_figure4b_metrics.csv.

Usage:
    python supplementary_figure4.py --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

import style as ps
import common as C

ps.apply_house_style()

PATHS = ["P1", "P2", "P3", "P4", "P5"]

_CMAP = LinearSegmentedColormap.from_list(
    "house_blue", [ps.MODEL_BG, ps.MODEL_LIGHT, ps.MODEL, ps.MODEL_SERIES[2]])
_CMAP.set_bad(color="none")


# --------------------------------------------------------------------------- #
# 4a — pairwise inter-rater correlation matrices (one row of five panels)
# --------------------------------------------------------------------------- #
def _corr_with_avg(frame: pd.DataFrame):
    corr = frame.corr(method="spearman")
    raters = list(corr.columns)
    mat = corr.to_numpy()
    n = len(raters)
    avg = (mat.sum(axis=1) - np.diag(mat)) / (n - 1)
    return raters, mat, avg


def _draw_matrix(ax, raters, mat, avg, title, norm):
    n = len(raters)
    disp = np.full((n, n + 1), np.nan)
    lower = ~np.triu(np.ones((n, n), dtype=bool), k=1)
    disp[:, :n][lower] = mat[lower]
    disp[:, n] = avg
    ax.imshow(disp, cmap=_CMAP, norm=norm, aspect="equal")
    for r in range(n):
        for c in range(n + 1):
            v = disp[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm(v) > 0.55 else "#2b333b")
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(raters + ["Avg"], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(raters, fontsize=7.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.axvline(n - 0.5, color=ps.THRESHOLD_LINE, linewidth=1.0)
    ax.set_xticks(np.arange(-0.5, n + 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def figure_4a(frames, outdir):
    panels = []
    for tk in C.TASKS:
        frame = frames[tk][[r for r in C.RATERS if r in frames[tk].columns]]
        panels.append((C.TASKS[tk]["label"], *_corr_with_avg(frame)))
    norm = Normalize(vmin=0.4, vmax=1.0)

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4))
    for ax, (title, raters, mat, avg) in zip(axes, panels):
        _draw_matrix(ax, raters, mat, avg, title, norm)
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, aspect=25, pad=0.01)
    cbar.set_label("Spearman ρ", fontsize=9)
    fig.suptitle("Inter-rater agreement — pairwise Spearman correlation "
                 "(LUCAID vs pathologists)", fontsize=12.5, fontweight="semibold")
    ps.save_figure(fig, Path(outdir) / "supplementary_figure4a_rater_correlations")


# --------------------------------------------------------------------------- #
# 4b — MAE (95% CI) + Spearman correlation table
# --------------------------------------------------------------------------- #
ROWS_4B = ["P1", "P2", "P3", "P4", "P5", "LUCAID", "LUCAID (area)"]
COLS_4B = [  # (display title, column key)
    ("Cellularity\n(vs Consensus, %)", "cell_cons"),
    ("Cellularity\n(vs KRAS AF, %)", "cell_kras"),
    ("PD-L1 TPS\n(%)", "pdl1"),
    ("c-MET\nH-score", "cmet"),
    ("TROP-2 Membrane\nH-score", "tmem"),
    ("TROP-2 Cytoplasm\nH-score", "tcyt"),
]


def _metrics_table(frames):
    """{(col_key, row_label): (mae, lo, hi, corr)} — Spearman ρ vs the consensus,
    Pearson r vs the molecular KRAS reference (as in Figure 5)."""
    m = {}

    def put(col_key, row_label, pred, ref, method="spearman"):
        res = C.annotator_metrics(pred, ref, method=method)
        if res:
            m[(col_key, row_label)] = (res["mae"], res["mae_ci_lower"],
                                       res["mae_ci_upper"], res["correlation"])

    # vs consensus, per task
    task_col = {"cellularity": "cell_cons", "pdl1": "pdl1", "cmet": "cmet",
                "trop2_membrane": "tmem", "trop2_cytoplasm": "tcyt"}
    for tk, ck in task_col.items():
        df = frames[tk]
        for r in PATHS:
            put(ck, r, df[r], df["reference"])
        put(ck, "LUCAID", df["LUCAID"], df["reference"])
        if tk == "cellularity":
            put(ck, "LUCAID (area)", df["LUCAID_area"], df["reference"])

    # cellularity vs KRAS (2 x VAF) on the prospective cohort. Like Figure 5, drop
    # cases with 2 x VAF > 100 % (VAF > 50 %, which break the heterozygous-diploid
    # assumption) and report Pearson r against the molecular reference.
    cell = frames["cellularity"].copy()
    cell["tf"] = 2.0 * cell["kras_vaf"]
    cell = cell[cell["tf"] <= 100.0]
    for r in PATHS:
        put("cell_kras", r, cell[r], cell["tf"], method="pearson")
    put("cell_kras", "LUCAID", cell["LUCAID"], cell["tf"], method="pearson")
    put("cell_kras", "LUCAID (area)", cell["LUCAID_area"], cell["tf"], method="pearson")
    return m


def figure_4b(frames, outdir):
    m = _metrics_table(frames)

    # CSV with the full numbers
    rows = [dict(comparison=title.replace("\n", " "), rater=rl, mae=v[0],
                 ci_lo=v[1], ci_hi=v[2], correlation=v[3],
                 corr_method="pearson" if ck == "cell_kras" else "spearman")
            for title, ck in COLS_4B for rl in ROWS_4B
            if (v := m.get((ck, rl))) is not None]
    pd.DataFrame(rows).to_csv(Path(outdir) / "supplementary_figure4b_metrics.csv",
                              index=False)

    # Rendered table
    ncol = len(COLS_4B)
    x_label = 0.02
    x0 = 0.16                       # first task block start
    block = (0.995 - x0) / ncol
    y_top, y_sub, y_first, dy = 0.93, 0.82, 0.72, 0.095

    fig, ax = plt.subplots(figsize=(15.5, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for j, (title, ck) in enumerate(COLS_4B):
        cx = x0 + block * (j + 0.5)
        ax.text(cx, y_top, title, ha="center", va="center", fontsize=9,
                fontweight="bold")
        ax.text(x0 + block * j + block * 0.34, y_sub, "MAE (95% CI)",
                ha="center", va="center", fontsize=8, color="#4a5663")
        # Pearson r for the molecular KRAS column; Spearman rho for the rest.
        ax.text(x0 + block * j + block * 0.85, y_sub, "r" if ck == "cell_kras" else "ρ",
                ha="center", va="center", fontsize=8, style="italic", color="#4a5663")
        ax.axvline(x0 + block * j, ymin=0.06, ymax=0.87, color="#e6e9ec", lw=0.8)

    for i, rl in enumerate(ROWS_4B):
        y = y_first - i * dy
        bold = rl.startswith("LUCAID")
        fw = "bold" if bold else "normal"
        if rl == "LUCAID":                       # separator above the model rows
            ax.axhline(y + dy * 0.5, xmin=0.01, xmax=0.99, color="#c8ccd0", lw=1.0)
        ax.text(x_label, y, rl, ha="left", va="center", fontsize=8.5, fontweight=fw)
        for j, (_, ck) in enumerate(COLS_4B):
            v = m.get((ck, rl))
            mae_x = x0 + block * j + block * 0.34
            rho_x = x0 + block * j + block * 0.85
            if v is None:
                ax.text(mae_x, y, "—", ha="center", va="center", fontsize=8.5,
                        color="#9aa4ac")
                continue
            mae, lo, hi, rho = v
            ax.text(mae_x, y, f"{mae:.1f} ({lo:.1f}–{hi:.1f})", ha="center",
                    va="center", fontsize=8.5, fontweight=fw)
            ax.text(rho_x, y, f"{rho:.2f}", ha="center", va="center", fontsize=8.5,
                    fontweight=fw)

    ax.set_title("Rater agreement with the reference standard — MAE (95% CI) and "
                 "correlation (Spearman ρ; Pearson r for KRAS)", fontsize=12,
                 fontweight="bold", y=1.02)
    ps.save_figure(fig, Path(outdir) / "supplementary_figure4b_table")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = {tk: C.load_task(tk) for tk in C.TASKS}
    figure_4a(frames, outdir)
    m = figure_4b(frames, outdir)
    print("Supplementary Figure 4b (MAE / 95% CI / correlation):")
    for _, r in m.iterrows():
        print(f"  {r['comparison']:26s} {r['rater']:14s} "
              f"MAE={r['mae']:6.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] "
              f"{r['corr_method'][:4]}={r['correlation']:.3f}")
    print(f"Saved Suppl Fig 4a/4b + metrics CSV to {outdir}/")


if __name__ == "__main__":
    main()
