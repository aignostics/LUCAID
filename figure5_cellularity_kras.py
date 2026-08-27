#!/usr/bin/env python3
"""Figure 5f-h — tumor cellularity vs KRAS variant allele frequency.

Orthogonal molecular validation of tumor-cellularity estimates against KRAS VAF
in a retrospective NSCLC cohort (n = 115). The molecular reference is the
estimated tumor cell fraction = 2 x VAF (for a heterozygous somatic mutation);
cases with 2 x VAF > 100 % (VAF > 50 %) break the heterozygous-diploid
assumption and are dropped rather than clipped. Every panel is scored on the
same matched cohort (cases with a routine pathologist estimate and both LUCAID
estimates), so all three share one n.

Panels (each vs 2 x VAF, with Pearson r, Spearman rho and MAE):
    5f  routine pathologist cellularity estimate
    5g  LUCAID cell-count-based estimate
    5h  LUCAID nuclear-area-based estimate

Input: data/cellularity_vs_kras.csv (case_id, kras_vaf, patho_routine,
LUCAID_count, LUCAID_area, P1..P5).

Usage:
    python figure5_cellularity_kras.py --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, t as student_t

import style as ps

ps.apply_house_style()

SRC = Path(__file__).resolve().parent / "data" / "cellularity_vs_kras.csv"

# (panel letter, column, title, is-model). Molecular reference on X, estimate on Y.
PANELS = [
    ("f", "patho_routine", "Pathologist", False),
    ("g", "LUCAID_count", "LUCAID (Cell Count)", True),
    ("h", "LUCAID_area", "LUCAID (Nucleus Area)", True),
]
MOL_LABEL = "Estimated tumor cell fraction (2×VAF) [%]"


def _fit_with_ci(x, y, n_pts: int = 100, ci: float = 0.95):
    """Least-squares line + 95% confidence band for the mean response."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), n_pts)
    ys = intercept + slope * xs
    dof = n - 2
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid ** 2) / dof)
    x_bar = x.mean()
    sxx = np.sum((x - x_bar) ** 2)
    t_val = student_t.ppf(0.5 + ci / 2.0, dof)
    half = t_val * s_err * np.sqrt(1.0 / n + (xs - x_bar) ** 2 / sxx)
    return xs, ys, ys - half, ys + half


def load_matched() -> pd.DataFrame:
    """Case-level table restricted to the matched cohort with 2xVAF <= 100%."""
    df = pd.read_csv(SRC)
    df["tumor_fraction"] = 2.0 * df["kras_vaf"]
    df = df[df["tumor_fraction"] <= 100.0]
    df = df.dropna(subset=["patho_routine", "LUCAID_count", "LUCAID_area"])
    return df


def panel(letter, col, title, is_model, df, outdir):
    d = df[[col, "tumor_fraction"]].dropna()
    x, y = d["tumor_fraction"].values, d[col].values
    r, pr = pearsonr(y, x)
    rho, _ = spearmanr(y, x)
    mae = float(np.mean(np.abs(y - x)))

    fig, ax = plt.subplots(figsize=(4.7, 4.6))
    xs, ys, lo, hi = _fit_with_ci(x, y)
    ax.fill_between(xs, lo, hi, color=ps.ACCENT, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, ys, color=ps.ACCENT, lw=1.9, zorder=2.5)
    ax.scatter(x, y, marker=ps.MODEL_MARKER if is_model else ps.PATHOLOGIST_MARKER,
               s=55 if is_model else 48, color=ps.MODEL if is_model else ps.PATHOLOGIST,
               alpha=0.85, edgecolors="white", linewidth=0.4, zorder=3)
    ax.text(0.04, 0.96,
            f"r = {r:.2f} {ps.significance_stars(pr)}\nMAE = {mae:.1f}\n"
            f"p = {pr:.1e}\nn = {len(d)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=ps.THRESHOLD_LINE, alpha=0.9))
    ax.set_xlabel(MOL_LABEL, fontsize=11)
    ax.set_ylabel("Tumor cellularity [%]", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ps.style_axes(ax, grid=True)
    ps.save_figure(fig, Path(outdir) / f"figure5{letter}_cellularity_kras_{col}")
    return dict(panel=letter, variant=title, n=len(d), r=r, rho=rho, mae=mae, p=pr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_matched()
    print(f"Cellularity vs KRAS (2×VAF) — matched cohort n = {len(df)}")
    print(f"{'panel':5s} {'variant':22s} {'n':>4s} {'Pearson r':>9s} "
          f"{'Spearman':>9s} {'MAE':>6s}")
    rows = []
    for letter, col, title, is_model in PANELS:
        s = panel(letter, col, title, is_model, df, outdir)
        rows.append(s)
        print(f"5{s['panel']:4s} {s['variant']:22s} {s['n']:>4d} {s['r']:>9.3f} "
              f"{s['rho']:>9.3f} {s['mae']:>6.2f}")
    pd.DataFrame(rows).to_csv(outdir / "figure5_cellularity_kras_stats.csv", index=False)
    print(f"Saved panels 5f/g/h + stats to {outdir}/")


if __name__ == "__main__":
    main()
