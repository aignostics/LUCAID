#!/usr/bin/env python3
"""Figure 6i — single UICC-adjusted Cox forest over the full metric panel.

Reads the forest results (forest_uicc_analysis.py) and draws every feature (established
circles, novel-spatial diamonds, UICC-stage square), ordered by HR, bold where the feature
is nominally significant (stage-adjusted Wald p < 0.05). All features are shown, including
the non-significant ones, so the novel panel is not selectively displayed. No model fitting.
Writes lung_forest_final_labels.png (+ lung_final_label_map.csv).
"""
from pathlib import Path
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
          '/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Arial', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 180})
LUNG = Path(os.environ.get('LUNG_DATA', 'lung_data'))
RISK, PROT = '#E6AF00', '#10288C'

FINAL = {
 'dens_TIL': 'Lym / mm²', 'TIL_pct_stromal': 'Stromal TIL %',
 'TIL_pct_intratumoral': 'Intratumoral TIL %',
 'immune_infiltration_score': 'Immune infiltration score',
 'total_immune_density': 'Total immune / mm²', 'tNLR': 'Tissue NLR',
 'LGR': 'Lym/Gran ratio', 'LMR': 'Lym/Mac ratio', 'TIL_tumour_ratio': 'TIL / CaC ratio',
 'dens_macrophage': 'Mac / mm²', 'mac_tumour_ratio': 'Mac / CaC ratio',
 'dens_granulocyte': 'Gran / mm²', 'gra_tumour_ratio': 'Gran / CaC ratio',
 'stroma_cell_density': 'Stromal cell / mm²', 'stroma_pct': 'Stroma area %',
 'dens_fibroblast': 'Fib / mm²', 'CAF_tumour_ratio': 'CAF / CaC ratio',
 'TSR': 'Tumor – stroma ratio', 'invasivity': 'Carcinoma delineation',
 'immune_exclusion_density': 'Immune exclusion index',
 'interface_immunity_ref_20um': 'Interface immunity (20µm)',
 'dens_endothelial_vascularization': 'Vascularization index',
 'dens_tumour': 'Tumor cell / mm²', 'tumour_area_pct': 'Tumor area %',
 'normal_pct': 'Normal tissue %',
 'DEL_COMMTYPE_FRAC_LYMPHOID': 'Lymphoid niche (TLS-like)',
 'DEL_EDGE_FRAC_CAR_PLA': 'CaC – PC adjacency',
 'DEL_EDGE_FRAC_CAR_LYM': 'CaC – Lym adjacency',
 'd_MAC_CAR_carc_median': 'Mac – CaC distance',
 'd_CAR_LYM_carc_median': 'CaC – Lym distance',
 'd_LYM_LYM_median': 'Lym dispersion',
 'd_END_LYM_median': 'EC – Lym distance',
 'dens_plasma': 'PC / mm²', 'pct_carcinoma': 'Carcinoma %', 'pct_lymphocyte': 'Lym %',
 'pct_plasma': 'PC %', 'pct_macrophage': 'Mac %', 'pct_granulocyte': 'Gran %',
 'pct_fibroblast': 'Fib %', 'pct_endothelial': 'Endothelial %',
 'necrosis_pct': 'Necrosis %', 'vessel_pct': 'Vessel %', 'blood_pct': 'Blood %',
 'epithelial_pct': 'Epithelial %', 'UICC_stage': 'UICC stage',
}

R = pd.read_csv(LUNG / 'lung_forest_uicc_panel.csv').sort_values('HR').reset_index(drop=True)
uicc = pd.read_csv(LUNG / 'lung_forest_uicc_panel_uicc.csv')
missing = set(R.m) - set(FINAL)
assert not missing, f'unmapped metrics: {missing}'
pd.DataFrame([dict(metric=m, internal_label=l, final_label=FINAL[m])
              for m, l in zip(pd.concat([R, uicc]).m, pd.concat([R, uicc]).lab)]) \
  .to_csv(LUNG / 'lung_final_label_map.csv', index=False)
R = pd.concat([R, uicc], ignore_index=True)

n = len(R)
FL = 1.48
fig, ax = plt.subplots(figsize=(14.0, 0.205 * n + 1.2))
for i, (_, r) in enumerate(R.iterrows()):
    y = n - 1 - i
    clin = r.fam == 'clinical'
    col = RISK if r.HR > 1 else PROT
    sig = r.p < 0.05
    nov = r.fam not in ('established', 'composition', 'clinical')
    ax.plot([r.lo, r.hi95], [y, y], color=col, lw=3.0 if (sig or clin) else 1.5,
            alpha=1 if (sig or clin) else .5, solid_capstyle='round', zorder=1)
    mk = 's' if clin else ('D' if nov else 'o')
    ax.plot(r.HR, y, mk, mfc=col, mec='#222' if clin else 'white',
            mew=1.4 if clin else 1.1,
            ms=10 if clin else ((9 if sig else 6) if nov else (8.5 if sig else 5.5)),
            alpha=1 if (sig or clin) else .6, zorder=2)
    ps = 'p<0.0001' if r.p < 1e-4 else f'p={r.p:.2g}'
    ax.text(1.80, y, f'{r.HR:.2f} ({r.lo:.2f}–{r.hi95:.2f})  {ps}' + (' *' if sig else ''),
            va='center', ha='left', fontsize=9.2 * FL,
            fontweight='bold' if sig else 'normal', color=col, clip_on=False)
ax.axvline(1, color='#444', ls='--', lw=1.1)
ax.set_yticks(range(n))
yl = ax.set_yticklabels([FINAL[m] for m in R.m[::-1]], fontsize=10.3 * FL)
for t, (_, r) in zip(yl, R[::-1].iterrows()):
    t.set_fontweight('bold' if r.p < 0.05 else 'normal')
ax.set_xscale('log')
ax.set_ylim(-0.8, n - 0.2)
ax.set_xlim(0.52, 2.75)
ax.set_xticks([0.6, 0.8, 1.0, 1.5, 2.0])
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.tick_params(labelsize=10 * FL)
ax.set_xlabel('Overall-survival hazard ratio (high vs low, adaptive split) '
              '· each feature UICC-adjusted · 95% CI', fontsize=10.5 * FL)
ax.legend(handles=[
    Line2D([0], [0], marker='o', ls='', mfc=PROT, mec='none', ms=9, label='Protective (HR<1)'),
    Line2D([0], [0], marker='o', ls='', mfc=RISK, mec='none', ms=9, label='Risk (HR>1)'),
    Line2D([0], [0], marker='D', ls='', mfc=PROT, mec='white', mew=1.0, ms=9,
           label='Novel spatial feature'),
    Line2D([0], [0], marker='s', ls='', mfc=RISK, mec='#222', mew=1.2, ms=9,
           label='UICC stage (clinical anchor)')],
    fontsize=9.6 * FL, loc='upper center', bbox_to_anchor=(0.5, -0.055), frameon=False,
    ncol=4, handletextpad=0.4, columnspacing=1.8)
fig.tight_layout()
fig.savefig(LUNG / 'lung_forest_final_labels.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'wrote lung_forest_final_labels.png ({n} rows) + lung_final_label_map.csv')
