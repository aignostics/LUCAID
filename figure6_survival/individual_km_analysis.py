#!/usr/bin/env python3
"""Figure 6 F–I metrics: adaptive split, UICC-adjusted Cox HR (95% CI, Wald p), log-rank p. No plotting."""
import os
from pathlib import Path
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lung_split import smart_split

LUNG = Path(os.environ.get('LUNG_DATA', 'lung_data')); MIN_OS = 3.0
U = lambda s: s.astype(str).str.upper().str.strip()
man = pd.read_csv(LUNG / 'lung_manifest_core2case.csv'); man['core_uuid'] = U(man.core_uuid); man['case_id'] = U(man.case_id)
lp = pd.read_csv(LUNG / 'lung_patient_landscape_ext.csv'); lp['CASE'] = U(lp.CASE)
lp = lp[(lp.OS_M >= MIN_OS) & lp.EVENT.isin([0, 1])].copy(); lp['E'] = (lp.EVENT == 1).astype(int); lp['stage'] = pd.to_numeric(lp.stage_num, errors='coerce')

def agg1(fn, col):
    d = pd.read_csv(LUNG / fn, usecols=lambda c: c == col or c.lower() in ('core_uuid', 'tcga_slide_uuid', 'slide_uuid'), low_memory=False)
    key = [c for c in d.columns if c.lower() in ('core_uuid', 'tcga_slide_uuid', 'slide_uuid')][0]; d[key] = U(d[key])
    a = d[[key, col]].rename(columns={key: 'core_uuid'}).merge(man, on='core_uuid', how='inner').groupby('case_id')[col].mean()
    a.index = U(pd.Series(a.index)); a.name = col; a.index.name = 'CASE'; return a

# final F–I selection: 2 novel spatial (F, G) + 2 established (H, I)
SPECS = [
    ('DEL_EDGE_FRAC_CAR_PLA', 'curator_arms_expanded/delaunay_niches_features.csv', 'Carcinoma–plasma cell adjacency', 'F_carcinoma_plasma_adjacency', 'novel'),
    ('d_END_LYM_median', 'curator_arms/cell_distances_full.csv', 'Endothelial cell–lymphocyte distance', 'G_endothelial_lymphocyte_distance', 'novel'),
    ('TIL_pct_stromal', None, 'Stromal TIL %', 'H_stromal_TIL', 'established'),
    ('tNLR', None, 'Tissue NLR', 'I_tissue_NLR', 'established'),
]
for col, src, _, _, _ in SPECS:
    if src and col not in lp.columns:
        lp = lp.merge(agg1(src, col).reset_index(), on='CASE', how='left')

T = pd.to_numeric(lp.OS_M, errors='coerce'); E = pd.to_numeric(lp.E, errors='coerce')
rows = []
for col, src, label, panel, kind in SPECS:
    v = pd.to_numeric(lp[col], errors='coerce'); cut, rule = smart_split(v.dropna().values)
    df = pd.DataFrame({'T': T.values, 'E': E.values, 'hi': (v > cut).astype(float).mask(v.isna()).values, 'stage': lp.stage.values}).dropna()
    s = CoxPHFitter().fit(df, 'T', 'E').summary.loc['hi']
    adjHR, adjLo, adjHi, cox_p = float(s['exp(coef)']), float(s['exp(coef) lower 95%']), float(s['exp(coef) upper 95%']), float(s['p'])
    ok = v.notna() & T.notna() & E.notna(); vv, Tt, Ee = v[ok].values, T[ok].values, E[ok].values; g = vv > cut
    logrank_p = logrank_test(Tt[g], Tt[~g], Ee[g], Ee[~g]).p_value
    rows.append(dict(metric=col, label=label, panel=panel, kind=kind, split_rule=rule, split_cut=round(float(cut), 4),
                     adjHR=adjHR, adjLo=adjLo, adjHi=adjHi, cox_p=cox_p, logrank_p=logrank_p,
                     n_cox=int(len(df)), n_logrank=int(ok.sum()), fav_high=bool(adjHR < 1)))

out = pd.DataFrame(rows)
out.round(4).to_csv(LUNG / 'lung_individual_km_metrics.csv', index=False)
print(out[['metric', 'adjHR', 'cox_p', 'logrank_p', 'split_rule', 'fav_high']].to_string(index=False))
print(f'wrote lung_individual_km_metrics.csv ({len(out)} metrics)')
