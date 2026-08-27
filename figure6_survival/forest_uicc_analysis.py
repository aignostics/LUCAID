#!/usr/bin/env python3
"""Forest analysis — per-feature UICC-adjusted Cox over the full metric panel (source of panel i).

Each feature is fitted in its own bivariate Cox model OS ~ feature(high/low) + UICC stage, so its
direction is the stage-adjusted marginal (no co-fit direction flips). Reports HR, 95% CI, two-sided
Wald p and Benjamini–Hochberg q across the panel. Writes the full-metrics results table and, in a
separate file, the UICC-stage reference row. No plotting — fig6i_forest_single.py draws
panel i from these CSVs.
Env: CONT=1 continuous per-SD; MEDIANONLY=1 force median split.
"""
from pathlib import Path
import os, numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lung_split import smart_split

LUNG = Path(os.environ.get('LUNG_DATA', 'lung_data')); MIN_OS = 3.0
U = lambda s: s.astype(str).str.upper().str.strip()
man = pd.read_csv(LUNG / 'lung_manifest_core2case.csv'); man['core_uuid'] = U(man.core_uuid); man['case_id'] = U(man.case_id)


def agg1(fn, col):
    d = pd.read_csv(LUNG / fn, usecols=lambda c: c == col or c.lower() in ('core_uuid', 'tcga_slide_uuid', 'slide_uuid'), low_memory=False)
    k = [c for c in d.columns if c.lower() in ('core_uuid', 'tcga_slide_uuid', 'slide_uuid')][0]; d[k] = U(d[k])
    a = d[[k, col]].rename(columns={k: 'core_uuid'}).merge(man, on='core_uuid', how='inner').groupby('case_id')[col].mean()
    a.index = U(pd.Series(a.index)); a.index.name = 'CASE'; a.name = col; return a


# established markers (25; UICC-adjusted per marker) — the established-marker list mapped to our data
EST = {'dens_TIL': 'Lymphocytes/mm² (whole tumor)', 'TIL_pct_stromal': 'Stromal TIL %', 'TIL_pct_intratumoral': 'Intratumoral TIL %',
       'immune_infiltration_score': 'Immune infiltration score', 'total_immune_density': 'Total immune density',
       'tNLR': 'Tissue NLR', 'LGR': 'Lympho/granulocyte ratio', 'LMR': 'Lympho/macrophage ratio', 'TIL_tumour_ratio': 'TIL/tumor ratio',
       'dens_macrophage': 'Macrophage density', 'mac_tumour_ratio': 'Macrophage/tumor', 'dens_granulocyte': 'Granulocyte density',
       'gra_tumour_ratio': 'Granulocyte/tumor', 'stroma_cell_density': 'Stromal cell density', 'stroma_pct': 'Stroma area %',
       'dens_fibroblast': 'Fibroblast density', 'CAF_tumour_ratio': 'CAF/tumor ratio', 'TSR': 'Tumor–stroma ratio',
       'invasivity': 'Carcinoma delineation', 'immune_exclusion_density': 'Immune exclusion index', 'interface_immunity_ref_20um': 'Interface immunity (20µm)',
       'dens_endothelial_vascularization': 'Vascularization index', 'dens_tumour': 'Tumor cell density',
       'tumour_area_pct': 'Tumor area %', 'normal_pct': 'Normal tissue %'}
# novel panel = 7 spatial features selected from the 586-metric screen (analysis_novel_screen.py):
# 4 FDR-significant + 3 nominal-only, chosen for interpretability, family coverage, and both risk directions;
# adverse distances carry a 'Higher' prefix so the risk direction reads directly on the label.
DEL = 'curator_arms_expanded/delaunay_niches_features.csv'; DIST = 'curator_arms/cell_distances_full.csv'
NOV = {'DEL_COMMTYPE_FRAC_LYMPHOID': ('Lymphoid niche (TLS-like)', DEL, 'community'),
       'DEL_EDGE_FRAC_CAR_PLA': ('Carcinoma–plasma cell adjacency', DEL, 'adjacency'),
       'DEL_EDGE_FRAC_CAR_LYM': ('Carcinoma–lymphocyte adjacency', DEL, 'adjacency'),
       'd_MAC_CAR_carc_median': ('Macrophage–carcinoma cell distance', DIST, 'distance'),
       'd_CAR_LYM_carc_median': ('Higher carcinoma cell–lymphocyte distance', DIST, 'distance'),
       'd_LYM_LYM_median': ('Higher lymphocyte dispersion', DIST, 'distance'),
       'd_END_LYM_median': ('Higher endothelial cell–lymphocyte distance', DIST, 'distance')}

pat = pd.read_csv(LUNG / 'lung_patient_landscape_ext.csv'); pat['CASE'] = U(pat.CASE)
d = pat[(pat.OS_M >= MIN_OS) & pat.EVENT.isin([0, 1])].copy(); d['E'] = (d.EVENT == 1).astype(int); d['stage'] = pd.to_numeric(d.stage_num, errors='coerce')
for col, (lab, fn, fam) in NOV.items():
    d = d.merge(agg1(fn, col).reset_index(), on='CASE', how='left')
T, E = d.OS_M.values, d.E.values

# composition: % of each cell phenotype + tissue-compartment % (from spot_master RELATIVE_AREA_*)
sm = pd.read_csv(LUNG / 'lung_spot_master.csv', usecols=lambda c: c == 'core_uuid' or c.startswith('RELATIVE_AREA_'), low_memory=False); sm['core_uuid'] = U(sm.core_uuid)
_CP = {'RELATIVE_AREA_NECROSIS': 'necrosis_pct', 'RELATIVE_AREA_VESSEL': 'vessel_pct', 'RELATIVE_AREA_BLOOD': 'blood_pct', 'RELATIVE_AREA_EPITHELIAL_TISSUE': 'epithelial_pct'}
_cp = sm.merge(man, on='core_uuid', how='inner').groupby('case_id')[[c for c in _CP if c in sm.columns]].mean().rename(columns=_CP)
_cp.index = U(pd.Series(_cp.index)); _cp.index.name = 'CASE'; d = d.merge(_cp.reset_index(), on='CASE', how='left')
COMP = {'dens_plasma': 'Plasma cell density', 'pct_carcinoma': 'Carcinoma %', 'pct_lymphocyte': 'Lymphocytes %',
        'pct_plasma': 'Plasma cells %', 'pct_macrophage': 'Macrophages %', 'pct_granulocyte': 'Granulocytes %',
        'pct_fibroblast': 'Fibroblasts %', 'pct_endothelial': 'Endothelial %',
        'necrosis_pct': 'Necrosis %', 'vessel_pct': 'Vessel %', 'blood_pct': 'Blood %', 'epithelial_pct': 'Epithelial %'}
LAB = {**EST, **{k: v[0] for k, v in NOV.items()}, **COMP}
FAM = {**{k: 'established' for k in EST}, **{k: v[2] for k, v in NOV.items()}, **{k: 'composition' for k in COMP}}

CONT = os.getenv('CONT') == '1'                # continuous per-SD (primary); else dichotomised adaptive split
MEDIANONLY = os.getenv('MEDIANONLY') == '1'    # force a plain median split for every feature (simplified variant)
rows = []
for m in [x for x in LAB if x in d.columns]:
    v = pd.to_numeric(d[m], errors='coerce')
    cut, rule = (float(np.nanmedian(v)), 'median') if MEDIANONLY else smart_split(v)
    if CONT:
        vv = v.astype(float)
        if (v.dropna() >= 0).all() and float(pd.Series(v.dropna()).skew()) > 1.0:
            vv = np.log1p(v)
        xcol = ((vv - vv.mean()) / vv.std()).values
    else:
        xcol = (v > cut).astype(float).mask(v.isna()).values   # NaN stays NaN -> dropped below (not forced low)
    df = pd.DataFrame({'T': T, 'E': E, 'hi': xcol, 'stage': d.stage.values}).dropna()
    if df.hi.std() == 0 or df.E.sum() < 20:
        continue
    s = CoxPHFitter().fit(df, 'T', 'E').summary.loc['hi']
    rows.append(dict(m=m, fam=FAM[m], HR=float(s['exp(coef)']), lo=float(s['exp(coef) lower 95%']),
                     hi95=float(s['exp(coef) upper 95%']), p=float(s['p']), n=int(len(df)), events=int(df.E.sum()),
                     split_rule=rule, split_cut=round(float(cut), 4)))
R = pd.DataFrame(rows).sort_values('HR').reset_index(drop=True)

# Benjamini–Hochberg q across the whole feature panel (p-rank order, monotone-enforced)
pv = R.p.values; order = np.argsort(pv); ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]; q = np.empty_like(q_sorted); q[order] = np.clip(q_sorted, 0, 1); R['q'] = q

# merge in the screen-wide BH q (from analysis_novel_screen.py) if present, so the panel
# carries both the post-selection panel q and the full-screen q for the novel features
try:
    scr = pd.read_csv(LUNG / 'lung_novel_screen.csv')[['metric', 'q', 'redundant', 'WINNER']] \
        .rename(columns={'metric': 'm', 'q': 'screen_q', 'redundant': 'screen_redundant', 'WINNER': 'screen_winner'})
    R = R.merge(scr, on='m', how='left')
except FileNotFoundError:
    pass

R['lab'] = R.m.map(lambda m: LAB.get(m, m)); R['novel'] = R.fam != 'established'

TAG = ('cont' if CONT else 'panel') + ('_median' if MEDIANONLY else '')
R.round(4).to_csv(LUNG / f'lung_forest_uicc_{TAG}.csv', index=False)

# UICC-stage reference row (OS ~ stage only) — clinical anchor, not part of the feature FDR (q = NaN)
_du = pd.DataFrame({'T': T, 'E': E, 'stage': d.stage.values}).dropna(); _su = CoxPHFitter().fit(_du, 'T', 'E').summary.loc['stage']
pd.DataFrame([dict(m='UICC_stage', fam='clinical', HR=float(_su['exp(coef)']), lo=float(_su['exp(coef) lower 95%']),
                   hi95=float(_su['exp(coef) upper 95%']), p=float(_su['p']), n=int(len(_du)), events=int(_du.E.sum()),
                   q=np.nan, lab='UICC stage (per step)', novel=False)]
             ).round(4).to_csv(LUNG / f'lung_forest_uicc_{TAG}_uicc.csv', index=False)

nsig = int((R.p < 0.05).sum()); nsq = int((R.q < 0.05).sum()); nnov = int(((R.p < 0.05) & (R.fam != 'established')).sum())
print(f'{len(R)} features · {nsig} nominal p<0.05 ({nnov} novel) · {nsq} survive BH-FDR')
print(R[R.p < 0.05][['m', 'fam', 'HR', 'p', 'q']].to_string(index=False))
print(f'wrote lung_forest_uicc_{TAG}.csv (+ _uicc.csv)')
