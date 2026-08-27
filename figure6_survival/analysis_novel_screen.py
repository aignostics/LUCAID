#!/usr/bin/env python3
"""Screen of the full candidate spatial-TME metric family for overall-survival association.

Every metric across the six spatial families (distance, niche, ripley, delaunay, perivascular,
interface) is fitted in its own UICC-stage-adjusted Cox model (high/low at the adaptive split),
with Benjamini-Hochberg FDR applied across the whole screen and a redundancy flag against the
established/basic markers (|Spearman rho| > 0.70 -> redundant). WINNER = FDR-significant AND
non-redundant. Writes lung_novel_screen.csv. The seven novel features shown in the Figure 6i
forest are selected from this screen for interpretability, family coverage and both risk
directions (see forest_uicc_analysis.py), which is why they include FDR-significant and
nominal-only features.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from lung_split import smart_split
LUNG=Path(os.environ.get('LUNG_DATA', 'lung_data')); MIN_OS=3.0; RHO_RED=0.70; U=lambda s:s.astype(str).str.upper().str.strip()
man=pd.read_csv(LUNG/'lung_manifest_core2case.csv'); man['core_uuid']=U(man.core_uuid); man['case_id']=U(man.case_id)
def aggcols(fn,cols):
    key_opts=('core_uuid','tcga_slide_uuid','slide_uuid')
    d=pd.read_csv(LUNG/fn,low_memory=False); k=[c for c in d.columns if c.lower() in key_opts][0]; d[k]=U(d[k])
    keep=[c for c in cols if c in d.columns]
    a=d[[k]+keep].rename(columns={k:'core_uuid'}).merge(man,on='core_uuid',how='inner').groupby('case_id')[keep].mean()
    a.index=U(pd.Series(a.index)); a.index.name='CASE'; return a.reset_index()
def feat_cols(fn,fam):
    cols=list(pd.read_csv(LUNG/fn,nrows=1).columns)
    bad=lambda c:(c.lower() in ('tcga_slide_uuid','_cohort','core_uuid') or c.startswith('n_') or c.startswith('DEL_N_')
        or 'TOTAL' in c or 'COUNT' in c.upper() or c.endswith('_count') or '_area_mm2' in c or c in ('endothelial_count','vessel_cell_frac','DEL_TOTAL_EDGES_RAW','DEL_TOTAL_EDGES_PRUNED','DEL_PRUNED_EDGE_FRAC'))
    if fam=='distance': sel=[c for c in cols if c.startswith('d_') and c.endswith('_median')]+[c for c in cols if c.startswith('ratio_')]
    elif fam=='niche':  sel=[c for c in cols if c.startswith('niche_')]
    elif fam=='ripley': sel=[c for c in cols if c.startswith('ripley_')]
    elif fam=='delaunay':sel=[c for c in cols if c.startswith('DEL_') and not bad(c)]
    elif fam=='perivascular':sel=[c for c in cols if (c.startswith('peri_') or c.startswith('perivascular_')) and not bad(c)]
    elif fam=='interface':sel=[c for c in cols if ('interface' in c.lower() or 'gradient' in c.lower()) and not bad(c)]
    else: sel=[c for c in cols if not bad(c)]
    return sel
FAM={'distance':'curator_arms/cell_distances_full.csv','niche':'curator_arms/niches_kmeans.csv',
     'ripley':'curator_arms_expanded/cell_pair_colocalisation_features.csv','delaunay':'curator_arms_expanded/delaunay_niches_features.csv',
     'perivascular':'curator_arms_expanded/perivascular_barrier_features.csv','interface':'curator_arms_expanded/interface_scores_features.csv'}

lp=pd.read_csv(LUNG/'lung_patient_landscape_ext.csv'); lp['CASE']=U(lp.CASE)
REF=[c for c in lp.columns if c.startswith(('dens_','pct_','TIL_','stroma','tumour','normal')) or c in
     ('tNLR','LGR','LMR','immune_infiltration_score','total_immune_density','immune_exclusion_density',
      'interface_immunity_ref_20um','TSR','invasivity','CAF_tumour_ratio','mac_tumour_ratio','gra_tumour_ratio','dens_endothelial_vascularization')]
REF=sorted(set(REF)); refM=lp.set_index('CASE')[REF].apply(pd.to_numeric,errors='coerce')
d=lp[(lp.OS_M>=MIN_OS)&lp.EVENT.isin([0,1])].copy(); d['E']=(d.EVENT==1).astype(int); d['stage']=pd.to_numeric(d.stage_num,errors='coerce')
d=d[['CASE','OS_M','E','stage']].copy()

rows=[]
for fam,fn in FAM.items():
    sel=feat_cols(fn,fam)
    A=aggcols(fn,sel); m=d.merge(A,on='CASE',how='left').merge(refM.reset_index(),on='CASE',how='left')
    print(f'{fam}: screening {len(sel)} metrics',flush=True)
    for col in sel:
        v=pd.to_numeric(m[col],errors='coerce')
        if v.notna().sum()<100 or v.nunique()<5: continue
        if v.round(6).value_counts(normalize=True).iloc[0]>0.90: continue
        cut,_=smart_split(v); hi=(v>cut).astype(float)
        df=pd.DataFrame({'T':m.OS_M,'E':m.E,'stage':m.stage,'hi':hi}).dropna()
        if df.hi.std()==0 or df.E.sum()<20: continue
        try: s=CoxPHFitter().fit(df,'T','E').summary.loc['hi']
        except Exception: continue
        rr=[abs(spearmanr(v,m[rc],nan_policy='omit')[0]) for rc in REF]; maxrho=float(np.nanmax(rr)) if rr else np.nan
        near=REF[int(np.nanargmax(rr))] if rr else ''
        rows.append(dict(family=fam,metric=col,HR=round(float(s['exp(coef)']),3),p=round(float(s['p']),5),
                         max_rho_vs_ref=round(maxrho,3),nearest_ref=near))
R=pd.DataFrame(rows); o=np.argsort(R.p.values); ranked=R.p.values[o]*len(R)/np.arange(1,len(R)+1)
q=np.minimum.accumulate(ranked[::-1])[::-1]; qq=np.empty_like(q); qq[o]=np.clip(q,0,1); R['q']=np.round(qq,4)
R['redundant']=R.max_rho_vs_ref>RHO_RED; R['WINNER']=(R.q<0.05)&(~R.redundant)
R.sort_values('p').to_csv(LUNG/'lung_novel_screen.csv',index=False)
print(f'\nScreened {len(R)} novel metrics · nominal p<0.05={int((R.p<0.05).sum())} · FDR q<0.05={int((R.q<0.05).sum())} · '
      f'non-redundant FDR WINNERS={int(R.WINNER.sum())}')
print('\n=== WINNERS (FDR-significant AND non-redundant vs established+basic) ===')
print(R[R.WINNER].sort_values('p')[['family','metric','HR','p','q','max_rho_vs_ref','nearest_ref']].to_string(index=False))
print('\n=== FDR-significant but REDUNDANT (rho>0.70 → dropped) ===')
print(R[(R.q<0.05)&R.redundant].sort_values('max_rho_vs_ref',ascending=False)[['family','metric','HR','p','max_rho_vs_ref','nearest_ref']].head(15).to_string(index=False))
