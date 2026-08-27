#!/usr/bin/env python3
"""Figure 6 F–I — per-metric landscape + Kaplan–Meier from lung_individual_km_metrics.csv. Env TITLELESS=1 drops titles."""
from pathlib import Path
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
for p in ['/System/Library/Fonts/Supplemental/Arial.ttf','/Library/Fonts/Arial.ttf','/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
    if os.path.exists(p): fm.fontManager.addfont(p)
INK='#333333'; FAV='#10288C'; RISKC='#E6AF00'
plt.rcParams.update({'font.family':'Arial','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180,
    'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#888','xtick.color':INK,'ytick.color':INK})
LUNG=Path(os.environ.get('LUNG_DATA', 'lung_data')); MIN_OS=3.0; U=lambda s:s.astype(str).str.upper().str.strip()
OUT=LUNG/'individual_survival'; OUT.mkdir(exist_ok=True)
man=pd.read_csv(LUNG/'lung_manifest_core2case.csv'); man['core_uuid']=U(man.core_uuid); man['case_id']=U(man.case_id)
lp=pd.read_csv(LUNG/'lung_patient_landscape_ext.csv'); lp['CASE']=U(lp.CASE)
lp=lp[(lp.OS_M>=MIN_OS)&lp.EVENT.isin([0,1])].copy(); lp['E']=(lp.EVENT==1).astype(int)
def agg1(fn,col):
    d=pd.read_csv(LUNG/fn,usecols=lambda c: c==col or c.lower() in ('core_uuid','tcga_slide_uuid','slide_uuid'),low_memory=False)
    key=[c for c in d.columns if c.lower() in ('core_uuid','tcga_slide_uuid','slide_uuid')][0]; d[key]=U(d[key])
    a=d[[key,col]].rename(columns={key:'core_uuid'}).merge(man,on='core_uuid',how='inner').groupby('case_id')[col].mean()
    a.index=U(pd.Series(a.index)); a.name=col; a.index.name='CASE'; return a
# final F–I selection: 2 novel spatial (F, G) + 2 established (H, I)
SPECS=[
 ('DEL_EDGE_FRAC_CAR_PLA','curator_arms_expanded/delaunay_niches_features.csv','Carcinoma–plasma cell adjacency','F_carcinoma_plasma_adjacency'),
 ('d_END_LYM_median','curator_arms/cell_distances_full.csv','Endothelial cell–lymphocyte distance','G_endothelial_lymphocyte_distance'),
 ('TIL_pct_stromal',None,'Stromal TIL %','H_stromal_TIL'),
 ('tNLR',None,'Tissue NLR','I_tissue_NLR'),
]
for col,src,_,_ in SPECS:
    if src and col not in lp.columns: lp=lp.merge(agg1(src,col).reset_index(),on='CASE',how='left')
MET=pd.read_csv(LUNG/'lung_individual_km_metrics.csv').set_index('metric')

def wraplab(s):
    if len(s)<=20: return s
    if '–' in s: i=s.index('–'); return s[:i+1]+'\n'+s[i+1:]
    if ' ' in s:
        mid=len(s)/2; sp=[i for i,c in enumerate(s) if c==' ']; i=min(sp,key=lambda j:abs(j-mid)); return s[:i]+'\n'+s[i+1:]
    return s

def landscape_png(v,label,cut,rule,fav_high,title,fname):
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    v=pd.to_numeric(v,errors='coerce'); vs=np.sort(v.dropna().values); n=len(vs); hi=vs>cut
    sk=float(pd.Series(vs).skew()) if n>8 else 0
    if abs(sk)>1.5 and vs.min()>0: ax.set_yscale('log')
    fav=hi if fav_high else ~hi
    ax.scatter(np.arange(n)[fav],vs[fav],s=8,c=FAV,edgecolors='none',zorder=2)
    ax.scatter(np.arange(n)[~fav],vs[~fav],s=8,c=RISKC,edgecolors='none',zorder=2)
    ax.axhline(cut,color=INK,ls='--',lw=1.2,zorder=3)
    ax.text(0.015,cut,(f'median {cut:.3g}' if rule=='median' else f'{rule} {cut:.3g}'),transform=ax.get_yaxis_transform(),ha='left',va='bottom',fontsize=15,color=INK,zorder=5)
    ax.set_xlim(0,n); ax.set_xlabel('Patients',fontsize=18); ax.set_ylabel(wraplab(label),fontsize=18); ax.tick_params(labelsize=16)
    if os.getenv('TITLELESS')!='1': ax.set_title(title,fontsize=11.5,fontweight='bold',loc='left',color=INK,pad=8)
    hi_c=FAV if fav_high else RISKC; lo_c=RISKC if fav_high else FAV; word='median' if rule=='median' else 'threshold'
    ax.legend(handles=[Line2D([0],[0],marker='o',ls='',mfc=hi_c,mec='none',ms=10,label=f'> {word}'),
                       Line2D([0],[0],marker='o',ls='',mfc=lo_c,mec='none',ms=10,label=f'< {word}')],fontsize=15,loc='upper left',frameon=False)
    fig.tight_layout(); fig.savefig(OUT/fname,bbox_inches='tight',pad_inches=0.25,dpi=300); plt.close(fig)

def km_png(v,T,E,cut,rule,label,fav_high,adjHR,adjLo,adjHi,lr,accent,title,fname):
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    v=pd.to_numeric(v,errors='coerce'); ok=v.notna()&T.notna()&E.notna()
    v,Tt,Ee=v[ok].values,T[ok].values,E[ok].values; g=v>cut; fav=g if fav_high else ~g
    leglines=[]; word='median' if rule=='median' else 'threshold'
    for mask,lab,c in [(fav,(f'> {word}' if fav_high else f'< {word}'),FAV),(~fav,(f'< {word}' if fav_high else f'> {word}'),RISKC)]:
        kmf=KaplanMeierFitter(label=lab).fit(Tt[mask],Ee[mask]); med=kmf.median_survival_time_
        ms='NR' if not np.isfinite(med) else f'{med:.0f} mo'
        kmf.plot_survival_function(ax=ax,ci_show=True,ci_alpha=0.18,show_censors=False,color=c,lw=2.6)
        leglines.append((f'{lab} (median OS {ms})',c))
    ax.set_xlim(0,60); ax.set_xticks(np.arange(0,61,12)); ax.set_ylim(0,1.02); ax.tick_params(labelsize=16)
    ax.set_xlabel('Overall survival (months)',fontsize=18); ax.set_ylabel('Overall survival',fontsize=18)
    ax.legend([Line2D([0],[0],color=c,lw=2.8) for _,c in leglines],[t for t,_ in leglines],fontsize=14,loc='lower left',frameon=False)
    lrs='log-rank p < 0.001' if lr<0.001 else f'log-rank p = {lr:.3f}'
    ax.text(0.975,0.125,f'HR {adjHR:.2f} ({adjLo:.2f}–{adjHi:.2f})',transform=ax.transAxes,ha='right',va='bottom',fontsize=15,fontweight='bold',color=accent)
    ax.text(0.975,0.04,lrs,transform=ax.transAxes,ha='right',va='bottom',fontsize=15,color=INK)
    if os.getenv('TITLELESS')!='1': ax.set_title(f'{title} · adj. HR {adjHR:.2f}',fontsize=13,fontweight='bold',loc='left',color=accent,pad=6)
    fig.tight_layout(); fig.savefig(OUT/fname,bbox_inches='tight',pad_inches=0.25,dpi=300); plt.close(fig)

T=pd.to_numeric(lp.OS_M,errors='coerce'); E=pd.to_numeric(lp.E,errors='coerce')
for col,src,label,base in SPECS:
    m=MET.loc[col]; cut,rule,fav_high=float(m.split_cut),m.split_rule,bool(m.fav_high)
    adjHR,adjLo,adjHi,lr=float(m.adjHR),float(m.adjLo),float(m.adjHi),float(m.logrank_p)
    accent=FAV if fav_high else RISKC; v=pd.to_numeric(lp[col],errors='coerce')
    landscape_png(v,label,cut,rule,fav_high,f'{label} — patient landscape',f'{base}_landscape.png')
    km_png(v,T,E,cut,rule,label,fav_high,adjHR,adjLo,adjHi,lr,accent,f'{label} — overall survival (high vs low)',f'{base}_KM.png')
    print(f'{col:<24} adjHR {adjHR:.2f}  {base}')
print(f'wrote {2*len(SPECS)} PNGs (dpi300) → {OUT}/')
