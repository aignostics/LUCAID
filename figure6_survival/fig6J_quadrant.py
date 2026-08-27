#!/usr/bin/env python3
"""Figure 6J (for record) — PD-L1 × TIL immune-phenotype panel as three standardized plots:
  (1) lung_J_quadrant.png       — PD-L1 × TIL quadrant (both axes split at their median)
  (2) lung_J_pdl1_landscape.png — PD-L1 TPS ranked patient landscape (median split line)
  (3) lung_J_til_landscape.png  — TIL density ranked patient landscape (median split line)
PD-L1 = AI-estimated tumour-proportion score (model_TPS, a smooth continuous score that replaces the
coarse pathologist grid), averaged per case across its spots (PD-L1_case+spotIDs_with_model_TPS.xlsx).
Title-less; Arial, standardized fonts/sizes.
"""
from pathlib import Path
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for p in ['/System/Library/Fonts/Supplemental/Arial.ttf','/Library/Fonts/Arial.ttf','/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
    if os.path.exists(p): fm.fontManager.addfont(p)
INK='#333333'; HI='#2E7D32'; LO='#E8B10A'
plt.rcParams.update({'font.family':'Arial','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180,
    'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#888','xtick.color':INK,'ytick.color':INK})
FS_LAB=23; FS_TICK=21; FS_GRP=19   # enlarged axis titles + scale numbers + quadrant labels (Arial)
LUNG=Path(os.environ.get('LUNG_DATA', 'lung_data')); U=lambda s:s.astype(str).str.upper().str.strip()
QCOL={'Th/Ph':'#1B7A3D','Th/Pl':'#8CCB6E','Tl/Ph':'#F0A93B','Tl/Pl':'#D9702B'}
QLAB={'Th/Ph':'PD-L1+ / TIL+','Th/Pl':'PD-L1− / TIL+','Tl/Ph':'PD-L1+ / TIL−','Tl/Pl':'PD-L1− / TIL−'}

pat=pd.read_csv(LUNG/'lung_patient_landscape_ext.csv'); pat['CASE']=U(pat.CASE)
pdl1=pd.read_excel(LUNG/'PD-L1_case+spotIDs_with_model_TPS.xlsx',sheet_name='Sheet1')
pdl1['CASE']=U(pdl1.Case_UUID); pdl1['TPS']=pd.to_numeric(pdl1.model_TPS,errors='coerce')   # AI-estimated continuous TPS
tps=pdl1.dropna(subset=['TPS']).groupby('CASE')['TPS'].mean().reset_index()                 # mean per case across spots
pat=pat.merge(tps,on='CASE',how='left')
til=pd.to_numeric(pat.dens_TIL,errors='coerce'); tp=pd.to_numeric(pat.TPS,errors='coerce')
ok=til.notna()&tp.notna()&(til>0); til,tp=til[ok].values,tp[ok].values; N=len(til)
PCUT=float(np.median(tp)); TCUT=float(np.median(til))                                  # median split, both axes
qkey=np.where(til>TCUT,'Th','Tl')+'/'+np.where(tp>PCUT,'Ph','Pl'); qn={k:int((qkey==k).sum()) for k in QCOL}

# ---------- (1) quadrant ----------
fig,axQ=plt.subplots(figsize=(7.2,6.5))
axQ.set_xscale('function',functions=(np.sqrt,np.square)); axQ.set_yscale('log')   # √ x-axis → median split near centre
for k in QCOL:
    m=qkey==k; axQ.scatter(np.clip(tp[m],0,None),til[m],s=17,c=QCOL[k],alpha=0.8,lw=0)
axQ.axvline(PCUT,color=INK,ls='--',lw=1.5); axQ.axhline(TCUT,color=INK,ls='--',lw=1.5)
axQ.set_xlim(0,102); axQ.set_xticks([0,1,5,25,50,100]); axQ.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axQ.set_xlabel('PD-L1 TPS (%)',fontsize=FS_LAB)
axQ.set_ylabel('Lymphocyte density (cells/mm²)',fontsize=FS_LAB); axQ.tick_params(labelsize=FS_TICK)
for k,(fx,fy,ha,va) in {'Th/Ph':(0.97,0.97,'right','top'),'Th/Pl':(0.03,0.97,'left','top'),'Tl/Ph':(0.97,0.03,'right','bottom'),'Tl/Pl':(0.03,0.03,'left','bottom')}.items():
    axQ.text(fx,fy,f'{QLAB[k]}\nn={qn[k]} ({qn[k]/N*100:.0f}%)',transform=axQ.transAxes,ha=ha,va=va,fontsize=FS_GRP,
             fontweight='bold',color='#222',bbox=dict(boxstyle='round,pad=0.3',fc=QCOL[k],ec='none',alpha=0.28))
fig.tight_layout(); fig.savefig(LUNG/'lung_J_quadrant.png',dpi=300,bbox_inches='tight',pad_inches=0.25,facecolor='white'); plt.close(fig)

# ---------- shared ranked-landscape renderer ----------
def landscape(vals,cut,ylab,fname,ylog=True,sqrt=False,step=False,xticks=None):
    v=np.sort(vals); n=len(v); x=np.arange(n); hi=v>cut
    fig,ax=plt.subplots(figsize=(6.6,4.0))
    if step:
        sp=int(np.sum(v<=cut))
        ax.plot(x[:sp+1],v[:sp+1],drawstyle='steps-post',color=LO,lw=2.4); ax.plot(x[sp:],v[sp:],drawstyle='steps-post',color=HI,lw=2.4)
    else:
        ax.plot(x,v,color='#9aa0a6',lw=1.1,alpha=0.7,zorder=1)
        ax.scatter(x[~hi],v[~hi],s=7,c=LO,lw=0,zorder=2); ax.scatter(x[hi],v[hi],s=7,c=HI,lw=0,zorder=2)
    if ylog: ax.set_yscale('log')
    elif sqrt: ax.set_yscale('function',functions=(np.sqrt,np.square)); ax.set_yticks([0,1,5,25,50,100]); ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())   # √ scale — matches the quadrant PD-L1 axis
    ax.axhline(cut,color=INK,ls='--',lw=1.3)
    ax.set_xlim(0,n); ax.set_xticks(xticks if xticks is not None else [t for t in (0,200,400,600) if t<=n])   # default: same x-ticks on both landscapes
    ax.set_xlabel('Patients',fontsize=FS_LAB); ax.set_ylabel(ylab,fontsize=FS_LAB); ax.tick_params(labelsize=FS_TICK)
    fig.subplots_adjust(left=0.22,right=0.97,top=0.95,bottom=0.18)   # FIXED axes box → same Patients-axis length on every landscape (they stack up/down)
    fig.savefig(LUNG/fname,dpi=300,facecolor='white'); plt.close(fig)

landscape(tp,PCUT,'PD-L1 TPS (%)','lung_J_pdl1_landscape.png',ylog=False,sqrt=True,step=False)
landscape(til,TCUT,'Lymphocytes/mm²\n(whole tumor)','lung_J_til_landscape.png',ylog=True)
landscape(tp,PCUT,'PD-L1 TPS (%)','lung_J_pdl1_landscape_dense.png',ylog=False,sqrt=True,step=False,xticks=list(range(0,701,100)))     # alt: dense 0/100/…/700 ticks
landscape(til,TCUT,'Lymphocytes/mm²\n(whole tumor)','lung_J_til_landscape_dense.png',ylog=True,xticks=list(range(0,701,100)))       # alt: dense 0/100/…/700 ticks

print(f'N={N}  PD-L1 median(TPS)={PCUT:.2f}  TIL median={TCUT:.0f}')
print('groups:',qn)
print('wrote lung_J_quadrant.png + lung_J_pdl1_landscape.png + lung_J_til_landscape.png')
