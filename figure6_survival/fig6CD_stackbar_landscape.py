#!/usr/bin/env python3
"""Cohort composition landscapes (all centres).
  A  tissue COMPARTMENTS per patient (segmentation): carcinoma / epithelial / stroma / necrosis / vessel / blood / other
  B  CELL TYPES per patient: carcinoma / lymphocyte / plasma / macrophage / granulocyte / fibroblast / endothelial
Each bar = one patient; y = relative frequency (0–1); samples sorted by carcinoma-cell content (descending), as in the
reference. Compartments from slide_readouts RELATIVE_AREA_* (core→case mean); cell types from the pct_* landscape."""
from pathlib import Path
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Patch
for f in ['/System/Library/Fonts/Supplemental/Arial.ttf','/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
    if os.path.exists(f): fm.fontManager.addfont(f)
INK='#333333'; GREY='#6b6b6b'
plt.rcParams.update({'font.family':'Arial','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180,
    'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#888','xtick.color':INK,'ytick.color':INK})
LUNG=Path(os.environ.get('LUNG_DATA', 'lung_data')); U=lambda s:s.astype(str).str.upper().str.strip()
man=pd.read_csv(LUNG/'lung_manifest_core2case.csv'); man['core_uuid']=U(man.core_uuid); man['case_id']=U(man.case_id)
clin=pd.read_csv(LUNG/'validation_cohorts_lung_metadata _luncohort_metadata_UUID.csv',low_memory=False)
clin['CASE']=U(clin.spot_case_uuid); clin['city']=clin.ENR.astype(str).str.extract(r'^([A-Za-z]+)')[0].str[0].map({'E':'Centre A','C':'Centre B'})
HASMETA=set(clin.CASE)   # default: no centre filter — every annotated case incl. the third-source cases (n≈1,054)
if os.getenv('CENTRE_BK')=='1':   # manuscript C/D composition cohort: the two main centres only (drops the third-source cases → n≈1,017)
    HASMETA=set(clin.CASE[clin.city.isin(['Centre A','Centre B'])])

# ── A: COMPARTMENTS (segmentation) from slide-level RELATIVE_AREA_* ──
COMP=['CARCINOMA','EPITHELIAL_TISSUE','NECROSIS','STROMA','VESSEL','BLOOD','OTHER']
CLAB={'CARCINOMA':'carcinoma','EPITHELIAL_TISSUE':'epithelial','NECROSIS':'necrosis','STROMA':'stroma','VESSEL':'vessel','BLOOD':'blood','OTHER':'other'}
# EXACT reference tissue-segmentation colour codes (provided)
CCOL={'CARCINOMA':'#B57075','EPITHELIAL_TISSUE':'#7BA4B2','STROMA':'#E0BD98','NECROSIS':'#9D9FD6','VESSEL':'#BCE2B2','BLOOD':'#C0AE7B','OTHER':'#F7F1F6'}
sm=pd.read_csv(LUNG/'lung_spot_master.csv',usecols=['core_uuid']+[f'RELATIVE_AREA_{c}' for c in COMP],low_memory=False)
sm['core_uuid']=U(sm.core_uuid); sm=sm.merge(man,on='core_uuid',how='inner')
compC=sm.groupby('case_id')[[f'RELATIVE_AREA_{c}' for c in COMP]].mean()
compC.columns=COMP; compC=compC.reset_index().rename(columns={'case_id':'CASE'}); compC['CASE']=U(compC.CASE)
compC=compC[compC.CASE.isin(HASMETA)].copy()                      # no centre filter (any centre, annotated cases)
compM=compC.set_index('CASE')[COMP].copy()
compM=compM[compM.sum(axis=1)>0]                                   # drop cases with no segmented tissue (would render as blank bars)
compM=compM.div(compM.sum(axis=1),axis=0)                          # normalise to segmented tissue

# ── B: CELL TYPES — all NINE classifier classes from raw per-class CELL_COUNT_* (core→case sum) ──
#    epithelial-cell and other-cell ARE detected classes; they were dropped when the 7-class pct_* landscape was
#    derived, so build the phenotype panel here directly from the raw counts → the complete classification.
CT=['carcinoma','epithelial','lymphocyte','plasma','macrophage','granulocyte','fibroblast','endothelial','other']
CLSMAP={'carcinoma':'CARCINOMA_CELL','epithelial':'EPITHELIAL_CELL','lymphocyte':'LYMPHOCYTE','plasma':'PLASMA_CELL',
        'macrophage':'MACROPHAGE','granulocyte':'GRANULOCYTE','fibroblast':'FIBROBLAST','endothelial':'ENDOTHELIAL_CELL','other':'OTHER'}
# EXACT frozen cell-classification colour codes (all nine classes)
CTCOL={'carcinoma':'#7514F5','epithelial':'#377EF6','lymphocyte':'#A1FC4E','plasma':'#EA33F6','macrophage':'#1200F4',
       'granulocyte':'#EF8733','fibroblast':'#FEFF54','endothelial':'#EA3323','other':'#75FBFD'}
CLAB_CELL={'carcinoma':'carcinoma cell','epithelial':'epithelial cell','lymphocyte':'lymphocyte','plasma':'plasma cell',
           'macrophage':'macrophage','granulocyte':'granulocyte','fibroblast':'fibroblast','endothelial':'endothelial cell','other':'other cell'}
ccnt=[f'CELL_COUNT_{CLSMAP[c]}' for c in CT]
cc=pd.read_csv(LUNG/'lung_spot_master.csv',usecols=['core_uuid']+ccnt,low_memory=False); cc['core_uuid']=U(cc.core_uuid)
cc=cc.merge(man,on='core_uuid',how='inner').groupby('case_id')[ccnt].sum(); cc.index=U(pd.Series(cc.index))
cellM=cc[cc.index.isin(HASMETA)].copy(); cellM.columns=CT        # no centre filter (any centre, annotated cases)
cellM=cellM[cellM.sum(axis=1)>0]                                  # drop cases with no analysed cells (unanalysed → blank white bars)
cellM=cellM.div(cellM.sum(axis=1),axis=0); cellM.index.name='CASE'

# ── each panel is sorted INDEPENDENTLY by its OWN carcinoma fraction, high → low (no cross-panel alignment) ──
cmap=dict(zip(clin.CASE,clin.city))
compM['city']=[cmap.get(c) for c in compM.index]; cellM['city']=[cmap.get(c) for c in cellM.index]

def stack(ax,M,order,cols,sortcol,title,xlab):
    d=M.sort_values(sortcol,ascending=False)          # sort by this panel's own carcinoma fraction, high → low
    n=len(d); x=np.arange(n); cum=np.zeros(n)
    for k in order:
        ax.bar(x,d[k].values,bottom=cum,width=1.0,color=cols[k],linewidth=0,zorder=2); cum+=d[k].values
    ax.set_xlim(0,n); ax.set_ylim(0,1.0); ax.set_xticks([]); ax.set_ylabel('Relative frequency',fontsize=19); ax.tick_params(labelsize=17)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor('#cfcfcf'); sp.set_linewidth(0.7)  # light box so panel extent is visible even where 'other' is near-white
    ax.set_xlabel(xlab,fontsize=19,color=INK)
    if os.getenv('TITLELESS')!='1': ax.set_title(title,fontsize=15,fontweight='bold',loc='left',color=INK)

# ── (1) SEPARATE high-resolution single-panel files (dpi 300) ──
def standalone(M,order,cols,labels,sortcol,title,supt,legtitle,fname,ncol):
    fig,ax=plt.subplots(figsize=(13.5,4.6))
    stack(ax,M,order,cols,sortcol,title,f'Patients (n = {len(M):,})')   # x-axis aligned with the landscape panels
    if os.getenv('TITLELESS')!='1': fig.suptitle(supt,fontsize=15,fontweight='bold',y=1.05,color=INK)
    fig.tight_layout()
    if os.getenv('LEGENDLESS')!='1':   # colour key moves to the figure caption when composed (LEGENDLESS=1)
        ax.legend(handles=[Patch(fc=cols[k],label=labels[k]) for k in reversed(order)],loc='upper center',bbox_to_anchor=(0.5,-0.16),
                  ncol=ncol,fontsize=12.5,frameon=False,columnspacing=1.5,handletextpad=0.5)   # bottom horizontal legend
    fig.savefig(LUNG/fname,bbox_inches='tight',dpi=300); plt.close(fig)
standalone(compM,COMP,CCOL,CLAB,'CARCINOMA','Tissue compartments (segmentation)',
    f'Tissue-compartment abundances per tumor sample (n = {len(compM):,})','Compartment','lung_stackbar_compartments.png',4)
standalone(cellM,CT,CTCOL,CLAB_CELL,'carcinoma','Cell phenotypes (nine-class classification)',
    f'Cell-phenotype abundances per tumor sample (n = {len(cellM):,})','Cell type','lung_stackbar_cells.png',5)

# ── (2) combined two-panel figure (for the overview) — each panel independently carcinoma-sorted ──
fig,(axA,axB)=plt.subplots(2,1,figsize=(13.5,8.4))
stack(axA,compM,COMP,CCOL,'CARCINOMA','a   Tissue compartments (segmentation)',f'Tumor samples (n = {len(compM):,}, sorted by carcinoma area, high → low)')
stack(axB,cellM,CT,CTCOL,'carcinoma','b   Cell phenotypes',f'Tumor samples (n = {len(cellM):,}, sorted by carcinoma-cell fraction, high → low)')
fig.suptitle(f'Stacked bar plots of tissue-compartment (a) and cell-phenotype (b) abundances for each tumor sample\n'
             f'(n = {len(compM):,} patients; each panel independently sorted by its own\n'
             f'carcinoma fraction, high → low; cell panel = full nine-class classification)',
             fontsize=12,fontweight='bold',y=1.03,color=INK)
fig.tight_layout(rect=[0,0,0.86,1])
fig.legend(handles=[Patch(fc=CCOL[k],label=CLAB[k]) for k in COMP],loc='center left',bbox_to_anchor=(0.87,0.74),fontsize=8.2,frameon=False,title='Compartment',title_fontsize=8.6)
fig.legend(handles=[Patch(fc=CTCOL[k],label=CLAB_CELL[k]) for k in CT],loc='center left',bbox_to_anchor=(0.87,0.24),fontsize=8.2,frameon=False,title='Cell type',title_fontsize=8.6)
fig.savefig(LUNG/'lung_stackbar_landscape.png',bbox_inches='tight',dpi=200); plt.close(fig)

# ── (3) split by centre (2×2), each panel independently carcinoma-sorted; column widths ∝ n → constant bar width ──
nB_=int((compM.city=='Centre A').sum()); nK_=int((compM.city=='Centre B').sum())
fig2,axes=plt.subplots(2,2,figsize=(17.5,8.6),gridspec_kw={'width_ratios':[nB_,nK_]})
PL={0:['a','b'],1:['c','d']}
for j,cc in enumerate(['Centre A','Centre B']):
    nA=int((compM.city==cc).sum()); nBc=int((cellM.city==cc).sum())
    stack(axes[0,j],compM[compM.city==cc],COMP,CCOL,'CARCINOMA',f'{PL[0][j]}   Tissue compartments — {cc} (n = {nA})',f'sorted by carcinoma area, high → low')
    stack(axes[1,j],cellM[cellM.city==cc],CT,CTCOL,'carcinoma',f'{PL[1][j]}   Cell phenotypes — {cc} (n = {nBc})',f'sorted by carcinoma-cell fraction, high → low')
fig2.suptitle('Cell-phenotype and tissue-compartment abundances per tumor sample, by centre — Centre A vs Centre B\n'
              '(each panel independently sorted by its own carcinoma fraction, high → low; bar width constant, so Centre B is narrower)',
              fontsize=12,fontweight='bold',y=1.02,color=INK)
fig2.tight_layout(rect=[0,0,0.9,1])
fig2.legend(handles=[Patch(fc=CCOL[k],label=CLAB[k]) for k in COMP],loc='center left',bbox_to_anchor=(0.91,0.73),fontsize=8.4,frameon=False,title='Compartment',title_fontsize=8.8)
fig2.legend(handles=[Patch(fc=CTCOL[k],label=CLAB_CELL[k]) for k in CT],loc='center left',bbox_to_anchor=(0.91,0.24),fontsize=8.4,frameon=False,title='Cell type',title_fontsize=8.8)
fig2.savefig(LUNG/'lung_stackbar_landscape_bycentre.png',bbox_inches='tight',dpi=200); plt.close(fig2)
print(f'merged n={len(compM)} · Centre A={(compM.city=="Centre A").sum()} · Centre B={(compM.city=="Centre B").sum()}')
print('wrote lung_stackbar_compartments.png + lung_stackbar_cells.png (dpi300) + landscape.png + bycentre.png')
