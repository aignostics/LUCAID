# LUCAID — slide-level correlation and clinical-action analyses

Analysis code and anonymized data to reproduce the slide-level result panels of
the LUCAID study: agreement of LUCAID's automated readouts with molecular and
pathologist reference standards (correlation), and clinical-action concordance
(CAC) against an expert-panel–adjudicated reference standard. Each script
regenerates the panels of one figure from a small, fully anonymized table.

## Figures reproduced

| Script | Figure | What |
|---|---|---|
| `figure5_cellularity_kras.py` | **5f–h** | tumour cellularity vs KRAS VAF (pathologist / LUCAID cell-count / LUCAID nuclear-area), n = 115 |
| `figure6_regression_grading.py` | **6e/f** | tissue-compartment regression grading, LUCAID vs pathologist, n = 140 |
| `figure8_clinical_validation.py` | **8a, c, d, e** | prospective validation: rater-vs-reference calibration, MAE, per-task and case-level clinical-action concordance |
| `supplementary_figure4.py` | **S4a, b** | inter-rater correlation matrices; MAE (95% CI) + correlation by task |
| `figure6_survival/` | **6g–n** | spatial-TME prognostic analysis — code and input-table schema only (see [Figure 6g–n](#figure-6gn-spatial-tme-survival)) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python figure5_cellularity_kras.py
python figure6_regression_grading.py
python figure8_clinical_validation.py
python supplementary_figure4.py
```

Each script reads its table from `data/`, writes PNG + PDF + SVG panels (and a
stats CSV) to `figures/`, and prints the key numbers. The four scripts above need
only `numpy`, `pandas`, `scipy`, `matplotlib` and `pillow`. Tested with Python 3.11.

## Layout

```
lucaid-analysis/
├── figure5_cellularity_kras.py       # Fig 5f-h
├── figure6_regression_grading.py     # Fig 6e/f
├── figure8_clinical_validation.py    # Fig 8a,c,d,e
├── supplementary_figure4.py          # Suppl Fig 4a,b
├── style.py                          # shared figure style (palette, export)
├── common.py                         # thresholds, categories, MAE + bootstrap CI
├── data/                             # anonymized inputs (+ DATA_DICTIONARY.md)
├── figures/                          # generated panels
└── figure6_survival/                 # Fig 6g-n spatial-TME survival analysis
```

## Data and anonymization

All inputs live in `data/` and are documented column-by-column in
[`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md). Every table is fully
anonymized: no case, slide, spot or patient identifiers, no dates, no institution
names, no free text, and no molecular data beyond the single KRAS allele
frequency used in Figure 5. Cases carry sequential surrogate IDs; the five
pathologists are coded P1–P5 (consistently across the prospective tables) and the
LUCAID readout is labelled `LUCAID`. The tables are pre-joined, so no
identifier-bearing keys are needed to run the analyses.

## Statistics

Correlation with the pathologist consensus is reported as **Spearman ρ**;
correlation with the molecular KRAS reference as **Pearson r** — in both the
retrospective Figure 5f–h (n = 115) and the prospective Supplementary Figure 4b
KRAS column. Both KRAS analyses exclude cases with 2 × VAF > 100 % (VAF > 50 %,
which break the heterozygous-diploid assumption). All correlations are two-sided.
Mean absolute error is reported with a **95% bootstrap confidence interval**
(1,000 resamples; fixed random seed for reproducibility). Clinical-action
concordance uses the study thresholds: cellularity < 10 % vs ≥ 10 %; PD-L1 TPS
< 1 %, 1–49 %, ≥ 50 %; MET and TROP-2 H-scores < 100, 100–199, ≥ 200 (see
`common.py`).

All prospective analyses (Figure 8 and Supplementary Figure 4) are restricted to the
common **70-case prospective cohort** — the cases scored for at least one IHC marker
— so every task is evaluated on the same patients.

## Figure 6g–n (spatial-TME survival)

The Figure 6g–n panels (compartment/cell-type composition, UICC-adjusted hazard
forest, Kaplan–Meier curves, and the PD-L1 × TIL quadrant) use per-patient data
from an external NSCLC validation cohort — spatial tumour-microenvironment
features together with survival endpoints. That patient-level data is not
redistributed here, so [`figure6_survival/`](figure6_survival/) provides the
analysis code and the required input-table schema; see
[`figure6_survival/README.md`](figure6_survival/README.md) for how to run it
against that table.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) — free
for non-commercial academic research use with attribution; commercial use is not
permitted. © 2026 Aignostics GmbH. See [`LICENSE`](LICENSE).

## Citation

If you use this code or data, please cite the LUCAID paper (citation to be added
on publication).
