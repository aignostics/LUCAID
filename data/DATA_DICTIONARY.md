# Data dictionary

All tables are **fully anonymized**: no case, slide, spot or patient identifiers,
no dates, no institution names, no free text, and no molecular data beyond the
single KRAS allele frequency used in Figure 5. Every score is on the scale a
pathologist reports it (percent for cellularity and PD-L1 TPS; 0–300 for
H-scores). Missing scores are blank (a rater did not score that case).

`case_id` is a sequential surrogate. The **five pathologists are coded P1–P5**
consistently across every prospective table (the same `P3` is the same person in
each file), so the case-level grid in Figure 8e is a genuine per-case view. The
retrospective KRAS table (`cellularity_vs_kras.csv`) uses its own case numbering.

## Prospective clinical-validation tables (Figure 8, Supplementary Figure 4)

One row per case; `reference` is the expert-panel adjudicated consensus. All
prospective analyses are restricted to the common **70-case cohort** (cases scored
for at least one IHC marker); the tables themselves are unfiltered and the
restriction is applied in `common.py` (`prospective_cohort`).

| Column | Meaning |
|---|---|
| `case_id` | surrogate case identifier (shared across the five task files) |
| `LUCAID` | LUCAID model score for the task |
| `reference` | adjudicated pathologist consensus (the reference standard) |
| `P1`–`P5` | the five individual pathologists' scores |

Files and their score, with units:

| File | Task | Units |
|---|---|---|
| `prospective_cellularity.csv` | tumour cellularity | % carcinoma |
| `prospective_pdl1_tps.csv` | PD-L1 tumour proportion score | % (TPS) |
| `prospective_met_hscore.csv` | c-MET H-score | 0–300 |
| `prospective_trop2_membrane.csv` | TROP-2 membranous H-score | 0–300 |
| `prospective_trop2_cytoplasm.csv` | TROP-2 cytoplasmic H-score | 0–300 |

`prospective_cellularity.csv` has two extra columns:

| Column | Meaning |
|---|---|
| `LUCAID_area` | LUCAID nuclear-**area**-based cellularity (the default `LUCAID` is cell-**count** based) |
| `kras_vaf` | KRAS variant allele frequency (%), where molecularly tested — used for the cellularity-vs-KRAS arm of Supplementary Figure 4b (tumour cell fraction = 2 × VAF; cases with 2 × VAF > 100 % excluded, as in Figure 5) |

## Retrospective KRAS cohort (Figure 5f–h)

`cellularity_vs_kras.csv` — one row per case; a separate cohort from the
prospective tables.

| Column | Meaning |
|---|---|
| `case_id` | surrogate case identifier (`kras_###`) |
| `kras_vaf` | KRAS variant allele frequency (%); the molecular reference is the estimated tumour cell fraction = 2 × VAF |
| `patho_routine` | routine (rater-independent) pathologist cellularity estimate — Figure 5f |
| `LUCAID_count` | LUCAID cell-count-based cellularity — Figure 5g |
| `LUCAID_area` | LUCAID nuclear-area-based cellularity — Figure 5h |
| `P1`–`P5` | the five study pathologists' cellularity estimates (for completeness) |

Figure 5f–h uses the matched cohort (a routine estimate and both LUCAID
estimates present) after dropping cases with 2 × VAF > 100 % (VAF > 50 %,
which breaks the heterozygous-diploid assumption): **n = 115**.

## Regression grading (Figure 6e/f)

`regression_grading.csv` — one row per resection case (n = 140). Compartment
fractions (0–1) for LUCAID and the joint pathologist assessment.

| Column | Meaning |
|---|---|
| `case_id` | surrogate case identifier |
| `Carcinoma_LUCAID_%`, `Necrosis_LUCAID_%`, `Stroma_LUCAID_%` | LUCAID compartment fractions (sum ≈ 1) |
| `Carcinoma_pathologist_%`, `Necrosis_pathologist_%`, `Stroma_pathologist_%` | pathologist compartment fractions (independent coarse estimates, so the three occasionally sum slightly above 1) |
