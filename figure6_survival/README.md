# Figure 6g–n — spatial-TME prognostic analysis

External NSCLC validation of spatial tumour-microenvironment (TME) prognostic
features: cohort composition landscapes, a UICC-adjusted hazard-ratio forest,
Kaplan–Meier survival curves, and the PD-L1 × TIL immune-phenotype quadrant.

> **Patient-level data not included.** These panels use per-patient data from an
> external NSCLC validation cohort (spatial TME features together with survival
> endpoints), which is not redistributed with this repository. The analysis code
> and the expected input-table schema are provided here so the method is
> transparent and can be rerun against an equivalent per-patient table placed
> under `$LUNG_DATA`.

## Panels → scripts

Script filenames use an earlier panel lettering (C–J); the mapping to the
manuscript's Figure 6 panels **g–n** is:

| Manuscript | What | Script |
|---|---|---|
| **6g, 6h** | compartment / cell-type composition stacked bars | `fig6CD_stackbar_landscape.py` |
| **6i** | per-feature UICC-adjusted Cox forest | `analysis_novel_screen.py` (candidate screen) + `forest_uicc_analysis.py` → `fig6i_forest_single.py` |
| **6j–m** | landscape + Kaplan–Meier for selected features | `individual_km_analysis.py` → `fig6FI_km_plot.py` |
| **6n** | PD-L1 × TIL immune-phenotype quadrant | `fig6J_quadrant.py` |

`lung_split.py` is the shared adaptive high/low split (median, or a Gaussian-mixture
trough for genuinely bimodal markers) used by both the forest and the KM analyses,
so the Cox cut and the plotted KM cut are always identical.

## Dependencies

`pandas numpy scipy scikit-learn lifelines matplotlib openpyxl` (the `openpyxl`
requirement is only for `fig6J_quadrant.py`, which reads a PD-L1 `.xlsx`). Arial is
used if installed and falls back to the default sans-serif otherwise.

## Input tables

All I/O is under `$LUNG_DATA` (defaults to `./lung_data`). The scripts read a small
set of per-patient / per-core tables (no cell-level data):

| File | Content |
|---|---|
| `lung_patient_landscape.csv` | per-patient table (schema below) |
| `lung_manifest_core2case.csv` | `core_uuid` → `case_id` map |
| `lung_spot_master.csv` | per-core `RELATIVE_AREA_*` (compartments) and `CELL_COUNT_*` (cell types) |
| `curator_arms_expanded/delaunay_niches_features.csv` | per-slide niche/adjacency features |
| `curator_arms/cell_distances_full.csv` | per-slide cell–cell distance features |
| `validation_cohorts_lung_metadata …_UUID.csv` | clinical metadata (centre; panels 6g/h only) |
| `PD-L1_case+spotIDs_with_model_TPS.xlsx` | per-spot PD-L1 model TPS (panel 6n only) |

**Per-patient landscape schema** (one row per `CASE`):

- required: `CASE`, `OS_M` (overall survival, **months**), `EVENT` (0/1),
  `stage_num` (UICC stage, numeric);
- 25 established TME markers (`dens_TIL`, `TIL_pct_stromal`, `tNLR`, `LGR`, `LMR`,
  `stroma_pct`, `invasivity`, … — see `forest_uicc_analysis.py:EST`);
- 8 composition fractions (`pct_carcinoma`, `pct_lymphocyte`, `pct_plasma`,
  `dens_plasma`, …).

The 7 novel spatial features (`DEL_*`, `d_*_median`) are aggregated at run time
from the curator-arm tables via the manifest. They are a curated subset of a
586-metric candidate screen (`analysis_novel_screen.py`, Benjamini–Hochberg FDR
across the whole screen → `lung_novel_screen.csv`): four are FDR-significant and
three are nominal-only, chosen for interpretability, spatial-family coverage and
both risk directions. `forest_uicc_analysis.py` merges the screen-wide q back onto
the panel. `analysis_novel_screen.py` additionally reads the other curator-arm
family tables (`curator_arms/niches_kmeans.csv`,
`curator_arms_expanded/{cell_pair_colocalisation,perivascular_barrier,interface_scores}_features.csv`).
Those per-core / per-slide feature tables are, in turn, produced from raw tissue
readouts by a feature-generation step that is upstream of this folder.

Missing values are handled per feature: a patient lacking a given feature is
excluded from that feature's Cox model and log-rank test (a feature-specific `n`,
rather than being forced into the low group).

## Run order

```bash
export LUNG_DATA=/path/to/lung_data        # per-patient tables live here
export PYTHONPATH=$PWD                      # so the scripts can import lung_split

python build_patient_landscape.py          # freeze the per-patient landscape
python analysis_novel_screen.py            # -> lung_novel_screen.csv (586-metric FDR screen)
python forest_uicc_analysis.py             # -> lung_forest_uicc_panel.csv (+ _uicc.csv)
python individual_km_analysis.py           # -> lung_individual_km_metrics.csv
python fig6CD_stackbar_landscape.py        # 6g/6h
python fig6i_forest_single.py              # 6i
python fig6FI_km_plot.py                   # 6j–m
python fig6J_quadrant.py                   # 6n
```
