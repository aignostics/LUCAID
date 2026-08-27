#!/usr/bin/env python3
"""Freeze the per-patient analysis landscape read by the survival analyses (critical path).

Writes lung_patient_landscape_ext.csv — the per-patient landscape of established + composition +
clinical + survival columns that forest_uicc_analysis.py and individual_km_analysis.py
read. The 7 novel spatial features shown in panel E are aggregated per-feature
DOWNSTREAM in those scripts (directly from the curator CSVs), so they are not appended here.
"""
import os
from pathlib import Path
import pandas as pd

LUNG = Path(os.environ.get('LUNG_DATA', 'lung_data'))
U = lambda s: s.astype(str).str.upper().str.strip()
land = pd.read_csv(LUNG / 'lung_patient_landscape.csv'); land['CASE'] = U(land.CASE)
land.to_csv(LUNG / 'lung_patient_landscape_ext.csv', index=False)
print(f'wrote lung_patient_landscape_ext.csv ({len(land)} patients · {land.shape[1]} cols)')
