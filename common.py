"""
Shared analysis helpers for the LUCAID clinical-validation figures
=================================================================
Small, self-contained numeric helpers shared by the Figure 8 and Supplementary
Figure 4 scripts: where the tidy CSVs live, the clinical decision thresholds,
the clinical-action category of a score, mean absolute error with a bootstrap
confidence interval, and the 3-way concordance split against the reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path(__file__).resolve().parent / "data"

# The five prospectively scored tasks, in paper order. `cuts` are the clinical
# decision thresholds (a score's category = how many cuts it is >=), `vmax` the
# axis maximum. LUCAID + five pathologists (P1-P5) score every task.
RATERS = ["LUCAID", "P1", "P2", "P3", "P4", "P5"]

TASKS: Dict[str, dict] = {
    "cellularity":     dict(csv="prospective_cellularity.csv",     label="Cellularity",
                            cuts=[10],       vmax=100),
    "pdl1":            dict(csv="prospective_pdl1_tps.csv",        label="PD-L1 TPS",
                            cuts=[1, 50],    vmax=100),
    "cmet":            dict(csv="prospective_met_hscore.csv",      label="c-MET H-score",
                            cuts=[100, 200], vmax=300),
    "trop2_membrane":  dict(csv="prospective_trop2_membrane.csv",  label="TROP-2 mem.",
                            cuts=[100, 200], vmax=300),
    "trop2_cytoplasm": dict(csv="prospective_trop2_cytoplasm.csv", label="TROP-2 cyt.",
                            cuts=[100, 200], vmax=300),
}


def _load_raw(task_key: str) -> pd.DataFrame:
    """One tidy per-task CSV, indexed by case_id, all scores numeric (unfiltered)."""
    df = pd.read_csv(DATA_DIR / TASKS[task_key]["csv"]).set_index("case_id")
    return df.apply(pd.to_numeric, errors="coerce")


# The prospective validation cohort = every case scored (LUCAID + reference) for at
# least one IHC task (PD-L1, c-MET, TROP-2) -> 70 cases. Every prospective analysis
# (Figure 8, Supplementary Figure 4) is restricted to this common set so all tasks
# share the same patients; cellularity-only cases outside it are dropped.
_COHORT: Optional[set] = None


def prospective_cohort() -> set:
    """Case IDs of the common 70-case prospective cohort (see note above)."""
    global _COHORT
    if _COHORT is None:
        cohort: set = set()
        for tk in ("pdl1", "cmet", "trop2_membrane", "trop2_cytoplasm"):
            df = _load_raw(tk)
            cohort |= set(df.index[df["LUCAID"].notna() & df["reference"].notna()])
        _COHORT = cohort
    return _COHORT


def load_task(task_key: str) -> pd.DataFrame:
    """Load one tidy per-task CSV (case_id index, numeric), restricted to the common
    prospective cohort (prospective_cohort()).

    Columns: LUCAID, reference, P1..P5 (cellularity also LUCAID_area, kras_vaf).
    Row order is preserved from the file so bootstrap resampling is reproducible.
    """
    df = _load_raw(task_key)
    return df[df.index.isin(prospective_cohort())]


def category(value: float, cuts: List[float]) -> int:
    """Clinical-action category of a score: the number of thresholds it is >=."""
    return sum(1 for c in cuts if value >= c)


def annotator_metrics(pred: pd.Series, ref: pd.Series, method: str = "spearman",
                      n_bootstrap: int = 1000, random_state: int = 42) -> Optional[Dict]:
    """Correlation + MAE (with 95% bootstrap CI) between a rater and the reference.

    `method` selects Spearman rho (vs the pathologist
    consensus) or Pearson r (vs the molecular KRAS reference).
    """
    valid = pd.concat([pred, ref], axis=1).dropna()
    if len(valid) < 3:
        return None
    pred_vals = valid.iloc[:, 0]
    ref_vals = valid.iloc[:, 1]

    corr, p_value = (pearsonr if method == "pearson" else spearmanr)(pred_vals, ref_vals)
    mae = float(np.mean(np.abs(pred_vals.values - ref_vals.values)))

    np.random.seed(random_state)
    boot = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(pred_vals), size=len(pred_vals), replace=True)
        boot.append(float(np.mean(np.abs(ref_vals.iloc[idx].values - pred_vals.iloc[idx].values))))

    return {
        "correlation": corr, "p_value": p_value, "method": method, "mae": mae,
        "mae_ci_lower": float(np.percentile(boot, 2.5)),
        "mae_ci_upper": float(np.percentile(boot, 97.5)),
        "n": len(valid),
    }


def concordance_split(pred: pd.Series, ref: pd.Series, cuts: List[float]) -> tuple:
    """Percentages of paired cases that are concordant / off-by-1 / off-by->=2."""
    d = pd.concat([pred, ref], axis=1).dropna()
    dist = [abs(category(a, cuts) - category(b, cuts))
            for a, b in zip(d.iloc[:, 0], d.iloc[:, 1])]
    n = len(dist)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    return (100 * sum(x == 0 for x in dist) / n,
            100 * sum(x == 1 for x in dist) / n,
            100 * sum(x >= 2 for x in dist) / n)
