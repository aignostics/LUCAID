"""Canonical high/low split shared by the forest and every Kaplan–Meier panel, so the Cox cut
and the plotted KM cut are always identical. Adaptive rule, returns (threshold, rule):
  gmm    — genuinely bimodal (BIC prefers k>=2; interior antimode >=20% below both modes): cut at the trough.
  peak   — boundary-inflated (>=10% of mass at the exact min or max): cut at the KDE peak.
  median — otherwise, or when a gmm/peak cut would leave a group smaller than min_frac.
"""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


def _balanced(x, thr, min_frac):
    f = float(np.mean(x > thr))
    return min_frac <= f <= 1 - min_frac


def _gmm_antimode(x, min_frac, q03, q97):
    xr = x.reshape(-1, 1)
    models = {k: GaussianMixture(k, random_state=0, n_init=4).fit(xr) for k in (1, 2, 3)}
    bic = {k: m.bic(xr) for k, m in models.items()}
    kbest = min(bic, key=bic.get)
    if kbest < 2 or bic[1] - bic[kbest] <= 10:
        return None
    grid = np.linspace(x.min(), x.max(), 2001)
    dens = np.exp(models[kbest].score_samples(grid.reshape(-1, 1)))

    def real_dip(i):
        lmax = dens[:i].max() if i > 0 else 0
        rmax = dens[i + 1:].max() if i < len(dens) - 1 else 0
        return dens[i] < 0.80 * min(lmax, rmax)

    valleys = [i for i in range(1, len(grid) - 1)
               if dens[i] < dens[i - 1] and dens[i] <= dens[i + 1] and q03 < grid[i] < q97
               and _balanced(x, grid[i], min_frac) and real_dip(i)]
    valleys.sort(key=lambda i: dens[i])
    return float(grid[valleys[0]]) if valleys else None


def smart_split(v, min_n=30, min_frac=0.15):
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if len(x) < min_n or np.std(x) == 0:
        return (float(np.median(x)) if len(x) else np.nan), 'median'

    q03, q97 = np.quantile(x, .03), np.quantile(x, .97)
    if len(np.unique(x)) >= 10:
        cut = _gmm_antimode(x, min_frac, q03, q97)
        if cut is not None:
            return cut, 'gmm'

    inflated = np.mean(x == x.min()) >= 0.10 or np.mean(x == x.max()) >= 0.10
    if inflated:
        xs = np.linspace(np.quantile(x, .02), np.quantile(x, .98), 256)
        pk = float(xs[np.argmax(gaussian_kde(x)(xs))])
        if q03 < pk < q97 and _balanced(x, pk, min_frac):
            return pk, 'peak'

    return float(np.median(x)), 'median'
