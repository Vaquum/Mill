"""Request-local factorization of exact QC pair prerequisites.

Pairwise missing-value masks and sample positions remain pair-specific. Columns
are prepared once per identical retained-row cohort, within a byte budget.
The joint KNN search remains per pair; marginal radius counts use sorted 1-D
columns and the same Chebyshev comparison as sklearn's KDTree.
"""
from __future__ import annotations

import warnings

import numpy as np
from numba import njit
from scipy import stats
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale

from .memo import ByteLRU
from .settings import memory_budget


@njit(cache=True)
def radius_counts(values, ordered, radii):
    counts = np.empty(values.size, dtype=np.int64)
    for i in range(values.size):
        value, radius = values[i], radii[i]
        lo, hi = 0, ordered.size
        while lo < hi:
            mid = (lo + hi) // 2
            if ordered[mid] < value and value - ordered[mid] > radius:
                lo = mid + 1
            else:
                hi = mid
        first = lo
        lo, hi = 0, ordered.size
        while lo < hi:
            mid = (lo + hi) // 2
            if ordered[mid] > value and ordered[mid] - value > radius:
                hi = mid
            else:
                lo = mid + 1
        counts[i] = lo - first
    return counts


def _weight(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, tuple):
        return sum(_weight(v) for v in value)
    return 64


class Workspace:
    def __init__(self, arrays, *, max_n, mi_max_n, min_n):
        self.arrays = arrays
        self.n = next(iter(arrays.values())).size if arrays else 0
        self.masks = {name: np.packbits(~np.isnan(values)) for name, values in arrays.items()}
        self.max_n, self.mi_max_n, self.min_n = max_n, mi_max_n, min_n
        # Retained preparations, masks and indices are request-local, never
        # keyed by array identity across requests. Keys include exact mask bytes.
        self.prepared = ByteLRU(memory_budget("pair_work", 64), lambda entry: _weight(entry[0]) + entry[1])

    def _get(self, key, build):
        entry = self.prepared.get(key)
        if entry is not None:
            return entry[0]
        hit = build()
        self.prepared[key] = (hit, 256 + sum(len(k) for k in key if isinstance(k, bytes)))
        return hit

    def _indices(self, cohort, cap):
        def build():
            rows = np.flatnonzero(np.unpackbits(np.frombuffer(cohort, dtype=np.uint8), count=self.n))
            if rows.size > cap:
                rows = rows[np.linspace(0, rows.size - 1, cap).astype(np.int64)]
            return rows
        return self._get(("indices", cohort, cap), build)

    def _ranks(self, name, cohort):
        return self._get(("ranks", name, cohort), lambda: stats.rankdata(
            self.arrays[name][self._indices(cohort, self.max_n)]))

    def _marginal(self, name, cohort, role):
        def build():
            rows = self._indices(cohort, self.mi_max_n)
            values = self.arrays[name][rows]
            if not np.isfinite(values).all():
                return (None, None)
            # Preserve sklearn's one-feature RNG: first N normals for X,
            # next N for y, independent of feature count or pair traversal.
            def noise_vectors():
                rng = np.random.RandomState(0)
                return rng.standard_normal(values.size), rng.standard_normal(values.size)
            noises = self._get(("noise", cohort), noise_vectors)
            noise = noises[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if role == "left":
                    values = scale(values[:, None], with_mean=False).ravel()
                    means = np.maximum(1, np.mean(np.abs(values[:, None]), axis=0))[0]
                else:
                    values = scale(values, with_mean=False)
                    means = np.maximum(1, np.mean(np.abs(values)))
                    noise = noises[1]
            values += 1e-10 * means * noise
            return values, np.sort(values)
        return self._get(("marginal", name, cohort, role), build)

    def pair(self, left, right):
        cohort = np.bitwise_and(self.masks[left], self.masks[right]).tobytes()
        rows = self._indices(cohort, self.max_n)
        if rows.size < self.min_n:
            return None, None
        a, b = self._ranks(left, cohort), self._ranks(right, cohort)
        with np.errstate(invalid="ignore", divide="ignore"):
            # scipy.spearmanr returns [1, 0]; the two normalized covariance
            # entries can differ by one ULP, so preserve that direction too.
            rho = np.corrcoef(a, b)[1, 0]
        rho = abs(float(rho)) if np.isfinite(rho) else None
        x, sx = self._marginal(left, cohort, "left")
        y, sy = self._marginal(right, cohort, "right")
        if x is None or y is None:
            return rho, None
        try:
            nn = NearestNeighbors(metric="chebyshev", n_neighbors=3).fit(np.column_stack((x, y)))
            radius = np.nextafter(nn.kneighbors()[0][:, -1], 0)
            nx, ny = radius_counts(x, sx, radius), radius_counts(y, sy, radius)
            mi = max(0, digamma(x.size) + digamma(3)
                     - np.mean(digamma(nx)) - np.mean(digamma(ny)))
        except Exception:
            return rho, None
        return rho, float(mi) if np.isfinite(mi) else None
