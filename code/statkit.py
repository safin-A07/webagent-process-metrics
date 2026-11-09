"""Statistical helpers (pre-registered in notes/preregistration.md). Seed 42 everywhere."""
from __future__ import annotations

import numpy as np
from scipy import stats as st

SEED = 42
B = 1000


def rng(offset=0):
    return np.random.default_rng(SEED + offset)


def cluster_weights(n_clusters: int, b: int = B, offset: int = 0) -> np.ndarray:
    """Bootstrap resample of clusters as multiplicity weights, shape (b, n_clusters)."""
    r = rng(offset)
    idx = r.integers(0, n_clusters, size=(b, n_clusters))
    w = np.zeros((b, n_clusters))
    for i in range(b):
        np.add.at(w[i], idx[i], 1)
    return w


def pct_ci(samples, alpha=0.05):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2)))


def kendall_b(x, y):
    t = st.kendalltau(x, y, variant="b")
    return float(t.statistic)


def paired_permutation(d, n_perm=10000, offset=0):
    """Two-sided sign-flip permutation test for mean of paired differences d."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0 or np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    r = rng(offset)
    signs = r.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(perm >= obs - 1e-12)) / (n_perm + 1))


def hodges_lehmann(d):
    """One-sample Hodges-Lehmann estimator: median of Walsh averages."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def tost_paired_binary(x1, x2, margin=0.10, alpha=0.05):
    """TOST for equivalence of two paired success rates (normal approximation on paired differences).
    Returns (diff, p_tost); equivalence if p_tost < alpha."""
    d = np.asarray(x1, float) - np.asarray(x2, float)
    n = len(d)
    diff = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
    if se == 0:
        return float(diff), (0.0 if abs(diff) < margin else 1.0)
    p_lower = 1 - st.norm.cdf((diff + margin) / se)
    p_upper = st.norm.cdf((diff - margin) / se)
    return float(diff), float(max(p_lower, p_upper))


def mcnemar_exact(x1, x2):
    x1, x2 = np.asarray(x1, int), np.asarray(x2, int)
    b = int(((x1 == 1) & (x2 == 0)).sum())
    c = int(((x1 == 0) & (x2 == 1)).sum())
    if b + c == 0:
        return 1.0
    return float(st.binomtest(b, b + c, 0.5).pvalue)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(st.spearmanr(x[ok], y[ok]).statistic)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank-based partial correlation)."""
    x, y, z = (st.rankdata(np.asarray(v, float)) for v in (x, y, z))
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))
