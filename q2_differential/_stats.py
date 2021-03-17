from scipy.stats import f as f_distrib
from scipy.spatial.distance import pdist, euclidean
import numpy as np


def hotelling_ttest(X : np.array, to_alr=False):
    """ Tests if table is centered around zero.

    Parameters
    ----------
    X : np.array
       Rows are posterior draws, columns are features,

    Returns
    -------
    t2 : np.float32
       T2 statistic
    pval : np.float32
       P-value from chi-square distribution.

    Notes
    -----
    It is a strict requirement that n > p.
    """
    # convert table to ALR coordinates
    if to_alr:
        X_ = X - X[:, 0].reshape(-1, 1)
        X_ = X_[:, 1:]
    else:
        X_ = X
    muX = X_.mean(axis=0)
    nx, p = X_.shape
    if nx < p :
        raise ValueError(f'{nx} < {p}, need more samples.')
    covX = np.cov(X_.T)
    invcovX = np.linalg.pinv(covX)
    t2 = muX @ invcovX @ muX.T
    stat = (t2 * (nx - p) / ((nx - 1) * p))
    npval = np.squeeze(f_distrib.cdf(stat, p, nx - p))
    pval = 1 - npval
    return t2, pval


def spherical_test(X : np.array):
    """ Fits a sphere that contains all of the points in X
    and tests to see if 0 is inside of that sphere.

    Parameters
    ----------
    X : np.array
       Rows are posterior draws, columns are features,

    Returns
    -------
    True if zero is inside of sphere, False if not.
    """
    X_ = X - X[:, 0].reshape(-1, 1)
    X_ = X_[:, 1:]
    muX = X_.mean(axis=0)
    dists = pdist(X_)
    r = np.max(dists) / 2   # radius of sphere
    p = np.zeros_like(muX)
    d = euclidean(muX, p)
    return d < r, r, d


def rank_test(X : np.array):
    """ Computes a cumulative rank test.

    This computes the probability of P(x > X) where x
    is the feature of interest, and X are all of the features
    that have a mean rank less than x.
    """
    X_ = X - X[:, 0].reshape(-1, 1)
    X_ = X_[:, 1:]
    muX = X_.mean(axis=0)
    idx = np.argsort(muX)
    X_ = X_[:, idx]
    d = len(muX)
    n = len(X_)
    pval = np.zeros(d)
    for i in range(d):
        T = 0
        for j in range(n):
            T += np.sum(X[j, i] > X[j, i + 1:])
        p = (T + 1) / (n * (d - i) + 1)
        pval[i] = p
    return idx, pval
