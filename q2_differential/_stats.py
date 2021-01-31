from scipy.stats import f as f_distrib
from scipy.spatial.distance import pdist, euclidean
import numpy as np


def hotelling_ttest(X):
    """ Tests if table is centered around zero.

    Parameters
    ----------
    X : pd.DataFrame
       Rows are posterior draws, columns are features,

    Returns
    -------
    t2 : np.float32
       T2 statistic
    pval : np.float32
       P-value
    """
    # convert table to ALR coordinates
    X_ = X - X[:, 0].reshape(-1, 1)
    X_ = X_[:, 1:]
    muX = X_.mean(axis=0)
    nx, p = X_.shape
    covX = np.cov(X_.T)
    invcovX = np.linalg.pinv(covX)
    t2 = muX @ invcovX @ muX.T
    stat = (t2 * (nx - p) / ((nx - 1) * p))
    pval = 1 - np.squeeze(f_distrib.cdf(stat, p, nx - p))
    return t2, pval


def spherical_test(X):
    """ Fits a sphere that contains all of the points in X
    and tests to see if 0 is inside of that sphere.

    Parameters
    ----------
    X : pd.DataFrame
       Rows are posterior draws, columns are features,

    Returns
    -------
    True if zero is inside of sphere, False if not.
    """
    X_ = X - X[:, 0].reshape(-1, 1)
    X_ = X_[:, 1:]
    muX = X_.mean(axis=0)
    dists = pdist(X_)
    r = np.max(dists)   # radius of sphere
    p = np.zeros_like(muX)
    d = euclidean(muX, p)
    return d < r


