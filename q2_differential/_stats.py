from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import f as f_distrib
import numpy as np


def hotelling_ttest(X):
    """ Tests if table is centered around zero.

    Parameters
    ----------
    X : pd.DataFrame
       Rows are posterior draws, columns are features,
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


def convex_hull_test(X):
    """ Tests if zero is included within the convex hull.

    Parameters
    ----------
    x : np.array
       Rows are features, columns are posterior draws
    """
    # convert table to ALR coordinates
    X_ = X - X[:, 0].reshape(-1, 1)
    X_ = X_[:, 1:]
    # Construct convex hull
    hull = ConvexHull(X_)
    hull = Delaunay(hull.points)
    # Test if zero is contained within the convex hull
    p = np.zeros(X_.shape[1])
    ans = hull.find_simplex(p) >= 0
    return ans
