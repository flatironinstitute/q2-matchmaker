from scipy.stats import f as f_distrib
from scipy.stats import ttest_1samp
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
import numpy as np


def hotelling_ttest(X: np.array, to_alr=False):
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
    if nx < p:
        raise ValueError(f'{nx} < {p}, need more samples.')
    covX = np.cov(X_.T)
    invcovX = np.linalg.pinv(covX)
    t2 = muX @ invcovX @ muX.T
    stat = (t2 * (nx - p) / ((nx - 1) * p))
    npval = np.squeeze(f_distrib.cdf(stat, p, nx - p))
    pval = 1 - npval
    return t2, pval


def spherical_test(X: np.array, p=0.95, center=True, median=False, radius=False):

    """ Fits a sphere that contains all of the points in X
    and tests to see if 0 is inside of that sphere.

    Parameters
    ----------
    X : np.array
       Rows are posterior draws, columns are features,
    p : float
       Confidence interval
    center : bool
       Recenter the data (i.e. CLR transform)
    median : bool
       Compute medoid. If false, then the mean is computed instead

    Returns
    -------
    True if zero is inside of sphere, False if not.
    """
    if center:
        X_ = X - X.mean(axis=1).reshape(-1, 1)
    else:
        X_ = X

    muX = X_.mean(axis=0).reshape(1, -1)

    dists = cdist(X_, muX)
    r = np.percentile(dists, p)   # radius of sphere
    p = np.zeros_like(muX)
    d = euclidean(muX, p)
    return d < r, r, d


def effect_size(x: pd.DataFrame) -> pd.DataFrame:
    """ aldex2 style estimate of effect size. """
    y = x - x.mean(axis=0)   # CLR transform posterior
    ym, ys = y.mean(axis=1), y.var(axis=1, ddof=1)
    ye = ym / ys
    diffs = x.copy()
    diffs['effect_size'] = ye
    diffs['effect_std'] = 1 / ys
    # Compute effect size pvalues
    tt, pvals = ttest_1samp(y.values, popmean=0, axis=1)
    diffs['tstat'] = tt
    diffs['pvalue'] = pvals
    return diffs


def logodds_ranking(x: pd.DataFrame) -> pd.Series:
    """ Computes log p(max) / p(min) from posterior distribution. """
    b = x.apply(np.argmin, axis=0)
    t = x.apply(np.argmax, axis=0)
    countb = b.value_counts()
    countt = t.value_counts()
    countb.index = x.index[countb.index]
    countt.index = x.index[countt.index]
    countb.name = 'counts_bot'
    countt.name = 'counts_top'
    diffs = pd.merge(x, countb,
                     left_index=True, right_index=True, how='left')
    diffs = pd.merge(x, countt,
                     left_index=True, right_index=True, how='left')
    diffs = diffs.fillna(0)
    ct, cb = diffs['counts_top'], diffs['counts_bot']
    diffs['prob_top'] = (ct + 1) / (ct + 1).sum()
    diffs['prob_bot'] = (cb + 1) / (cb + 1).sum()
    diffs['prob_lr'] = diffs.apply(
        lambda x: np.log(x['prob_top'] / x['prob_bot']), axis=1)
    diffs = diffs.replace([np.inf, -np.inf, np.nan], 0)
    return diffs['prob_lr']
