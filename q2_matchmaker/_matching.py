from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import pandas as pd


def _standardize(x):
    return (x - x.min()) / (x.max() - x.min())


def _matchmaker(metadata, status, match_columns, types):
    """ Computes matching ids.

    Parameters
    ----------
    metadata : pd.DataFrame
        Sample metadata
    status : str
        Column for specifying case-control status
    match_columns : list of str
        List of metadata categories
    types : list of bool
        Specifies if it is categorical or not.
        True for categorical, False for continuous

    Returns
    -------
    pd.Series : List of matching ids
    """
    md = metadata.sort_values(by=status)
    dummies = []
    for col, cat in zip(match_columns, types):
        if cat:
            df = pd.get_dummies(md[col])
            dummies.append(df)
        else:
            df = pd.DataFrame(_standardize(md[col]))
            dummies.append(df)
    dm = sum(map(lambda x: squareform(pdist(x)) ** 2, dummies))
    i = (md[status] == md[status][0]).values.sum()
    x, y = linear_sum_assignment(dm[:i, i:])
    y = y + i
    md.loc[md.index[x], 'matching_id'] = x
    md.loc[md.index[y], 'matching_id'] = x
    return md['matching_id']
