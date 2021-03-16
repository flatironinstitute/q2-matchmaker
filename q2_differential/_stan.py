import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from skbio.stats.composition import (clr, closure, alr, alr_inv,
                                     multiplicative_replacement)
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from skbio.stats.composition import ilr_inv
import matplotlib.pyplot as plt
import pickle
from cmdstanpy import CmdStanModel
from sklearn.preprocessing import LabelEncoder
import tempfile
import json
import arviz as az


def _case_control_data(counts : np.array, case_ctrl_ids : np.array,
                       case_member : np.array,
                       depth : int,
                       reference : str):
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)
    #initialization for controls
    idx = ~(case_member==reference).astype(np.bool)
    dat = {
        'N' : counts.shape[0],
        'D' : counts.shape[1],
        'C' : int(max(case_ids) + 1),
        'depth' : list(np.log(depth)),
        'y' : counts.astype(int).tolist(),
        'cc_bool' : list(map(int, case_member)),
        'cc_ids' : list(map(int, case_ids + 1))
    }
    return dat

def _case_control_full(counts : np.array, case_ctrl_ids : np.array,
                       case_member : np.array,
                       depth : int,
                       reference : str,
                       mc_samples : int=1000,
                       seed : int = None) -> dict:
    dat = _case_control_data(counts, case_ctrl_ids,
                             case_member, depth, reference)
    #initialization for controls
    idx = ~(case_member==reference).astype(np.bool)
    init_ctrl = alr(multiplicative_replacement(
        closure(counts[idx])))
    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/nb_case_control.stan')
    sm = CmdStanModel(stan_file=code)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        # see https://mattocci27.github.io/assets/poilog.html
        # for recommended parameters for poisson log normal
        prior = sm.sample(data=data_path, iter_sampling=100, chains=4,
                          iter_warmup=1)
        posterior = sm.sample(data=data_path, iter_sampling=mc_samples,
                              chains=4, iter_warmup=mc_samples // 2,
                              inits={'control': init_ctrl}, seed = seed,
                              adapt_delta = 0.95, max_treedepth = 20)
        posterior.diagnose()
        return sm, posterior, prior


def _case_control_sim(n=100, d=10, depth=50):
    """ Simulate case-controls from Multinomial distribution

    Parameters
    ----------
    n : int
       Number of samples (must be divisible by 2).
    d : int
       Number of microbes
    depth : int
       Sequencing depth

    Returns
    -------
    table : pd.DataFrame
        Simulated counts
    md : pd.DataFrame
        Simulated metadata
    """
    noise = 0.1
    diff = np.random.randn(d - 1)
    ref = np.random.randn(d - 1)
    table = np.zeros((n, d))
    batch_md = np.zeros(n)
    diff_md = np.zeros(n)
    rep_md = np.zeros(n)
    for i in range(n // 2):
        delta = np.random.randn(d - 1) * noise
        N = np.random.poisson(depth)
        r1 = ref + delta
        r2 = r1 + diff
        p1 = np.random.dirichlet(alr_inv(r1))
        p2 = np.random.dirichlet(alr_inv(r2))
        diff_md[i] = 0
        diff_md[(n // 2) + i] = 1
        rep_md[i] = i
        rep_md[(n // 2) + i] = i
        table[i] = np.random.multinomial(N, p1)
        table[(n // 2) + i] = np.random.multinomial(N, p2)
    oids = [f'o{x}' for x in range(d)]
    sids = [f's{x}' for x in range(n)]
    table = pd.DataFrame(table, index=sids, columns=oids)
    md = pd.DataFrame({'diff': diff_md.astype(np.int64).astype(np.str),
                       'reps': rep_md.astype(np.int64).astype(np.str)},
                      index=sids)
    md.index.name = 'sampleid'
    return table, md, diff
