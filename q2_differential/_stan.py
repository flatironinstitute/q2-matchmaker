import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
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


def _case_control_func(counts : np.array, case_ctrl_ids : np.array,
                       case_member : np.array,
                       depth : int,
                       mc_samples : int=1000) -> dict:
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)

    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/nb_case_control.stan')
    sm = CmdStanModel(stan_file=code)
    dat = {
        'N' : len(counts),
        'C' : int(max(case_ids) + 1),
        'depth' : list(np.log(depth)),
        'y' : list(map(int, counts.astype(np.int64))),
        'cc_bool' : list(map(int, case_member)),
        'cc_ids' : list(map(int, case_ids + 1))
    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        # see https://mattocci27.github.io/assets/poilog.html
        # for recommended parameters for poisson log normal
        fit = sm.sample(data=data_path, iter_sampling=mc_samples, chains=4,
                        iter_warmup=mc_samples // 2,
                        adapt_delta = 0.9, max_treedepth = 20)
        fit.diagnose()
        mu = fit.stan_variable('mu')
        sigma = fit.stan_variable('sigma')
        disp = fit.stan_variable('disp')
        res = pd.DataFrame({
            'mu': mu,
            'sigma': sigma,
            'disp_ctrl': disp[:, 0],
            'disp_case': disp[:, 1]
        })
        # TODO: this doesn't seem to work atm, but its fixed upstream
        # res = fit.summary()
        return res


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
        p1 = np.random.dirichlet(ilr_inv(r1))
        p2 = np.random.dirichlet(ilr_inv(r2))
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
    return table, md
