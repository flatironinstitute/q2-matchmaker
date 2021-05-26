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
import dask_jobqueue
from dask.distributed import Client
from dask.distributed import wait
import tempfile
import json
import arviz as az
import biom
import dask
import time
from birdman import BaseModel
from typing import List, Sequence


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
    md.index.name = 'sample id'
    return table, md, diff


def _case_control_full(counts : np.array,
                       case_ctrl_ids : np.array,
                       case_member : np.array,
                       depth : int,
                       mc_samples : int=1000,
                       seed : int = None) -> dict:
    dat = _case_control_data(counts, case_ctrl_ids,
                             case_member, depth)
    #initialization for controls
    init_ctrl = alr(multiplicative_replacement(
        closure(counts[~(case_member).astype(np.bool)])))
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


def _case_control_single(counts : np.array, case_ctrl_ids : np.array,
                         case_member : np.array,
                         depth : int,
                         mc_samples : int=1000,
                         chains : int=1) -> dict:
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)

    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/nb_case_control_single.stan')
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
        fit = sm.sample(data=data_path, iter_sampling=mc_samples,
                        chains=chains, iter_warmup=mc_samples // 2,
                        adapt_delta = 0.9, max_treedepth = 20)
        fit.diagnose()
        mu = fit.stan_variable('mu')
        sigma = fit.stan_variable('sigma')
        diff = fit.stan_variable('diff')
        disp = fit.stan_variable('disp')
        res = pd.DataFrame({
            'mu': mu,
            'sigma': sigma,
            'diff' : diff,
            'disp_ctrl': disp[:, 0],
            'disp_case': disp[:, 1]
        })
        # TODO: this doesn't seem to work atm, but its fixed upstream
        # res = fit.summary()
        return res


class NegativeBinomialCaseControl(BaseModel):
    """Fit count data with case-control design with negative binomial model.

    Parameters:
    -----------
    table: biom.table.Table
        Feature table (features x samples)
    status_column : str
        Column that specifies `status` of interest, usually the
        experimental condition of interest.
    matching_column : str
        Column that specifies case-control matchings
    metadata: pd.DataFrame
        Metadata for matching and status covariates.
    num_iter: int
        Number of posterior sample draws, defaults to 1000
    num_warmup: int
        Number of posterior draws used for warmup, defaults to 500
    chains: int
        Number of chains to use in MCMC, defaults to 4
    seed: float
        Random seed to use for sampling, defaults to 42
    mu_scale : float
        Standard deviation for prior distribution for mu
    sigma_scale : float
        Standard deviation for prior distribution for sigma
    disp_scale : float
        Standard deviation for prior distribution for disp
    control_scale : float
        Standard deviation for prior distribution for control

    Notes
    -----
    The default priors are choosen for amplicon data.
    Make sure to adjust for other omics datasets.
    """
    def __init__(self,
                 table: biom.table.Table,
                 status_column : str,
                 reference_status : str,
                 matching_column : str,
                 metadata: pd.DataFrame,
                 num_iter: int = 1000,
                 num_warmup: int = 500,
                 adapt_delta: float = 0.9,
                 max_treedepth: float = 20,
                 chains: int = 4,
                 seed: float = 42,
                 mu_scale: float = 10,
                 sigma_scale: float = 1,
                 disp_scale: float = 1,
                 control_loc: float = -5,
                 control_scale: float = 3):
        model_path = os.path.join(
            os.path.dirname(__file__),
            'assets/nb_case_control_single.stan')
        super(NegativeBinomialCaseControl, self).__init__(
            table, metadata, model_path,
            num_iter, num_warmup, chains, seed,
            parallelize_across = "features")
        case_ctrl_ids = metadata[matching_column]
        status = metadata[status_column]
        case_member = (status == reference_status).astype(np.int64)
        case_encoder = LabelEncoder()
        case_encoder.fit(case_ctrl_ids)
        case_ids = case_encoder.transform(case_ctrl_ids)
        self.status_names = metadata[status_column].value_counts().index
        self.param_names = ["mu", "sigma", "disp", "diff", "control"]
        self.dat = {
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0],                 # number of features
            "N": table.shape[1],                 # number of samples
            'C' : int(max(case_ids) + 1),        # number of controls
            'depth' : np.log(table.sum(axis='sample')),
            'cc_bool' : list(map(int, case_member)),
            'cc_ids' : list(map(int, case_ids + 1))
        }
        param_dict = {
            "mu_scale": mu_scale,
            "sigma_scale": sigma_scale,
            "disp_scale": disp_scale,
            "control_loc": control_loc,
            "control_scale": control_scale
        }
        self.add_parameters(param_dict)
        self.specify_model(
            params=["mu", "sigma", "disp", "diff", "control"],
            dims={
                "mu": ["feature"],
                "sigma": ["feature"],
                "diff": ["feature"],
                "disp": ["feature", "covariate"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "convariate": ['control', 'case'],
                "feature": self.feature_names,
                "tbl_sample": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )


def _case_control_data(counts : np.array, case_ctrl_ids : np.array,
                       case_member : np.array,
                       depth : int):
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)
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
