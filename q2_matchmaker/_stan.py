import os
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
from cmdstanpy import CmdStanModel, CmdStanMCMC
from sklearn.preprocessing import LabelEncoder
import tempfile
import json
import arviz as az
import biom



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


def _case_control_normal_sim(n=100, d=10):
    """ Simulate case-controls from positive-bounded Normal distribution

    Parameters
    ----------
    n : int
       Number of samples (must be divisible by 2).
    d : int
       Number of microbes

    Returns
    -------
    table : pd.DataFrame
        Simulated counts
    md : pd.DataFrame
        Simulated metadata
    """
    offset = 100    # to make sure it is strictly positive
    noise = 0.1
    diff = np.random.randn(d)
    ref = np.random.randn(d) + offset
    table = np.zeros((n, d))
    diff_md = np.zeros(n)
    rep_md = np.zeros(n)
    for i in range(n // 2):
        delta = np.random.randn(d) * noise
        r1 = ref + delta
        r2 = r1 + diff
        p1 = np.random.normal(r1)
        p2 = np.random.normal(r2)
        diff_md[i] = 0
        diff_md[(n // 2) + i] = 1
        rep_md[i] = i
        rep_md[(n // 2) + i] = i
        table[i] = np.random.normal(p1, 1)
        table[(n // 2) + i] = np.random.normal(p2, 1)
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
                       seed : int = None) -> (CmdStanMCMC, az.InferenceData):
    dat = _case_control_data(counts, case_ctrl_ids,
                             case_member, depth)
    #initialization for controls
    init_ctrl = alr(multiplicative_replacement(

        closure(counts[~np.array(dat['cc_bool'])] + 1)))

    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/nb_case_control.stan')
    sm = CmdStanModel(stan_file=code)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        posterior = sm.sample(data=data_path, iter_sampling=mc_samples,
                              chains=4, iter_warmup=mc_samples,
                              inits={'control': init_ctrl},
                              seed = seed,
                              adapt_delta = 0.95, max_treedepth = 20)
        posterior.diagnose()
        return sm, posterior


def _case_control_normal_full(
        counts : np.array,
        case_ctrl_ids : np.array,
        case_member : np.array,
        mc_samples : int=1000,
        seed : int = None,
        mu_scale : float = 100,
        sigma_scale : float = 1,
        disp_scale : float = 1,
        control_loc : float = 100,
        control_scale : float = 100) -> (CmdStanMCMC, az.InferenceData):
    dat = _case_control_data(counts, case_ctrl_ids,
                             case_member)
    dat['mu_scale'] = mu_scale
    dat['sigma_scale'] = sigma_scale
    dat['disp_scale'] = disp_scale
    dat['control_loc'] = control_loc
    dat['control_scale'] = control_scale
    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/normal_case_control.stan')
    sm = CmdStanModel(stan_file=code)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        posterior = sm.sample(data=data_path, iter_sampling=mc_samples,
                              chains=4, iter_warmup=mc_samples,
                              seed = seed,
                              adapt_delta = 0.95, max_treedepth = 20)
        posterior.diagnose()
        return sm, posterior


def _case_control_single(counts : np.array, case_ctrl_ids : np.array,
                         case_member : np.array,
                         depth : int,
                         mu_scale : float=10,
                         control_loc : float=0,
                         control_scale : float=10,
                         mc_samples : int=1000,
                         chains : int=1) -> (CmdStanModel, CmdStanMCMC):
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
        'cc_ids' : list(map(int, case_ids + 1)),
        'mu_scale': mu_scale,
        'sigma_scale': 1,
        'disp_scale': 1,
        'control_loc': control_loc,
        'control_scale': control_scale,

    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        # see https://mattocci27.github.io/assets/poilog.html
        # for recommended parameters for poisson log normal
        fit = sm.sample(data=data_path, iter_sampling=mc_samples,
                        chains=chains, iter_warmup=mc_samples,
                        adapt_delta = 0.9, max_treedepth = 20)
        fit.diagnose()
        inf = az.from_cmdstanpy(fit,
                                posterior_predictive='y_predict',
                                log_likelihood='log_lhood')
        return inf



def _case_control_data(counts : np.array, case_ctrl_ids : np.array,
                       case_member : np.array,
                       depth : int = None):
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)
    dat = {
        'N' : counts.shape[0],
        'D' : counts.shape[1],
        'C' : int(max(case_ids) + 1),
        'y' : counts.astype(int).tolist(),
        'cc_bool' : list(map(int, case_member)),
        'cc_ids' : list(map(int, case_ids + 1))
    }
    if depth is not None:
        dat['depth'] = list(np.log(depth))
    return dat


def merge_inferences(inf_list, log_likelihood, posterior_predictive,
                     coords, concatenation_name='features'):
    group_list = []
    group_list.append(dask.persist(*[x.posterior for x in inf_list]))
    group_list.append(dask.persist(*[x.sample_stats for x in inf_list]))
    if log_likelihood is not None:
        group_list.append(dask.persist(*[x.log_likelihood for x in inf_list]))
    if posterior_predictive is not None:
        group_list.append(
            dask.persist(*[x.posterior_predictive for x in inf_list])
        )

    group_list = dask.compute(*group_list)
    po_ds = xr.concat(group_list[0], concatenation_name)
    ss_ds = xr.concat(group_list[1], concatenation_name)
    group_dict = {"posterior": po_ds, "sample_stats": ss_ds}

    if log_likelihood is not None:
        ll_ds = xr.concat(group_list[2], concatenation_name)
        group_dict["log_likelihood"] = ll_ds
    if posterior_predictive is not None:
        pp_ds = xr.concat(group_list[3], concatenation_name)
        group_dict["posterior_predictive"] = pp_ds

    all_group_inferences = []
    for group in group_dict:
        # Set concatenation dim coords
        group_ds = group_dict[group].assign_coords(
            {concatenation_name: coords[concatenation_name]}
        )

        group_inf = az.InferenceData(**{group: group_ds})  # hacky
        all_group_inferences.append(group_inf)

    return az.concat(*all_group_inferences)
