import os
import numpy as np
import pandas as pd
from skbio.stats.composition import (closure, alr, alr_inv,
                                     multiplicative_replacement)
from sklearn.preprocessing import LabelEncoder
from cmdstanpy import CmdStanModel, CmdStanMCMC
import tempfile
import json
import xarray as xr
import arviz as az
from scipy.stats import nbinom


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


def negative_binomial_rvs(mu, alpha, state=None):
    """ Uses mean / phi reparameterization of scipy negative binomial"""

    sigma2 = mu + alpha * mu ** 2
    p = mu / sigma2
    n = (mu ** 2) / (sigma2 - mu)
    return nbinom.rvs(n, p, random_state=state)


def _case_control_negative_binomial_sim(n=100, b=2, d=10, depth=50,
                                        disp_scale = 0.1,
                                        batch_scale = 0.1,
                                        diff_scale = 1,
                                        control_loc = None,
                                        control_scale = 0.1,
                                        state=None, params=dict()):
    """ Simulate case-controls from Negative Binomial distribution

    Parameters
    ----------
    n : int
       Number of samples (must be divisible by 2).
    b : int
       Number of batches (must be able to divide n).
    d : int
       Number of microbes
    depth : int
       Sequencing depth
    state : np.random.RandomState or int or None
       Random number generator.
    params : dict
       Dictionary of parameters to initialize simulations

    Returns
    -------
    table : pd.DataFrame
        Simulated counts
    md : pd.DataFrame
        Simulated metadata
    diff : pd.DataFrame
        Ground truth differentials
    """
    if state is None:
        state = np.random.RandomState(0)
    else:
        state = np.random.RandomState(state)

    # dimensionality
    c = n // 2
    # setup scaling parameters
    if control_loc is None:
        control_loc = np.log(1 / d)
    eps = 0.1      # random effects for intercepts
    delta = 0.1    # size of overdispersion
    # setup priors
    a1 = state.normal(eps, eps, size=d)
    diff = params.get('diff', state.normal(0, diff_scale, size=d))
    disp = params.get('disp', state.lognormal(np.log(delta), disp_scale, size=(2, d)))

    batch_mu = params.get('batch_mu', state.normal(0, 1, size=(b, d)))
    batch_disp = params.get('batch_disp', state.lognormal(np.log(delta), batch_scale, size=(b, d)))
    control_mu = params.get('control_mu', state.normal(control_loc, 1, size=(d)))
    control_sigma = params.get('control_sigma', state.lognormal(np.log(delta), control_scale, size=(d)))
    control = np.vstack([state.normal(control_mu, control_sigma) for _ in range(c)])

    depth = np.log(state.poisson(depth, size=n))
    # depth = np.array([np.log(depth)] * n)  # for debugging
    # look up tables
    bs = n // b  # batch size
    batch_ids = np.repeat(np.arange(b), bs)
    batch_ids = np.hstack((
        batch_ids,
        np.array([b - 1] * (n - len(batch_ids)))
    )).astype(np.int64)
    cc_bool = np.arange(n) % 2  # case or control
    cc_ids = np.repeat(np.arange(c), 2)
    y = np.zeros((n, d))
    # model simulation
    for s in range(n):
        for i in range(d):
            # control counts
            lam = depth[s] + batch_mu[batch_ids[s], i] + control[cc_ids[s], i]
            # case counts (if applicable)
            if cc_bool[s] == 1:
                lam += diff[i]
            alpha = (np.exp(a1[i]) / np.exp(lam))
            alpha += disp[cc_bool[s], i]
            alpha += batch_disp[batch_ids[s], i]
            # phi = (1 / alpha)  # stan's parameterization
            nb = negative_binomial_rvs(np.exp(lam), alpha, state)
            y[s, i] = nb
    oids = [f'o{x}' for x in range(d)]
    sids = [f's{x}' for x in range(n)]
    table = pd.DataFrame(y, index=sids, columns=oids)
    md = pd.DataFrame({'cc_bool': cc_bool.astype(np.str),
                       'cc_ids': cc_ids.astype(np.str),
                       'batch_ids': batch_ids.astype(np.str)},
                      index=sids)
    md.index.name = 'sample id'
    return table, md, diff


def _case_control_full(counts: np.array,
                       case_ctrl_ids: np.array,
                       case_member: np.array,
                       depth: int,
                       mc_samples: int = 1000,
                       seed: int = None) -> (CmdStanModel, CmdStanMCMC):
    dat = _case_control_data(counts, case_ctrl_ids,
                             case_member, depth)
    # initialization for controls
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
                              chains=4, iter_warmup=1000,
                              inits={'control': init_ctrl},
                              seed=seed, adapt_delta=0.95,
                              max_treedepth=20)
        return sm, posterior


def _case_control_single(counts: np.array,
                         case_ctrl_ids: np.array,
                         case_member: np.array,
                         batch_ids: np.array,
                         depth: int,
                         diff_scale: float = 5,
                         disp_scale: float = 1,
                         control_loc: float = 0,
                         control_scale: float = 5,
                         batch_scale: float = 3,
                         mc_samples: int = 1000,
                         num_warmup: int = 2000,
                         chains: int = 1) -> (CmdStanModel, CmdStanMCMC):
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)

    batch_encoder = LabelEncoder()
    batch_encoder.fit(batch_ids)
    batch_ids = batch_encoder.transform(batch_ids)

    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/nb_case_control_single.stan')
    sm = CmdStanModel(stan_file=code)
    dat = {
        'N': len(counts),
        'C': int(max(case_ids) + 1),
        'B': int(max(batch_ids) + 1),
        'depth': list(np.log(depth)),
        'y': list(map(int, counts.astype(np.int64))),
        'cc_bool': list(map(int, case_member)),
        'cc_ids': list(map(int, case_ids + 1)),
        'batch_ids': list(map(int, batch_ids + 1)),
        'diff_scale': diff_scale,
        'disp_scale': disp_scale,
        'batch_scale': batch_scale,
        'control_loc': control_loc,
        'control_scale': control_scale,
    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        fit = sm.sample(data=data_path, iter_sampling=mc_samples,
                        chains=chains, iter_warmup=num_warmup,
                        adapt_delta=0.95, max_treedepth=20)
        inf = az.from_cmdstanpy(fit, posterior_predictive='y_predict',
                                log_likelihood='log_lhood')
        # delete useless variables
        del inf['posterior']['lam']
        del inf['posterior']['phi']
        del inf['posterior']['control']
        return inf


def _case_control_data(counts: np.array, case_ctrl_ids: np.array,
                       case_member: np.array,
                       depth: int = None):
    case_encoder = LabelEncoder()
    case_encoder.fit(case_ctrl_ids)
    case_ids = case_encoder.transform(case_ctrl_ids)
    dat = {
        'N': counts.shape[0],
        'D': counts.shape[1],
        'C': int(max(case_ids) + 1),
        'y': counts.astype(int).tolist(),
        'cc_bool': list(map(int, case_member)),
        'cc_ids': list(map(int, case_ids + 1))
    }
    if depth is not None:
        dat['depth'] = list(np.log(depth))
    return dat


def merge_inferences(inf_list, log_likelihood, posterior_predictive,
                     coords, concatenation_name='features',
                     sample_name='samples'):
    group_list = []
    group_list.append([x.posterior for x in inf_list])
    group_list.append([x.sample_stats for x in inf_list])
    if log_likelihood is not None:
        group_list.append([x.log_likelihood for x in inf_list])
    if posterior_predictive is not None:
        group_list.append(
            [x.posterior_predictive for x in inf_list]
        )

    po_ds = xr.concat(group_list[0], concatenation_name)
    ss_ds = xr.concat(group_list[1], concatenation_name)
    group_dict = {"posterior": po_ds, "sample_stats": ss_ds}

    if log_likelihood is not None:
        ll_ds = xr.concat(group_list[2], concatenation_name)
        ll_ds = ll_ds.rename_dims({'log_lhood_dim_0': sample_name})
        group_dict["log_likelihood"] = ll_ds
    if posterior_predictive is not None:
        pp_ds = xr.concat(group_list[3], concatenation_name)
        pp_ds = pp_ds.rename_dims({'y_predict_dim_0': sample_name})
        group_dict["posterior_predictive"] = pp_ds

    all_group_inferences = []
    for group in group_dict:
        # Set concatenation dim coords
        group_ds = group_dict[group].assign_coords(
            {concatenation_name: coords[concatenation_name],
             sample_name: coords[sample_name]}
        )
        group_inf = az.InferenceData(**{group: group_ds})  # hacky
        all_group_inferences.append(group_inf)

    return az.concat(*all_group_inferences)
