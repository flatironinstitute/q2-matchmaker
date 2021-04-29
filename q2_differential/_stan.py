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

# Birdman dependencies
# TODO: get rid of these once have finalized merging BaseModel
# into the Birdman repo
from birdman.model_util import (single_fit_to_inference,
                                multiple_fits_to_inference)
from cmdstanpy import CmdStanModel, CmdStanMCMC
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


class BaseModel:
    """Base Stan model.
    :param table: Feature table (features x samples)
    :type table: biom.table.Table
    :param formula: Design formula to use in model
    :type formula: str
    :param metadata: Metadata for design matrix
    :type metadata: pd.DataFrame
    :param model_path: Filepath to Stan model
    :type model_path: str
    :param num_iter: Number of posterior sample draws, defaults to 500
    :type num_iter: int
    :param num_warmup: Number of posterior draws used for warmup, defaults to
        num_iter
    :type num_warmup: int
    :param chains: Number of chains to use in MCMC, defaults to 4
    :type chains: int
    :param seed: Random seed to use for sampling, defaults to 42
    :type seed: float
    :param parallelize_across: Whether to parallelize across features or
        chains, defaults to 'chains'
    :type parallelize_across: str
    """
    def __init__(
        self,
        table: biom.table.Table,
        metadata: pd.DataFrame,
        model_path: str,
        num_iter: int = 1000,
        num_warmup: int = 500,
        chains: int = 4,
        seed: float = 42,
        #tmp_directory='/tmp',
        parallelize_across: str = "chains"
    ):
        self.table = table
        self.metadata = metadata
        self.num_iter = num_iter
        if num_warmup is None:
            self.num_warmup = num_iter
        else:
            self.num_warmup = num_warmup
        self.chains = chains
        self.seed = seed
        self.feature_names = table.ids(axis="observation")
        self.sample_names = table.ids(axis="sample")
        self.model_path = model_path
        #self.tmp_directory = tmp_directory
        self.sm = None
        self.fit = None
        self.parallelize_across = parallelize_across

        self.dat = {
            "y": table.matrix_data.todense().T.astype(int),
            "D": table.shape[0],
            "N": table.shape[1],
            "depth": np.log(table.sum(axis='sample'))
        }

    def compile_model(self) -> None:
        """Compile Stan model."""
        self.sm = CmdStanModel(stan_file=self.model_path)

    def add_parameters(self, param_dict=None) -> None:
        """Add parameters from dict to be passed to Stan."""
        self.dat.update(param_dict)

    def fit_model(
        self,
        sampler_args: dict = None,
        dask_cluster: dask_jobqueue.JobQueueCluster = None,
        jobs: int = 4,
    ) -> None:
        """Fit model according to parallelization configuration.
        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict
        :param dask_cluster: Dask jobqueue to run parallel jobs if
            parallelizing across features (optional)
        :type dask_cluster: dask_jobqueue
        :param jobs: Number of jobs to run in parallel jobs if parallelizing
            across features, defaults to 4
        :type jobs: int
        """
        if self.parallelize_across == "features":
            self.fit = self._fit_parallel(dask_cluster=dask_cluster, jobs=jobs,
                                          sampler_args=sampler_args)
        elif self.parallelize_across == "chains":
            if None not in [dask_cluster, jobs]:
                warnings.warn(
                    "dask_cluster and jobs ignored when parallelizing"
                    " across chains."
                )
            self.fit = self._fit_serial(sampler_args)
        else:
            raise ValueError("parallelize_across must be features or chains!")

    def _fit_serial(self, sampler_args: dict = None) -> CmdStanMCMC:
        """Fit model by parallelizing across chains.
        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict
        """
        if sampler_args is None:
            sampler_args = dict()

        fit = self.sm.sample(
            chains=self.chains,
            parallel_chains=self.chains,  # run all chains in parallel
            data=self.dat,
            iter_warmup=self.num_warmup,
            iter_sampling=self.num_iter,
            seed=self.seed,
            **sampler_args
        )
        return fit

    def _fit_parallel(
        self,
        dask_cluster: dask_jobqueue.JobQueueCluster = None,
        jobs: int = 4,
        sampler_args: dict = None,
    ) -> List[CmdStanMCMC]:
        """Fit model by parallelizing across features.
        :param dask_cluster: Dask jobqueue to run parallel jobs (optional)
        :type dask_cluster: dask_jobqueue
        :param jobs: Number of jobs to run parallel in parallel, defaults to 4
        :type jobs: int
        :param sampler_args: Additional parameters to pass to CmdStanPy
            sampler (optional)
        :type sampler_args: dict
        """
        if sampler_args is None:
            sampler_args = dict()

        if dask_cluster is None:
            raise ValueError('Parallelization can only be performed '
                             'on a cluster')
        dask_cluster.scale(jobs=jobs)
        client = Client(dask_cluster)
        client.wait_for_workers(jobs)
        time.sleep(60)

        @dask.delayed
        def _fit_single(self, values):
            dat = self.dat
            dat["y"] = values.astype(int)
            _fit = self.sm.sample(
                chains=self.chains,
                parallel_chains=1,            # run all chains in serial
                data=dat,
                iter_warmup=self.num_warmup,
                iter_sampling=self.num_iter,
                #output_dir=self.tmp_directory,
                seed=self.seed,
                **sampler_args
            )
            _fit.diagnose()  # won't print otherwise ...
            mu = _fit.stan_variable('mu')
            sigma = _fit.stan_variable('sigma')
            diff = _fit.stan_variable('diff')
            print('mu', mu.mean(), mu.std())
            print('sigma', sigma.mean(), sigma.std())
            print('diff', diff.mean(), diff.std())
            return _fit

        _fits = []
        for v, i, d in self.table.iter(axis="observation"):
            _fit = _fit_single(self, v)
            _fits.append(_fit)

        futures = dask.persist(*_fits)
        all_fits = dask.compute(futures)[0]
        # Set data back to full table
        self.dat["y"] = self.table.matrix_data.todense().T.astype(int)
        return all_fits

    def to_inference_object(
        self,
        params: Sequence[str],
        coords: dict,
        dims: dict,
        concatenation_name: str = "feature",
        alr_params: Sequence[str] = None,
        include_observed_data: bool = False,
        posterior_predictive: str = None,
        log_likelihood: str = None,
        dask_cluster: dask_jobqueue.JobQueueCluster = None,
        jobs: int = 4
    ) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.
        Example for a simple Negative Binomial model:
        .. code-block:: python
            inf_obj = model.to_inference_object(
                params=['beta', 'phi'],
                coords={
                    'feature': model.feature_names,
                    'covariate': model.colnames
                },
                dims={
                    'beta': ['covariate', 'feature'],
                    'phi': ['feature']
                },
                alr_params=['beta']
            )
        :param params: Posterior fitted parameters to include
        :type params: Sequence[str]
        :param coords: Mapping of entries in dims to labels
        :type coords: dict
        :param dims: Dimensions of parameters in the model
        :type dims: dict
        :param concatenation_name: Name to aggregate features when combining
            multiple fits, defaults to 'feature'
        :type concatentation_name: str
        :param alr_params: Parameters to convert from ALR to CLR (this will
            be ignored if the model has been parallelized across features)
        :type alr_params: Sequence[str], optional
        :param include_observed_data: Whether to include the original feature
            table values into the ``arviz`` InferenceData object, default is
            False
        :type include_observed_data: bool
        :param posterior_predictive: Name of posterior predictive values from
            Stan model to include in ``arviz`` InferenceData object
        :type posterior_predictive: str, optional
        :param log_likelihood: Name of log likelihood values from Stan model
            to include in ``arviz`` InferenceData object
        :type log_likelihood: str, optional
        :param dask_cluster: Dask jobqueue to run parallel jobs (optional)
        :type dask_cluster: dask_jobqueue
        :param jobs: Number of jobs to run in parallel, defaults to 4
        :type jobs: int
        :returns: ``arviz`` InferenceData object with selected values
        :rtype: az.InferenceData
        """
        if self.fit is None:
            raise ValueError("Model has not been fit!")

        args = {
            "params": params,
            "coords": coords,
            "dims": dims,
            "posterior_predictive": posterior_predictive,
            "log_likelihood": log_likelihood,
        }
        if isinstance(self.fit, CmdStanMCMC):
            fit_to_inference = single_fit_to_inference
            args["alr_params"] = alr_params
        elif isinstance(self.fit, Sequence):
            fit_to_inference = multiple_fits_to_inference
            args["concatenation_name"] = concatenation_name
            args["dask_cluster"] = dask_cluster
            args["jobs"] = jobs
            # TODO: Check that dims and concatenation_match

            if alr_params is not None:
                warnings.warn("ALR to CLR not performed on parallel models.",
                              UserWarning)
        else:
            raise ValueError("Unrecognized fit type!")

        inference = fit_to_inference(self.fit, **args)
        if include_observed_data:
            obs = az.from_dict(
                observed_data={"observed": self.dat["y"]},
                coords={
                    "tbl_sample": self.sample_names,
                    "feature": self.feature_names
                },
                dims={"observed": ["tbl_sample", "feature"]}
            )
            inference = az.concat(inference, obs)
        return inference

    def diagnose(self):
        """Use built-in diagnosis function of ``cmdstanpy``."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")
        if self.parallelize_across == "chains":
            return self.fit.diagnose()
        if self.parallelize_across == "features":
            return [x.diagnose() for x in self.fit]

    def summary(self):
        """Use built-in summary function of ``cmdstanpy``."""
        if self.fit is None:
            raise ValueError("Model has not been fit!")
        if self.parallelize_across == "chains":
            return self.fit.summary()
        if self.parallelize_across == "features":
            return [x.summary() for x in self.fit]


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
                 #tmp_directory='/tmp',
                 mu_scale: float = 10,
                 sigma_scale: float = 1,
                 disp_scale: float = 1,
                 control_scale: float = 10):
        model_path = os.path.join(
            os.path.dirname(__file__),
            'assets/nb_case_control_single.stan')
        super(NegativeBinomialCaseControl, self).__init__(
            table, metadata, model_path,
            num_iter, num_warmup, chains, seed,
            #tmp_directory=tmp_directory,
            parallelize_across = "features")
        case_ctrl_ids = self.metadata[matching_column]
        status = self.metadata[status_column]
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
            "control_scale": control_scale
        }
        self.add_parameters(param_dict)

    def to_inference_object(
        self,
        dask_cluster: dask_jobqueue.JobQueueCluster = None,
        jobs: int = 4
    ) -> az.InferenceData:
        """Convert fitted Stan model into ``arviz`` InferenceData object.

        Parameters
        ----------
        dask_cluster dask_jobqueue.JobQueueCluster, optional
            Dask jobqueue to run parallel jobs (optional)
        jobs : int
            Number of jobs to run in parallel, defaults to 4

        Returns
        -------
        az.InferenceData : ``arviz`` InferenceData object with selected values
        """
        dims = {
            "mu": ["feature"],
            "sigma": ["feature"],
            "disp": ["status", "feature"],
            "diff": ["feature"],
            "log_lhood": ["tbl_sample", "feature"],
            "y_predict": ["tbl_sample", "feature"]
        }
        coords = {
            "status": self.status_names,
            "feature": self.feature_names,
            "tbl_sample": self.sample_names
        }

        # TODO: May want to allow not passing PP/LL/OD in the future
        args = dict()
        args["dask_cluster"] = dask_cluster
        args["jobs"] = jobs

        inf = super().to_inference_object(
            params=self.param_names,
            dims=dims,
            coords=coords,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood",
            include_observed_data=True,
            **args
        )
        return inf


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
