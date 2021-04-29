
import unittest
import numpy as np
from q2_differential._stan import (
    _case_control_sim, _case_control_full,
    _case_control_data, _case_control_single,
    NegativeBinomialCaseControl
)
from biom import Table
from birdman.diagnostics import r2_score
from dask.distributed import wait
import arviz as az

try:
    from dask_jobqueue import SLURMCluster
    import dask
    no_dask = False
except:
    no_dask = True


np.random.seed(0)
table, metadata, diff = _case_control_sim(
    n=200, d=20, depth=1000)


biom_table = Table(table.values.T,
                   list(table.columns),
                   list(table.index))
cluster = SLURMCluster(cores=5,
                       processes=4,
                       memory='16GB',
                       walltime='01:00:00',
                       interface='ib0',
                       nanny=True,
                       death_timeout='300s',
                       local_directory='/scratch',
                       shebang='#!/usr/bin/env bash',
                       env_extra=["export TBB_CXX_TYPE=gcc"],
                       queue='ccb')

nb = NegativeBinomialCaseControl(
    table=biom_table,
    matching_column="reps",
    status_column="diff",
    metadata=metadata,
    #tmp_directory='/scratch',
    reference_status='1',
    num_warmup=1000,
    mu_scale=5,
    control_scale=5,
    chains=1,
    seed=42)
nb.compile_model()
nb.fit_model(dask_cluster=cluster, jobs=4)
print(nb.fit)
print(nb.fit[0])
for n in nb.fit:
    try:
        az.from_cmdstanpy(posterior=n)
    except: continue
inf = nb.to_inference_object(dask_cluster=cluster, jobs=4)
loo = az.loo(inf)
bfmi = az.bfmi(inf)
rhat = az.rhat(inf, var_names=nb.param_names)
ess = az.ess(inf, var_names=nb.param_names)
r2 = r2_score(inf)
print('loo', loo)
print('bfmi', bfmi)
print('rhat', rhat)
print('r2', r2)
