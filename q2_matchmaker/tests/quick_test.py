
import unittest
import numpy as np
from q2_matchmaker._stan import (
    _case_control_sim, _case_control_full,
    _case_control_data, _case_control_single,
    NegativeBinomialCaseControl
)
from biom import Table
from birdman.diagnostics import r2_score
from dask.distributed import wait
import arviz as az
from dask.distributed import Client, LocalCluster
import dask

# from dask_jobqueue import SLURMCluster
# import dask


np.random.seed(0)
table, metadata, diff = _case_control_sim(
    n=200, d=20, depth=1000)


biom_table = Table(table.values.T,
                   list(table.columns),
                   list(table.index))
# cluster = SLURMCluster(cores=5,
#                        processes=4,
#                        memory='16GB',
#                        walltime='01:00:00',
#                        interface='ib0',
#                        nanny=True,
#                        death_timeout='300s',
#                        local_directory='/scratch',
#                        shebang='#!/usr/bin/env bash',
#                        env_extra=["export TBB_CXX_TYPE=gcc"],
#                        queue='ccb')

dask_args={'n_workers': 5, 'threads_per_worker': 2}
cluster = LocalCluster(**dask_args)
cluster.scale(dask_args['n_workers'])
client = Client(cluster)

nb = NegativeBinomialCaseControl(
    table=biom_table,
    matching_column="reps",
    status_column="diff",
    metadata=metadata,
    #tmp_directory='/scratch',
    reference_status='1',
    num_warmup=1000,
    mu_scale=1,
    control_loc=-5,
    control_scale=3,
    chains=2,
    seed=42)
nb.compile_model()
nb.fit_model()
inf = nb.to_inference_object()
loo = az.loo(inf)
bfmi = az.bfmi(inf)
rhat = az.rhat(inf, var_names=nb.param_names)
ess = az.ess(inf, var_names=nb.param_names)
r2 = r2_score(inf)
print('loo', loo)
print('bfmi', bfmi)
print('rhat', rhat)
print('r2', r2)
