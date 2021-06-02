import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
import dask.array as da
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_matchmaker._stan import (_case_control_single, _case_control_sim,
                                 merge_inferences)

import time
import arviz as az


np.random.seed(0)
counts, metadata, diff = _case_control_sim(
    n=50, d=10, depth=100)


jobs=4
cluster = SLURMCluster(cores=4,
                       processes=jobs,
                       memory='16GB',
                       walltime='01:00:00',
                       interface='ib0',
                       nanny=True,
                       death_timeout='300s',
                       local_directory='/scratch',
                       shebang='#!/usr/bin/env bash',
                       env_extra=["export TBB_CXX_TYPE=gcc"],
                       queue='ccb')
print(cluster.job_script())
cluster.scale(jobs=jobs)
client = Client(cluster)
print(client)
client.wait_for_workers(jobs)
time.sleep(60)
print(cluster.scheduler.workers)

# take intersection
depth = counts.sum(axis=1)
pfunc = lambda x: _case_control_single(
    x, case_ctrl_ids=metadata['reps'],
    case_member=metadata['diff'],
    depth=depth, mc_samples=1000)
dcounts = da.from_array(counts.values.T, chunks=(counts.T.shape))

res = []
for d in range(dcounts.shape[0]):
    r = dask.delayed(pfunc)(dcounts[d])
    res.append(r)
# Retrieve InferenceData objects and save them to disk
futures = dask.persist(*res)
resdf = dask.compute(futures)
inf_list = list(resdf[0])
# coords={'features' : table.ids(axis='observation'),
#         'monte_carlo_samples' : np.arange(1000)}
coords={'features' : counts.columns, 'monte_carlo_samples' : np.arange(1000)}

samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)
#samples = xr.concat([df.to_xarray() for df in data_df], dim="features")
#samples = samples.assign_coords()

# Get summary statistics
# loo = az.loo(samples)   # broken due to nans
# bfmi = az.bfmi(samples) # broken due to nans
param_names = ['mu', 'sigma', 'diff', 'disp', 'control']
rhat = az.rhat(samples, var_names=param_names)
ess = az.ess(samples, var_names=param_names)
y_pred = samples['posterior_predictive'].stack(
    sample=("chain", "draw"))['y_predict'].fillna(0).values.T
r2 = az.r2_score(counts.values, y_pred)

summary_stats = loo
summary_stats.loc['bfmi'] = [bfmi.mean().values, bfmi.std().values]
summary_stats.loc['r2'] = r2.values

# Save everything to a file
# os.mkdir(args.output_directory)
# posterior_file = os.path.join(args.output_directory,
#                               'differential_posterior.nc')
# summary_file = os.path.join(args.output_directory,
#                             'summary_statistics.nc')
# rhat_file = os.path.join(args.output_directory, 'rhat.nc')
# samples.to_netcdf(posterior_file)
# summary_stats.to_netcdf(summary_file)
# rhat.to_netcdf(rhat_file)
