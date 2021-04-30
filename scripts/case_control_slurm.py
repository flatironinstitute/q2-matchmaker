import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
import dask.array as da
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_differential._stan import _case_control_single, merge_inferences
import time
import logging
import os
import arviz as az


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--biom-table', help='Biom table of counts.', required=True)
parser.add_argument(
    '--metadata-file', help='Sample metadata file.', required=True)
parser.add_argument(
    '--matching-ids', help='Column specifying matchings.', required=True)
parser.add_argument(
    '--groups', help='Column specifying groups.', required=True)
parser.add_argument(
    '--reference-group', help='The name of the reference group.', required=True)
parser.add_argument(
    '--monte-carlo-samples', help='Number of monte carlo samples.',
    type=int, required=False, default=1000)
parser.add_argument(
    '--cores', help='Number of cores per process.', type=int, required=False, default=1)
parser.add_argument(
    '--processes', help='Number of processes per node.', type=int, required=False, default=1)
parser.add_argument(
    '--nodes', help='Number of nodes.', type=int, required=False, default=1)
parser.add_argument(
    '--memory', help='Memory allocation size.', type=str, required=False, default='16GB')
parser.add_argument(
    '--walltime', help='Walltime.', type=str, required=False, default='01:00:00')
parser.add_argument(
    '--interface', help='Interface for communication', type=str, required=False, default='eth0')
parser.add_argument(
    '--queue', help='Queue to submit job to.', type=str, required=True)
parser.add_argument(
    '--local-directory', help='Scratch directory to deposit dask logs.', type=str, required=False,
    default='/scratch')

parser.add_argument(
    '--output-directory', help=('Output directory to store posterior distributions '
                                ' and diagnostics.'),
    type=str, required=True)
args = parser.parse_args()
print(args)
cluster = SLURMCluster(cores=args.cores,
                       processes=args.processes,
                       memory=args.memory,
                       walltime=args.walltime,
                       interface=args.interface,
                       nanny=True,
                       death_timeout='300s',
                       local_directory=args.local_directory,
                       shebang='#!/usr/bin/env bash',
                       env_extra=["export TBB_CXX_TYPE=gcc"],
                       queue=args.queue)
print(cluster.job_script())
cluster.scale(jobs=args.nodes)
client = Client(cluster)
print(client)
client.wait_for_workers(args.nodes)
time.sleep(60)
print(cluster.scheduler.workers)
table = load_table(args.biom_table)
counts = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                      index=table.ids(),
                      columns=table.ids(axis='observation'))
metadata = pd.read_table(args.metadata_file, index_col=0)
matching_ids = metadata[args.matching_ids]
groups = metadata[args.groups]
metadata = pd.DataFrame({'cc_ids': matching_ids,
                         'groups': groups})
metadata['groups'] = (metadata['groups'] == args.reference_group).astype(np.int64)

# take intersection
idx = list(set(metadata.index) & set(counts.index))
counts = counts.loc[idx]
metadata = metadata.loc[idx]
depth = counts.sum(axis=1)
pfunc = lambda x: _case_control_single(
    x, case_ctrl_ids=metadata['cc_ids'],
    case_member=metadata['groups'],
    depth=depth, mc_samples=args.monte_carlo_samples)
dcounts = da.from_array(counts.values.T, chunks=(counts.T.shape))

res = []
for d in range(dcounts.shape[0]):
    r = dask.delayed(pfunc)(dcounts[d])
    res.append(r)
# Retrieve InferenceData objects and save them to disk
futures = dask.persist(*res)
resdf = dask.compute(futures)
print('Runs complete')
inf_list = list(resdf[0])
coords={'features' : counts.columns,
        'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}
samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)
print('Merging results')
# Get summary statistics
#loo = az.loo(samples)
#bfmi = az.bfmi(samples)
param_names = ['mu', 'sigma', 'diff', 'disp', 'control']
#rhat = az.rhat(samples, var_names=param_names)
#ess = az.ess(samples, var_names=param_names)

# Get Bayesian r2
y_pred = samples['posterior_predictive'].stack(
    sample=("chain", "draw"))['y_predict'].fillna(0).values.T
r2 = az.r2_score(counts.values, y_pred)

summary_stats = r2
#summary_stats.loc['bfmi'] = [bfmi.mean().values, bfmi.std().values]
#summary_stats.loc['r2'] = r2.values

# Save everything to a file
os.mkdir(args.output_directory)
posterior_file = os.path.join(args.output_directory,
                              'differential_posterior.nc')
summary_file = os.path.join(args.output_directory,
                            'summary_statistics.txt')
# rhat_file = os.path.join(args.output_directory, 'rhat.nc')
# ess_file = os.path.join(args.output_directory, 'ess.nc')
# bfmi_file = os.path.join(args.output_directory, 'bfmi.nc')
samples.to_netcdf(posterior_file)
#rhat.to_netcdf(rhat_file)
#ess.to_netcdf(ess_file)
#bfmi.to_netcdf(bfmi_file)
summary_stats.to_csv(summary_file, sep='\t')
