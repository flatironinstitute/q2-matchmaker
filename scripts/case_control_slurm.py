import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from biom import load_table
from q2_matchmaker._method import _negative_binomial_case_control
import time
import logging
from gneiss.util import match
import pandas as pd
from birdman.diagnostics import r2_score
import os


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
    '--reference-group', help='The name of the reference group.',
    required=True)
parser.add_argument(
    '--monte-carlo-samples', help='Number of monte carlo samples.',
    type=int, required=False, default=1000)
parser.add_argument(
    '--chains', help='Number of parallel chains.',
    type=int, required=False, default=4)
parser.add_argument(
    '--cores', help='Number of cores per process.',
    type=int, required=False, default=1)
parser.add_argument(
    '--nodes', help='Number of nodes.',
    type=int, required=False, default=1)
parser.add_argument(
    '--processes', help='Number of processes.',
    type=int, required=False, default=1)
parser.add_argument(
    '--memory', help='Memory allocation size.',
    type=str, required=False, default='16GB')
parser.add_argument(
    '--walltime', help='Walltime.', type=str,
    required=False, default='01:00:00')
parser.add_argument(
    '--interface', help='Interface for communication',
    type=str, required=False, default='eth0')
parser.add_argument(
    '--queue', help='Queue to submit job to.', type=str, required=True)
parser.add_argument(
    '--local-directory', help='Scratch directory to deposit dask logs.',
    type=str, required=False, default='/scratch')
parser.add_argument(
    '--output-directory',
    help='Output directory of differential tensor and diagnosis stats.',
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

# load relevant files
table = load_table(args.biom_table)
metadata = pd.read_table(args.metadata_file, index_col=0)
table, metadata = match(table, metadata)

samples = _negative_binomial_case_control(
    table,
    metadata[args.matching_ids],
    metadata[args.groups],
    reference_group=args.reference_group,
    mu_scale=1,
    sigma_scale=1,
    disp_scale=1,
    control_loc=-5,
    control_scale=3,
    num_iter=args.monte_carlo_samples,
    chains=args.chains
)

# Save files to output directory
os.mkdir(args.output_directory)
posterior_file = os.path.join(args.output_directory,
                              'differential_posterior.nc')
samples.to_netcdf(posterior_file)


# Get summary statistics
# summary_stats = r2_score(samples)
# summary_file = os.path.join(args.output_directory,
#                             'summary_statistics.txt')
# summary_stats.to_csv(summary_file, sep='\t')
