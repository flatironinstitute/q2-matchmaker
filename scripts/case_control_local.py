import argparse
from dask.distributed import Client, LocalCluster
from biom import load_table
from q2_matchmaker._method import _negative_binomial_case_control
import time
import logging
from gneiss.util import match
import pandas as pd
from birdman.diagnostics import r2_score
import os


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
if __name__ == "__main__":
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
        '--chunksize', help='Number of features to analyze per process.',
        type=int, required=False, default=50)
    parser.add_argument(
        '--cores', help='Number of cores per process.',
        type=int, required=False, default=1)
    parser.add_argument(
        '--output-tensor', help='Output tensor.', type=str, required=True)
    args = parser.parse_args()
    print(args)
    dask_args = {'n_workers': args.cores, 'threads_per_worker': 1}
    cluster = LocalCluster(**dask_args)
    cluster.scale(dask_args['n_workers'])
    client = Client(cluster)
    print(client)

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
        chains=args.chains,
        chunksize=args.chunksize
    )

    # Save files to output directory
    samples.to_netcdf(args.output_tensor)
