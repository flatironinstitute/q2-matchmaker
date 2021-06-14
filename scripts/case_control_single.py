import qiime2
import argparse
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_matchmaker._stan import _case_control_single
import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--biom-table', help='Biom table of counts.', required=True)
    parser.add_argument(
        '--metadata-file', help='Sample metadata file.', required=True)
    parser.add_argument(
        '--matching-ids', help='Column specifying matchings.', required=True)
    parser.add_argument(
        '--groups', help=('Column specifying groups '
                          '(i.e. treatment vs control groups).'),
        required=True)
    parser.add_argument(
        '--control-group', help='The name of the control group.', required=True)
    parser.add_argument(
        '--feature-id', help='Feature to analyze.', type=str, required=True)
    parser.add_argument(
        '--mu-scale', help='Scale of differentials.',
        type=float, required=False, default=10)
    parser.add_argument(
        '--control-loc', help='Center of control log proportions.',
        type=float, required=False, default=None)
    parser.add_argument(
        '--control-scale', help='Scale of control log proportions.',
        type=float, required=False, default=10)
    parser.add_argument(
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=1000)
    parser.add_argument(
        '--chains', help='Number of MCMC chains.', type=int,
        required=False, default=4)
    parser.add_argument(
        '--output-tensor', help='Output tensor.', type=str, required=True)

    args = parser.parse_args()

    table = load_table(args.biom_table)
    counts = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                          index=table.ids(),
                          columns=table.ids(axis='observation'))
    metadata = pd.read_table(args.metadata_file, index_col=0)
    matching_ids = metadata[args.matching_ids]
    groups = metadata[args.groups]
    # match everything up
    idx = list(set(counts.index) & set(matching_ids.index) & set(groups.index))
    counts, matching_ids, groups = [x.loc[idx] for x in
                                    (counts, matching_ids, groups)]
    matching_ids, groups = matching_ids.values, groups.values
    groups = (groups == args.control_group).astype(np.int64)
    depth = counts.sum(axis=1)

    if args.control_loc is None:
        # Dirichilet-like prior
        control_loc = np.log(1 / counts.shape[1])
    else:
        control_loc = args.control_loc

    samples = _case_control_single(
        x, matching_ids, groups,
        depth, args.monte_carlo_samples,
        chains=args.chains,
        mu_scale=args.mu_scale,
        control_loc=control_loc,
        control_scale=args.control_scale)

    samples.to_netcdf(args.output_tensor)
