import argparse
from biom import load_table
import pandas as pd
import numpy as np
from q2_matchmaker._stan import _case_control_single


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
        '--treatment-group', help='The name of the control group.',
        required=True)
    parser.add_argument(
        '--batch-ids', help='Column specifying batch_ids.', required=False)
    parser.add_argument(
        '--feature-id', help='Feature to analyze.', type=str, required=True)
    parser.add_argument(
        '--diff-scale', help='Scale of differentials.',
        type=float, required=False, default=5)
    parser.add_argument(
        '--disp-scale', help='Scale of dispersion.',
        type=float, required=False, default=0.1)
    parser.add_argument(
        '--batch-scale', help='Scale of batch effects.',
        type=float, required=False, default=0.1)
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
    metadata = pd.read_table(args.metadata_file, comment='#', dtype=str)
    metadata = metadata.set_index(metadata.columns[0])
    matching_ids = metadata[args.matching_ids]
    groups = metadata[args.groups]
    if args.batch_ids is None or args.batch_ids == 'None':
        batch_ids = pd.Series(np.zeros(len(metadata)), index=metadata.index)
    else:
        batch_ids = metadata[args.batch_ids]

    # match everything up
    idx = list(set(counts.index) & set(metadata.index))
    varz = (matching_ids, groups, batch_ids)
    matching_ids, groups, batch_ids = [x.loc[idx].values for x in varz]
    counts = counts.loc[idx]
    groups = (groups == args.treatment_group).astype(np.int64)
    depth = counts.sum(axis=1)
    if len(counts) == 0 or len(groups) == 0:
        raise ValueError('No samples overlap with biom table or metadata. '
                         'Double check your sample names.')
    if args.control_loc is None:
        # Dirichilet-like prior
        control_loc = np.log(1 / counts.shape[1])
    else:
        control_loc = args.control_loc

    x = counts[args.feature_id]

    samples = _case_control_single(
        x, matching_ids, groups, batch_ids,
        depth=depth,
        mc_samples=args.monte_carlo_samples,
        chains=args.chains,
        diff_scale=args.diff_scale,
        disp_scale=args.disp_scale,
        control_loc=control_loc,
        control_scale=args.control_scale)

    samples.to_netcdf(args.output_tensor)
