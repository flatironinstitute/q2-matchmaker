import argparse
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
import arviz as az
from q2_matchmaker._stan import _case_control_single, merge_inferences
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--biom-table', help='Biom table of counts.', required=True)
    parser.add_argument(
        '--inference-files',
        metavar='N', type=str, nargs='+',
        help='List of inference files.', required=True)
    parser.add_argument(
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=1000)
    parser.add_argument(
        '--output-inference', help='Merged inference file.', type=str, required=True)

    args = parser.parse_args()
    # A little redundant, but necessary for getting ids
    table = load_table(args.biom_table)
    counts = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                          index=table.ids(),
                          columns=table.ids(axis='observation'))
    metadata = pd.read_table(args.metadata_file, index_col=0)
    replicates = metadata[args.replicates]
    batches = metadata[args.batches]
    idx = list(set(counts.index) & set(replicates.index) & set(batches.index))

    inf_list = [az.from_netcdf(x) for x in args.inference_files]
    coords={'features' : counts.columns,
            'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}
    samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)

    samples.to_netcdf(args.output_inference)
