import argparse
from biom import load_table
import numpy as np
import arviz as az
from q2_matchmaker._stan import merge_inferences


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
        '--output-inference', help='Merged inference file.',
        type=str, required=True)

    args = parser.parse_args()
    # A little redundant, but necessary for getting ids
    table = load_table(args.biom_table)
    inf_list = []
    for x in args.inference_files:
        inf = az.from_netcdf(x)
        # delete useless variables
        if hasattr(inf['posterior'], 'lam'):
            del inf['posterior']['lam']
        if hasattr(inf['posterior'], 'phi'):
            del inf['posterior']['phi']
        if hasattr(inf['posterior'], 'a1'):
            del inf['posterior']['a1']
        if hasattr(inf['posterior'], 'control'):
            del inf['posterior']['control']
        inf_list.append(inf)

    coords = {'features': table.ids(axis='observation'),
              'monte_carlo_samples': np.arange(args.monte_carlo_samples)}
    samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)

    samples.to_netcdf(args.output_inference)
