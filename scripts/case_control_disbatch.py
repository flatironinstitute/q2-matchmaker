import qiime2
import argparse
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_matchmaker._stan import _case_control_single, merge_inferences
import time
import logging
import subprocess, os
import tempfile
import arviz as az


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
        '--local-directory',
        help=('Scratch directory to deposit logs '
              'and intermediate files.'),
        type=str, required=False, default='/scratch')
    parser.add_argument(
        '--job-extra',
        help=('Additional job arguments, like loading modules.'),
        type=str, required=False, default='')
    parser.add_argument(
        '--output-inference', help='Output inference tensor.',
        type=str, required=True)

    args = parser.parse_args()
    print(args)
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

    if args.control_loc is None:
        # Dirichilet-like prior
        control_loc = np.log(1 / counts.shape[1])
    else:
        control_loc = args.control_loc

    # Launch disbatch
    ## First create a temporary file with all of the tasks
    with tempfile.TemporaryDirectory() as temp_dir_name:
        print(temp_dir_name)
        task_fp = os.path.join(temp_dir_name, 'tasks.txt')
        print(task_fp)
        with open(task_fp, 'w') as fh:
            for feature_id in counts.columns:
                cmd_ = ('case_control_single.py '
                        f'--biom-table {args.biom_table} '
                        f'--metadata-file {args.metadata_file} '
                        f'--matching-ids {args.matching_ids} '
                        f'--groups {args.group} '
                        f'--control-group {args.control_group} '
                        f'--feature-id {feature_id} '
                        f'--mu-scale {args.mu_scale} '
                        f'--control-loc {control_loc} '
                        f'--control-scale {args.control_scale} '
                        f'--monte-carlo-samples {args.monte_carlo_samples} '
                        f'--chains {args.chains} '
                        f'--output-tensor {args.local_directory}/{feature_id}.nc'
                        # slurm logs
                        f' &> {args.local_directory}/{feature_id}.log\n')
                print(cmd_)
                fh.write(cmd_)
        ## Run disBatch with the SLURM environmental parameters
        cmd = f'disBatch {task_fp}'
        cmd = f'{args.job_extra}; {cmd}'
        slurm_env = os.environ.copy()
        print(cmd)
        try:
            output = subprocess.run(cmd, env=slurm_env, check=True, shell=True)
        except subprocess.CalledProcessError as exc:
            print("Status : FAIL", exc.returncode, exc.output)
        else:
            print("Output: \n{}\n".format(output))

    # Aggregate results
    inference_files = [f'{args.local_directory}/{feature_id}.nc'
                       for feature_id in counts.columns]
    inf_list = [az.from_netcdf(x) for x in inference_files]
    coords={'features' : counts.columns,
            'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}
    samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)
    samples.to_netcdf(args.output_inference)
