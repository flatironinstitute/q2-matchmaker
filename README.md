# q2-matchmaker
A qiime2 plugin for case-control differential abundance analysis

I want to stress that this repository is highly experimental.  There are many holes in the source code and the  documentation is going to be patchy until version 1 is released. So feel free to use at your own risk.

# Installation (for Linux)

First install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

If you want to use the qiime2 version, go ahead and install the most up-to-date version.

Then install qiime2
```
wget https://data.qiime2.org/distro/core/qiime2-2021.4-py36-linux-conda.yml
conda env create -n qiime2-2021.4 --file qiime2-2021.4-py36-linux-conda.yml
rm qiime2-2020.11-py36-linux-conda.yml
```

There are a few additional dependencies that you will need to install, namely
```
pip install git+git@github.com:mortonjt/q2-types.git
pip install arviz
pip install cmdstanpy
```

Finally, install this repository
```
git clone https://github.com/flatironinstitute/q2-matchmaker.git
cd q2-matchmaker
source install.sh
pip install -e .
qiime dev refresh-cache   # this is optional. 
```

# qiime2 Tutorial

If you want to just a feel how to run this pipeline, there is a qiime2 interface available.
But do note that this pipeline is much too slow to handle practically sized datasets.
Go to SLURM tutorial to see how to speed up the computation on a high performance compute cluster.

First go to the examples folder via `cd examples`.
Then load the biom table as a qiime2 Artifact.
```bash
qiime tools import --input-path table.biom --output-path table.qza --type FeatureTable[Frequency]
```

```bash
qiime matchmaker negative-binomial-case-control \
    --i-table table.qza \
    --m-matching-ids-file sample_metadata.txt --m-matching-ids-column reps \
    --m-groups-file sample_metadata.txt --m-groups-column diff \
    --p-control-group 0 \
    --o-differentials differentials.qza
```

There is a chance that you may get an error, that looks like as follows
```
Plugin error from matchmaker:

  Unable to compile Stan model file: XXX ...

Debug info has been saved to XXX ...
```

The solution is to run `export TBB_CXX_TYPE=gcc` first, then rerun the above qiime2 command.

Once you have the differentials file you can upload the inference data object into [Arviz](https://arviz-devs.github.io/arviz/index.html) in Python as follows

```python
import qiime2
import arviz as az
inf = qiime2.Artifact.load('differentials.nc.qza').view(az.InferenceData)
```
Investigating the `inf   object will yield something like
```
Inference data with groups:
        > posterior
        > sample_stats
        > observed_data
```
where `inf.posterior` contains all of the posterior draws of the variables of interest, `inf.sample_stats` contains information about MCMC diagnostics and `inf.observed_data` contains the preprocessed data formatted for the MCMC model.

Investigating `inf.posterior` will yield something like
```
<xarray.Dataset>
Dimensions:          (chain: 4, control_dim_0: 450, diff_dim_0: 9, disp_dim_0: 20, draw: 2000, mu_dim_0: 9, sigma_dim_0: 9, y_predict_dim_0: 1000)
Coordinates:
  * chain            (chain) int64 0 1 2 3
  * draw             (draw) int64 0 1 2 3 4 5 ... 1994 1995 1996 1997 1998 1999
  * control_dim_0    (control_dim_0) int64 0 1 2 3 4 5 ... 445 446 447 448 449
  * diff_dim_0       (diff_dim_0) int64 0 1 2 3 4 5 6 7 8
  * mu_dim_0         (mu_dim_0) int64 0 1 2 3 4 5 6 7 8
  * sigma_dim_0      (sigma_dim_0) int64 0 1 2 3 4 5 6 7 8
  * disp_dim_0       (disp_dim_0) int64 0 1 2 3 4 5 6 7 ... 13 14 15 16 17 18 19
  * y_predict_dim_0  (y_predict_dim_0) int64 0 1 2 3 4 5 ... 995 996 997 998 999
Data variables:
    control          (chain, draw, control_dim_0) float64 ...
    diff             (chain, draw, diff_dim_0) float64 ...
    mu               (chain, draw, mu_dim_0) float64 ...
    sigma            (chain, draw, sigma_dim_0) float64 ...
    disp             (chain, draw, disp_dim_0) float64 ...
    y_predict        (chain, draw, y_predict_dim_0) float64 ...
Attributes:
    created_at:                 2021-06-14T23:14:42.049366
    arviz_version:              0.11.2
    inference_library:          cmdstanpy
    inference_library_version:  0.9.68
```
Here, we had 100 biological samples, and 10 microbial species.  The differentials are in an ALR representation with the first species being the reference frame.

The most relevant variable here is `diff` which measures the differentials between the cases and the controls.  You can extract that via `inf.posterior['diff]`.

Otherwise, make sure to check out [Arviz](https://arviz-devs.github.io/arviz/index.html), since it provides an extremely comprehensive API for diagnostics, so it is recommended to check it out.

# SLURM Tutorial

If you really want apply this to typical microbiome datasets, chances are, you will need to use the SLURM interface. However, note that it will be considerably more difficult to use compared to the qiime2 interface, since it will require some cluster configuration. To use this, you will need to install [disBatch](https://github.com/flatironinstitute/disBatch). This can be installed via
```
pip install git+https://github.com/flatironinstitute/disBatch.git
```

An example SLURM script would look like as follows

```bash
#!/bin/sh
#SBATCH ... # whatever slurm params you need
conda activate qiime2-2021.4
export TBB_CXX_TYPE=gcc
case_control_disbatch.py \
    --biom-table table.biom \
    --metadata-file sample_metadata.txt \
    --matching-ids reps \
    --groups diff \
    --control-group 0 \
    --monte-carlo-samples 1000 \
    --local-directory /scratch \
    --output-inference differentials.nc
```

If this script is saved in a file called `launch.sh`, it can be run as follows
```
sbatch -n 10 -c 4 --mem 8GB launch.sh
```
Where it would be run using 10 processes, each allocated with 4 cores and 8GB per process.

The command is very similar to the qiime2 command, but there are some very nuisanced differences, The resources need to be carefully allocated depending on the dataset.  In addition, large datasets with thousands of microbes will generate thousands of microbes, which will put stress on a networked file system.  Therefore, it is important to specify the `--local-directory` to save intermediate files to node local storage, which is much faster to access than the networked file system.  Every system is different, your favorite systems admin will know the location of the node local storage.

Finally, persistence is never a guarantee with cluster systems, jobs will fail, in which you may need to relaunch jobs.  You may need to modify the `biom-table` to filter out ids that succeed (see [here](https://biom-format.org/documentation/generated/biom.table.Table.filter.html)).  You may also need to rescue your intermediate files if you saved them to node local storage. If you remember your node ids, you can recover them via ssh as follows
```bash
for remote in {node1,node2,...}
do
   ssh $remote 'cp /scratch/*.nc your_directory')
done
```
Once this is done, you can stitch together your individual runs using the `case_control_merge.py` command as follows
```
case_control_merge.py \
    --biom-table table.biom \
    --inference-files your_directory/*
    --output-inference differentials.nc
```
And wahla, you now have an arviz object that you can open in python via
```python
import arviz as az
inf = az.from_netcdf('differentials.nc')
```

