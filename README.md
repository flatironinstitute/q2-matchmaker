# q2-matchmaker
A software package for case-control differential abundance analysis

I want to stress that this software package is research software and may be challenging to run.  Running this software requires a super computing cluster with 100s of cores, in addition to a very involved downstream analysis.
For examples on how the outputs can be processed, see the notebooks for the [ASD multiomics analysis](https://github.com/mortonjt/asd_multiomics_analyses).

All of the dependencies of this software package is frozen, meaning that we have provided a conda environment with the exact version numbers to faciliate the installation. For future iterations that will focus on lowering the barriers of this Bayesian modeling strategy see [Birdman](https://github.com/gibsramen/BIRDMAn).

Feel free to raise questions through the issue tracker and I will try my best to be responsive.

# Installation (for Linux)

First install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Then install the q2_matchmaker dependencies through the following conda environment

```
conda create -f environment.yml
```

Then you'll need to install the `q2_matchmaker` package.  One way to do this is through pip via

```
pip install git+https://github.com/flatironinstitute/q2-matchmaker.git
```

And voila!  You should be able to activate your environment via `conda activate q2_matchmaker` and access the basic command line utilities.


# Basic Linux tutorial

If you have access to a compute cluster that has nodes with >30 cores, then you can apply the the `case_control_parallel.py` workflow to your data!

See the `examples` folder for some example datasets and scripts.  For instance, you try out the following command

```
export TBB_CXX_TYPE=gcc
case_control_parallel.py \
    --biom-table table.biom \
    --metadata-file sample_metadata.txt \
    --matching-ids reps \
    --groups diff \
    --treatment-group 0 \
    --monte-carlo-samples 100 \
    --processes 40 \
    --output-inference differentials.nc
```

We can dissect this command.  The input counts (`table.biom`) are passed in as a [biom format](https://biom-format.org/), which is a sparse matrix representation of the microbial counts.
The `sample_metadata.txt` contains the metadata associated with each biological sample.  There are a couple of important points here.  First, the `reps` column denotes the case-control matching ids -- each case-control pair is given a unique id.  Second, the `diff` column denotes the which subjects are allocated to case and control.  Here I put in zeros and ones, but it is ok to pass in categorical values (see the [ASD multiomics analysis](https://github.com/mortonjt/asd_multiomics_analyses) scripts for more examples).  If you don't know how to obtain your matching ids, don't worry, keep reading.

The option `--treatment-group` tells us the group of interest (i.e. which subjects correspond to the sick cohort, rather than the control cohort).  This is important for getting the directionality of the log-fold changes correct.

The option `--monte-carlo-samples` specifies the number of draws from the posterior distribution will be obtained.  This is important for calculating Bayesian credible intervals, pvalues and effect sizes.  We found that 100 samples is more than enough, amounting to only shuffling around a few GBs.  One can do 1000+ samples, but we have found it to be too computationally expensive, since it will blow up memory in downstream steps and requires us to shuffle multiple TBs (not fun!).

The option `--processes` specifies the number of parallel processes to be performed. It is important to note that the total number of processes to be launched is actually `processes x chains` (see the `--chains` option).  You can launch multiple parallel chains to help with debugging, but you should be careful NOT to launch more processes than what your computer has.  So if you have 40 cores on your computer and you specify 40 processes and 4 chains, you have 160 processes fightning for 40 cores.  Then you machine will thrash and take forever to complete.  Alternatively, you could launch 40 processes with 1 chain, and then you can process 40 microbes at a time (abet less capacity to perform diagnostics like Rhat).

The option `--output-inference` specifies the output file location.  The output format is a netcdf file that can be opened using [Arviz](https://github.com/arviz-devs/arviz).
This is a tensor format that stores all of the output results.  You can open one of these files and view its contents with the following command

```python
import arviz as az
inf = az.from_netcdf('differentials.nc')
inf
```
you may see the following output.
```
Inference data with groups:
          > posterior
          > posterior_predictive
          > log_likelihood
          > sample_stats
```
You will note that there are new fields, namely `posterior_predictive` and `log_likelihood`.  These new objects can aid with additional diagnostics such as Bayesian R2 or other out-of-distribution statistics.  For more, check out the [Stan documentation](https://mc-stan.org/users/documentation/).

This object is also better labeled, the feature names and sample names are all intact.  Specifically, investigating `inf.posterior` will yield
```
Dimensions:        (chain: 4, control_dim_0: 50, disp_dim_0: 2, draw: 1000, features: 10, samples: 100)
Coordinates:
  * chain          (chain) int64 0 1 2 3
  * draw           (draw) int64 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
  * control_dim_0  (control_dim_0) int64 0 1 2 3 4 5 6 ... 43 44 45 46 47 48 49
  * disp_dim_0     (disp_dim_0) int64 0 1
  * features       (features) object 'o0' 'o1' 'o2' 'o3' ... 'o6' 'o7' 'o8' 'o9'
  * samples        (samples) object 's75' 's98' 's22' ... 's48' 's69' 's54'
Data variables:
    control        (features, chain, draw, control_dim_0) float64 ...
    diff           (features, chain, draw) float64 ...
    mu             (features, chain, draw) float64 ...
    sigma          (features, chain, draw) float64 ...
    disp           (features, chain, draw, disp_dim_0) float64 ...
Attributes:
    created_at:                 2021-06-16T04:40:37.233277
    arviz_version:              0.11.2
    inference_library:          cmdstanpy
    inference_library_version:  0.9.68

```

The variable that is most likely of interest is `diff` which specifies the computed log-fold changes.  You can pull format this into a readable format via
```python
diffs = inf['posterior']['diff'].to_dataframe().reset_index().pivot(
        index='features', columns=['chain', 'draw'], values='diff')
```
This will give you a pandas dataframe, where the rows are microbes and the columns are posterior samples.
From this, you can compute posterior statistics such as per-microbe credible intervals via

```python
import numpy as np
lo = np.percentile(diffs, q=5, axis=1)  # gives the 5% credible interval
hi = np.percentile(diffs, q=95, axis=1) # gives the 95% credible interval
```
We can also test to see if the differential itself is statistically significant (like PERMANOVA in beta diversity).
```python
from q2_matchmaker.stats import spherical_test
reject, radius, zero_dist = spherical_test(diffs.values)
```
The idea here is that we fit a sphere around the posterior distribution of differentials (why? It's because convex hulls in higher dimensions is NP-hard).
Once we have a sphere, we test to see if zero is contained in the sphere.  If it is, we reject model, and acknowledge that there could no statistical significance (i.e. none of the microbes had a statistically significant difference).  If `reject` is False, then at least 1 microbe is significantly different -- but we can't say for sure which one due to the compositional bias.  This is where the strength of differential ranking is important, ranking the log-fold changes in `diff` to prioritize the likely candidate microbes that have changed in abundance.

It can be a bit overwhelming to analyze for those not familiar with Bayesian statistics, so make sure to check out to check out the analysis notebooks, particularly [this notebook](https://github.com/mortonjt/asd_multiomics_analyses/blob/main/ipynb/main-differential-notebook.ipynb)

## Computing Matching IDs
You may ask, how exactly does one obtain the matching ids?  Sometimes this is provided upfront (i.e. household matching).  Other times you need to compute this yourself.
If you are in the later category, we have the [utility functions](https://github.com/flatironinstitute/q2-matchmaker/blob/main/q2_matchmaker/_matching.py#L10) just for you!  You can run the template for the following python script

```python
import pandas as pd
md = pd.read_table("<your metadata.txt>", index_col=0)

from q2_matchmaker._matching import _matchmaker

status = 'status'                # specifies if a sample came from case or control (i.e. sick vs healthy)
match_columns = ['Age', 'Sex']   # all of the confounders you'd like to control for
match_types = [False, True]      # specifies if continuous (False) or categorical (True)
match_ids = _matchmaker(md, status, match_columns, match_types)
md['Match_IDs'] = match_ids
md.to_csv("<your updated metadata.txt>", sep='\t')
```
Note that this is just a template for a python script -- you may massage your input data (i.e. drop NAs) before applying the matching.
You also may need to drop NAs after the matching (i.e. if you have an even number of cases, and an odd number of controls, some body is not going to be matched).
The [pandas dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html) will be particularly useful for this.


# SLURM Tutorial (advanced)

If you really want scale this to larger datasets (i.e. >10k features), chances are, you will need to use the SLURM interface. However, note that it will be considerably more difficult to use compared to the qiime2 interface, since it will require some cluster configuration. To use this, you will need to install [disBatch](https://github.com/flatironinstitute/disBatch). This can be installed via
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
    --treatment-group 0 \
    --monte-carlo-samples 1000 \
    --output-inference differentials.nc
```

If this script is saved in a file called `launch.sh`, it can be run as follows
```
sbatch -n 10 -c 4 --mem 8GB launch.sh
```
Where it would be run using 10 processes, each allocated with 4 cores and 8GB per process. An example slurm script is also provided in the examples folder.
You may see a ton of files being generated - these are diagnostics files primarily for debugging.  See the Other considerations sections.

## Other considerations

It is important to note that the computing resources need to be carefully allocated depending on the dataset.  In addition, large datasets with thousands of microbes will generate thousands of microbes, which will put stress on a networked file system.  Therefore, it is important to specify the `--local-directory` to save intermediate files to node local storage, which is much faster to access than the networked file system.  Every system is different, your favorite systems admin will know the location of the node local storage.

## What to do when crap hits the fan?
Persistence is never a guarantee with cluster systems, jobs will fail, in which you may need to relaunch jobs.  You may need to modify the `biom-table` to filter out ids that succeed (see [here](https://biom-format.org/documentation/generated/biom.table.Table.filter.html)). Now q2_matchmaker will store intermediate files for you -- if you didn't specify an intermediate folder, it would be stored until newly created folder called `intermediate`.  Once your relaunched jobs have completed, you can stitch together your individual runs using the `case_control_merge.py` command as follows
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


# Experimental QIIME2 plugin

We did attempt to implement a qiime2 plugin, but fully fleshing this out turned out to be a major software engineering challenge.
The basic MCMC pipeline is available as a qiime2 plugin, but there are no downstream analyses that are currently implemented.

If you still want to use the qiime2 version, go ahead and install the most up-to-date version.

Then install qiime2
```
wget https://data.qiime2.org/distro/core/qiime2-2021.4-py36-linux-conda.yml
conda env create -n qiime2-2021.4 --file qiime2-2021.4-py36-linux-conda.yml
rm qiime2-2020.11-py36-linux-conda.yml
```

There are a few additional dependencies that you will need to install, namely
```
pip install https://github.com/mortonjt/q2-types.git
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


# qiime2 Tutorial (Work In Progress)

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
    --p-treatment-group 0 \
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

The most relevant variable here is `diff` which measures the differentials between the cases and the controls.  You can extract that via `inf.posterior['diff']`.

Otherwise, make sure to check out [Arviz](https://arviz-devs.github.io/arviz/index.html), since it provides an extremely comprehensive API for diagnostics, so it is recommended to check it out.
