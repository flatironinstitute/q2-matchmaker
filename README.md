# q2-differential
A qiime2 plugin for differential abundance analysis

# Installation (for Linux)

First install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Then install qiime2
```
wget https://data.qiime2.org/distro/core/qiime2-2020.11-py36-linux-conda.yml
conda env create -n qiime2-2020.11 --file qiime2-2020.11-py36-linux-conda.yml
rm qiime2-2020.11-py36-linux-conda.yml
```

Then install q2-differential
```
pip install git+https://github.com/mortonjt/q2-differential.git
qiime dev refresh-cache
```

# Tutorial

Download the files from the Moving Pictures tutorial
```
cd example
wget \
  -O "sample-metadata.tsv" \
  "https://data.qiime2.org/2020.11/tutorials/moving-pictures/sample_metadata.tsv"

wget https://view.qiime2.org/?src=https%3A%2F%2Fdocs.qiime2.org%2F2020.11%2Fdata%2Ftutorials%2Fmoving-pictures%2Ftable-dada2.qza
```

Run the dirichlet multinomial example
```
qiime differential dirichlet-multinomial \
    --i-table table.qza \
    --m-groups-file sample-metadata.tsv \
    --m-groups-column subject \
    --o-differentials ranks.qza
```
Now run tests to summarize the results of dirichlet-multinomial
```
qiime differential rankplot \
    --i-differentials ranks.qza \
    --o-visualization ranks.qzv
```

To visualize the results in a qiime environment run
```
qiime tools view ranks.qzv
```

Note that qiime qzv files are just zip files packaged with html / json files.
Those can be unpackaged with
```
unzip ranks.qzv
```
