#!/bin/sh

source ~/.bashrc
conda activate qiime2-2021.4
export TBB_CXX_TYPE=gcc

case_control_parallel.py \
    --biom-table table.biom \
    --metadata-file sample_metadata.txt \
    --matching-ids reps \
    --groups diff \
    --treatment-group 0 \
    --monte-carlo-samples 100 \
    --processes 10 \
    --output-inference differentials.nc
