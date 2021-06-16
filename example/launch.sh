#!/bin/sh

source ~/.bashrc
conda activate qiime2-2021.4
module load disBatch/2.0-beta
export TBB_CXX_TYPE=gcc
case_control_disbatch.py \
    --biom-table table.biom \
    --metadata-file sample_metadata.txt \
    --matching-ids reps \
    --groups diff \
    --treatment-group 0 \
    --monte-carlo-samples 1000 \
    --local-directory /scratch \
    --output-inference differentials.nc
