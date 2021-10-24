#!/bin/bash

source ~/.bashrc
conda activate qiime2-2021.4
export TBB_CXX_TYPE=gcc
module load disBatch/2.0-beta
cd /mnt/home/jmorton/software/q2-matchmaker/q2_matchmaker/tests/stress/data/amplicon_comparison


BIOM_Z=zurita.biom
BIOM_D=dan.biom
BIOM_B=berding.biom
MD=sample_metadata.txt
TMP=~/ceph/scratch/q2-matchmaker

# clean up
rm differentials_z.nc
rm differentials_d.nc
rm differentials_b.nc

mkdir $TMP/intermediate_test_z
case_control_disbatch.py \
    --biom-table $BIOM_Z \
    --metadata-file sample_metadata.txt \
    --matching-ids Match_IDs \
    --groups Status \
    --treatment-group 'ASD' \
    --monte-carlo-samples 100 \
    --intermediate-directory $TMP/intermediate_test_z \
    --output-inference differentials_z.nc

mkdir $TMP/intermediate_test_d
case_control_disbatch.py \
    --biom-table $BIOM_D \
    --metadata-file sample_metadata.txt \
    --matching-ids Match_IDs \
    --groups Status \
    --treatment-group 'ASD' \
    --monte-carlo-samples 100 \
    --intermediate-directory $TMP/intermediate_test_d \
    --output-inference differentials_d.nc

mkdir $TMP/intermediate_test_b
case_control_disbatch.py \
    --biom-table $BIOM_B \
    --metadata-file sample_metadata.txt \
    --matching-ids Match_IDs \
    --groups Status \
    --treatment-group 'ASD' \
    --monte-carlo-samples 100 \
    --intermediate-directory $TMP/intermediate_test_b \
    --output-inference differentials_b.nc
