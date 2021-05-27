import qiime2
import numpy as np
import pandas as pd
import arviz as az
import biom
from q2_matchmaker._matching import _matchmaker
from q2_matchmaker._stan import NegativeBinomialCaseControl
from gneiss.util import match
from dask.distributed import Client, LocalCluster
from typing import List


def _negative_binomial_case_control(
        table, matching_ids,
        groups, reference_group, **sampler_args):
    if reference_group is None:
        reference_group = groups.iloc[0]
    groups_ = (groups == reference_group).astype(np.int64)
    metadata = pd.DataFrame({
        groups_.name: groups_,
        matching_ids.name: matching_ids})
    table, metadata = match(table, metadata)
    nb = NegativeBinomialCaseControl(
        table=table,
        matching_column=matching_ids.name,
        status_column=groups.name,
        metadata=metadata,
        reference_status=reference_group,
        **sampler_args)
    # Fit the model and extract diagnostics
    nb.compile_model()
    nb.fit_model(convert_to_inference=True)
    samples = nb.to_inference_object()
    return samples


def negative_binomial_case_control(
        table: biom.Table,
        matching_ids: qiime2.CategoricalMetadataColumn,
        groups: qiime2.CategoricalMetadataColumn,
        reference_group: str = None,
        cores: int = 1) -> az.InferenceData:
    # Build me a cluster!
    dask_args = {'n_workers': cores, 'threads_per_worker': 1}
    cluster = LocalCluster(**dask_args)
    cluster.scale(dask_args['n_workers'])
    Client(cluster)
    samples = _negative_binomial_case_control(
        table, matching_ids.to_series(),
        groups.to_series(),
        reference_group)
    return samples


def matching(sample_metadata: qiime2.Metadata,
             status: str,
             match_columns: List[str],
             prefix: str = None) -> qiime2.Metadata:
    new_column = 'matching_id'
    columns = [sample_metadata.get_column(col) for col in match_columns]
    types = [isinstance(m, qiime2.CategoricalMetadataColumn) for m in columns]
    sample_metadata = sample_metadata.to_dataframe()
    match_ids = _matchmaker(sample_metadata, status, match_columns, types)
    new_metadata = sample_metadata.copy()
    new_metadata[new_column] = match_ids
    # drop any nans that may appear due to lack of matching
    new_metadata = new_metadata.dropna(subset=[new_column])
    new_metadata[new_column] = new_metadata[new_column].astype(
        np.int64).astype(str)
    if prefix is not None:
        new_metadata[new_column] = new_metadata[new_column].apply(
            lambda x: f'{prefix}_{x}')
    return qiime2.Metadata(new_metadata)
