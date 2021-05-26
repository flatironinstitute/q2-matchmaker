import qiime2
import numpy as np
import pandas as pd
import arviz as az
import biom
from q2_matchmaker._stan import (
    _case_control_full, _case_control_data,
    _case_control_single)
from q2_differential._matching import _matchmaker
from typing import List, Dict
from birdman.diagnostics import r2_score


def _negative_binomial_case_control(
        table, matching_ids,
        groups, monte_carlo_samples,
        reference_group):
    if reference_group is None:
        reference_group = groups.iloc[0]
    groups_ = (groups == reference_group).astype(np.int64)
    idx = list(set(metadata.index) & set(table.index))
    counts = table.loc[idx]
    metadata = metadata.loc[idx]
    depth = counts.sum(axis=1)
    nb = NegativeBinomialCaseControl(
        table=biom_table,
        matching_column="reps",
        status_column="diff",
        metadata=self.metadata,
        reference_status='1',
        chains=1,
        seed=42)
    # Fit the model and extract diagnostics
    nb.fit_model()
    inf = nb.to_inference_object()
    res = dict(
        r2=r2_score(inf),
        loo=az.loo(inf),
        bfmi=az.bfmi(inf),
        rhat=az.rhat(inf, var_names=nb.param_names),
        ess=az.ess(inf, var_names=nb.param_names)
    )
    return samples, res


def negative_binomial_case_control(
        table: pd.DataFrame,
        matching_ids: qiime2.CategoricalMetadataColumn,
        groups: qiime2.CategoricalMetadataColumn,
        monte_carlo_samples: int = 2000,
        reference_group: str = None,
        cores: int = 1) -> az.InferenceData:
    # Build me a cluster!
    dask_args={'n_workers': cores, 'threads_per_worker': 1}
    cluster = LocalCluster(**dask_args)
    cluster.scale(dask_args['n_workers'])
    client = Client(cluster)
    samples, res = _negative_binomial_case_control(
        table, matching_ids.to_series(),
        groups.to_series(),
        monte_carlo_samples,
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
