import qiime2
import numpy as np
import pandas as pd
import arviz as az
from q2_matchmaker._matching import _matchmaker
from typing import List
from q2_matchmaker._stan import _case_control_data
from q2_matchmaker._stan import _case_control_full


def negative_binomial_case_control(
        table: pd.DataFrame,
        matching_ids: qiime2.CategoricalMetadataColumn,
        groups: qiime2.CategoricalMetadataColumn,
        treatment_group: str,
        monte_carlo_samples: int = 2000) -> az.InferenceData:
    reference_group = treatment_group
    metadata = pd.DataFrame({'cc_ids': matching_ids.to_series(),
                             'groups': groups.to_series()})
    metadata['groups'] = (metadata['groups'] == reference_group)
    metadata['groups'] = metadata['groups'].astype(np.int64)

    # take intersection
    idx = list(set(metadata.index) & set(table.index))
    counts = table.loc[idx]
    metadata = metadata.loc[idx]
    depth = counts.sum(axis=1)
    dat = _case_control_data(counts.values,
                             metadata['cc_ids'].values,
                             metadata['groups'].values, depth)
    _, posterior = _case_control_full(
        counts=counts.values,
        case_ctrl_ids=metadata['cc_ids'].values,
        case_member=metadata['groups'].values,
        depth=depth,
        mc_samples=monte_carlo_samples)
    opts = {
        'observed_data': dat,
        'coords': {'diff': list(table.columns[1:])}
    }
    samples = az.from_cmdstanpy(posterior=posterior, **opts)

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
