import qiime2
import numpy as np
import xarray as xr
import biom


def dirichlet_multinomial(
        table: biom.Table,
        groups: qiime2.CategoricalMetadataColumn,
        training_samples: qiime2.CategoricalMetadataColumn = None,
        percent_test_examples: float = 0.1,
        monte_carlo_samples: int = 1000,
        reference_group: str = None) -> xr.DataArray:
    # Perform train/test split
    groups = groups.to_series()
    if training_samples is not None:
        idx = np.random.random(len(groups)) < percent_test_examples
    else:
        idx = training_samples == 'Test'
    train_samples = set(groups.loc[~idx].index)
    func = lambda v, i, m: i in train_samples
    train_table = table.filter(func, inplace=False)
    func = lambda v, i, m: i not in train_samples
    test_table = table.filter(func, inplace=False)
    cats = list(set(metadata))
    if reference_group is None:
        ref_idx = 0
        reference_group = cats[0]
    else:
        ref_idx = cats.index(reference_group)
    # Compute Multinomial probabilities
    groups = {}
    N, D = train_table.shape
    C = len(cats)
    samples = np.zeros((C, D, monte_carlo_samples))
    for j, c in enumerate(cats):
        samples = set(metadata.index[metadata == c])
        func = lambda v, i, m: i in samples
        subtable = table.filter(func, inplace=False)
        groups[c] = subtable.sum(axis='observations') + 1
        # Draw MCMC samples
        for i in range(monte_carlo_samples):
            samples[j, :, i] = np.random.dirichlet(
                groups[c], size=monte_carlo_samples)
    # Build x-array object
    diffs = samples / samples[ref_idx]
    diffs = diffs[np.array(cats) != ref_idx]
    samples = xr.DataArray(
        diffs,
        dims=['differences', 'features', 'monte_carlo_samples'],
        coords=dict(
            groups=cats,
            features=train_table.ids(axis='observation'),
            monte_carlo_samples=np.arange(monte_carlo_samples)
        ),
        attrs=dict(
            description='Posterior samples of groups',
            reference=reference_group
        )
    )
    return samples
