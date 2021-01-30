from os.path import join
import pandas as pd
import qiime2
import biom
import pkg_resources
import q2templates
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


TEMPLATES = pkg_resources.resource_filename('q2_differential', 'assets')


def _plot(ax, data, cat, reference):
    idx = np.arange(len(data))
    ax.errorbar(idx, data['mean'].values, data['std'].values)
    ax.set_title(f'Category: {cat}, Reference: {reference}')
    ax.set_ylabel('Log fold change + const.')


def rankplot(output_dir: str,
             differentials: xr.DataArray) -> None:
    reference = differentials.attrs['reference']
    contrasts = list(differentials.coords['contrasts'])
    diffmean = differentials.mean(dim='monte_carlo_samples')
    diffstd = differentials.std(dim='monte_carlo_samples')
    diffmean = diffmean.to_dataframe(name='mean').reset_index()
    diffstd = diffstd.to_dataframe(name='std').reset_index()
    # TODO: perform global tests, namely
    # Convex Hull Test
    # Hotelling T-test
    # TODO: perform rank tests to compute per-microbe pvalues
    fig, ax = plt.subplots(len(contrasts), 1)
    for i, cat in enumerate(contrasts):
        cat = cat.values
        mean = diffmean.loc[diffmean['contrasts'] == cat, 'mean'].values
        std = diffstd.loc[diffstd['contrasts'] == cat, 'std'].values
        data = pd.DataFrame({'mean': mean, 'std': std})
        data = data.sort_values('mean')
        if len(contrasts) > 1:
            _plot(ax[i], data, cat, reference)
        else:
            _plot(ax, data, cat, reference)

    fig.savefig(join(output_dir, 'ranks.pdf'), bbox_inches='tight')
    fig.savefig(join(output_dir, 'ranks.png'), bbox_inches='tight')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'Rank Plot',
        'pdf_fp': 'ranks.pdf',
        'png_fp': 'ranks.png'})
