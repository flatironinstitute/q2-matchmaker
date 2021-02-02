from os.path import join
import pandas as pd
import qiime2
import biom
import pkg_resources
import q2templates
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from q2_differential._stats import hotelling_ttest, spherical_test, rank_test


TEMPLATES = pkg_resources.resource_filename('q2_differential', 'assets')


def _plot(ax, data, cat, reference):
    idx = np.arange(len(data))
    ax.errorbar(idx, data['mean'].values, data['std'].values)
    ax.set_title(f'Category: {cat}, Reference: {reference}')
    ax.set_ylabel('Log fold change + const.')


def rankplot(output_dir: str,
             differentials: xr.DataArray) -> None:
    reference = differentials.attrs['reference']
    contrasts = list(differentials.coords['contrasts'].values)
    diffmean = differentials.mean(dim='monte_carlo_samples')
    diffstd = differentials.std(dim='monte_carlo_samples')
    diffmean = diffmean.to_dataframe(name='mean').reset_index()
    diffstd = diffstd.to_dataframe(name='std').reset_index()
    features = np.array(diffmean['features'].values)
    res, rank_res = dict(), dict()
    for i, cat in enumerate(contrasts):
        diffs = np.array(differentials.values[i]).T
        # Compute Spherical test
        s, r, d = spherical_test(diffs)
        # Compute Hotelling t-test
        #t2, pval = hotelling_ttest(diffs)
        res[cat] = {
            'Posterior-radius': r,
            '0-dist.': d,
            '0-in-sphere': s,
            'Hotelling-T': t2,
            'Hotelling-pval': pval
        }
        # Compute rank tests
        fwd_idx, fwd_rank = rank_test(diffs)
        #bwd_idx, bwd_rank = rank_test(-diffs)  # TODO: add invert option
        rank_res[cat] = pd.Series(fwd_rank, index=features[fwd_idx])
    rank_stats = pd.DataFrame(rank_res).T
    stats = pd.DataFrame(res)
    stats.name = reference
    # Plot results
    fig, ax = plt.subplots(len(contrasts), 1)
    for i, cat in enumerate(contrasts):
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
    rank_stats.to_csv(join(output_dir, 'rank_stats.csv'))
    # TODO: Need to figure out how to download the rank test dataframe
    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'Rank Plot',
        'stats': stats.to_html(),
        'rank_stats_fp': 'rank_stats.csv',
        'pdf_fp': 'ranks.pdf',
        'png_fp': 'ranks.png'})
