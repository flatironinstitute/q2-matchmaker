import unittest
import qiime2
from q2_differential._method import (
    dirichlet_multinomial, negative_binomial_case_control,
    parallel_negative_binomial_case_control)
from q2_differential._stan import (
    _case_control_sim, _case_control_full, _case_control_data)
from skbio.stats.composition import clr_inv
import biom
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
from scipy.stats import pearsonr


def sim_multinomial(N, D, C, depth=1000):
    """ Simulate Multinomial counts. """
    counts = np.zeros((N, D))
    cats = np.arange(C)
    groups = np.random.choice(cats, size=N)
    means = np.random.randn(C, D)
    differentials = np.log((clr_inv(means) / clr_inv(means[0]))[1:])
    for i in range(N):
        p = clr_inv(means[groups[i]])
        n = np.random.poisson(depth)
        counts[i] = np.random.multinomial(n, p)
    return counts, groups, differentials


class TestDirichiletMultinomial(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        N = 50
        D = 100
        C = 3
        self.counts, groups, self.diffs = sim_multinomial(N, D, C)
        sids = [f's{i}' for i in range(N)]
        oids = [f'f{i}' for i in range(D)]
        self.table = biom.Table(self.counts.T, oids, sids)
        self.groups = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, groups)),
                      index=pd.Index(sids, name='id'),
                      name='n')
        )
        self.N, self.D, self.C = N, D, C

    def test_dirichlet_multinomial(self):
        mc_samples = 100
        res_diffs = dirichlet_multinomial(self.table, self.groups,
                                          monte_carlo_samples=mc_samples,
                                          reference_group='0')
        res_diffs_np = res_diffs.values
        self.assertEqual(tuple(list(res_diffs_np.shape)),
                         (self.C - 1, self.D, mc_samples))
        for i in range(res_diffs_np.shape[0]):
            for j in range(res_diffs_np.shape[2]):
                r, p = pearsonr(res_diffs_np[i, :, j], self.diffs[i])
                self.assertGreater(r, 0.95)
                self.assertLess(p, 1e-8)


class TestNegativeBinomialCaseControl(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.N, self.D = 50, 4
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=4, depth=100)

    def test_negative_binomial_case_control(self):
        sids = [f's{i}' for i in range(self.N)]
        oids = [f'f{i}' for i in range(self.D)]
        matchings = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, self.metadata['reps'])),
                      index=pd.Index(sids, name='id'),
                      name='n'))
        diffs = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, self.metadata['diff'])),
                      index=pd.Index(sids, name='id'),
                      name='n'))
        res = negative_binomial_case_control(
            self.table,
            matchings, diffs,
            monte_carlo_samples = 100,
            reference_group = '0',
            cores = 1)
        self.assertIsInstance(res, az.InferenceData)


class TestNegativeBinomialCaseControlParallel(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.N, self.D = 50, 4
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=4, depth=100)

    def test_negative_binomial_case_control_parallel(self):
        sids = [f's{i}' for i in range(self.N)]
        oids = [f'f{i}' for i in range(self.D)]
        matchings = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, self.metadata['reps'])),
                      index=pd.Index(sids, name='id'),
                      name='n'))
        diffs = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, self.metadata['diff'])),
                      index=pd.Index(sids, name='id'),
                      name='n'))
        res = parallel_negative_binomial_case_control(
            self.table,
            matchings, diffs,
            monte_carlo_samples = 100,
            reference_group = '0',
            cores = 4)
        self.assertIsInstance(res, xr.Dataset)


if __name__ == '__main__':
    unittest.main()
