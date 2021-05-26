import unittest
import qiime2
from q2_matchmaker._method import (
    amplicon_case_control,
    matching)
from q2_matchmaker._stan import _case_control_sim
from skbio.stats.composition import clr, clr_inv, alr_inv
import biom
import numpy as np
import pandas as pd
import arviz as az
import pandas.util.testing as pdt


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


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.index = pd.Index(['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4'],
                              name='sampleid')
        self.metadata = qiime2.Metadata(pd.DataFrame(
            [
                ['male', 6, 'ASD'],
                ['male', 6, 'Control'],
                ['female', 6, 'ASD'],
                ['female', 6, 'Control'],
                ['male', 7, 'ASD'],
                ['male', 7, 'Control'],
                ['female', 8, 'ASD'],
                ['female', 8, 'Control'],
            ],
            index=self.index,
            columns=['Sex', 'Age', 'Diagnosis']
        ))

    def test_matching(self):
        matched_metadata = matching(self.metadata, 'Diagnosis', ['Age', 'Sex'])
        matched_metadata = matched_metadata.to_dataframe()
        pdt.assert_series_equal(
            matched_metadata['matching_id'],
            pd.Series(['0', '0', '1', '1', '2', '2', '3', '3'],
                      index=self.index, name='matching_id')
        )

    def test_matching_prefix(self):
        matched_metadata = matching(
            self.metadata, 'Diagnosis', ['Age', 'Sex'], prefix='cool')
        matched_metadata = matched_metadata.to_dataframe()
        pdt.assert_series_equal(
            matched_metadata['matching_id'],
            pd.Series(['cool_0', 'cool_0',
                       'cool_1', 'cool_1',
                       'cool_2', 'cool_2',
                       'cool_3', 'cool_3'],
                      index=self.index, name='matching_id')
        )

    def test_matching_nans(self):
        self.index = pd.Index(['a1', 'a2', 'a3', 'a4',
                               'b1', 'b2', 'b3', 'b4', 'b5'],
                              name='sampleid')

        metadata = qiime2.Metadata(pd.DataFrame(
            [
                ['male', 6, 'ASD'],
                ['male', 6, 'Control'],
                ['female', 6, 'ASD'],
                ['female', 6, 'Control'],
                ['male', 7, 'ASD'],
                ['male', 7, 'Control'],
                ['female', 8, 'ASD'],
                ['female', 8, 'Control'],
                ['female', 10, 'Control'],
            ],
            index=self.index,
            columns=['Sex', 'Age', 'Diagnosis']
        ))

        matched_metadata = matching(
            self.metadata, 'Diagnosis', ['Age', 'Sex'])
        matched_metadata = matched_metadata.to_dataframe()

        index = pd.Index(['a1', 'a2', 'a3', 'a4',
                          'b1', 'b2', 'b3', 'b4'],
                         name='sampleid')
        pdt.assert_series_equal(
            matched_metadata['matching_id'],
            pd.Series(['0', '0', '1', '1', '2', '2', '3', '3'],
                      index=index, name='matching_id')
        )


class TestNegativeBinomialCaseControl(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.N, self.D = 50, 3
        self.table, self.metadata, self.diff = _case_control_sim(
            n=self.N, d=self.D, depth=100)

    def test_amplicon_case_control(self):
        sids = [f's{i}' for i in range(self.N)]
        oids = [f'f{i}' for i in range(self.D)]
        biom_table = biom.Table(self.table.values.T,
                                list(self.table.columns),
                                list(self.table.index))
        matchings = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, self.metadata['reps'])),
                      index=pd.Index(sids, name='id'),
                      name='n'))
        diffs = qiime2.CategoricalMetadataColumn(
            pd.Series(list(map(str, self.metadata['diff'])),
                      index=pd.Index(sids, name='id'),
                      name='n'))
        samples = amplicon_case_control(
            biom_table,
            matchings, diffs,
            reference_group = '0',
            cores = 4)
        self.assertIsInstance(samples, az.InferenceData)


if __name__ == '__main__':
    unittest.main()
