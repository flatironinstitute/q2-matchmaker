import unittest
import qiime2
from q2_matchmaker._method import (
    negative_binomial_case_control,
    matching)
from q2_matchmaker._stan import _case_control_sim

import biom
import numpy as np
import pandas as pd
import arviz as az
import pandas.util.testing as pdt


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

    @unittest.skip('qiime2 not supported yet.')
    def test_matching(self):
        matched_metadata = matching(self.metadata, 'Diagnosis', ['Age', 'Sex'])
        matched_metadata = matched_metadata.to_dataframe()
        pdt.assert_series_equal(
            matched_metadata['matching_id'],
            pd.Series(['0', '0', '1', '1', '2', '2', '3', '3'],
                      index=self.index, name='matching_id')
        )

    @unittest.skip('qiime2 not supported yet.')
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

    @unittest.skip('qiime2 not supported yet.')
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
            metadata, 'Diagnosis', ['Age', 'Sex'])
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

    @unittest.skip('qiime2 not supported yet.')
    def test_negative_binomial_case_control(self):
        sids = [f's{i}' for i in range(self.N)]
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
        res = negative_binomial_case_control(
            biom_table,
            matchings, diffs,
            monte_carlo_samples=100,
            reference_group='0')
        self.assertIsInstance(res, az.InferenceData)


if __name__ == '__main__':
    unittest.main()
