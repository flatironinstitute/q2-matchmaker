import importlib
import qiime2.plugin
import qiime2.sdk
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool, List,
                           MetadataColumn, Categorical)
from q2_matchmaker import __version__
from q2_matchmaker._type import Matching
from q2_matchmaker._format import MatchingFormat, MatchingDirectoryFormat
from q2_types._type import MonteCarloTensor
from q2_types._format import (
    MonteCarloTensorFormat, MonteCarloTensorDirectoryFormat
)
from q2_matchmaker._method import (
    negative_binomial_case_control,
    parallel_negative_binomial_case_control,
    slurm_negative_binomial_case_control, matching
)
from q2_matchmaker._visualizer import rankplot
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.ordination import PCoAResults
from q2_types.sample_data import SampleData
import xarray as xr


plugin = qiime2.plugin.Plugin(
    name='matchmaker',
    version=__version__,
    website="https://github.com/mortonjt/q2-matchmaker",
    citations=[],
    short_description=('Plugin for matchmaker abundance analysis '
                       'via count-based models.'),
    description=('This is a QIIME 2 plugin supporting statistical models on '
                 'feature tables and metadata.'),
    package='q2-matchmaker')


plugin.methods.register_function(
    function=negative_binomial_case_control,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'matching_ids': MetadataColumn[Categorical],
        'groups': MetadataColumn[Categorical],
        'monte_carlo_samples': Int,
        'reference_group': Str,
        'cores': Int
    },
    outputs=[
        ('differentials', MonteCarloTensor)
    ],
    input_descriptions={
        "table": "Input table of counts.",
    },
    output_descriptions={
        'matchmakers': ('Output posterior matchmakers learned from the '
                          'Negative Binomial model.'),
    },
    parameter_descriptions={
        'matching_ids': ('The matching ids to link case-control samples '),
        'groups': ('The categorical sample metadata column to test for '
                     'matchmaker abundance across.'),
        "monte_carlo_samples": (
            'Number of monte carlo samples to draw from '
            'posterior distribution.'
        ),
        "reference_group": (
            'Reference category to compute log-fold change from.'
        ),
        "cores" : ('Number of cores to utilize for parallelism.')
    },
    name='Negative Binomial Case Control Estimation',
    description=("Fits a Negative Binomial model to estimate "
                 "biased log-fold change"),
    citations=[]
)


plugin.methods.register_function(
    function=parallel_negative_binomial_case_control,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'matching_ids': MetadataColumn[Categorical],
        'groups': MetadataColumn[Categorical],
        'monte_carlo_samples': Int,
        'reference_group': Str,
        'cores': Int
    },
    outputs=[
        ('matchmakers', MonteCarloTensor)
    ],
    input_descriptions={
        "table": "Input table of counts.",
    },
    output_descriptions={
        'matchmakers': ('Output posterior matchmakers learned from the '
                          'Negative Binomial model.'),
    },
    parameter_descriptions={
        'matching_ids': ('The matching ids to link case-control samples '),
        'groups': ('The categorical sample metadata column to test for '
                     'matchmaker abundance across.'),
        "monte_carlo_samples": (
            'Number of monte carlo samples to draw from '
            'posterior distribution.'
        ),
        "reference_group": (
            'Reference category to compute log-fold change from.'
        ),
        "cores" : ('Number of cores to utilize for parallelism.')
    },
    name='Negative Binomial Case Contro Parallel Estimation',
    description=("Fits a Negative Binomial model to estimate "
                 "biased log-fold change"),
    citations=[]
)


plugin.methods.register_function(
    function=matching,
    inputs={},
    parameters={
        'sample_metadata': qiime2.plugin.Metadata,
        'status' : Str,
        'match_columns' : List[Str],
        'prefix': Str
    },
    outputs=[
        ('matched_metadata', SampleData[Matching])
    ],
    input_descriptions={
    },
    output_descriptions={
        "matched_metadata": ("Modified metadata with matching ids.")
    },
    parameter_descriptions={
        "sample_metadata": ("Information about the metadata that allows for "
                            "case-control matching across confounders "
                            "such as age, sex and household."),
        'status': ('The experimental condition to be investigated.'),
        'match_columns': ('The confounder covariates to match on.'),
        'prefix': ('A prefix to add to the matching ids'),
    },
    name='Matching',
    description=("Creates matching ids to enable case-control matching."),
    citations=[]
)


plugin.register_formats(MonteCarloTensorFormat, MonteCarloTensorDirectoryFormat,
                        MatchingFormat, MatchingDirectoryFormat)
plugin.register_semantic_types(MonteCarloTensor, Matching)
plugin.register_semantic_type_to_format(
    MonteCarloTensor, MonteCarloTensorDirectoryFormat)
plugin.register_semantic_type_to_format(
    SampleData[Matching], MatchingDirectoryFormat)

importlib.import_module('q2_matchmaker._transformer')
