import importlib
import qiime2.plugin
import qiime2.sdk
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool, List,
                           MetadataColumn, Categorical)
from q2_differential import __version__
from q2_differential._type import FeatureTensor, Matching
from q2_differential._format import (
    FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt,
    MatchingFormat, MatchingDirectoryFormat
    # DifferentialStatsFormat, DifferentialStatsDirectoryFormat
)
from q2_differential._method import (
    dirichlet_multinomial, negative_binomial_case_control,
    parallel_negative_binomial_case_control,
    slurm_negative_binomial_case_control, matching
)
from q2_differential._visualizer import rankplot
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.ordination import PCoAResults
from q2_types.sample_data import SampleData
from q2_types.feature_data import (FeatureData, Differential)
import xarray as xr


plugin = qiime2.plugin.Plugin(
    name='differential',
    version=__version__,
    website="https://github.com/mortonjt/q2-differential",
    citations=[],
    short_description=('Plugin for differential abundance analysis '
                       'via count-based models.'),
    description=('This is a QIIME 2 plugin supporting statistical models on '
                 'feature tables and metadata.'),
    package='q2-differential')


plugin.methods.register_function(
    function=dirichlet_multinomial,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'groups': MetadataColumn[Categorical],
        'training_samples': MetadataColumn[Categorical],
        'percent_test_examples': Float,
        'monte_carlo_samples': Int,
        'reference_group': Str
    },
    outputs=[
        ('differentials', FeatureTensor)
    ],
    input_descriptions={
        "table": "Input table of counts.",
    },
    output_descriptions={
        'differentials': ('Output posterior differentials learned from the '
                          'Dirichlet Multinomial.'),
    },
    parameter_descriptions={
        'groups': ('The categorical sample metadata column to test for '
                     'differential abundance across.'),
        "training_samples": (
            'The column in the metadata file used to '
            'specify training and testing. These columns '
            'should be specifically labeled (Train) and (Test).'
        ),
        "percent_test_examples": (
            'Percentage of random samples to hold out for cross-validation if '
            'a training column is not specified.'
        ),
        "monte_carlo_samples": (
            'Number of monte carlo samples to draw from '
            'posterior distribution.'
        ),
        "reference_group": (
            'Reference category to compute log-fold change from.'
        )
    },
    name='Dirichilet Multinomial',
    description=("Fits a Dirchilet Multinomial model and computes biased"
                 "log-fold change."),
    citations=[]
)


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
        ('differentials', FeatureTensor)
    ],
    input_descriptions={
        "table": "Input table of counts.",
    },
    output_descriptions={
        'differentials': ('Output posterior differentials learned from the '
                          'Negative Binomial model.'),
    },
    parameter_descriptions={
        'matching_ids': ('The matching ids to link case-control samples '),
        'groups': ('The categorical sample metadata column to test for '
                     'differential abundance across.'),
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
        ('differentials', FeatureTensor)
    ],
    input_descriptions={
        "table": "Input table of counts.",
    },
    output_descriptions={
        'differentials': ('Output posterior differentials learned from the '
                          'Negative Binomial model.'),
    },
    parameter_descriptions={
        'matching_ids': ('The matching ids to link case-control samples '),
        'groups': ('The categorical sample metadata column to test for '
                     'differential abundance across.'),
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

plugin.visualizers.register_function(
    function=rankplot,
    inputs={'differentials': FeatureTensor},
    parameters={},
    input_descriptions={'differentials': 'Differentials or log-fold changes.'},
    parameter_descriptions={},
    name='Rank plot',
    description="Generate a rank plot of the log-fold changes",
    citations=[]
)


plugin.register_formats(FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt,
                        MatchingFormat, MatchingDirectoryFormat)
plugin.register_semantic_types(FeatureTensor, Matching)
plugin.register_semantic_type_to_format(
    FeatureTensor, FeatureTensorNetCDFDirFmt)
plugin.register_semantic_type_to_format(
    SampleData[Matching], MatchingDirectoryFormat)

importlib.import_module('q2_differential._transformer')
