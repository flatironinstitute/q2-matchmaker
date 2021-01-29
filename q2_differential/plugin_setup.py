import importlib
import qiime2.plugin
import qiime2.sdk
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool,
                           MetadataColumn, Categorical)
from q2_differential import __version__
from q2_differential._type import FeatureTensor
from q2_differential._format import FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt
from q2_differential._method import dirichlet_multinomial
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
        # ('differential_stats', SampleData[DifferentialStats]),
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

plugin.register_semantic_types(FeatureTensor)
plugin.register_views(FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt,
                      xr.DataArray)
# citations.register_formats(FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt)

importlib.import_module('q2_differential._transformer')
