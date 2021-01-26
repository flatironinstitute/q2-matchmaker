import importlib
import qiime2.plugin
import qiime2.sdk
from q2_differential import __version__
from q2_types._format import FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt
from q2_types._type import FeatureTensor


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
        'metadata': MetadataColumn[Categorical],
        'training_column': Str,
        'num_random_test_examples': Int
        'monte_carlo_samples': Int
    },
    outputs=[
        ('differentials', FeatureTensor),
        ('differential_stats', SampleData[DifferentialStats]),
    ],
    input_descriptions={
        "table": "Input table of counts.",
    },
    output_descriptions={
        'differentials': ('Output posterior differentials learned from the '
                          'Dirichlet ultinomial.'),
    },
    parameter_descriptions={
        'metadata': ('The categorical sample metadata column to test for '
                     'differential abundance across.'),
        "training-column": (
            'The column in the metadata file used to '
            'specify training and testing. These columns '
            'should be specifically labeled (Train) and (Test).'
        ),
        "num-random-test-examples": (
            'Number of random samples to hold out for cross-validation if '
            'a training column is not specified.'
        )
    },
    name='Dirichilet Multinomial',
    description=("Fits a Dirchilet Multinomial model and computes biased"
                 "log-fold change."),
    citations=[]
)

plugin.register_formats(FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt)
plugin.register_semantic_types(FeatureTensor)

importlib.import_module('songbird.q2._transformer')
