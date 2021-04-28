from qiime2.plugin import SemanticType, model
from q2_types.sample_data import SampleData


# Tensor type sorting information about features
FeatureTensor = SemanticType('FeatureTensor')


# differential stats summarizing training / testing error
DifferentialStats = SemanticType('DifferentialStats',
                                 variant_of=SampleData.field['type'])

Matching = SemanticType('Matching',
                        variant_of=SampleData.field['type'])
