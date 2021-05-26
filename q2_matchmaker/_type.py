from qiime2.plugin import SemanticType, model
from q2_types.sample_data import SampleData


Matching = SemanticType('Matching',
                        variant_of=SampleData.field['type'])
