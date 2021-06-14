import qiime2
from q2_matchmaker.plugin_setup import plugin
from q2_matchmaker._format import MatchingFormat


@plugin.register_transformer
def _106(ff: MatchingFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))


@plugin.register_transformer
def _107(obj: qiime2.Metadata) -> MatchingFormat:
    ff = MatchingFormat()
    obj.save(str(ff))
    return ff
