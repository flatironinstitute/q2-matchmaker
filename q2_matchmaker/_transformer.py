import qiime2
from q2_matchmaker.plugin_setup import plugin
from q2_matchmaker._format import MatchingFormat
import pandas as pd


def _read_matching(fh):
    # Using `dtype=object` and `set_index` to avoid type casting/inference
    # of any columns or the index.
    df = pd.read_csv(fh, sep='\t', header=0, dtype=object)
    df.set_index(df.columns[0], drop=True, append=False, inplace=True)
    df.index.name = None
    return df


@plugin.register_transformer
def _106(ff: MatchingFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))


@plugin.register_transformer
def _107(obj: qiime2.Metadata) -> MatchingFormat:
    ff = MatchingFormat()
    obj.save(str(ff))
    return ff


@plugin.register_transformer
def _108(ff: MatchingFormat) -> pd.Series:
    df = _read_matching(ff)
    df.index.name = 'Sample ID'
    return df


@plugin.register_transformer
def _109(obj: pd.Series) -> MatchingFormat:
    ff = MatchingFormat()
    with ff.open() as fh:
        obj.to_csv(fh, sep='\t', header=True)
    return ff
