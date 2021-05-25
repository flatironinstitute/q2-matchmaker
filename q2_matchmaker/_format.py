import qiime2.plugin.model as model


class MatchingFormat(model.TextFileFormat):
    def validate(*args):
        pass


MatchingDirectoryFormat = model.SingleFileDirectoryFormat(
    'MatchingDirectoryFormat', 'matching.tsv', MatchingFormat)
