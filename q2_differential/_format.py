import xarray as xr
import qiime2.plugin.model as model


class FeatureTensorNetCDFFormat(model.BinaryFileFormat):

    def sniff(self):
        try:
            xr.open_dataset(str(self))
            return True
        except Exception:
            return False


# TODO: add text representation

FeatureTensorNetCDFDirFmt = model.SingleFileDirectoryFormat(
    'FeatureTensorNetCDFDirFmt', 'feature-tensor.nc', FeatureTensorNetCDFFormat)


class MatchingFormat(model.TextFileFormat):
    def validate(*args):
        pass


MatchingDirectoryFormat = model.SingleFileDirectoryFormat(
    'MatchingDirectoryFormat', 'matching.tsv',
    MatchingFormat)


# class DifferentialStatsFormat(model.TextFileFormat):
#     def validate(*args):
#         pass
#
#
# DifferentialStatsDirFmt = model.SingleFileDirectoryFormat(
#     'DifferentialStatsDirFmt', 'stats.tsv', DifferentialStatsFormat)
