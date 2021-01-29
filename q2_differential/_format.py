import xarray as xr
import qiime2.plugin.model as model
from q2_differential.plugin_setup import plugin


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


plugin.register_views(FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt,
                      xr.DataArray)


# class DifferentialStatsFormat(model.TextFileFormat):
#     def validate(*args):
#         pass
#
#
# DifferentialStatsDirFmt = model.SingleFileDirectoryFormat(
#     'DifferentialStatsDirFmt', 'stats.tsv', DifferentialStatsFormat)
