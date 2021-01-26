import xarray as xr


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


plugin.register_views(FeatureTensorCDFFormat, FeatureTensorDirFmt,
                      xr.DataArray)


# class DifferentialStatsFormat(model.TextFileFormat):
#     def validate(*args):
#         pass
#
#
# DifferentialStatsDirFmt = model.SingleFileDirectoryFormat(
#     'DifferentialStatsDirFmt', 'stats.tsv', DifferentialStatsFormat)
