import qiime2
from q2_differential.plugin_setup import plugin
from q2_differential._format import FeatureTensorNetCDFFormat, MatchingFormat
import xarray as xr
import arviz as az


@plugin.register_transformer
def _100(ff : FeatureTensorNetCDFFormat) -> xr.DataArray:
    return xr.open_dataarray(str(ff))


@plugin.register_transformer
def _101(tensor : xr.DataArray) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff

@plugin.register_transformer
def _102(ff : FeatureTensorNetCDFFormat) -> xr.Dataset:
    return xr.open_dataset(str(ff))


@plugin.register_transformer
def _103(tensor : xr.Dataset) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff


@plugin.register_transformer
def _104(ff : FeatureTensorNetCDFFormat) -> az.InferenceData:
    return xr.from_netcdf(str(ff))


@plugin.register_transformer
def _105(tensor : az.InferenceData) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff


@plugin.register_transformer
def _106(ff: MatchingFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))


@plugin.register_transformer
def _107(obj: qiime2.Metadata) -> MatchingFormat:
    ff = MatchingFormat()
    obj.save(str(ff))
    return ff


# @plugin.register_transformer
# def _108(ff: DifferentialStatsFormat) -> qiime2.Metadata:
#     return qiime2.Metadata.load(str(ff))
#
#
# @plugin.register_transformer
# def _109(obj: qiime2.Metadata) -> DifferentialStatsFormat:
#     ff = DifferentialStatsFormat()
#     obj.save(str(ff))
#     return ff
