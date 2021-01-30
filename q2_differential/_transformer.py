from q2_differential.plugin_setup import plugin
from q2_differential._format import FeatureTensorNetCDFFormat
import xarray as xr


@plugin.register_transformer
def _100(ff : FeatureTensorNetCDFFormat) -> xr.DataArray:
    return xr.open_dataarray(str(ff))


@plugin.register_transformer
def _101(tensor : xr.DataArray) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff
