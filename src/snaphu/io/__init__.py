from ._dataset import InputDataset, OutputDataset

__all__ = [
    "InputDataset",
    "OutputDataset",
    "Raster",
]


# The `_raster` submodule depends on the `rasterio` package (an optional dependency).
# Failing to import the module should be non-fatal, but we shouldn't expose its contents
# in that case.
# Note: This implementation is preferred over PEP 562-style `__getattr__` and `__dir__`
# functions due to better compatibility with Pylance.
try:
    from ._raster import Raster
except ModuleNotFoundError:
    __all__.remove("Raster")
