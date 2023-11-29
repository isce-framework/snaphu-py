from typing import Any

from ._dataset import InputDataset, OutputDataset

__all__ = [
    "InputDataset",
    "OutputDataset",
]


def __getattr__(name: str) -> Any:
    if name == "Raster":
        # This module depends on `rasterio` so load it lazily to avoid an ImportError
        # for users that don't need `Raster`.
        from ._raster import Raster

        return Raster

    if name in __all__:
        return globals()[name]

    errmsg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(errmsg)


def __dir__() -> list[str]:
    try:
        from ._raster import Raster
    except ModuleNotFoundError:
        return __all__
    else:
        return sorted([*__all__, "Raster"])
