from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, TypeVar

import numpy as np
import rasterio
from numpy.typing import DTypeLike

from ._dataset import InputDataset, OutputDataset

__all__ = [
    "Raster",
]


RasterT = TypeVar("RasterT", bound="Raster")


class Raster(InputDataset, OutputDataset, AbstractContextManager["Raster"]):
    """
    A single raster band in a GDAL-compatible dataset containing one or more bands.

    `Raster` provides a convenient interface for using SNAPHU to unwrap ground-projected
    interferograms in raster formats supported by the Geospatial Data Abstraction
    Library (GDAL). It acts as a thin wrapper around a Rasterio dataset and a band
    index, providing NumPy-like access to the underlying raster data.

    Data access is performed lazily -- the raster contents are not stored in memory
    unless/until they are explicitly accessed by an indexing operation.

    `Raster` objects must be closed after use in order to ensure that any data written
    to them is flushed to disk and any associated file objects are closed. The `Raster`
    class implements Python's context manager protocol, which can be used to reliably
    ensure that the raster is closed upon exiting the context manager.
    """

    def __init__(
        self,
        fp: str | os.PathLike[str],
        mode: str = "r",
        band: int = 1,
        driver: str | None = None,
    ):
        """
        Open an existing raster dataset.

        Parameters
        ----------
        fp : str or path-like
            File system path or URL of the local or remote dataset.
        mode : {'r', 'r+'}, optional
            The file access mode -- 'r' for read-only (the default) or 'r+' for
            read/write access.
        band : int, optional
            The (1-based) band index of the raster band. Defaults to 1.
        driver : str or None, optional
            Raster format driver name. If None, registered drivers will be tried
            sequentially until a match is found. Defaults to None.
        """
        # Check `mode` argument.
        if mode in {"w", "w+"}:
            clsname = type(self).__name__
            errmsg = (
                f"{clsname}.__init__() only supports 'r' and 'r+' modes, use"
                f" {clsname}.create() for dataset creation"
            )
            raise ValueError(errmsg)
        if mode not in {"r", "r+"}:
            errmsg = f"unsupported mode {mode!r}, must be one of {{'r', 'r+'}}"
            raise ValueError(errmsg)

        # Open the dataset.
        dataset = rasterio.open(fp, mode=mode, driver=driver)

        # Check that `band` is a valid band index in the dataset.
        nbands = dataset.count
        if not (1 <= band <= nbands):
            errmsg = (
                f"band index {band} out of range: dataset contains {nbands} raster"
                " band(s)"
            )
            raise IndexError(errmsg)

        self._dataset = dataset
        self._band = band

    @classmethod
    def create(
        cls: type[RasterT],
        fp: str | os.PathLike[str],
        width: int | None = None,
        height: int | None = None,
        dtype: DTypeLike | None = None,
        driver: str | None = None,
        crs: str | Mapping[str, str] | rasterio.crs.CRS | None = None,
        transform: rasterio.transform.Affine | None = None,
        nodata: float | None = None,
        *,
        like: Raster | None = None,
        **kwargs: Any,
    ) -> RasterT:
        """
        Create a new single-band raster dataset.

        If another raster is passed via the `like` argument, the new dataset will
        inherit the shape, data-type, driver, coordinate reference system (CRS), and
        geotransform of the reference raster. Driver-specific dataset creation options
        such as chunk size and compression flags may also be inherited.

        All other arguments take precedence over `like` and may be used to override
        attributes of the reference raster when creating the new raster.

        Parameters
        ----------
        fp : str or path-like
            File system path or URL of the local or remote dataset.
        width, height : int or None, optional
            The numbers of columns and rows of the raster dataset. Required if `like` is
            not specified. Otherwise, if None, the new dataset is created with the same
            width/height as `like`. Defaults to None.
        dtype : data-type or None, optional
            Data-type of the raster dataset's elements. Must be convertible to a
            `numpy.dtype` object and must correspond to a valid GDAL datatype. Required
            if `like` is not specified. Otherwise, if None, the new dataset is created
            with the same data-type as `like`. Defaults to None.
        driver : str or None, optional
            Raster format driver name. If None, the method will attempt to infer the
            driver from the file extension. Defaults to None.
        crs : str, dict, rasterio.crs.CRS, or None; optional
            The coordinate reference system. If None, the CRS of `like` will be used, if
            available, otherwise the raster will not be georeferenced. Defaults to None.
        transform : rasterio.transform.Affine or None, optional
            Affine transformation mapping the pixel space to geographic space. If None,
            the geotransform of `like` will be used, if available, otherwise the default
            transform will be used. Defaults to None.
        nodata : float or None, optional
            Defines the pixel value to be interpreted as not valid data. If None, the
            nodata value of `like` will be used, if available, otherwise no nodata value
            will be populated. Only real-valued nodata values are supported, even if the
            raster data is complex-valued. Defaults to None.
        like : Raster or None, optional
            An optional reference raster. If not None, the new raster will be created
            with the same metadata (shape, data-type, driver, CRS/geotransform, etc) as
            the reference raster. All other arguments will override the corresponding
            attribute of the reference raster. Defaults to None.
        **kwargs : dict, optional
            Additional driver-specific creation options passed to `rasterio.open`.
        """
        if like is not None:
            kwargs = like.dataset.profile | kwargs

        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if dtype is not None:
            kwargs["dtype"] = np.dtype(dtype)
        if driver is not None:
            kwargs["driver"] = driver
        if crs is not None:
            kwargs["crs"] = crs
        if transform is not None:
            kwargs["transform"] = transform
        if nodata is not None:
            kwargs["nodata"] = nodata

        # Always create a single-band dataset, even if `like` was part of a multi-band
        # dataset.
        kwargs["count"] = 1

        # Create the new single-band dataset.
        dataset = rasterio.open(fp, mode="w+", **kwargs)

        # XXX We need this gross hack in order to bypass calling `__init__` (which only
        # supports opening existing datasets).
        raster = cls.__new__(cls)
        raster._dataset = dataset  # noqa: SLF001
        raster._band = 1  # noqa: SLF001

        return raster

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dataset.dtypes[self.band - 1])

    @property
    def height(self) -> int:
        """int : The number of rows in the raster."""  # noqa: D403
        return self.dataset.height  # type: ignore[no-any-return]

    @property
    def width(self) -> int:
        """int : The number of columns in the raster."""  # noqa: D403
        return self.dataset.width  # type: ignore[no-any-return]

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dataset(self) -> rasterio.io.DatasetReader | rasterio.io.DatasetWriter:
        """
        rasterio.io.DatasetReader or rasterio.io.DatasetWriter : The underlying dataset.
        """  # noqa: D200
        return self._dataset

    @property
    def band(self) -> int:
        """int : Band index (1-based)."""  # noqa: D403
        return self._band

    @property
    def mode(self) -> str:
        """str : File access mode."""  # noqa: D403
        return self.dataset.mode  # type: ignore[no-any-return]

    @property
    def driver(self) -> str:
        """str : Raster format driver name."""  # noqa: D403
        return self.dataset.driver  # type: ignore[no-any-return]

    @property
    def crs(self) -> rasterio.crs.CRS:
        """rasterio.crs.CRS : The dataset's coordinate reference system."""
        return self.dataset.crs

    @property
    def transform(self) -> rasterio.transform.Affine:
        """
        rasterio.transform.Affine : The dataset's georeferencing transformation matrix.

        This transform maps pixel row/column coordinates to coordinates in the dataset's
        coordinate reference system.
        """
        return self.dataset.transform

    @property
    def nodata(self) -> float | None:
        """
        float or None : The raster's nodata value (may be unset).

        The raster's nodata value, or None if no nodata value was set.
        """
        return self.dataset.nodatavals[self.band - 1]  # type: ignore[no-any-return]

    @property
    def closed(self) -> bool:
        """bool : True if the dataset is closed."""  # noqa: D403
        return self.dataset.closed  # type: ignore[no-any-return]

    def close(self) -> None:
        """
        Close the underlying dataset.

        Has no effect if the dataset is already closed.
        """
        if not self.closed:
            self.dataset.close()

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()

    def __array__(self) -> np.ndarray:
        return self.dataset.read(self.band)

    def _window_from_slices(
        self, key: slice | tuple[slice, ...]
    ) -> rasterio.windows.Window:
        if isinstance(key, slice):
            row_slice = key
            col_slice = slice(None)
        else:
            row_slice, col_slice = key

        return rasterio.windows.Window.from_slices(
            row_slice, col_slice, height=self.height, width=self.width
        )

    def __getitem__(self, key: slice | tuple[slice, ...], /) -> np.ndarray:
        window = self._window_from_slices(key)
        return self.dataset.read(self.band, window=window)

    def __setitem__(self, key: slice | tuple[slice, ...], value: np.ndarray, /) -> None:
        window = self._window_from_slices(key)
        self.dataset.write(value, self.band, window=window)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(dataset={self.dataset!r}, band={self.band!r})"
