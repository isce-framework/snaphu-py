from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import DTypeLike

import snaphu

# Maximum value representable by 32-bit unsigned integer type.
UINT32_MAX = np.iinfo(np.uint32).max


def has_rasterio() -> bool:
    """Check if `rasterio` can be imported."""
    return importlib.util.find_spec("rasterio") is not None


@contextmanager
def make_geotiff_raster(
    fp: str | os.PathLike[str],
    shape: tuple[int, int] = (1024, 512),
    dtype: DTypeLike = np.int32,
    epsg: int = 4326,
    extents: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    nodata: float | None = None,
) -> Generator[snaphu.io.Raster, None, None]:
    """
    Make a dummy GeoTiff raster for testing.

    Upon exiting the context manager, the raster dataset is closed.

    Parameters
    ----------
    fp : path-like
        File system path of the dataset. The parent directory must exist.
    shape : (int, int), optional
        The raster shape (height and width).
    dtype : data-type, optional
        The raster data-type. Must be convertible to `numpy.dtype`.
    epsg : int, optional
        An EPSG code defining the coordinate reference system (CRS) of the raster.
    extents : (float, float, float, float), optional
        The geospatial extents of the raster image, in coordinates defined by the CRS
        represented by the `epsg` code, in the following order: West, South, East,
        North.
    nodata : float or None, optional
        Defines the pixel value to be interpreted as not valid data.

    Yields
    ------
    raster : snaphu.io.Raster
        The raster.
    """
    import rasterio

    height, width = shape
    crs = rasterio.crs.CRS.from_epsg(epsg)
    transform = rasterio.transform.from_bounds(*extents, width=width, height=height)
    with snaphu.io.Raster.create(
        fp,
        width=width,
        height=height,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        driver="GTiff",
    ) as raster:
        yield raster


@contextmanager
def make_temp_geotiff_raster() -> Generator[snaphu.io.Raster, None, None]:
    """
    Make a temporary dummy GeoTiff raster dataset for testing.

    Upon exiting the context manager, the raster dataset is closed and the file is
    removed from the file system.

    Yields
    ------
    raster : snaphu.io.Raster
        The raster.
    """
    with (
        tempfile.NamedTemporaryFile(suffix=".tif") as file_,
        make_geotiff_raster(file_.name) as raster,
    ):
        yield raster


@pytest.mark.skipif(not has_rasterio(), reason="requires rasterio package")
class TestRaster:
    @pytest.fixture(scope="class")
    def geotiff_raster(self) -> Generator[snaphu.io.Raster, None, None]:
        with make_temp_geotiff_raster() as raster:
            yield raster

    def test_is_input_dataset(self, geotiff_raster: snaphu.io.Raster):
        assert isinstance(geotiff_raster, snaphu.io.InputDataset)

    def test_is_output_dataset(self, geotiff_raster: snaphu.io.Raster):
        assert isinstance(geotiff_raster, snaphu.io.OutputDataset)

    def test_open_raster(self):
        with tempfile.NamedTemporaryFile(suffix=".tif") as file_:
            with make_geotiff_raster(file_.name):
                pass

            assert Path(file_.name).is_file()

            with snaphu.io.Raster(file_.name) as raster:
                assert raster.mode == "r"

            with snaphu.io.Raster(file_.name, mode="r+") as raster:
                assert raster.mode == "r+"

    def test_create_like(self, geotiff_raster: snaphu.io.Raster):
        with (
            tempfile.NamedTemporaryFile(suffix=".tif") as file_,
            snaphu.io.Raster.create(file_.name, like=geotiff_raster) as new_raster,
        ):
            assert new_raster.shape == geotiff_raster.shape
            assert new_raster.dtype == geotiff_raster.dtype
            assert new_raster.driver == geotiff_raster.driver
            assert new_raster.crs == geotiff_raster.crs
            assert new_raster.transform == geotiff_raster.transform

            assert new_raster.band == 1
            assert new_raster.dataset.count == 1

    def test_create_like_override(self, geotiff_raster: snaphu.io.Raster):
        with tempfile.TemporaryDirectory() as dir_:
            fp = Path(dir_) / "raster"
            with snaphu.io.Raster.create(
                fp,
                like=geotiff_raster,
                dtype="f8",
                driver="ENVI",
            ) as new_raster:
                assert new_raster.shape == geotiff_raster.shape
                assert new_raster.crs == geotiff_raster.crs
                assert new_raster.transform == geotiff_raster.transform
                assert new_raster.dtype == np.float64
                assert new_raster.driver == "ENVI"

    def test_shape(self, geotiff_raster: snaphu.io.Raster):
        shape = (1024, 512)
        assert geotiff_raster.shape == shape
        assert geotiff_raster.height == shape[0]
        assert geotiff_raster.width == shape[1]

    def test_dtype(self, geotiff_raster: snaphu.io.Raster):
        assert isinstance(geotiff_raster.dtype, np.dtype)
        assert geotiff_raster.dtype == np.int32

    def test_ndim(self, geotiff_raster: snaphu.io.Raster):
        assert geotiff_raster.ndim == 2

    def test_band(self, geotiff_raster: snaphu.io.Raster):
        assert geotiff_raster.band == 1

    def test_mode(self, geotiff_raster: snaphu.io.Raster):
        assert geotiff_raster.mode == "w+"

    def test_driver(self, geotiff_raster: snaphu.io.Raster):
        assert geotiff_raster.driver == "GTiff"

    def test_crs(self, geotiff_raster: snaphu.io.Raster):
        import rasterio

        assert geotiff_raster.crs == rasterio.crs.CRS.from_epsg(4326)

    def test_transform(self, geotiff_raster: snaphu.io.Raster):
        import rasterio

        height, width = 1024, 512
        transform = rasterio.transform.from_bounds(0.0, 0.0, 1.0, 1.0, width, height)
        assert geotiff_raster.transform == transform

    @pytest.mark.parametrize(
        ("dtype", "nodata"),
        [
            (np.int32, -999.0),
            (np.uint32, UINT32_MAX),
            (np.float64, 123.456),
            (np.complex64, 0.0),
            (np.float32, None),
        ],
    )
    def test_nodata(self, dtype: DTypeLike, nodata: float | None):
        with (
            tempfile.NamedTemporaryFile(suffix=".tif") as file_,
            make_geotiff_raster(file_.name, dtype=dtype, nodata=nodata) as raster,
        ):
            assert raster.nodata == nodata

    def test_open_closed(self):
        with make_temp_geotiff_raster() as raster:
            assert not raster.closed
        assert raster.closed

    def test_arraylike(self, geotiff_raster: snaphu.io.Raster):
        arr = np.asarray(geotiff_raster)
        assert arr.shape == geotiff_raster.shape
        assert arr.dtype == geotiff_raster.dtype

    def test_setitem_getitem_roundtrip(self, geotiff_raster: snaphu.io.Raster):
        data = np.arange(20, dtype=np.int32).reshape(4, 5)
        idx = np.s_[100:104, 200:205]
        geotiff_raster[idx] = data
        out = geotiff_raster[idx]
        np.testing.assert_array_equal(out, data)

    def test_repr(self, geotiff_raster: snaphu.io.Raster):
        dataset = repr(geotiff_raster.dataset)
        band = repr(geotiff_raster.band)
        assert repr(geotiff_raster) == f"Raster(dataset={dataset}, band={band})"

    def test_bad_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".tif") as file_:
            fp = file_.name
            with make_geotiff_raster(fp):
                pass

            regex = re.compile(r"unsupported mode 'q', must be one of \{'r', 'r\+'\}")
            with pytest.raises(ValueError, match=regex):
                snaphu.io.Raster(fp, mode="q")

    def test_bad_band(self):
        with tempfile.NamedTemporaryFile(suffix=".tif") as file_:
            fp = file_.name
            with make_geotiff_raster(fp):
                pass

            regex = re.compile(
                r"band index 2 out of range: dataset contains 1 raster band\(s\)"
            )
            with pytest.raises(IndexError, match=regex):
                snaphu.io.Raster(fp, band=2)
