import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import DTypeLike

from snaphu._util import read_from_file, scratch_directory, slices, write_to_file


class TestSlices:
    def test_is_iterator(self):
        s = slices(0, 100, 10)
        next(s)

    def test_start_stop(self):
        s = list(slices(0, 3))
        assert s == [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None)]

    def test_start_stop_step(self):
        s = list(slices(10, 20, 4))
        assert s == [slice(10, 14, None), slice(14, 18, None), slice(18, 20, None)]

    @pytest.mark.parametrize("step", [0, -1])
    def test_bad_step(self, step: int):
        errmsg = f"^step must be >= 1, instead got {step}$"
        with pytest.raises(ValueError, match=errmsg):
            list(slices(0, 100, step))

    def test_empty(self):
        s1 = list(slices(100, 100))
        assert s1 == []
        s2 = list(slices(101, 100))
        assert s2 == []


class TestWriteToFile:
    def test_filelike(self):
        dtype = np.int32
        shape = (100, 20)
        in_arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        with tempfile.TemporaryFile() as file:
            write_to_file(in_arr, file, batchsize=10)
            file.seek(0)
            out_arr = np.fromfile(file, dtype=dtype).reshape(shape)
        np.testing.assert_array_equal(in_arr, out_arr)

    def test_pathlike(self):
        dtype = np.int32
        shape = (100, 20)
        in_arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        with tempfile.TemporaryDirectory() as dir_:
            file = Path(dir_, "test.i4")
            write_to_file(in_arr, file, batchsize=10)
            out_arr = np.fromfile(file, dtype=dtype).reshape(shape)
        np.testing.assert_array_equal(in_arr, out_arr)

    @pytest.mark.parametrize("dtype", ["<i4", ">i4"])
    def test_endian(self, dtype: DTypeLike):
        in_arr = np.arange(1000, dtype=np.int32)
        with tempfile.TemporaryFile() as file:
            write_to_file(in_arr, file, dtype=dtype)
            file.seek(0)
            out_arr = np.fromfile(file, dtype=dtype)
        np.testing.assert_array_equal(in_arr, out_arr)

    def test_transform(self):
        phase = np.linspace(-np.pi, np.pi, num=1001)
        in_arr = np.exp(1j * phase)
        with tempfile.TemporaryFile() as file:
            write_to_file(in_arr, file, transform=np.angle)
            file.seek(0)
            out_arr = np.fromfile(file, dtype=np.float64)
        np.testing.assert_allclose(out_arr, phase, atol=1e-6)

    def test_0d(self):
        in_arr = np.int32(0)
        with tempfile.TemporaryFile() as file:
            errmsg = r"^dataset must be at least 1-D, instead got dataset\.ndim=0$"
            with pytest.raises(ValueError, match=errmsg):
                write_to_file(in_arr, file)


class TestReadFromFile:
    def test_filelike(self):
        dtype = np.int32
        shape = (101, 20)
        in_arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        out_arr = np.empty_like(in_arr)
        with tempfile.TemporaryFile() as file:
            in_arr.tofile(file)
            file.seek(0)
            read_from_file(out_arr, file, batchsize=10)
        np.testing.assert_array_equal(in_arr, out_arr)

    def test_pathlike(self):
        dtype = np.int32
        shape = (101, 20)
        in_arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        out_arr = np.empty_like(in_arr)
        with tempfile.TemporaryDirectory() as dir_:
            file = Path(dir_, "test.i4")
            in_arr.tofile(file)
            read_from_file(out_arr, file, batchsize=10)
        np.testing.assert_array_equal(in_arr, out_arr)

    def test_0d(self):
        in_arr = np.int32(123)
        out_arr = np.int32(0)
        with tempfile.TemporaryFile() as file:
            in_arr.tofile(file)
            file.seek(0)
            errmsg = r"^dataset must be at least 1-D, instead got dataset\.ndim=0$"
            with pytest.raises(ValueError, match=errmsg):
                read_from_file(out_arr, file)


class TestScratchDirectory:
    def test_temp_directory(self):
        with scratch_directory() as scratchdir:
            assert scratchdir.is_dir()
        assert not scratchdir.exists()

    def test_create_directory(self):
        # Create a temporary parent directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            # The file path to create the scratch directory at.
            path = Path(tmpdir) / "a/s/d/f"
            assert not path.is_dir()

            # Create a new scratch directory.
            with scratch_directory(path) as scratchdir:
                # Check that a directory was created in the right place.
                assert scratchdir == path
                assert scratchdir.is_dir()

    @pytest.mark.parametrize("delete", [True, False])
    def test_existing_directory(self, *, delete: bool):
        with tempfile.TemporaryDirectory() as tmpdir:
            with scratch_directory(tmpdir, delete=delete) as scratchdir:
                assert scratchdir == Path(tmpdir)

            # An existing directory should not be cleaned up by the context manager,
            # regardless of the `delete` parameter.
            assert Path(tmpdir).is_dir()

    @pytest.mark.parametrize("delete", [True, False])
    def test_delete(self, *, delete: bool):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a scratch directory at the specified path.
            path = Path(tmpdir) / "asdf"
            with scratch_directory(path, delete=delete) as scratchdir:
                assert scratchdir.is_dir()

            # Check that the scratch directory was removed if `delete` was True.
            if delete:
                assert not scratchdir.exists()
            else:
                assert scratchdir.is_dir()

    @pytest.mark.parametrize("delete", [True, False])
    def test_tempdir_delete(self, *, delete: bool):
        try:
            # Create a scratch directory using `mkdtemp()`.
            with scratch_directory(delete=delete) as scratchdir:
                assert scratchdir.is_dir()

            # Check that the scratch directory was removed if `delete` was True.
            if delete:
                assert not scratchdir.exists()
            else:
                assert scratchdir.exists()
        finally:
            # Clean up the directory if it wasn't automatically removed.
            if scratchdir.is_dir():
                shutil.rmtree(scratchdir)
