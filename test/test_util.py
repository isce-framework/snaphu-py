import itertools
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from snaphu._util import BlockIterator, ceil_divide, scratch_directory


class TestCeilDivide:
    def test_positive(self):
        assert ceil_divide(3, 2) == 2
        assert ceil_divide(4, 2) == 2
        assert ceil_divide(1, 1_000_000) == 1

    def test_negative(self):
        assert ceil_divide(-3, 2) == -1
        assert ceil_divide(3, -2) == -1
        assert ceil_divide(-3, -2) == 2

        assert ceil_divide(-4, 2) == -2
        assert ceil_divide(4, -2) == -2
        assert ceil_divide(-4, -2) == 2

    def test_zero(self):
        assert ceil_divide(0, 1) == 0
        assert ceil_divide(-0, 1) == 0

    def test_divide_by_zero(self):
        with pytest.warns(RuntimeWarning) as record:
            ceil_divide(1, 0)

        assert len(record) == 1
        warning = record[0].message.args[0]
        assert "divide by zero encountered" in warning

    def test_arraylike(self):
        result = ceil_divide([1, 2, 3, 4, 5], 2)
        expected = [1, 1, 2, 2, 3]
        np.testing.assert_array_equal(result, expected)


class TestBlockIterator:
    @pytest.fixture
    def blocks2d(self) -> BlockIterator:
        return BlockIterator(shape=(100, 101), chunks=(25, 34))

    def test_is_iterable(self, blocks2d: BlockIterator):
        assert isinstance(blocks2d, Iterable)

    def test_attrs(self, blocks2d: BlockIterator):
        assert blocks2d.shape == (100, 101)
        assert blocks2d.chunks == (25, 34)

    def test_nblocks(self, blocks2d: BlockIterator):
        nblocks = len(list(blocks2d))
        assert nblocks == 12

    def test_iter(self, blocks2d: BlockIterator):
        arr = np.zeros(blocks2d.shape, dtype=np.int32)
        for block in blocks2d:
            arr[block] += 1
        np.testing.assert_array_equal(arr, 1)

    def test_blocks1d(self):
        blocks = BlockIterator(shape=99, chunks=25)
        assert blocks.shape == (99,)
        assert blocks.chunks == (25,)

        starts = [0, 25, 50, 75]
        stops = [25, 50, 75, 99]
        for block, start, stop in itertools.zip_longest(blocks, starts, stops):
            (slice_,) = block
            assert slice_.start == start
            assert slice_.stop == stop
            assert slice_.step is None

    def test_shape_chunks_mismatch(self):
        pattern = (
            "^size mismatch: shape and chunks must have the same number of elements,"
            r" instead got len\(shape\) != len\(chunks\) \(2 != 3\)$"
        )
        with pytest.raises(ValueError, match=pattern):
            BlockIterator(shape=(300, 400), chunks=(3, 4, 5))

    def test_bad_shape(self):
        pattern = r"^shape elements must all be > 0, instead got \(100, -1\)$"
        with pytest.raises(ValueError, match=pattern):
            BlockIterator(shape=(100, -1), chunks=(10, 10))

    def test_bad_chunks(self):
        pattern = r"^chunk elements must all be > 0, instead got \(10, -1\)$"
        with pytest.raises(ValueError, match=pattern):
            BlockIterator(shape=(100, 100), chunks=(10, -1))

    def test_repr(self, blocks2d: BlockIterator):
        assert repr(blocks2d) == "BlockIterator(shape=(100, 101), chunks=(25, 34))"


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

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: SIM117
            with scratch_directory(tmpdir, delete=False) as scratchdir:
                assert scratchdir == Path(tmpdir)

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
