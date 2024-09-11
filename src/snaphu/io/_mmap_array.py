from __future__ import annotations

import mmap
import os
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from ._dataset import InputDataset, OutputDataset

__all__ = [
    "MMapArray",
]


# TODO(Python 3.10): change return type to `TypeGuard[Iterable[Any]]`
# (see https://docs.python.org/3/library/typing.html#typing.TypeGuard)
def is_iterable(obj: Any) -> bool:
    """
    Check if the input object is iterable.

    Parameters
    ----------
    obj : object
        The input object.

    Returns
    -------
    bool
        True if the argument is iterable; otherwise False.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def tuple_of_ints(ints: int | Iterable[int]) -> tuple[int, ...]:
    """
    Convert the input to a tuple of ints.

    Parameters
    ----------
    ints : int or iterable of int
        One or more integers.

    Returns
    -------
    tuple of int
        Tuple containing the inputs.
    """
    if is_iterable(ints):
        return tuple(int(i) for i in ints)
    else:
        return int(ints),


def create_or_extend_file(filepath: str | os.PathLike[str], size: int) -> None:
    """
    Create a file with the specified size or extend an existing file to the same size.

    Parameters
    ----------
    filepath : str or path-like
        File path.
    size : int
        The size, in bytes, of the file.
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        # If the file does not exist, then create it with the specified size.
        with filepath.open("wb") as fd:
            fd.truncate(size)
    else:
        # If the file exists but is smaller than the requested size, extend the file
        # length.
        filesize = filepath.stat().st_size
        if filesize < size:
            with filepath.open("r+b") as fd:
                fd.truncate(size)


class MMapArray(InputDataset, OutputDataset, AbstractContextManager["MMapArray"]):
    """
    """

    def __init__(
        self,
        filepath: str | os.PathLike[str],
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ):
        """
        Create a new `MMapArray` object.

        If the file does not exist, it will be created. If the file does exist but is
        smaller than the array, it will be extended to the size (in bytes) of the array.

        Parameters
        ----------
        filepath : str or path-like
            The file path.
        shape : tuple of int
            Tuple of array dimensions.
        dtype : data-type
            Data-type of the array's elements. Must be convertible to a `numpy.dtype`
            object.
        """

        self._filepath = Path(filepath)
        self._shape = tuple_of_ints(shape)
        self._dtype = np.dtype(dtype)

        # Get array size in bytes.
        size = np.prod(self.shape) * self.dtype.itemsize

        # If the file doesn't exist, create it with the required size. Else, ensure that
        # the file size is at least `size` bytes.
        create_or_extend_file(self.filepath, size)

        # Open & memory-map the file.
        self._fd = self.filepath.open("r+b")
        try:
            self._mmap = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_WRITE)
        except:
            self._fd.close()
            raise

    @property
    def filepath(self) -> Path:
        """pathlib.Path : The file path."""
        return self._filepath

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int : Tuple of array dimensions."""
        return self._shape

    @property
    def ndim(self) -> int:
        """int : Number of array dimensions."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype : Data-type of the array's elements."""
        return self._dtype

    @property
    def closed(self) -> bool:
        """bool : True if the memory-mapped file is closed."""
        return self._fd.closed

    def flush(self) -> None:
        """Flushes changes made to the in-memory copy of a file back to disk."""
        self._mmap.flush()

    def close(self) -> None:
        """
        Close the underlying dataset.

        Has no effect if the dataset is already closed.
        """
        if self.closed:
            return

        self.flush()

        try:
            self._mmap.close()
        finally:
            self._fd.close()

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()

    def __array__(self) -> np.ndarray:
        return np.frombuffer(self._mmap, dtype=self.dtype).reshape(self.shape)

    def __getitem__(self, key: slice | tuple[slice, ...], /) -> np.ndarray:
        arr = np.asanyarray(self)
        return arr[key]

    def __setitem__(self, key: slice | tuple[slice, ...], value: np.ndarray, /) -> None:
        arr = np.asanyarray(self)
        arr[key] = value

    def __repr__(self) -> str:
        filepath = self.filepath
        shape = self.shape
        dtype = self.dtype
        return f"{type(self).__name__}({filepath=}, {shape=}, {dtype=})"
