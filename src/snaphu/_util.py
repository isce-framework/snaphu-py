from __future__ import annotations

import io
import os
import shutil
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .io import InputDataset, OutputDataset

__all__ = [
    "nan_to_zero",
    "read_from_file",
    "scratch_directory",
    "write_to_file",
]


def slices(start: int, stop: int, step: int = 1) -> Iterator[slice]:
    """
    Iterate over slices spanning a range.

    Yield successive non-overlapping slices of length `step` that span the half-open
    interval from `start` to `stop` (excluding `stop` itself).

    Parameters
    ----------
    start : int
        The start of the range.
    stop : int
        The end of the range.
    step : int, optional
        The maximum length of each slice. The final slice may be smaller. Must be >= 1.
        Defaults to 1.

    Yields
    ------
    slice
        A slice representing a sub-range of the specified range.
    """
    if step < 1:
        errmsg = f"step must be >= 1, instead got {step}"
        raise ValueError(errmsg)

    while start < stop:
        end = min(start + step, stop)
        yield slice(start, end)
        start = end


def write_to_file(
    dataset: InputDataset,
    file: str | os.PathLike[str] | io.IOBase,
    *,
    batchsize: int = 512,
    transform: Callable[[ArrayLike], np.ndarray] = np.asanyarray,
    dtype: DTypeLike | None = None,
) -> None:
    """
    Write the dataset contents to a file.

    The data is written in batches by slicing along the leading axis of the dataset in
    order to avoid holding the entire dataset in memory at once.

    Parameters
    ----------
    dataset : snaphu.io.InputDataset
        The input dataset.
    file : path-like or file-like
        An open file object or valid file path. If the path to an existing file is
        provided, the file will be overwritten.
    batchsize : int, optional
        The maximum length of each batch of data along the leading axis of `dataset`.
        Defaults to 512.
    transform : callable, optional
        An function that is applied to each batch of data from `dataset` before writing
        it to the file. The function should take a single array_like parameter and
        return a NumPy array. Defaults to `numpy.asanyarray`.
    dtype : data-type or None, optional
        The datatype used to store the dataset contents in the file. Each batch of data
        will be cast to this datatype before writing it to the file. If None, uses the
        datatype of the input dataset. Defaults to None.
    """
    # If the `file` argument was a path, open the file for writing in binary mode and
    # truncate the file if it exists.
    if isinstance(file, (str, os.PathLike)):
        with Path(file).open("w+b") as file:
            write_to_file(
                dataset,
                file,
                batchsize=batchsize,
                dtype=dtype,
                transform=transform,
            )
        return

    # The input dataset must be at least 1-D.
    if dataset.ndim < 1:
        errmsg = f"dataset must be at least 1-D, instead got {dataset.ndim=}"
        raise ValueError(errmsg)

    # If `dtype` was not specified, default to the dataset's dtype.
    if dtype is None:
        dtype = dataset.dtype

    # Iterate over batches of data by slicing the dataset along its leading axis.
    for slice_ in slices(0, dataset.shape[0], batchsize):
        # Transform the batch of data as necessary and write it to the file.
        arr = transform(dataset[slice_]).astype(dtype, copy=False)
        arr.tofile(file)


def read_from_file(
    dataset: OutputDataset,
    file: str | os.PathLike[str] | io.IOBase,
    *,
    batchsize: int = 512,
    dtype: DTypeLike | None = None,
) -> None:
    """
    Populate the dataset contents by reading from a file.

    The data is read in batches by slicing along the leading axis of the dataset in
    order to avoid holding the entire dataset in memory at once.

    The file size must not be less than the size of the dataset contents.

    Parameters
    ----------
    dataset : snaphu.io.OutputDataset
        The output dataset.
    file : path-like or file-like
        An open file object or valid path to an existing file.
    batchsize : int, optional
        The maximum length of each batch of data along the leading axis of `dataset`.
        Defaults to 512.
    dtype : data-type or None, optional
        The datatype of the contents of the file. If None, the file contents will be
        assumed to have the same datatype as the output dataset (including byte order).
        Defaults to None.
    """
    # If the `file` argument was a path, open the file for reading in binary mode.
    if isinstance(file, (str, os.PathLike)):
        with Path(file).open("rb") as f:
            read_from_file(dataset, f, batchsize=batchsize, dtype=dtype)
        return

    # The input dataset must be at least 1-D.
    if dataset.ndim < 1:
        errmsg = f"dataset must be at least 1-D, instead got {dataset.ndim=}"
        raise ValueError(errmsg)

    # If `dtype` was not specified, default to the dataset's dtype.
    if dtype is None:
        dtype = dataset.dtype

    # Iterate over batches of data by slicing the dataset along its leading axis.
    n = dataset.shape[0]
    for slice_ in slices(0, n, batchsize):
        # Infer the shape and size of the corresponding slice of the dataset.
        shape = (slice_.stop - slice_.start,) + dataset.shape[1:]
        size = np.prod(shape)

        # Read a batch of data from the file.
        dataset[slice_] = np.fromfile(file, dtype=dtype, count=size).reshape(shape)


def nan_to_zero(arr: ArrayLike) -> np.ndarray:
    """
    Replace Not a Number (NaN) values with zeros.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    np.ndarray
        A copy of the input array with NaN values replaced with zeros.
    """
    return np.where(np.isnan(arr), 0, arr)


@contextmanager
def scratch_directory(
    dir_: str | os.PathLike[str] | None = None, *, delete: bool = True
) -> Generator[Path, None, None]:
    """
    Context manager that creates a (possibly temporary) file system directory.

    If `dir_` is a path-like object, a directory will be created at the specified
    file system path if it did not already exist. Otherwise, if `dir_` is None, a
    temporary directory will instead be created as though by ``tempfile.mkdtemp()``.

    If a directory was created this way, it may be automatically removed from the file
    system upon exiting the context manager, depending on the `delete` argument. If the
    directory already existed, it will not be removed.

    Parameters
    ----------
    dir_ : path-like or None, optional
        Scratch directory path. If None, a temporary directory will be created. Defaults
        to None.
    delete : bool, optional
        If True, the directory and its contents are recursively removed from the
        file system upon exiting the context manager. This parameter is ignored if the
        specified path was an existing directory. Defaults to True.

    Yields
    ------
    pathlib.Path
        Scratch directory path. If `delete` was True, the directory will be removed from
        the file system upon exiting the context manager scope.
    """
    if dir_ is None:
        scratchdir = Path(mkdtemp())
    else:
        scratchdir = Path(dir_)

        # If the directory already existed, don't delete it upon exiting the context
        # manager. Otherwise, create the directory.
        if scratchdir.exists():
            delete = False
        else:
            scratchdir.mkdir(parents=True)

    yield scratchdir

    if delete:
        shutil.rmtree(scratchdir)
