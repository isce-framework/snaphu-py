from __future__ import annotations

import itertools
import os
import shutil
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "BlockIterator",
    "ceil_divide",
    "scratch_directory",
]


def ceil_divide(n: ArrayLike, d: ArrayLike) -> np.ndarray:
    """
    Return the smallest integer greater than or equal to the quotient of the inputs.

    Computes integer division of dividend `n` by divisor `d`, rounding up instead of
    truncating.

    Parameters
    ----------
    n : array_like
        The numerator.
    d : array_like
        The denominator.

    Returns
    -------
    q : numpy.ndarray
        The quotient, rounded up to the next integer.
    """
    n = np.asanyarray(n)
    d = np.asanyarray(d)
    return (n + d - np.sign(d)) // d


def as_tuple_of_int(ints: int | Iterable[int]) -> tuple[int, ...]:
    """
    Convert the input to a tuple of ints.

    Parameters
    ----------
    ints : int or iterable of int
        One or more integers.

    Returns
    -------
    out : tuple of int
        Tuple containing the input(s).
    """
    try:
        return (int(ints),)  # type: ignore[arg-type]
    except TypeError:
        return tuple([int(i) for i in ints])  # type: ignore[union-attr]


@dataclass(frozen=True)
class BlockIterator(Iterable[tuple[slice, ...]]):
    """
    An iterable over chunks of an N-dimensional array.

    `BlockIterator` represents a partitioning of a multidimensional array into
    regularly-sized non-overlapping blocks. Each block is represented by an index
    expression (i.e. a tuple of `slice` objects) that can be used to access the
    corresponding block of data from the partitioned array. The full set of blocks spans
    the entire array.

    Iterating over a `BlockIterator` object yields each block in unspecified order.
    """

    shape: tuple[int, ...]
    """tuple of int : The shape of the array to be partitioned into blocks."""
    chunks: tuple[int, ...]
    """
    tuple of int : The shape of a typical block. The last block along each axis may be
    smaller.
    """

    def __init__(self, shape: int | Iterable[int], chunks: int | Iterable[int]):
        """
        Construct a new `BlockIterator` object.

        Parameters
        ----------
        shape : int or iterable of int
            The shape of the array to be partitioned into blocks. Each dimension must be
            > 0.
        chunks : int or iterable of int
            The shape of a typical block. Must be the same length as `shape`. Each chunk
            dimension must be > 0.
        """
        # Normalize `shape` and `chunks` into tuples of ints.
        shape = as_tuple_of_int(shape)
        chunks = as_tuple_of_int(chunks)

        if len(chunks) != len(shape):
            errmsg = (
                "size mismatch: shape and chunks must have the same number of elements,"
                f" instead got len(shape) != len(chunks) ({len(shape)} !="
                f" {len(chunks)})"
            )
            raise ValueError(errmsg)

        if not all(n > 0 for n in shape):
            errmsg = f"shape elements must all be > 0, instead got {shape}"
            raise ValueError(errmsg)
        if any(n <= 0 for n in chunks):
            errmsg = f"chunk elements must all be > 0, instead got {chunks}"
            raise ValueError(errmsg)

        # XXX Workaround for `frozen=True`.
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "chunks", chunks)

    def __iter__(self) -> Iterator[tuple[slice, ...]]:
        """
        Iterate over blocks in unspecified order.

        Yields
        ------
        block : tuple of slice
            A tuple of slices that can be used to access the corresponding block of data
            from an array.
        """
        # Number of blocks along each array axis.
        nblocks = ceil_divide(self.shape, self.chunks)

        # Iterate over blocks.
        for block_ind in itertools.product(*[range(n) for n in nblocks]):
            # Get the lower & upper index bounds for the current block.
            start = np.multiply(block_ind, self.chunks)
            stop = np.minimum(start + self.chunks, self.shape)

            # Yield a tuple of slice objects.
            yield tuple(itertools.starmap(slice, zip(start, stop)))


@contextmanager
def scratch_directory(
    dir_: str | os.PathLike[str] | None = None, *, delete: bool = True
) -> Generator[Path, None, None]:
    """
    Context manager that creates a (possibly temporary) file system directory.

    If `dir_` is a path-like object, a directory will be created at the specified
    file system path if it did not already exist. Otherwise, if `dir_` is None, a
    temporary directory will instead be created as though by ``tempfile.mkdtemp()``.

    The directory may be automatically removed from the file system upon exiting the
    context manager.

    Parameters
    ----------
    dir_ : path-like or None, optional
        Scratch directory path. If None, a temporary directory will be created. Defaults
        to None.
    delete : bool, optional
        If True, the directory and its contents are recursively removed from the
        file system upon exiting the context manager. Defaults to True.

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
        scratchdir.mkdir(parents=True, exist_ok=True)

    yield scratchdir

    if delete:
        shutil.rmtree(scratchdir)
