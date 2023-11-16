from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "InputDataset",
    "OutputDataset",
]


@runtime_checkable
class InputDataset(Protocol):
    """
    An array-like interface for reading input datasets.

    `InputDataset` defines the abstract interface that types must conform to in order
    to be valid inputs to the ``snaphu.unwrap()`` function. Such objects must export
    NumPy-like `dtype`, `shape`, and `ndim` attributes and must support NumPy-style
    slice-based indexing.

    See Also
    --------
    OutputDataset
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""
    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""
    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: slice | tuple[slice, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...  # pragma: no cover


@runtime_checkable
class OutputDataset(Protocol):
    """
    An array-like interface for writing output datasets.

    `OutputDataset` defines the abstract interface that types must conform to in order
    to be valid outputs of the ``snaphu.unwrap()`` function. Such objects must export
    NumPy-like `dtype`, `shape`, and `ndim` attributes and must support NumPy-style
    slice-based indexing.

    See Also
    --------
    InputDataset
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""
    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""
    ndim: int
    """int : Number of array dimensions."""

    def __setitem__(self, key: slice | tuple[slice, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""
        ...  # pragma: no cover
