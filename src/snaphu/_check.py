from __future__ import annotations

import numpy as np

from .io import InputDataset, OutputDataset

__all__ = [
    "check_2d_shapes",
    "check_bool_or_byte_dtype",
    "check_complex_dtype",
    "check_cost_mode",
    "check_dataset_shapes",
    "check_float_dtype",
    "check_integer_dtype",
]


def check_2d_shapes(**shapes: tuple[int, ...]) -> None:
    """
    Ensure that the input tuples are valid 2-D shapes.

    Parameters
    ----------
    **shapes : dict, optional
        Input tuples to check for validity. Inputs must be pairs of positive integers.
        The name of each keyword argument is used to format the error message in case of
        an invalid input.

    Raises
    ------
    ValueError
        If an input had invalid length or contained invalid (negative or zero) values.
    """
    for name, shape in shapes.items():
        if len(shape) != 2:
            errmsg = f"{name} must be a pair of ints, instead got {name}={shape}"
            raise ValueError(errmsg)
        if not all(n >= 1 for n in shape):
            errmsg = (
                f"{name} may not contain negative or zero values, got {name}={shape}"
            )
            raise ValueError(errmsg)


def check_dataset_shapes(
    shape: tuple[int, ...], **datasets: InputDataset | OutputDataset
) -> None:
    """
    Ensure that one or more datasets has a specified shape.

    Parameters
    ----------
    shape : tuple of int
        The expected shape of each dataset.
    **datasets : dict, optional
        Datasets whose shape must be equal to `shape`. The name of each keyword argument
        is used to format the error message in case of a shape mismatch.

    Raises
    ------
    ValueError
        If any dataset had a different shape.
    """
    for name, arr in datasets.items():
        if arr.shape != shape:
            errmsg = (
                f"shape mismatch: {name} dataset must have shape {shape}, instead got"
                f" {name}.shape={arr.shape}"
            )
            raise ValueError(errmsg)


def check_complex_dtype(**datasets: InputDataset | OutputDataset) -> None:
    """
    Ensure that one or more datasets is complex-valued.

    Parameters
    ----------
    **datasets : dict, optional
        Datasets whose datatype must be a complex floating-point type. The name of each
        keyword argument is used to format the error message in case of a different
        datatype.

    Raises
    ------
    TypeError
        If any dataset had a different datatype.
    """
    for name, arr in datasets.items():
        if not np.issubdtype(arr.dtype, np.complexfloating):
            errmsg = (
                f"{name} dataset must be complex-valued, instead got dtype={arr.dtype}"
            )
            raise TypeError(errmsg)


def check_float_dtype(**datasets: InputDataset | OutputDataset) -> None:
    """
    Ensure that one or more datasets is real-valued.

    Parameters
    ----------
    **datasets : dict, optional
        Datasets whose datatype must be a floating-point type. The name of each keyword
        argument is used to format the error message in case of a different datatype.

    Raises
    ------
    TypeError
        If any dataset had a different datatype.
    """
    for name, arr in datasets.items():
        if not np.issubdtype(arr.dtype, np.floating):
            errmsg = (
                f"{name} dataset must be real-valued, instead got dtype={arr.dtype}"
            )
            raise TypeError(errmsg)


def check_integer_dtype(**datasets: InputDataset | OutputDataset) -> None:
    """
    Ensure that one or more datasets is integer-valued.

    Parameters
    ----------
    **datasets : dict, optional
        Datasets whose datatype must be a (signed or unsigned) integer type. The name of
        each keyword argument is used to format the error message in case of a different
        datatype.

    Raises
    ------
    TypeError
        If any dataset had a different datatype.
    """
    for name, arr in datasets.items():
        if not np.issubdtype(arr.dtype, np.integer):
            errmsg = (
                f"{name} dataset must be integer-valued, instead got dtype={arr.dtype}"
            )
            raise TypeError(errmsg)


def check_bool_or_byte_dtype(**datasets: InputDataset | OutputDataset) -> None:
    """
    Ensure that one or more datasets is boolean or 8-bit integer-valued.

    Parameters
    ----------
    **datasets : dict, optional
        Datasets whose datatype must be a boolean or 8-bit (signed or unsigned) integer
        type. The name of each keyword argument is used to format the error message in
        case of a different datatype.

    Raises
    ------
    TypeError
        If any dataset had a different datatype.
    """
    for name, arr in datasets.items():
        dtype = arr.dtype
        if (dtype != np.bool_) and (dtype != np.uint8) and (dtype != np.int8):
            errmsg = (
                f"{name} dataset must be a boolean or 8-bit integer-valued, instead got"
                f" dtype={dtype}"
            )
            raise TypeError(errmsg)


def check_cost_mode(cost: str) -> None:
    """
    Ensure that the input SNAPHU cost mode is valid and supported.

    Parameters
    ----------
    cost : str
        The cost mode.

    Raises
    ------
    NotImplementedError
        If the specified cost mode is not supported.
    ValueError
        If the specified cost mode is not a valid SNAPHU cost option.
    """
    cost_modes = {"defo", "smooth"}

    if cost == "topo":
        errmsg = (
            "'topo' cost mode is not currently supported, cost must be one of"
            f" {cost_modes}"
        )
        raise NotImplementedError(errmsg)

    if cost not in cost_modes:
        errmsg = f"cost mode must be in {cost_modes}, instead got {cost!r}"
        raise ValueError(errmsg)
