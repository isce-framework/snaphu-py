from __future__ import annotations

import os
import textwrap
from pathlib import Path
from tempfile import mkstemp
from typing import overload

import numpy as np

from ._check import (
    check_bool_or_byte_dtype,
    check_cost_mode,
    check_dataset_shapes,
    check_float_dtype,
    check_integer_dtype,
)
from ._snaphu import run_snaphu
from ._util import copy_blockwise, nan_to_zero, scratch_directory
from .io import InputDataset, OutputDataset

__all__ = [
    "grow_conncomps",
    "regrow_conncomp_from_unw",
]


def regrow_conncomp_from_unw(
    unw_file: str | os.PathLike[str],
    corr_file: str | os.PathLike[str],
    conncomp_file: str | os.PathLike[str],
    line_length: int,
    nlooks: float,
    cost: str,
    mag_file: str | os.PathLike[str] | None = None,
    mask_file: str | os.PathLike[str] | None = None,
    min_conncomp_frac: float = 0.01,
    scratchdir: str | os.PathLike[str] | None = None,
) -> None:
    """
    Run SNAPHU to regrow connected components from an unwrapped input.

    This is particularly useful if SNAPHU was initially run in tiled mode, resulting in
    a set of connected components that are disjoint across tile boundaries. The
    connected components will be recomputed as though by a single tile, so that
    components are no longer delimited by the extents of each tile.

    This function does not compute a new unwrapped solution.

    Parameters
    ----------
    unw_file : path-like
        The input unwrapped phase file path.
    corr_file : path-like
        The input coherence file path.
    conncomp_file : path-like
        The output connected component labels file path.
    line_length : int
        The line length, in samples, of the input & output arrays.
    nlooks : float
        The equivalent number of independent looks used to form the sample coherence.
    cost : str
        Statistical cost mode.
    mag_file : path-like or None, optional
        The input interferogram magnitude file path. If None, magnitude data is not
        used. Defaults to None.
    mask_file : path-like or None, optional
        An optional file path of a byte mask file. If None, no mask is applied. Defaults
        to None.
    min_conncomp_frac : float, optional
        Minimum size of a single connected component, as a fraction of the total number
        of pixels in the array. Defaults to 0.01.
    scratchdir : path-like or None, optional
        The scratch directory where the config file will be written. If None, a
        default temporary directory is chosen as though by ``tempfile.gettempdir()``.
        Defaults to None.
    """
    # In REGROWCONNCOMPS mode, SNAPHU recomputes the cost arrays (but does not
    # re-unwrap), so we should pass in all parameters necessary to compute costs, but
    # don't need to pass an INITMETHOD, for example.
    config = textwrap.dedent(
        f"""\
        REGROWCONNCOMPS TRUE
        INFILE {os.fspath(unw_file)}
        INFILEFORMAT FLOAT_DATA
        UNWRAPPEDINFILEFORMAT FLOAT_DATA
        CORRFILE {os.fspath(corr_file)}
        CORRFILEFORMAT FLOAT_DATA
        CONNCOMPFILE {os.fspath(conncomp_file)}
        CONNCOMPOUTTYPE UINT
        LINELENGTH {line_length}
        NCORRLOOKS {nlooks}
        STATCOSTMODE {cost.upper()}
        MINCONNCOMPFRAC {min_conncomp_frac}
        """
    )
    if mag_file is not None:
        config += f"MAGFILE {os.fspath(mag_file)}\nMAGFILEFORMAT FLOAT_DATA\n"
    if mask_file is not None:
        config += f"BYTEMASKFILE {os.fspath(mask_file)}\n"

    # Write config parameters to file. The config file should have a descriptive name to
    # disambiguate it from the config file used for unwrapping.
    _, config_file = mkstemp(
        dir=scratchdir, prefix="snaphu-regrow-conncomps.config.", suffix=".txt"
    )
    Path(config_file).write_text(config)

    # Run SNAPHU in REGROWCONNCOMPS mode to generate new connected component labels.
    run_snaphu(config_file)


@overload
def grow_conncomps(
    unw: InputDataset,
    corr: InputDataset,
    nlooks: float,
    cost: str = "smooth",
    *,
    mag: InputDataset | None = None,
    mask: InputDataset | None = None,
    min_conncomp_frac: float = 0.01,
    scratchdir: str | os.PathLike[str] | None = None,
    delete_scratch: bool = True,
    conncomp: OutputDataset,
) -> OutputDataset: ...  # pragma: no cover


# If `conncomp` is not specified, return the output as a NumPy array.
@overload
def grow_conncomps(
    unw: InputDataset,
    corr: InputDataset,
    nlooks: float,
    cost: str = "smooth",
    *,
    mag: InputDataset | None = None,
    mask: InputDataset | None = None,
    min_conncomp_frac: float = 0.01,
    scratchdir: str | os.PathLike[str] | None = None,
    delete_scratch: bool = True,
) -> np.ndarray: ...  # pragma: no cover


def grow_conncomps(  # type: ignore[no-untyped-def]
    unw,
    corr,
    nlooks,
    cost="smooth",
    *,
    mag=None,
    mask=None,
    min_conncomp_frac=0.01,
    scratchdir=None,
    delete_scratch=True,
    conncomp=None,
):
    r"""
    Compute connected component labels using SNAPHU.

    Grow connected component labels for the input unwrapped phase field using a
    statistical segmentation framework\ [1]_. Connected components are generated as
    though the phase was unwrapped by the Statistical-Cost, Network-Flow Algorithm for
    Phase Unwrapping (SNAPHU) as a single tile with the specified inputs. The unwrapped
    phase itself is not modified or re-computed.

    This may be useful if the unwrapped phase was formed using a different unwrapping
    algorithm or if the phase was modified after unwrapping by some post-processing
    step(s).

    Each connected component is a region of phase samples that is believed to have been
    unwrapped in an internally self-consistent manner. Each distinct region is assigned
    a unique positive integer label. Pixels not belonging to any component are assigned
    a label of zero.

    Parameters
    ----------
    unw : snaphu.io.InputDataset
        The input unwrapped phase, in radians. A 2-D real-valued array. If the dataset
        contains Not a Number (NaN) values, they will be replaced with zeros.
    corr : snaphu.io.InputDataset
        The sample coherence magnitude. Must be a floating-point array with the same
        dimensions as the input unwrapped phase. Valid coherence values are in the range
        [0, 1]. NaN values in the array will be replaced with zeros.
    nlooks : float
        The equivalent number of independent looks used to form the sample coherence. An
        estimate of the number of statistically independent samples averaged in the
        multilooked data, taking into account spatial correlation due to
        oversampling/filtering (see `Notes`_).
    cost : {'defo', 'smooth'}, optional
        Statistical cost mode. Defaults to 'smooth'.
    mag : snaphu.io.InputDataset or None, optional
        An optional array of interferogram magnitude data. If provided, it must be a
        real-valued array with the same dimensions as the input unwrapped phase.
        Zero-magnitude pixels are treated as invalid (i.e. masked-out). NaN values in
        the array will be replaced with zeros. Defaults to None.
    mask : snaphu.io.InputDataset or None, optional
        Binary mask of valid pixels. Zeros in this raster indicate interferogram
        pixels that should be masked out. If provided, it must have the same dimensions
        as the input unwrapped phase and boolean or 8-bit integer datatype. Defaults to
        None.
    min_conncomp_frac : float, optional
        Minimum size of a single connected component, as a fraction of the total number
        of pixels in the array. Defaults to 0.01.
    scratchdir : path-like or None, optional
        Scratch directory where intermediate processing artifacts are written.
        If the specified directory does not exist, it will be created. If None,
        a temporary directory will be created as though by
        ``tempfile.TemporaryDirectory()``. Defaults to None.
    delete_scratch : bool, optional
        If True, if a scratch directory was created by this function, it will be
        automatically removed from the file system when the function exits. Otherwise,
        the scratch directory will be preserved. Existing directories are not deleted.
        Defaults to True.
    conncomp : snaphu.io.OutputDataset or None, optional
        An optional output dataset to store the connected component labels. If provided,
        it must have the same dimensions as the input unwrapped phase and any integer
        datatype. If None, an output array will be allocated internally. Defaults to
        None.

    Returns
    -------
    conncomp : snaphu.io.OutputDataset
        The output connected component labels.

    Notes
    -----
    An estimate of the equivalent number of independent looks may be obtained by

    .. math:: n_e = k_r k_a \frac{d_r d_a}{\rho_r \rho_a}

    where :math:`k_r` and :math:`k_a` are the number of looks in range and azimuth,
    :math:`d_r` and :math:`d_a` are the (single-look) sample spacing in range and
    azimuth, and :math:`\rho_r` and :math:`\rho_a are the range and azimuth resolution.

    References
    ----------
    .. [1] C. W. Chen and H. A. Zebker, "Phase unwrapping for large SAR interferograms:
       Statistical segmentation and generalized network models," IEEE Transactions on
       Geoscience and Remote Sensing, vol. 40, pp. 1709-1719 (2002).
    """
    # If a `conncomp` output dataset was not provided, create an array to store the
    # results.
    if conncomp is None:
        conncomp = np.zeros(shape=unw.shape, dtype=np.uint32)

    if unw.ndim != 2:
        errmsg = f"unw must be 2-D, instead got ndim={unw.ndim}"
        raise ValueError(errmsg)

    # Ensure that input & output datasets have matching shapes.
    check_dataset_shapes(unw.shape, corr=corr, conncomp=conncomp)
    if mag is not None:
        check_dataset_shapes(unw.shape, mag=mag)
    if mask is not None:
        check_dataset_shapes(unw.shape, mask=mask)

    # Ensure that input & output datasets have valid datatypes.
    check_float_dtype(unw=unw, corr=corr)
    check_integer_dtype(conncomp=conncomp)
    if mag is not None:
        check_float_dtype(mag=mag)
    if mask is not None:
        check_bool_or_byte_dtype(mask=mask)

    if nlooks < 1.0:
        errmsg = f"nlooks must be >= 1, instead got {nlooks}"
        raise ValueError(errmsg)

    check_cost_mode(cost)

    with scratch_directory(scratchdir, delete=delete_scratch) as dir_:
        # Create a raw binary file in the scratch directory for the unwrapped phase and
        # copy the input data to it. (`mkstemp` is used to avoid data races in case the
        # same scratch directory was used for multiple SNAPHU processes.)
        _, tmp_unw = mkstemp(dir=dir_, prefix="snaphu.unw.", suffix=".f4")
        tmp_unw_mmap = np.memmap(tmp_unw, dtype=np.float32, shape=unw.shape)
        copy_blockwise(unw, tmp_unw_mmap, transform=nan_to_zero)
        tmp_unw_mmap.flush()

        # Copy the input coherence data to a raw binary file in the scratch directory.
        _, tmp_corr = mkstemp(dir=dir_, prefix="snaphu.corr.", suffix=".f4")
        tmp_corr_mmap = np.memmap(tmp_corr, dtype=np.float32, shape=corr.shape)
        copy_blockwise(corr, tmp_corr_mmap, transform=nan_to_zero)
        tmp_corr_mmap.flush()

        # If magnitude data was provided, copy it to a raw binary file in the scratch
        # directory.
        if mag is None:
            tmp_mag = None
        else:
            _, tmp_mag = mkstemp(dir=dir_, prefix="snaphu.mag.", suffix=".f4")
            tmp_mag_mmap = np.memmap(tmp_mag, dtype=np.float32, shape=mag.shape)
            copy_blockwise(mag, tmp_mag_mmap, transform=nan_to_zero)
            tmp_mag_mmap.flush()

        # If a mask was provided, copy the mask data to a raw binary file in the scratch
        # directory.
        if mask is None:
            tmp_mask = None
        else:
            _, tmp_mask = mkstemp(dir=dir_, prefix="snaphu.mask.", suffix=".u1")
            tmp_mask_mmap = np.memmap(tmp_mask, dtype=np.bool_, shape=mask.shape)
            copy_blockwise(mask, tmp_mask_mmap)
            tmp_mask_mmap.flush()

        # Create a raw file in the scratch directory for the output connected
        # components.
        _, tmp_conncomp = mkstemp(dir=dir_, prefix="snaphu.conncomp.", suffix=".u4")

        regrow_conncomp_from_unw(
            unw_file=tmp_unw,
            corr_file=tmp_corr,
            conncomp_file=tmp_conncomp,
            line_length=unw.shape[1],
            nlooks=nlooks,
            cost=cost,
            mag_file=tmp_mag,
            mask_file=tmp_mask,
            min_conncomp_frac=min_conncomp_frac,
            scratchdir=dir_,
        )

        # Get the output connected component labels.
        tmp_cc_mmap = np.memmap(tmp_conncomp, dtype=np.uint32, shape=conncomp.shape)
        copy_blockwise(tmp_cc_mmap, conncomp)

    return conncomp
