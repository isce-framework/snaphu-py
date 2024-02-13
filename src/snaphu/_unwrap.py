from __future__ import annotations

import os
import textwrap
from collections.abc import Callable
from pathlib import Path
from tempfile import mkstemp
from typing import cast, overload

import numpy as np
from numpy.typing import ArrayLike

from ._snaphu import run_snaphu
from ._util import BlockIterator, scratch_directory
from .io import InputDataset, OutputDataset

__all__ = [
    "unwrap",
]


def check_shapes(
    igram: InputDataset,
    corr: InputDataset,
    unw: OutputDataset,
    conncomp: OutputDataset,
    mask: InputDataset | None = None,
) -> None:
    """
    Check that the arguments are 2-D arrays with matching shapes.

    Parameters
    ----------
    igram : snaphu.io.InputDataset
        The input interferogram. Must be a 2-D array.
    corr : snaphu.io.InputDataset
        The input coherence. Must be a 2-D array with the same shape as `igram`.
    unw : snaphu.io.OutputDataset
        The output unwrapped phase. Must be a 2-D array with the same shape as `igram`.
    conncomp : snaphu.io.OutputDataset
        The output connected component labels. Must be a 2-D array with the same shape
        as `igram`.
    mask : snaphu.io.InputDataset or None, optional
        An optional binary mask of valid samples. If not None, it must be a 2-D array
        with the same shape as `igram`. Defaults to None.

    Raises
    ------
    ValueError
        If any array is not 2-D or has a different shape than the others.
    """
    if igram.ndim != 2:
        errmsg = f"igram must be 2-D, instead got ndim={igram.ndim}"
        raise ValueError(errmsg)

    def raise_shape_mismatch_error(
        name: str,
        arr: InputDataset | OutputDataset,
    ) -> None:
        errmsg = (
            f"shape mismatch: {name} and igram must have the same shape, instead got"
            f" {name}.shape={arr.shape} and {igram.shape=}"
        )
        raise ValueError(errmsg)

    if corr.shape != igram.shape:
        raise_shape_mismatch_error("corr", corr)
    if unw.shape != igram.shape:
        raise_shape_mismatch_error("unw", unw)
    if conncomp.shape != igram.shape:
        raise_shape_mismatch_error("conncomp", conncomp)
    if (mask is not None) and (mask.shape != igram.shape):
        raise_shape_mismatch_error("mask", mask)


def check_dtypes(
    igram: InputDataset,
    corr: InputDataset,
    unw: OutputDataset,
    conncomp: OutputDataset,
    mask: InputDataset | None = None,
) -> None:
    """
    Check that the arguments have valid datatypes.

    Parameters
    ----------
    igram : snaphu.io.InputDataset
        The input interferogram. Must be a complex-valued array.
    corr : snaphu.io.InputDataset
        The input coherence. Must be a real-valued array.
    unw : snaphu.io.OutputDataset
        The output unwrapped phase. Must be a real-valued array.
    conncomp : snaphu.io.OutputDataset
        The output connected component labels. Must be an integer-valued array.
    mask : snaphu.io.InputDataset or None, optional
        An optional binary mask of valid samples. If not None, it must be a boolean or
        8-bit integer array.

    Raises
    ------
    TypeError
        If any array has an unexpected datatype.
    """
    if not np.issubdtype(igram.dtype, np.complexfloating):
        errmsg = (
            f"igram must be a complex-valued array, instead got dtype={igram.dtype}"
        )
        raise TypeError(errmsg)

    if not np.issubdtype(corr.dtype, np.floating):
        errmsg = f"corr must be a real-valued array, instead got dtype={corr.dtype}"
        raise TypeError(errmsg)

    if not np.issubdtype(unw.dtype, np.floating):
        errmsg = f"unw must be a real-valued array, instead got dtype={unw.dtype}"
        raise TypeError(errmsg)

    if not np.issubdtype(conncomp.dtype, np.integer):
        errmsg = (
            "conncomp must be an integer-valued array, instead got"
            f" dtype={conncomp.dtype}"
        )
        raise TypeError(errmsg)

    if (
        (mask is not None)
        and (mask.dtype != np.bool_)
        and (mask.dtype != np.uint8)
        and (mask.dtype != np.int8)
    ):
        errmsg = (
            "mask must be a boolean or 8-bit integer array (or None), instead got"
            f" dtype={mask.dtype}"
        )
        raise TypeError(errmsg)


def normalize_and_validate_tiling_params(
    ntiles: tuple[int, int],
    tile_overlap: int | tuple[int, int],
    nproc: int,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """
    Normalize and validate inputs related to tiling and multiprocessing.

    Parameters
    ----------
    ntiles : (int, int)
        Number of tiles along the row/column directions.
    tile_overlap : int or (int, int)
        Overlap, in pixels, between neighboring tiles.
    nproc : int
        Maximum number of child processes to spawn for parallel tile unwrapping.

    Returns
    -------
    ntiles : (int, int)
        `ntiles` normalized to a pair of positive integers.
    tile_overlap : (int, int)
        `tile_overlap` normalized to a pair of nonnegative integers.
    nproc : int
        `nproc` as a positive integer.
    """
    # Normalize `ntiles` to a tuple and ensure its contents are two positive-valued
    # integers.
    ntiles = tuple(ntiles)  # type: ignore[assignment]
    if len(ntiles) != 2:
        errmsg = f"ntiles must be a pair of ints, instead got {ntiles=}"
        raise ValueError(errmsg)
    if not all(n >= 1 for n in ntiles):
        errmsg = f"ntiles may not contain negative or zero values, got {ntiles=}"
        raise ValueError(errmsg)

    # If `tile_overlap` is iterable, ensure it's a tuple. Otherwise, assume it's a
    # single integer.
    try:
        tile_overlap = tuple(tile_overlap)  # type: ignore[arg-type,assignment]
    except TypeError:
        tile_overlap = (tile_overlap, tile_overlap)  # type: ignore[assignment]

    # Convince static type checkers that `tile_overlap` is now a pair of ints.
    tile_overlap = cast(tuple[int, int], tile_overlap)

    # Ensure the contents of `tile_overlap` are two nonnegative integers.
    if len(tile_overlap) != 2:
        errmsg = (
            f"tile_overlap must be an int or pair of ints, instead got {tile_overlap=}"
        )
        raise ValueError(errmsg)
    if not all(n >= 0 for n in tile_overlap):
        errmsg = f"tile_overlap may not contain negative values, got {tile_overlap=}"
        raise ValueError(errmsg)

    # If `nproc` is less than 1, use all available processors. Fall back to serial
    # execution if the number of available processors cannot be determined.
    if nproc < 1:
        nproc = os.cpu_count() or 1

    return ntiles, tile_overlap, nproc


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


def copy_blockwise(
    src: InputDataset,
    dst: OutputDataset,
    chunks: tuple[int, int] = (1024, 1024),
    *,
    transform: Callable[[ArrayLike], np.ndarray] | None = None,
) -> None:
    """
    Copy the contents of `src` to `dst` block-by-block.

    Parameters
    ----------
    src : snaphu.io.InputDataset
        Source dataset.
    dst : snaphu.io.OutputDataset
        Destination dataset.
    chunks : (int, int), optional
        Block dimensions. Defaults to (1024, 1024).
    transform : callable or None, optional
        An optional function object that is applied to each input block of data from
        `src` to produce the corresponding output block in `dst`. The function should
        take a single array_like parameter and return a NumPy array. If None, no
        transform is applied. Defaults to None.
    """
    shape = src.shape
    if dst.shape != shape:
        errmsg = (
            "shape mismatch: src and dst must have the same shape, instead got"
            f" {src.shape=} and {dst.shape=}"
        )
        raise ValueError(errmsg)

    for block in BlockIterator(shape, chunks):
        if transform is None:
            dst[block] = src[block]
        else:
            dst[block] = transform(src[block])


def regrow_conncomp_from_unw(
    unw_file: str | os.PathLike[str],
    corr_file: str | os.PathLike[str],
    mag_file: str | os.PathLike[str],
    conncomp_file: str | os.PathLike[str],
    line_length: int,
    nlooks: float,
    cost: str,
    mask_file: str | os.PathLike[str] | None = None,
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
    mag_file : path-like
        The input interferogram magnitude file path.
    conncomp_file : path-like
        The output connected component labels file path.
    line_length : int
        The line length, in samples, of the input & output arrays.
    nlooks : float
        The equivalent number of independent looks used to form the sample coherence.
    cost : str
        Statistical cost mode.
    mask_file : path-like or None, optional
        An optional file path of a byte mask file. If None, no mask is applied. Defaults
        to None.
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
        MAGFILE {os.fspath(mag_file)}
        MAGFILEFORMAT FLOAT_DATA
        CONNCOMPFILE {os.fspath(conncomp_file)}
        CONNCOMPOUTTYPE UINT
        LINELENGTH {line_length}
        NCORRLOOKS {nlooks}
        STATCOSTMODE {cost.upper()}
        """
    )
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
def unwrap(
    igram: InputDataset,
    corr: InputDataset,
    nlooks: float,
    cost: str = "smooth",
    init: str = "mcf",
    *,
    mask: InputDataset | None = None,
    ntiles: tuple[int, int] = (1, 1),
    tile_overlap: int | tuple[int, int] = 0,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
    regrow_conncomps: bool = True,
    scratchdir: str | os.PathLike[str] | None = None,
    delete_scratch: bool = True,
    unw: OutputDataset,
    conncomp: OutputDataset,
) -> tuple[OutputDataset, OutputDataset]: ...  # pragma: no cover


# If `unw` and `conncomp` aren't specified, return the outputs as two NumPy arrays.
@overload
def unwrap(
    igram: InputDataset,
    corr: InputDataset,
    nlooks: float,
    cost: str = "smooth",
    init: str = "mcf",
    *,
    mask: InputDataset | None = None,
    ntiles: tuple[int, int] = (1, 1),
    tile_overlap: int | tuple[int, int] = 0,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
    regrow_conncomps: bool = True,
    scratchdir: str | os.PathLike[str] | None = None,
    delete_scratch: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...  # pragma: no cover


def unwrap(  # type: ignore[no-untyped-def]
    igram,
    corr,
    nlooks,
    cost="smooth",
    init="mcf",
    *,
    mask=None,
    ntiles=(1, 1),
    tile_overlap=0,
    nproc=1,
    tile_cost_thresh=500,
    min_region_size=100,
    regrow_conncomps=True,
    scratchdir=None,
    delete_scratch=True,
    unw=None,
    conncomp=None,
):
    r"""
    Unwrap an interferogram using SNAPHU.

    Performs 2-D phase unwrapping using the Statistical-Cost, Network-Flow Algorithm for
    Phase Unwrapping (SNAPHU)\ [1]_. The algorithm produces a Maximum a Posteriori (MAP)
    estimate of the unwrapped phase field by (approximately) solving a nonlinear network
    flow optimization problem.

    The outputs include the unwrapped phase and a corresponding array of connected
    component labels. Each connected component is a region of pixels in the solution
    that is believed to have been unwrapped in an internally self-consistent manner\
    [2]_. Each distinct region is assigned a unique positive integer label. Pixels not
    belonging to any component are assigned a label of zero.

    Parameters
    ----------
    igram : snaphu.io.InputDataset
        The input interferogram. A 2-D complex-valued array. Not a Number (NaN) values
        in the array will be replaced with zeros.
    corr : snaphu.io.InputDataset
        The sample coherence magnitude. Must be a floating-point array with the same
        dimensions as the input interferogram. Valid coherence values are in the range
        [0, 1]. NaN values in the array will be replaced with zeros.
    nlooks : float
        The equivalent number of independent looks used to form the sample coherence. An
        estimate of the number of statistically independent samples averaged in the
        multilooked data, taking into account spatial correlation due to
        oversampling/filtering (see `Notes`_).
    cost : {'defo', 'smooth'}, optional
        Statistical cost mode. Defaults to 'smooth'.
    init : {'mst', 'mcf'}, optional
        Algorithm used for initialization of unwrapped phase gradients.
        Supported algorithms include Minimum Spanning Tree ('mst') and Minimum
        Cost Flow ('mcf'). Defaults to 'mcf'.
    mask : snaphu.io.InputDataset or None, optional
        Binary mask of valid pixels. Zeros in this raster indicate interferogram
        pixels that should be masked out. If provided, it must have the same dimensions
        as the input interferogram and boolean or 8-bit integer datatype. Defaults to
        None.
    ntiles : (int, int), optional
        Number of tiles along the row/column directions. If `ntiles` is (1, 1), then the
        interferogram will be unwrapped as a single tile. Increasing the number of tiles
        may improve runtime and reduce peak memory utilization, but may also introduce
        tile boundary artifacts in the unwrapped result. Defaults to (1, 1).
    tile_overlap : int or (int, int), optional
        Overlap, in pixels, between neighboring tiles. Increasing overlap may help to
        avoid phase discontinuities between tiles. If `tile_overlap` is a scalar
        integer, the number of overlapping rows and columns will be the same. Defaults
        to 0.
    nproc : int, optional
        Maximum number of child processes to spawn for parallel tile unwrapping. If
        `nproc` is less than 1, use all available processors. Defaults to 1.
    tile_cost_thresh : int, optional
        Cost threshold to use for determining boundaries of reliable regions
        (dimensionless; scaled according to other cost constants). Larger cost threshold
        implies smaller regions -- safer, but more expensive computationally. Defaults
        to 500.
    min_region_size : int, optional
        Minimum size, in pixels, of a reliable region in tile mode. Defaults to 100.
    regrow_conncomps : bool, optional
        If True, the connected component labels will be re-computed using a single tile
        after first unwrapping with multiple tiles. This option is disregarded when
        `ntiles` is (1, 1). Defaults to True.
    scratchdir : path-like or None, optional
        Scratch directory where intermediate processing artifacts are written.
        If the specified directory does not exist, it will be created. If None,
        a temporary directory will be created as though by
        ``tempfile.TemporaryDirectory()``. Defaults to None.
    delete_scratch : bool, optional
        If True, the scratch directory will be automatically removed from the file
        system when the function exits. Otherwise, the scratch directory will be
        preserved. Defaults to True.
    unw : snaphu.io.OutputDataset or None, optional
        An optional output dataset to store the unwrapped phase, in radians. If
        provided, it must have the same dimensions as the input interferogram and
        floating-point datatype. If None, an output array will be allocated internally.
        Defaults to None.
    conncomp : snaphu.io.OutputDataset or None, optional
        An optional output dataset to store the connected component labels. If provided,
        it must have the same dimensions as the input interferogram and any integer
        datatype. If None, an output array will be allocated internally. Defaults to
        None.

    Returns
    -------
    unw : snaphu.io.OutputDataset
        The output unwrapped phase, in radians.
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
    .. [1] C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping with use of
       statistical models for cost functions in nonlinear optimization," Journal of the
       Optical Society of America A, vol. 18, pp. 338-351 (2001).
    .. [2] C. W. Chen and H. A. Zebker, "Phase unwrapping for large SAR interferograms:
       Statistical segmentation and generalized network models," IEEE Transactions on
       Geoscience and Remote Sensing, vol. 40, pp. 1709-1719 (2002).
    """
    # If the `unw` and/or `conncomp` output datasets were not provided, allocate arrays
    # to store these outputs.
    if unw is None:
        unw = np.zeros(shape=igram.shape, dtype=np.float32)
    if conncomp is None:
        conncomp = np.zeros(shape=igram.shape, dtype=np.uint32)

    # Ensure that input & output datasets have valid dimensions and datatypes.
    check_shapes(igram, corr, unw, conncomp, mask)
    check_dtypes(igram, corr, unw, conncomp, mask)

    if nlooks < 1.0:
        errmsg = f"nlooks must be >= 1, instead got {nlooks}"
        raise ValueError(errmsg)

    if cost == "topo":
        errmsg = "'topo' cost mode is not currently supported"
        raise NotImplementedError(errmsg)

    cost_modes = {"defo", "smooth"}
    if cost not in cost_modes:
        errmsg = f"cost mode must be in {cost_modes}, instead got {cost!r}"
        raise ValueError(errmsg)

    init_methods = {"mst", "mcf"}
    if init not in init_methods:
        errmsg = f"init method must be in {init_methods}, instead got {init!r}"
        raise ValueError(errmsg)

    # Validate inputs related to tiling and coerce them to the expected types.
    ntiles, tile_overlap, nproc = normalize_and_validate_tiling_params(
        ntiles=ntiles, tile_overlap=tile_overlap, nproc=nproc
    )

    with scratch_directory(scratchdir, delete=delete_scratch) as dir_:
        # Create a raw binary file in the scratch directory for the interferogram and
        # copy the input data to it. (`mkstemp` is used to avoid data races in case the
        # same scratch directory was used for multiple SNAPHU processes.)
        _, tmp_igram = mkstemp(dir=dir_, prefix="snaphu.igram.", suffix=".c8")
        tmp_igram_mmap = np.memmap(tmp_igram, dtype=np.complex64, shape=igram.shape)
        copy_blockwise(igram, tmp_igram_mmap, transform=nan_to_zero)
        tmp_igram_mmap.flush()

        # Copy the input coherence data to a raw binary file in the scratch directory.
        _, tmp_corr = mkstemp(dir=dir_, prefix="snaphu.corr.", suffix=".f4")
        tmp_corr_mmap = np.memmap(tmp_corr, dtype=np.float32, shape=corr.shape)
        copy_blockwise(corr, tmp_corr_mmap, transform=nan_to_zero)
        tmp_corr_mmap.flush()

        # If a mask was provided, copy the mask data to a raw binary file in the scratch
        # directory.
        if mask is None:
            tmp_mask = None
        else:
            _, tmp_mask = mkstemp(dir=dir_, prefix="snaphu.mask.", suffix=".u1")
            tmp_mask_mmap = np.memmap(tmp_mask, dtype=np.bool_, shape=mask.shape)
            copy_blockwise(mask, tmp_mask_mmap)
            tmp_mask_mmap.flush()

        # Create files in the scratch directory for SNAPHU outputs.
        _, tmp_unw = mkstemp(dir=dir_, prefix="snaphu.unw.", suffix=".f4")
        _, tmp_conncomp = mkstemp(dir=dir_, prefix="snaphu.conncomp.", suffix=".u4")

        config = textwrap.dedent(
            f"""\
            INFILE {tmp_igram}
            INFILEFORMAT COMPLEX_DATA
            CORRFILE {tmp_corr}
            CORRFILEFORMAT FLOAT_DATA
            OUTFILE {tmp_unw}
            OUTFILEFORMAT FLOAT_DATA
            CONNCOMPFILE {tmp_conncomp}
            CONNCOMPOUTTYPE UINT
            LINELENGTH {igram.shape[1]}
            NCORRLOOKS {nlooks}
            STATCOSTMODE {cost.upper()}
            INITMETHOD {init.upper()}
            NTILEROW {ntiles[0]}
            NTILECOL {ntiles[1]}
            ROWOVRLP {tile_overlap[0]}
            COLOVRLP {tile_overlap[1]}
            NPROC {nproc}
            TILECOSTTHRESH {tile_cost_thresh}
            MINREGIONSIZE {min_region_size}
            """
        )
        if mask is not None:
            config += f"BYTEMASKFILE {tmp_mask}\n"

        # Write config parameters to file.
        _, config_file = mkstemp(dir=dir_, prefix="snaphu.config.", suffix=".txt")
        Path(config_file).write_text(config)

        # Run SNAPHU with the specified parameters.
        run_snaphu(config_file)

        # Optionally regrow connected component labels from the unwrapped phase if
        # SNAPHU was run in tiled mode. This step should have no effect if SNAPHU was
        # previously run in single-tile mode, so always skip it in that case.
        single_tile = ntiles == (1, 1)
        if (not single_tile) and regrow_conncomps:
            # The connected component regrowing step takes as input the unwrapped phase,
            # which is missing magnitude information. The magnitude may be necessary,
            # for example, to detect zero-magnitude pixels which should be masked out
            # (i.e. connected component label set to 0). So compute the interferogram
            # magnitude and pass it as a separate input file.
            _, tmp_mag = mkstemp(dir=dir_, prefix="snaphu.mag.", suffix=".f4")
            tmp_mag_mmap = np.memmap(tmp_mag, dtype=np.float32, shape=igram.shape)
            copy_blockwise(tmp_igram_mmap, tmp_mag_mmap, transform=np.abs)
            tmp_mag_mmap.flush()

            # Re-run SNAPHU to compute new connected components from the unwrapped phase
            # as though in single-tile mode, overwriting the original connected
            # components file.
            regrow_conncomp_from_unw(
                unw_file=tmp_unw,
                corr_file=tmp_corr,
                mag_file=tmp_mag,
                conncomp_file=tmp_conncomp,
                line_length=igram.shape[1],
                nlooks=nlooks,
                cost=cost,
                scratchdir=dir_,
                mask_file=tmp_mask,
            )

        # Get the output unwrapped phase data.
        tmp_unw_mmap = np.memmap(tmp_unw, dtype=np.float32, shape=unw.shape)
        copy_blockwise(tmp_unw_mmap, unw)

        # Get the output connected component labels.
        tmp_cc_mmap = np.memmap(tmp_conncomp, dtype=np.uint32, shape=conncomp.shape)
        copy_blockwise(tmp_cc_mmap, conncomp)

    return unw, conncomp
