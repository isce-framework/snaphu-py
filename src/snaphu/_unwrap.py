from __future__ import annotations

import io
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import cast, overload

import numpy as np

from ._snaphu import run_snaphu
from ._util import BlockIterator, scratch_directory
from .io import InputDataset, OutputDataset

__all__ = [
    "unwrap",
]


@dataclass(frozen=True)
class TilingParams:
    """
    SNAPHU configuration parameters affecting scene tiling and parallel processing.

    Parameters
    ----------
    ntilerow, ntilecol : int, optional
        Number of tiles along the row/column directions. If `ntilerow` and `ntilecol`
        are both 1 (the default), the interferogram will be unwrapped as a single tile.
    rowovrlp, colovrlp : int, optional
        Overlap, in number of rows/columns, between neighboring tiles. Defaults to 0.
    nproc : int, optional
        Maximum number of child processes to spawn for parallel tile unwrapping.
        Defaults to 1.
    """

    ntilerow: int = 1
    ntilecol: int = 1
    rowovrlp: int = 0
    colovrlp: int = 0
    nproc: int = 1

    def to_string(self) -> str:
        """
        Write SNAPHU tiling parameters to a string.

        Creates a multi-line string in SNAPHU configuration file format.

        Returns
        -------
        str
            The output string.
        """
        return textwrap.dedent(f"""\
            NTILEROW {self.ntilerow}
            NTILECOL {self.ntilecol}
            ROWOVRLP {self.rowovrlp}
            COLOVRLP {self.colovrlp}
            NPROC {self.nproc}
        """)


@dataclass(frozen=True)
class SnaphuConfig:
    """
    SNAPHU configuration parameters.

    Parameters
    ----------
    infile : path-like
        The input interferogram file path.
    corrfile : path-like
        The input coherence file path.
    outfile : path-like
        The output unwrapped phase file path.
    conncompfile : path-like
        The output connected component labels file path.
    linelength : int
        The line length, in samples, of the input interferogram data array.
    ncorrlooks : float
        The equivalent number of independent looks used to form the coherence data.
    statcostmode : str
        The statistical cost mode.
    initmethod : str
        The algorithm used for initializing the network solver routine.
    bytemaskfile : path-like or None, optional
        An optional file path of a byte mask file. If None, no mask is applied. Defaults
        to None.
    tiling_params : TilingParams or None, optional
        Optional additional configuration parameters affecting scene tiling and parallel
        processing. Defaults to None.
    """

    infile: str | os.PathLike[str]
    corrfile: str | os.PathLike[str]
    outfile: str | os.PathLike[str]
    conncompfile: str | os.PathLike[str]
    linelength: int
    ncorrlooks: float
    statcostmode: str
    initmethod: str
    bytemaskfile: str | os.PathLike[str] | None = None
    tiling_params: TilingParams | None = None

    def to_string(self) -> str:
        """
        Write SNAPHU configuration parameters to a string.

        Creates a multi-line string in SNAPHU configuration file format.

        Returns
        -------
        str
            The output string.
        """
        config = textwrap.dedent(f"""\
            INFILE {os.fspath(self.infile)}
            INFILEFORMAT COMPLEX_DATA
            CORRFILE {os.fspath(self.corrfile)}
            CORRFILEFORMAT FLOAT_DATA
            OUTFILE {os.fspath(self.outfile)}
            OUTFILEFORMAT FLOAT_DATA
            CONNCOMPFILE {os.fspath(self.conncompfile)}
            CONNCOMPOUTTYPE UINT
            LINELENGTH {self.linelength}
            NCORRLOOKS {self.ncorrlooks}
            STATCOSTMODE {self.statcostmode.upper()}
            INITMETHOD {self.initmethod.upper()}
        """)

        if self.bytemaskfile is not None:
            config += f"BYTEMASKFILE {os.fspath(self.bytemaskfile)}\n"
        if self.tiling_params is not None:
            config += self.tiling_params.to_string()

        return config

    def _to_file_textio(self, file_: io.TextIOBase, /) -> None:
        # Write config params to file.
        s = self.to_string()
        count = file_.write(s)

        # Check that the full text was successfully written to the file.
        if count != len(s):
            errmsg = "failed to write config params to file"
            raise RuntimeError(errmsg)

    def _to_file_pathlike(self, file_: str | os.PathLike[str], /) -> None:
        # Create the file's parent directory(ies) if they didn't already exist.
        p = Path(file_)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Write config params to file.
        s = self.to_string()
        p.write_text(s)

    def to_file(self, file_: str | os.PathLike[str] | io.TextIOBase, /) -> None:
        """
        Write SNAPHU configuration parameters to a file.

        The resulting file is suitable for passing to the SNAPHU executable as a
        configuration file.

        Parameters
        ----------
        file_ : path-like or file-like
            The output file. May be an open text file or a file path. If the file
            and any of its parent directories do not exist, they will be created. If the
            path to an existing file is specified, the file will be overwritten.
        """
        if isinstance(file_, io.TextIOBase):
            self._to_file_textio(file_)
        elif isinstance(file_, (str, os.PathLike)):
            self._to_file_pathlike(file_)
        else:
            errmsg = (
                "to_file argument must be a path-like or file-like object, instead got"
                f" type={type(file_)}"
            )
            raise TypeError(errmsg)


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


def copy_blockwise(
    src: InputDataset,
    dst: OutputDataset,
    chunks: tuple[int, int] = (1024, 1024),
    *,
    nan_to_zero: bool = False,
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
    nan_to_zero : bool, optional
        If True, replace Not a Number (NaN) values in the array with zeros in the
        output. Defaults to False.
    """
    shape = src.shape
    if dst.shape != shape:
        errmsg = (
            "shape mismatch: src and dst must have the same shape, instead got"
            f" {src.shape=} and {dst.shape=}"
        )
        raise ValueError(errmsg)

    for block in BlockIterator(shape, chunks):
        if nan_to_zero:
            nan_mask = np.isnan(src[block])
            dst[block] = np.where(nan_mask, 0.0, src[block])
        else:
            dst[block] = src[block]


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
        copy_blockwise(igram, tmp_igram_mmap, nan_to_zero=True)

        # Copy the input coherence data to a raw binary file in the scratch directory.
        _, tmp_corr = mkstemp(dir=dir_, prefix="snaphu.corr.", suffix=".f4")
        tmp_corr_mmap = np.memmap(tmp_corr, dtype=np.float32, shape=corr.shape)
        copy_blockwise(corr, tmp_corr_mmap, nan_to_zero=True)

        # If a mask was provided, copy the mask data to a raw binary file in the scratch
        # directory.
        if mask is None:
            tmp_mask = None
        else:
            _, tmp_mask = mkstemp(dir=dir_, prefix="snaphu.mask.", suffix=".u1")
            tmp_mask_mmap = np.memmap(tmp_mask, dtype=np.bool_, shape=mask.shape)
            copy_blockwise(mask, tmp_mask_mmap)

        # Create files in the scratch directory for SNAPHU outputs.
        _, tmp_unw = mkstemp(dir=dir_, prefix="snaphu.unw.", suffix=".f4")
        _, tmp_conncomp = mkstemp(dir=dir_, prefix="snaphu.conncomp.", suffix=".u4")

        tiling_params = TilingParams(
            ntilerow=ntiles[0],
            ntilecol=ntiles[1],
            rowovrlp=tile_overlap[0],
            colovrlp=tile_overlap[1],
            nproc=nproc,
        )

        config = SnaphuConfig(
            infile=tmp_igram,
            corrfile=tmp_corr,
            outfile=tmp_unw,
            conncompfile=tmp_conncomp,
            linelength=igram.shape[1],
            ncorrlooks=nlooks,
            statcostmode=cost,
            initmethod=init,
            bytemaskfile=tmp_mask,
            tiling_params=tiling_params,
        )

        # Write config parameters to file.
        config_file = dir_ / "snaphu.conf"
        config.to_file(config_file)

        # Run SNAPHU with the specified parameters.
        run_snaphu(config_file)

        # Get the output unwrapped phase data.
        tmp_unw_mmap = np.memmap(tmp_unw, dtype=np.float32, shape=unw.shape)
        copy_blockwise(tmp_unw_mmap, unw)

        # Get the output connected component labels.
        tmp_cc_mmap = np.memmap(tmp_conncomp, dtype=np.uint32, shape=conncomp.shape)
        copy_blockwise(tmp_cc_mmap, conncomp)

    return unw, conncomp
