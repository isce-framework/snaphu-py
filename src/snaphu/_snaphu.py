from __future__ import annotations

import importlib.resources
import os
import re
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "get_snaphu_executable",
    "get_snaphu_version",
    "run_snaphu",
]


@contextmanager
def get_snaphu_executable() -> Generator[Path, None, None]:
    """
    Get the file path of the SNAPHU executable.

    Returns a context manager for use in a ``with`` statement. The SNAPHU executable is
    included in this Python package. If the package was installed as a zip archive, the
    executable will be extracted to a temporary file that is cleaned up upon exiting the
    context manager.

    Yields
    ------
    pathlib.Path
        The file path of the SNAPHU executable.
    """
    files = importlib.resources.files(__package__)
    with importlib.resources.as_file(files / "snaphu") as snaphu:
        yield snaphu


def get_snaphu_version() -> str:
    """
    Get the SNAPHU version, in '<major>.<minor>.<patch>' format.

    Returns
    -------
    str
        The version string reported by the SNAPHU executable.
    """
    # XXX Currently, this runs `snaphu -h` to get the help message and then parses that
    # with a regex to extract the version string. Alternatively, we could:
    #   * parse the `snaphu.h` header file (not zip-safe), or
    #   * simply hard-code the version string here (not easily maintainable)

    # Run `snaphu -h` to get the help message.
    with get_snaphu_executable() as snaphu:
        args = [os.fspath(snaphu), "-h"]
        result = subprocess.run(args, capture_output=True, text=True)  # noqa: PLW1510

        # On success, `snaphu -h` returns 1 (not 0!) and doesn't write to stderr.
        if (result.returncode != 1) or (result.stderr != ""):
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )

    # Parse the output to extract the version string.
    regex = re.compile(r"^snaphu v(?P<version>[0-9]+(?:\.[0-9]+)*)$", re.MULTILINE)
    match = regex.search(result.stdout)
    if match is None:
        errmsg = "failed to get version string"
        raise RuntimeError(errmsg)

    return match.group("version")


def run_snaphu(config_file: str | os.PathLike[str]) -> None:
    """
    Run SNAPHU with the specified config file.

    Parameters
    ----------
    config_file : path-like
        The file path of a text file storing configuration parameters to pass to SNAPHU.
    """
    if not Path(config_file).is_file():
        errmsg = f"config file not found: {config_file}"
        raise FileNotFoundError(errmsg)

    with get_snaphu_executable() as snaphu:
        args = [os.fspath(snaphu), "-f", os.fspath(config_file)]
        try:
            subprocess.run(args, stderr=subprocess.PIPE, check=True, text=True)
        except subprocess.CalledProcessError as e:
            errmsg = e.stderr.strip()
            raise RuntimeError(errmsg) from e
