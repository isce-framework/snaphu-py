import importlib.resources
import os
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "get_snaphu_executable",
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
