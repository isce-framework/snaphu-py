import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

__all__ = [
    "scratch_directory",
]


@contextmanager
def scratch_directory(
    dir_: str | os.PathLike[str] | None = None, *, delete: bool = True
) -> Generator[Path, None, None]:
    """
    Context manager that creates a (possibly temporary) file system directory.

    If `dir_` is a path-like object, a directory will be created at the specified
    file system path if it did not already exist. Otherwise, if `dir_` is None, a
    temporary directory will instead be created as though by
    ``tempfile.TemporaryDirectory()``.

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
        with TemporaryDirectory(delete=delete) as tmpdir:
            yield Path(tmpdir)
    else:
        scratchdir = Path(dir_)
        scratchdir.mkdir(parents=True, exist_ok=True)

        yield scratchdir

        if delete:
            shutil.rmtree(scratchdir)
