import shutil
import tempfile
from pathlib import Path

import pytest

from snaphu._util import scratch_directory


class TestScratchDirectory:
    def test_temp_directory(self):
        with scratch_directory() as scratchdir:
            assert scratchdir.is_dir()
        assert not scratchdir.exists()

    def test_create_directory(self):
        # Create a temporary parent directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            # The file path to create the scratch directory at.
            path = Path(tmpdir) / "a/s/d/f"
            assert not path.is_dir()

            # Create a new scratch directory.
            with scratch_directory(path) as scratchdir:
                # Check that a directory was created in the right place.
                assert scratchdir == path
                assert scratchdir.is_dir()

    @pytest.mark.parametrize("delete", [True, False])
    def test_existing_directory(self, *, delete: bool):
        with tempfile.TemporaryDirectory() as tmpdir:
            with scratch_directory(tmpdir, delete=delete) as scratchdir:
                assert scratchdir == Path(tmpdir)

            # An existing directory should not be cleaned up by the context manager,
            # regardless of the `delete` parameter.
            assert Path(tmpdir).is_dir()

    @pytest.mark.parametrize("delete", [True, False])
    def test_delete(self, *, delete: bool):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a scratch directory at the specified path.
            path = Path(tmpdir) / "asdf"
            with scratch_directory(path, delete=delete) as scratchdir:
                assert scratchdir.is_dir()

            # Check that the scratch directory was removed if `delete` was True.
            if delete:
                assert not scratchdir.exists()
            else:
                assert scratchdir.is_dir()

    @pytest.mark.parametrize("delete", [True, False])
    def test_tempdir_delete(self, *, delete: bool):
        try:
            # Create a scratch directory using `mkdtemp()`.
            with scratch_directory(delete=delete) as scratchdir:
                assert scratchdir.is_dir()

            # Check that the scratch directory was removed if `delete` was True.
            if delete:
                assert not scratchdir.exists()
            else:
                assert scratchdir.exists()
        finally:
            # Clean up the directory if it wasn't automatically removed.
            if scratchdir.is_dir():
                shutil.rmtree(scratchdir)
