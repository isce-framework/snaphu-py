from typing import Any

import numpy as np
import pytest

import snaphu


class TestUnwrap:
    @pytest.mark.parametrize("cost", ["defo", "smooth"])
    @pytest.mark.parametrize("init", ["mst", "mcf"])
    def test_unwrapped_phase(self, cost: str, init: str):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape, dtype=np.float32)

        # Unwrap.
        unw, _ = snaphu.unwrap(igram, corr, nlooks=1.0, cost=cost, init=init)

        # The unwrapped phase may differ from the true phase by a fixed integer multiple
        # of 2pi.
        mean_diff = np.mean(unw - phase)
        offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
        np.testing.assert_allclose(unw, phase + offset, atol=1e-3)

    def test_mask(self):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape, dtype=np.float32)

        # Create a binary mask of valid samples.
        mask = np.zeros(igram.shape, dtype=np.bool_)
        mask[128:-128] = True

        # Unwrap.
        unw, conncomp = snaphu.unwrap(igram, corr, nlooks=1.0, mask=mask)

        # The unwrapped phase may differ from the true phase by a fixed integer multiple
        # of 2pi.
        mean_diff = np.mean(unw[mask] - phase[mask])
        offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
        np.testing.assert_allclose(unw[mask], phase[mask] + offset, atol=1e-3)

        # There should be a single connected component (labeled 1) that contains all of
        # the valid pixels and none of the invalid pixels.
        np.testing.assert_array_equal(conncomp, mask.astype(np.int32))

    @pytest.mark.parametrize("nan", [np.nan, 1j * np.nan])
    def test_nans(self, nan: complex):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape, dtype=np.float32)

        # Create a binary mask of valid samples.
        mask = np.zeros(igram.shape, dtype=np.bool_)
        mask[128:-128] = True

        # Set masked-out interferogram pixels to NaN.
        igram[~mask] = nan

        # Unwrap.
        unw, conncomp = snaphu.unwrap(igram, corr, nlooks=1.0)

        # The unwrapped phase may differ from the true phase by a fixed integer multiple
        # of 2pi.
        mean_diff = np.mean(unw[mask] - phase[mask])
        offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
        np.testing.assert_allclose(unw[mask], phase[mask] + offset, atol=1e-3)

        # There should be a single connected component (labeled 1) that contains all of
        # the valid pixels and none of the invalid pixels.
        np.testing.assert_array_equal(conncomp, mask.astype(np.int32))

    @pytest.mark.parametrize("nproc", [1, 2, -1])
    def test_tiling(self, nproc: int):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-6:6:1024j, -6:6:1024j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape, dtype=np.float32)

        # Unwrap.
        unw, _ = snaphu.unwrap(
            igram,
            corr,
            nlooks=1.0,
            ntiles=(2, 2),
            tile_overlap=(128, 128),
            nproc=nproc,
        )

        # The unwrapped phase may differ from the true phase by a fixed integer multiple
        # of 2pi.
        mean_diff = np.mean(unw - phase)
        offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
        np.testing.assert_allclose(unw, phase + offset, atol=1e-2)

    def test_regrow_conncomps(self):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Create a binary mask containing a single U-shaped region of valid data that
        # spans the 4 quadrants of the interferogram.
        mask = np.zeros(igram.shape, dtype=np.bool_)
        mask[:, :128] = True
        mask[-128:, :] = True
        mask[:, -128:] = True

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape, dtype=np.float32)

        # Unwrap using a 2x2 grid of tiles. Without regrowing connected components, this
        # would result in 4 distinct connected component labels (one per tile).
        _, conncomp = snaphu.unwrap(
            igram,
            corr,
            nlooks=1.0,
            mask=mask,
            ntiles=(2, 2),
            tile_overlap=(64, 64),
            regrow_conncomps=True,
        )

        # There should be a single connected component (labeled 1) that contains all of
        # the valid pixels and none of the invalid pixels.
        np.testing.assert_array_equal(conncomp, mask.astype(np.int32))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"ntiles": (2, 2), "tile_overlap": (64, 64), "regrow_conncomps": False},
            {"ntiles": (2, 2), "tile_overlap": (64, 64), "regrow_conncomps": True},
        ],
    )
    def test_zero_magnitude(self, kwargs: dict[str, Any]):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape, dtype=np.float32)

        # Create a binary mask of valid (nonzero-magnitude) pixels.
        mask = np.zeros(igram.shape, dtype=np.bool_)
        mask[128:-128] = True

        # Set magnitude & coherence of invalid pixels to zero.
        igram[~mask] = 0.0
        corr[~mask] = 0.0

        # Unwrap with the specified keyword arguments.
        _, conncomp = snaphu.unwrap(igram, corr, nlooks=1.0, **kwargs)

        # The zero-magnitude pixels should all have connected component label 0. Nonzero
        # pixels should all have a nonzero connected component label.
        np.testing.assert_array_equal(conncomp != 0, mask)

    def test_shape_mismatch(self):
        igram = np.empty(shape=(128, 128), dtype=np.complex64)
        corr = np.empty(shape=(128, 129), dtype=np.float32)
        pattern = (
            "^shape mismatch: corr and igram must have the same shape, instead got"
            r" corr.shape=\(128, 129\) and igram.shape=\(128, 128\)$"
        )
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0)

    def test_bad_igram_dtype(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.float64)
        corr = np.empty(shape, dtype=np.float32)
        pattern = r"^igram must be a complex-valued array, instead got dtype=float64$"
        with pytest.raises(TypeError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0)

    def test_bad_corr_dtype(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.complex64)
        pattern = r"^corr must be a real-valued array, instead got dtype=complex64$"
        with pytest.raises(TypeError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0)

    def test_bad_nlooks(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.float32)
        pattern = "^nlooks must be >= 1, instead got 0$"
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=0)

    def test_bad_cost(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.float32)
        pattern = r"^cost mode must be in \{.*\}, instead got 'asdf'$"
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0, cost="asdf")

    def test_bad_init(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.float32)
        pattern = r"^init method must be in \{.*\}, instead got 'asdf'$"
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0, init="asdf")

    def test_bad_ntiles(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.float32)

        pattern = r"^ntiles must be a pair of ints, instead got ntiles=\(1, 2, 3\)$"
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0, ntiles=(1, 2, 3))  # type: ignore[arg-type]

        pattern = (
            r"^ntiles may not contain negative or zero values, got ntiles=\(1, 0\)$"
        )
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0, ntiles=(1, 0))

    def test_bad_tile_overlap(self):
        shape = (128, 128)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.float32)

        pattern = (
            r"^tile_overlap must be an int or pair of ints, instead got"
            r" tile_overlap=\(1, 2, 3\)$"
        )
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0, tile_overlap=(1, 2, 3))  # type: ignore[arg-type]

        pattern = (
            r"^tile_overlap may not contain negative values, got"
            r" tile_overlap=\(0, -1\)$"
        )
        with pytest.raises(ValueError, match=pattern):
            snaphu.unwrap(igram, corr, nlooks=100.0, tile_overlap=(0, -1))

    def test_bad_min_region_size(self):
        shape = (256, 256)
        igram = np.empty(shape, dtype=np.complex64)
        corr = np.empty(shape, dtype=np.float32)

        pattern = r"^minimum region size too large for given tile parameters$"
        with pytest.raises(RuntimeError, match=pattern):
            snaphu.unwrap(
                igram,
                corr,
                nlooks=100.0,
                ntiles=(2, 2),
                min_region_size=1_000_000,
            )
