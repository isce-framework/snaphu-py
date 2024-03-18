import numpy as np
import pytest
from numpy.typing import ArrayLike

import snaphu


def jaccard_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the Jaccard index of two binary masks.

    Returns the Jaccard similarity coefficient (Intersect-over-Union) of the two input
    boolean arrays. A value of 1 indicates that the two masks are identical. A value of
    0 indicates that the two masks are non-overlapping. Values between 0 and 1 indicate
    partial overlap.

    Parameters
    ----------
    a, b : array_like
        The input binary masks.

    Returns
    -------
    float
        The Jaccard similarity coefficient.
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)

    if (a.dtype != np.bool_) or (b.dtype != np.bool_):
        errmsg = (
            f"the input arrays must each have boolean dtype, instead got {a.dtype=} and"
            f" {b.dtype=}"
        )
        raise TypeError(errmsg)

    return np.sum(a & b) / np.sum(a | b)  # type: ignore[no-any-return]


def simulate_phase_noise(
    corr: ArrayLike, nlooks: float, *, seed: int | None = None
) -> np.ndarray:
    r"""
    Simulate InSAR phase noise.

    Generate pseudo-random noise samples that approximately match the expected
    distribution of multilooked interferogram phase.

    The resulting samples are zero-mean Gaussian distributed, with variance equal to the
    Cramer-Rao bound of the Maximum Likelihood Estimator for the interferometric phase\
    [1]_. This simple approximation is most accurate for high coherence and large number
    of looks. The true phase distribution is more complicated\ [2]_.

    Parameters
    ----------
    corr : array_like
        The interferometric coherence magnitude.
    nlooks : float
        The number of independent looks used to form the interferogram.
    seed : int or None, optional
        A seed for initializing pseudo-random number generator state. Must be
        nonnegative. If None, then the generator will be initialized randomly. Defaults
        to None.

    Returns
    -------
    numpy.ndarray
        Phase noise samples, in radians.

    References
    ----------
    .. [1] E. Rodriguez, and J. M. Martin, "Theory and design of interferometric
       synthetic aperture radars," IEE Proceedings-F, vol. 139, no. 2, pp. 147-159,
       April 1992.
    .. [2] J. S. Lee, K. W. Hoppel, S. A. Mango, and A. R. Miller, "Intensity and phase
       statistics of multilook polarimetric and interferometric SAR imagery," IEEE
       Trans. Geosci. Remote Sens. 32, 1017-1028 (1994).
    """
    # Setup a pseudo-random number generator.
    rng = np.random.default_rng(seed)

    # Approximate interferometric phase standard deviation using a simple approximation
    # that holds for high coherence / number of looks.
    corr = np.asanyarray(corr)
    sigma = 1.0 / np.sqrt(2.0 * nlooks) * np.sqrt(1.0 - corr**2) / corr

    # Generate zero-mean Gaussian-distributed phase noise samples.
    return rng.normal(scale=sigma)


class TestGrowConnComps:
    def test_single_component(self):
        shape = (128, 129)
        unw = np.zeros(shape=shape, dtype=np.float32)
        corr = np.ones(shape=shape, dtype=np.float32)

        conncomp = snaphu.grow_conncomps(unw, corr, nlooks=1.0)

        np.testing.assert_array_equal(conncomp, 1)
        assert conncomp.shape == unw.shape

    def test_output_parameter(self):
        unw = np.zeros(shape=(128, 128), dtype=np.float32)
        corr = np.ones(unw.shape, dtype=np.float32)

        # Test passing a `conncomp` array as an output parameter.
        conncomp = np.zeros(unw.shape, dtype=np.uint8)
        snaphu.grow_conncomps(unw, corr, nlooks=1.0, conncomp=conncomp)

        np.testing.assert_array_equal(conncomp, 1)

    def test_low_coherence(self):
        # Simulate a diagonal phase ramp.
        y, x = np.ogrid[-3:3:1024j, -3:3:1024j]
        unw = np.pi * (x + y)

        # Simulate a single square-shaped island of high-coherence surrounded by
        # low-coherence pixels.
        corr = np.full(unw.shape, fill_value=0.1, dtype=np.float32)
        corr[256:-256, 256:-256] = 1.0

        # Add simulated phase noise to unwrapped phase.
        nlooks = 100.0
        unw += simulate_phase_noise(corr, nlooks, seed=1234)

        # Grow connected components.
        conncomp = snaphu.grow_conncomps(unw, corr, nlooks=1.0)

        # Create a mask of pixels that are expected to have nonzero connected component
        # label.
        mask = corr > 0.5

        # Check approximate locations of nonzero connected component labels.
        assert jaccard_similarity(conncomp != 0, mask) > 0.9

    # Helper method to check connected component labels for the next few test cases.
    def check_conncomp_4_quadrants(self, conncomp: np.ndarray) -> None:
        # Check unique connected component labels.
        assert set(np.unique(conncomp)) == {0, 1, 2, 3, 4}

        # Check each quadrants' connected component label. Ignore the outermost
        # row/column of each quadrant since the way SNAPHU grows connected components
        # sometimes erodes corners.
        np.testing.assert_array_equal(conncomp[:255, :255], 1)
        np.testing.assert_array_equal(conncomp[:255, -254:], 2)
        np.testing.assert_array_equal(conncomp[-254:, :255], 3)
        np.testing.assert_array_equal(conncomp[-254:, -254:], 4)

    def test_mask(self):
        # Simulate a diagonal phase ramp.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        unw = np.pi * (x + y)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(unw.shape, dtype=np.float32)

        # Create a binary mask that subdivides the array into 4 disjoint quadrants of
        # valid samples separated by a single row & column of invalid samples.
        mask = np.ones(unw.shape, dtype=np.bool_)
        mask[256, :] = False
        mask[:, 256] = False

        # Grow connected components.
        conncomp = snaphu.grow_conncomps(unw, corr, nlooks=1.0, mask=mask)

        self.check_conncomp_4_quadrants(conncomp)

    def test_zero_magnitude(self):
        # Simulate a diagonal phase ramp.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        unw = np.pi * (x + y)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(unw.shape, dtype=np.float32)

        # Create an array of interferogram magnitude data. A single row & column of
        # zero-magnitude pixels subdivides the array into 4 disjoint quadrants of
        # nonzero-magnitude pixels.
        mag = np.ones(unw.shape, dtype=np.float32)
        mag[256, :] = 0.0
        mag[:, 256] = 0.0

        # Grow connected components.
        conncomp = snaphu.grow_conncomps(unw, corr, nlooks=1.0, mag=mag)

        self.check_conncomp_4_quadrants(conncomp)

    def test_nans(self):
        # Simulate a diagonal phase ramp.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        unw = np.pi * (x + y)

        # Sample coherence for an interferogram with no noise.
        corr = np.ones(unw.shape, dtype=np.float32)

        # Simulate interferogram magnitudes
        mag = np.ones(unw.shape, dtype=np.float32)

        # Insert a single row & column of NaN values to the magnitude & unwrapped phase
        # data such that each array is subdivided into 4 disjoint quadrants of valid
        # pixels separated by NaNs.
        mag[256, :] = np.nan
        mag[:, 256] = np.nan
        unw[256, :] = np.nan
        unw[:, 256] = np.nan

        # Grow connected components.
        conncomp = snaphu.grow_conncomps(unw, corr, nlooks=1.0, mag=mag)

        self.check_conncomp_4_quadrants(conncomp)

    def test_shape_mismatch(self):
        unw = np.empty(shape=(128, 128), dtype=np.float32)
        corr = np.empty(shape=(128, 129), dtype=np.float32)
        pattern = (
            "^shape mismatch: corr and unw must have the same shape, instead got"
            r" corr.shape=\(128, 129\) and unw.shape=\(128, 128\)$"
        )
        with pytest.raises(ValueError, match=pattern):
            snaphu.grow_conncomps(unw, corr, nlooks=100.0)

    def test_bad_unw_dtype(self):
        unw = np.empty(shape=(128, 128), dtype=np.complex64)
        corr = np.empty(unw.shape, dtype=np.float32)
        pattern = r"^unw must be a real-valued array, instead got dtype=complex64$"
        with pytest.raises(TypeError, match=pattern):
            snaphu.grow_conncomps(unw, corr, nlooks=100.0)

    def test_bad_corr_dtype(self):
        unw = np.empty(shape=(128, 128), dtype=np.float32)
        corr = np.empty(unw.shape, dtype=np.complex64)
        pattern = r"^corr must be a real-valued array, instead got dtype=complex64$"
        with pytest.raises(TypeError, match=pattern):
            snaphu.grow_conncomps(unw, corr, nlooks=100.0)

    def test_bad_nlooks(self):
        unw = np.empty(shape=(128, 128), dtype=np.float32)
        corr = np.empty(unw.shape, dtype=np.float32)
        pattern = "^nlooks must be >= 1, instead got 0.5$"
        with pytest.raises(ValueError, match=pattern):
            snaphu.grow_conncomps(unw, corr, nlooks=0.5)

    def test_bad_cost(self):
        unw = np.empty(shape=(128, 128), dtype=np.float32)
        corr = np.empty(unw.shape, dtype=np.float32)
        pattern = r"^cost mode must be in \{.*\}, instead got 'asdf'$"
        with pytest.raises(ValueError, match=pattern):
            snaphu.grow_conncomps(unw, corr, nlooks=100.0, cost="asdf")
