import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from baselines.kowcpi.model import WeightedNadarayaWatson


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "KOWCPI_Codes" / "weighted_nw.py"
ReferenceNadarayaWatson = None
if REFERENCE_PATH.is_file():
    specification = importlib.util.spec_from_file_location("_kowcpi_bandwidth_reference", REFERENCE_PATH)
    reference_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference_module)
    ReferenceNadarayaWatson = reference_module.WeightedNadarayaWatson


class KOWCPIBandwidthTests(unittest.TestCase):
    def setUp(self):
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.simplefilter("error", RuntimeWarning)
        self.x = np.arange(10, dtype=float)[:, None]

    @unittest.skipIf(ReferenceNadarayaWatson is None, "Optional bundled KOWCPI source is unavailable.")
    def test_fast_aic_matches_source_for_normal_and_tiny_positive_rss(self):
        # Every candidate has positive RSS and a positive AIC denominator.
        # Multiplying y by 1e-8 moves every RSS below the former 1e-12 floor.
        bandwidths = np.array([2.0, 3.0, 5.0])
        for kernel in ("epanechnikov", "gaussian"):
            reference = ReferenceNadarayaWatson(kernel=kernel)
            reference._prepare_pairwise(self.x)
            baseline = WeightedNadarayaWatson(kernel=kernel)
            baseline._prepare_pairwise(self.x)
            for scale in (1.0, 1e-8):
                with self.subTest(kernel=kernel, scale=scale):
                    y = self.x[:, 0] * scale
                    expected = reference._aic_for_bandwidths_fast(self.x, y, bandwidths)
                    actual = baseline._aic_for_bandwidths_fast(y, bandwidths)
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @unittest.skipIf(ReferenceNadarayaWatson is None, "Optional bundled KOWCPI source is unavailable.")
    def test_direct_and_coarse_fine_search_match_source(self):
        for kernel in ("epanechnikov", "gaussian"):
            for bandwidths in (np.array([2.0, 3.0, 5.0]), np.linspace(2.0, 5.0, 30)):
                selections = []
                for scale in (1.0, 1e-8):
                    with self.subTest(kernel=kernel, grid_size=len(bandwidths), scale=scale):
                        y = self.x[:, 0] * scale
                        settings = dict(bandwidth=None, kernel=kernel, bandwidth_range=bandwidths)
                        reference = ReferenceNadarayaWatson(**settings).fit(self.x, y)
                        baseline = WeightedNadarayaWatson(**settings).fit(self.x, y)
                        self.assertAlmostEqual(baseline.bandwidth, reference.bandwidth, places=12)
                        selections.append(baseline.bandwidth)
                self.assertAlmostEqual(selections[0], selections[1], places=12)

    def test_target_rescaling_preserves_known_bandwidth_for_both_searches(self):
        for kernel in ("epanechnikov", "gaussian"):
            for bandwidths in (np.array([2.0, 3.0, 5.0]), np.linspace(2.0, 5.0, 30)):
                for scale in (1.0, 1e-8):
                    with self.subTest(kernel=kernel, grid_size=len(bandwidths), scale=scale):
                        baseline = WeightedNadarayaWatson(
                            bandwidth=None, kernel=kernel, bandwidth_range=bandwidths,
                        ).fit(self.x, self.x[:, 0] * scale)
                        # Both source searches select 2.0 on this fixture;
                        # the former RSS floor changed the tiny-scale result to 5.0.
                        self.assertAlmostEqual(baseline.bandwidth, 2.0, places=12)

    def test_fixed_and_cached_bandwidths_skip_reselection(self):
        y = self.x[:, 0]
        fixed = WeightedNadarayaWatson(bandwidth=3.0)
        with patch.object(fixed, "_aic_for_bandwidths_fast", side_effect=AssertionError("Unexpected AIC search")):
            fixed.fit(self.x, y)
        self.assertEqual(fixed.bandwidth, 3.0)

        cached = WeightedNadarayaWatson(bandwidth=None, bandwidth_range=[2.0, 3.0, 5.0])
        cached.fit(self.x, y)
        selected = cached.bandwidth
        with patch.object(cached, "_aic_for_bandwidths_fast", side_effect=AssertionError("Unexpected AIC search")):
            cached.fit(self.x, y[::-1])
        self.assertEqual(cached.bandwidth, selected)

    def test_zero_rss_and_short_histories_keep_finite_aic(self):
        # These degenerate cases intentionally retain baseline guardrails;
        # the unguarded source formula can divide by zero or take log(0).
        for kernel in ("epanechnikov", "gaussian"):
            for count in (1, 2, 3):
                with self.subTest(kernel=kernel, count=count):
                    x = np.arange(count, dtype=float)[:, None]
                    y = np.zeros(count)
                    bandwidths = np.array([0.1, 2.0, 5.0])
                    baseline = WeightedNadarayaWatson(
                        bandwidth=None, kernel=kernel, bandwidth_range=bandwidths,
                    ).fit(x, y)
                    self.assertTrue(np.isfinite(baseline.bandwidth))
                    baseline._prepare_pairwise(x)
                    self.assertTrue(np.isfinite(baseline._aic_for_bandwidths_fast(y, bandwidths)).all())


if __name__ == "__main__":
    unittest.main()
