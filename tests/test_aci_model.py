"""Numerical and state-machine checks against the original ACI equations."""

import math
import unittest

import numpy as np

from baselines.aci.model import ACI


class ACIModelTests(unittest.TestCase):
    def test_default_quantile_matches_author_r_type_seven(self):
        model = ACI().fit([0., 1., 4., 10.])
        self.assertAlmostEqual(model.predict(), 8.2)
        self.assertEqual(model.gamma, .005)
        self.assertEqual(model.window_size, 1250)

    def test_reference_trajectory_and_rolling_history(self):
        calibration = [1., 4., 2., 4., 8., 3.]
        observations = [20., 0., 4., 1., 80., 0., 0., 2., 1., 5.]
        for method in ("linear", "inverted_cdf"):
            with self.subTest(method=method):
                model = ACI(alpha=.2, gamma=.1, window_size=4, quantile_method=method).fit(calibration)
                history, level = calibration[-4:], .2
                errors = []
                for score in observations:
                    expected = (math.inf if level <= 0 else -math.inf if level >= 1
                                else np.quantile(history, 1-level, method=method))
                    self.assertAlmostEqual(model.predict(), expected)
                    diagnostic = model.observe(score)
                    error = int(score > expected)
                    errors.append(error)
                    self.assertEqual(diagnostic["error"], error)
                    self.assertAlmostEqual(diagnostic["alpha_used"], level)
                    level += .1 * (.2-error)
                    history = (history + [score])[-4:]
                    self.assertAlmostEqual(model.alpha_t, level)
                    self.assertEqual(model.history_size, 4)
                self.assertAlmostEqual(model.alpha_t, .2 + .1 * (len(observations)*.2 - sum(errors)))

    def test_full_sets_are_not_clipped_and_continue_updating(self):
        model = ACI(alpha=.1, gamma=.05, initial_alpha=.02).fit([0., 1.])
        model.predict()
        model.observe(10)
        self.assertAlmostEqual(model.alpha_t, -.025)
        self.assertEqual(model.predict(), math.inf)
        diagnostic = model.observe(1e6)
        self.assertEqual(diagnostic["error"], 0)
        self.assertAlmostEqual(model.alpha_t, -.02)

    def test_inclusive_boundary_conventions_and_ties(self):
        for initial, expected, error in ((0., math.inf, 0), (1., -math.inf, 1)):
            model = ACI(initial_alpha=initial).fit([1., 2.])
            self.assertEqual(model.predict(), expected)
            self.assertEqual(model.observe(0)["error"], error)
        model = ACI().fit([3., 3.])
        self.assertEqual(model.predict(), 3.)
        self.assertEqual(model.observe(3.)["error"], 0)

    def test_fit_resets_alpha_and_only_seeds_history(self):
        model = ACI(window_size=2).fit([1., 2., 3.])
        model.predict()
        model.observe(100)
        self.assertNotEqual(model.alpha_t, .1)
        model.fit([4., 5., 6.])
        self.assertEqual(model.alpha_t, .1)
        self.assertEqual(model.history_size, 2)
        self.assertAlmostEqual(model.predict(), 5.9)

    def test_issue_observe_order_is_enforced(self):
        model = ACI()
        with self.assertRaises(RuntimeError):
            model.predict()
        model.fit([1.])
        with self.assertRaises(RuntimeError):
            model.observe(1.)
        model.predict()
        with self.assertRaises(RuntimeError):
            model.predict()
        model.observe(1.)

    def test_expanding_window_and_fixed_level_control(self):
        model = ACI(gamma=0, window_size=None).fit([1., 2.])
        model.predict()
        model.observe(100)
        self.assertEqual(model.history_size, 3)
        self.assertEqual(model.alpha_t, .1)
        self.assertAlmostEqual(model.predict(), 80.4)

    def test_invalid_inputs_rejected(self):
        for kwargs in ({"alpha": 0}, {"alpha": 1}, {"alpha": True}, {"gamma": -1},
                       {"gamma": math.inf}, {"window_size": 0}, {"window_size": True},
                       {"initial_alpha": -.1}, {"initial_alpha": 1.1},
                       {"quantile_method": "nearest"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ACI(**kwargs)
        for scores in ([], [-1.], [math.nan], [math.inf], [[1., 2.]], ["1"]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                ACI().fit(scores)
        model = ACI().fit([1.])
        model.predict()
        for score in (-1, math.nan, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                model.observe(score)


if __name__ == "__main__":
    unittest.main()
