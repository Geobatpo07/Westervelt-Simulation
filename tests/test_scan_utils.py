import unittest

import numpy as np

from utils import build_scan_grid, get_scan_axes


class TestScanUtils(unittest.TestCase):
    def test_scan_axes_accept_solver_amplitude_key(self):
        results = [
            {"dt": 1e-8, "amplitude_u0": 0.5, "amplitude_u1": 0.0, "stable": True},
            {"dt": 2e-8, "amplitude_u0": 0.5, "amplitude_u1": 0.0, "stable": False},
            {"dt": 1e-8, "amplitude_u0": 1.0, "amplitude_u1": 0.0, "stable": True},
        ]

        dt_vals, amp_vals = get_scan_axes(results)

        self.assertEqual(dt_vals, [1e-8, 2e-8])
        self.assertEqual(amp_vals, [0.5, 1.0])

    def test_build_scan_grid_accepts_solver_amplitude_key(self):
        results = [
            {"dt": 1e-8, "amplitude_u0": 0.5, "amplitude_u1": 0.0, "stable": True},
            {"dt": 2e-8, "amplitude_u0": 0.5, "amplitude_u1": 0.0, "stable": False},
            {"dt": 1e-8, "amplitude_u0": 1.0, "amplitude_u1": 0.0, "stable": True},
        ]
        dt_vals, amp_vals = get_scan_axes(results)

        grid = build_scan_grid(
            results,
            dt_vals,
            amp_vals,
            lambda r: 1.0 if r["stable"] else 0.0,
            default=np.nan,
        )

        expected = np.array([[1.0, 0.0], [1.0, np.nan]])
        self.assertTrue(np.allclose(grid, expected, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
