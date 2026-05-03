import unittest

import numpy as np

from core.explicite import step_explicit
from core.kernels_numba import (
    allocate_semi_implicit_workspace,
    compute_energy_numba,
    step_explicit_inplace,
    step_semi_implicit_inplace,
)
from core.numerics import compute_energy
from core.semi_implicite import step_semi_implicit


class TestNumbaKernels(unittest.TestCase):
    def test_explicit_kernel_matches_reference(self):
        c = 1500.0
        b = 1.0e-5
        k = 2.0e-4
        dt = 1.0e-7
        dx = 5.0e-3
        u = np.array([0.0, 0.3, -0.2, 0.1, 0.04, 0.0], dtype=np.float64)
        F = np.array([0.0, 0.02, -0.03, 0.01, -0.02, 0.0], dtype=np.float64)

        for bc_type in (0, 1):
            with self.subTest(bc_type=bc_type):
                expected_u, expected_F = step_explicit(u, F, c, b, k, dt, dx, bc_type)
                actual_u = np.empty_like(u)
                actual_F = np.empty_like(F)

                step_explicit_inplace(u, F, actual_u, actual_F, c, b, k, dt, dx, bc_type)

                self.assertTrue(np.allclose(actual_F, expected_F))
                self.assertTrue(np.allclose(actual_u, expected_u))

    def test_semi_implicit_kernel_matches_reference(self):
        c = 1500.0
        b = 1.0e-5
        k = 2.0e-4
        dt = 1.0e-7
        dx = 5.0e-3
        u = np.array([0.0, 0.3, -0.2, 0.1, 0.04, 0.0], dtype=np.float64)
        F = np.array([0.0, 0.02, -0.03, 0.01, -0.02, 0.0], dtype=np.float64)

        for bc_type in (0, 1):
            with self.subTest(bc_type=bc_type):
                expected_u, expected_F = step_semi_implicit(u, F, c, b, k, dt, dx, bc_type)
                actual_u = np.empty_like(u)
                actual_F = np.empty_like(F)
                workspace = allocate_semi_implicit_workspace(u.size)

                step_semi_implicit_inplace(
                    u,
                    F,
                    actual_u,
                    actual_F,
                    c,
                    b,
                    k,
                    dt,
                    dx,
                    bc_type,
                    *workspace,
                )

                self.assertTrue(np.allclose(actual_F, expected_F))
                self.assertTrue(np.allclose(actual_u, expected_u))

    def test_numba_energy_matches_reference(self):
        x = np.linspace(0.0, 1.0, 16)
        u = np.sin(np.pi * x)
        u_prev = 0.9 * u
        c = 2.0
        dt = 0.01
        dx = x[1] - x[0]

        expected = compute_energy(u, u_prev, c, dt, dx)
        actual = compute_energy_numba(u, u_prev, c, dt, dx)

        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
