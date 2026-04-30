import unittest
import numpy as np

from core.explicite import step_explicit
from core.semi_implicite import step_semi_implicit
from core.numerics import _laplacian_all, _safe_denominator, compute_energy, update_F
from core.solver import WesterveltParams


class TestWesterveltValidation(unittest.TestCase):
    def test_parameter_calculation(self):
        c = 1500.0
        rho0 = 1000.0
        beta = 4.8
        mu_v = 1.2e-3
        params = WesterveltParams(c=c, rho0=rho0, beta=beta, mu_v=mu_v)

        expected_b = mu_v / rho0
        expected_k = beta / (rho0 * c**2)

        self.assertAlmostEqual(params.beta, beta, places=12)
        self.assertAlmostEqual(params.b, expected_b, places=12)
        self.assertAlmostEqual(params.k, expected_k, places=12)

    def test_update_F_matches_formula(self):
        c = 1500.0
        dt = 1.0e-6
        dx = 1.0e-2

        u = np.array([0.0, 0.2, 0.1, -0.1, 0.0], dtype=float)
        F = np.array([0.0, 1.0, -2.0, 0.5, 0.0], dtype=float)

        dx2 = dx * dx
        lap = _laplacian_all(u, dx2)
        expected = F + dt * (c * c * lap)
        expected[0] = 0.0
        expected[-1] = 0.0

        computed = update_F(F, u, dt, dx, c, bc_type=0)
        self.assertTrue(np.allclose(computed, expected))

    def test_compute_energy_uses_dx_not_dx_squared(self):
        dx = 0.1
        x = np.linspace(0.0, 1.0, 11)
        u = x.copy()
        u_prev = u.copy()

        energy = compute_energy(u, u_prev, c=2.0, dt=0.01, dx=dx)

        self.assertAlmostEqual(energy, 1.8, places=12)

    def test_explicit_step_matches_formula(self):
        c = 1500.0
        b = 1.0e-5
        k = 2.0e-4
        dt = 1.0e-7
        dx = 5.0e-3

        u = np.array([0.0, 0.3, -0.2, 0.1, 0.0], dtype=float)
        F = np.array([0.0, 0.02, -0.03, 0.01, 0.0], dtype=float)

        F_expected = update_F(F, u, dt, dx, c, bc_type=0)
        lap_u = _laplacian_all(u, dx * dx)
        denom = _safe_denominator(1.0 - 2.0 * k * u)
        u_expected = u + dt * (F_expected + b * lap_u) / denom
        u_expected[0] = 0.0
        u_expected[-1] = 0.0

        u_next, F_next = step_explicit(u, F, c, b, k, dt, dx, bc_type=0)

        self.assertTrue(np.allclose(F_next, F_expected))
        self.assertTrue(np.allclose(u_next, u_expected))

    def test_semi_implicit_with_b_zero_reduces_to_explicit_update(self):
        c = 1500.0
        b = 0.0
        k = 1.0e-4
        dt = 1.0e-7
        dx = 1.0e-2

        u = np.array([0.0, 0.3, -0.2, 0.1, 0.0], dtype=float)
        F = np.array([0.0, 0.02, -0.03, 0.01, 0.0], dtype=float)

        denom = _safe_denominator(1.0 - 2.0 * k * u)
        F_expected = update_F(F, u, dt, dx, c, bc_type=0)
        u_expected = u + dt * F_expected / denom
        u_expected[0] = 0.0
        u_expected[-1] = 0.0

        u_next, F_next = step_semi_implicit(u, F, c, b, k, dt, dx, bc_type=0)

        self.assertTrue(np.allclose(F_next, F_expected))
        self.assertTrue(np.allclose(u_next, u_expected))


if __name__ == "__main__":
    unittest.main()
