import unittest
import numpy as np

from core.numerics import _laplacian_all
from core.solver import WesterveltSolver, WesterveltParams


class TestWesterveltSolver(unittest.TestCase):
    def test_initialization(self):
        params = WesterveltParams(c=1500, nx=100)
        solver = WesterveltSolver(params)
        solver.initialize(u0_type="gaussian", u1_type="gaussian_derivative", A1=2.0, A2=0.5)

        expected_u0 = solver._initial_profile("gaussian", amplitude=2.0)
        expected_u1 = solver._initial_profile("gaussian_derivative", amplitude=0.5)
        expected_u0[0] = expected_u0[-1] = 0.0
        expected_u1[0] = expected_u1[-1] = 0.0

        self.assertEqual(len(solver.u), 100)
        self.assertGreater(float(np.max(solver.u)), 0.0)
        self.assertTrue(np.allclose(solver.u, expected_u0))
        self.assertTrue(np.allclose(solver.u_prev, expected_u0 - params.dt * expected_u1))
        expected_F = (1.0 - 2.0 * params.k * expected_u0) * expected_u1 - params.b * _laplacian_all(expected_u0, params.dx ** 2)
        expected_F[0] = expected_F[-1] = 0.0
        self.assertTrue(np.allclose(solver.F, expected_F))
        self.assertEqual(len(solver.energy_history), 1)

    def test_explicit_step_updates_u_and_F(self):
        params = WesterveltParams(
            c=1500,
            beta=3.5,
            mu_v=1e-3,
            dx=1e-4,
            dt=1e-8,
            nx=128,
            nt=5,
            scheme="explicit",
        )
        solver = WesterveltSolver(params)
        solver.initialize(u0_type="gaussian")

        u_initial = solver.u.copy()
        F_initial = solver.F.copy()
        solver.step()

        self.assertFalse(np.array_equal(solver.u, u_initial))
        self.assertFalse(np.array_equal(solver.F, F_initial))

    def test_semi_implicit_step_updates_u_and_F(self):
        params = WesterveltParams(
            c=1500,
            beta=3.5,
            mu_v=1e-3,
            dx=1e-4,
            dt=1e-8,
            nx=128,
            nt=5,
            scheme="semi_implicit",
        )
        solver = WesterveltSolver(params)
        solver.initialize(u0_type="gaussian")

        u_initial = solver.u.copy()
        F_initial = solver.F.copy()
        solver.step()

        self.assertFalse(np.array_equal(solver.u, u_initial))
        self.assertFalse(np.array_equal(solver.F, F_initial))

    def test_explicit_and_semi_implicit_different(self):
        p_exp = WesterveltParams(c=1500, beta=12.0, mu_v=1.0e3, dx=1e-3, dt=2e-7, nx=128, nt=5, scheme="explicit")
        p_si = WesterveltParams(c=1500, beta=12.0, mu_v=1.0e3, dx=1e-3, dt=2e-7, nx=128, nt=5, scheme="semi_implicit")

        s_exp = WesterveltSolver(p_exp)
        s_si = WesterveltSolver(p_si)
        s_exp.initialize(u0_type="gaussian")
        s_si.initialize(u0_type="gaussian")

        for _ in range(5):
            s_exp.step()
            s_si.step()

        self.assertFalse(np.allclose(s_exp.u, s_si.u))

    def test_snapshots_and_energy(self):
        params = WesterveltParams(c=1500, dx=1e-3, dt=1e-7, nx=100, nt=20, scheme="semi_implicit")
        solver = WesterveltSolver(params)
        solver.initialize(u0_type="gaussian")

        snaps = solver.run_with_snapshots([0.0, 5e-7, 1e-6], store_energy=True)

        self.assertGreaterEqual(len(snaps), 2)
        self.assertEqual(len(solver.energy_history), params.nt + 1)
        self.assertTrue(np.all(np.isfinite(np.array(solver.energy_history))))

    def test_stability_scan(self):
        params = WesterveltParams(c=1500, dx=1e-3, dt=1e-7, nx=100, nt=5, scheme="semi_implicit")
        solver = WesterveltSolver(params)
        solver.initialize(u0_type="gaussian")

        results = solver.run_stability_scan([8e-8, 1e-7], [0.8, 1.2], blowup_threshold=1e4)
        self.assertEqual(len(results), 4)
        self.assertTrue(all("stable" in r for r in results))

    def test_invalid_scheme_raises(self):
        with self.assertRaises(ValueError):
            WesterveltParams(c=1500, scheme="quasilinear")

    def test_invalid_boundary_condition_raises(self):
        with self.assertRaises(ValueError):
            WesterveltParams(c=1500, bc="periodic")


if __name__ == "__main__":
    unittest.main()
