"""
Tests unitaires pour le module de validation (cas fabriqué).
Vérifie la présence et le comportement des utilitaires de validation
pour la solution fabriquée (manufactured solution).
"""

import unittest
import numpy as np

from core.validation import (
    make_time_grid,
    initialize_manufactured_solver,
    make_manufactured_source,
    evaluate_exact_solution,
    run_manufactured_case,
    compute_manufatured_errors,
    convergence_study_manufactured,
    print_convergence_table,
)
from core.symbolics import build_numerics_function
from core.solver import WesterveltParams


class TestManufacturedValidation(unittest.TestCase):
    def setUp(self):
        self.funcs = build_numerics_function()
        self.A = 1.0
        self.L = 1.0
        self.omega = 2.0
        self.gamma = 0.5
        self.kappa = 0.1

    def test_make_time_grid_adjusts_dt(self):
        T = 1.0
        dt = 0.3
        times = make_time_grid(T, dt)
        # dt adjusted should divide T exactly
        self.assertAlmostEqual(times[-1], T, places=12)
        self.assertGreaterEqual(len(times), 2)

    def test_initialize_manufactured_solver_sets_fields(self):
        params = WesterveltParams(c=1500.0, dx=0.1, dt=0.01, nx=11, nt=10, bc='dirichlet')
        from core.solver import WesterveltSolver

        solver = WesterveltSolver(params)
        initialize_manufactured_solver(solver, self.funcs, self.A, self.L, self.omega, self.gamma, self.kappa)

        # shapes
        self.assertEqual(solver.u.shape[0], params.nx)
        self.assertEqual(solver.u_prev.shape[0], params.nx)

        # boundaries should respect Dirichlet (zero)
        self.assertAlmostEqual(solver.u[0], 0.0, places=12)
        self.assertAlmostEqual(solver.u[-1], 0.0, places=12)
        self.assertAlmostEqual(solver.u_prev[0], 0.0, places=12)
        self.assertAlmostEqual(solver.u_prev[-1], 0.0, places=12)

    def test_make_manufactured_source_callable(self):
        c = 1500.0
        b = 1e-9
        k = 1e-9
        src = make_manufactured_source(self.funcs, self.A, self.L, self.omega, self.gamma, self.kappa, c, b, k)
        x = np.linspace(0, 1, 11)
        val = src(x, 0.0)
        self.assertEqual(val.shape, x.shape)

    def test_run_manufactured_case_returns_shapes(self):
        # small case for speed
        params = WesterveltParams(c=1500.0, dx=0.05, dt=1e-5, nx=21, nt=10, bc='dirichlet', scheme='explicit')
        res = run_manufactured_case(params, self.funcs, self.A, self.L, self.omega, self.gamma, self.kappa, store_energy=False)

        self.assertIn('U_num', res)
        self.assertIn('U_ref', res)
        U_num = res['U_num']
        U_ref = res['U_ref']
        times = res['times']

        self.assertEqual(U_num.shape, (len(times), params.nx))
        self.assertEqual(U_ref.shape, (len(times), params.nx))

        # boundaries should be zero for Dirichlet
        self.assertTrue(np.allclose(U_ref[:, 0], 0.0))
        self.assertTrue(np.allclose(U_ref[:, -1], 0.0))

    def test_compute_manufatured_errors_returns_metrics(self):
        params = WesterveltParams(c=1500.0, dx=0.05, dt=1e-5, nx=21, nt=5, bc='dirichlet')
        res = run_manufactured_case(params, self.funcs, self.A, self.L, self.omega, self.gamma, self.kappa, store_energy=False)
        errs = compute_manufatured_errors(res['U_num'], res['U_ref'], params.dx, bc_type='dirichlet')
        # expect the keys to be present and numeric
        for key in ['Linf_L2', 'Linf_H1', 'Linf_grad', 'Linf_Linf']:
            self.assertIn(key, errs)
            self.assertIsInstance(errs[key], float)

    def test_convergence_study_manufactured_runs(self):
        # small levels & small base_nx for test speed
        results = convergence_study_manufactured(self.funcs, levels=[0, 1], L=1.0, T=1e-5, base_nx=10, cfl_factor=0.2)
        # check structure
        self.assertIn('errors_L2', results)
        self.assertIn('orders_L2', results)
        self.assertIn('mesh_sizes', results)
        # keys for levels
        self.assertTrue(0 in results['errors_L2'])
        self.assertTrue(1 in results['errors_L2'])

    def test_convergence_errors_decrease(self):
        results = convergence_study_manufactured(
            self.funcs,
            levels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            L=1.0,
            T=1e-5,
            base_nx=10,
            cfl_factor=0.2,
            scheme="explicit",
        )

        self.assertLess(results["errors_L2"][1], results["errors_L2"][0])
        self.assertLess(results["errors_L2"][2], results["errors_L2"][1])


if __name__ == '__main__':
    unittest.main()

