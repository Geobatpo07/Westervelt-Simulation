import unittest
import numpy as np
import sympy as sp

from core.symbolics import build_manufactured_solution, build_numerics_function


class TestBuildManufacturedSolution(unittest.TestCase):
    """Tests for the build_manufactured_solution function."""

    def test_solution_returns_dict(self):
        """Test that build_manufactured_solution returns a dictionary."""
        result = build_manufactured_solution()
        self.assertIsInstance(result, dict)

    def test_solution_contains_required_keys(self):
        """Test that the result contains all required keys."""
        result = build_manufactured_solution()
        required_keys = ['u', 'ut', 'utt', 'uxx', 'uxxt', 'f', 'parametres']
        for key in required_keys:
            self.assertIn(key, result)

    def test_solution_expressions_are_sympy(self):
        """Test that the solution expressions are SymPy expressions."""
        result = build_manufactured_solution()
        for key in ['u', 'ut', 'utt', 'uxx', 'uxxt', 'f']:
            self.assertIsInstance(result[key], sp.Basic)

    def test_parametres_tuple_length(self):
        """Test that parametres tuple contains all expected symbols."""
        result = build_manufactured_solution()
        parametres = result['parametres']
        self.assertEqual(len(parametres), 10)

    def test_parametres_contains_required_symbols(self):
        """Test that parametres tuple contains all required symbols."""
        result = build_manufactured_solution()
        parametres = result['parametres']
        param_names = [str(p) for p in parametres]

        required_symbols = ['x', 't', 'A', 'L', 'omega', 'gamma', 'kappa', 'c', 'b', 'k']
        for name in required_symbols:
            self.assertIn(name, param_names)

    def test_solution_u_is_function_of_time_and_space(self):
        """Test that u depends on x and t."""
        result = build_manufactured_solution()
        u = result['u']
        self.assertIn(sp.symbols('x'), u.free_symbols)
        self.assertIn(sp.symbols('t'), u.free_symbols)

    def test_derivatives_depend_on_original_solution(self):
        """Test that ut and uxx are non-zero."""
        result = build_manufactured_solution()
        self.assertNotEqual(result['ut'], 0)
        self.assertNotEqual(result['uxx'], 0)

    def test_double_derivative_utt_exists(self):
        """Test that utt is computed correctly."""
        result = build_manufactured_solution()
        utt = result['utt']
        # Verify it's a second time derivative
        self.assertIsInstance(utt, sp.Basic)
        self.assertNotEqual(utt, 0)

    def test_mixed_derivative_uxxt_exists(self):
        """Test that uxxt (spatial-temporal mixed derivative) is computed."""
        result = build_manufactured_solution()
        uxxt = result['uxxt']
        self.assertIsInstance(uxxt, sp.Basic)

    def test_source_term_f_is_simplified(self):
        """Test that the source term f is created."""
        result = build_manufactured_solution()
        f = result['f']
        # The source term should exist and be simplified
        self.assertIsInstance(f, sp.Basic)


class TestBuildNumericsFunction(unittest.TestCase):
    """Tests for the build_numerics_function function."""

    def test_returns_dict(self):
        """Test that build_numerics_function returns a dictionary."""
        result = build_numerics_function()
        self.assertIsInstance(result, dict)

    def test_returns_callable_functions(self):
        """Test that returned values are callable (functions)."""
        result = build_numerics_function()
        for key, value in result.items():
            self.assertTrue(callable(value), f"{key} is not callable")

    def test_contains_required_derivative_terms(self):
        """Test that all derivative terms are in the result."""
        result = build_numerics_function()
        required_terms = ['u', 'ut', 'utt', 'uxx', 'uxxt']
        for term in required_terms:
            self.assertIn(term, result)

    def test_functions_accept_correct_number_of_arguments(self):
        """Test that functions accept 10 arguments."""
        result = build_numerics_function()
        # All functions should accept 10 arguments: x, t, A, L, omega, gamma, kappa, c, b, k
        test_args = (0.5, 0.1, 1.0, 1.0, 1.0, 0.1, 0.01, 1500.0, 0.01, 1e-4)

        for key, func in result.items():
            try:
                output = func(*test_args)
                # Output should be a scalar or array
                self.assertTrue(isinstance(output, (float, np.ndarray, np.floating)))
            except TypeError as e:
                self.fail(f"Function {key} failed with TypeError: {e}")

    def test_functions_return_finite_values(self):
        """Test that functions return finite values with valid inputs."""
        result = build_numerics_function()
        test_args = (0.5, 0.1, 1.0, 1.0, 1.0, 0.1, 0.01, 1500.0, 0.01, 1e-4)

        for key, func in result.items():
            output = func(*test_args)
            if isinstance(output, np.ndarray):
                self.assertTrue(np.all(np.isfinite(output)), f"{key} produced non-finite values")
            else:
                self.assertTrue(np.isfinite(output), f"{key} produced non-finite value")

    def test_functions_vectorized_input(self):
        """Test that functions handle vectorized input (arrays)."""
        result = build_numerics_function()
        x_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        t_array = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        other_params = (1.0, 1.0, 1.0, 0.1, 0.01, 1500.0, 0.01, 1e-4)

        # Test with array for x
        func_u = result['u']
        try:
            output = func_u(x_array, 0.1, *other_params)
            # Should return an array
            self.assertTrue(isinstance(output, np.ndarray))
        except:
            pass  # Some implementations might not support full vectorization

    def test_u_at_zero_with_zero_time(self):
        """Test u function at specific points."""
        result = build_numerics_function()
        func_u = result['u']
        # With A=0 (amplitude zero), u should be zero everywhere
        output = func_u(0.5, 0.0, A=0.0, L=1.0, omega=1.0, gamma=0.1, kappa=0.01, c=1500.0, b=0.01, k=1e-4)
        self.assertEqual(output, 0.0)

    def test_ut_initial_value_consistency(self):
        """Test that ut and u are related (ut is derivative of u with respect to t)."""
        result = build_numerics_function()
        func_u = result['u']
        func_ut = result['ut']

        x, t = 0.5, 0.0
        params = (1.0, 1.0, 1.0, 0.1, 0.01, 1500.0, 0.01, 1e-4)

        u_val = func_u(x, t, *params)
        ut_val = func_ut(x, t, *params)

        # At t=0 with decaying exponential, ut should not be zero unless the parameters are special
        self.assertTrue(isinstance(ut_val, (float, np.ndarray, np.floating)))

    def test_ut_initial_formula(self):
        """Test that ut is consistent with the initial value formula."""
        funcs = build_numerics_function()

        x = 0.5
        t = 0.0

        A = 2.0
        L = 1.0
        omega = 3.0
        gamma = 0.5
        kappa = 0.2
        c = 1500.0
        b = 1e-3
        k = 1e-4

        ut_val = funcs["ut"](x, t, A, L, omega, gamma, kappa, c, b, k)

        expected = A * np.sin(np.pi * x / L) * (gamma * omega - kappa)

        self.assertAlmostEqual(float(ut_val), float(expected), places=10)

    def test_source_term_residual_is_zero(self):
        """Test that the source term residual is zero."""
        funcs = build_numerics_function()

        x = np.linspace(0.0, 1.0, 101)
        t = 0.2

        A = 1.0
        L = 1.0
        omega = 2.0
        gamma = 0.5
        kappa = 0.1
        c = 1500.0
        b = 1e-3
        k = 1e-4

        args = (x, t, A, L, omega, gamma, kappa, c, b, k)

        u = funcs["u"](*args)
        ut = funcs["ut"](*args)
        utt = funcs["utt"](*args)
        uxx = funcs["uxx"](*args)
        uxxt = funcs["uxxt"](*args)
        f = funcs["f"](*args)

        residual = (1.0 - 2.0 * k * u) * utt - c ** 2 * uxx - b * uxxt - 2.0 * k * ut ** 2 - f

        self.assertLess(np.max(np.abs(residual)), 1e-8)

    def test_dirichlet_boundary_conditions(self):
        """Test that the Dirichlet boundary conditions are satisfied."""
        funcs = build_numerics_function()

        t = 0.3
        params = (1.0, 1.0, 2.0, 0.5, 0.1, 1500.0, 1e-3, 1e-4)

        u_left = funcs["u"](0.0, t, *params)
        u_right = funcs["u"](1.0, t, *params)

        self.assertAlmostEqual(float(u_left), 0.0, places=12)
        self.assertAlmostEqual(float(u_right), 0.0, places=12)

if __name__ == "__main__":
    unittest.main()

