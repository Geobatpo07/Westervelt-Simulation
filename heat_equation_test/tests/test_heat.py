import unittest
import numpy as np
import sys
import os

# Ajouter le chemin vers core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from heat_solver import solve_heat_explicit, solve_heat_implicit

class TestHeatSolver(unittest.TestCase):
    def test_explicit_heat(self):
        nx = 50
        u0 = np.zeros(nx)
        u0[nx//2] = 1.0
        alpha = 0.01
        dx = 0.1
        dt = 0.001
        nt = 10
        u_final = solve_heat_explicit(u0, alpha, dx, dt, nt)
        self.assertEqual(len(u_final), nx)
        self.assertTrue(np.sum(u_final) > 0)

    def test_implicit_heat(self):
        nx = 50
        u0 = np.zeros(nx)
        u0[nx//2] = 1.0
        alpha = 0.01
        dx = 0.1
        dt = 0.001
        nt = 10
        u_final = solve_heat_implicit(u0, alpha, dx, dt, nt)
        self.assertEqual(len(u_final), nx)
        self.assertTrue(np.sum(u_final) > 0)

if __name__ == '__main__':
    unittest.main()
