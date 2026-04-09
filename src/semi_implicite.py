# src/semi_implicite.py

import numpy as np
from src.numerics import (
    _apply_boundary,
    assemble_semi_implicit_system,
    solve_tridiagonal,
    update_F,
)


def step_semi_implicit(u, u_prev, F, c, b, k, dt, dx, bc_type):
    """
    Schéma semi-implicite:
      F^{n+1} = F^n + dt(c^2 Δu^n + 2k(D_t^-u^n)^2)
      u^{n+1} = u^n + dt(F^{n+1} + bΔu^{n+1})/(1 - 2k u^n)
    """
    F_next = update_F(F, u, u_prev, dt, dx, c, k, bc_type)

    lower, diag, upper, rhs = assemble_semi_implicit_system(u, F_next, dt, dx, b, k, bc_type)
    interior = solve_tridiagonal(lower, diag, upper, rhs)

    u_next = np.zeros_like(u)
    u_next[1:-1] = interior
    _apply_boundary(u_next, bc_type)
    return u_next, F_next
