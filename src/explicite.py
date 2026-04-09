# src/explicite.py

from src.numerics import _apply_boundary, _laplacian_all, _safe_denominator, update_F


def step_explicit(u, u_prev, F, c, b, k, dt, dx, bc_type):
    """
    Schéma explicite:
      F^{n+1} = F^n + dt(c^2 Δu^n + 2k(D_t^-u^n)^2)
      u^{n+1} = u^n + dt(F^{n+1} + bΔu^n)/(1 - 2k u^n)
    """
    F_next = update_F(F, u, u_prev, dt, dx, c, k, bc_type)

    dx2 = dx * dx
    lap_u = _laplacian_all(u, dx2)
    denom = _safe_denominator(1.0 - 2.0 * k * u)
    u_next = u + dt * (F_next + b * lap_u) / denom

    _apply_boundary(u_next, bc_type)
    return u_next, F_next
