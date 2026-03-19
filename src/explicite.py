# src/explicite.py

import numpy as np
from numba import njit


# OPERATEUR SPATIAL
@njit
def _laplacian(u, i, dx2):
    """Laplacien 1D centré."""
    return (u[i+1] - 2*u[i] + u[i-1]) / dx2


# CALCUL u_tt
@njit
def _compute_utt(u_i, u_prev_i, lap, lap_prev, c, a, beta, dt, nonlinear):
    """
    Calcule u_tt selon l'équation de Westervelt
    """

    c2 = c**2

    # terme diffusion
    lap_ut = (lap - lap_prev) / dt

    utt = c2 * lap + a * lap_ut

    if nonlinear:
        denom = 1.0 - beta * u_i
        utt = utt / denom

    return utt


# CONDITIONS DE BORD
@njit
def _apply_boundary(u, bc_type):

    if bc_type == 0:  # Dirichlet
        u[0] = 0.0
        u[-1] = 0.0

    elif bc_type == 1:  # Neumann
        u[0] = u[1]
        u[-1] = u[-2]


# PAS DE TEMPS EXPLICITE
@njit
def step_explicit(u, u_prev, c, a, beta, dt, dx, nonlinear, bc_type):
    """
    Effectue un pas de temps explicite pour l'équation de Westervelt
    """

    nx = u.shape[0]

    u_next = np.zeros_like(u)

    dx2 = dx * dx
    dt2 = dt * dt

    for i in range(1, nx-1):

        lap = _laplacian(u, i, dx2)
        lap_prev = _laplacian(u_prev, i, dx2)

        utt = _compute_utt(
            u[i], u_prev[i],
            lap, lap_prev,
            c, a, beta, dt,
            nonlinear
        )

        u_next[i] = 2*u[i] - u_prev[i] + dt2 * utt

    _apply_boundary(u_next, bc_type)

    return u_next