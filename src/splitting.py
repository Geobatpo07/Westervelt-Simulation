# src/splitting.py

import numpy as np
from src.numerics import _laplacian, _apply_boundary


# OPERATEUR ONDE

def step_wave(u, u_prev, c, dt, dx, bc_type):

    nx = u.shape[0]

    u_next = np.zeros_like(u)

    dx2 = dx*dx
    dt2 = dt*dt
    c2 = c*c

    for i in range(1, nx-1):

        lap = _laplacian(u, i, dx2)

        u_next[i] = 2*u[i] - u_prev[i] + dt2*c2*lap

    _apply_boundary(u_next, bc_type)

    return u_next


# OPERATEUR DISSIPATION

def step_diffusion(u, u_prev, a, dt, dx, bc_type):

    nx = u.shape[0]

    u_next = np.zeros_like(u)

    dx2 = dx*dx
    dt2 = dt*dt

    for i in range(1, nx-1):

        lap = _laplacian(u, i, dx2)
        lap_prev = _laplacian(u_prev, i, dx2)

        lap_ut = (lap - lap_prev) / dt

        u_next[i] = 2*u[i] - u_prev[i] + dt2*a*lap_ut

    _apply_boundary(u_next, bc_type)

    return u_next


# OPERATEUR NON LINEAIRE

def step_nonlinear(u, u_prev, beta, dt, bc_type):

    nx = u.shape[0]

    u_next = np.zeros_like(u)

    dt2 = dt*dt

    for i in range(1, nx-1):

        ut = (u[i] - u_prev[i]) / dt

        denom = 1.0 - 2.0*beta*u[i]

        utt = (2.0*beta*ut*ut) / denom

        u_next[i] = 2*u[i] - u_prev[i] + dt2*utt

    _apply_boundary(u_next, bc_type)

    return u_next


# STRANG SPLITTING COMPLET

def step_splitting(u, u_prev, c, a, beta, dt, dx, bc_type):
    """
    Un pas de temps avec Strang splitting.

    1) demi onde
    2) demi diffusion
    3) non linéaire
    4) demi diffusion
    5) demi onde
    """

    dt_half = 0.5 * dt

    # 1) demi onde
    u1 = step_wave(u, u_prev, c, dt_half, dx, bc_type)

    # 2) demi diffusion
    u2 = step_diffusion(u1, u, a, dt_half, dx, bc_type)

    # 3) non linéaire
    u3 = step_nonlinear(u2, u1, beta, dt, bc_type)

    # 4) demi diffusion
    u4 = step_diffusion(u3, u2, a, dt_half, dx, bc_type)

    # 5) demi onde
    u_next = step_wave(u4, u3, c, dt_half, dx, bc_type)

    return u_next