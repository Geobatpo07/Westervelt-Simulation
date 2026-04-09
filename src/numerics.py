# src/numerics.py

import numpy as np


def _laplacian(u, i, dx2):
    """Laplacien 1D centré au point i."""
    return (u[i + 1] - 2.0 * u[i] + u[i - 1]) / dx2


def _laplacian_all(u, dx2):
    """Laplacien centré sur tout le domaine (bords laissés à 0)."""
    nx = u.shape[0]
    out = np.zeros_like(u)
    for i in range(1, nx - 1):
        out[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / dx2
    return out


def _time_derivative(u, u_prev, dt):
    """D_t^- u^n."""
    return (u - u_prev) / dt


def _safe_denominator(denom, eps=1e-12):
    """Evite les divisions par 0 dans 1 - 2k u^n."""
    safe = denom.copy()
    mask = np.abs(safe) < eps
    safe[mask] = eps * np.where(safe[mask] >= 0.0, 1.0, -1.0)
    return safe


def _apply_boundary(u, bc_type):
    """
    Applique les conditions aux limites.

    bc_type:
        0: Dirichlet homogène (u=0)
        1: Neumann homogène (du/dx=0)
    """
    if bc_type == 0:
        u[0] = 0.0
        u[-1] = 0.0
    elif bc_type == 1:
        u[0] = u[1]
        u[-1] = u[-2]
    else:
        raise ValueError("bc_type doit valoir 0 (Dirichlet) ou 1 (Neumann).")


def update_F(F_n, u_n, u_prev, dt, dx, c, k, bc_type):
    """Mise à jour: F^{n+1} = F^n + dt(c^2 Δu^n + 2k(D_t^-u^n)^2)."""
    dx2 = dx * dx
    c2 = c * c
    lap_u = _laplacian_all(u_n, dx2)
    ut = _time_derivative(u_n, u_prev, dt)
    F_next = F_n + dt * (c2 * lap_u + 2.0 * k * ut ** 2)
    _apply_boundary(F_next, bc_type)
    return F_next


def assemble_semi_implicit_system(u_n, F_next, dt, dx, b, k, bc_type):
    """Assemble le système tri-diagonal du schéma semi-implicite pour u^{n+1}."""
    nx = u_n.shape[0]
    denom = _safe_denominator(1.0 - 2.0 * k * u_n)
    rhs_full = u_n + dt * F_next / denom
    lam_full = dt * b / (denom * dx ** 2)

    n_int = nx - 2
    if n_int <= 0:
        raise ValueError("nx doit être >= 3 pour le schéma semi-implicite.")

    lower = np.zeros(n_int - 1, dtype=u_n.dtype)
    diag = np.zeros(n_int, dtype=u_n.dtype)
    upper = np.zeros(n_int - 1, dtype=u_n.dtype)
    rhs = rhs_full[1:-1].copy()

    for j in range(n_int):
        i = j + 1
        lam = lam_full[i]

        if bc_type == 0: # Dirichlet
            diag[j] = 1.0 + 2.0 * lam
            if j > 0:
                lower[j - 1] = -lam
            if j < n_int - 1:
                upper[j] = -lam
        elif bc_type == 1: # Neumann
            if j == 0 or j == n_int - 1:
                diag[j] = 1.0 + lam
            else:
                diag[j] = 1.0 + 2.0 * lam
            if j > 0:
                lower[j - 1] = -lam
            if j < n_int - 1:
                upper[j] = -lam
        else:
            raise ValueError("bc_type doit valoir 0 (Dirichlet) ou 1 (Neumann).")

    return lower, diag, upper, rhs


def solve_tridiagonal(lower, diag, upper, rhs):
    """Solveur de Thomas pour matrice tri-diagonale."""
    n = diag.size
    c_prime = np.zeros(n - 1, dtype=diag.dtype)
    d_prime = np.zeros(n, dtype=rhs.dtype)

    piv = diag[0]
    if np.abs(piv) < 1e-14:
        raise ZeroDivisionError("Pivot nul dans solve_tridiagonal.")

    if n > 1:
        c_prime[0] = upper[0] / piv
    d_prime[0] = rhs[0] / piv

    for i in range(1, n):
        piv = diag[i] - lower[i - 1] * c_prime[i - 1]
        if np.abs(piv) < 1e-14:
            raise ZeroDivisionError("Pivot nul dans solve_tridiagonal.")
        if i < n - 1:
            c_prime[i] = upper[i] / piv
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / piv

    x = np.zeros(n, dtype=rhs.dtype)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def compute_energy(u, u_prev, c, dt, dx):
    """Energie discrète: 0.5 * int((u_t)^2 + c^2 (u_x)^2) dx."""
    ut = _time_derivative(u, u_prev, dt)

    ux = np.zeros_like(u)
    inv_2dx = 0.5 / dx
    for i in range(1, u.shape[0] - 1):
        ux[i] = (u[i + 1] - u[i - 1]) * inv_2dx

    return 0.5 * dx * np.sum(ut**2 + c**2 * ux**2)
