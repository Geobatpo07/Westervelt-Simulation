"""Kernels numeriques in-place pour les simulations temps reel.

Les fonctions restent utilisables sans Numba: le decorateur devient alors un
no-op, ce qui garde l'app Streamlit fonctionnelle dans un environnement leger.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import get_num_threads, njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - depend de l'environnement local
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def get_num_threads():  # type: ignore[no-redef]
        return 1


NUMBA_PARALLEL_ENABLED = False
SAFE_DENOMINATOR_EPS = 1.0e-12
PIVOT_EPS = 1.0e-14


def allocate_semi_implicit_workspace(nx: int, dtype=np.float64):
    """
    Alloue les tableaux temporaires requis par le solveur de Thomas.

    Args:
        nx: Nombre de points spatiaux.
        dtype: Type de données.

    Returns:
        Tuple: (lower, diag, upper, rhs, c_prime, d_prime).
    """
    n_int = max(int(nx) - 2, 0)
    n_offdiag = max(n_int - 1, 0)
    return (
        np.empty(n_offdiag, dtype=dtype),
        np.empty(n_int, dtype=dtype),
        np.empty(n_offdiag, dtype=dtype),
        np.empty(n_int, dtype=dtype),
        np.empty(n_offdiag, dtype=dtype),
        np.empty(n_int, dtype=dtype),
    )


@njit(cache=True)
def _safe_scalar_denominator(value):
    if abs(value) < SAFE_DENOMINATOR_EPS:
        if value >= 0.0:
            return SAFE_DENOMINATOR_EPS
        return -SAFE_DENOMINATOR_EPS
    return value


@njit(cache=True)
def apply_boundary_inplace(values, bc_type):
    """
    Applique une condition de bord au tableau fourni (Dirichlet ou Neumann).

    Args:
        values: Tableau de données modifié in-place.
        bc_type: 0 pour Dirichlet, 1 pour Neumann.
    """
    if bc_type == 0:
        values[0] = 0.0
        values[values.shape[0] - 1] = 0.0
    elif bc_type == 1:
        values[0] = values[1]
        values[values.shape[0] - 1] = values[values.shape[0] - 2]
    else:
        raise ValueError("bc_type doit valoir 0 (Dirichlet) ou 1 (Neumann).")


@njit(cache=True, parallel=NUMBA_PARALLEL_ENABLED, nogil=True)
def update_F_inplace(F, u, F_next, c, dt, dx, bc_type):
    """
    Calcule F^{n+1} dans F_next à partir de F^n et u^n.

    Args:
        F, u: Tableaux à l'instant n.
        F_next: Tableau de destination.
        c, dt, dx: Paramètres numériques.
        bc_type: Type de conditions aux limites.
    """
    nx = u.shape[0]
    inv_dx2 = 1.0 / (dx ** 2)
    c2 = c ** 2

    for i in prange(1, nx - 1):
        lap_u = (u[i + 1] - 2.0 * u[i] + u[i - 1]) * inv_dx2
        F_next[i] = F[i] + dt * c2 * lap_u

    apply_boundary_inplace(F_next, bc_type)


@njit(cache=True, parallel=NUMBA_PARALLEL_ENABLED, nogil=True)
def step_explicit_inplace(u, F, u_next, F_next, c, b, k, dt, dx, bc_type):
    """
    Effectue un pas explicite dans les buffers fournis.

    Args:
        u, F: Champs à l'instant n.
        u_next, F_next: Champs à l'instant n+1 (modifiés in-place).
        c, b, k, dt, dx: Paramètres physiques et numériques.
        bc_type: Type de conditions aux limites.
    """
    nx = u.shape[0]
    inv_dx2 = 1.0 / (dx ** 2)
    c2 = c ** 2

    for i in prange(1, nx - 1):
        lap_u = (u[i + 1] - 2.0 * u[i] + u[i - 1]) * inv_dx2
        F_next[i] = F[i] + dt * c2 * lap_u
        denom = _safe_scalar_denominator(1.0 - 2.0 * k * u[i])
        u_next[i] = u[i] + dt * (F_next[i] + b * lap_u) / denom

    apply_boundary_inplace(F_next, bc_type)
    apply_boundary_inplace(u_next, bc_type)


@njit(cache=True, parallel=NUMBA_PARALLEL_ENABLED, nogil=True)
def step_semi_implicit_inplace(
    u,
    F,
    u_next,
    F_next,
    c,
    b,
    k,
    dt,
    dx,
    bc_type,
    lower,
    diag,
    upper,
    rhs,
    c_prime,
    d_prime,
):
    """
    Effectue un pas semi-implicite avec solveur de Thomas in-place.

    Args:
        u, F: Champs à l'instant n.
        u_next, F_next: Champs à l'instant n+1.
        c, b, k, dt, dx: Paramètres physiques et numériques.
        bc_type: Type de conditions aux limites.
        lower, diag, upper: Vecteurs de la matrice tridiagonale.
        rhs: Second membre.
        c_prime, d_prime: Vecteurs auxiliaires pour Thomas.
    """
    nx = u.shape[0]
    n_int = nx - 2
    if n_int <= 0:
        raise ValueError("nx doit etre >= 3 pour le schema semi-implicite.")

    inv_dx2 = 1.0 / (dx * dx)
    c2 = c * c

    for j in prange(n_int):
        i = j + 1
        lap_u = (u[i + 1] - 2.0 * u[i] + u[i - 1]) * inv_dx2
        F_next[i] = F[i] + dt * c2 * lap_u

        denom = _safe_scalar_denominator(1.0 - 2.0 * k * u[i])
        lam = dt * b * inv_dx2 / denom
        rhs[j] = u[i] + dt * F_next[i] / denom

        if bc_type == 1 and (j == 0 or j == n_int - 1):
            diag[j] = 1.0 + lam
        else:
            diag[j] = 1.0 + 2.0 * lam

        if j > 0:
            lower[j - 1] = -lam
        if j < n_int - 1:
            upper[j] = -lam

    apply_boundary_inplace(F_next, bc_type)

    piv = diag[0]
    if abs(piv) < PIVOT_EPS:
        raise ZeroDivisionError("Pivot nul dans solve_tridiagonal.")

    if n_int > 1:
        c_prime[0] = upper[0] / piv
    d_prime[0] = rhs[0] / piv

    for i in range(1, n_int):
        piv = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(piv) < PIVOT_EPS:
            raise ZeroDivisionError("Pivot nul dans solve_tridiagonal.")
        if i < n_int - 1:
            c_prime[i] = upper[i] / piv
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / piv

    u_next[nx - 2] = d_prime[n_int - 1]
    for j in range(n_int - 2, -1, -1):
        u_next[j + 1] = d_prime[j] - c_prime[j] * u_next[j + 2]

    apply_boundary_inplace(u_next, bc_type)


@njit(cache=True, parallel=NUMBA_PARALLEL_ENABLED, nogil=True)
def compute_energy_numba(u, u_prev, c, dt, dx):
    """
    Calcule l'énergie discrète via Numba.

    Args:
        u, u_prev: Champs u^n et u^{n-1}.
        c, dt, dx: Paramètres.

    Returns:
        float: Énergie totale calculée.
    """
    nx = u.shape[0]
    inv_dt = 1.0 / dt
    inv_2dx = 1.0 / (2.0 * dx)
    c2 = c * c
    total = 0.0

    for i in prange(nx):
        ut = (u[i] - u_prev[i]) * inv_dt
        ux = 0.0
        if 0 < i < nx - 1:
            ux = (u[i + 1] - u[i - 1]) * inv_2dx
        total += ut ** 2 + c2 * ux ** 2

    return 0.5 * dx * total


def numba_thread_count() -> int:
    """
    Retourne le nombre de threads utilisés par les kernels Numba.

    Returns:
        int: Nombre de threads.
    """
    return int(get_num_threads()) if NUMBA_AVAILABLE and NUMBA_PARALLEL_ENABLED else 1


def warm_up_numba() -> None:
    """
    Déclenche la compilation JIT sur un petit cas test pour éviter les délais au runtime.
    """
    if not NUMBA_AVAILABLE:
        return

    nx = 8
    u = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    u[0] = 0.0
    u[-1] = 0.0
    u_prev = u.copy()
    F = np.zeros(nx, dtype=np.float64)
    u_next = np.empty_like(u)
    F_next = np.empty_like(F)
    c = 1500.0
    b = 1.0e-9
    k = 1.0e-9
    dt = 1.0e-8
    dx = 1.0e-4
    bc_type = 0

    step_explicit_inplace(u, F, u_next, F_next, c, b, k, dt, dx, bc_type)
    workspace = allocate_semi_implicit_workspace(nx)
    step_semi_implicit_inplace(u, F, u_next, F_next, c, b, k, dt, dx, bc_type, *workspace)
    compute_energy_numba(u, u_prev, c, dt, dx)