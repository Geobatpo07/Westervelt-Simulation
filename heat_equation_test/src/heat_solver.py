import numpy as np
from numba import njit, prange
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

@njit(parallel=True)
def solve_heat_explicit(u0, alpha, dx, dt, nt):
    """
    Résout l'équation de la chaleur 1D en utilisant un schéma explicite d'Euler.
    
    u_t = alpha * u_xx
    
    Parameters:
    u0 : array_like
        Condition initiale.
    alpha : float
        Coefficient de diffusivité thermique.
    dx : float
        Pas d'espace.
    dt : float
        Pas de temps.
    nt : int
        Nombre de pas de temps.
        
    Returns:
    u : ndarray
        La solution finale après nt pas de temps.
    """
    u = u0.copy()
    n = len(u)
    r = alpha * dt / dx**2
    
    if r > 0.5:
        # Note: Numba ne supporte pas les warnings standard facilement, 
        # mais on peut imprimer un message (bien que déconseillé en njit pur parfois)
        pass

    for _ in prange(nt):
        un = u.copy()
        for i in prange(1, n - 1):
            u[i] = un[i] + r * (un[i+1] - 2*un[i] + un[i-1])
        # Conditions aux limites (Dirichlet ici, maintenues par u0 si on ne les change pas)
    return u

def solve_heat_implicit(u0, alpha, dx, dt, nt):
    """
    Résout l'équation de la chaleur 1D en utilisant un schéma implicite (Crank-Nicolson).
    
    Parameters:
    u0 : array_like
        Condition initiale.
    alpha : float
        Coefficient de diffusivité thermique.
    dx : float
        Pas d'espace.
    dt : float
        Pas de temps.
    nt : int
        Nombre de pas de temps.
        
    Returns:
    u : ndarray
        La solution finale après nt pas de temps.
    """
    n = len(u0)
    r = alpha * dt / (2 * dx**2)
    
    # Construction de la matrice pour Crank-Nicolson: (I - r*D) u^{n+1} = (I + r*D) u^n
    # D est l'opérateur de dérivée seconde
    
    main_diag = np.full(n, 1 + 2*r)
    off_diag = np.full(n-1, -r)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsr()
    
    # Conditions aux limites de Dirichlet (on modifie la matrice pour u[0] et u[n-1])
    # Pour simplifier, on garde les bords constants
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[-1, -1] = 1.0
    A[-1, -2] = 0.0
    
    main_diag_b = np.full(n, 1 - 2*r)
    off_diag_b = np.full(n-1, r)
    B = diags([off_diag_b, main_diag_b, off_diag_b], [-1, 0, 1]).tocsr()
    B[0, 0] = 1.0
    B[0, 1] = 0.0
    B[-1, -1] = 1.0
    B[-1, -2] = 0.0

    u = u0.copy()
    for _ in prange(nt):
        b = B.dot(u)
        u = spsolve(A, b)
        
    return u
