# ./core/symbolics.py

"""
Module de calcul symbolique pour la génération de solutions fabriquées.

Ce module utilise SymPy pour construire des solutions analytiques de l'équation de Westervelt
et générer les termes sources correspondants, ainsi que leurs versions numérisées (lambdifiées).
"""

import sympy as sp

def build_manufactured_solution(bc_type: str = 'dirichlet'):
    """
    Construit une solution fabriquée symbolique pour l'équation de Westervelt.

    Définit u(t, x) selon le type de conditions aux limites et calcule
    ses dérivées ainsi que le terme source f associé.

    Parameters
    ----------
    bc_type : str
        Type de conditions aux limites ('dirichlet', 'neumann', ou 'periodic').
        Par défaut 'dirichlet'.

    Returns
    -------
    dict
        Dictionnaire contenant :
        - 'u' : Expression SymPy de la solution.
        - 'ut', 'utt', 'uxx', 'uxxt' : Expressions des dérivées.
        - 'f' : Expression du terme source simplifié.
        - 'parametres' : Tuple des symboles utilisés (x, t, A, L, omega, gamma, kappa, c, b, k).

    Raises
    ------
    ValueError
        Si le `bc_type` n'est pas reconnu.
    """
    # Variables
    x, t = sp.symbols('x t')

    # Paramètres
    c, b, k = sp.symbols('c b k')
    A, L, omega, gamma, kappa = sp.symbols('A L omega gamma kappa')

    # Solution fabriquée
    if bc_type == 'dirichlet':
        spatial = sp.sin(sp.pi * x / L)
    elif bc_type == 'neumann':
        spatial = sp.cos(sp.pi * x / L)
    elif bc_type == 'periodic':
        spatial = sp.sin(2 * sp.pi * x / L)
    else:
        raise ValueError(f"Type de condition de borne {bc_type} non reconnu.")

    u = A * spatial * sp.exp(-kappa * t) * (sp.cos(omega * t) + gamma * sp.sin(omega * t))

    # Dérivées
    ut = sp.diff(u, t)
    utt = sp.diff(u, t, 2)
    uxx = sp.diff(u, x, 2)
    uxxt = sp.diff(uxx, t)

    # Terme source
    f = (1- 2 * k * u) * utt - c ** 2 * uxx - b * uxxt - 2 * k * ut ** 2

    return {
        'u': u,
        'ut': ut,
        'utt': utt,
        'uxx': uxx,
        'uxxt': uxxt,
        'f': sp.simplify(f),
        'parametres': (x, t, A, L, omega, gamma, kappa, c, b, k),
    }


def build_numerics_function(bc_type='dirichlet'):
    """
    Génère des fonctions numériques (NumPy) à partir des solutions symboliques.

    Convertit les expressions SymPy en fonctions exécutables acceptant des tableaux NumPy
    pour une évaluation efficace lors des simulations.

    Parameters
    ----------
    bc_type : str, optional
        Type de conditions aux limites pour la solution fabriquée. Par défaut 'dirichlet'.

    Returns
    -------
    dict
        Dictionnaire où chaque clé est un nom de terme ('u', 'ut', 'utt', 'uxx', 'uxxt', 'f')
        et chaque valeur est une fonction lambdifiée correspondante.
    """
    data = build_manufactured_solution(bc_type=bc_type)

    DERIV_TERMES = ['u', 'ut', 'utt', 'uxx', 'uxxt', 'f',]

    x, t, A, L, omega, gamma, kappa, c, b, k = data['parametres']

    funcs = {}

    for key in DERIV_TERMES:
        funcs[key] = sp.lambdify(
            (x, t, A, L, omega, gamma, kappa, c, b, k),
            data[key],
            'numpy'
        )

    funcs['bc_type'] = bc_type

    return funcs