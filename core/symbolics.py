# ./core/symbolics.py
# ici les symboles pour l'implementation d'une solution analytique

import sympy as sp

def build_manufactured_solution():
    # Variables
    x, t = sp.symbols('x t')

    # Paramètres
    c, b, k = sp.symbols('c b k')
    A, L, omega, gamma, kappa = sp.symbols('A L omega gamma kappa')

    # Solution fabriquée
    u = A * sp.sin(sp.pi * x / L) * sp.exp(-kappa * t) * (sp.cos(omega * t) + gamma * sp.sin(omega * t))

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


def build_numerics_function():
    data = build_manufactured_solution()

    DERIV_TERMES = ['u', 'ut', 'utt', 'uxx', 'uxxt', 'f']

    x, t, A, L, omega, gamma, kappa, c, b, k = data['parametres']

    funcs = {}

    for key in DERIV_TERMES:
        funcs[key] = sp.lambdify(
            (x, t, A, L, omega, gamma, kappa, c, b, k),
            data[key],
            'numpy'
        )

    return funcs