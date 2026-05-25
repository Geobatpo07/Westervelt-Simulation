# /core/stability_analysis.py

import numpy as np

def discrete_mu(theta, dx):
    """
    Calcule la valeur discrète de mu pour une onde de nombre d'onde theta et un pas spatial dx.

    Args:
        theta (float or np.ndarray): Nombre d'onde ou vecteur de nombres d'onde.
        dx (float): Pas d'espace.

    Returns:
        float or np.ndarray: Valeur discrète de mu.
    """
    theta = np.asarray(theta)
    return 4.0 * np.sin(theta / 2) ** 2 / dx ** 2


def amplification_matrix_explicite(mu, dt, c, b):
    """
    Calcule la matrice d'amplification pour le schéma explicite linéarisé (k = 0).

    Args:
        mu (float): Valeur discrète de mu.
        dt (float): Pas de temps.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.

    Returns:
        np.ndarray: Matrice d'amplification 2x2.
    """
    return np.array(
        [
            [1.0 - dt ** 2 * c ** 2 * mu - dt * b * mu, dt],
            [- dt * c ** 2 * mu, 1.0],
        ],
        dtype=float,
    )


def amplification_matrix_semi_implicite(mu, dt, c, b, alpha=1.0):
    """
    Calcule la matrice d'amplification pour le schéma semi-implicite linéarisé par un gel de alpha.

    Args:
        mu (float): Valeur discrète de mu.
        dt (float): Pas de temps.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.
        alpha (float): Paramètre de gel pour la linéarisation (défaut: 1.0).

    Returns:
        np.ndarray: Matrice d'amplification 2x2.
    """
    beta = b * dt / alpha
    gamma = c ** 2 * dt ** 2 / alpha
    tau = dt / alpha
    return np.array(
        [
            [(1.0 - gamma * mu) / (1 + beta * mu), tau / (1 + beta * mu)],
            [- dt * c ** 2 * mu, 1.0],
        ],
        dtype=float,
    )


def eigenvalues_amplification(A):
    """
    Calcule les valeurs propres de la matrice d'amplification A.

    Args:
        A (np.ndarray): Matrice d'amplification.

    Returns:
        np.ndarray: Valeurs propres de la matrice.
    """
    return np.linalg.eigvals(A)


def spectral_radius(A):
    """
    Calcule le rayon spectral de la matrice A.

    Args:
        A (np.ndarray): Matrice carrée.

    Returns:
        float: Rayon spectral (maximum des modules des valeurs propres).
    """
    eigenvalues = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigenvalues)))


def explicit_stability_margin(dt, dx, c, b):
    """
    Calcule la marge de stabilité pour le schéma explicite linéarisé (k = 0).

    Args:
        dt (float): Pas de temps.
        dx (float): Pas d'espace.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.

    Returns:
        float: Valeur de la marge de stabilité (doit être >= 0 pour la stabilité).
    """
    return dx ** 2 - (c ** 2 * dt ** 2 + 2 * b * dt)


def explicit_theoretical_stable(dt, dx, c, b):
    """
    Vérifie si la condition de stabilité théorique pour le schéma explicite linéarisé (k = 0) est respectée.

    Args:
        dt (float): Pas de temps.
        dx (float): Pas d'espace.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.

    Returns:
        bool: True si le schéma est théoriquement stable.
    """
    return explicit_stability_margin(dt, dx, c, b) >= 0.0


def critical_dt_explicit(dx, c, b):
    return (-b + np.sqrt(b ** 2 + c ** 2 * dx ** 2)) / c ** 2


def semi_implicit_stability_margin(dt, dx, c, b, alpha=1.0):
    """
    Calcule la marge de stabilité pour le schéma semi-implicite linéarisé par un gel de alpha.

    Args:
        dt (float): Pas de temps.
        dx (float): Pas d'espace.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.
        alpha (float): Paramètre de gel (défaut: 1.0).

    Returns:
        float: Valeur de la marge de stabilité (doit être >= 0 pour la stabilité).
    """
    return alpha * dx ** 2 - (c ** 2 * dt ** 2 - 2 * b * dt)


def semi_implicit_theoretical_stable(dt, dx, c, b, alpha=1.0):
    """
    Vérifie si la condition de stabilité théorique pour le schéma semi-implicite linéarisé par un gel de alpha.

    Args:
        dt (float): Pas de temps.
        dx (float): Pas d'espace.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.
        alpha (float): Paramètre de gel (défaut: 1.0).

    Returns:
        bool: True si le schéma est théoriquement stable.
    """
    return semi_implicit_stability_margin(dt, dx, c, b, alpha) >= 0.0


def scan_spectral_radius_explicit(dt, dx, c, b, ntheta=500):
    """
    Scanne le rayon spectral de la matrice d'amplification du schéma explicite linéarisé.

    Args:
        dt (float): Pas de temps.
        dx (float): Pas d'espace.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.
        ntheta (int): Nombre de points pour la discrétisation du nombre d'onde (défaut: 500).

    Returns:
        dict: Dictionnaire contenant les vecteurs theta, mu, rho et la valeur rho_max.
    """
    thetas = np.linspace(0, np.pi, ntheta)
    mus = discrete_mu(thetas, dx)
    radii = np.empty_like(mus)

    for i, mu in enumerate(mus):
        A = amplification_matrix_explicite(mu, dt, c, b)
        radii[i] = spectral_radius(A)

    return {
        "theta": thetas,
        "mu": mus,
        "rho": radii,
        "rho_max": float(np.max(radii)),
    }


def scan_spectral_radius_semi_implicit(dt, dx, c, b, alpha=1.0, ntheta=500):
    """
    Scanne le rayon spectral de la matrice d'amplification du schéma semi-implicite linéarisé.

    Args:
        dt (float): Pas de temps.
        dx (float): Pas d'espace.
        c (float): Vitesse de l'onde.
        b (float): Paramètre de diffusion.
        alpha (float): Paramètre de gel (défaut: 1.0).
        ntheta (int): Nombre de points pour la discrétisation du nombre d'onde (défaut: 500).

    Returns:
        dict: Dictionnaire contenant les vecteurs theta, mu, rho et la valeur rho_max.
    """
    thetas = np.linspace(0, np.pi, ntheta)
    mus = discrete_mu(thetas, dx)
    radii = np.empty_like(mus)

    for i, mu in enumerate(mus):
        A = amplification_matrix_semi_implicite(mu, dt, c, b, alpha=alpha)
        radii[i] = spectral_radius(A)

    return {
        "theta": thetas,
        "mu": mus,
        "rho": radii,
        "rho_max": float(np.max(radii)),
    }


def dominant_eigenvalue(A):
    """
    Retourne la valeur propre dominante de la matrice A.

    Args:
        A (np.ndarray): Matrice carrée.

    Returns:
        complex: Valeur propre de plus grand module.
    """
    eigenvalues = np.linalg.eigvals(A)
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx]


def amplification_modulus(A):
    """
    Retourne le module de la valeur propre dominante.

    Args:
        A (np.ndarray): Matrice d'amplification.

    Returns:
        float: Module de la valeur propre dominante.
    """
    return float(np.abs(dominant_eigenvalue(A)))


def phase_dominant_eigenvalue(A):
    """
    Retourne la phase de la valeur propre dominante de la matrice A.

    Args:
        A (np.ndarray): Matrice carrée.

    Returns:
        float: Angle (phase) de la valeur propre dominante.
    """
    return np.angle(dominant_eigenvalue(A))

