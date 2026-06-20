# ./core/beamer_style.py
"""
Style et utilitaires partagés pour les figures de soutenance (Beamer).

Utilisé par plots_simulation.py (et utilisable par plots_stability.py) :

    from core.beamer_style import (
        beamer, BEAMER_RC, COL_EXP, COL_IMP, sci,
        time_colors, slope_triangle, fitted_order, annotate_max,
    )

    with beamer():
        fig, ax = plt.subplots(...)
"""

import contextlib

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Style "Beamer" : gros caractères, traits épais, lisible depuis le fond
# ----------------------------------------------------------------------------
BEAMER_RC = {
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "axes.linewidth": 1.2,
    "figure.constrained_layout.use": True,
    "mathtext.fontset": "cm",
}

# Couleurs cohérentes sur toutes les figures de la soutenance
COL_EXP = "#1f77b4"    # schéma explicite
COL_IMP = "#d62728"    # schéma semi-implicite
COL_REF = "#2c2c2c"    # solution de référence / exacte
COL_OK = "#2ca02c"     # stable / valide
COL_KO = "#d62728"     # instable / danger


def beamer():
    """Contexte rc pour une figure de soutenance : `with beamer(): ...`."""
    return plt.rc_context(BEAMER_RC)


def sci(x, digits=2):
    """Formate x en notation scientifique LaTeX : 6.66 \\times 10^{-9}."""
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m} \times 10^{{{int(e)}}}"


def time_colors(n, cmap="plasma", lo=0.0, hi=0.92):
    """n couleurs ordonnées pour des instantanés successifs (clair -> foncé)."""
    return plt.get_cmap(cmap)(np.linspace(lo, hi, n))


def fitted_order(x, y):
    """Ordre de convergence global par moindres carrés en échelle log-log.

    Retourne la pente p de log(y) = p log(x) + c, c'est-à-dire l'ordre
    expérimental moyen sur tous les niveaux de raffinement (plus robuste que
    les ordres locaux entre niveaux consécutifs, sensibles au bruit).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return np.nan
    p, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(p)


def slope_triangle(ax, x0, x1, y0, order, label=None, color="0.3",
                   above=False, fontsize=13):
    """Triangle de pente standard sur un graphe log-log.

    Dessine le triangle rectangle reliant (x0, y0) à (x1, y0 (x1/x0)^order),
    avec la mention de l'ordre sur l'hypoténuse. Convention classique des
    figures de convergence en analyse numérique.

    Args:
        ax: axes en échelle log-log.
        x0, x1: abscisses des deux sommets (x1 > x0).
        y0: ordonnée du sommet gauche.
        order: pente visée (1, 2, ...).
        above: place le triangle au-dessus de la courbe (cathètes en haut).
    """
    y1 = y0 * (x1 / x0) ** order
    if above:
        xs = [x0, x0, x1, x0]
        ys = [y0, y1, y1, y0]
        tx, ty = x0 * 0.93, np.sqrt(y0 * y1)
        ha = "right"
    else:
        xs = [x0, x1, x1, x0]
        ys = [y0, y0, y1, y0]
        tx, ty = x1 * 1.07, np.sqrt(y0 * y1)
        ha = "left"
    ax.plot(xs, ys, color=color, lw=1.4)
    txt = label if label is not None else rf"${order:g}$"
    ax.text(tx, ty, txt, color=color, fontsize=fontsize,
            ha=ha, va="center")


def annotate_max(ax, x, y, color, unit="", fmt=".3g", dy=12):
    """Marque le maximum d'une courbe et l'annote."""
    i = int(np.argmax(y))
    ax.plot(x[i], y[i], "o", color=color, markersize=8, zorder=5)
    ax.annotate(rf"max $= {y[i]:{fmt}}${unit}",
                xy=(x[i], y[i]), xytext=(0, dy),
                textcoords="offset points", ha="center",
                fontsize=13, color=color)
    return i
