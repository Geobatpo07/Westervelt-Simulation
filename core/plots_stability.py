# ./core/plots_stability.py
"""
Figures de stabilité orientées présentation (Beamer).

Principes retenus pour des graphiques "parlants" en soutenance :
  - une figure = un message (titre explicite, annotations directement sur la courbe) ;
  - zone instable rho > 1 ombrée en rouge, zone stable en vert ;
  - frontière théorique du chapitre 4 superposée aux résultats numériques ;
  - point de fonctionnement (Delta t utilisé) marqué sur les cartes ;
  - polices et traits épais, lisibles depuis le fond de la salle.

Conventions :
  - theta in [0, pi] est l'angle du mode discret, mu(theta) = 4 sin^2(theta/2) / dx^2 ;
  - theta = 0 correspond au mode du noyau (mu_0 = 0), theta = pi au mode le plus
    oscillant (mu_max = 4/dx^2), qui dicte la condition CFL.
"""

import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

# Ajout du dossier racine au path pour les imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.stability_analysis import (
    discrete_mu,
    scan_spectral_radius_explicit,
    scan_spectral_radius_semi_implicit,
    explicit_theoretical_stable,
    semi_implicit_theoretical_stable,
    amplification_matrix_explicite,
    amplification_matrix_semi_implicite,
    spectral_radius,
    critical_dt_explicit,
    eigenvalues_amplification,
)
from utils import set_style, save_figure_with_version

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ----------------------------------------------------------------------------
# Style "Beamer" : gros caractères, traits épais, mise en page compacte
# ----------------------------------------------------------------------------
BEAMER_RC = {
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.6,
    "axes.linewidth": 1.2,
    "figure.constrained_layout.use": True,
    "mathtext.fontset": "cm",
}

COL_EXP = "#1f77b4"      # explicite
COL_IMP = "#d62728"      # semi-implicite
COL_OK = "#2ca02c"       # stable
COL_KO = "#d62728"       # instable
SHADE_KO = "#d62728"     # ombrage zone instable


def critical_dt_semi_implicit(dx, c, b, alpha=1.0):
    """Pas de temps critique du schéma semi-implicite :
    c^2 dt^2 - 2 b dt <= alpha dx^2  =>  dt <= (b + sqrt(b^2 + alpha c^2 dx^2)) / c^2.
    (Formule utilisée au chapitre 5, tableau 5.6.)
    """
    return (b + np.sqrt(b ** 2 + alpha * c ** 2 * dx ** 2)) / c ** 2


def _shade_instability(ax, ymax=None):
    """Ombre la zone rho > 1 et trace la ligne rho = 1."""
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.6, zorder=3)
    lo, hi = ax.get_ylim()
    hi = ymax if ymax is not None else hi
    if hi > 1.0:
        ax.axhspan(1.0, hi, color=SHADE_KO, alpha=0.10, zorder=0)
        ax.text(0.02, 1.0, " instable", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=12, color=COL_KO, alpha=0.9)
    ax.set_ylim(lo, hi)


def _sci(x, digits=2):
    """Formate x en notation scientifique LaTeX : 6.66 x 10^-9."""
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m} \times 10^{{{int(e)}}}"


def _annotate_rho_max(ax, theta, rho, color, dy=12):
    """Marque et annote le maximum du rayon spectral.

    Si le maximum est atteint au mode du noyau (theta = 0, rho = 1 exactement,
    cf. Prop. 4.5 / 4.15), on le signale comme tel plutôt que comme un rho_max
    "ordinaire" : c'est le cas neutre traite par la condition F0(0) = 0.
    """
    i = int(np.argmax(rho))
    ax.plot(theta[i], rho[i], "o", color=color, markersize=8, zorder=5)
    if i == 0:
        ax.annotate(r"mode du noyau : $\rho = 1$",
                    xy=(theta[i], rho[i]), xytext=(20, -26),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=13, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    else:
        ax.annotate(rf"$\rho_{{\max}} = {rho[i]:.4f}$",
                    xy=(theta[i], rho[i]), xytext=(-10, dy),
                    textcoords="offset points", ha="right",
                    fontsize=13, color=color)


def _theta_axis(ax):
    """Axe des modes : theta/pi avec rappel des modes extrêmes."""
    ax.set_xlabel(r"Angle du mode discret $\theta$  " 
                  r"($\mu = \frac{4}{\Delta x^2}\sin^2\frac{\theta}{2}$)")
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlim(0, np.pi)


def _cfl_text_explicit(dt, dx, c, b):
    ratio = (c ** 2 * dt ** 2 + 2 * b * dt) / dx ** 2
    return (rf"$\dfrac{{c^2\Delta t^2 + 2b\Delta t}}{{\Delta x^2}} = {ratio:.3f}$"
            + ("  $\\leq 1$" if ratio <= 1 else "  $> 1$"))


def _cfl_text_semi_implicit(dt, dx, c, b, alpha):
    ratio = (c ** 2 * dt ** 2 - 2 * b * dt) / (alpha * dx ** 2)
    return (rf"$\dfrac{{c^2\Delta t^2 - 2b\Delta t}}{{\alpha\,\Delta x^2}} = {ratio:.3f}$"
            + ("  $\\leq 1$" if ratio <= 1 else "  $> 1$"))


# ----------------------------------------------------------------------------
# 1. Rayon spectral en fonction du mode — un schéma
# ----------------------------------------------------------------------------
def plot_rho_theta_explicit(dt, dx, c, b, ntheta=500):
    data = scan_spectral_radius_explicit(dt, dx, c, b, ntheta)
    stable = explicit_theoretical_stable(dt, dx, c, b)

    with plt.rc_context(BEAMER_RC):
        fig, ax = plt.subplots(figsize=(9, 5.4))

        ax.plot(data["theta"], data["rho"], color=COL_EXP,
                label=r"$\rho(A(\mu_m))$")
        _annotate_rho_max(ax, data["theta"], data["rho"], COL_EXP)

        # zoom automatique autour de [min, max] avec marge, en gardant rho=1 visible
        lo = min(data["rho"].min(), 1.0)
        hi = max(data["rho"].max(), 1.0)
        pad = 0.15 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        _shade_instability(ax)

        _theta_axis(ax)
        ax.set_ylabel(r"Rayon spectral $\rho(A)$")

        status = "STABLE" if stable else "INSTABLE"
        color = COL_OK if stable else COL_KO
        ax.set_title("Schéma explicite — analyse de Von Neumann")
        ax.text(0.985, 0.06, status, transform=ax.transAxes, ha="right",
                fontsize=20, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=2))
        ax.text(0.03, 0.06, _cfl_text_explicit(dt, dx, c, b),
                transform=ax.transAxes, fontsize=13, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    return fig


def plot_rho_theta_semi_implicit(dt, dx, c, b, alpha=1.0, ntheta=500):
    data = scan_spectral_radius_semi_implicit(dt, dx, c, b, alpha, ntheta)
    stable = semi_implicit_theoretical_stable(dt, dx, c, b, alpha=alpha)

    with plt.rc_context(BEAMER_RC):
        fig, ax = plt.subplots(figsize=(9, 5.4))

        ax.plot(data["theta"], data["rho"], color=COL_IMP,
                label=r"$\rho(A(\mu_m))$")
        _annotate_rho_max(ax, data["theta"], data["rho"], COL_IMP, dy=-26)

        # zoom automatique (remplace l'ancien ylim codé en dur)
        lo = min(data["rho"].min(), 1.0)
        hi = max(data["rho"].max(), 1.0)
        pad = 0.15 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        _shade_instability(ax)

        _theta_axis(ax)
        ax.set_ylabel(r"Rayon spectral $\rho(A)$")
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

        status = "STABLE" if stable else "INSTABLE"
        color = COL_OK if stable else COL_KO
        ax.set_title(rf"Schéma semi-implicite ($\alpha = {alpha:g}$) — Von Neumann")
        ax.text(0.985, 0.06, status, transform=ax.transAxes, ha="right",
                fontsize=20, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=2))
        ax.text(0.03, 0.06, _cfl_text_semi_implicit(dt, dx, c, b, alpha),
                transform=ax.transAxes, fontsize=13, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))

        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.18))

    return fig


# ----------------------------------------------------------------------------
# 2. Comparaison directe explicite / semi-implicite (même Delta t)
# ----------------------------------------------------------------------------
def plot_compare_rho_vs_theta(dt, dx, c, b, alpha=1.0, ntheta=500):
    data_exp = scan_spectral_radius_explicit(dt, dx, c, b, ntheta)
    data_imp = scan_spectral_radius_semi_implicit(dt, dx, c, b, alpha, ntheta)

    with plt.rc_context(BEAMER_RC):
        fig, ax = plt.subplots(figsize=(9, 5.4))

        ax.plot(data_exp["theta"], data_exp["rho"], color=COL_EXP,
                label=f"Explicite  ($\\rho_{{\\max}}={data_exp['rho_max']:.4f}$)")
        ax.plot(data_imp["theta"], data_imp["rho"], color=COL_IMP,
                label=f"Semi-implicite  ($\\rho_{{\\max}}={data_imp['rho_max']:.4f}$)")

        lo = min(data_exp["rho"].min(), data_imp["rho"].min(), 1.0)
        hi = max(data_exp["rho"].max(), data_imp["rho"].max(), 1.0)
        pad = 0.15 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        _shade_instability(ax)

        _theta_axis(ax)
        ax.set_ylabel(r"Rayon spectral $\rho(A)$")
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.set_title(rf"Même $\Delta t = {_sci(dt)}$ s — qui reste stable ?")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    return fig


def plot_compare_stable_unstable(dx, c, b, alpha=1.0, ntheta=500,
                                 frac_stable=0.9, frac_unstable=1.2):
    """Figure "choc" pour le Beamer : deux panneaux côte à côte.

    Gauche  : Delta t = frac_stable   * Delta t_crit(explicite) -> les deux stables.
    Droite  : Delta t = frac_unstable * Delta t_crit(explicite) -> l'explicite
              franchit rho = 1, le semi-implicite reste sous la barre.
    Le message : le traitement implicite de la dissipation relaxe la CFL.
    """
    dt_crit = critical_dt_explicit(dx, c, b)
    cases = [
        (frac_stable * dt_crit,
         rf"$\Delta t = {frac_stable:g}\,\Delta t_c$  (sous la CFL)"),
        (frac_unstable * dt_crit,
         rf"$\Delta t = {frac_unstable:g}\,\Delta t_c$  (au-delà de la CFL)"),
    ]

    with plt.rc_context(BEAMER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)

        for ax, (dt, subtitle) in zip(axes, cases):
            d_exp = scan_spectral_radius_explicit(dt, dx, c, b, ntheta)
            d_imp = scan_spectral_radius_semi_implicit(dt, dx, c, b, alpha, ntheta)

            ax.plot(d_exp["theta"], d_exp["rho"], color=COL_EXP, label="Explicite")
            ax.plot(d_imp["theta"], d_imp["rho"], color=COL_IMP, label="Semi-implicite")

            lo = min(d_exp["rho"].min(), d_imp["rho"].min(), 1.0)
            hi = max(d_exp["rho"].max(), d_imp["rho"].max(), 1.0)
            pad = 0.15 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            _shade_instability(ax)
            _annotate_rho_max(ax, d_exp["theta"], d_exp["rho"], COL_EXP)
            # sur le panneau instable, rappeler que le semi-implicite reste <= 1
            if d_exp["rho_max"] > 1.0:
                ax.annotate(
                    rf"semi-implicite : $\rho_{{\max}} = {d_imp['rho_max']:.4f}$",
                    xy=(np.pi / 2, 1.0), xytext=(0, -26),
                    textcoords="offset points", ha="center",
                    fontsize=13, color=COL_IMP,
                    arrowprops=dict(arrowstyle="->", color=COL_IMP, lw=1.4),
                )

            _theta_axis(ax)
            ax.set_title(subtitle, fontsize=15)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r"Rayon spectral $\rho(A)$")
        axes[0].legend(loc="lower left")
        fig.suptitle(
            rf"Le semi-implicite relaxe la contrainte CFL ($\Delta t_c = {_sci(dt_crit)}$ s)", fontsize=17,)

    return fig


# ----------------------------------------------------------------------------
# 3. Cartes de stabilité (theta, Delta t) avec frontière théorique
# ----------------------------------------------------------------------------
def _theoretical_dt_boundary_explicit(theta, dx, c, b):
    """Pour chaque mode, dt limite :  mu (c^2 dt^2 + 2 b dt) = 4."""
    mu = discrete_mu(theta, dx)
    mu = np.where(mu > 0, mu, np.nan)  # le mode du noyau n'impose rien
    return (-b + np.sqrt(b ** 2 + 4.0 * c ** 2 / mu)) / c ** 2


def _theoretical_dt_boundary_semi_implicit(theta, dx, c, b, alpha):
    """Pour chaque mode, dt limite :  mu (c^2 dt^2 - 2 b dt) = 4 alpha."""
    mu = discrete_mu(theta, dx)
    mu = np.where(mu > 0, mu, np.nan)
    return (b + np.sqrt(b ** 2 + 4.0 * alpha * c ** 2 / mu)) / c ** 2


def _stability_map(amp_matrix, boundary_fn, dt_crit_global, title,
                   dx, c, b, dt_min, dt_max, dt_op=None,
                   n_dt=300, ntheta=300, **amp_kwargs):
    theta_values = np.linspace(0.0, np.pi, ntheta)
    dt_values = np.linspace(dt_min, dt_max, n_dt)

    rho = np.zeros((n_dt, ntheta))
    for i, dt in enumerate(dt_values):
        for j, theta in enumerate(theta_values):
            mu = discrete_mu(theta, dx)
            A = amp_matrix(mu, dt, c, b, **amp_kwargs)
            rho[i, j] = spectral_radius(A)

    with plt.rc_context(BEAMER_RC):
        fig, ax = plt.subplots(figsize=(9.5, 6))

        # Échelle de couleurs centrée sur rho = 1 (bleu = stable, rouge = instable).
        # On écrête rho pour la lisibilité : au-delà de la frontière, seul le
        # franchissement compte, pas la valeur exacte.
        rho_min = float(rho.min())
        cap = 1.0 + 1.5 * max(1.0 - rho_min, 1e-6)
        rho_disp = np.minimum(rho, cap)
        norm = TwoSlopeNorm(vmin=rho_min, vcenter=1.0, vmax=cap)
        contour = ax.contourf(theta_values, dt_values, rho_disp,
                              levels=np.linspace(rho_min, cap, 61),
                              norm=norm, cmap="RdBu_r", extend="max")
        cbar = fig.colorbar(contour, ax=ax, label=r"$\rho(A(\mu))$")
        cbar.ax.axhline(1.0, color="k", lw=1.5)

        # frontière numérique rho = 1
        ax.contour(theta_values, dt_values, rho, levels=[1.0], colors="k", linestyles="-", linewidths=2.2)

        # frontière théorique du chapitre 4 (mode par mode), écrêtée au cadre
        dt_bound = boundary_fn(theta_values)
        ax.plot(theta_values, np.clip(dt_bound, dt_min, dt_max), color="#cc00cc", linestyle="--", linewidth=2.4)

        # condition suffisante globale (pire mode mu = 4/dx^2)
        if dt_min < dt_crit_global < dt_max:
            ax.axhline(dt_crit_global, color="#ffd700", linestyle=":", linewidth=2.6)

        # point de fonctionnement
        if dt_op is not None and dt_min <= dt_op <= dt_max:
            ax.plot(np.pi * 0.97, dt_op, "*", color="k", markersize=18, markeredgecolor="w", markeredgewidth=1.2, zorder=6)
            ax.annotate(r"$\Delta t$ utilisé", xy=(np.pi * 0.97, dt_op), xytext=(-100, 8), textcoords="offset points", fontsize=13, fontweight="bold")

        # étiquettes des régions
        ax.text(0.40, 0.06, "STABLE", transform=ax.transAxes, fontsize=18, fontweight="bold", color="#0b3d91", alpha=0.9)
        ax.text(0.62, 0.92, "INSTABLE", transform=ax.transAxes, fontsize=18, fontweight="bold", color="white", alpha=0.95)

        ax.set_ylim(dt_values[0], dt_values[-1])  # la frontière divergente en
        # theta -> 0 ne doit pas dilater l'axe

        _theta_axis(ax)
        ax.set_ylabel(r"Pas de temps $\Delta t$ [s]")
        ax.set_title(title)

        handles = [
            Line2D([], [], color="k", lw=2.2, label=r"$\rho = 1$ (numérique)"),
            Line2D([], [], color="#cc00cc", lw=2.4, ls="--", label="frontière théorique"),
            Line2D([], [], color="#ffd700", lw=2.6, ls=":", label=r"$\Delta t_c$ global"),
        ]
        ax.legend(handles=handles, loc="center left", framealpha=0.9, fontsize=12)
        ax.grid(False)

    return fig


def stability_map_explicit(dx, c, b, dt_min, dt_max, dt_op=None, n_dt=300, ntheta=300):
    return _stability_map(
        amplification_matrix_explicite,
        lambda th: _theoretical_dt_boundary_explicit(th, dx, c, b),
        critical_dt_explicit(dx, c, b),
        "Domaine de stabilité — schéma explicite",
        dx, c, b, dt_min, dt_max, dt_op=dt_op, n_dt=n_dt, ntheta=ntheta,
    )


def stability_map_semi_implicit(dx, c, b, dt_min, dt_max, alpha=1.0, dt_op=None, n_dt=300, ntheta=300):
    return _stability_map(
        amplification_matrix_semi_implicite,
        lambda th: _theoretical_dt_boundary_semi_implicit(th, dx, c, b, alpha),
        critical_dt_semi_implicit(dx, c, b, alpha),
        rf"Domaine de stabilité — schéma semi-implicite ($\alpha={alpha:g}$)",
        dx, c, b, dt_min, dt_max, dt_op=dt_op, n_dt=n_dt, ntheta=ntheta,
        alpha=alpha,
    )


# ----------------------------------------------------------------------------
# 4. Pas de temps critique en fonction de Delta x : la CFL relaxée
# ----------------------------------------------------------------------------
def plot_critical_dt_vs_dx(media=None, alpha=1.0,
                           dx_min=1e-7, dx_max=1e-3, n=300):
    """Compare Delta t_max(Delta x) des deux schémas pour plusieurs milieux.

    Message Beamer : l'écart entre explicite et semi-implicite est le gain
    apporté par le traitement implicite de la dissipation. Il est négligeable
    dans l'eau (b petit) mais atteint plusieurs ordres de grandeur dans le
    glycérol sur maillage fin, là où la contrainte parabolique
    Delta t <~ Delta x^2 / (2b) domine la CFL hyperbolique Delta t ~ Delta x / c.
    (Paramètres physiques : Annexe B du mémoire.)

    Args:
        media: liste de tuples (nom, c, b, couleur). Par défaut : eau, glycérol.
    """
    if media is None:
        media = [
            ("Eau",      1500.0, 2.478e-6, "#1f77b4"),
            ("Glycérol", 1930.0, 1.674e-3, "#d62728"),
        ]
    dx = np.logspace(np.log10(dx_min), np.log10(dx_max), n)

    with plt.rc_context(BEAMER_RC):
        fig, ax = plt.subplots(figsize=(9.5, 6))

        for name, c, b, color in media:
            dt_exp = critical_dt_explicit(dx, c, b)
            dt_imp = critical_dt_semi_implicit(dx, c, b, alpha)

            ax.loglog(dx, dt_exp, color=color, ls="-",
                      label=f"{name} - explicite")
            ax.loglog(dx, dt_imp, color=color, ls="--",
                      label=f"{name} - semi-implicite")
            ax.fill_between(dx, dt_exp, dt_imp, color=color, alpha=0.10)

            # gain au point le plus contraint (maillage le plus fin)
            gain = dt_imp[0] / dt_exp[0]
            if gain >= 2.0:
                gain_txt = f"{gain:.0f}" if gain >= 10 else f"{gain:.1f}"
                ax.annotate(rf"gain $\times\,{gain_txt}$",
                            xy=(dx[0], dt_imp[0]),
                            xytext=(14, 4), textcoords="offset points",
                            ha="left", va="bottom",
                            fontsize=14, fontweight="bold", color=color)

        # pentes de référence : régimes hyperbolique (dt ~ dx) et
        # parabolique (dt ~ dx^2)
        ax.loglog(dx, dx / 1700.0, color="0.55", ls=":", lw=1.6)
        ax.text(dx[-1], dx[-1] / 1700.0, r"  $\propto \Delta x$",
                color="0.4", fontsize=13, va="center")

        ax.set_xlabel(r"Pas d'espace $\Delta x$ [m]")
        ax.set_ylabel(r"Pas de temps critique $\Delta t_c$ [s]")
        ax.set_title("Quand le schéma semi-implicite devient avantageux")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=12)

    return fig


# ----------------------------------------------------------------------------
# 5. Valeurs propres dans le plan complexe (cercle unité)
# ----------------------------------------------------------------------------
def plot_eigenvalue_locus(dt, dx, c, b, alpha=1.0, ntheta=400):
    """Trajectoires des valeurs propres de A(mu) quand theta parcourt [0, pi].

    Très visuel en soutenance : la stabilité = "tout reste dans le disque unité".
    On voit aussi le mode du noyau (lambda double = 1) discuté aux Prop. 4.5/4.15.
    """
    thetas = np.linspace(0.0, np.pi, ntheta)

    def locus(matrix_fn, **kw):
        eigs = np.empty((ntheta, 2), dtype=complex)
        for i, th in enumerate(thetas):
            mu = discrete_mu(th, dx)
            eigs[i] = eigenvalues_amplification(matrix_fn(mu, dt, c, b, **kw))
        return eigs

    eig_exp = locus(amplification_matrix_explicite)
    eig_imp = locus(amplification_matrix_semi_implicite, alpha=alpha)

    with plt.rc_context(BEAMER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)

        for ax, eigs, color, name in (
            (axes[0], eig_exp, COL_EXP, "Explicite"),
            (axes[1], eig_imp, COL_IMP, "Semi-implicite"),
        ):
            phi = np.linspace(0, 2 * np.pi, 400)
            ax.plot(np.cos(phi), np.sin(phi), color="k", lw=1.4, ls="--",
                    label="cercle unité")
            ax.fill(np.cos(phi), np.sin(phi), color=COL_OK, alpha=0.06)

            sc = ax.scatter(eigs.real.ravel(), eigs.imag.ravel(),
                            c=np.repeat(thetas, 2), cmap="viridis",
                            s=10, zorder=4)

            # mode du noyau : lambda = 1 double
            ax.plot(1.0, 0.0, "s", color=COL_KO, markersize=10, zorder=6)
            ax.annotate(r"noyau : $\lambda = 1$ (double)", xy=(1.0, 0.0),
                        xytext=(-10, -22), textcoords="offset points",
                        ha="right", fontsize=12, color=COL_KO)

            ax.set_title(name)
            ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r"$\mathrm{Im}\,\lambda$")
        cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
        cbar.set_label(r"angle du mode $\theta$")
        fig.suptitle(
            rf"Spectre de la matrice d'amplification "
            rf"($\Delta t = {_sci(dt)}$ s)",
            fontsize=17,
        )

    return fig


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_stability_plots(
        theme: str,
        show: bool = True,
        save: bool = True,
        save_path: pathlib.Path | str | None = None,
):
    set_style(theme)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Paramètres de la simulation
    c = 1500.0
    b = 6e-6
    dx = 1e-5
    alpha = 2.0

    # On exprime Delta t relativement au Delta t critique de l'explicite :
    # les figures racontent alors une histoire cohérente quel que soit (c, b, dx).
    dt_c = critical_dt_explicit(dx, c, b)
    dt_stable = 0.9 * dt_c
    dt_unstable = 1.2 * dt_c

    figures = {
        # 1) un schéma à la fois, cas stable
        "rho_theta_explicit": plot_rho_theta_explicit(dt_stable, dx, c, b),
        "rho_theta_semi_implicit": plot_rho_theta_semi_implicit(
            dt_stable, dx, c, b, alpha),
        # 2) comparaison au même dt + figure "choc" stable/instable
        "compare_rho_theta": plot_compare_rho_vs_theta(
            dt_unstable, dx, c, b, alpha),
        "compare_stable_unstable": plot_compare_stable_unstable(
            dx, c, b, alpha),
        # 3) cartes (theta, dt) avec frontière théorique et point utilisé
        "stability_map_explicit": stability_map_explicit(
            dx, c, b, dt_min=0.3 * dt_c, dt_max=2.0 * dt_c, dt_op=dt_stable),
        "stability_map_semi_implicit": stability_map_semi_implicit(
            dx, c, b, dt_min=0.3 * dt_c, dt_max=2.0 * dt_c,
            alpha=alpha, dt_op=dt_stable),
        # 4) CFL relaxée : comparaison eau / glycérol (Annexe B).
        # alpha = 1 ici : dans les régimes simulés 2k||u|| << 1, et cela
        # isole l'effet du traitement implicite de la dissipation.
        "critical_dt_vs_dx": plot_critical_dt_vs_dx(alpha=1.0),
        # 5) spectre dans le plan complexe
        "eigenvalue_locus": plot_eigenvalue_locus(dt_stable, dx, c, b, alpha),
    }

    if save and save_path is not None:
        for name, fig in figures.items():
            save_figure_with_version(fig, name, output_dir=save_path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return {"figures": list(figures.values()), "named_figures": figures}


if __name__ == "__main__":
    stability_dir = OUTPUTS_DIR / "stability_plots"
    run_stability_plots(theme="scientific", show=True, save=True, save_path=stability_dir)
