# ./core/plots_stability.py

import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# Ajout du dossier racine au path pour les imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.stability_analysis import (
    scan_spectral_radius_explicit,
    scan_spectral_radius_semi_implicit,
    explicit_theoretical_stable,
    semi_implicit_theoretical_stable,
    amplification_matrix_explicite,
    amplification_matrix_semi_implicite,
    spectral_radius,
)
from utils import set_style, save_figure_with_version

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def plot_rho_theta_explicit(dt, dx, c, b, ntheta=500):
    data = scan_spectral_radius_explicit(dt, dx, c, b, ntheta)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(data['theta'], data['rho'], label=r'$\rho(A(\mu))$',)
    ax.axhline(1.0, linestyle='--', linewidth=1.2, label=r'$\rho(A(\mu)) = 1$')

    stable = explicit_theoretical_stable(dt, dx, c, b)
    status = 'stable' if stable else 'instable'

    ax.set_xlabel(r"Nombre d'onde discret $\mu$")
    ax.set_ylabel(r'Rayon spectral $\rho$')
    ax.set_title(f'Schéma explicite - rayon spectral ({status})')
    ax.grid(True, alpha=0.35)
    ax.legend()

    return fig


def plot_rho_theta_semi_implicit(dt, dx, c, b, alpha=1.0, ntheta=500):
    data = scan_spectral_radius_semi_implicit(dt, dx, c, b, alpha, ntheta)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(data['theta'], data['rho'], label=r'$\rho(A(\mu))$',)
    ax.axhline(1.0, linestyle='--', linewidth=1.2, label=r'$\rho(A(\mu)) = 1$')

    stable = semi_implicit_theoretical_stable(dt, dx, c, b, alpha=alpha)
    status = 'stable' if stable else 'instable'

    ax.set_title(f'Schéma semi-implicite - rayon spectral ({status})')
    ax.set_xlabel(r"Nombre d'onde discret $\mu$")
    ax.set_ylabel(r'Rayon spectral $\rho$')
    ax.set_ylim(0.9998, 1.00001)
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0), useMathText=True)
    ax.grid(True, alpha=0.35)
    ax.legend()

    return fig


def plot_compare_rho_vs_theta(dt, dx, c, b, alpha=1.0, ntheta=500):
    fig, ax = plt.subplots(figsize=(8, 6))

    data_exp = scan_spectral_radius_explicit(dt, dx, c, b, ntheta)
    data_imp = scan_spectral_radius_semi_implicit(dt, dx, c, b, alpha, ntheta)

    ax.plot(data_exp["theta"], data_exp['rho'], label='Explicite')
    ax.plot(data_imp["theta"], data_imp['rho'], label='Semi-implicite')
    ax.axhline(1.0, linestyle='--', linewidth=1.2, label=r'$\rho(A(\mu)) = 1$')

    rho_max = max(np.max(data_exp['rho']), np.max(data_imp['rho']))
    rho_min = min(np.min(data_exp['rho']), np.min(data_imp['rho']))

    margin = 1e-5

    ax.set_ylim(rho_min - margin, rho_max + margin)

    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0), useMathText=True)

    ax.set_xlabel(r"Nombre d'onde discret $\mu$")
    ax.set_ylabel(r'Rayon spectral $\rho$')
    ax.set_title('Comparaisons des rayons spectraux')
    ax.grid(True, alpha=0.35)
    ax.legend()

    return fig


def stability_map_explicit(
        dx,
        c,
        b,
        dt_min,
        dt_max,
        n_dt=500,
        ntheta=500,
):
    theta_values = np.linspace(0, np.pi, ntheta)
    dt_values = np.linspace(dt_min, dt_max, n_dt)

    rho = np.zeros((n_dt, ntheta))

    for i, dt in enumerate(dt_values):
        for j, theta in enumerate(theta_values):
            mu = 4.0 * np.sin(theta / 2.0) ** 2 / dx ** 2
            A = amplification_matrix_explicite(mu, dt, c, b)
            rho[i, j] = spectral_radius(A)

    fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.linspace(0, 2, 60)

    contour = ax.contourf(theta_values, dt_values, rho, levels=levels, cmap='RdYlBu_r')
    fig.colorbar(contour, ax=ax, label=r'$\rho(A(\mu))$')

    ax.contour(theta_values, dt_values, rho, levels=[1.0], colors='k', linestyles='--', linewidths=1.8)

    ax.set_title('Domaine de stabilité - schéma explicite')
    ax.set_xlabel(r"Nombre d'onde discret $\mu$")
    ax.set_ylabel(r'Pas de temps $\Delta t$')

    return fig


def stability_map_semi_implicit(
        dx,
        c,
        b,
        dt_min,
        dt_max,
        alpha=1.0,
        n_dt=500,
        ntheta=500,
):
    theta_values = np.linspace(0, np.pi, ntheta)
    dt_values = np.linspace(dt_min, dt_max, n_dt)

    rho = np.zeros((n_dt, ntheta))

    for i, dt in enumerate(dt_values):
        for j, theta in enumerate(theta_values):
            mu = 4.0 * np.sin(theta / 2.0) ** 2 / dx ** 2
            A = amplification_matrix_semi_implicite(mu, dt, c, b, alpha=alpha)
            rho[i, j] = spectral_radius(A)

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(theta_values, dt_values, rho, levels=60, cmap='RdYlBu_r')
    fig.colorbar(contour, ax=ax, label=r'$\rho(A(\mu))$')

    ax.contour(theta_values, dt_values, rho, levels=[1.0], colors='k', linestyles='--', linewidths=1.8)

    ax.set_title('Domaine de stabilité - schéma semi-implicite')
    ax.set_xlabel(r"Nombre d'onde discret $\mu$")
    ax.set_ylabel(r'Pas de temps $\Delta t$')

    return fig


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
    c = 1500
    b = 6e-6
    dx = 1e-5
    dt = 1.8e-8
    alpha = 2.0

    # Plots
    fig1 = plot_rho_theta_explicit(dt, dx, c, b)
    fig2 = plot_rho_theta_semi_implicit(dt, dx, c, b, alpha)
    fig3 = plot_compare_rho_vs_theta(dt, dx, c, b, alpha)
    fig4 = stability_map_explicit(dx, c, b, dt_min=1e-8, dt_max=3e-8)
    fig5 = stability_map_semi_implicit(dx, c, b, dt_min=1e-8, dt_max=3e-8, alpha=alpha)

    if save and save_path is not None:
        save_figure_with_version(fig1, "rho_theta_explicit",  output_dir=save_path)
        save_figure_with_version(fig2, "rho_theta_semi_implicit",  output_dir=save_path)
        save_figure_with_version(fig3, "compare_rho_theta",  output_dir=save_path)
        save_figure_with_version(fig4, "stability_map_explicit",  output_dir=save_path)
        save_figure_with_version(fig5, "stability_map_semi_implicit",  output_dir=save_path)

    if show:
        plt.show()
    else:
        plt.close('all')

    return {
        "figures": [fig1, fig2, fig3, fig4, fig5],
    }


if __name__ == "__main__":
    stability_dir = OUTPUTS_DIR / "stability_plots"
    run_stability_plots(theme="scientific", show=True, save=False, save_path=stability_dir)
