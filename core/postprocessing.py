# core/postprocessing.py

import numpy as np
import matplotlib.pyplot as plt
from core.solver import WesterveltSolver
from utils import build_scan_grid, get_scan_axes


def run_snapshots(solver: WesterveltSolver, times_to_save):
    """Exécute la simulation et renvoie les snapshots demandés."""
    return solver.run_with_snapshots(times_to_save, store_energy=True)


def get_energy_history(solver: WesterveltSolver):
    """Renvoie l'historique d'énergie déjà stocké par le solver."""
    return np.array(solver.energy_history, dtype=float)


def plot_stability_scan(results):
    """Trace une carte simple stable/instable selon dt et amplitude."""
    if len(results) == 0:
        return

    dt_vals, amp_vals = get_scan_axes(results)
    grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r["stable"] else 0.0)

    plt.figure(figsize=(8, 5))
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    plt.colorbar(label="1 = stable, 0 = instable")
    plt.xlabel("dt")
    plt.ylabel("Amplitude initiale")
    plt.title("Carte de stabilité observée")
    plt.grid(False)
    plt.show()


def plot_stability_detailed(results):
    """
    Analyse comparative détaillée de la stabilité avec subplot_mosaic.
    Affiche :
    - stabilité observée
    - marge de stabilité théorique
    - minimum de 1 - 2ku
    - maximum de |u|
    """
    if len(results) == 0:
        return

    dt_vals, amp_vals = get_scan_axes(results)

    # Construire les grilles de metriques a partir d'un utilitaire commun
    stable_grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r.get("stable", False) else 0.0)
    margin_grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: r.get("stability_margin", np.nan))
    min_denom_grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: r.get("min_denom", np.nan))
    max_u_grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: r.get("max_abs_u", np.nan))

    # Création des sous-figures avec subplot_mosaic pour une meilleure organisation visuelle
    fig, axs = plt.subplot_mosaic(
        [["stability", "margin"], ["min_denom", "max_u"]],
        figsize=(14, 10),
        constrained_layout=True
    )

    # Affichage des sous-figures
    im1 = axs["stability"].imshow(
        stable_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["stability"].set_xlabel("dt (s)")
    axs["stability"].set_ylabel("Amplitude initiale")
    axs["stability"].set_title("Stabilité observée")
    plt.colorbar(im1, ax=axs["stability"], label="1 = stable")

    im2 = axs["margin"].imshow(
        margin_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        cmap="coolwarm",
    )
    axs["margin"].set_xlabel("dt (s)")
    axs["margin"].set_ylabel("Amplitude initiale")
    axs["margin"].set_title("Marge de stabilité théorique")
    plt.colorbar(im2, ax=axs["margin"], label="marge")

    im3 = axs["min_denom"].imshow(
        min_denom_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        cmap="viridis",
    )
    axs["min_denom"].set_xlabel("dt (s)")
    axs["min_denom"].set_ylabel("Amplitude initiale")
    axs["min_denom"].set_title("Minimum de 1 - 2ku")
    plt.colorbar(im3, ax=axs["min_denom"], label="min denom")

    im4 = axs["max_u"].imshow(
        max_u_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        cmap="plasma",
    )
    axs["max_u"].set_xlabel("dt (s)")
    axs["max_u"].set_ylabel("Amplitude initiale")
    axs["max_u"].set_title("Valeur maximale de |u| durant la simulation")
    plt.colorbar(im4, ax=axs["max_u"], label="max|u|")

    plt.suptitle("Analyse détaillée de stabilité", fontsize=14, fontweight="bold")
    plt.show()


def plot_theory_vs_observed(results):
    """
    Affiche côte à côte la stabilité théorique et observée, ainsi que leur différence.
    Utilise subplot_mosaic pour une meilleure organisation visuelle.
    """
    if len(results) == 0:
        return

    dt_vals, amp_vals = get_scan_axes(results)

    theoretical_grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r.get("theoretical_stable", False) else 0.0)
    observed_grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r.get("stable", False) else 0.0)

    diff_grid = observed_grid - theoretical_grid

    fig, axs = plt.subplot_mosaic(
        [["theory", "observed"], ["difference", "difference"]],
        figsize=(14, 10),
        constrained_layout=True
    )

    im1 = axs["theory"].imshow(
        theoretical_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["theory"].set_title("Stabilité théorique")
    axs["theory"].set_xlabel("dt (s)")
    axs["theory"].set_ylabel("Amplitude initiale")
    plt.colorbar(im1, ax=axs["theory"], label="théorie")

    im2 = axs["observed"].imshow(
        observed_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["observed"].set_title("Stabilité observée")
    axs["observed"].set_xlabel("dt (s)")
    axs["observed"].set_ylabel("Amplitude initiale")
    plt.colorbar(im2, ax=axs["observed"], label="observé")

    im3 = axs["difference"].imshow(
        diff_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )
    axs["difference"].set_xlabel("dt (s)")
    axs["difference"].set_ylabel("Amplitude initiale")
    axs["difference"].set_title("Différence (Observé - Théorie)")
    axs["difference"].text(
        0.5, 1.05,
        "Bleu: Observé plus stable | Rouge: Théorie plus stable",
        ha="center",
        transform=axs["difference"].transAxes,
        fontsize=9
    )
    plt.colorbar(im3, ax=axs["difference"], label="différence")

    plt.suptitle("Comparaison théorie / expérience", fontsize=14, fontweight="bold")
    plt.show()


def plot_snapshots_energy_comparison(solver: WesterveltSolver, snapshots, title_prefix=""):
    """
    Affiche les snapshots et l'énergie côte à côte avec subplot_mosaic.
    Facilite la comparaison entre l'évolution spatiale et énergétique.
    """
    if not snapshots:
        print("Aucun snapshot disponible.")
        return

    fig, axs = plt.subplot_mosaic(
        [["snapshots", "energy"]],
        figsize=(14, 5),
        constrained_layout=True
    )

    # Snapshots spatiaux
    for t in sorted(snapshots.keys()):
        axs["snapshots"].plot(solver.x, snapshots[t], label=f"t = {t * 1e6:.2f} µs")
    axs["snapshots"].set_xlabel("x (m)")
    axs["snapshots"].set_ylabel("u(x,t)")
    axs["snapshots"].set_title(f"{title_prefix}Evolution spatiale")
    axs["snapshots"].legend(fontsize=8)
    axs["snapshots"].grid(True, alpha=0.3)

    # Historique d'énergie
    if solver.energy_history:
        t_energy = np.arange(len(solver.energy_history)) * solver.param.dt
        axs["energy"].plot(t_energy, solver.energy_history, linewidth=1.5, color="darkblue")
        axs["energy"].set_xlabel("t (s)")
        axs["energy"].set_ylabel("Energie discrète")
        axs["energy"].set_title(f"{title_prefix}Evolution énergétique")
        axs["energy"].grid(True, alpha=0.3)
    else:
        axs["energy"].text(0.5, 0.5, "Pas d'historique d'énergie",
                          ha="center", va="center", transform=axs["energy"].transAxes)

    plt.suptitle(f"{title_prefix}Analyse spatiale vs énergétique", fontsize=13, fontweight="bold")
    plt.show()


def plot_scheme_comparison(results_explicit, results_semi_implicit):
    """
    Compare la stabilité entre le schéma explicite et semi-implicite.
    Utilise subplot_mosaic pour une visualisation côte à côte.
    """
    if not results_explicit or not results_semi_implicit:
        print("Résultats incomplets pour la comparaison.")
        return

    dt_vals, amp_vals = get_scan_axes(results_explicit)

    explicit_grid = build_scan_grid(results_explicit, dt_vals, amp_vals, lambda r: 1.0 if r["stable"] else 0.0)
    semi_implicit_grid = build_scan_grid(results_semi_implicit, dt_vals, amp_vals, lambda r: 1.0 if r["stable"] else 0.0)
    diff_grid = semi_implicit_grid - explicit_grid  # >0 si semi-implicite est meilleur

    fig, axs = plt.subplot_mosaic(
        [["explicit", "semi_implicit"], ["difference", "difference"]],
        figsize=(14, 10),
        constrained_layout=True
    )

    # Schéma explicite
    im1 = axs["explicit"].imshow(
        explicit_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["explicit"].set_xlabel("dt (s)")
    axs["explicit"].set_ylabel("Amplitude initiale")
    axs["explicit"].set_title("Schéma explicite")
    plt.colorbar(im1, ax=axs["explicit"], label="stable")

    # Schéma semi-implicite
    im2 = axs["semi_implicit"].imshow(
        semi_implicit_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["semi_implicit"].set_xlabel("dt (s)")
    axs["semi_implicit"].set_ylabel("Amplitude initiale")
    axs["semi_implicit"].set_title("Schéma semi-implicite")
    plt.colorbar(im2, ax=axs["semi_implicit"], label="stable")

    # Différence (semi-implicite - explicite)
    im3 = axs["difference"].imshow(
        diff_grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )
    axs["difference"].set_xlabel("dt (s)")
    axs["difference"].set_ylabel("Amplitude initiale")
    axs["difference"].set_title("Différence (Semi-implicite - Explicite)")
    axs["difference"].text(0.5, 1.05, "Bleu: Semi-implicite plus stable | Rouge: Explicite plus stable",
                          ha="center", transform=axs["difference"].transAxes, fontsize=9)
    plt.colorbar(im3, ax=axs["difference"], label="différence")

    plt.suptitle("Comparaison de stabilité entre les schémas numériques", fontsize=14, fontweight="bold")
    plt.show()

