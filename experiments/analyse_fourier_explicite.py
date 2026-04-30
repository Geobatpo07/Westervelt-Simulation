"""
Analyse de Fourier du schéma explicite de Westervelt.

Ce script réalise une analyse de stabilité, de dispersion et d’amortissement
dans le régime linéarisé (k = 0) sur une grille périodique.

On considère un mode de Fourier de la forme :
    u_i^n = G^n exp(i i theta),
où theta = kappa * dx est le nombre d’onde réduit.

Pour le schéma explicite, on obtient la récurrence :
    u^{n+1} = [2 - sigma (CFL^2 + D)] u^n + [-1 + sigma D] u^{n-1}

avec :
    sigma = 4 sin^2(theta / 2)
    CFL   = c dt / dx
    D     = b dt / dx^2   (nombre de diffusion discret sans dimension)

Le facteur d’amplification G est donné par les racines du polynôme :
    G^2 - [2 - sigma (CFL^2 + D)] G + [1 - sigma D] = 0

Le script s’appuie sur la matrice d’amplification pour :
    - calculer le rayon spectral (critère de stabilité),
    - sélectionner la branche physique du facteur d’amplification,
    - comparer la réponse numérique à la réponse analytique du modèle continu linéarisé.

Trois types de graphiques sont produits :
1. Module du facteur d’amplification et dispersion numérique en fonction de theta,
2. Représentation paramétrique du compromis amplification / dispersion,
3. Effet du nombre de diffusion sur l’amortissement des modes de Fourier.

Le taux d’amortissement est défini par :
    alpha_num(theta) = -log(|G(theta)|) / dt

Les résultats numériques sont comparés aux valeurs analytiques issues de la
relation de dispersion du modèle continu linéarisé.

Les figures sont enregistrées dans :
    outputs/analysis/explicit-fourier-analysis
avec versioning automatique.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FOURIER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "explicit-fourier-analysis"

from core.solver import WesterveltParams
from core.stability_analysis import (
    amplification_matrix_explicite,
    discrete_mu,
    explicit_stability_margin,
    explicit_theoretical_stable,
)
from utils import save_figure_with_version, set_style


STABILITY_TOL = 1e-10


@dataclass(frozen=True)
class FourierCurve:
    """Courbe Fourier pour un couple (CFL, diffusion) donne."""

    cfl: float
    D: float
    label: str
    color: str


def _theta_grid(num_points: int = 500) -> np.ndarray:
    """Grille de nombres d'onde reduits theta dans [0, pi]."""
    return np.linspace(0.0, np.pi, num_points)


def select_physical_branch_from_matrix(theta, params, dt, b):
    mu_values = discrete_mu(theta, params.dx)

    physical = np.empty_like(theta, dtype=complex)
    rho = np.empty_like(theta, dtype=float)

    physical[0] = 1.0 + 0.0j
    rho[0] = 1.0
    previous = 1.0 + 0.0j

    for i, mu in enumerate(mu_values):
        A = amplification_matrix_explicite(float(mu), dt, params.c, b)
        eigenvalues = np.linalg.eigvals(A)
        rho[i] = np.max(np.abs(eigenvalues))

        omega_candidates = - np.angle(eigenvalues) / dt

        # racines physiques admissibles
        valid_idx = np.where(np.abs(omega_candidates) >= 0.0)[0]

        if len(valid_idx) == 1:
            chosen = eigenvalues[valid_idx[0]]
        elif len(valid_idx) == 2:
            # si les deux sont admissibles, on garde la continuite
            idx_local = valid_idx[np.argmin(np.abs(eigenvalues[valid_idx] - previous))]
            chosen = eigenvalues[idx_local]
        else:
            idx_local = np.argmin(np.abs(eigenvalues - previous))
            chosen = eigenvalues[idx_local]

        physical[i] = chosen
        previous = chosen

    return physical, rho

def compute_numerical_response(
    theta: np.ndarray,
    params: WesterveltParams,
    cfl: float,
    D: float,
) -> dict[str, np.ndarray]:
    """Retourne les indicateurs numeriques via la matrice d'amplification."""
    dt_eff, b_eff = _effective_dt_and_b(params, cfl, D)

    g_phys, rho = select_physical_branch_from_matrix(theta, params, dt_eff, b_eff)

    phase = np.unwrap(np.angle(g_phys))
    omega_num = phase / dt_eff
    kappa = theta / params.dx

    phase_velocity_ratio = np.ones_like(theta)
    nonzero = kappa > 0.0
    phase_velocity_ratio[nonzero] = omega_num[nonzero] / (params.c * kappa[nonzero])

    amortissement_rate = -np.log(np.maximum(np.abs(g_phys), 1e-15)) / dt_eff
    return {
        "physical_root": g_phys,
        "spectral_radius": rho,
        "phase_velocity_ratio": phase_velocity_ratio,
        "amortissement_rate": amortissement_rate,
    }


def compute_exact_continuous_response(theta: np.ndarray, params: WesterveltParams) -> dict[str, np.ndarray]:
    """Reponse analytique de l'equation linearisee continue."""
    kappa = theta / params.dx
    disc = 4.0 * (params.c**2) * (kappa**2) - (params.b**2) * (kappa**4)
    omega = 0.5 * (np.emath.sqrt(disc) - 1j * params.b * (kappa**2))

    phase_velocity_ratio = np.ones_like(theta)
    nonzero = kappa > 0.0
    phase_velocity_ratio[nonzero] = np.real(omega[nonzero]) / (params.c * kappa[nonzero])

    amortissement_rate = -np.imag(omega)
    return {
        "omega": omega,
        "phase_velocity_ratio": phase_velocity_ratio,
        "amortissement_rate": amortissement_rate,
    }


def build_analysis_cases() -> list[FourierCurve]:
    """Cas representatifs pour visualiser l'effet de la diffusion."""
    return [
        FourierCurve(cfl=0.25, D=0.00, label="D = 0", color="#1f77b4"),
        FourierCurve(cfl=0.25, D=0.02, label="D = 0.02", color="#ff7f0e"),
        FourierCurve(cfl=0.25, D=0.05, label="D = 0.05", color="#2ca02c"),
    ]


def build_reference_params() -> WesterveltParams:
    """Parametres de reference utilises pour l'echelle physique des courbes."""
    return WesterveltParams(
        c=1500.0,
        rho0=1000.0,
        beta=4.8,
        mu_v=1e-3,
        dx=1e-4,
        dt=1.67e-8,
        nx=200,
        nt=1,
        scheme="explicit",
        bc="dirichlet",
    )


def _effective_dt_and_b(params: WesterveltParams, cfl: float, D: float) -> tuple[float, float]:
    """Convertit (CFL, nu) en (dt_effectif, b_effectif) pour les criteres theoriques."""
    dt_eff = cfl * params.dx / params.c
    if dt_eff <= 0.0:
        raise ValueError("Le CFL doit etre strictement positif pour evaluer la stabilite.")
    b_eff = D * params.dx**2 / dt_eff
    return float(dt_eff), float(b_eff)


def compute_case_stability_diagnostics(
    theta: np.ndarray,
    params: WesterveltParams,
    cfl: float,
    D: float,
) -> dict[str, float | bool]:
    """Calcule les nouveaux criteres et le diagnostic observe pour un cas Fourier."""
    dt_eff, b_eff = _effective_dt_and_b(params, cfl, D)
    margin = explicit_stability_margin(dt_eff, params.dx, params.c, b_eff)
    theoretical_stable = explicit_theoretical_stable(dt_eff, params.dx, params.c, b_eff)

    response = compute_numerical_response(theta, params, cfl, D)
    rho_max = float(np.max(response["spectral_radius"]))
    observed_stable = bool(rho_max <= 1.0 + STABILITY_TOL)

    return {
        "cfl": float(cfl),
        "D": float(D),
        "stability_margin": float(margin),
        "reduced_margin": float(1.0 - cfl**2 - 2.0 * D),
        "theoretical_stable": bool(theoretical_stable),
        "spectral_radius_max": rho_max,
        "observed_stable": observed_stable,
        "theory_observed_match": bool(theoretical_stable == observed_stable),
    }


def _save_figure(fig: plt.Figure, filename: str, metadata: dict[str, object]) -> dict[str, Path]:
    """Sauvegarde une figure avec metadonnees versionnees."""
    return save_figure_with_version(
        fig,
        filename=filename,
        output_dir=str(FOURIER_OUTPUT_DIR),
        formats=["png", "pdf"],
        metadata=metadata,
        tight_layout=False,
    )


def plot_amplification_and_dispersion(
    theta: np.ndarray,
    params: WesterveltParams,
    cases: Iterable[FourierCurve],
    show: bool = False,
    savefig: bool = True,
) -> tuple[plt.Figure, dict[str, Path]]:
    """Trace le module du facteur d'amplification et la dispersion numerique."""
    cases = list(cases)
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    stability_summary = []

    for case in cases:
        response = compute_numerical_response(theta, params, case.cfl, case.D)
        diag = compute_case_stability_diagnostics(theta, params, case.cfl, case.D)
        stability_summary.append(diag)
        dt_eff, _ = _effective_dt_and_b(params, case.cfl, case.D)

        exact_params = WesterveltParams(
            c=params.c,
            rho0=params.rho0,
            beta=params.beta,
            mu_v=case.D * params.dx**2 * params.rho0 / dt_eff,
            dx=params.dx,
            dt=dt_eff,
            nx=params.nx,
            nt=params.nt,
            scheme=params.scheme,
            bc=params.bc,
        )
        exact = compute_exact_continuous_response(theta, exact_params)

        status = "stable" if diag["theoretical_stable"] else "instable"
        label = f"{case.label} ({status} theo)"
        axs[0].plot(theta, np.abs(response["physical_root"]), color=case.color, linewidth=2.0, label=label)
        axs[0].plot(theta, response["spectral_radius"], color=case.color, linestyle="--", alpha=0.55)

        axs[1].plot(theta, response["phase_velocity_ratio"], color=case.color, linewidth=2.0, label=label)
        axs[1].plot(theta, exact["phase_velocity_ratio"], color=case.color, linestyle="--", alpha=0.55)

    axs[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="|G| = 1")
    axs[0].set_ylabel("Module du facteur d'amplification")
    axs[0].set_title("Schema explicite : amplification numerique")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="Dispersion nulle")
    axs[1].set_xlabel(r"Nombre d'onde reduit $\theta = k\Delta x$")
    axs[1].set_ylabel(r"Vitesse de phase numerique / $c$")
    axs[1].set_title("Schema explicite : dispersion numerique")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    metadata = {
        "analysis": "Fourier explicite",
        "reference_c": params.c,
        "reference_dx": params.dx,
        "reference_dt": params.dt,
        "reference_b": params.b,
        "cfl_values": [case.cfl for case in cases],
        "D_values": [case.D for case in cases],
        "stability_summary": stability_summary,
        "stability_criteria": ["stability_margin", "theoretical_stable", "spectral_radius_max"],
        "note": "Les courbes pointillees montrent le module spectral et la dispersion exacte continue.",
    }
    paths = {}
    if savefig:
        paths = _save_figure(fig, "analyse_fourier_explicite_amplification_dispersion", metadata)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, paths


def plot_amplification_vs_dispersion(
    theta: np.ndarray,
    params: WesterveltParams,
    cases: Iterable[FourierCurve],
    show: bool = False,
    savefig: bool = True,
) -> tuple[plt.Figure, dict[str, Path]]:
    """Trace le lien entre amplification et dispersion sous forme parametrique."""
    cases = list(cases)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    stability_summary = []

    for case in cases:
        response = compute_numerical_response(theta, params, case.cfl, case.D)
        diag = compute_case_stability_diagnostics(theta, params, case.cfl, case.D)
        stability_summary.append(diag)

        x = response["phase_velocity_ratio"]
        y = np.abs(response["physical_root"])
        status = "stable" if diag["theoretical_stable"] else "instable"
        ax.plot(x, y, color=case.color, linewidth=2.0, label=f"{case.label} ({status} theo)")
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], color=case.color, s=25)

    ax.scatter([1.0], [1.0], color="black", marker="x", s=80, label="Reference ideale (1, 1)")
    ax.set_xlabel(r"Vitesse de phase numerique / $c$")
    ax.set_ylabel(r"$|G|$")
    ax.set_title("Amplification vs dispersion")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    metadata = {
        "analysis": "Amplification vs dispersion",
        "reference_c": params.c,
        "reference_dx": params.dx,
        "reference_dt": params.dt,
        "reference_b": params.b,
        "cfl_values": [case.cfl for case in cases],
        "D_values": [case.D for case in cases],
        "stability_summary": stability_summary,
    }
    paths = {}
    if savefig:
        paths = _save_figure(fig, "analyse_fourier_explicite_amplification_vs_dispersion", metadata)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, paths


def plot_diffusion_effect(
    theta: np.ndarray,
    params: WesterveltParams,
    cfl: float,
    D_values: Iterable[float],
    show: bool = False,
    savefig: bool = True,
) -> tuple[plt.Figure, dict[str, Path]]:
    """Mets en evidence l'effet de la diffusion sur l'amplification et l'amortissement."""
    paths = {}
    D_values = list(D_values)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    stability_summary = []

    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(D_values)))
    for D, color in zip(D_values, colors):
        response = compute_numerical_response(theta, params, cfl, D)
        diag = compute_case_stability_diagnostics(theta, params, cfl, D)
        stability_summary.append(diag)
        dt_eff, _ = _effective_dt_and_b(params, cfl, D)

        exact_params = WesterveltParams(
            c=params.c,
            rho0=params.rho0,
            beta=params.beta,
            mu_v=D * params.dx**2 * params.rho0 / dt_eff,
            dx=params.dx,
            dt=dt_eff,
            nx=params.nx,
            nt=params.nt,
            scheme=params.scheme,
            bc=params.bc,
        )
        exact = compute_exact_continuous_response(theta, exact_params)

        state = "stable" if diag["theoretical_stable"] else "instable"
        label = rf"$D={D:.3f}$ ({state} theo)"
        axs[0].plot(theta, np.abs(response["physical_root"]), color=color, linewidth=2.0, label=label)
        axs[1].plot(theta, response["amortissement_rate"], color=color, linewidth=2.0, label=f"num {label}")
        axs[1].plot(theta, exact["amortissement_rate"], color=color, linestyle="--", alpha=0.6, label=f"exact {label}")

    axs[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axs[0].set_xlabel(r"Nombre d'onde reduit $\theta = k\Delta x$")
    axs[0].set_ylabel(r"$|G|$")
    axs[0].set_title("Effet de la diffusion sur le module de G")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].set_xlabel(r"Nombre d'onde reduit $\theta = k\Delta x$")
    axs[1].set_ylabel(r"Taux d'amortissement $\alpha$ (s$^{-1}$)")
    axs[1].set_title("Diffusion: amortissement des hautes frequences")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize=8)

    metadata = {
        "analysis": "Effet de la diffusion",
        "reference_c": params.c,
        "reference_dx": params.dx,
        "reference_dt": params.dt,
        "reference_b": params.b,
        "cfl": cfl,
        "D_values": D_values,
        "stability_summary": stability_summary,
    }
    if savefig:
        paths = _save_figure(fig, "analyse_fourier_explicite_effet_diffusion", metadata)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, paths


def main(show: bool = False, savefig: bool = False) -> dict[str, dict[str, Path]]:
    """Point d'entree du script."""
    set_style()
    params = build_reference_params()
    theta = _theta_grid(600)
    cases = build_analysis_cases()

    print("Analyse Fourier du schema explicite")
    print(f"  c = {params.c:.3g} m/s")
    print(f"  dx = {params.dx:.3g} m")
    print(f"  dt = {params.dt:.3g} s")
    print(f"  b = {params.b:.3g} m^2/s")
    print(f"  CFL de reference = {cases[0].cfl:.3f}")
    print(f"  Cas de diffusion = {[case.D for case in cases]}")

    print("Nouveaux criteres de stabilite (schema explicite):")
    for case in cases:
        diag = compute_case_stability_diagnostics(theta, params, case.cfl, case.D)
        print(
            "  "
            f"{case.label}: margin={diag['stability_margin']:.3e}, "
            f"theoretical_stable={diag['theoretical_stable']}, "
            f"rho_max={diag['spectral_radius_max']:.6f}, "
            f"observed_stable={diag['observed_stable']}"
        )

    _, paths1 = plot_amplification_and_dispersion(theta, params, cases, show=show, savefig=savefig)
    _, paths2 = plot_amplification_vs_dispersion(theta, params, cases, show=show, savefig=savefig)
    _, paths3 = plot_diffusion_effect(
        theta,
        params,
        cfl=cases[0].cfl,
        D_values=[0.0, 0.02, 0.05],
        show=show,
        savefig=savefig,
    )

    if savefig:
        print(f"Figures enregistrees dans {FOURIER_OUTPUT_DIR}")
        for label, paths in {
            "amplification_dispersion": paths1,
            "amplification_vs_dispersion": paths2,
            "effet_diffusion": paths3,
        }.items():
            print(f"  - {label}: {paths}")
    else:
        print("Enregistrement des figures desactive (savefig=False).")

    return {
        "amplification_dispersion": paths1,
        "amplification_vs_dispersion": paths2,
        "effet_diffusion": paths3,
    }


if __name__ == "__main__":
    main(show=True, savefig=False)

