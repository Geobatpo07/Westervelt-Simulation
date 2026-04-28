"""
Analyse de Fourier du schema semi-implicite de Westervelt.

Ce script realise une analyse de stabilite, de dispersion et d'amortissement
dans le regime linearise par gel de alpha sur une grille periodique.

On considere un mode de Fourier de la forme :
    u_i^n = G^n exp(i i theta),
ou theta = kappa * dx est le nombre d'onde reduit.

Pour le schema semi-implicite (gel de alpha), on utilise directement la
matrice d'amplification deja implementee dans src.stability_analysis.

Le script s'appuie sur la matrice d'amplification pour :
    - calculer le rayon spectral (critere de stabilite),
    - selectionner la branche physique du facteur d'amplification,
    - comparer la reponse numerique a la reponse analytique du modele continu linearise gele.

Le mode nul (theta = 0, donc mu = 0) est traite separement, conformement
a l'analyse theorique : il est neutre (rho(A(0)) = 1), mais ne releve pas
de la meme lecture dispersion/amortissement que les modes non nuls.

Trois types de graphiques sont produits :
1. Module du facteur d'amplification et dispersion numerique en fonction de theta,
2. Representation parametrique du compromis amplification / dispersion,
3. Effet du nombre de diffusion sur l'amortissement des modes de Fourier.

Le taux d'amortissement est defini par :
    alpha_num(theta) = -log(|G(theta)|) / dt

Les figures sont enregistrees dans :
    outputs/analysis/semi-implicit-fourier-analysis
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

FOURIER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "semi-implicit-fourier-analysis"
ALPHA_LINEARIZED = 0.1
STABILITY_TOL = 1e-10

from src.solver import WesterveltParams
from src.stability_analysis import (
    amplification_matrix_semi_implicite,
    discrete_mu,
    semi_implicit_stability_margin,
    semi_implicit_theoretical_stable,
)
from utils import save_figure_with_version, set_style


@dataclass(frozen=True)
class FourierCurve:
    """Courbe Fourier pour un couple (CFL, D) donne."""

    cfl: float
    D: float
    label: str
    color: str


def _theta_grid(num_points: int = 500) -> np.ndarray:
    """Grille de nombres d'onde reduits theta dans [0, pi]."""
    return np.linspace(0.0, np.pi, num_points)


def _effective_dt_and_b(params: WesterveltParams, cfl: float, D: float) -> tuple[float, float]:
    """Convertit (CFL, D) en (dt_effectif, b_effectif)."""
    dt_eff = cfl * params.dx / params.c
    if dt_eff <= 0.0:
        raise ValueError("Le CFL doit etre strictement positif pour evaluer la stabilite.")
    b_eff = D * params.dx**2 / dt_eff
    return float(dt_eff), float(b_eff)


def analyze_zero_mode(dt: float, alpha: float) -> dict[str, float | bool | str]:
    """
    Analyse du mode nul mu = 0 pour le schema semi-implicite.

    Pour mu = 0, la matrice d'amplification theorique vaut :
        A(0) = [[1, tau], [0, 1]], tau = dt / alpha
    donc rho(A(0)) = 1, mais A(0) n'est pas diagonalisable.
    """
    tau = dt / alpha
    return {
        "tau": float(tau),
        "spectral_radius": 1.0,
        "neutral_mode": True,
        "diagonalisable": False,
        "comment": (
            "Le mode nul est neutre (rho=1), mais la matrice d'amplification "
            "n'est pas diagonalisable. Une croissance lineaire en temps peut apparaitre "
            "si la composante moyenne initiale de F n'est pas nulle."
        ),
    }


def select_physical_branch_from_matrix(
    theta: np.ndarray,
    params: WesterveltParams,
    dt: float,
    b: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selectionne une branche physique continue parmi les valeurs propres
    pour les modes non nuls, et traite separement le mode nul.
    """
    mu_values = discrete_mu(theta, params.dx)

    physical = np.empty_like(theta, dtype=complex)
    rho = np.empty_like(theta, dtype=float)

    # Mode nul : traitement separe
    physical[0] = 1.0 + 0.0j
    rho[0] = 1.0

    previous = physical[0]

    # Modes non nuls : analyse spectrale
    for i, mu in enumerate(mu_values[1:], start=1):
        A = amplification_matrix_semi_implicite(float(mu), dt, params.c, b, alpha=alpha)
        eigenvalues = np.linalg.eigvals(A)
        rho[i] = np.max(np.abs(eigenvalues))

        # Selection par continuite de branche
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
    alpha: float = ALPHA_LINEARIZED,
) -> dict[str, np.ndarray]:
    """Retourne les indicateurs numeriques via la matrice d'amplification semi-implicite."""
    dt_eff, b_eff = _effective_dt_and_b(params, cfl, D)

    g_phys, rho = select_physical_branch_from_matrix(theta, params, dt_eff, b_eff, alpha=alpha)

    phase = np.unwrap(np.angle(g_phys))
    omega_num = phase / dt_eff
    kappa = theta / params.dx

    phase_velocity_ratio = np.full_like(theta, np.nan, dtype=float)
    amortissement_rate = np.full_like(theta, np.nan, dtype=float)

    # Mode nul traite a part
    phase_velocity_ratio[0] = 1.0 / np.sqrt(alpha)
    amortissement_rate[0] = 0.0

    # Modes non nuls
    nonzero = kappa > 0.0
    phase_velocity_ratio[nonzero] = omega_num[nonzero] / (params.c * kappa[nonzero])
    amortissement_rate[nonzero] = -np.log(np.maximum(np.abs(g_phys[nonzero]), 1e-15)) / dt_eff

    return {
        "physical_root": g_phys,
        "spectral_radius": rho,
        "phase_velocity_ratio": phase_velocity_ratio,
        "amortissement_rate": amortissement_rate,
    }


def compute_exact_continuous_response(
    theta: np.ndarray,
    params: WesterveltParams,
    alpha: float = ALPHA_LINEARIZED,
) -> dict[str, np.ndarray]:
    """
    Reponse analytique de l'equation continue linearisee gelee :
        alpha * u_tt - c^2 u_xx - b u_xxt = 0
    """
    kappa = theta / params.dx
    disc = 4.0 * alpha * (params.c**2) * (kappa**2) - (params.b**2) * (kappa**4)
    omega = 0.5 / alpha * (np.emath.sqrt(disc) - 1j * params.b * (kappa**2))

    phase_velocity_ratio = np.full_like(theta, np.nan, dtype=float)
    amortissement_rate = np.full_like(theta, np.nan, dtype=float)

    phase_velocity_ratio[0] = 1.0 / np.sqrt(alpha)
    amortissement_rate[0] = 0.0

    nonzero = kappa > 0.0
    phase_velocity_ratio[nonzero] = np.real(omega[nonzero]) / (params.c * kappa[nonzero])
    amortissement_rate[nonzero] = -np.imag(omega[nonzero])

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
        scheme="semi_implicit",
        bc="dirichlet",
    )


def compute_case_stability_diagnostics(
    theta: np.ndarray,
    params: WesterveltParams,
    cfl: float,
    D: float,
    alpha: float = ALPHA_LINEARIZED,
) -> dict[str, float | bool]:
    """Calcule les criteres semi-implicites et le diagnostic observe pour un cas Fourier."""
    dt_eff, b_eff = _effective_dt_and_b(params, cfl, D)
    margin = semi_implicit_stability_margin(dt_eff, params.dx, params.c, b_eff, alpha=alpha)
    theoretical_stable = semi_implicit_theoretical_stable(dt_eff, params.dx, params.c, b_eff, alpha=alpha)

    response = compute_numerical_response(theta, params, cfl, D, alpha=alpha)
    rho_max = float(np.max(response["spectral_radius"][1:]))  # modes non nuls seulement
    observed_stable = bool(rho_max <= 1.0 + STABILITY_TOL)

    return {
        "cfl": float(cfl),
        "D": float(D),
        "alpha": float(alpha),
        "stability_margin": float(margin),
        "reduced_margin": float(alpha - cfl**2 + 2.0 * D),
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
    alpha: float = ALPHA_LINEARIZED,
    show: bool = False,
    savefig: bool = True,
) -> tuple[plt.Figure, dict[str, Path]]:
    """Trace le module du facteur d'amplification et la dispersion numerique."""
    cases = list(cases)
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    stability_summary = []

    for case in cases:
        response = compute_numerical_response(theta, params, case.cfl, case.D, alpha=alpha)
        diag = compute_case_stability_diagnostics(theta, params, case.cfl, case.D, alpha=alpha)
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
        exact = compute_exact_continuous_response(theta, exact_params, alpha=alpha)

        status = "stable" if diag["theoretical_stable"] else "instable"
        label = f"{case.label} ({status} theo)"
        axs[0].plot(theta, np.abs(response["physical_root"]), color=case.color, linewidth=2.0, label=label)
        axs[0].plot(theta, response["spectral_radius"], color=case.color, linestyle="--", alpha=0.55)

        axs[1].plot(theta, response["phase_velocity_ratio"], color=case.color, linewidth=2.0, label=label)
        axs[1].plot(theta, exact["phase_velocity_ratio"], color=case.color, linestyle="--", alpha=0.55)

    axs[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="|G| = 1")
    axs[0].set_ylabel("Module du facteur d'amplification")
    axs[0].set_title("Schema semi-implicite : amplification numerique")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].axhline(1.0 / np.sqrt(alpha), color="black", linestyle=":", linewidth=1.0, label="Reference basse frequence")
    axs[1].set_xlabel(r"Nombre d'onde reduit $\theta = \kappa \Delta x$")
    axs[1].set_ylabel(r"Vitesse de phase numerique / $c$")
    axs[1].set_title("Schema semi-implicite : dispersion numerique")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    metadata = {
        "analysis": "Fourier semi-implicite",
        "alpha": alpha,
        "reference_c": params.c,
        "reference_dx": params.dx,
        "reference_dt": params.dt,
        "reference_b": params.b,
        "cfl_values": [case.cfl for case in cases],
        "D_values": [case.D for case in cases],
        "stability_summary": stability_summary,
        "stability_criteria": ["stability_margin", "theoretical_stable", "spectral_radius_max"],
        "note": "Le mode nul est traite separement. Les courbes pointillees representent la reponse continue exacte gelee.",
    }
    paths = {}
    if savefig:
        paths = _save_figure(fig, "analyse_fourier_semi_implicite_amplification_dispersion", metadata)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, paths


def plot_amplification_vs_dispersion(
    theta: np.ndarray,
    params: WesterveltParams,
    cases: Iterable[FourierCurve],
    alpha: float = ALPHA_LINEARIZED,
    show: bool = False,
    savefig: bool = True,
) -> tuple[plt.Figure, dict[str, Path]]:
    """Trace le lien entre amplification et dispersion sous forme parametrique."""
    cases = list(cases)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    stability_summary = []

    for case in cases:
        response = compute_numerical_response(theta, params, case.cfl, case.D, alpha=alpha)
        diag = compute_case_stability_diagnostics(theta, params, case.cfl, case.D, alpha=alpha)
        stability_summary.append(diag)

        x = response["phase_velocity_ratio"]
        y = np.abs(response["physical_root"])
        status = "stable" if diag["theoretical_stable"] else "instable"
        ax.plot(x, y, color=case.color, linewidth=2.0, label=f"{case.label} ({status} theo)")
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], color=case.color, s=25)

    ax.scatter([1.0 / np.sqrt(alpha)], [1.0], color="black", marker="x", s=80, label="Reference ideale basse frequence")
    ax.set_xlabel(r"Vitesse de phase numerique / $c$")
    ax.set_ylabel(r"$|G|$")
    ax.set_title("Compromis amplification / dispersion")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    metadata = {
        "analysis": "Amplification vs dispersion",
        "alpha": alpha,
        "reference_c": params.c,
        "reference_dx": params.dx,
        "reference_dt": params.dt,
        "reference_b": params.b,
        "cfl_values": [case.cfl for case in cases],
        "D_values": [case.D for case in cases],
        "stability_summary": stability_summary,
        "note": "Le mode nul est traite separement.",
    }
    paths = {}
    if savefig:
        paths = _save_figure(fig, "analyse_fourier_semi_implicite_amplification_vs_dispersion", metadata)

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
    alpha: float = ALPHA_LINEARIZED,
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
        response = compute_numerical_response(theta, params, cfl, D, alpha=alpha)
        diag = compute_case_stability_diagnostics(theta, params, cfl, D, alpha=alpha)
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
        exact = compute_exact_continuous_response(theta, exact_params, alpha=alpha)

        state = "stable" if diag["theoretical_stable"] else "instable"
        label = rf"$D={D:.3f}$ ({state} theo)"
        axs[0].plot(theta, np.abs(response["physical_root"]), color=color, linewidth=2.0, label=label)
        axs[1].plot(theta, response["amortissement_rate"], color=color, linewidth=2.0, label=f"num {label}")
        axs[1].plot(theta, exact["amortissement_rate"], color=color, linestyle="--", alpha=0.6, label=f"exact {label}")

    axs[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axs[0].set_xlabel(r"Nombre d'onde reduit $\theta = \kappa \Delta x$")
    axs[0].set_ylabel(r"$|G|$")
    axs[0].set_title("Effet de la diffusion sur le module de G")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].set_xlabel(r"Nombre d'onde reduit $\theta = \kappa \Delta x$")
    axs[1].set_ylabel(r"Taux d'amortissement $\alpha$ (s$^{-1}$)")
    axs[1].set_title("Diffusion : amortissement des hautes frequences")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize=8)

    metadata = {
        "analysis": "Effet de la diffusion",
        "alpha": alpha,
        "reference_c": params.c,
        "reference_dx": params.dx,
        "reference_dt": params.dt,
        "reference_b": params.b,
        "cfl": cfl,
        "D_values": D_values,
        "stability_summary": stability_summary,
        "note": "Le mode nul est traite separement.",
    }
    if savefig:
        paths = _save_figure(fig, "analyse_fourier_semi_implicite_effet_diffusion", metadata)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, paths


def main(
    show: bool = False,
    savefig: bool = False,
    alpha: float = ALPHA_LINEARIZED,
) -> dict[str, dict[str, Path]]:
    """Point d'entree du script."""
    set_style()
    params = build_reference_params()
    theta = _theta_grid(600)
    cases = build_analysis_cases()

    print("Analyse Fourier du schema semi-implicite")
    print(f"  alpha (gel lineaire) = {alpha:.3f}")
    print(f"  c = {params.c:.3g} m/s")
    print(f"  dx = {params.dx:.3g} m")
    print(f"  dt = {params.dt:.3g} s")
    print(f"  b = {params.b:.3g} m^2/s")
    print(f"  CFL de reference = {cases[0].cfl:.3f}")
    print(f"  Cas de diffusion = {[case.D for case in cases]}")

    zero_mode_info = analyze_zero_mode(
        dt=_effective_dt_and_b(params, cases[0].cfl, cases[0].D)[0],
        alpha=alpha,
    )

    print("Mode nul (mu = 0) :")
    print(f"  tau = {zero_mode_info['tau']:.3e}")
    print(f"  rho(A(0)) = {zero_mode_info['spectral_radius']:.3f}")
    print(f"  diagonalisable = {zero_mode_info['diagonalisable']}")
    print(f"  commentaire : {zero_mode_info['comment']}")

    print("Criteres de stabilite (modes non nuls) :")
    for case in cases:
        diag = compute_case_stability_diagnostics(theta, params, case.cfl, case.D, alpha=alpha)
        print(
            "  "
            f"{case.label}: margin={diag['stability_margin']:.3e}, "
            f"theoretical_stable={diag['theoretical_stable']}, "
            f"rho_max={diag['spectral_radius_max']:.6f}, "
            f"observed_stable={diag['observed_stable']}"
        )

    _, paths1 = plot_amplification_and_dispersion(theta, params, cases, alpha=alpha, show=show, savefig=savefig)
    _, paths2 = plot_amplification_vs_dispersion(theta, params, cases, alpha=alpha, show=show, savefig=savefig)
    _, paths3 = plot_diffusion_effect(
        theta,
        params,
        cfl=cases[0].cfl,
        D_values=[0.0, 0.02, 0.05],
        alpha=alpha,
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