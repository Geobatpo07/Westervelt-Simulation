"""Analyse de Fourier du schéma explicite de Westervelt.

Ce script réalise une analyse de stabilité/dispersion dans le régime linéarisé
(k ≈ 0) et sur une grille périodique. On considère un mode de Fourier
u_i^n = G^n exp(i i θ), avec θ = k Δx.

Pour le schéma explicite, on obtient la récurrence :

	u^{n+1} = [2 - σ( CFL^2 + ν )] u^n + [ -1 + σ ν ] u^{n-1}

avec :
	σ = 4 sin^2(θ/2)
	CFL = c Δt / Δx
	ν   = b Δt / Δx^2

Le facteur d'amplification G est donné par les racines de :

	G^2 - [2 - σ(CFL^2 + ν)] G + [1 - σν] = 0

Le script produit trois graphiques :
1. module du facteur d'amplification et dispersion numérique,
2. amplification vs dispersion,
3. effet de la diffusion via le taux d'atténuation.

Les figures sont enregistrées dans `outputs/analysis/` avec versioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.solver import WesterveltParams
from utils import save_figure_with_version, set_style


@dataclass(frozen=True)
class FourierCurve:
	"""Courbe Fourier pour un couple (CFL, diffusion) donné."""

	cfl: float
	diffusion_number: float
	label: str
	color: str


def _theta_grid(num_points: int = 500) -> np.ndarray:
	"""Grille de nombres d'onde réduits θ ∈ [0, π]."""
	return np.linspace(0.0, np.pi, num_points)


def _sigma(theta: np.ndarray) -> np.ndarray:
	"""Symbole du laplacien 1D centré, en version réduite."""
	return 4.0 * np.sin(theta / 2.0) ** 2


def compute_quadratic_roots(theta: np.ndarray, cfl: float, diffusion_number: float) -> np.ndarray:
	"""Calcule les deux racines du facteur d'amplification pour chaque θ.

	Args:
		theta: tableau des nombres d'onde réduits.
		cfl: nombre CFL = c Δt / Δx.
		diffusion_number: ν = b Δt / Δx².

	Returns:
		Array complexe de forme (N, 2) contenant les deux racines.
	"""
	sigma = _sigma(theta)
	a = 2.0 - sigma * (cfl**2 + diffusion_number)
	b = 1.0 - sigma * diffusion_number
	disc = a**2 - 4.0 * b
	sqrt_disc = np.lib.scimath.sqrt(disc)
	g_plus = 0.5 * (a + sqrt_disc)
	g_minus = 0.5 * (a - sqrt_disc)
	return np.stack([g_plus, g_minus], axis=-1)


def select_physical_branch(roots: np.ndarray) -> np.ndarray:
	"""Sélectionne la branche physique par continuité depuis θ = 0.

	On choisit à chaque pas la racine la plus proche de la racine précédente.
	Cela évite les sauts de branche quand les racines deviennent complexes.
	"""
	physical = np.empty(roots.shape[0], dtype=complex)
	previous = 1.0 + 0.0j

	for i, pair in enumerate(roots):
		idx = int(np.argmin(np.abs(pair - previous)))
		physical[i] = pair[idx]
		previous = physical[i]

	return physical


def compute_numerical_response(
	theta: np.ndarray,
	cfl: float,
	diffusion_number: float,
	dt: float,
	dx: float,
) -> dict[str, np.ndarray]:
	"""Retourne les indicateurs numériques du schéma explicite."""
	roots = compute_quadratic_roots(theta, cfl, diffusion_number)
	g_phys = select_physical_branch(roots)
	spectral_radius = np.max(np.abs(roots), axis=-1)

	phase = np.unwrap(np.angle(g_phys))
	omega_num = -phase / dt
	kappa = theta / dx

	phase_velocity_ratio = np.ones_like(theta)
	nonzero = kappa > 0.0
	phase_velocity_ratio[nonzero] = omega_num[nonzero] / (kappa[nonzero] * (cfl * dx / dt))

	attenuation_rate = -np.log(np.maximum(np.abs(g_phys), 1e-15)) / dt
	return {
		"roots": roots,
		"physical_root": g_phys,
		"spectral_radius": spectral_radius,
		"phase_velocity_ratio": phase_velocity_ratio,
		"attenuation_rate": attenuation_rate,
	}


def compute_exact_continuous_response(theta: np.ndarray, params: WesterveltParams) -> dict[str, np.ndarray]:
	"""Réponse analytique de l'équation linéarisée continue.

	Pour u_tt - c² u_xx - b u_xxt = 0, on obtient :
		ω = 1/2 * (sqrt(4 c² κ² - b² κ^4) - i b κ²)

	avec κ = θ/Δx.
	"""
	kappa = theta / params.dx
	disc = 4.0 * (params.c**2) * (kappa**2) - (params.b**2) * (kappa**4)
	omega = 0.5 * (np.lib.scimath.sqrt(disc) - 1j * params.b * (kappa**2))

	phase_velocity_ratio = np.ones_like(theta)
	nonzero = kappa > 0.0
	phase_velocity_ratio[nonzero] = np.real(omega[nonzero]) / (params.c * kappa[nonzero])

	attenuation_rate = -np.imag(omega)
	return {
		"omega": omega,
		"phase_velocity_ratio": phase_velocity_ratio,
		"attenuation_rate": attenuation_rate,
	}


def build_analysis_cases() -> list[FourierCurve]:
	"""Cas représentatifs pour visualiser l'effet de la diffusion."""
	return [
		FourierCurve(cfl=0.25, diffusion_number=0.00, label="ν = 0", color="#1f77b4"),
		FourierCurve(cfl=0.25, diffusion_number=0.02, label="ν = 0.02", color="#ff7f0e"),
		FourierCurve(cfl=0.25, diffusion_number=0.05, label="ν = 0.05", color="#2ca02c"),
	]


def build_reference_params() -> WesterveltParams:
	"""Paramètres de référence utilisés pour l'échelle physique des courbes."""
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


def _save_figure(fig: plt.Figure, filename: str, metadata: dict[str, object]) -> dict[str, Path]:
	"""Sauvegarde une figure avec métadonnées versionnées."""
	return save_figure_with_version(
		fig,
		filename=filename,
		output_dir="outputs/analysis",
		formats=["png", "pdf"],
		metadata=metadata,
		tight_layout=False,
	)


def plot_amplification_and_dispersion(
	theta: np.ndarray,
	params: WesterveltParams,
	cases: Iterable[FourierCurve],
	show: bool = False,
) -> tuple[plt.Figure, dict[str, Path]]:
	"""Trace le module du facteur d'amplification et la dispersion numérique."""
	fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

	for case in cases:
		response = compute_numerical_response(theta, case.cfl, case.diffusion_number, params.dt, params.dx)
		exact_params = WesterveltParams(
			c=params.c,
			rho0=params.rho0,
			beta=params.beta,
			mu_v=case.diffusion_number * params.dx**2 * params.rho0 / params.dt,
			dx=params.dx,
			dt=params.dt,
			nx=params.nx,
			nt=params.nt,
			scheme=params.scheme,
			bc=params.bc,
		)
		exact = compute_exact_continuous_response(theta, exact_params)

		axs[0].plot(theta, np.abs(response["physical_root"]), color=case.color, linewidth=2.0, label=case.label)
		axs[0].plot(theta, response["spectral_radius"], color=case.color, linestyle="--", alpha=0.55)

		axs[1].plot(theta, response["phase_velocity_ratio"], color=case.color, linewidth=2.0, label=case.label)
		axs[1].plot(theta, exact["phase_velocity_ratio"], color=case.color, linestyle="--", alpha=0.55)

	axs[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="|G| = 1")
	axs[0].set_ylabel("Module du facteur d'amplification")
	axs[0].set_title("Schéma explicite : amplification numérique")
	axs[0].grid(True, alpha=0.3)
	axs[0].legend(loc="best")

	axs[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="Dispersion nulle")
	axs[1].set_xlabel(r"Nombre d'onde réduit $\theta = k\Delta x$")
	axs[1].set_ylabel(r"Vitesse de phase numérique / $c$")
	axs[1].set_title("Schéma explicite : dispersion numérique")
	axs[1].grid(True, alpha=0.3)
	axs[1].legend(loc="best")

	metadata = {
		"analysis": "Fourier explicite",
		"reference_c": params.c,
		"reference_dx": params.dx,
		"reference_dt": params.dt,
		"reference_b": params.b,
		"cfl_values": [case.cfl for case in cases],
		"diffusion_numbers": [case.diffusion_number for case in cases],
		"note": "Les courbes pointillées montrent le module spectral et la dispersion exacte continue.",
	}
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
) -> tuple[plt.Figure, dict[str, Path]]:
	"""Trace le lien entre amplification et dispersion sous forme paramétrique."""
	fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)

	for case in cases:
		response = compute_numerical_response(theta, case.cfl, case.diffusion_number, params.dt, params.dx)
		x = response["phase_velocity_ratio"]
		y = np.abs(response["physical_root"])
		ax.plot(x, y, color=case.color, linewidth=2.0, label=case.label)
		ax.scatter([x[0], x[-1]], [y[0], y[-1]], color=case.color, s=25)

	ax.scatter([1.0], [1.0], color="black", marker="x", s=80, label="Référence idéale (1, 1)")
	ax.set_xlabel(r"Vitesse de phase numérique / $c$")
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
		"diffusion_numbers": [case.diffusion_number for case in cases],
	}
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
	diffusion_numbers: Iterable[float],
	show: bool = False,
) -> tuple[plt.Figure, dict[str, Path]]:
	"""Mets en évidence l'effet de la diffusion sur l'amplification et l'atténuation."""
	diffusion_numbers = list(diffusion_numbers)
	fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

	colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(diffusion_numbers)))
	for diffusion_number, color in zip(diffusion_numbers, colors):
		response = compute_numerical_response(theta, cfl, diffusion_number, params.dt, params.dx)
		exact_params = WesterveltParams(
			c=params.c,
			rho0=params.rho0,
			beta=params.beta,
			mu_v=diffusion_number * params.dx**2 * params.rho0 / params.dt,
			dx=params.dx,
			dt=params.dt,
			nx=params.nx,
			nt=params.nt,
			scheme=params.scheme,
			bc=params.bc,
		)
		exact = compute_exact_continuous_response(theta, exact_params)

		label = rf"$\nu={diffusion_number:.3f}$"
		axs[0].plot(theta, np.abs(response["physical_root"]), color=color, linewidth=2.0, label=label)
		axs[1].plot(theta, response["attenuation_rate"], color=color, linewidth=2.0, label=f"num {label}")
		axs[1].plot(theta, exact["attenuation_rate"], color=color, linestyle="--", alpha=0.6, label=f"exact {label}")

	axs[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
	axs[0].set_xlabel(r"Nombre d'onde réduit $\theta = k\Delta x$")
	axs[0].set_ylabel(r"$|G|$")
	axs[0].set_title("Effet de la diffusion sur le module de G")
	axs[0].grid(True, alpha=0.3)
	axs[0].legend(loc="best")

	axs[1].set_xlabel(r"Nombre d'onde réduit $\theta = k\Delta x$")
	axs[1].set_ylabel(r"Taux d'atténuation $\alpha$ (s$^{-1}$)")
	axs[1].set_title("Diffusion: atténuation des hautes fréquences")
	axs[1].grid(True, alpha=0.3)
	axs[1].legend(loc="best", fontsize=8)

	metadata = {
		"analysis": "Effet de la diffusion",
		"reference_c": params.c,
		"reference_dx": params.dx,
		"reference_dt": params.dt,
		"reference_b": params.b,
		"cfl": cfl,
		"diffusion_numbers": diffusion_numbers,
	}
	paths = _save_figure(fig, "analyse_fourier_explicite_effet_diffusion", metadata)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return fig, paths


def main(show: bool = False) -> dict[str, dict[str, Path]]:
	"""Point d'entrée du script."""
	set_style()
	params = build_reference_params()
	theta = _theta_grid(600)
	cases = build_analysis_cases()

	print("Analyse Fourier du schéma explicite")
	print(f"  c = {params.c:.3g} m/s")
	print(f"  dx = {params.dx:.3g} m")
	print(f"  dt = {params.dt:.3g} s")
	print(f"  b = {params.b:.3g} m²/s")
	print(f"  CFL de référence = {cases[0].cfl:.3f}")
	print(f"  Cas de diffusion = {[case.diffusion_number for case in cases]}")

	_, paths1 = plot_amplification_and_dispersion(theta, params, cases, show=show)
	_, paths2 = plot_amplification_vs_dispersion(theta, params, cases, show=show)
	_, paths3 = plot_diffusion_effect(theta, params, cfl=cases[0].cfl, diffusion_numbers=[0.0, 0.02, 0.05], show=show)

	print("Figures enregistrées dans outputs/analysis/")
	for label, paths in {
		"amplification_dispersion": paths1,
		"amplification_vs_dispersion": paths2,
		"effet_diffusion": paths3,
	}.items():
		print(f"  - {label}: {paths}")

	return {
		"amplification_dispersion": paths1,
		"amplification_vs_dispersion": paths2,
		"effet_diffusion": paths3,
	}


if __name__ == "__main__":
	main(show=False)





