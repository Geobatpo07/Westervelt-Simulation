"""
Analyse des simulations de Westervelt inspirees du cas Nikolic-Wohlmuth.

Ce script reprend les parametres du notebook `notebooks/test_solver_nikolic.ipynb`
et produit une analyse reproductible du cas 1D :

1. snapshots de la solution u(x,t),
2. energie discrete et indicateurs de non-degenerescence,
3. carte de stabilite observee pour le balayage (dt, amplitude u0).

Les figures peuvent etre enregistrees dans :
    outputs/analysis/nikolic-simulation-analysis
avec versioning automatique.

Execution typique :
    python experiments/analyse_nikolic_simulations.py --savefig --no-show

Execution rapide :
    python experiments/analyse_nikolic_simulations.py --quick --savefig --no-show
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "nikolic-simulation-analysis"

from core.solver import WesterveltParams, WesterveltSolver
from utils import build_scan_grid, get_scan_axes, save_data_with_version, save_figure_with_version, set_style


@dataclass(frozen=True)
class NikolicConfig:
    """Parametres physiques, numeriques et initiaux du cas Nikolic."""

    c: float = 1500.0
    rho0: float = 1000.0
    beta: float = 3.5
    mu_v: float = 6e-6
    length: float = 0.2
    nx: int = 2000
    t_final: float = 37e-6
    nt: int = 2000
    scheme: str = "explicit"
    bc: str = "dirichlet"

    u0_type: str = "gaussian"
    u1_type: str = "gaussian_derivative"
    amplitude_u0: float = 1.2e8
    amplitude_u1: float = 1.0e11
    mu: float = 0.1
    sigma1: float = 0.015
    sigma2: float = 0.02

    blowup_threshold: float = 1e10

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dt(self) -> float:
        return self.t_final / self.nt


SNAPSHOT_TIMES = (0.0, 5e-6, 9.25e-6, 15e-6, 20e-6, 25e-6, 30e-6, 37e-6)
DT_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0)
AMPLITUDE_VALUES = (0.5e8, 0.8e8, 1.2e8, 1.6e8, 2.0e8)


def make_config(quick: bool = False, scheme: str = "explicit") -> NikolicConfig:
    """
    Construit la configuration de base.

    Le mode quick conserve les memes echelles physiques et initiales, mais reduit
    nx et nt pour verifier rapidement le pipeline de figures.
    """
    if quick:
        return NikolicConfig(nx=400, nt=300, t_final=37e-6, scheme=scheme)
    return NikolicConfig(scheme=scheme)


def make_params(config: NikolicConfig, *, dt: float | None = None, amplitude_scheme: str | None = None) -> WesterveltParams:
    return WesterveltParams(
        c=config.c,
        rho0=config.rho0,
        beta=config.beta,
        mu_v=config.mu_v,
        dx=config.dx,
        dt=config.dt if dt is None else float(dt),
        nx=config.nx,
        nt=config.nt,
        scheme=config.scheme if amplitude_scheme is None else amplitude_scheme,
        bc=config.bc,
    )


def initialize_solver(params: WesterveltParams, config: NikolicConfig, amplitude_u0: float | None = None) -> WesterveltSolver:
    with contextlib.redirect_stdout(io.StringIO()):
        solver = WesterveltSolver(params)
        solver.initialize(
            u0_type=config.u0_type,
            u1_type=config.u1_type,
            A1=config.amplitude_u0 if amplitude_u0 is None else float(amplitude_u0),
            A2=config.amplitude_u1,
            mu=config.mu,
            sigma1=config.sigma1,
            sigma2=config.sigma2,
        )
    return solver


def run_nikolic_simulation(config: NikolicConfig, snapshot_times: tuple[float, ...] = SNAPSHOT_TIMES) -> dict[str, object]:
    """Execute la simulation et stocke snapshots, energie et diagnostics temporels."""
    params = make_params(config)
    solver = initialize_solver(params, config)

    snapshot_indices = {
        int(round(float(time) / params.dt)): float(time)
        for time in snapshot_times
        if 0.0 <= float(time) <= params.nt * params.dt
    }

    energy = np.empty(params.nt + 1, dtype=float)
    max_abs_u = np.empty(params.nt + 1, dtype=float)
    min_denom = np.empty(params.nt + 1, dtype=float)
    snapshots = {}

    with contextlib.redirect_stdout(io.StringIO()):
        for n in range(params.nt + 1):
            if n in snapshot_indices:
                snapshots[snapshot_indices[n]] = solver.u.copy()

            denom = 1.0 - 2.0 * params.k * solver.u
            energy[n] = solver.compute_energy()
            max_abs_u[n] = float(np.max(np.abs(solver.u)))
            min_denom[n] = float(np.min(denom))

            if n < params.nt:
                solver.step()

    return {
        "params": params,
        "x": solver.x.copy(),
        "time": np.arange(params.nt + 1, dtype=float) * params.dt,
        "snapshots": snapshots,
        "energy": energy,
        "max_abs_u": max_abs_u,
        "min_denom": min_denom,
        "u_final": solver.u.copy(),
        "F_final": solver.F.copy(),
    }


def run_nikolic_stability_scan(config: NikolicConfig) -> list[dict]:
    """Reproduit le balayage de stabilite du notebook Nikolic."""
    params = make_params(config)
    solver = initialize_solver(params, config)

    dt_values = [factor * config.dt for factor in DT_MULTIPLIERS]
    with contextlib.redirect_stdout(io.StringIO()):
        results = solver.run_stability_scan(
            dt_values=dt_values,
            amplitude_values=list(AMPLITUDE_VALUES),
            u0_type=config.u0_type,
            u1_type=config.u1_type,
            velocity_amplitude=config.amplitude_u1,
            mu=config.mu,
            sigma1=config.sigma1,
            sigma2=config.sigma2,
            blowup_threshold=config.blowup_threshold,
        )
    return results


def metadata(config: NikolicConfig, params: WesterveltParams) -> dict[str, object]:
    cfl = params.c * params.dt / params.dx
    return {
        "analysis": "nikolic_simulation",
        "config": asdict(config),
        "derived": {
            "dx": params.dx,
            "dt": params.dt,
            "t_final": params.nt * params.dt,
            "b": params.b,
            "k": params.k,
            "cfl": cfl,
            "lambda": params.c**2 * params.dt / params.dx**2,
            "nondegeneracy_limit": 1.0 / (2.0 * params.k),
        },
    }


def save_figure(fig: plt.Figure, filename: str, md: dict[str, object], savefig: bool) -> dict[str, Path]:
    if not savefig:
        return {}
    return save_figure_with_version(
        fig,
        filename=filename,
        output_dir=str(OUTPUT_DIR),
        formats=["png", "pdf"],
        metadata=md,
        tight_layout=False,
    )


def plot_snapshots(simulation: dict[str, object], md: dict[str, object], show: bool = False, savefig: bool = True) -> tuple[plt.Figure, dict[str, Path]]:
    x = simulation["x"]
    snapshots = simulation["snapshots"]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for time, values in sorted(snapshots.items()):
        ax.plot(x, values, linewidth=1.6, label=f"{time * 1e6:.2f} us")

    ax.set_title("Cas Nikolic-Wohlmuth: evolution spatiale")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u(x,t) (Pa)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    paths = save_figure(fig, "nikolic_snapshots", md, savefig)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, paths


def plot_energy_and_diagnostics(simulation: dict[str, object], md: dict[str, object], show: bool = False, savefig: bool = True) -> tuple[plt.Figure, dict[str, Path]]:
    time_us = simulation["time"] * 1e6
    energy = simulation["energy"]
    max_abs_u = simulation["max_abs_u"]
    min_denom = simulation["min_denom"]

    fig, axs = plt.subplot_mosaic(
        [["energy", "energy"], ["max_u", "denom"]],
        figsize=(12, 7.5),
        constrained_layout=True,
    )

    axs["energy"].plot(time_us, energy, linewidth=1.6, color="#315c8a")
    axs["energy"].set_title("Energie discrete")
    axs["energy"].set_xlabel("t (us)")
    axs["energy"].set_ylabel("E(t)")
    axs["energy"].grid(True, alpha=0.3)

    axs["max_u"].plot(time_us, max_abs_u, linewidth=1.6, color="#467a4b")
    axs["max_u"].set_title("Amplitude maximale")
    axs["max_u"].set_xlabel("t (us)")
    axs["max_u"].set_ylabel("max |u|")
    axs["max_u"].grid(True, alpha=0.3)

    axs["denom"].plot(time_us, min_denom, linewidth=1.6, color="#8a4f31")
    axs["denom"].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axs["denom"].set_title("Non-degenerescence")
    axs["denom"].set_xlabel("t (us)")
    axs["denom"].set_ylabel("min(1 - 2ku)")
    axs["denom"].grid(True, alpha=0.3)

    paths = save_figure(fig, "nikolic_energy_diagnostics", md, savefig)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, paths


def plot_stability_scan(results: list[dict], md: dict[str, object], show: bool = False, savefig: bool = True) -> tuple[plt.Figure, dict[str, Path]]:
    dt_vals, amp_vals = get_scan_axes(results)
    observed = build_scan_grid(results, dt_vals, amp_vals, lambda row: 1.0 if row.get("stable", False) else 0.0, default=np.nan)
    theoretical = build_scan_grid(results, dt_vals, amp_vals, lambda row: 1.0 if row.get("theoretical_stable", False) else 0.0, default=np.nan)
    max_u = build_scan_grid(results, dt_vals, amp_vals, lambda row: row.get("max_abs_u", np.nan), default=np.nan)
    min_denom = build_scan_grid(results, dt_vals, amp_vals, lambda row: row.get("min_denom", np.nan), default=np.nan)

    extent = [min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)]
    fig, axs = plt.subplot_mosaic(
        [["observed", "theory"], ["max_u", "denom"]],
        figsize=(13, 9),
        constrained_layout=True,
    )

    im0 = axs["observed"].imshow(observed, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axs["observed"].set_title("Stabilite observee")
    axs["observed"].set_xlabel("dt (s)")
    axs["observed"].set_ylabel("Amplitude u0")
    plt.colorbar(im0, ax=axs["observed"], label="stable")

    im1 = axs["theory"].imshow(theoretical, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axs["theory"].set_title("Stabilite theorique")
    axs["theory"].set_xlabel("dt (s)")
    axs["theory"].set_ylabel("Amplitude u0")
    plt.colorbar(im1, ax=axs["theory"], label="stable")

    im2 = axs["max_u"].imshow(max_u, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axs["max_u"].set_title("Maximum de |u|")
    axs["max_u"].set_xlabel("dt (s)")
    axs["max_u"].set_ylabel("Amplitude u0")
    plt.colorbar(im2, ax=axs["max_u"], label="max |u|")

    im3 = axs["denom"].imshow(min_denom, origin="lower", aspect="auto", extent=extent, cmap="plasma")
    axs["denom"].set_title("Minimum de 1 - 2ku")
    axs["denom"].set_xlabel("dt (s)")
    axs["denom"].set_ylabel("Amplitude u0")
    plt.colorbar(im3, ax=axs["denom"], label="min denom")

    stable_count = sum(1 for row in results if row.get("stable", False))
    fig.suptitle(f"Scan Nikolic: {stable_count}/{len(results)} configurations stables", fontsize=14)

    scan_md = dict(md)
    scan_md["scan"] = {
        "dt_multipliers": list(DT_MULTIPLIERS),
        "amplitude_values": list(AMPLITUDE_VALUES),
        "stable_count": stable_count,
        "total": len(results),
    }
    paths = save_figure(fig, "nikolic_stability_scan", scan_md, savefig)
    if savefig:
        save_data_with_version(
            results,
            filename="nikolic_stability_scan",
            output_dir=str(OUTPUT_DIR),
            fmt="json",
            metadata=scan_md,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, paths


def main(show: bool = False, savefig: bool = False, quick: bool = False, run_scan: bool = True) -> dict[str, dict[str, Path]]:
    """Point d'entree de l'experience Nikolic."""
    set_style()
    config = make_config(quick=quick)
    params = make_params(config)

    print("Analyse des simulations Nikolic-Wohlmuth")
    print(f"  scheme = {config.scheme}")
    print(f"  nx = {config.nx}, nt = {config.nt}")
    print(f"  dx = {params.dx:.3e} m")
    print(f"  dt = {params.dt:.3e} s")
    print(f"  T = {params.nt * params.dt:.3e} s")
    print(f"  b = {params.b:.3e} m^2/s")
    print(f"  k = {params.k:.3e} Pa^-1")
    print(f"  A1 = {config.amplitude_u0:.3e}, A2 = {config.amplitude_u1:.3e}")

    simulation = run_nikolic_simulation(config)
    md = metadata(config, params)

    print("Diagnostics simulation:")
    print(f"  max |u(T)| = {np.max(np.abs(simulation['u_final'])):.3e}")
    print(f"  min_t min(1 - 2ku) = {np.min(simulation['min_denom']):.6f}")
    print(f"  energie initiale = {simulation['energy'][0]:.6e}")
    print(f"  energie finale   = {simulation['energy'][-1]:.6e}")

    _, paths_snapshots = plot_snapshots(simulation, md, show=show, savefig=savefig)
    _, paths_energy = plot_energy_and_diagnostics(simulation, md, show=show, savefig=savefig)

    paths_scan = {}
    if run_scan:
        scan_results = run_nikolic_stability_scan(config)
        stable_count = sum(1 for row in scan_results if row.get("stable", False))
        print(f"Scan stabilite: {stable_count}/{len(scan_results)} configurations stables")
        _, paths_scan = plot_stability_scan(scan_results, md, show=show, savefig=savefig)
    else:
        print("Scan stabilite ignore.")

    if savefig:
        print(f"Sorties enregistrees dans {OUTPUT_DIR}")
    else:
        print("Enregistrement des figures desactive (savefig=False).")

    return {
        "snapshots": paths_snapshots,
        "energy_diagnostics": paths_energy,
        "stability_scan": paths_scan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse du cas Nikolic-Wohlmuth pour le modele de Westervelt.")
    parser.add_argument("--savefig", action="store_true", help="Enregistre les figures et donnees dans outputs/analysis.")
    parser.add_argument("--no-show", action="store_true", help="N'affiche pas les figures.")
    parser.add_argument("--quick", action="store_true", help="Utilise une grille reduite pour verifier rapidement le pipeline.")
    parser.add_argument("--no-scan", action="store_true", help="Desactive le scan de stabilite.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        show=not args.no_show,
        savefig=args.savefig,
        quick=args.quick,
        run_scan=not args.no_scan,
    )
