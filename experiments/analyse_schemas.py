# experiments/analyse_schemas.py

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

# Permet l'execution directe: python experiments/analyse_schemas.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.solver import WesterveltParams, WesterveltSolver
from src.stability_analysis import (
    explicit_stability_margin,
    explicit_theoretical_stable,
    scan_spectral_radius_explicit,
    scan_spectral_radius_semi_implicit,
    semi_implicit_stability_margin,
    semi_implicit_theoretical_stable,
)
from utils import (
    build_scan_grid,
    compute_stable_ratio,
    ensure_output_dir,
    get_scan_axes,
    save_data_with_version,
    save_figure_with_version,
)


def _make_dirs(base_output: str) -> dict[str, str]:
    base = Path(base_output)
    dirs = {
        "solutions": base / "analysis" / "schema_analysis" / "solutions",
        "stability": base / "analysis" / "schema_analysis" / "stability",
        "comparisons": base / "analysis" / "schema_analysis" / "comparisons",
        "spectral": base / "analysis" / "schema_analysis" / "spectral_radius",
        "data": base / "analysis" / "schema_analysis" / "data",
    }
    for p in dirs.values():
        ensure_output_dir(str(p))
    return {k: str(v) for k, v in dirs.items()}


def _run_solver_scheme(params: WesterveltParams, u0_type: str = "gaussian", amplitude: float = 1.0, u1_type: str = "zero", velocity_amplitude: float = 0.0):
    solver = WesterveltSolver(params)
    solver.initialize(u0_type=u0_type, u1_type=u1_type, A1=float(amplitude), A2=float(velocity_amplitude))
    solver.run(store_energy=True)
    return {
        "x": solver.x.copy(),
        "u": solver.u.copy(),
        "energy": np.array(solver.energy_history, dtype=float),
        "stable_ratio": None,
    }


def _plot_final_solutions(solution_by_scheme: dict, output_dir: str, metadata: dict):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for scheme, data in solution_by_scheme.items():
        ax.plot(data["x"], data["u"], linewidth=1.6, label=scheme)
    ax.set_title("Solution finale par schema")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u(x, T)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure_with_version(
        fig,
        filename="final_solutions_all_schemes",
        output_dir=output_dir,
        tight_layout=False,
        metadata=metadata,
    )
    plt.close(fig)


def _plot_theory_vs_observed(results, scheme: str, output_dir: str, metadata: dict):
    dt_vals, amp_vals = get_scan_axes(results)
    observed = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r.get("stable", False) else 0.0)
    theoretical = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r.get("theoretical_stable", False) else 0.0)
    diff = observed - theoretical

    fig, axs = plt.subplot_mosaic(
        [["theory", "observed"], ["diff", "diff"]],
        figsize=(13.5, 9),
        constrained_layout=True,
    )

    im_t = axs["theory"].imshow(
        theoretical,
        origin="lower",
        aspect="auto",
        extent=(min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)),
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["theory"].set_title(f"Theorie ({scheme})")
    axs["theory"].set_xlabel("dt (s)")
    axs["theory"].set_ylabel("Amplitude")
    plt.colorbar(im_t, ax=axs["theory"], label="stable")

    im_o = axs["observed"].imshow(
        observed,
        origin="lower",
        aspect="auto",
        extent=(min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)),
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    axs["observed"].set_title(f"Observe ({scheme})")
    axs["observed"].set_xlabel("dt (s)")
    axs["observed"].set_ylabel("Amplitude")
    plt.colorbar(im_o, ax=axs["observed"], label="stable")

    im_d = axs["diff"].imshow(
        diff,
        origin="lower",
        aspect="auto",
        extent=(min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)),
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )
    axs["diff"].set_title("Difference observe - theorie")
    axs["diff"].set_xlabel("dt (s)")
    axs["diff"].set_ylabel("Amplitude")
    plt.colorbar(im_d, ax=axs["diff"], label="diff")

    save_figure_with_version(
        fig,
        filename=f"theory_vs_observed_{scheme}",
        output_dir=output_dir,
        tight_layout=False,
        metadata=metadata,
    )
    plt.close(fig)


def _plot_scheme_stability_comparison(results_map: dict, output_dir: str, metadata: dict):
    dt_vals, amp_vals = get_scan_axes(results_map["explicit"])

    explicit_grid = build_scan_grid(
        results_map["explicit"], dt_vals, amp_vals, lambda r: 1.0 if r.get("stable", False) else 0.0
    )
    semi_grid = build_scan_grid(
        results_map["semi_implicit"], dt_vals, amp_vals, lambda r: 1.0 if r.get("stable", False) else 0.0
    )
    diff_grid = semi_grid - explicit_grid

    fig, axs = plt.subplot_mosaic(
        [["explicit", "semi"], ["diff", "diff"]],
        figsize=(13.5, 9),
        constrained_layout=True,
    )

    for key, grid, title in [
        ("explicit", explicit_grid, "Stabilite observee - explicit"),
        ("semi", semi_grid, "Stabilite observee - semi_implicit"),
    ]:
        im = axs[key].imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=(min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)),
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
        )
        axs[key].set_title(title)
        axs[key].set_xlabel("dt (s)")
        axs[key].set_ylabel("Amplitude")
        plt.colorbar(im, ax=axs[key], label="stable")

    imd = axs["diff"].imshow(
        diff_grid,
        origin="lower",
        aspect="auto",
        extent=(min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)),
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )
    axs["diff"].set_title("Difference semi_implicit - explicit")
    axs["diff"].set_xlabel("dt (s)")
    axs["diff"].set_ylabel("Amplitude")
    plt.colorbar(imd, ax=axs["diff"], label="diff")

    save_figure_with_version(
        fig,
        filename="observed_stability_comparison_schemes",
        output_dir=output_dir,
        tight_layout=False,
        metadata=metadata,
    )
    plt.close(fig)


def _plot_stable_ratio_bar(results_map: dict, output_dir: str, metadata: dict):
    schemes = ["explicit", "semi_implicit"]
    ratios = [compute_stable_ratio(results_map[s]) for s in schemes]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(schemes, ratios, color=["#cb6d51", "#2a9d8f"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Ratio stable")
    ax.set_title("Comparaison globale des schemas")
    ax.grid(axis="y", alpha=0.25)

    for i, val in enumerate(ratios):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center")

    save_figure_with_version(
        fig,
        filename="stable_ratio_comparison",
        output_dir=output_dir,
        tight_layout=False,
        metadata=metadata,
    )
    plt.close(fig)


def _plot_spectral_radius(params: WesterveltParams, output_dir: str, metadata: dict):
    s_exp = scan_spectral_radius_explicit(dt=params.dt, dx=params.dx, c=params.c, b=params.b, ntheta=500)
    s_semi = scan_spectral_radius_semi_implicit(dt=params.dt, dx=params.dx, c=params.c, b=params.b, alpha=1.0, ntheta=500)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s_exp["theta"], s_exp["rho"], label=f"explicit (rho_max={s_exp['rho_max']:.3f})")
    ax.plot(s_semi["theta"], s_semi["rho"], label=f"semi_implicit (rho_max={s_semi['rho_max']:.3f})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="rho = 1")
    ax.set_xlabel("theta")
    ax.set_ylabel("Rayon spectral")
    ax.set_title("Analyse de stabilite lineaire (rayon spectral)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure_with_version(
        fig,
        filename="spectral_radius_explicit_vs_semi_implicit",
        output_dir=output_dir,
        tight_layout=False,
        metadata={
            **metadata,
            "rho_max_explicit": float(s_exp["rho_max"]),
            "rho_max_semi_implicit": float(s_semi["rho_max"]),
        },
    )
    plt.close(fig)


def _build_scan_config(quick: bool):
    if quick:
        dt_values = [1.0e-8, 1.5e-8, 2.0e-8]
        amp_values = [0.5, 1.0, 1.5]
    else:
        dt_values = [1.0e-8, 1.4e-8, 1.8e-8, 2.2e-8, 2.6e-8, 3.0e-8]
        amp_values = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
    return dt_values, amp_values


def _clone_params_with_scheme(params: WesterveltParams, scheme: str) -> WesterveltParams:
    return WesterveltParams(
        c=params.c,
        rho0=params.rho0,
        beta=params.beta,
        mu_v=params.mu_v,
        B_over_A=params.B_over_A,
        nu=params.nu,
        dx=params.dx,
        dt=params.dt,
        nx=params.nx,
        nt=params.nt,
        bc=params.bc,
        scheme=scheme,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyses de schema Westervelt: solutions, stabilite, comparaisons."
    )
    parser.add_argument("--quick", action="store_true", help="Mode rapide pour validation locale.")
    parser.add_argument("--output-dir", default="outputs", help="Repertoire racine de sortie.")
    parser.add_argument("--blowup-threshold", type=float, default=1e5, help="Seuil de divergence observee.")
    parser.add_argument("--show", action="store_true", help="Affiche les figures en plus de la sauvegarde.")
    parser.add_argument("velocity_amplitude", type=float, nargs="?", default=1.0e11, help="Amplitude de la vitesse (par defaut 1e11).")
    args = parser.parse_args()

    nx = 220 if args.quick else 900
    nt = 140 if args.quick else 900
    base_dt = 1.8e-8

    base_params = WesterveltParams(
        c=1500,
        rho0=1000,
        beta=4.8,
        mu_v=6e-6,
        dx=2.8e-5,
        dt=base_dt,
        nx=nx,
        nt=nt,
        bc="dirichlet",
        scheme="semi_implicit",
    )

    dirs = _make_dirs(args.output_dir)
    dt_values, amp_values = _build_scan_config(args.quick)

    metadata = {
        "script": "experiments/analyse_schemas.py",
        "quick": bool(args.quick),
        "params": asdict(base_params),
        "dt_values": [float(v) for v in dt_values],
        "amp_values": [float(v) for v in amp_values],
    }

    print("[1/5] Simulation des solutions finales...")
    solution_by_scheme = {}

    p_exp = _clone_params_with_scheme(base_params, "explicit")
    p_semi = _clone_params_with_scheme(base_params, "semi_implicit")

    solution_by_scheme["explicit"] = _run_solver_scheme(p_exp)
    solution_by_scheme["semi_implicit"] = _run_solver_scheme(p_semi)

    _plot_final_solutions(solution_by_scheme, dirs["solutions"], metadata)

    print("[2/5] Scans de stabilite observee...")
    solver_exp = WesterveltSolver(p_exp)
    solver_semi = WesterveltSolver(p_semi)
    results_exp = solver_exp.run_stability_scan(
        dt_values=dt_values,
        amplitude_values=amp_values,
        u0_type="gaussian",
        u1_type="gaussian_derivative",
        velocity_amplitude=args.velocity_amplitude,
        blowup_threshold=args.blowup_threshold,
    )
    results_semi = solver_semi.run_stability_scan(
        dt_values=dt_values,
        amplitude_values=amp_values,
        u0_type="gaussian",
        u1_type="gaussian_derivative",
        velocity_amplitude=1.0e11,
        blowup_threshold=args.blowup_threshold,
    )
    print("[3/5] Theorie vs observation (explicit/semi_implicit)...")
    _plot_theory_vs_observed(results_exp, "explicit", dirs["stability"], metadata)
    _plot_theory_vs_observed(results_semi, "semi_implicit", dirs["stability"], metadata)

    print("[4/5] Comparaison entre schemas...")
    results_map = {
        "explicit": results_exp,
        "semi_implicit": results_semi,
    }
    _plot_scheme_stability_comparison(results_map, dirs["comparisons"], metadata)
    _plot_stable_ratio_bar(results_map, dirs["comparisons"], metadata)

    print("[5/5] Analyse spectrale theorique...")
    _plot_spectral_radius(base_params, dirs["spectral"], metadata)

    # Tableau de synthese theorie pour explicite et semi-implicite.
    theory_table = {
        "explicit": {
            "margin": float(explicit_stability_margin(base_params.dt, base_params.dx, base_params.c, base_params.b)),
            "stable": bool(explicit_theoretical_stable(base_params.dt, base_params.dx, base_params.c, base_params.b)),
        },
        "semi_implicit": {
            "margin": float(semi_implicit_stability_margin(base_params.dt, base_params.dx, base_params.c, base_params.b, alpha=1.0)),
            "stable": bool(semi_implicit_theoretical_stable(base_params.dt, base_params.dx, base_params.c, base_params.b, alpha=1.0)),
        },
        "stable_ratio_observed": {
            "explicit": float(compute_stable_ratio(results_exp)),
            "semi_implicit": float(compute_stable_ratio(results_semi)),
        },
    }

    save_data_with_version(
        data=theory_table,
        filename="stability_summary",
        output_dir=dirs["data"],
        fmt="json",
        metadata=metadata,
    )
    save_data_with_version(
        data={
            "explicit": results_exp,
            "semi_implicit": results_semi,
        },
        filename="stability_scan_results",
        output_dir=dirs["data"],
        fmt="json",
        metadata=metadata,
    )

    if args.show:
        plt.show()

    print("Termine. Figures et donnees enregistrees dans outputs/analysis/schema_analysis/.")


if __name__ == "__main__":
    main()




