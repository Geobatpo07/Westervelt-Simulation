from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.solver import WesterveltParams, WesterveltSolver
from utils import build_scan_grid, get_scan_axes


PROFILE_TYPES = {
    "Gaussien": "gaussian",
    "Derivee gaussienne": "gaussian_derivative",
    "Zero": "zero",
    "Uniforme aleatoire": "uniform",
}

SCHEMES = {
    "Semi-implicite": "semi_implicit",
    "Explicite": "explicit",
}

BOUNDARIES = {
    "Dirichlet homogene": "dirichlet",
    "Neumann homogene": "neumann",
}

NIKOLIC_SNAPSHOT_TIMES = (0.0, 5e-6, 9.25e-6, 15e-6, 20e-6, 25e-6, 30e-6, 37e-6)

PRESETS = {
    "Nikolic-Wohlmuth": {
        "model": dict(
            c=1500.0,
            rho0=1000.0,
            beta=3.5,
            mu_v=6e-6,
            dx=1.0e-4,
            dt=1.85e-8,
            nx=2000,
            nt=2000,
            scheme="explicit",
            bc="dirichlet",
        ),
        "init": dict(
            u0_type="gaussian",
            u1_type="gaussian_derivative",
            A1=1.2e8,
            A2=1.0e11,
            mu=0.1,
            sigma1=0.015,
            sigma2=0.02,
        ),
        "scan": dict(
            dt_multipliers=(0.5, 1.0, 1.5, 2.0, 3.0),
            amplitudes=(0.5e8, 0.8e8, 1.2e8, 1.6e8, 2.0e8),
            blowup_threshold=1.0e10,
        ),
        "snapshot_times": NIKOLIC_SNAPSHOT_TIMES,
    },
    "Demo rapide": {
        "model": dict(
            c=1500.0,
            rho0=1000.0,
            beta=3.5,
            mu_v=6e-6,
            dx=1.0e-4,
            dt=1.85e-8,
            nx=400,
            nt=300,
            scheme="explicit",
            bc="dirichlet",
        ),
        "init": dict(
            u0_type="gaussian",
            u1_type="gaussian_derivative",
            A1=1.2e8,
            A2=1.0e11,
            mu=0.02,
            sigma1=0.003,
            sigma2=0.004,
        ),
        "scan": dict(
            dt_multipliers=(0.5, 1.0, 1.5, 2.0, 3.0),
            amplitudes=(0.5e8, 0.8e8, 1.2e8, 1.6e8, 2.0e8),
            blowup_threshold=1.0e10,
        ),
        "snapshot_times": None,
    },
    "Semi-implicite exploration": {
        "model": dict(
            c=1500.0,
            rho0=1000.0,
            beta=3.5,
            mu_v=6e-6,
            dx=1.0e-4,
            dt=1.85e-8,
            nx=800,
            nt=700,
            scheme="semi_implicit",
            bc="dirichlet",
        ),
        "init": dict(
            u0_type="gaussian",
            u1_type="gaussian_derivative",
            A1=1.2e8,
            A2=1.0e11,
            mu=0.04,
            sigma1=0.006,
            sigma2=0.008,
        ),
        "scan": dict(
            dt_multipliers=(0.5, 1.0, 1.5, 2.0, 3.0),
            amplitudes=(0.5e8, 0.8e8, 1.2e8, 1.6e8, 2.0e8),
            blowup_threshold=1.0e10,
        ),
        "snapshot_times": None,
    },
}


def label_from_value(options: dict[str, str], value: str) -> str:
    for label, option_value in options.items():
        if option_value == value:
            return label
    return next(iter(options))


def make_params(
    c: float,
    rho0: float,
    beta: float,
    mu_v: float,
    dx: float,
    dt: float,
    nx: int,
    nt: int,
    scheme: str,
    bc: str,
) -> WesterveltParams:
    return WesterveltParams(
        c=float(c),
        rho0=float(rho0),
        beta=float(beta),
        mu_v=float(mu_v),
        dx=float(dx),
        dt=float(dt),
        nx=int(nx),
        nt=int(nt),
        scheme=scheme,
        bc=bc,
    )


def stability_numbers(params: WesterveltParams) -> dict[str, float | bool]:
    cfl = params.c * params.dt / params.dx
    lambda_number = params.c**2 * params.dt / params.dx**2
    if params.scheme == "explicit":
        margin = params.dx**2 - (params.c**2 * params.dt**2 + 2.0 * params.b * params.dt)
    else:
        margin = params.dx**2 - (params.c**2 * params.dt**2 - 2.0 * params.b * params.dt)

    return {
        "cfl": float(cfl),
        "lambda": float(lambda_number),
        "margin": float(margin),
        "stable_margin": bool(margin >= 0.0),
    }


def create_solver(params: WesterveltParams, init: dict) -> WesterveltSolver:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        solver = WesterveltSolver(params)
        solver.initialize(
            u0_type=init["u0_type"],
            u1_type=init["u1_type"],
            A1=init["A1"],
            A2=init["A2"],
            mu=init["mu"],
            sigma1=init["sigma1"],
            sigma2=init["sigma2"],
        )
    return solver


@st.cache_data(show_spinner=False)
def run_simulation_cached(params_payload: dict, init_payload: dict, snapshot_times: tuple[float, ...]):
    params = make_params(**params_payload)
    solver = create_solver(params, init_payload)

    with contextlib.redirect_stdout(io.StringIO()):
        snapshots = solver.run_with_snapshots(snapshot_times, store_energy=True)

    energy = np.asarray(solver.energy_history, dtype=float)
    denominator = 1.0 - 2.0 * solver.param.k * solver.u

    return {
        "x": solver.x,
        "u": solver.u,
        "u_prev": solver.u_prev,
        "F": solver.F,
        "energy": energy,
        "snapshots": {float(t): np.asarray(values, dtype=float) for t, values in snapshots.items()},
        "max_abs_u": float(np.max(np.abs(solver.u))),
        "min_denom": float(np.min(denominator)),
        "finite": bool(np.all(np.isfinite(solver.u)) and np.all(np.isfinite(energy))),
    }


@st.cache_data(show_spinner=False)
def run_scan_cached(params_payload: dict, init_payload: dict, dt_values: tuple[float, ...], amp_values: tuple[float, ...], blowup_threshold: float):
    params = make_params(**params_payload)
    solver = create_solver(params, init_payload)
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.run_stability_scan(
            dt_values=list(dt_values),
            amplitude_values=list(amp_values),
            u0_type=init_payload["u0_type"],
            u1_type=init_payload["u1_type"],
            velocity_amplitude=init_payload["A2"],
            mu=init_payload["mu"],
            sigma1=init_payload["sigma1"],
            sigma2=init_payload["sigma2"],
            blowup_threshold=float(blowup_threshold),
        )


def plot_snapshots(x: np.ndarray, snapshots: dict[float, np.ndarray]):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for t, values in sorted(snapshots.items()):
        ax.plot(x, values, linewidth=1.5, label=f"{t * 1e6:.2f} us")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Snapshots spatiaux")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_energy(energy: np.ndarray, dt: float):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    time = np.arange(energy.size) * dt
    ax.plot(time, energy, linewidth=1.6, color="#315c8a")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Energie discrete")
    ax.set_title("Evolution de l'energie")
    ax.grid(alpha=0.25)
    return fig


def plot_final_solution(x: np.ndarray, u: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, u, linewidth=1.7, color="#467a4b")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u final")
    ax.set_title("Solution finale")
    ax.grid(alpha=0.25)
    return fig


def plot_scan(results: list[dict]):
    dt_vals, amp_vals = get_scan_axes(results)
    grid = build_scan_grid(results, dt_vals, amp_vals, lambda r: 1.0 if r.get("stable", False) else 0.0, default=np.nan)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(dt_vals), max(dt_vals), min(amp_vals), max(amp_vals)],
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    ax.set_xlabel("dt (s)")
    ax.set_ylabel("Amplitude u0")
    ax.set_title("Carte de stabilite observee")
    fig.colorbar(image, ax=ax, label="1 = stable, 0 = instable")
    return fig


def results_to_csv(results: list[dict]) -> str:
    if not results:
        return ""

    columns = [
        "dt",
        "amplitude",
        "amplitude_u0",
        "amplitude_u1",
        "stable",
        "max_abs_u",
        "cfl",
        "lambda_legacy",
        "min_denom",
        "nondegenerate",
        "stability_margin",
        "theoretical_stable",
    ]
    rows = [",".join(columns)]
    for row in results:
        rows.append(",".join(str(row.get(column, "")) for column in columns))
    return "\n".join(rows)


def sidebar_controls():
    with st.sidebar:
        st.header("Parametres")

        preset = st.selectbox(
            "Preset",
            list(PRESETS.keys()),
            index=0,
        )

        preset_config = PRESETS[preset]
        defaults = preset_config["model"]
        init_defaults = preset_config["init"]

        scheme_default = label_from_value(SCHEMES, defaults["scheme"])
        bc_default = label_from_value(BOUNDARIES, defaults["bc"])
        scheme_label = st.segmented_control("Schema", list(SCHEMES.keys()), default=scheme_default)
        bc_label = st.selectbox("Condition limite", list(BOUNDARIES.keys()), index=list(BOUNDARIES.keys()).index(bc_default))

        c = st.number_input("c (m/s)", min_value=1.0, value=defaults["c"], step=50.0)
        rho0 = st.number_input("rho0 (kg/m3)", min_value=1.0, value=defaults["rho0"], step=50.0)
        beta = st.number_input("beta", min_value=1e-9, value=defaults["beta"], step=0.1, format="%.6g")
        mu_v = st.number_input("mu_v (Pa.s)", min_value=0.0, value=defaults["mu_v"], step=1e-6, format="%.6g")

        dx = st.number_input("dx (m)", min_value=1e-12, value=defaults["dx"], step=1e-6, format="%.8g")
        dt = st.number_input("dt (s)", min_value=1e-15, value=defaults["dt"], step=1e-9, format="%.8g")
        nx = st.number_input("nx", min_value=8, max_value=10000, value=defaults["nx"], step=50)
        nt = st.number_input("nt", min_value=1, max_value=20000, value=defaults["nt"], step=50)

        st.header("Initialisation")
        u0_default = label_from_value(PROFILE_TYPES, init_defaults["u0_type"])
        u1_default = label_from_value(PROFILE_TYPES, init_defaults["u1_type"])
        u0_label = st.selectbox("Profil u0", list(PROFILE_TYPES.keys()), index=list(PROFILE_TYPES.keys()).index(u0_default))
        A1 = st.number_input("Amplitude u0", value=init_defaults["A1"], step=1e7, format="%.6g")
        u1_label = st.selectbox("Profil u_t0", list(PROFILE_TYPES.keys()), index=list(PROFILE_TYPES.keys()).index(u1_default))
        A2 = st.number_input("Amplitude u_t0", value=init_defaults["A2"], step=1e10, format="%.6g")

        length = float(dx) * (int(nx) - 1)
        mu_default = float(np.clip(init_defaults["mu"], 0.0, length))
        sigma1_default = float(max(init_defaults["sigma1"], 1e-12))
        sigma2_default = float(max(init_defaults["sigma2"], 1e-12))
        mu = st.number_input("mu centre (m)", min_value=0.0, max_value=max(length, 1e-12), value=mu_default, step=max(length / 100.0, 1e-6), format="%.8g")
        sigma1 = st.number_input("sigma1 u0 (m)", min_value=1e-12, value=sigma1_default, step=max(length / 200.0, 1e-6), format="%.8g")
        sigma2 = st.number_input("sigma2 u_t0 (m)", min_value=1e-12, value=sigma2_default, step=max(length / 200.0, 1e-6), format="%.8g")

        st.header("Sortie")
        snapshot_default = len(preset_config["snapshot_times"]) if preset_config["snapshot_times"] else 5
        snapshot_count = st.slider("Nombre de snapshots", 2, 8, snapshot_default)
        run_clicked = st.button("Lancer la simulation", type="primary", use_container_width=True)

    params_payload = {
        "c": c,
        "rho0": rho0,
        "beta": beta,
        "mu_v": mu_v,
        "dx": dx,
        "dt": dt,
        "nx": int(nx),
        "nt": int(nt),
        "scheme": SCHEMES[scheme_label],
        "bc": BOUNDARIES[bc_label],
    }
    init_payload = {
        "u0_type": PROFILE_TYPES[u0_label],
        "u1_type": PROFILE_TYPES[u1_label],
        "A1": float(A1),
        "A2": float(A2),
        "mu": float(mu),
        "sigma1": float(max(sigma1, 1e-12)),
        "sigma2": float(max(sigma2, 1e-12)),
    }
    scan_defaults = preset_config["scan"]
    return params_payload, init_payload, int(snapshot_count), run_clicked, preset, scan_defaults


def snapshot_times_for_preset(preset: str, total_time: float, snapshot_count: int) -> tuple[float, ...]:
    configured_times = PRESETS[preset].get("snapshot_times")
    if configured_times and snapshot_count == len(configured_times):
        return tuple(float(t) for t in configured_times if 0.0 <= float(t) <= total_time)
    return tuple(np.linspace(0.0, total_time, snapshot_count))


def main():
    st.set_page_config(page_title="Westervelt Lab", page_icon=None, layout="wide")
    st.title("Westervelt Lab")

    params_payload, init_payload, snapshot_count, run_clicked, preset, scan_defaults = sidebar_controls()
    params = make_params(**params_payload)
    numbers = stability_numbers(params)
    total_time = params.nt * params.dt
    domain_length = params.dx * (params.nx - 1)

    metric_cols = st.columns(5)
    metric_cols[0].metric("CFL", f"{numbers['cfl']:.4g}")
    metric_cols[1].metric("lambda", f"{numbers['lambda']:.4g}")
    metric_cols[2].metric("Marge", f"{numbers['margin']:.3g}")
    metric_cols[3].metric("Duree", f"{total_time * 1e6:.3g} us")
    metric_cols[4].metric("Domaine", f"{domain_length * 1e3:.3g} mm")

    if params.scheme == "explicit" and not numbers["stable_margin"]:
        st.warning("Le schema explicite est hors marge de stabilite lineaire pour ces pas.")

    if not run_clicked and "last_simulation" not in st.session_state:
        run_clicked = True

    if run_clicked:
        snapshot_times = snapshot_times_for_preset(preset, total_time, snapshot_count)
        with st.spinner("Simulation en cours..."):
            st.session_state["last_simulation"] = run_simulation_cached(params_payload, init_payload, snapshot_times)
            st.session_state["last_payload"] = (params_payload, init_payload)

    simulation = st.session_state.get("last_simulation")

    if simulation is None:
        st.info("Lance une simulation pour afficher les resultats.")
        return

    status_cols = st.columns(3)
    status_cols[0].metric("max |u| final", f"{simulation['max_abs_u']:.5g}")
    status_cols[1].metric("min(1 - 2ku)", f"{simulation['min_denom']:.5g}")
    status_cols[2].metric("Etat numerique", "fini" if simulation["finite"] else "non fini")

    tab_solution, tab_energy, tab_scan, tab_data = st.tabs(["Solution", "Energie", "Scan stabilite", "Donnees"])

    with tab_solution:
        st.pyplot(plot_snapshots(simulation["x"], simulation["snapshots"]), clear_figure=True)
        st.pyplot(plot_final_solution(simulation["x"], simulation["u"]), clear_figure=True)

    with tab_energy:
        st.pyplot(plot_energy(simulation["energy"], params.dt), clear_figure=True)

    with tab_scan:
        left, right = st.columns([0.34, 0.66])
        with left:
            st.subheader("Balayage")
            dt_multipliers = scan_defaults["dt_multipliers"]
            scan_amplitudes = scan_defaults["amplitudes"]
            dt_min = st.number_input("dt min", min_value=1e-15, value=params.dt * min(dt_multipliers), format="%.8g")
            dt_max = st.number_input("dt max", min_value=1e-15, value=params.dt * max(dt_multipliers), format="%.8g")
            dt_count = st.slider("Pas dt", 2, 12, len(dt_multipliers))
            amp_min = st.number_input("Amplitude min", value=min(scan_amplitudes), format="%.6g")
            amp_max = st.number_input("Amplitude max", value=max(scan_amplitudes), format="%.6g")
            amp_count = st.slider("Pas amplitude", 2, 12, len(scan_amplitudes))
            blowup_threshold = st.number_input("Seuil divergence |u|", min_value=1.0, value=scan_defaults["blowup_threshold"], format="%.6g")
            run_scan = st.button("Lancer le scan", use_container_width=True)

        with right:
            if run_scan:
                dt_values = tuple(np.linspace(float(dt_min), float(dt_max), int(dt_count)))
                amp_values = tuple(np.linspace(float(amp_min), float(amp_max), int(amp_count)))
                with st.spinner("Scan de stabilite en cours..."):
                    st.session_state["last_scan"] = run_scan_cached(
                        params_payload,
                        init_payload,
                        dt_values,
                        amp_values,
                        float(blowup_threshold),
                    )

            scan_results = st.session_state.get("last_scan")
            if scan_results:
                stable_count = sum(1 for row in scan_results if row.get("stable", False))
                st.metric("Configurations stables", f"{stable_count}/{len(scan_results)}")
                st.pyplot(plot_scan(scan_results), clear_figure=True)
                st.download_button(
                    "Telecharger le scan CSV",
                    data=results_to_csv(scan_results),
                    file_name="westervelt_scan.csv",
                    mime="text/csv",
                )
            else:
                st.info("Lance un scan pour visualiser une carte de stabilite.")

    with tab_data:
        st.subheader("Parametres effectifs")
        st.json(
            {
                "modele": params_payload,
                "initialisation": init_payload,
                "coefficients": {"b": params.b, "k": params.k},
                "diagnostics": numbers,
            }
        )
        st.subheader("Apercu solution finale")
        preview_idx = np.linspace(0, len(simulation["x"]) - 1, min(250, len(simulation["x"])), dtype=int)
        st.dataframe(
            {
                "x": simulation["x"][preview_idx],
                "u_final": simulation["u"][preview_idx],
                "F_final": simulation["F"][preview_idx],
            },
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
