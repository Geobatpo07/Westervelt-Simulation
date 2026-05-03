# ./app/westervelt_lab.py

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
from core.kernels_numba import (
    NUMBA_AVAILABLE,
    allocate_semi_implicit_workspace,
    compute_energy_numba,
    numba_thread_count,
    step_explicit_inplace,
    step_semi_implicit_inplace,
    warm_up_numba,
)
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


@st.cache_resource(show_spinner=False)
def warm_up_numba_cached():
    warm_up_numba()
    return True


def finite_max_abs(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite_values)))


def snapshot_indices(snapshot_times: tuple[float, ...], dt: float, nt: int) -> dict[int, float]:
    indices_to_save = {}
    for t in snapshot_times:
        n = int(round(float(t) / dt))
        if 0 <= n <= nt:
            indices_to_save[n] = float(t)
    return indices_to_save


def run_simulation_numba(
    params_payload: dict,
    init_payload: dict,
    snapshot_times: tuple[float, ...],
    render_every: int | None = None,
    live_container=None,
):
    params = make_params(**params_payload)
    solver = create_solver(params, init_payload)

    if NUMBA_AVAILABLE:
        warm_up_numba_cached()

    x = solver.x.copy()
    u_prev = np.asarray(solver.u_prev, dtype=np.float64).copy()
    u = np.asarray(solver.u, dtype=np.float64).copy()
    F = np.asarray(solver.F, dtype=np.float64).copy()
    u_next = np.empty_like(u)
    F_next = np.empty_like(F)
    workspace = allocate_semi_implicit_workspace(params.nx, dtype=u.dtype)

    dt = float(params.dt)
    nt = int(params.nt)
    indices_to_save = snapshot_indices(snapshot_times, dt, nt)
    snapshots = {}
    energy_values = [float(compute_energy_numba(u, u_prev, params.c, dt, params.dx))]

    frame_slot = None
    progress_slot = None
    status_slot = None
    y_limit = max(finite_max_abs(u) * 1.15, 1.0e-12)
    render_stride = max(1, int(render_every or nt or 1))

    if live_container is not None:
        with live_container.container():
            st.subheader("Propagation en direct")
            frame_slot = st.empty()
            progress_slot = st.progress(0.0)
            status_slot = st.empty()

    for n in range(nt + 1):
        if n in indices_to_save:
            snapshots[indices_to_save[n]] = u.copy()

        should_render = live_container is not None and (n == 0 or n == nt or n % render_stride == 0)
        if should_render and frame_slot is not None:
            current_max = finite_max_abs(u)
            if np.isfinite(current_max):
                y_limit = max(y_limit, current_max * 1.15, 1.0e-12)
            fig = plot_live_solution(x, u, params, n, y_limit)
            frame_slot.pyplot(fig, clear_figure=True)
            plt.close(fig)
            if progress_slot is not None:
                progress_slot.progress(1.0 if nt == 0 else n / nt)
            if status_slot is not None:
                status_slot.caption(f"pas {n}/{nt} | t = {n * dt * 1e6:.3f} us | max |u| = {current_max:.5g}")

        if n >= nt:
            continue

        if params.scheme == "semi_implicit":
            step_semi_implicit_inplace(
                u,
                F,
                u_next,
                F_next,
                params.c,
                params.b,
                params.k,
                dt,
                params.dx,
                solver.bc_type,
                *workspace,
            )
        else:
            step_explicit_inplace(
                u,
                F,
                u_next,
                F_next,
                params.c,
                params.b,
                params.k,
                dt,
                params.dx,
                solver.bc_type,
            )

        previous_u = u
        u_prev = previous_u
        u = u_next
        u_next = previous_u
        F, F_next = F_next, F

        energy_values.append(float(compute_energy_numba(u, u_prev, params.c, dt, params.dx)))

    energy = np.asarray(energy_values, dtype=float)
    denominator = 1.0 - 2.0 * params.k * u

    return {
        "x": x,
        "u": u.copy(),
        "u_prev": u_prev.copy(),
        "F": F.copy(),
        "energy": energy,
        "snapshots": {float(t): np.asarray(values, dtype=float) for t, values in snapshots.items()},
        "max_abs_u": finite_max_abs(u),
        "min_denom": float(np.min(denominator)),
        "finite": bool(np.all(np.isfinite(u)) and np.all(np.isfinite(energy))),
        "engine": "numba-parallel" if NUMBA_AVAILABLE else "python-fallback",
        "threads": numba_thread_count(),
    }


@st.cache_data(show_spinner=False)
def run_simulation_cached(params_payload: dict, init_payload: dict, snapshot_times: tuple[float, ...]):
    return run_simulation_numba(params_payload, init_payload, snapshot_times)


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


def plot_live_solution(x: np.ndarray, u: np.ndarray, params: WesterveltParams, step: int, y_limit: float):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, u, linewidth=1.5, color="#315c8a")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.45)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Propagation de l'onde - t = {step * params.dt * 1e6:.3f} us")
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
        live_enabled = st.checkbox("Propagation en direct", value=False)
        render_every = st.slider("Pas entre images", 1, 250, min(25, int(nt)), disabled=not live_enabled)
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
    live_options = {
        "enabled": bool(live_enabled),
        "render_every": int(render_every),
    }
    return params_payload, init_payload, int(snapshot_count), run_clicked, preset, scan_defaults, live_options


def snapshot_times_for_preset(preset: str, total_time: float, snapshot_count: int) -> tuple[float, ...]:
    configured_times = PRESETS[preset].get("snapshot_times")
    if configured_times and snapshot_count == len(configured_times):
        return tuple(float(t) for t in configured_times if 0.0 <= float(t) <= total_time)
    return tuple(np.linspace(0.0, total_time, snapshot_count))


def render_header(params: WesterveltParams, numbers: dict[str, float | bool], total_time: float, domain_length: float):
    st.title("Westervelt Lab")

    metric_cols = st.columns(5)
    metric_cols[0].metric("CFL", f"{numbers['cfl']:.4g}")
    metric_cols[1].metric("lambda", f"{numbers['lambda']:.4g}")
    metric_cols[2].metric("Marge", f"{numbers['margin']:.3g}")
    metric_cols[3].metric("Duree", f"{total_time * 1e6:.3g} us")
    metric_cols[4].metric("Domaine", f"{domain_length * 1e3:.3g} mm")

    if params.scheme == "explicit" and not numbers["stable_margin"]:
        st.warning("Le schema explicite est hors marge de stabilite lineaire pour ces pas.")


def render_home_page(params: WesterveltParams, numbers: dict[str, float | bool]):
    st.subheader("Propagation acoustique non lineaire")
    st.markdown(
        """
        Westervelt Lab est un espace d'exploration numerique pour visualiser la
        propagation d'une onde acoustique lorsque les effets non lineaires,
        dissipatifs et les contraintes de stabilite deviennent importants.

        Le modele de Westervelt apparait notamment en acoustique medicale,
        ultrasons focalises, propagation de fortes amplitudes et analyse de
        schemas pour equations hyperboliques non lineaires. Ici, l'objectif est
        de rendre ces dynamiques visibles: evolution spatiale de l'onde,
        energie discrete, stabilite observee et sensibilite aux parametres.
        """
    )

    st.latex(r"(1 - 2ku)u_{tt} - c^2 \Delta u + b \Delta u_t = 2k(u_t)^2")

    left, right = st.columns([0.52, 0.48])
    with left:
        st.markdown(
            """
            **Ce lab met l'accent sur**

            - la comparaison entre schema explicite et semi-implicite;
            - l'effet des conditions de bord Dirichlet ou Neumann;
            - le role des amplitudes initiales et du terme non lineaire;
            - la marge de stabilite liee a `dt`, `dx`, `c`, `b` et `k`;
            - l'affichage temps reel accelere par kernels Numba paralleles.
            """
        )

    with right:
        st.markdown(
            f"""
            **Configuration actuellement selectionnee**

            - Schema: `{params.scheme}`
            - Condition limite: `{params.bc}`
            - Grille: `{params.nx}` points
            - Pas de temps: `{params.dt:.3e}` s
            - CFL: `{numbers["cfl"]:.4g}`
            - Moteur live: `numba-parallel` si Numba est disponible
            """
        )

    st.info("Regle les parametres dans la sidebar, puis clique sur **Lancer la simulation** pour demarrer.")


def render_simulation_results(
    simulation: dict,
    params: WesterveltParams,
    params_payload: dict,
    init_payload: dict,
    numbers: dict[str, float | bool],
    scan_defaults: dict,
):
    status_cols = st.columns(4)
    status_cols[0].metric("max |u| final", f"{simulation['max_abs_u']:.5g}")
    status_cols[1].metric("min(1 - 2ku)", f"{simulation['min_denom']:.5g}")
    status_cols[2].metric("Etat numerique", "fini" if simulation["finite"] else "non fini")
    status_cols[3].metric("Moteur", simulation.get("engine", "solver"), f"{simulation.get('threads', 1)} thread(s)")

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


def main():
    st.set_page_config(page_title="Westervelt Lab", page_icon=None, layout="wide")

    params_payload, init_payload, snapshot_count, run_clicked, preset, scan_defaults, live_options = sidebar_controls()
    params = make_params(**params_payload)
    numbers = stability_numbers(params)
    total_time = params.nt * params.dt
    domain_length = params.dx * (params.nx - 1)

    live_run = bool(run_clicked and live_options["enabled"])
    header_area = st.empty()
    live_area = st.empty()
    results_area = st.empty()

    if not live_run:
        with header_area.container():
            render_header(params, numbers, total_time, domain_length)

    if run_clicked:
        snapshot_times = snapshot_times_for_preset(preset, total_time, snapshot_count)
        if live_run:
            header_area.empty()
            results_area.empty()
            if NUMBA_AVAILABLE:
                with st.spinner("Compilation Numba..."):
                    warm_up_numba_cached()
            st.session_state["last_simulation"] = run_simulation_numba(
                params_payload,
                init_payload,
                snapshot_times,
                render_every=live_options["render_every"],
                live_container=live_area,
            )
            live_area.empty()
        else:
            with st.spinner("Simulation en cours..."):
                st.session_state["last_simulation"] = run_simulation_cached(params_payload, init_payload, snapshot_times)
        st.session_state["last_payload"] = (params_payload, init_payload)

    simulation = st.session_state.get("last_simulation")

    if simulation is None:
        with results_area.container():
            render_home_page(params, numbers)
        return

    if live_run:
        with header_area.container():
            render_header(params, numbers, total_time, domain_length)

    with results_area.container():
        render_simulation_results(simulation, params, params_payload, init_payload, numbers, scan_defaults)


if __name__ == "__main__":
    main()
