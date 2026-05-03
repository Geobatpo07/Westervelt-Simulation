# ./core/validation.py

import numpy as np
from typing import Dict, Any, Tuple, Callable, List, Optional, Iterable

from core.numerics import _apply_boundary
from core.solver import WesterveltSolver, WesterveltParams
from utils.utils import compute_linf_time_error, compute_convergence_orders


def make_time_grid(T: float, dt: float) -> np.ndarray:
    """Crée une grille temporelle de 0 à T avec un pas de dt ajusté pour que T soit un multiple de dt."""
    nt = int(np.ceil(T / dt))
    dt_adjusted = T / nt

    return np.arange(nt + 1) * dt_adjusted

def initialize_manufactured_solver(
        solver: WesterveltSolver,
        funcs: Dict[str, callable],
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
) -> None:
    """Initialise le solveur avec la solution fabriquée à t=0 et sa dérivée temporelle."""
    c = solver.param.c
    b = solver.param.b
    k = solver.param.k

    u_exact = funcs['u']
    ut_exact = funcs['ut']

    u0 = u_exact(solver.x, 0.0, A, L, omega, gamma, kappa, c, b, k)
    u1 = ut_exact(solver.x, 0.0, A, L, omega, gamma, kappa, c, b, k)

    solver.u = np.asarray(u0, dtype=float).copy()
    _apply_boundary(solver.u, solver.bc_type)

    u1 = np.asarray(u1, dtype=float).copy()
    _apply_boundary(u1, solver.bc_type)

    solver.u_prev = solver.u - solver.param.dt * u1
    _apply_boundary(solver.u_prev, solver.bc_type)

    solver.reset_auxiliary_field(u_t0=u1)

    solver.energy_history = [solver.compute_energy()]


def make_manufactured_source(
        funcs: Dict[str, callable],
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
        c: float,
        b: float,
        k: float,
):
    """Crée une fonction source f(x, t) à partir de la solution fabriquée."""
    f = funcs['f']

    def source(x, t):
        return f(x, t, A, L, omega, gamma, kappa, c, b, k)

    return source


def evaluate_exact_solution(
        funcs: Dict[str, callable],
        x: np.ndarray,
        times: np.ndarray,
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
        c: float,
        b: float,
        k: float,
) -> np.ndarray:
    """Évalue la solution exacte u(x, t) à partir de la fonction fabriquée."""
    u_exact = funcs['u']

    return np.array([u_exact(x, t, A, L, omega, gamma, kappa, c, b, k) for t in times])


def run_manufactured_case(
        params: WesterveltParams,
        funcs: Dict[str, callable],
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
        times_to_save: Iterable[float] | None = None,
        store_energy: bool = False,
) -> Dict[str, Any]:
    """
    Lance une simulation forcée par la source fabriquée.

    Retourne :
        - solver
        - x
        - times
        - U_num
        - U_ref
        - snapshots
    """
    solver = WesterveltSolver(params)

    initialize_manufactured_solver(solver, funcs, A, L, omega, gamma, kappa)

    if times_to_save is None:
        times = np.arange(params.nt + 1) * params.dt
    else:
        times = np.asarray(list(times_to_save), dtype=float)

    source = make_manufactured_source(funcs, A, L, omega, gamma, kappa, params.c, params.b, params.k)

    snapshots = solver.run_with_snapshots(times, source=source, store_energy=store_energy)

    U_num = np.array([snapshots[t] for t in times])

    U_ref = evaluate_exact_solution(funcs, solver.x, times, A, L, omega, gamma, kappa, params.c, params.b, params.k)

    return {
        'solver': solver,
        'x': solver.x,
        'times': times,
        'U_num': U_num,
        'U_ref': U_ref,
        'snapshots': snapshots,
    }


def compute_manufatured_errors(
        U_num: np.ndarray,
        U_ref: np.ndarray,
        dx: float,
        bc_type: str = "dirichlet",
) -> Dict[str, float]:
    return {
        "Linf_L2": compute_linf_time_error(
            U_num, U_ref, dx, norm_type="L2", bc_type=bc_type
        ),
        "Linf_H1": compute_linf_time_error(
            U_num, U_ref, dx, norm_type="H1", bc_type=bc_type
        ),
        "Linf_grad": compute_linf_time_error(
            U_num, U_ref, dx, norm_type="grad", bc_type=bc_type
        ),
        "Linf_Linf": compute_linf_time_error(
            U_num, U_ref, dx, norm_type="Linf", bc_type=bc_type
        )
    }


def convergence_study_manufactured(
        funcs: Dict[str, callable],
        levels: Iterable[int],
        L: float = 1.0,
        T: float = 1e-4,
        c: float = 1500.0,
        rho0: float = 1000.0,
        beta: float = 3.5,
        mu_v: float = 6e-6,
        A: float = 1e-3,
        omega: float = 2.0 * np.pi * 1e4,
        gamma: float = 1.0,
        kappa: float = 1e4,
        scheme: str = "explicit",
        base_nx: int = 50,
        dt_mode: str = "cfl",
        dt_factor: float = 0.2,
) -> Dict[str, Any]:
    """
    Étude de convergence avec solution fabriquée.

    levels:
        niveaux de raffinement N.
        nx = base_nx * 2**N + 1
    """
    errors_L2 = {}
    errors_H1 = {}
    errors_grad = {}
    errors_Linf = {}
    mesh_sizes = {}
    times_steps = {}

    cases = {}

    for N in levels:
        nx = base_nx * 2 ** N + 1
        dx = L / (nx - 1)

        if dt_mode == "cfl":
            dt = dt_factor * dx / c
        elif dt_mode == "quadratic":
            dt = dt_factor * dx ** 2
        else:
            raise ValueError(f"Mode de temps inconnu : {dt_mode} | Choix : cfl, quadratic")

        nt = int(np.ceil(T / dt))
        dt = T / nt

        params = WesterveltParams(
            c=c,
            rho0=rho0,
            beta=beta,
            mu_v=mu_v,
            dx=dx,
            dt=dt,
            nx=nx,
            nt=nt,
            bc="dirichlet",
            scheme=scheme,
        )

        times = np.arange(nt + 1) * dt

        case = run_manufactured_case(
            params, funcs, A, L, omega, gamma, kappa,
            times_to_save=times,
            store_energy=False,
        )

        errs = compute_manufatured_errors(
            case["U_num"], case["U_ref"], dx, bc_type="dirichlet"
        )

        errors_L2[N] = errs["Linf_L2"]
        errors_H1[N] = errs["Linf_H1"]
        errors_grad[N] = errs["Linf_grad"]
        errors_Linf[N] = errs["Linf_Linf"]
        mesh_sizes[N] = dx
        times_steps[N] = dt
        cases[N] = case

    orders_L2 = compute_convergence_orders(errors_L2)
    orders_H1 = compute_convergence_orders(errors_H1)
    orders_grad = compute_convergence_orders(errors_grad)
    orders_Linf = compute_convergence_orders(errors_Linf)

    return {
        "errors_L2": errors_L2,
        "errors_H1": errors_H1,
        "errors_grad": errors_grad,
        "errors_Linf": errors_Linf,
        "orders_L2": orders_L2,
        "orders_H1": orders_H1,
        "orders_grad": orders_grad,
        "orders_Linf": orders_Linf,
        "mesh_sizes": mesh_sizes,
        "time_steps": times_steps,
        "cases": cases,
    }


def print_convergence_table(results: Dict[str, Any]) -> None:
    """Affiche un tableau simple des erreurs et ordres."""

    errors_L2 = results["errors_L2"]
    errors_H1 = results["errors_H1"]
    errors_grad = results["errors_grad"]
    errors_Linf = results["errors_Linf"]

    orders_L2 = results["orders_L2"]
    orders_H1 = results["orders_H1"]
    orders_grad = results["orders_grad"]
    orders_Linf = results["orders_Linf"]

    mesh_sizes = results["mesh_sizes"]
    time_steps = results["time_steps"]

    levels = sorted(errors_L2.keys())

    print("\nTable de convergence - solution fabriquée")
    print("-" * 130)
    print(
        f"{'N':>4} | {'dx':>12} | {'dt':>12} | "
        f"{'LinfL2':>12} | {'ord':>6} | "
        f"{'H1':>12} | {'ord':>6} | "
        f"{'LinfGrad':>12} | {'ord':>6} | "
        f"{'Linf':>12} | {'ord':>6}"
    )
    print("-" * 130)

    for N in levels:
        o_l2 = orders_L2.get(N, np.nan)
        o_h1 = orders_H1.get(N, np.nan)
        o_grad = orders_grad.get(N, np.nan)
        o_linf = orders_Linf.get(N, np.nan)

        print(
            f"{N:>4} | "
            f"{mesh_sizes[N]:>12.4e} | "
            f"{time_steps[N]:>12.4e} | "
            f"{errors_L2[N]:>12.4e} | "
            f"{o_l2:>6.3f} | "
            f"{errors_H1[N]:>12.4e} | "
            f"{o_h1:>6.3f} |"
            f"{errors_grad[N]:>12.4e} | "
            f"{o_grad:>6.3f} | "
            f"{errors_Linf[N]:>12.4e} | "
            f"{o_linf:>6.3f}"
        )

    print("-" * 130)
