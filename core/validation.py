# ./core/validation.py

"""
Outils de validation numérique pour le projet Westervelt Simulation.

Fournit des fonctionnalités pour:
- La génération de grilles spatio-temporelles.
- La validation par méthode des solutions fabriquées (MMS).
- Les études de convergence par raffinement de maillage (H-refinement).
- Le calcul et l'affichage de tables d'erreurs et d'ordres de convergence.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Callable, List, Optional, Iterable


from core.numerics import _apply_boundary
from core.solver import WesterveltSolver, WesterveltParams
from utils import (
    compute_linf_time_error,
    compute_error_metrics,
    compute_convergence_orders,
    save_error_table_csv,
    save_solution_npz,
    load_solution_npz,
)


def make_time_grid(T: float, dt: float) -> np.ndarray:
    """
    Crée une grille temporelle uniforme ajustée.

    Ajuste le pas de temps pour que le temps final T soit un multiple exact de dt.

    Args:
        T: Temps final (s).
        dt: Pas de temps cible (s).

    Returns:
        np.ndarray: Vecteur des instants temporels de 0 à T.
    """
    nt = int(np.ceil(T / dt))
    dt_adjusted = T / nt

    return np.arange(nt + 1) * dt_adjusted


def make_spatial_grid(L: float, nx: int) -> np.ndarray:
    """
    Crée une grille spatiale uniforme.

    Args:
        L: Longueur du domaine (m).
        nx: Nombre de points spatiaux.

    Returns:
        np.ndarray: Vecteur des coordonnées spatiales de 0 à L.
    """
    return np.linspace(0.0, L, nx)


def nx_from_level(N: int) -> int:
    """
    Calcule le nombre de points nx à partir d'un niveau de raffinement.

    La formule utilisée est nx = 100 * 2^(N-1) + 1.

    Args:
        N: Niveau de raffinement (1, 2, 3, ...).

    Returns:
        int: Nombre de points correspondants.
    """
    return 100 * 2 ** (N - 1) + 1


def nx_from_mesh_size(dx: float, L: float) -> int:
    """
    Calcule le nombre de points nx à partir du pas spatial dx.

    Args:
        dx: Pas spatial (m).
        L: Longueur totale (m).

    Returns:
        int: Nombre de points requis pour couvrir L avec un pas dx.
    """
    return int(np.ceil(L / dx)) + 1

# ------------------------------------------------------------------------------------------------------------------------
# VALIDATION PAR SOLUTION FABRIQUEE
# ------------------------------------------------------------------------------------------------------------------------

def initialize_manufactured_solver(
        solver: WesterveltSolver,
        funcs: Dict[str, Callable],
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
) -> None:
    """
    Initialise le solveur pour une solution fabriquée.

    Définit u(t=0) et calcule u(t=-dt) pour initialiser le schéma à deux pas
    en utilisant la dérivée temporelle exacte à t=0.

    Args:
        solver: Instance du solveur à initialiser.
        funcs: Dictionnaire contenant les fonctions 'u' et 'ut' exactes.
        A: Amplitude de la solution.
        L: Paramètre spatial de la solution.
        omega: Pulsation angulaire.
        gamma: Paramètre de forme temporelle.
        kappa: Paramètre de forme spatiale.
    """
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
        funcs: Dict[str, Callable],
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
        c: float,
        b: float,
        k: float,
) -> Callable:
    """
    Crée une fonction source f(t, x) pour la solution fabriquée.

    Args:
        funcs: Dictionnaire contenant la fonction de source 'f'.
        A: Amplitude.
        L: Paramètre L.
        omega: Pulsation.
        gamma: Paramètre gamma.
        kappa: Paramètre kappa.
        c: Vitesse du son.
        b: Paramètre de viscosité.
        k: Paramètre de non-linéarité.

    Returns:
        Callable: Fonction f(t, x) prête à être utilisée par le solveur.
    """
    f = funcs['f']

    def source(t, x):
        return f(x, t, A, L, omega, gamma, kappa, c, b, k)

    return source


def evaluate_exact_solution(
        funcs: Dict[str, Callable],
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
    """
    Évalue la solution exacte sur une grille spatio-temporelle.

    Args:
        funcs: Dictionnaire contenant la fonction 'u' exacte.
        x: Grille spatiale.
        times: Grille temporelle.
        A, L, omega, gamma, kappa: Paramètres de la solution.
        c, b, k: Paramètres physiques.

    Returns:
        np.ndarray: Matrice (nt, nx) contenant la solution exacte.
    """
    u_exact = funcs['u']

    return np.array([u_exact(x, t, A, L, omega, gamma, kappa, c, b, k) for t in times])


def run_manufactured_case(
        params: WesterveltParams,
        funcs: Dict[str, Callable],
        A: float,
        L: float,
        omega: float,
        gamma: float,
        kappa: float,
        times_to_save: Iterable[float] | None = None,
        store_energy: bool = False,
) -> Dict[str, Any]:
    """
    Exécute une simulation complète avec une solution fabriquée.

    Lance le solveur avec une source forcée et compare le résultat numérique
    avec la solution analytique.

    Args:
        params: Paramètres de la simulation.
        funcs: Dictionnaire des fonctions analytiques ('u', 'ut', 'f').
        A, L, omega, gamma, kappa: Paramètres de la solution fabriquée.
        times_to_save: Instants auxquels sauvegarder la solution (optionnel).
        store_energy: Si True, enregistre l'évolution de l'énergie.

    Returns:
        Dict[str, Any]: Dictionnaire contenant:
            - 'solver': Instance du solveur.
            - 'x': Grille spatiale.
            - 'times': Grille temporelle.
            - 'U_num': Solution numérique (nt, nx).
            - 'U_ref': Solution exacte (nt, nx).
            - 'snapshots': Dictionnaire des snapshots.
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


def compute_manufactured_errors(
        U_num: np.ndarray,
        U_ref: np.ndarray,
        dx: float,
        bc_type: str = "dirichlet",
) -> Dict[str, float]:
    """
    Calcule les métriques d'erreur pour une solution fabriquée.

    Args:
        U_num: Solution numérique.
        U_ref: Solution de référence.
        dx: Pas spatial.
        bc_type: Conditions aux limites.

    Returns:
        Dict[str, float]: Erreurs Linf en temps pour diverses normes spatiales (L2, H1, grad, Linf).
    """
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


def compute_manufactured_error_norm_over_time(
        U_num: np.ndarray,
        U_ref: np.ndarray,
        dx: float,
        norm_type: str = "L2",
        bc_type: str = "dirichlet",
) -> np.ndarray:
    values = []

    for u_num, u_ref in zip(U_num, U_ref):
        metrics = compute_error_metrics(
            u_num,
            u_ref,
            dx=dx,
            compute_l2=(norm_type == "L2"),
            compute_h1=(norm_type == "H1"),
            compute_linf=(norm_type == "Linf"),
            bc_type=bc_type,
        )

        if norm_type == "grad":
            diff = u_num - u_ref
            grad = (diff[2:] - diff[:-2]) / (2.0 * dx)
            value = np.sqrt(dx * np.sum(grad**2))
        else:
            value = metrics[norm_type]

        values.append(value)

    return np.asarray(values)


def convergence_study_manufactured(
        funcs: Dict[str, Callable],
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
    Réalise une étude de convergence complète (MMS).

    Exécute la simulation pour plusieurs niveaux de raffinement et calcule
    les erreurs et les ordres de convergence.

    Args:
        funcs: Fonctions de la solution fabriquée.
        levels: Liste des niveaux de raffinement N.
        L, T: Paramètres du domaine.
        c, rho0, beta, mu_v: Paramètres physiques.
        A, omega, gamma, kappa: Paramètres de la solution.
        scheme: Schéma numérique.
        base_nx: Nombre de points de base pour nx.
        dt_mode: 'cfl' ou 'quadratic'.
        dt_factor: Facteur de sécurité pour dt.

    Returns:
        Dict[str, Any]: Résultats complets incluant erreurs, ordres et cas individuels.
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

        errs = compute_manufactured_errors(
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


def build_manufactured_convergence_table(results: Dict[str, Any],) -> pd.DataFrame:
    """
    Construit un DataFrame pandas récapitulant les erreurs et ordres de convergence.

    Args:
        results: Dictionnaire issu de `convergence_study_manufactured`.

    Returns:
        pd.DataFrame: Tableau contenant N, dx, dt, les erreurs et les ordres pour chaque métrique.
    """
    rows = []

    levels = sorted(results["errors_L2"].keys())

    for N in levels:
        rows.append({
            "N": N,
            "dx": results["mesh_sizes"][N],
            "dt": results["time_steps"][N],
            "Linf_L2": results["errors_L2"][N],
            "order_Linf_L2": results["orders_L2"].get(N, np.nan),
            "Linf_H1": results["errors_H1"][N],
            "order_Linf_H1": results["orders_H1"].get(N, np.nan),
            "Linf_grad": results["errors_grad"][N],
            "order_Linf_grad": results["orders_grad"].get(N, np.nan),
            "Linf_Linf": results["errors_Linf"][N],
            "order_Linf_Linf": results["orders_Linf"].get(N, np.nan),
        })

    return pd.DataFrame(rows)


def print_convergence_table_manufactured(results: Dict[str, Any]) -> None:
    """
    Affiche un tableau formaté des erreurs et ordres de convergence (MMS).

    Args:
        results: Dictionnaire issu de `convergence_study_manufactured`.
    """

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
    print("-" * 132)
    print(
        f"{'N':>4} | {'dx':>12} | {'dt':>12} | "
        f"{'LinfL2':>12} | {'ord':>6} | "
        f"{'H1':>12} | {'ord':>6} | "
        f"{'LinfGrad':>12} | {'ord':>6} | "
        f"{'Linf':>12} | {'ord':>6}"
    )
    print("-" * 132)

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

    print("-" * 132)

# ------------------------------------------------------------------------------------------------------------------------
# VALIDATION PAR RAFFINEMENT DU MAILLAGE
# ------------------------------------------------------------------------------------------------------------------------

def check_nested_grids(
        nx_coarse: int,
        nt_coarse: int,
        nx_fine: int,
        nt_fine: int,
) -> Tuple[int, int]:
    """
    Vérifie que deux grilles sont imbriquées.

    Args:
        nx_coarse: Nombre de points spatiaux de la grille grossière.
        nt_coarse: Nombre d'itérations temporelles de la grille grossière.
        nx_fine: Nombre de points de la grille fine.
        nt_fine: Nombre d'itérations de la grille fine.

    Returns:
        Tuple[int, int]: (ratio_nx, ratio_nt) entre les deux grilles.

    Raises:
        ValueError: Si les grilles ne sont pas imbriquées.
    """
    if nx_coarse < 2 or nx_fine < 2:
        raise ValueError("nx_coarse et nx_fine doivent être au moins égaux à 2.")

    if nt_coarse < 1 or nt_fine < 1:
        raise ValueError("nt_coarse et nt_fine doivent être au moins égaux à 1.")

    if (nx_fine - 1) % (nx_coarse - 1) != 0:
        raise ValueError(
            f"Grilles spatiales non imbriquées : "
            f"nx_fine-1={nx_fine - 1} n'est pas un multiple de "
            f"nx_coarse-1={nx_coarse - 1}."
        )

    if nt_fine % nt_coarse != 0:
        raise ValueError(
            f"Grilles temporelles non imbriquées : "
            f"nt_fine={nt_fine} n'est pas un multiple de nt_coarse={nt_coarse}."
        )

    return (nx_fine - 1) // (nx_coarse - 1), nt_fine // nt_coarse


def restrict_fine_to_coarse(
        U_fine: np.ndarray,
        nx_coarse: int,
        nt_coarse: int,
        nx_fine: int,
        nt_fine: int,
) -> np.ndarray:
    """
    Restreint une solution d'une grille fine vers une grille grossière.

    Args:
        U_fine: Solution sur grille fine.
        nx_coarse, nt_coarse: Dimensions cibles.
        nx_fine, nt_fine: Dimensions sources.

    Returns:
        np.ndarray: Solution restreinte.
    """
    ratio_nx, ratio_nt = check_nested_grids(nx_coarse, nt_coarse, nx_fine, nt_fine)

    return U_fine[::ratio_nt, ::ratio_nx]


def compute_relative_linf_l2_error(
        U_coarse: np.ndarray,
        U_fine_restricted: np.ndarray,
        dx: float,
        eps: float = 1e-14,
) -> float:
    error = U_coarse - U_fine_restricted

    err_l2_t = np.sqrt(np.sum(error ** 2, axis=1))
    ref_l2_t = np.sqrt(np.sum(U_fine_restricted ** 2, axis=1))

    return np.max(err_l2_t / (ref_l2_t + eps))

def compute_refinement_error(
        U_coarse: np.ndarray,
        U_fine_restricted: np.ndarray,
        dx: float,
        bc_type: str = "dirichlet",
) -> Dict[str, float]:
    """
    Calcule l'erreur entre une solution grossière et une solution fine restreinte.

    Args:
        U_coarse: Solution calculée sur grille grossière.
        U_fine_restricted: Solution fine restreinte à la grille grossière.
        dx: Pas spatial.
        bc_type: Type de conditions aux limites.

    Returns:
        Dict[str, float]: Dictionnaire des erreurs.
    """
    return {
        "Linf_L2": compute_linf_time_error(
            U_coarse, U_fine_restricted, dx, norm_type="L2", bc_type=bc_type
        ),
        "Linf_rel_L2": compute_relative_linf_l2_error(
            U_coarse, U_fine_restricted, dx,
        ),
        "Linf_H1": compute_linf_time_error(
            U_coarse, U_fine_restricted, dx, norm_type="H1", bc_type=bc_type
        ),
        "Linf_grad": compute_linf_time_error(
            U_coarse, U_fine_restricted, dx, norm_type="grad", bc_type=bc_type
        ),
        "Linf_Linf": compute_linf_time_error(
            U_coarse, U_fine_restricted, dx, norm_type="Linf", bc_type=bc_type
        ),
    }


def run_case_direct(
        nx: int,
        nt: int,
        T_final: float = 37e-6,
        L: float = 0.2,
        c: float = 1500.0,
        rho0: float = 1000.0,
        beta: float = 3.5,
        mu_v: float = 6e-6,
        A1: float = 1.2e8,
        A2: float = 1.0e11,
        scheme: str = "semi_implicit",
        bc: str = "dirichlet",
        store_energy: bool = False,
) -> Dict[str, Any]:
    """
    Exécute un cas de simulation direct avec des paramètres spécifiés.

    Args:
        nx, nt: Nombre de points spatiaux et temporels.
        T_final, L: Paramètres du domaine.
        c, rho0, beta, mu_v: Paramètres physiques.
        scheme: Schéma numérique.
        bc: Conditions aux limites.
        store_energy: Enregistrer l'énergie.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les résultats et paramètres du cas.
    """
    dx = L / (nx - 1)
    dt = T_final / nt

    params = WesterveltParams(
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        dx=dx,
        dt=dt,
        nx=nx,
        nt=nt,
        bc=bc,
        scheme=scheme,
    )

    solver = WesterveltSolver(params)

    solver.initialize(
        u0_type="gaussian",
        u1_type="gaussian_zero_mean",
        A1=A1,
        A2=A2,
        mu=0.1,
        sigma1=0.015,
        sigma2=0.02,
    )

    times = np.arange(nt + 1) * dt

    snapshots = solver.run_with_snapshots(times, store_energy=store_energy)

    U = np.array([snapshots[t] for t in times])

    return {
        "solver": solver,
        "params": params,
        "x": solver.x,
        "times": times,
        "U": U,
        "dx": dx,
        "dt": dt,
        "nx": nx,
        "nt": nt,
        "T_final": T_final,
        "L": L,
        "scheme": scheme,
        "bc": bc,
    }


def refinement_validation_direct(
        coarse: dict,
        fine: dict,
        bc_type: str = "dirichlet",
) -> Dict[str, Any]:
    """
    Valide une solution grossière par rapport à une solution plus fine.

    Réalise la restriction de la solution fine et calcule les erreurs de différence.

    Args:
        coarse: Dictionnaire de résultats du cas grossier.
        fine: Dictionnaire de résultats du cas fin (référence).
        bc_type: Type de conditions aux limites.

    Returns:
        Dict[str, Any]: Comparaison détaillée incluant les erreurs.
    """
    U_fine_restricted = restrict_fine_to_coarse(
        U_fine=fine["U"],
        nx_coarse=coarse["nx"],
        nt_coarse=coarse["nt"],
        nx_fine=fine["nx"],
        nt_fine=fine["nt"],
    )

    if coarse['U'].shape != U_fine_restricted.shape:
        raise ValueError(
            f"Les solutions ne sont pas comparables après restriction : U_coarse.shape={coarse['U'].shape}, U_fine_restricted.shape={U_fine_restricted.shape}"
        )

    error = coarse['U'] - U_fine_restricted

    errors = compute_refinement_error(
        U_coarse=coarse['U'],
        U_fine_restricted=U_fine_restricted,
        dx=coarse['dx'],
        bc_type=bc_type,
    )

    return {
        "U_coarse": coarse['U'],
        "U_fine_restricted": U_fine_restricted,
        "error": error,
        "errors": errors,
    }


def convergence_study_refinement(
        levels: List[Tuple[int, int]],
        T_final: float = 37e-6,
        L_final: float = 0.2,
        scheme: str = "explicit",
        bc: str = "dirichlet",
        store_energy: bool = False,
) -> Dict[str, Any]:
    """
    Réalise une étude de convergence par raffinement successif.

    Compare chaque niveau de la liste `levels` au niveau suivant le plus fin (ou au dernier).

    Args:
        levels: Liste de tuples (nx, nt) à tester.
        T_final, L_final: Domaine spatio-temporel.
        scheme: Schéma numérique.
        bc: Conditions aux limites.
        store_energy: Enregistrer l'énergie.

    Returns:
        Dict[str, Any]: Résultats de l'étude de convergence.
    """
    cases = {}

    for nx, nt in levels:
        case = run_case_direct(
            nx=nx,
            nt=nt,
            T_final=T_final,
            L=L_final,
            scheme=scheme,
            bc=bc,
            store_energy=store_energy,
        )
        cases[(nx, nt)] = case

    nx_ref, nt_ref = levels[-1]
    fine = cases[(nx_ref, nt_ref)]

    rows = []
    results = {}

    for nx, nt in levels[:-1]:
        coarse = cases[(nx, nt)]

        comparison = refinement_validation_direct(
            coarse=coarse,
            fine=fine,
            bc_type=bc,
        )

        row = {
            "nx": nx,
            "nt": nt,
            "dx": coarse["dx"],
            "dt": coarse["dt"],
            "nx_ref": nx_ref,
            "nt_ref": nt_ref,
            **comparison["errors"],
        }

        rows.append(row)
        results[(nx, nt)] = comparison

    return {
        "cases": cases,
        "reference": fine,
        "comparisons": results,
        "rows": rows,
    }


def build_convergence_table_refinement(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Construit un DataFrame pandas récapitulant la convergence.

    Calcule automatiquement les ordres de convergence pour toutes les colonnes d'erreur.

    Args:
        rows: Liste de dictionnaires (lignes de données).

    Returns:
        pd.DataFrame: Tableau de convergence avec ordres calculés.
    """
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("dx", ascending=False).reset_index(drop=True)

    error_columns = [col for col in df.columns if col.startswith("Linf")]

    for col in error_columns:
        error_dict = {
            row["nx"]: row[col] for _, row in df.iterrows()
        }

        orders = compute_convergence_orders(error_dict)

        order_values = [orders.get(nx, np.nan) for nx in df["nx"]]

        df[f"order_{col}"] = order_values

    return df


def scan_critical_amplitudes(
        amplitudes,
        nx: int = 801,
        nt: int = 1000,
        scheme: str = "explicit",
        bc: str = "dirichlet",
        T_final: float = 37e-6,
        L: float = 0.2,
        alpha_tol: float = 1e-12,
        A2_factor: float = 1e3,
        stop_after_degeneracy: bool = False,
) -> pd.DataFrame:
    """
    Scanne les amplitudes initiales afin d'évaluer la non-dégénérescence
    numérique du modèle de Westervelt.

    Pour chaque amplitude A1, on lance une simulation et on calcule

        alpha(t,x) = 1 - 2 k u(t,x).

    Le critère retourné n'est pas une stabilité de Von Neumann. Il indique
    seulement si la simulation reste dans le régime non dégénéré au sens où

        min_{t,x} alpha(t,x) > alpha_tol.

    Args:
        amplitudes: suite d'amplitudes A1 à tester.
        nx, nt: résolution spatiale et temporelle.
        scheme: schéma numérique utilisé.
        bc: condition aux limites.
        T_final, L: domaine spatio-temporel.
        alpha_tol: marge de non-dégénérescence.
        A2_factor: rapport A2/A1 pour la vitesse initiale.
        stop_after_degeneracy: si True, arrête le scan après la première
            amplitude dégénérée.

    Returns:
        pd.DataFrame avec A1, A2, alpha_min, alpha_max, umax, non_degenerate.
    """
    rows = []

    for A1 in amplitudes:
        A1 = float(A1)
        A2 = float(A2_factor * A1)

        try:
            case = run_case_direct(
                nx=nx,
                nt=nt,
                T_final=T_final,
                L=L,
                A1=A1,
                A2=A2,
                scheme=scheme,
                bc=bc,
            )

            U = case["U"]
            k = case["params"].k

            finite = bool(np.isfinite(U).all())

            if finite:
                alpha = 1.0 - 2.0 * k * U
                alpha_min = float(np.min(alpha))
                alpha_max = float(np.max(alpha))
                umax = float(np.max(np.abs(U)))
                u_at_alpha_min = float(U[np.argmin(alpha)])
            else:
                alpha_min = np.nan
                alpha_max = np.nan
                umax = np.nan
                u_at_alpha_min = np.nan

            non_degenerate = finite and alpha_min > alpha_tol

        except Exception as exc:
            finite = False
            alpha_min = np.nan
            alpha_max = np.nan
            umax = np.nan
            u_at_alpha_min = np.nan
            non_degenerate = False
            error_message = str(exc)
        else:
            error_message = ""

        rows.append({
            "A1": A1,
            "A2": A2,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "umax": umax,
            "u_at_alpha_min": u_at_alpha_min,
            "finite": finite,
            "non_degenerate": non_degenerate,
            "alpha_tol": alpha_tol,
            "scheme": scheme,
            "error": error_message,
        })

        if stop_after_degeneracy and not non_degenerate:
            break

    return pd.DataFrame(rows)



