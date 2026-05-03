"""Validation study using the manufactured solution with Nikolic--Wohlmuth parameters.

Features:
- convergence study with Nikolic-like parameters
- convergence table display
- versioned figure saving via `save_figure_with_version`
- optional interactive display (`--mode show`) or saving (`--mode save`)
- additional criteria: non-dégénérescence, pointwise error, L2 error, energy history
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from core.symbolics import build_numerics_function
from core.validation import (
    convergence_study_manufactured,
    print_convergence_table,
    run_manufactured_case,
    compute_manufatured_errors,
)
from core.solver import WesterveltParams
from utils import save_figure_with_version


OUT_DIR = os.path.join('outputs', 'analysis', 'validate_nikolic')
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Validation study with manufactured solution (Nikolic parameters).')
    parser.add_argument(
        '--mode',
        choices=('save', 'show'),
        default='save',
        help='save: save versioned figures; show: display figures interactively (default: save)',
    )
    return parser.parse_args()


def finalize_figure(fig, base_name, mode='save', metadata=None, formats=None):
    """Either show or save a figure with versioned filenames."""
    if mode == 'show':
        plt.show()
        plt.close(fig)
        return {}

    paths = save_figure_with_version(
        fig,
        filename=base_name,
        output_dir=OUT_DIR,
        formats=formats or ['png', 'pdf'],
        metadata=metadata,
        dpi=200,
        tight_layout=False,
    )
    plt.close(fig)
    return paths


def plot_errors_loglog(results, mode='save', metadata=None):
    levels = sorted(results['errors_L2'].keys())
    dxs = [results['mesh_sizes'][N] for N in levels]
    err_L2 = [results['errors_L2'][N] for N in levels]
    err_H1 = [results['errors_H1'][N] for N in levels]
    err_grad = [results['errors_grad'][N] for N in levels]
    err_Linf = [results['errors_Linf'][N] for N in levels]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.loglog(dxs, err_L2, 'o-', label='L2')
    ax.loglog(dxs, err_H1, 's-', label='H1')
    ax.loglog(dxs, err_grad, 'd-', label='grad')
    ax.loglog(dxs, err_Linf, 'x-', label='Linf')

    # reference slopes (order 1 and 2)
    dx0 = dxs[0]
    err0 = err_L2[0]
    ax.loglog(dxs, [err0 * (dx/dx0) for dx in dxs], 'k--', label='slope 1')
    ax.loglog(dxs, [err0 * (dx/dx0)**2 for dx in dxs], 'k:', label='slope 2')

    ax.set_xlabel('dx')
    ax.set_ylabel('Erreur')
    ax.set_title('Etude de convergence (manufactured) - Nikolic params')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    return finalize_figure(fig, 'convergence_errors', mode=mode, metadata=metadata)


def plot_snapshots_compare(x, U_num, U_ref, times, mode='save', metadata=None):
    # plot a few snapshots side by side (numeric vs exact)
    n = len(times)
    fig, axs = plt.subplots(n, 1, figsize=(8, 2.5*n))
    if n == 1:
        axs = [axs]
    for i, t in enumerate(times):
        axs[i].plot(x, U_ref[i], 'r--', label='Exacte')
        axs[i].plot(x, U_num[i], 'b-', label='Numérique', alpha=0.8)
        axs[i].set_title(f't = {t:.3e} s')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    fig.tight_layout()
    return finalize_figure(fig, 'nikolic_case_snapshots', mode=mode, metadata=metadata)


def plot_pointwise_error(x, U_num, U_ref, t_index, mode='save', metadata=None):
    error = np.abs(U_num[t_index] - U_ref[t_index])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(x, error, 'g-')
    ax.set_xlabel('x')
    ax.set_ylabel('|u_num - u_ref|')
    ax.set_title(f'Erreur pointwise à t_index={t_index}')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    fig.tight_layout()
    return finalize_figure(fig, 'nikolic_case_final_error', mode=mode, metadata=metadata)


def main(mode='save', verbose=True):
    """
    Run the manufactured solution validation study with Nikolic parameters.

    Parameters
    ----------
    mode : str
        'save' to save versioned figures, 'show' to display interactively
    verbose : bool
        Whether to print convergence table and results (default: True)
    """
    if not isinstance(mode, str) or mode not in ('save', 'show'):
        raise ValueError(f"mode must be 'save' or 'show', got {mode}")
    args = parse_args()

    # Parameters from Nikolic notebook
    c = 1500.0
    rho0 = 1000.0
    beta = 3.5
    mu_v = 6e-6  # gives b = mu_v / rho0 = 6e-9
    T_final = 37e-6
    omega = 2.0 * np.pi / T_final


    funcs = build_numerics_function()

    # Override mode from CLI if provided (CLI takes precedence when called as script)
    if args.mode:
        mode = args.mode

    if verbose:
        print(f'\nRunning convergence study (Nikolic parameters) [mode={mode}]')
    results = convergence_study_manufactured(
        funcs=funcs,
        levels=[1,2,3],
        L=0.2,                # match notebook domain length
        T=T_final,
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        A=1e-3,
        omega=omega,  # one oscillation over [0, T_final]
        gamma=0.5,
        kappa=1e4,
        scheme='explicit',
        base_nx=50,
        cfl_factor=0.2,
    )

    if verbose:
        print_convergence_table(results)

    # Save plot of errors vs dx
    md = {
        'analysis': 'manufactured_validation_nikolic',
        'mode': mode,
        'levels': [1, 2, 3],
        'base_nx': 50,
        'T_final': T_final,
        'L': 0.2,
    }
    err_paths = plot_errors_loglog(results, mode=mode, metadata=md)
    if mode == 'save':
        print('Saved error plot:')
        for fmt, path in err_paths.items():
            print(f'  {fmt}: {path}')

    # Run a mid-resolution case and request snapshots at several times
    Nmid = 2
    case = results['cases'][Nmid]
    x = case['x']
    times = case['times']

    # pick a subset of times (start, mid, final)
    times_idx = [0, len(times)//2, len(times)-1]
    times_to_plot = [times[i] for i in times_idx]

    # run a fresh case with those times to ensure U_num returned as array
    dx_mid = results["mesh_sizes"][Nmid]
    dt_mid = results["time_steps"][Nmid]
    nx_mid = int(round(0.2 / dx_mid)) + 1
    nt_mid = int(round(T_final / dt_mid))

    params = WesterveltParams(
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        dx=dx_mid,
        dt=dt_mid,
        nx=nx_mid,
        nt=nt_mid,
        bc="dirichlet",
        scheme="explicit",
    )

    res = run_manufactured_case(params, funcs, A=1e-3, L=0.2, omega=omega, gamma=0.5, kappa=1e4, times_to_save=times_to_plot, store_energy=True)

    U_num = res['U_num']
    U_ref = res['U_ref']
    x = res['x']
    times_returned = res['times']

    snapshot_errors = compute_manufatured_errors(U_num, U_ref, results['mesh_sizes'][Nmid], bc_type='dirichlet')

    if verbose:
        print(f"\nSnapshots requested: {times_to_plot}")
        print(f"Returned {len(times_returned)} snapshots; U_num shape = {U_num.shape}")
        print('Snapshot error criteria:')
        for name, value in snapshot_errors.items():
            print(f'  {name}: {value:.6e}')

    # Plot compare snapshots
    out_pref = os.path.join(OUT_DIR, 'nikolic_case')
    snap_paths = plot_snapshots_compare(x, U_num, U_ref, times_returned, mode=mode, metadata=md)
    err_paths_snap = plot_pointwise_error(x, U_num, U_ref, -1, mode=mode, metadata=md)
    if mode == 'save':
        print('Saved snapshot comparison plot:')
        for fmt, path in snap_paths.items():
            print(f'  {fmt}: {path}')
        print('Saved final error plot:')
        for fmt, path in err_paths_snap.items():
            print(f'  {fmt}: {path}')

    # Additional criteria: non-degeneracy at final time
    k_param = results['cases'][Nmid]['solver'].param.k
    final_u_num = U_num[-1]
    min_denom = np.min(1.0 - 2.0 * k_param * final_u_num)
    max_point_error = np.max(np.abs(final_u_num - U_ref[-1]))
    l2_final = np.sqrt(results['mesh_sizes'][Nmid] * np.sum((final_u_num - U_ref[-1])**2))

    if verbose:
        print('\nAdditional criteria:')
        print(f"  min(1 - 2k u) at final time = {min_denom:.6e}")
        print(f"  max pointwise error at final time = {max_point_error:.6e}")
        print(f"  L2 error at final time = {l2_final:.6e}")

    criteria = {
        'min_denom': min_denom,
        'max_point_error': max_point_error,
        'l2_final': l2_final,
    }

    # Energy history if present
    en_paths = {}
    solver_for_case = res["solver"]
    if hasattr(solver_for_case, 'energy_history') and solver_for_case.energy_history:
        energy = solver_for_case.energy_history
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(np.arange(len(energy)) * results['time_steps'][Nmid], energy, '-o')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Discrete energy')
        ax.set_title('Energy history (sample case)')
        fig.tight_layout()
        en_paths = finalize_figure(fig, 'energy_history', mode=mode, metadata=md)
        if mode == 'save' and verbose:
            print('Saved energy history:')
            for fmt, path in en_paths.items():
                print(f'  {fmt}: {path}')
    elif verbose:
        print('No energy history available for the stored case.')

    return criteria, snapshot_errors, err_paths, snap_paths, err_paths_snap, en_paths


if __name__ == '__main__':
    args = parse_args()
    main(args.mode, verbose=True)

