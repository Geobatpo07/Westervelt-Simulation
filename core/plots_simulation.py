# ./core/plots_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from core.symbolics import build_numerics_function

from core.validation import (
    run_case_direct,
    refinement_validation_direct,
    convergence_study_refinement,
    build_convergence_table_refinement,
    compute_manufactured_error_norm_over_time,
    convergence_study_manufactured,
    build_manufactured_convergence_table,
    scan_critical_amplitudes
)
from utils import (
    set_style,
    save_figure_with_version,
    save_error_table_csv, compute_linf_time_error
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ------------------------------------------------------------------------------------------------------------------------
# VALIDATION PAR RAFFINEMENT DU MAILLAGE
# ------------------------------------------------------------------------------------------------------------------------

def plot_snapshots(case, title="Solution numérique", n_snapshots=5):
    x = case['x']
    times = case['times']
    U = case['U']

    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in indices:
        ax.plot(x, U[i], label=f't = {times[i]:.3e} s')

    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=10)

    return fig


def plot_coarse_vs_fine(comparison, coarse, n_snapshots, title="Comparaison fine vs coarse"):
    x = coarse['x']
    times = coarse['times']

    U_coarse = comparison['U_coarse']
    U_fine_restricted = comparison['U_fine_restricted']

    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in indices:
        ax.plot(x, U_coarse[i], linestyle='-', label=f'U coarse (t = {times[i]:.3e} s)')
        ax.plot(x, U_fine_restricted[i], linestyle='--', label=f'U fine (t = {times[i]:.3e} s)')

    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig


def plot_error_snapshots(comparison, coarse, n_snapshots=5):
    x = coarse['x']
    times = coarse['times']
    error = comparison['error']

    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in indices:
        ax.plot(x, error[i], label=f't = {times[i]:.3e} s')

    ax.set_xlabel('x')
    ax.set_ylabel('Erreur')
    ax.set_title('Erreur sur les snapshots')
    ax.grid(True)
    ax.legend()

    return fig


def plot_error_map(comparison, coarse):
    x = coarse['x']
    times = coarse['times']
    error = comparison['error']

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(error, cmap='RdYlBu_r', extent=[x[0], x[-1], times[0], times[-1]], aspect='auto')

    fig.colorbar(im, ax=ax, label='Erreur')
    ax.set_aspect('auto')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Carte spatio-temporelle des erreurs')
    ax.grid(False)

    return fig


def plot_convergence_table(df, error_col='Linf_L2'):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(df['dx'], df[error_col], 'o-', label='Erreur', markersize=6)

    ax.set_xlabel('Pas d\'espace dx')
    ax.set_ylabel('Erreur')
    ax.set_title(f'Tableau de convergence - {error_col.replace("_", " ")}')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    return fig


def plot_alpha_min_vs_amplitude(df_amp, alpha_tol=1e-6, zoom=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogx(df_amp['A1'], df_amp['alpha_min'], 'o-', label=r'$\min((1-2ku)$', markersize=6)
    ax.axhline(0.0, linestyle='--', linewidth=1, label=rf'$\alpha = 0$')
    ax.axhline(alpha_tol, linestyle=':', linewidth=1, color='k', label=rf'$\alpha = {alpha_tol:.1e}$')

    if zoom:
        ax.set_ylim(-1.0, 1.0)

    ax.set_xlabel(r'Amplitude initiale $A_1$')
    ax.set_ylabel(r'$\min_{t,x}(1-2ku)$')
    ax.set_title("Recherche de l'amplitude critique")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    return fig


def plot_error_vs_amplitude(df_amp):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(df_amp['A1'], df_amp['Linf_L2'], 'o-', label=r'Erreur $L^2$', markersize=6)
    ax.loglog(df_amp['A1'], df_amp['Linf_rel_L2'], 's-', label=r'Erreur relative $L^2$', markersize=6)

    ax.set_xlabel(r'Amplitude $A_1$')
    ax.set_ylabel(r'Erreur $L^2$')
    ax.set_title("Sensibilité du schéma à l'amplitude")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    return fig


def run_refinement_plots(
        scheme: str = "explicit",
        bc: str = "dirichlet",
        theme: str = "scientific",
        show: bool = True,
        save: bool = True,
        save_path: pathlib.Path | str | None = None
) -> dict:

    set_style(theme_name=theme)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Exécution des cas de validation
    case_coarse = run_case_direct(nx=2001, nt=2000, scheme=scheme)
    case_fine = run_case_direct(nx=10001, nt=10000, scheme=scheme)
    comparison = refinement_validation_direct(coarse=case_coarse, fine=case_fine, bc_type=bc)
    df_convergence = convergence_study_refinement(
        levels = [
            (101, 1000),
            (201, 2000),
            (401, 4000),
            (801, 8000),
            (1601, 16000),
            (3201, 32000),
            (6401, 64000),
            # (12801, 128000),
            # (25601, 256000),
            # (51201, 512000),
            # (102401, 1024000),
        ],
        scheme = scheme
    )
    table_convergence = build_convergence_table_refinement(df_convergence["rows"])

    amplitudes = np.logspace(3, 9, 25)
    df_amp = scan_critical_amplitudes(amplitudes, nx=2001, nt=2000, scheme=scheme)

    # Plots
    fig1 = plot_snapshots(case_coarse, title="Solution numérique - maille grossière")
    fig2 = plot_snapshots(case_fine, title="Solution numérique - maille fine")
    fig3 = plot_coarse_vs_fine(comparison, case_coarse, n_snapshots=5)
    fig4 = plot_error_snapshots(comparison, case_coarse, n_snapshots=5)
    fig5 = plot_error_map(comparison, case_coarse)
    fig6 = plot_convergence_table(table_convergence, error_col="Linf_L2")
    fig7 = plot_alpha_min_vs_amplitude(df_amp)
    fig8 = plot_alpha_min_vs_amplitude(df_amp, zoom=True)
    # fig8 = plot_error_vs_amplitude(df_amp)

    if save and save_path is not None:
        save_figure_with_version(fig1, "refinement_snapshots_coarse", output_dir=save_path)
        save_figure_with_version(fig2, "refinement_snapshots_fine", output_dir=save_path)
        save_figure_with_version(fig3, "refinement_comparison", output_dir=save_path)
        save_figure_with_version(fig4, "refinement_error_snapshots", output_dir=save_path)
        save_figure_with_version(fig5, "refinement_error_map", output_dir=save_path)
        save_figure_with_version(fig6, "refinement_convergence_Linf_L2", output_dir=save_path)
        save_figure_with_version(fig7, "refinement_alpha_min_vs_amplitude", output_dir=save_path)
        # save_figure_with_version(fig8, "refinement_error_vs_amplitude", output_dir=save_path)

        save_error_table_csv(
            save_path / "convergence_table_refinement.csv",
            table_convergence.to_dict(orient="records"),
        )

    print(f'Table de convergence :\n{table_convergence[["dx", "Linf_L2", "Linf_rel_L2", "order_Linf_L2"]]} ')

    if show:
        plt.show()
    else:
        plt.close('all')

    return {
        "case_coarse": case_coarse,
        "case_fine": case_fine,
        "comparison": comparison,
        "df_convergence": df_convergence,
        "table_convergence": table_convergence,
        "figures": [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8,],
       }

# ------------------------------------------------------------------------------------------------------------------------
# VALIDATION PAR SOLUTION FABRIQUEE
# ------------------------------------------------------------------------------------------------------------------------

def plot_manufactured_snapshots(case, field='U_num', title=None, n_snapshots=5):
    x = case['x']
    times = case['times']
    U = case[field]

    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx in indices:
        ax.plot(x, U[idx], label=f't = {times[idx]:.3e} s')

    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title(title if title else f'Solution numérique ({field})')
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_comparison(case, snapshot_index=-1):
    x = case['x']
    times = case['times']

    U_num = case['U_num']
    U_ref = case['U_ref']

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, U_num[snapshot_index], label="Solution numérique", linewidth=2, linestyle='-', color='b')
    ax.plot(x, U_ref[snapshot_index], label="Solution exacte", linewidth=2, linestyle='--', color='r')

    ax.set_title(f'Solution fabriquée : comparaison à t = {times[snapshot_index]:.3e} s')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_absolute_error_snapshots(case, n_snapshots=5):
    x = case['x']
    times = case['times']

    error = np.abs(case['U_num'] - case['U_ref'])

    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx in indices:
        ax.plot(x, error[idx], label=f't = {times[idx]:.3e} s')

    ax.set_title(r'Erreur méthode des solutions fabriquées: $\|u_{\mathrm{num}} - u_{\mathrm{ref}}\|$')
    ax.set_xlabel('x')
    ax.set_ylabel('Erreur')
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_error_norm_over_time(case, norm_type='L2', bc_type='dirichlet'):
    times = case['times']
    dx = case['x'][1] - case['x'][0]

    error_norm = compute_manufactured_error_norm_over_time(
        case['U_num'],
        case['U_ref'],
        dx=dx,
        norm_type=norm_type,
        bc_type=bc_type,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, error_norm, linewidth = 2, label=rf'Norme {norm_type}')
    ax.set_xlabel('t')
    ax.set_ylabel(rf"$\|u_{{num}}-u_{{exact}}\|_{{{norm_type}}}$")
    ax.set_title(rf"Évolution temporelle de la norme d'erreur ({norm_type})")
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_absolute_error_map(case):
    x = case['x']
    times = case['times']

    error = np.abs(case['U_num'] - case['U_ref'])

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(error, cmap='RdYlBu_r', extent=[x[0], x[-1], times[0], times[-1]], aspect='auto', origin='lower')

    fig.colorbar(im, ax=ax, label='Erreur')
    ax.set_title(r"Carte spatio-temporelle de l'erreur: $\|u_{\mathrm{num}} - u_{\mathrm{exact}}\|_{L^2}$")
    ax.set_aspect('auto')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.grid(False)

    return fig


def plot_convergence_curve(df, error_col='Linf_L2', x_col='dx', title=None, reference_order=1.0):
    fig, ax = plt.subplots(figsize=(8, 6))

    x = df[x_col].to_numpy()
    y = df[error_col].to_numpy()

    ax.loglog(df[x_col], df[error_col], 'o-', label='Erreur', markersize=6)

    x_ref = x
    y_ref = y[-1] * (x_ref / x_ref[-1]) ** reference_order

    ax.loglog(x_ref, y_ref, 'k--', label=rf'$\mathcal{{O}}(h^{reference_order:g})$', linewidth=1.5)

    ax.set_title(title if title else f'Courbe de convergence - {error_col.replace("_", " ")}')
    ax.set_xlabel("Pas d'espace dx")
    ax.set_ylabel('Erreur')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    return fig


def run_manufactured_plots(
        funcs,
        scheme: str = "explicit",
        theme: str =  "scientific",
        show: bool = True,
        save: bool = True,
        save_path: pathlib.Path | str | None = None
) -> dict:

    set_style(theme_name=theme)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Étude de la solution fabriquée
    study = convergence_study_manufactured(
        funcs=funcs,
        levels=[0, 1, 2, 3],
        scheme=scheme,
        T=1e-2,
        dt_mode="cfl",
        A=1e-6,
        gamma=0.1,
        kappa=1.0,
    )

    table_convergence = build_manufactured_convergence_table(study)

    finest_level = max(study["cases"].keys())
    finest_case = study["cases"][finest_level]

    fig1 = plot_manufactured_snapshots(finest_case, field='U_num', title="Solution numérique - solution fabriquée", n_snapshots=10)
    fig2 = plot_manufactured_snapshots(finest_case, field='U_ref', title="Solution exacte - solution fabriquée", n_snapshots=10)
    fig3 = plot_manufactured_comparison(finest_case, snapshot_index=-1)
    fig4 = plot_manufactured_absolute_error_snapshots(finest_case, n_snapshots=10)
    fig5 = plot_manufactured_absolute_error_map(finest_case)
    fig6 = plot_manufactured_error_norm_over_time(finest_case,  norm_type='L2', bc_type='dirichlet')
    fig7 = plot_manufactured_error_norm_over_time(finest_case, norm_type='Linf', bc_type='dirichlet')
    fig8 = plot_convergence_curve(table_convergence, error_col="Linf_L2", x_col="dx")

    if save and save_path is not None:
        save_figure_with_version(fig1, "manufactured_snapshots_num", output_dir=save_path)
        save_figure_with_version(fig2, "manufactured_snapshots_ref", output_dir=save_path)
        save_figure_with_version(fig3, "manufactured_comparison", output_dir=save_path)
        save_figure_with_version(fig4, "manufactured_absolute_error_snapshots", output_dir=save_path)
        save_figure_with_version(fig5, "manufactured_absolute_error_map", output_dir=save_path)
        save_figure_with_version(fig6, "manufactured_error_norm_L2", output_dir=save_path)
        save_figure_with_version(fig7, "manufactured_error_norm_Linf", output_dir=save_path)
        save_figure_with_version(fig8, "manufactured_convergence_curve", output_dir=save_path)

        save_error_table_csv(
            save_path / "convergence_table_manufactured.csv",
            table_convergence.to_dict(orient="records"),
        )

    print(f'Table de convergence :\n{table_convergence[["dx", "Linf_L2" ]]} ')

    if show:
        plt.show()
    else:
        plt.close('all')

    return {
        "study": study,
        "table_convergence": table_convergence,
        "finest_case": finest_case,
        "figures": [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8,]
    }


if __name__ == "__main__":

    funcs = build_numerics_function()

    refinement_dir = OUTPUTS_DIR / "refinement_plots"
    manufactured_dir = OUTPUTS_DIR / "manufactured_plots"

    # run_refinement_plots(show=True, save=False, save_path=refinement_dir, scheme="explicit")
    run_manufactured_plots(funcs, show=True, save=False,  save_path=manufactured_dir, scheme="explicit")