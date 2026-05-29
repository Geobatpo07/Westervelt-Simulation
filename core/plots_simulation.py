# ./core/plots_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from core.symbolics import build_numerics_function

from core.validation import (
    run_case_cached,
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
    save_error_table_csv,
    save_solution_npz, compute_gradient,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ------------------------------------------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------------------------------------------

def save_many_figures(figures: dict, output_dir: pathlib.Path, metadata=None):
    """
    Sauvegarde plusieurs figures dans un répertoire.

    Args:
        figures: Dictionnaire {nom: figure}.
        output_dir: Répertoire de destination.
        metadata: Métadonnées optionnelles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        save_figure_with_version(
            fig,
            name,
            output_dir=output_dir,
            formats=["png", "pdf"],
            dpi=300,
            metadata=metadata,
        )


def save_case_npz(case, path: pathlib.Path, metadata=None):
    """
    Sauvegarde les résultats d'un cas au format NPZ.

    Args:
        case: Dictionnaire de résultats.
        path: Chemin du fichier.
        metadata: Métadonnées optionnelles.
    """
    path.mkdir(parents=True, exist_ok=True)
    save_solution_npz(
        path=path,
        x=case['x'],
        times=case['times'],
        U=case['U'],
        metadata={
            "dx": case.get("dx"),
            "dt": case.get("dt"),
            "nx": case.get("nx"),
            "nt": case.get("nt"),
            "scheme": case.get("scheme"),
            "bc": case.get("bc"),
            **(metadata or {}),
        }
    )


# ------------------------------------------------------------------------------------------------------------------------
# VALIDATION PAR RAFFINEMENT DU MAILLAGE
# ------------------------------------------------------------------------------------------------------------------------

def plot_snapshots(case, title="Solution numérique", n_snapshots=5):
    """
    Affiche des snapshots de la solution.

    Args:
        case: Résultats de la simulation.
        title: Titre du graphique.
        n_snapshots: Nombre de snapshots.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
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
    """
    Compare graphiquement une solution grossière et fine.

    Args:
        comparison: Résultats de la comparaison.
        coarse: Données du cas grossier.
        n_snapshots: Nombre de snapshots.
        title: Titre du graphique.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
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
    """
    Affiche l'erreur spatiale à différents instants.

    Args:
        comparison: Résultats de la comparaison.
        coarse: Données du cas grossier.
        n_snapshots: Nombre de snapshots.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
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


def plot_error_norm_over_time(comparison, coarse, norm_type='L2', bc_type='dirichlet', title=None):
    """
    Affiche l'évolution de la norme d'erreur au cours du temps.

    Args:
        comparison: Résultats de la comparaison.
        coarse: Données du cas grossier.
        norm_type: Type de norme.
        bc_type: Conditions aux limites.
        title: Titre optionnel.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
    x = coarse['x']
    times = coarse['times']
    dx = coarse['dx']
    error = comparison['error']

    norms = []

    for n in range(error.shape[0]):
        e = error[n]

        if norm_type == "L2":
            val = np.sqrt(dx * np.sum(e**2))
        elif norm_type == "Linf":
            val = np.max(np.abs(e))
        elif norm_type == "grad":
            grad_e = compute_gradient(e, dx, bc_type=bc_type)
            val = np.sqrt(dx * np.sum(grad_e ** 2))
        elif norm_type == "H1":
            grad_e = compute_gradient(e, dx, bc_type=bc_type)
            val = np.sqrt(dx * (np.sum(e ** 2) + np.sum(grad_e ** 2)))
        else:
            raise ValueError(f"Norme inconnue : {norm_type}")

        norms.append(val)

    norms = np.asarray(norms)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, norms, linewidth=2, label=rf"Norme d'erreur {norm_type}")
    ax.set_xlabel('t')
    ax.set_ylabel(rf"$\|u_{{coarse}}-u_{{fine}}\|_{{{norm_type}}}$")
    ax.set_title(title if title else f"Évolution temporelle de la norme d'erreur ({norm_type})")
    ax.grid(True)
    ax.legend()

    return fig


def plot_error_map(comparison, coarse):
    """
    Affiche une carte spatio-temporelle de l'erreur.

    Args:
        comparison: Résultats de la comparaison.
        coarse: Données du cas grossier.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
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
    """
    Affiche la courbe de convergence.

    Args:
        df: DataFrame de convergence.
        error_col: Colonne d'erreur à afficher.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
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

    save_path = pathlib.Path(save_path or OUTPUTS_DIR / "refinement_plots")
    save_path.mkdir(parents=True, exist_ok=True)

    # Exécution des cas de validation
    case_coarse = run_case_cached(nx=2001, nt=2000, scheme=scheme, force_recompute=False)
    case_fine = run_case_cached(nx=4001, nt=4000, scheme=scheme, force_recompute=False)
    comparison = refinement_validation_direct(coarse=case_coarse, fine=case_fine, bc_type=bc)
    df_convergence = convergence_study_refinement(
        levels = [
            (101, 10000),
            (201, 10000),
            # (401, 10000),
            # (801, 10000),
            # (1601, 10000),
            # (3201, 10000),
            # (6401, 10000),
            # (12801, 10000),
            # (25601, 10000),
            # (51201, 10000),
            # (102401, 10000),
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
    fig5 = plot_convergence_table(table_convergence, error_col="Linf_L2")
    fig6 = plot_convergence_table(table_convergence, error_col="Linf_rel_L2")
    fig7 = plot_error_norm_over_time(comparison, case_coarse, norm_type='L2', bc_type=bc, title="Évolution temporelle de la norme d'erreur L2")
    fig8 = plot_error_norm_over_time(comparison, case_coarse, norm_type='Linf', bc_type=bc, title="Évolution temporelle de la norme d'erreur Linf")
    fig9 = plot_error_norm_over_time(comparison, case_coarse, norm_type='grad', bc_type=bc, title="Évolution temporelle de la norme d'erreur du gradient")

    if save and save_path is not None:
        figures_to_save = {
            "refinement_snapshots_coarse": fig1,
            "refinement_snapshots_fine": fig2,
            "refinement_comparison": fig3,
            "refinement_error_snapshots": fig4,
            "refinement_convergence_table_L2": fig5,
            "refinement_convergence_table_rel_L2": fig6,
            "refinement_error_norm_over_time_L2": fig7,
            "refinement_error_norm_over_time_Linf": fig8,
            "refinement_error_norm_over_time_grad": fig9,
        }
        save_many_figures(figures_to_save, output_dir=save_path)

        save_error_table_csv(
            "convergence_table_refinement",
            table_convergence.to_dict(orient="records"),
            output_dir=save_path,
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
        "figures": [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9],
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

    t = times[snapshot_index]

    exponent = int(np.floor(np.log10(abs(t))))
    mantissa = t / 10**exponent

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, U_num[snapshot_index], label="Solution numérique", linewidth=2, linestyle='-', color='b')
    ax.plot(x, U_ref[snapshot_index], label="Solution exacte", linewidth=2, linestyle='--', color='r')

    if np.isclose(mantissa, 1):
        time_label = rf"$t = 10^{{{exponent}}}\,\mathrm{{s}}$"
    else:
        time_label = rf"$t = {mantissa:.1f}\times10^{{{exponent}}}\,\mathrm{{s}}$"

    ax.set_title(f'Solution fabriquée : comparaison à {time_label}')
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


def plot_manufactured_scheme_comparison_over_time(case_explicit, case_semi_implicit, norm_type='L2', bc_type='dirichlet', title=None):
    times_exp = case_explicit['times']
    dx_exp = case_explicit['x'][1] - case_explicit['x'][0]

    times_semi = case_semi_implicit['times']
    dx_semi = case_semi_implicit['x'][1] - case_semi_implicit['x'][0]

    err_exp = compute_manufactured_error_norm_over_time(
        case_explicit['U_num'],
        case_explicit['U_ref'],
        dx=dx_exp,
        norm_type=norm_type,
        bc_type=bc_type,
    )

    err_semi = compute_manufactured_error_norm_over_time(
        case_semi_implicit['U_num'],
        case_semi_implicit['U_ref'],
        dx=dx_semi,
        norm_type=norm_type,
        bc_type=bc_type,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times_exp, err_exp, label=f'Explicite - {norm_type}', linewidth=2)
    ax.plot(times_semi, err_semi, label=f'Semi-implicite - {norm_type}', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel(rf"$\|u_{{num}}-u_{{exact}}\|_{{{norm_type}}}$")
    ax.set_title(title if title else rf"Comparaison des schémas explicite et semi-implicite - norme {norm_type}")
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
    ax.set_xlabel("dx")
    ax.set_ylabel('Erreur')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    return fig


def plot_manufactured_convergence_comparison(df_explicit, df_semi_implicit, error_col='Linf_L2', x_col='dx', reference_order=1.0, title=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    x_explicit = df_explicit[x_col].to_numpy()
    y_explicit = df_explicit[error_col].to_numpy()

    x_semi_implicit = df_semi_implicit[x_col].to_numpy()
    y_semi_implicit = df_semi_implicit[error_col].to_numpy()

    ax.loglog(x_explicit, y_explicit, 'o-', label='Explicite', markersize=6)
    ax.loglog(x_semi_implicit, y_semi_implicit, 's--', label='Semi-implicite', markersize=6)

    # x_ref = x_explicit
    # y_ref = y_explicit[-1] * (x_ref / x_ref[-1]) ** reference_order

    # ax.loglog(x_ref, y_ref, 'k--', label=rf'$\mathcal{{O}}(h^{reference_order:g})$', linewidth=1.5)

    ax.set_title(title if title else f'Courbe de convergence - {error_col.replace("_", " ")}')
    ax.set_xlabel("dx")
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
        save_path: pathlib.Path | str | None = None,
        levels: list[int] = [0, 1, 2, 3, 4, 5,],
        L: float = 1.0,
        T: float = 1e-4,
        A: float = 1e-3,
        gamma: float = 0.1,
        kappa: float = 1.0,
        dt_mode: str = "cfl",
) -> dict:

    figures_to_save = {}
    set_style(theme_name=theme)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Étude de la solution fabriquée
    study_exp = convergence_study_manufactured(
        funcs=funcs,
        levels=levels,
        scheme="explicit",
        L=L,
        T=T,
        dt_mode=dt_mode,
        A=A,
        mu_v=1e-6,
        gamma=gamma,
        kappa=kappa,
        force_recompute=False,
    )
    study_semi = convergence_study_manufactured(
        funcs=funcs,
        levels=levels,
        scheme="semi_implicit",
        L=L,
        T=T,
        dt_mode=dt_mode,
        A=A,
        mu_v=6e-6,
        gamma=gamma,
        kappa=kappa,
        force_recompute=False,
    )

    table_exp = build_manufactured_convergence_table(study_exp)
    table_semi = build_manufactured_convergence_table(study_semi)

    finest_level_exp = max(study_exp["cases"].keys())
    case_exp = study_exp["cases"][finest_level_exp]

    finest_level_semi = max(study_semi["cases"].keys())
    case_semi = study_semi["cases"][finest_level_semi]

    fig1 = plot_manufactured_snapshots(case_semi, field='U_num', title="Solution numérique - solution fabriquée", n_snapshots=6)
    fig2 = plot_manufactured_snapshots(case_semi, field='U_ref', title="Solution exacte - solution fabriquée", n_snapshots=6)
    fig3 = plot_manufactured_comparison(case_semi, snapshot_index=-1)
    fig4 = plot_manufactured_error_norm_over_time(case_semi,  norm_type='L2', bc_type='dirichlet')
    fig5 = plot_manufactured_error_norm_over_time(case_semi,  norm_type='H1', bc_type='dirichlet')
    fig6 = plot_manufactured_scheme_comparison_over_time(case_exp, case_semi, norm_type='L2', bc_type='dirichlet')
    fig7 = plot_manufactured_scheme_comparison_over_time(case_exp, case_semi, norm_type='Linf', bc_type='dirichlet')
    fig8 = plot_manufactured_convergence_comparison(table_exp, table_semi, error_col="Linf_L2", reference_order=1.0 if dt_mode=="cfl" else 2.0, x_col="dx")
    fig9 = plot_manufactured_convergence_comparison(table_exp, table_semi, error_col="Linf_H1", reference_order=1.0 if dt_mode=="cfl" else 2.0, x_col="dx")

    if save and save_path is not None:
        figures_to_save = {
            "manufactured_snapshots_semi": fig1,
            "manufactured_snapshots_ref": fig2,
            "manufactured_comparison": fig3,
            "manufactured_error_norm_over_time_L2": fig4,
            "manufactured_error_norm_over_time_H1": fig5,
            "manufactured_scheme_comparison_L2": fig6,
            "manufactured_scheme_comparison_Linf": fig7,
            "manufactured_convergence_comparison": fig8,
            "manufactured_convergence_comparison_H1": fig9,
        }
        save_many_figures(figures_to_save, output_dir=save_path)

        save_error_table_csv(
            "convergence_table_manufactured_explicit",
            table_exp.to_dict(orient="records"),
            output_dir=save_path,
        )

        save_error_table_csv(
            "convergence_table_manufactured_semi_implicit",
            table_semi.to_dict(orient="records"),
            output_dir=save_path,
        )

    print("\nTable explicite")
    print(table_exp[["N", "dx", "dt", "Linf_L2", "order_Linf_L2", "Linf_H1", "order_Linf_H1"]])

    print("\nTable semi-implicite")
    print(table_semi[["N", "dx", "dt", "Linf_L2", "order_Linf_L2", "Linf_H1", "order_Linf_H1"]])

    print("explicit cache:", case_exp["cache_path"])
    print("semi cache:", case_semi["cache_path"])

    print("loaded explicit:", case_exp["loaded_from_cache"])
    print("loaded semi:", case_semi["loaded_from_cache"])

    print("max |U_exp - U_semi| =", np.max(np.abs(case_exp["U_num"] - case_semi["U_num"])))
    print("max |U_exp - U_ref| =", np.max(np.abs(case_exp["U_num"] - case_exp["U_ref"])))
    print("max |U_semi - U_ref| =", np.max(np.abs(case_semi["U_num"] - case_semi["U_ref"])))

    if show:
        plt.show()
    else:
        plt.close('all')

    return {
        "study_explicit": study_exp,
        "study_semi_implicit": study_semi,
        "table_explicit": table_exp,
        "table_semi_implicit": table_semi,
        "case_explicit": case_exp,
        "case_semi_implicit": case_semi,
        "figures": figures_to_save,
    }


if __name__ == "__main__":

    funcs = build_numerics_function(bc_type="dirichlet")

    refinement_dir = OUTPUTS_DIR / "refinement_plots"
    manufactured_dir = OUTPUTS_DIR / "manufactured_plots"

    run_refinement_plots(show=True, save=False, save_path=refinement_dir, scheme="explicit", theme="scientific")
    run_manufactured_plots(
        funcs,
        show=True,
        save=False,
        save_path=manufactured_dir,
        theme="scientific",
        levels=[0, 1, 2, 3, 4, 5,],
        L=1.0,
        T=1e-3,
        A=1e-3,
        gamma=0.1,
        kappa=1.0,
        dt_mode="cfl",
    )