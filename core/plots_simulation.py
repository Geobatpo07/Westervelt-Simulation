# ./core/plots_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from core.symbolics import build_numerics_function

from core.beamer_style import (
    beamer,
    sci,
    time_colors,
    slope_triangle,
    fitted_order,
    COL_EXP,
    COL_IMP,
    COL_REF,
    COL_KO,
)

from core.validation import (
    run_case_cached,
    refinement_validation_direct,
    convergence_study_refinement,
    build_convergence_table_refinement,
    compute_manufactured_error_norm_over_time,
    convergence_study_manufactured,
    build_manufactured_convergence_table,
    scan_critical_amplitudes,
    compare_schemes_vs_b_final_time
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

def plot_snapshots(case, scale_mpa=True, title="Solution numérique", n_snapshots=5):
    """
    Affiche des snapshots de la solution, colorés par un dégradé temporel
    (clair = début, foncé = fin) : la flèche du temps se lit sans légende.
    Annote l'atténuation de l'amplitude entre t = 0 et t = T.

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

    U_plot = U / 1e6 if scale_mpa else U

    indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)
    colors = time_colors(n_snapshots)

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 6.5))

        for color, i in zip(colors, indices):
            ax.plot(x, U_plot[i], color=color,
                    label=rf'$t = {times[i] * 1e6:.1f}\,\mu s$')

        # atténuation de l'amplitude max entre t=0 et t=T
        a0 = np.max(np.abs(U_plot[indices[0]]))
        aT = np.max(np.abs(U_plot[indices[-1]]))
        if a0 > 0:
            ax.text(0.02, 0.97,
                    rf"atténuation : ${100 * (1 - aT / a0):.0f}\,\%$"
                    rf" sur $T$",
                    transform=ax.transAxes, va="top", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$u(t,x)$ [MPa]' if scale_mpa else r'$u(t,x)$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc="upper right")

    return fig


def plot_coarse_vs_fine(comparison, coarse,  n_snapshots, scale_mpa=True, title="Comparaison fine vs coarse"):
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

    if scale_mpa:
        U_coarse = comparison['U_coarse'] / 1e6
        U_fine_restricted = comparison['U_fine_restricted'] / 1e6
    else:
        U_coarse = comparison['U_coarse']
        U_fine_restricted = comparison['U_fine_restricted']

    indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)
    colors = time_colors(n_snapshots)

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.6))

        for color, i in zip(colors, indices):
            ax.plot(x, U_coarse[i], color=color, linestyle='-',
                    label=rf'$t = {times[i] * 1e6:.1f}\,\mu s$')
            ax.plot(x, U_fine_restricted[i], color=color, linestyle='--')

        # légende compacte : couleurs = instants, styles = grilles
        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([], [], color='k', ls='-', label=r'$u_h$ (grossier)'),
            Line2D([], [], color='k', ls='--', label=r'$u_{\mathrm{ref}}$ (fin restreint)'),
        ]
        leg1 = ax.legend(handles=style_handles, loc='upper right', fontsize=12)
        ax.add_artist(leg1)
        ax.legend(loc='upper left', fontsize=11)

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$u(t,x)$ [MPa]' if scale_mpa else r'$u(t,x)$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    return fig


def plot_coarse_vs_reference_final(comparison, coarse, title=None, scale_mpa=True):
    x = coarse['x']
    times = coarse['times']

    U_coarse = comparison['U_coarse']
    U_ref = comparison['U_fine_restricted']

    t_us = times[-1] * 1e6

    factor = 1e6 if scale_mpa else 1.0
    y_label = 'u(t,x) [MPa]' if scale_mpa else 'u(t,x)'

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, U_coarse[-1] / factor, label=rf'U coarse (t = {t_us:.2f} $\mu s)$', linewidth=2)
    ax.plot(x, U_ref[-1] / factor, label=rf'U référence (t = {t_us:.2f} $\mu s)$', linewidth=2, linestyle='--')

    ax.set_xlabel(r'x [$\mathrm{m}$]')
    ax.set_ylabel(y_label)
    ax.set_title(title if title else rf'Comparaison au temps $t={t_us:g}\,\mu s$')
    ax.grid(True)
    ax.legend()

    return fig


def plot_coarse_vs_reference_at_max_error(
        comparison,
        coarse,
        norm_type="L2",
        scale_mpa=True,
        title=None,
):
    x = coarse["x"]
    times = coarse["times"]
    dx = coarse["dx"]

    U_coarse = comparison["U_coarse"]
    U_ref = comparison["U_fine_restricted"]
    error = comparison["error"]

    norms = np.sqrt(dx * np.sum(error**2, axis=1))
    idx = int(np.argmax(norms))

    t_us = times[idx] * 1e6

    factor = 1e6 if scale_mpa else 1.0
    y_label = r"$u(t,x)$ [MPa]" if scale_mpa else r"$u(t,x)$"

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.6))

        # les marqueurs montrent les points effectifs de la grille grossière :
        # on *voit* la sous-résolution des fronts
        ax.plot(x, U_coarse[idx] / factor, "o-", color=COL_EXP,
                markersize=4.5, linewidth=1.8,
                label=rf"$u_h$  ($n_x = {len(x)}$)")
        ax.plot(x, U_ref[idx] / factor, "--", color=COL_REF, linewidth=2.2,
                label=r"$u_{\mathrm{ref}}$")

        # écart ponctuel |u_h - u_ref| en aire grisée (axe secondaire)
        gap = np.abs(error[idx]) / factor
        ax2 = ax.twinx()
        ax2.fill_between(x, gap, color=COL_KO, alpha=0.18, zorder=0)
        ax2.set_ylim(0, 4 * max(gap.max(), 1e-30))
        ax2.set_ylabel(r"$|u_h - u_{\mathrm{ref}}|$ [MPa]" if scale_mpa
                       else r"$|u_h - u_{\mathrm{ref}}|$",
                       color=COL_KO, fontsize=13)
        ax2.tick_params(axis="y", colors=COL_KO, labelsize=11)

        # localisation de l'écart maximal
        j = int(np.argmax(gap))
        ax.annotate(rf"écart max $= {gap[j]:.2g}$" + (" MPa" if scale_mpa else ""),
                    xy=(x[j], U_coarse[idx, j] / factor),
                    xytext=(0, 26), textcoords="offset points",
                    ha="center", fontsize=13, color=COL_KO,
                    arrowprops=dict(arrowstyle="->", color=COL_KO, lw=1.4))

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(y_label)
        ax.set_title(title if title else
                     rf"Instant d'erreur maximale ($t = {t_us:.1f}\,\mu s$)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", fontsize=12)

    return fig


def plot_scheme_comparison(
        case_explicit,
        case_semi_implicit,
        field="U",
        n_snapshots=5,
        scale_mpa=True,
        title=None,
):
    x_explicit = case_explicit['x']
    x_semi_implicit = case_semi_implicit['x']

    times_explicit = case_explicit['times']
    times_semi_implicit = case_semi_implicit['times']

    if not np.allclose(x_explicit, x_semi_implicit):
        raise ValueError("Les maillages spatiaux ne sont pas identiques")
    if not np.allclose(times_explicit, times_semi_implicit):
        raise ValueError("Les maillages temporels ne sont pas identiques")

    U_explicit = np.asarray(case_explicit[field])
    U_semi_implicit = np.asarray(case_semi_implicit[field])

    if U_explicit.shape != U_semi_implicit.shape:
        raise ValueError(f"Les solutions ne sont pas comparables: {U_explicit.shape} != {U_semi_implicit.shape}")

    factor = 1e6 if scale_mpa else 1.0

    indices = np.linspace(0, U_explicit.shape[0]-1, n_snapshots, dtype=int)

    fig, axs = plt.subplot_mosaic("AB;CC", figsize=(12, 10), )

    ax_explicit = axs['A']
    ax_semi_implicit = axs['B']
    ax_diff = axs['C']

    for i in indices:
        t_us = times_explicit[i] * 1e6

        label = rf"$t = {t_us:.2f}\,\mu s$"

        ax_explicit.plot(x_explicit, U_explicit[i] / factor, linewidth=2, label=label)
        ax_semi_implicit.plot(x_semi_implicit, U_semi_implicit[i] / factor, linewidth=2, label=label)
        ax_diff.plot(x_explicit, np.abs(U_explicit[i] - U_semi_implicit[i]), linewidth=2, label=label)

    unit = "MPa" if scale_mpa else "Pa"

    ax_diff.set_yscale('log')

    ax_explicit.set_ylabel(rf"$u(t,x)$ [{unit}]" if scale_mpa else rf"$u(t,x)$")
    ax_semi_implicit.set_ylabel(rf"$u(t,x)$ [{unit}]" if scale_mpa else rf"$u(t,x)$")
    ax_diff.set_ylabel(rf"$\Delta u(t,x)$")
    ax_diff.set_xlabel(r'x [$\mathrm{m}$]')

    ax_explicit.set_title("Schéma explicite")
    ax_semi_implicit.set_title("Schéma semi-implicite")
    ax_diff.set_title("Différence absolue entre les schémas")

    for ax in [ax_diff, ax_explicit, ax_semi_implicit]:
        ax.grid(True, alpha=0.4)

    handles, labels = ax_explicit.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.98))

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()

    return fig


def plot_scheme_difference_vs_b_final(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        df["b"],
        df["l2_rel_final"],
        "o-",
        linewidth=2,
        markersize=6,
    )

    ax.set_xlabel(r"Coefficient diffusif $b$ [$\mathrm{m}^2/\mathrm{s}$]")
    ax.set_ylabel(
        r"$\frac{\|u_{\mathrm{exp}}(T)-u_{\mathrm{semi}}(T)\|_{\ell^2}}"
        r"{\|u_{\mathrm{semi}}(T)\|_{\ell^2}}$"
    )

    ax.set_title(r"Écart relatif final entre les schémas en fonction de $b$")
    ax.grid(True, which="both", ls="--", alpha=0.5)

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
        ax.plot(x, error[i], label=rf'$t = {times[i] * 1e6:.2f}\,\mu s$')

    ax.set_xlabel(r'x [$\mathrm{m}$]')
    ax.set_ylabel('Erreur')
    ax.set_title('Erreur sur les snapshots')
    ax.grid(True)
    ax.legend()

    return fig


def plot_error_norm_over_time(comparison, coarse, norm_type='L2', bc_type='dirichlet', title=None):
    """
    Affiche l'évolution temporelle de l'erreur relative.

    E(t) = ||u_h - u_ref|| / ||u_ref||

    Args:
        comparison: Résultats de la comparaison.
        coarse: Données du cas grossier.
        norm_type: "L2", "Linf", "grad" ou "H1".
        bc_type: Conditions aux limites.
        title: Titre optionnel.

    Returns:
        plt.Figure: Figure Matplotlib.
    """
    times = coarse['times'] * 1e6
    dx = coarse['dx']
    error = comparison['error']
    U_ref = comparison['U_fine_restricted']

    relative_errors = []

    for n in range(error.shape[0]):
        e = error[n]
        u_ref = U_ref[n]

        if norm_type == 'L2':
            num = np.sqrt(dx*np.sum(e**2))
            deno = np.sqrt(dx*np.sum(u_ref**2))
        elif norm_type == 'Linf':
            num = np.max(np.abs(e))
            deno = np.max(np.abs(u_ref))
        elif norm_type == 'grad':
            grad_e = compute_gradient(e, dx, bc_type=bc_type)
            grad_ref = compute_gradient(u_ref, dx, bc_type=bc_type)

            num = np.sqrt(dx * np.sum(grad_e**2))
            deno = np.sqrt(dx * np.sum(grad_ref**2))
        elif norm_type == 'H1':
            grad_e = compute_gradient(e, dx, bc_type=bc_type)
            grad_ref = compute_gradient(u_ref, dx, bc_type=bc_type)

            e_inner = e[1:-1]
            ref_inner = u_ref[1:-1]

            num = np.sqrt(dx * np.sum(e_inner**2) + dx * np.sum(grad_e**2))
            deno = np.sqrt(dx * np.sum(ref_inner**2) + dx * np.sum(grad_ref**2))
        else:
            raise ValueError(f"Norm inconnue : {norm_type}")

        rel_error = num / deno if deno > 1e-15 else 0.0
        relative_errors.append(rel_error)

    relative_errors = np.asarray(relative_errors)

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.6))
        ax.plot(times, relative_errors, color=COL_EXP,
                label=rf"Erreur relative {norm_type}")

        # instant du maximum : celui retenu par la norme l^inf en temps
        i = int(np.argmax(relative_errors))
        ax.plot(times[i], relative_errors[i], 'o', color=COL_KO, markersize=8)
        ax.annotate(rf"max à $t = {times[i]:.1f}\,\mu s$",
                    xy=(times[i], relative_errors[i]),
                    xytext=(-10, -18), textcoords="offset points",
                    ha='right', fontsize=13, color=COL_KO)

        ax.set_xlabel(r'$t$ [$\mu$s]')

        if norm_type == "L2":
            ylabel = (
                r"$\frac{\|u_h-u_{\mathrm{ref}}\|_{\ell^2}}"
                r"{\|u_{\mathrm{ref}}\|_{\ell^2}}$"
            )
        elif norm_type == "H1":
            ylabel = (
                r"$\frac{\|u_h-u_{\mathrm{ref}}\|_{H_h^1}}"
                r"{\|u_{\mathrm{ref}}\|_{H_h^1}}$"
            )
        else:
            ylabel = f"Erreur relative {norm_type}"

        ax.set_ylabel(ylabel)
        ax.set_title(title if title else
                     f"Évolution temporelle de la norme d'erreur ({norm_type})")
        ax.grid(True, alpha=0.3)
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
    error = np.abs(comparison['error'])

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.8))

        # origin='lower' : sans cela, l'axe temporel est inversé
        im = ax.imshow(
            error, cmap='magma',
            extent=[x[0], x[-1], times[0] * 1e6, times[-1] * 1e6],
            aspect='auto', origin='lower',
        )
        fig.colorbar(im, ax=ax, label=r'$|u_h - u_{\mathrm{ref}}|$')

        # localisation du maximum spatio-temporel
        n_max, i_max = np.unravel_index(np.argmax(error), error.shape)
        ax.plot(x[i_max], times[n_max] * 1e6, '*', color='w',
                markersize=16, markeredgecolor='k', markeredgewidth=1.0)
        ax.annotate("erreur max", xy=(x[i_max], times[n_max] * 1e6),
                    xytext=(12, 10), textcoords="offset points",
                    color='w', fontsize=13, fontweight='bold')

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$t$ [$\mu$s]')
        ax.set_title("Carte spatio-temporelle de l'erreur")
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

    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel('Erreur')
    ax.set_title(f'Tableau de convergence - {error_col.replace("_", " ")}')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    return fig


def plot_refinement_convergence_comparison(
        df_explicit,
        df_semi_implicit,
        error_col='Linf_rel_L2',
        x_col='dx',
        title=None,
):
    x_explicit = df_explicit[x_col].to_numpy()
    x_semi_implicit = df_semi_implicit[x_col].to_numpy()

    y_explicit = df_explicit[error_col].to_numpy()
    y_semi_implicit = df_semi_implicit[error_col].to_numpy()

    p_exp = fitted_order(x_explicit, y_explicit)
    p_imp = fitted_order(x_semi_implicit, y_semi_implicit)
    reference_order = 1.0  # couplage dt = O(dx) dans cette expérience

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.8))

        ax.loglog(x_explicit, y_explicit, 'o-', color=COL_EXP,
                  label=rf'Explicite — $p = {p_exp:.2f}$')
        ax.loglog(x_semi_implicit, y_semi_implicit, 's--', color=COL_IMP,
                  label=rf'Semi-implicite — $p = {p_imp:.2f}$')

        x_ref = np.array([x_explicit.min(), x_explicit.max()])
        y_ref = y_explicit[np.argmin(x_explicit)] * (x_ref / x_ref.min()) ** reference_order
        ax.loglog(x_ref, y_ref * 0.5, 'k--', linewidth=1.4,
                  label=rf'$\mathcal{{O}}(\Delta x)$')
        i = len(x_explicit) // 2
        slope_triangle(ax, x_explicit[i] * 0.55, x_explicit[i],
                       y_explicit[i] * 0.30, reference_order)

        ax.set_xlabel(r"$\Delta x$ [m]")

        if error_col == "Linf_rel_L2":
            ylabel = (
                r"$\max_t "
                r"\frac{\|u_h-u_{\mathrm{ref}}\|_{\ell^2}}"
                r"{\|u_{\mathrm{ref}}\|_{\ell^2}}$"
            )
        elif error_col == "Linf_rel_H1":
            ylabel = (
                r"$\max_t "
                r"\frac{\|u_h-u_{\mathrm{ref}}\|_{H_h^1}}"
                r"{\|u_{\mathrm{ref}}\|_{H_h^1}}$"
            )
        else:
            ylabel = "Erreur"
        ax.set_ylabel(ylabel)
        ax.set_title(title if title else
                     f'Convergence (raffinement) — {error_col.replace("_", " ")}')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()

    return fig


def plot_alpha_min_vs_amplitude(df_amp, alpha_tol=1e-6, zoom=False):
    """Recherche de l'amplitude critique de dégénérescence.

    Trace min(1 - 2ku) en fonction de l'amplitude initiale ; la zone
    alpha <= 0 (rouge) est celle où l'hypothèse de non-dégénérescence (2.5)
    est violée.
    """
    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.6))

        ax.semilogx(df_amp['A1'], df_amp['alpha_min'], 'o-', color=COL_EXP,
                    label=r'$\min_{t,x}(1-2ku)$')
        ax.axhline(0.0, linestyle='--', linewidth=1.4, color='k',
                   label=r'dégénérescence ($\alpha = 0$)')
        ax.axhline(alpha_tol, linestyle=':', linewidth=1.2, color='0.4')

        # zone interdite
        lo, hi = ax.get_ylim()
        if lo < 0:
            ax.axhspan(lo, 0.0, color=COL_KO, alpha=0.12)
            ax.text(0.02, 0.04, r"$1 - 2ku \leq 0$ : modèle dégénéré",
                    transform=ax.transAxes, color=COL_KO, fontsize=13)
        ax.set_ylim(lo, hi)

        if zoom:
            ax.set_ylim(-1.0, 1.0)

        ax.set_xlabel(r'Amplitude initiale $A_1$ [Pa]')
        ax.set_ylabel(r'$\min_{t,x}(1-2ku)$')
        ax.set_title("Recherche de l'amplitude critique")
        ax.grid(True, which='both', ls='--', alpha=0.4)
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


def plot_scheme_solutions_only(
        case_explicit,
        case_semi_implicit,
        n_snapshots=5,
        scale_mpa=True,
        title=None,
):
    """
    Affiche les solutions des deux schémas côte à côte (sans la différence).
    Destiné à la présentation Beamer -- panneau large et lisible.

    Args:
        case_explicit: Résultats du schéma explicite.
        case_semi_implicit: Résultats du schéma semi-implicite.
        n_snapshots: Nombre de snapshots temporels.
        scale_mpa: Si True, affiche en MPa.
        title: Titre global optionnel.

    Returns:
        plt.Figure
    """
    x = case_explicit['x']
    times = case_explicit['times']

    if not np.allclose(x, case_semi_implicit['x']):
        raise ValueError("Les maillages spatiaux ne sont pas identiques.")
    if not np.allclose(times, case_semi_implicit['times']):
        raise ValueError("Les maillages temporels ne sont pas identiques.")

    factor = 1e6 if scale_mpa else 1.0
    unit = "MPa" if scale_mpa else "Pa"

    U_exp  = np.asarray(case_explicit['U']) / factor
    U_semi = np.asarray(case_semi_implicit['U']) / factor

    indices = np.linspace(0, len(times) - 1, n_snapshots, dtype=int)
    colors  = time_colors(n_snapshots)

    with beamer():
        fig, (ax_exp, ax_semi) = plt.subplots(
            1, 2, figsize=(12, 5.2), sharey=True
        )

        for color, i in zip(colors, indices):
            label = rf"$t = {times[i] * 1e6:.2f}\,\mu s$"
            ax_exp.plot( x, U_exp[i],  color=color, linewidth=1.8, label=label)
            ax_semi.plot(x, U_semi[i], color=color, linewidth=1.8, label=label)

        for ax, scheme_label in zip(
            [ax_exp, ax_semi],
            ["Schéma explicite", "Schéma semi-implicite"]
        ):
            ax.set_title(scheme_label, fontsize=14)
            ax.set_xlabel(r"$x$ [m]")
            ax.grid(True, alpha=0.3)

        ax_exp.set_ylabel(rf"$u(t,x)$ [{unit}]")
        ax_semi.set_ylabel("")  # axe partagé, étiquette inutile à droite

        # légende centrée au-dessus des deux panneaux
        handles, labels = ax_exp.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='upper center', ncol=n_snapshots,
            fontsize=11, bbox_to_anchor=(0.5, 1.0)
        )

        if title:
            fig.suptitle(title, fontsize=15, y=1.06)

        fig.tight_layout()

    return fig


def plot_scheme_difference_only(
        case_explicit,
        case_semi_implicit,
        n_snapshots=5,
        scale_mpa=True,
        title=None,
):
    """
    Affiche uniquement la différence absolue |u_exp - u_semi| en échelle log,
    pour plusieurs instants. Destiné à la présentation Beamer.

    Args:
        case_explicit: Résultats du schéma explicite.
        case_semi_implicit: Résultats du schéma semi-implicite.
        n_snapshots: Nombre de snapshots temporels.
        scale_mpa: Si True, affiche en MPa (différence en MPa).
        title: Titre optionnel.

    Returns:
        plt.Figure
    """
    x = case_explicit['x']
    times = case_explicit['times']

    if not np.allclose(x, case_semi_implicit['x']):
        raise ValueError("Les maillages spatiaux ne sont pas identiques.")
    if not np.allclose(times, case_semi_implicit['times']):
        raise ValueError("Les maillages temporels ne sont pas identiques.")

    factor = 1e6 if scale_mpa else 1.0
    unit = "MPa" if scale_mpa else "Pa"

    U_exp  = np.asarray(case_explicit['U'])
    U_semi = np.asarray(case_semi_implicit['U'])

    indices = np.linspace(0, len(times) - 1, n_snapshots + 1, dtype=int)[1:]
    colors = time_colors(n_snapshots + 1)[1:]

    # amplitude max globale pour annoter le ratio
    amp_max = np.max(np.abs(U_exp)) / factor
    diff_max = np.max(np.abs(U_exp - U_semi)) / factor

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.6))

        for color, i in zip(colors, indices):
            diff = np.abs(U_exp[i] - U_semi[i]) / factor
            # remplace les zéros exacts pour éviter log(0)
            diff = np.where(diff == 0, 1e-30, diff)
            ax.semilogy(
                x, diff,
                color=color, linewidth=1.8,
                label=rf"$t = {times[i] * 1e6:.2f}\,\mu s$"
            )

        # annotation du ratio diff_max / amp_max
        ax.text(
            0.97, 0.05,
            rf"$\Delta u_{{max}} / \|u\|_\infty \approx {diff_max / amp_max:.1e}$",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6")
        )

        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(
            rf"$|u_{{\mathrm{{exp}}}} - u_{{\mathrm{{semi}}}}|$ [{unit}]"
        )
        ax.set_title(
            title if title else
            r"Différence absolue $|u_{\mathrm{exp}} - u_{\mathrm{semi}}|$"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(
            fontsize=11,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=n_snapshots,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)

    return fig


def run_refinement_plots(
        c: float,
        rho0: float,
        beta: float,
        mu_v: float,
        zeta: float,
        scheme: str = "explicit",
        bc_type: str = "dirichlet",
        theme: str = "scientific",
        show: bool = True,
        save: bool = True,
        save_path: pathlib.Path | str | None = None,
        levels=None,
        force_recompute: bool = False,
) -> dict:

    set_style(theme_name=theme)

    if levels is None:
        levels = [
            (51, 50),
            (101, 100),
            (201, 200),
            (401, 400),
            (801, 800),
            (1601, 1600),
            (3201, 3200),
            # (6401, 6400),
            # (12801, 12800),
        ]

    save_path = pathlib.Path(save_path or OUTPUTS_DIR / "refinement_plots")
    save_path.mkdir(parents=True, exist_ok=True)

    params = {
        "T_final": 37e-6,
        "L_final": 0.2,
        "bc": bc_type,
        "store_energy": False,
        "force_recompute": force_recompute,
    }

    study = convergence_study_refinement(
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        zeta=zeta,
        levels=levels,
        scheme=scheme,
        **params,
    )

    table = build_convergence_table_refinement(study["rows"])

    coarse_key = levels[0]
    ref_key = levels[-1]

    case_coarse = study["cases"][coarse_key]
    case_ref = study["cases"][ref_key]
    comp = refinement_validation_direct(
        coarse=case_coarse,
        fine=case_ref,
        bc_type=bc_type,
    )

    k = beta / (rho0 * c ** 2)
    U_num = case_coarse["U"]
    min_coeff = np.min(1 - 2 * k * U_num)
    print(f"min(1 - 2ku) = {min_coeff:.6f}")

    fig1 = plot_snapshots(case_ref, title="Solution de référence", n_snapshots=5)

    fig2 = plot_coarse_vs_fine(
        comp,
        case_coarse,
        n_snapshots=5,
        title="Comparaison solution grossière / référence",
    )

    fig3 = plot_coarse_vs_reference_final(
        comp,
        case_coarse,
        title=r"Comparaison à l'instant final",
        scale_mpa=True,
    )

    fig4 = plot_coarse_vs_reference_at_max_error(
        comp,
        case_coarse,
        scale_mpa=True,
        title=r"Comparaison à l'instant d'erreur maximale",
    )

    fig5 = plot_error_norm_over_time(
        comp,
        case_coarse,
        norm_type="L2",
        bc_type=bc_type,
        title=r"Évolution temporelle de l'erreur de raffinement ($\ell^2$)",
    )

    fig6 = plot_error_norm_over_time(
        comp,
        case_coarse,
        norm_type="H1",
        bc_type=bc_type,
        title=r"Évolution temporelle de l'erreur de raffinement ($H^1_h$)",
    )

    study_exp, study_semi = convergence_study_refinement(
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        zeta=zeta,
        levels=levels,
        scheme='explicit',
        **params,
    ), convergence_study_refinement(
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        zeta=zeta,
        levels=levels,
        scheme='semi_implicit',
        **params,
    )

    case_explicit = study_exp["cases"][ref_key]
    case_semi_implicit = study_semi["cases"][ref_key]

    df_explicit = build_convergence_table_refinement(study_exp["rows"])
    df_semi_implicit = build_convergence_table_refinement(study_semi["rows"])

    # Comparaison des deux schémas sur le maillage le plus grossier
    fig_solutions = plot_scheme_solutions_only(
        case_explicit,  # cas explicite
        case_semi_implicit,  # cas semi-implicite
        title="Comparaison explicite vs semi-implicite",
        scale_mpa=True,
    )

    fig_diff = plot_scheme_difference_only(
        case_explicit,
        case_semi_implicit,
        scale_mpa=False,
    )

    # Convergence des deux schémas superposée
    fig_conv = plot_refinement_convergence_comparison(
        df_explicit,
        df_semi_implicit,
        error_col='Linf_rel_L2',
        title=r"Convergence par raffinement -- les deux schémas"
    )

    # fig7 = plot_refinement_convergence_comparison(table_exp, table_semi, error_col="Linf_rel_L2",)

    # fig8 = plot_scheme_comparison(case_exp_ref, case_semi_ref,"U",)

    # fig_b = plot_scheme_difference_vs_b_final(df_b)

    if scheme == "semi_implicit":
        figures_to_save = {
            "semi_refinement_snapshots_ref": fig1,
            "semi_refinement_coarse_vs_fine_ref": fig2,
            "semi_refinement_coarse_vs_final_ref": fig3,
            "semi_refinement_coarse_vs_max_error_ref": fig4,
            "semi_refinement_error_norm_over_time_ref": fig5,
            "semi_refinement_error_norm_over_time_h1_ref": fig6,
            "comparisons_semi_refinement_ref": fig_diff,
            "refinement_convergence_comparison_ref": fig_conv,
            "refinement_solutions_only_ref": fig_solutions,
        }

        if save:
            save_many_figures(figures_to_save, output_dir=save_path)

            save_error_table_csv(
                "convergence_table_refinement_semi_implicit",
                table.to_dict(orient="records"),
                output_dir=save_path,
            )
    elif scheme == "explicit":
        figures_to_save = {
            "exp_refinement_snapshots_ref": fig1,
            "exp_refinement_coarse_vs_fine_ref": fig2,
            "exp_refinement_coarse_vs_final_ref": fig3,
            "exp_refinement_coarse_vs_max_error_ref": fig4,
            "exp_refinement_error_norm_over_time_ref": fig5,
            "exp_refinement_error_norm_over_time_h1_ref": fig6,
            # "refinement_convergence_comparison_ref": fig7,
            # "refinement_scheme_comparison_ref": fig8,
            # "refinement_b_values_ref": fig_b,
        }

        if save:
            save_many_figures(figures_to_save, output_dir=save_path)

            save_error_table_csv(
                "convergence_table_refinement_explicit",
                table.to_dict(orient="records"),
                output_dir=save_path,
            )
    else:
        raise ValueError(f"Schéma inconnu : {scheme}. Utilisez 'explicit' ou 'semi_implicit'.")


    print("\nTable raffinement :")
    print(table[["nx", "dx", "dt", "Linf_rel_L2", "order_Linf_rel_L2", "Linf_rel_H1", "order_Linf_rel_H1"]])

    if show:
        plt.show()
    else:
        plt.close("all")

    if scheme == "explicit":
        return {
            "study_explicit": study,
            "table_explicit": table,
            "comparison_explicit": comp,
            "figures": figures_to_save,
        }
    elif scheme == "semi_implicit":
        return {
            "study_semi_implicit": study,
            "table_semi_implicit": table,
            "comparison_semi_implicit": comp,
            "figures": figures_to_save,
        }
    else:
        raise ValueError(f"Schéma inconnu : {scheme}. Utilisez 'explicit' ou 'semi_implicit'.")

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
        ax.plot(x, U[idx], label=rf'$t = {times[idx] * 1e6:.2f}\,\mu s$')

    ax.set_xlabel(r'x [$\mathrm{m}$]')
    ax.set_ylabel(r'$u(t,x)$')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_comparison(case, snapshot_index=-1):
    x = case['x']
    times = case['times']

    U_num = case['U_num']
    U_ref = case['U_ref']

    t = times[snapshot_index]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, U_num[snapshot_index], label="Solution numérique", linewidth=2, linestyle='-', color='b')
    ax.plot(x, U_ref[snapshot_index], label="Solution exacte", linewidth=2, linestyle='--', color='r')

    ax.set_title(rf"Comparaison au temps $t={t * 1e6:g}\,\mu s$")
    ax.set_xlabel(r'x [$\mathrm{m}$]')
    ax.set_ylabel(r'$u(t,x)$')
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_comparison_mosaic(case):
    x = case['x']
    times = case['times']

    U_num = case['U_num']
    U_ref = case['U_ref']

    indices = np.linspace(0, len(times) - 1, 4, dtype=int)

    with beamer():
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 8),
                                sharex=True, sharey=True)
        axs = axs.ravel()

        for ax, idx in zip(axs, indices):
            ax.plot(x, U_num[idx], color=COL_EXP, label="Numérique")
            ax.plot(x, U_ref[idx], color=COL_REF, linestyle="--",
                    label="Exacte")

            # erreur max du panneau, pour quantifier le "indiscernable"
            e_max = np.max(np.abs(U_num[idx] - U_ref[idx]))
            ax.set_title(rf"$t = {times[idx] * 1e6:.1f}\,\mu s$"
                         rf"  ($\|e\|_\infty = {sci(e_max, 1)}$)",
                         fontsize=14)
            ax.grid(True, alpha=0.3)

        axs[0].legend(fontsize=12)
        fig.supxlabel(r'$x$ [m]')
        fig.supylabel(r'$u(t,x)$')
        fig.suptitle('Solutions fabriquées : numérique vs exacte', fontsize=17)

    return fig


def plot_manufactured_absolute_error_snapshots(case, n_snapshots=5):
    x = case['x']
    times = case['times']

    error = np.abs(case['U_num'] - case['U_ref'])

    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx in indices:
        ax.plot(x, error[idx], label=rf'$t = {times[idx] * 1e6:.2f}\,\mu s$')

    ax.set_title(r'Erreur méthode des solutions fabriquées: $\|u_{\mathrm{num}} - u_{\mathrm{ref}}\|$')
    ax.set_xlabel(r'x [$\mathrm{m}$]')
    ax.set_ylabel('Erreur')
    ax.grid(True)
    ax.legend()

    return fig


def plot_manufactured_error_norm_over_time(case, norm_type='L2', bc_type='dirichlet', title=None):
    times = case['times']
    dx = case['x'][1] - case['x'][0]

    error_norm = compute_manufactured_error_norm_over_time(
        case['U_num'],
        case['U_ref'],
        dx=dx,
        norm_type=norm_type,
        bc_type=bc_type,
    )

    if norm_type == 'L2':
        norm_label = r'\ell^2'
    elif norm_type == 'H1':
        norm_label = r'H^1_h'
    else:
        norm_label = f'{norm_type}'

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times * 1e6, error_norm, linewidth = 2, label=rf'Norme ${norm_label}$')
    ax.set_xlabel(r't [$\mu\mathrm{s}$]')
    ax.set_ylabel(rf"$\|e(t)\|_{{{norm_label}}}$")
    ax.set_title(title if title is not None else rf"Évolution temporelle de la norme d'erreur ({norm_label})")
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

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.6))
        # NB : conversion en microsecondes (l'ancienne version traçait des
        # secondes sous une étiquette en microsecondes)
        ax.plot(times_exp * 1e6, err_exp, color=COL_EXP, label='Explicite')
        ax.plot(times_semi * 1e6, err_semi, color=COL_IMP, linestyle='--',
                label='Semi-implicite')
        ax.set_xlabel(r'$t$ [$\mu$s]')
        ax.set_ylabel(rf"$\|u_h - u_{{\mathrm{{ex}}}}\|_{{{norm_type}}}$")
        ax.set_title(title if title else
                     rf"Erreur au cours du temps — norme {norm_type}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    return fig


def plot_manufactured_absolute_error_map(case):
    x = case['x']
    times = case['times']

    error = np.abs(case['U_num'] - case['U_ref'])

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.8))

        im = ax.imshow(
            error, cmap='magma',
            extent=[x[0], x[-1], times[0] * 1e6, times[-1] * 1e6],
            aspect='auto', origin='lower',
        )
        fig.colorbar(im, ax=ax, label=r'$|u_h - u_{\mathrm{ex}}|$')

        n_max, i_max = np.unravel_index(np.argmax(error), error.shape)
        ax.plot(x[i_max], times[n_max] * 1e6, '*', color='w',
                markersize=16, markeredgecolor='k', markeredgewidth=1.0)

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$t$ [$\mu$s]')
        ax.set_title("Carte spatio-temporelle de l'erreur (MMS)")
        ax.grid(False)

    return fig


def plot_convergence_curve(df, error_col='Linf_L2', x_col='dx', title=None, reference_order=1.0):
    """Courbe de convergence avec triangle de pente et ordre ajusté.

    L'ordre expérimental affiché est la pente globale des moindres carrés
    en log-log (plus robuste que les ordres locaux entre niveaux).
    """
    x = df[x_col].to_numpy()
    y = df[error_col].to_numpy()

    p_fit = fitted_order(x, y)

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.8))

        ax.loglog(x, y, 'o-', color=COL_EXP,
                  label=rf'Erreur — ordre ajusté $p = {p_fit:.2f}$')

        # droite de référence calée sur le point le plus fin
        x_ref = np.array([x.min(), x.max()])
        y_ref = y[np.argmin(x)] * (x_ref / x.min()) ** reference_order
        ax.loglog(x_ref, y_ref, 'k--', linewidth=1.5,
                  label=rf'$\mathcal{{O}}(\Delta x^{{{reference_order:g}}})$')

        # triangle de pente (convention des figures de convergence)
        i = len(x) // 2
        slope_triangle(ax, x[i] * 0.55, x[i], y[i] * 0.45, reference_order)

        ax.set_title(title if title else
                     f'Courbe de convergence — {error_col.replace("_", " ")}')
        ax.set_xlabel(r"$\Delta x$ [m]")
        ax.set_ylabel('Erreur')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()

    return fig


def plot_manufactured_convergence_comparison(df_explicit, df_semi_implicit, error_col='Linf_L2', x_col='dx', title=None, reference_order=None):
    x_explicit = df_explicit[x_col].to_numpy()
    y_explicit = df_explicit[error_col].to_numpy()

    x_semi_implicit = df_semi_implicit[x_col].to_numpy()
    y_semi_implicit = df_semi_implicit[error_col].to_numpy()

    p_exp = fitted_order(x_explicit, y_explicit)
    p_imp = fitted_order(x_semi_implicit, y_semi_implicit)
    if reference_order is None:
        reference_order = round((p_exp + p_imp) / 2)

    with beamer():
        fig, ax = plt.subplots(figsize=(9, 5.8))

        ax.loglog(x_explicit, y_explicit, 'o-', color=COL_EXP,
                  label=rf'Explicite — $p = {p_exp:.2f}$')
        ax.loglog(x_semi_implicit, y_semi_implicit, 's--', color=COL_IMP,
                  label=rf'Semi-implicite — $p = {p_imp:.2f}$')

        # droite de référence et triangle de pente
        x_ref = np.array([x_explicit.min(), x_explicit.max()])
        y_ref = y_explicit[np.argmin(x_explicit)] * (x_ref / x_ref.min()) ** reference_order
        ax.loglog(x_ref, y_ref * 0.5, 'k--', linewidth=1.4,
                  label=rf'$\mathcal{{O}}(\Delta x^{{{reference_order:g}}})$')
        i = len(x_explicit) // 2
        slope_triangle(ax, x_explicit[i] * 0.55, x_explicit[i],
                       y_explicit[i] * 0.30, reference_order)

        if error_col == 'Linf_L2':
            y_label = r'\max_{0\leq t\leq T}\|e(t)\|_{\ell^2}'
        elif error_col == 'Linf_H1':
            y_label = r'\max_{0\leq t\leq T}\|e(t)\|_{H^1_h}'
        else:
            y_label = r'\mathrm{Erreur}'

        ax.set_title(title if title else
                     f'Convergence (MMS) — {error_col.replace("_", " ")}')
        ax.set_xlabel(r"$\Delta x$ [m]")
        ax.set_ylabel(rf'${y_label}$')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()

    return fig


def run_manufactured_plots(
        funcs,
        scheme: str = "explicit",
        bc_type: str = "dirichlet",
        theme: str =  "scientific",
        show: bool = True,
        save: bool = True,
        save_path: pathlib.Path | str | None = None,
        levels: list[int] = [0, 1, 2, 3],
        L: float = 0.2,
        T: float = 37e-6,
        A: float = 1e-3,
        gamma: float = 0.5,
        kappa: float = 1e4,
        c: float = 1500.0,
        rho0: float = 1000.0,
        beta: float = 3.5,
        mu_v: float = 6e-6,
        zeta: float = 0.0,
        dt_mode: str = "cfl",
        dt_factor: float = 0.2,
        force_recompute: bool = False,
) -> dict:

    figures_to_save = {}
    set_style(theme_name=theme)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Étude de la solution fabriquée
    study = convergence_study_manufactured(
        funcs=funcs,
        levels=levels,
        scheme=scheme,
        bc_type=bc_type,
        L=L,
        T=T,
        c=c,
        rho0=rho0,
        beta=beta,
        mu_v=mu_v,
        zeta=zeta,
        A=A,
        gamma=gamma,
        kappa=kappa,
        dt_mode=dt_mode,
        dt_factor=dt_factor,
        force_recompute=force_recompute,
    )


    table = build_manufactured_convergence_table(study)
    # table_semi = build_manufactured_convergence_table(study_semi)

    # finest_level_exp = max(study_exp["cases"].keys())
    # case_exp = study_exp["cases"][finest_level_exp]

    finest_level = max(study["cases"].keys())
    case = study["cases"][finest_level]

    fig1 = plot_manufactured_snapshots(case, field='U_num', title="Solution numérique", n_snapshots=8)
    fig2 = plot_manufactured_snapshots(case, field='U_ref', title="Solution exacte", n_snapshots=8)
    fig3 = plot_manufactured_comparison(case, snapshot_index=-1)
    fig4 = plot_manufactured_error_norm_over_time(case,  norm_type='L2', bc_type=bc_type, title=r"Évolution temporelle de la norme $\ell^2$ de l'erreur")
    fig5 = plot_manufactured_error_norm_over_time(case,  norm_type='H1', bc_type=bc_type, title=r"Évolution temporelle de la norme $H^1_h$ de l'erreur")
    # fig8 = plot_manufactured_convergence_comparison(table_exp, table_semi, error_col="Linf_L2", x_col="dx", title=r"Convergence en norme discrète $\ell^2$")

    if save and save_path is not None:
        if scheme == "semi_implicit":
            figures_to_save = {
                "semi_manufactured_snapshots_semi": fig1,
                "semi_manufactured_snapshots_ref": fig2,
                "semi_manufactured_comparison": fig3,
                "semi_manufactured_error_norm_over_time_L2": fig4,
                "semi_manufactured_error_norm_over_time_H1": fig5,
                # "manufactured_convergence_comparison": fig8,
            }
            save_many_figures(figures_to_save, output_dir=save_path)

            save_error_table_csv(
                "convergence_table_manufactured_semi_implicit",
                table.to_dict(orient="records"),
                output_dir=save_path,
            )
        elif scheme == "explicit":
            figures_to_save = {
                "exp_manufactured_snapshots": fig1,
                "exp_manufactured_snapshots_ref": fig2,
                "exp_manufactured_comparison": fig3,
                "exp_manufactured_error_norm_over_time_L2": fig4,
                "exp_manufactured_error_norm_over_time_H1": fig5,
            }
            save_many_figures(figures_to_save, output_dir=save_path)

            save_error_table_csv(
                "convergence_table_manufactured_explicit",
                table.to_dict(orient="records"),
                output_dir=save_path,
            )
        else:
            raise ValueError(f"Schéma {scheme} inconnu. Choisissez 'explicit' ou 'semi_implicit'.")


    print("\nTable de convergence.")
    print(table[["N", "dx", "dt", "Linf_L2", "order_Linf_L2", "Linf_H1", "order_Linf_H1"]])

    if show:
        plt.show()
    else:
        plt.close('all')

    return {
        "study": study,
        "table": table,
        "case": case,
        "figures": figures_to_save,
    }


if __name__ == "__main__":

    funcs = build_numerics_function(bc_type="neumann")

    refinement_dir = OUTPUTS_DIR / "refinement_plots"
    manufactured_dir = OUTPUTS_DIR / "manufactured_plots"

    run_refinement_plots(c=1930.0, rho0=1259.0, beta=5.4, mu_v=0.988, zeta=0.790, scheme="semi_implicit", bc_type="neumann", show=True, save=True, save_path=refinement_dir, theme="scientific", force_recompute=False)
    """
    run_manufactured_plots(
        funcs,
        bc_type="neumann",
        scheme="semi_implicit",
        show=True,
        save=False,
        save_path=manufactured_dir,
        theme="scientific",
        levels=[0, 1, 2, 3, 4, ],
        L=0.2,
        T=37e-6,
        A=1e-3,
        gamma=0.5,
        kappa=1e4,
        c=1930.0,  # glycérol (eau : 1500.0)
        rho0=1259.0,  # glycérol (eau : 1000.0)
        beta=5.4,  # glycérol (eau : 3.5)
        mu_v=0.988,  # glycérol (eau : 6e-6)
        zeta=0.790,  # glycérol — nouveau paramètre à ajouter
        dt_mode="cfl",
        dt_factor=0.2,
        force_recompute=True,  # invalider le cache eau
    )

    # """
