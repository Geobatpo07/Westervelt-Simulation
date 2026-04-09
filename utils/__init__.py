"""
Package d'utilitaires pour le projet Westervelt Simulation.

Contient toutes les fonctions utilitaires pour:
- Gestion des fichiers et versioning des figures
- Analyse de stabilité et des grilles de scans
- Validation numérique et analyse des schémas
- Utilitaires mathématiques et matplotlib
"""

from utils.utils import (
    # Gestion des fichiers et versioning
    ensure_output_dir,
    get_next_version,
    save_figure_with_version,
    save_data_with_version,

    # Gestion des scans
    build_scan_grid,
    get_scan_axes,
    compute_stable_ratio,

    # Utilitaires mathématiques
    compute_error_metrics,
    normalize_array,

    # Matplotlib
    set_style,
    create_comparison_figure,

    # Logging
    log_computation_params,
    print_progress,

    # Validation numérique
    compute_cfl_number,
    check_cfl_info,
    check_cfl_stability,
    compute_lambda_number,
    check_lambda_stability,
    compute_convergence_rate,
    estimate_error_bounds,

    # Analyse des schémas
    analyze_scheme_properties,
    compare_schemes,
    estimate_memory_usage,
    print_simulation_summary,
)

__all__ = [
    # Gestion des fichiers
    "ensure_output_dir",
    "get_next_version",
    "save_figure_with_version",
    "save_data_with_version",

    # Gestion des scans
    "build_scan_grid",
    "get_scan_axes",
    "compute_stable_ratio",

    # Utilitaires mathématiques
    "compute_error_metrics",
    "normalize_array",

    # Matplotlib
    "set_style",
    "create_comparison_figure",

    # Logging
    "log_computation_params",
    "print_progress",

    # Validation numérique
    "compute_cfl_number",
    "check_cfl_info",
    "check_cfl_stability",
    "compute_lambda_number",
    "check_lambda_stability",
    "compute_convergence_rate",
    "estimate_error_bounds",

    # Analyse des schémas
    "analyze_scheme_properties",
    "compare_schemes",
    "estimate_memory_usage",
    "print_simulation_summary",
]
