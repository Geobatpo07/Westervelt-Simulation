"""
Package d'utilitaires pour le projet Westervelt Simulation.

Contient des outils pour la gestion de fichiers, le versionnage, l'analyse
de stabilité, la validation numérique et la visualisation.
"""

from utils.utils import (
    # Décorateurs
    timer,
    log_execution,
    deprecated,
    validate_shape,
    profile,

    # Gestion du cache
    sanitize_float,
    build_cache_name,
    build_manufactured_cache_name,
    find_cached_solution,

    # Gestion des fichiers et versioning
    ensure_output_dir,
    get_next_version,
    append_profiler_record_csv,
    save_figure_with_version,
    save_data_with_version,

    # Gestion des solutions
    save_solution_npz,
    load_solution_npz,
    save_solution_npy,
    load_solution_npy,
    save_solution_csv_long,
    load_solution_csv_long,
    save_error_table_csv,
    save_manufactured_solution_npz,
    load_manufactured_solution_npz,

    # Gestion des scans
    build_scan_grid,
    get_scan_axes,
    compute_stable_ratio,

    # Utilitaires mathématiques
    compute_error_metrics,
    compute_gradient,
    compute_linf_time_error,
    compute_convergence_orders,
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
    # Décorateurs
    "timer",
    "log_execution",
    "deprecated",
    "validate_shape",
    "profile",

    # Gestion du cache
    "sanitize_float",
    "build_cache_name",
    "build_manufactured_cache_name",
    "find_cached_solution",

    # Gestion des fichiers
    "ensure_output_dir",
    "get_next_version",
    "append_profiler_record_csv",
    "save_figure_with_version",
    "save_data_with_version",
    "save_solution_npz",
    "load_solution_npz",
    "save_solution_npy",
    "load_solution_npy",
    "save_solution_csv_long",
    "load_solution_csv_long",
    "save_error_table_csv",
    "save_manufactured_solution_npz",
    "load_manufactured_solution_npz",

    # Gestion des scans
    "build_scan_grid",
    "get_scan_axes",
    "compute_stable_ratio",

    # Utilitaires mathématiques
    "compute_error_metrics",
    "compute_gradient",
    "compute_linf_time_error",
    "compute_convergence_orders",
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
