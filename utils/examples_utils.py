"""
Exemples d'utilisation des utilitaires consolidés.

Ce script montre comment utiliser les fonctions de utils.py
pour sauvegarder, analyser et valider les simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    # Versioning
    save_figure_with_version,
    save_data_with_version,
    
    # Analyse de scans
    get_scan_axes,
    build_scan_grid,
    compute_stable_ratio,
    
    # Validation numérique
    compute_cfl_number,
    check_cfl_stability,
    analyze_scheme_properties,
    compare_schemes,
    estimate_memory_usage,
    print_simulation_summary,
    
    # Utilitaires
    compute_error_metrics,
    normalize_array,
)


def exemple_versioning_figures():
    """Exemple: Sauvegarde de figures avec versioning automatique."""
    print("\n" + "="*70)
    print("EXEMPLE 1: Versioning automatique de figures")
    print("="*70)
    
    # Créer une figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")
    ax.legend()
    ax.set_title("Fonctions trigonométriques")
    
    # Sauvegarder avec métadonnées
    paths = save_figure_with_version(
        fig,
        "example_trigono",
        output_dir="../outputs/figures",
        formats=["png", "pdf"],
        metadata={
            "type": "example",
            "functions": "sin, cos",
            "domain": "[0, 10]"
        }
    )
    
    print(f"\n✓ Figure sauvegardée:")
    for fmt, path in paths.items():
        print(f"  - {fmt}: {path}")
    
    plt.close(fig)


def exemple_analyse_stabilite():
    """Exemple: Analyse de stabilité avec scan results."""
    print("\n" + "="*70)
    print("EXEMPLE 2: Analyse de stabilité")
    print("="*70)
    
    # Simuler des résultats de scan
    results = []
    dt_values = [1e-8, 2e-8, 3e-8]
    amp_values = [0.05, 0.1, 0.15]
    
    for dt in dt_values:
        for amp in amp_values:
            cfl = compute_cfl_number(1500, dt, 1e-4)
            results.append({
                "dt": dt,
                "amplitude": amp,
                "cfl": cfl,
                "stable": cfl <= np.sqrt(2),  # semi-implicite
                "max_u": amp * (1.5 if cfl > 0.5 else 1.0)
            })
    
    # Extraire les axes
    dt_vals, amp_vals = get_scan_axes(results)
    print(f"\nVin résultats de scan:")
    print(f"  • dt values: {dt_vals}")
    print(f"  • amplitude values: {amp_vals}")
    
    # Construire la grille de stabilité
    stability_grid = build_scan_grid(
        results,
        dt_vals,
        amp_vals,
        lambda r: 1.0 if r["stable"] else 0.0
    )
    
    # Ratio de stabilité
    stable_ratio = compute_stable_ratio(results)
    print(f"  • Ratio stable: {stable_ratio:.2%}")
    
    # Visualiser
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grille de stabilité
    im1 = ax1.imshow(stability_grid, cmap="RdYlGn", aspect="auto", origin="lower")
    ax1.set_xlabel("dt (index)")
    ax1.set_ylabel("amplitude (index)")
    ax1.set_title("Carte de stabilité")
    plt.colorbar(im1, ax=ax1)
    
    # Grille de max_u
    max_u_grid = build_scan_grid(
        results,
        dt_vals,
        amp_vals,
        lambda r: r["max_u"]
    )
    im2 = ax2.imshow(max_u_grid, cmap="viridis", aspect="auto", origin="lower")
    ax2.set_xlabel("dt (index)")
    ax2.set_ylabel("amplitude (index)")
    ax2.set_title("Maximum de |u|")
    plt.colorbar(im2, ax=ax2)
    
    save_figure_with_version(
        fig,
        "stability_analysis",
        output_dir="../outputs/analysis",
        metadata={"analysis": "stability_scan"}
    )
    
    plt.close(fig)


def exemple_validation_numerique():
    """Exemple: Validation numérique et analyse CFL."""
    print("\n" + "="*70)
    print("EXEMPLE 3: Validation numérique")
    print("="*70)
    
    # Paramètres
    c = 1500  # m/s
    dt = 2e-8  # s
    dx = 1e-4  # m
    
    # Calcul CFL
    cfl = compute_cfl_number(c, dt, dx)
    print(f"\nCalcul du nombre CFL:")
    print(f"  • c = {c} m/s")
    print(f"  • dt = {dt} s")
    print(f"  • dx = {dx} m")
    print(f"  • CFL = {cfl:.4f}")
    
    # Vérification de stabilité pour différents schémas
    print(f"\nVérification de stabilité par schéma:")
    for scheme in ["explicit", "semi_implicit", "implicit"]:
        is_stable, msg = check_cfl_stability(cfl, scheme)
        print(f"  • {msg}")


def exemple_analyse_schemas():
    """Exemple: Analyse des propriétés des schémas."""
    print("\n" + "="*70)
    print("EXEMPLE 4: Analyse des schémas numériques")
    print("="*70)
    
    # Paramètres
    params = {
        "c": 1500,
        "dt": 2e-8,
        "dx": 1e-4,
        "nx": 1200,
        "nt": 1200
    }
    
    # Résumé complet
    summary = print_simulation_summary(params, scheme_name="semi_implicit")
    
    # Comparaison des schémas
    print("\nComparaison des schémas:")
    comparaison = compare_schemes(
        c=params["c"],
        dt=params["dt"],
        dx=params["dx"],
        schemes=["explicit", "semi_implicit", "implicit"]
    )
    
    for scheme_info in comparaison:
        print(f"\n  {scheme_info['scheme'].upper()}:")
        print(f"    - Ordre spatial: {scheme_info['order_space']}")
        print(f"    - Ordre temporel: {scheme_info['order_time']}")
        print(f"    - Stabilité: {scheme_info['stability']}")
        print(f"    - Coût: {scheme_info['computational_cost']}")


def exemple_metriques_erreur():
    """Exemple: Calcul de métriques d'erreur."""
    print("\n" + "="*70)
    print("EXEMPLE 5: Métriques d'erreur et convergence")
    print("="*70)
    
    # Créer deux solutions (une "vraie", une approchée)
    x = np.linspace(0, 1, 100)
    solution_exacte = np.sin(np.pi * x)
    solution_approx = np.sin(np.pi * x) + 0.01 * np.random.randn(100)
    
    # Calcul des erreurs
    metrics = compute_error_metrics(solution_approx, solution_exacte)
    print(f"\nMétriques d'erreur:")
    for name, value in metrics.items():
        print(f"  • {name}: {value:.6e}")
    
    # Normalisations
    solution_normalized = normalize_array(solution_approx, mode="minmax")
    print(f"\nSolution normalisée (minmax):")
    print(f"  • Min: {solution_normalized.min():.6f}")
    print(f"  • Max: {solution_normalized.max():.6f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DÉMONSTRATION DES UTILITAIRES CONSOLIDÉS")
    print("="*70)
    
    # Exécuter les exemples
    exemple_versioning_figures()
    exemple_analyse_stabilite()
    exemple_validation_numerique()
    exemple_analyse_schemas()
    exemple_metriques_erreur()
    
    print("\n" + "="*70)
    print("✓ Tous les exemples ont été exécutés avec succès!")
    print("="*70 + "\n")

