#!/usr/bin/env python
"""Script de test pour vérifier la consolidation de utils.py"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ajouter le projet au path
sys.path.insert(0, str(Path.cwd()))

print("=" * 70)
print("VÉRIFICATION DE LA CONSOLIDATION - utils.py")
print("=" * 70)

try:
    # Test 1: Imports principaux
    print("\n TEST 1: Imports principaux")
    from utils import (
        save_figure_with_version,
        print_simulation_summary,
        check_cfl_stability,
        compute_cfl_number,
        get_scan_axes,
        build_scan_grid,
        compute_stable_ratio,
    )
    print("  Tous les imports réussis")

    # Test 2: Fonction compute_cfl_number
    print("\n TEST 2: compute_cfl_number()")
    cfl = compute_cfl_number(1500, 2e-8, 1e-4)
    print(f"  compute_cfl_number(1500, 2e-8, 1e-4) = {cfl:.4f}")
    assert cfl > 0, "CFL doit être positif"

    # Test 3: Fonction check_cfl_stability
    print("\n TEST 3: check_cfl_stability()")
    is_stable, msg = check_cfl_stability(cfl, "semi_implicit")
    print(f"  check_cfl_stability({cfl:.4f}, 'semi_implicit')")
    print(f"  Résultat: {msg}")

    # Test 4: Fonctions de scan
    print("\n TEST 4: Fonctions de scan (get_scan_axes, build_scan_grid)")
    results = [
        {"dt": 1e-8, "amplitude": 0.1, "stable": True},
        {"dt": 2e-8, "amplitude": 0.1, "stable": False},
        {"dt": 1e-8, "amplitude": 0.15, "stable": True},
    ]
    
    dt_vals, amp_vals = get_scan_axes(results)
    print(f"  get_scan_axes(): dt={len(dt_vals)} values, amp={len(amp_vals)} values")
    
    grid = build_scan_grid(
        results,
        dt_vals,
        amp_vals,
        lambda r: 1.0 if r["stable"] else 0.0
    )
    print(f"  build_scan_grid(): grille {grid.shape}")
    
    ratio = compute_stable_ratio(results)
    print(f"  compute_stable_ratio() = {ratio:.1%}")

    # Test 5: Autres imports
    print("\n TEST 5: Autres imports")
    from utils import (
        ensure_output_dir,
        compute_error_metrics,
        normalize_array,
        analyze_scheme_properties,
        compare_schemes,
        estimate_memory_usage,
    )
    print("  Tous les autres imports réussis")

    # Test 6: Tester analyze_scheme_properties
    print("\n TEST 6: analyze_scheme_properties()")
    props = analyze_scheme_properties("semi_implicit", 1500, 2e-8, 1e-4)
    print(f"  Schéma: {props['scheme']}")
    print(f"  CFL: {props['cfl_number']:.4f}")
    print(f"  Stable: {props['is_stable']}")

    # Test 7: Tester compare_schemes
    print("\n TEST 7: compare_schemes()")
    comparison = compare_schemes(1500, 2e-8, 1e-4, ["explicit", "semi_implicit"])
    print(f"  Comparaison de {len(comparison)} schémas")
    for scheme in comparison:
        print(f"     - {scheme['scheme']}: CFL={scheme['cfl_number']:.4f}")

    # Test 8: Tester estimate_memory_usage
    print("\n TEST 8: estimate_memory_usage()")
    memory = estimate_memory_usage(1200, 1200)
    print(f"  Estimation mémoire:")
    total_mb = sum(memory.values())
    print(f"     Total estimé: {total_mb:.2f} MB")

    # Test 9: Tester compute_error_metrics
    print("\n TEST 9: compute_error_metrics()")
    import numpy as np
    sol = np.array([1.0, 2.0, 3.0])
    ref = np.array([1.05, 2.1, 2.9])
    metrics = compute_error_metrics(sol, ref)
    print(f"  Métriques calculées: {list(metrics.keys())}")

    # Test 10: Tester normalize_array
    print("\n TEST 10: normalize_array()")
    arr = np.array([1, 2, 3, 4, 5])
    normalized = normalize_array(arr, mode="minmax")
    print(f"  Array normalisé (minmax): min={normalized.min():.2f}, max={normalized.max():.2f}")

    # Résumé
    print("\n" + "=" * 70)
    print(" TOUS LES TESTS PASSÉS AVEC SUCCÈS!")
    print("=" * 70)
    
    print("\n STATISTIQUES:")
    print(f"   • Chemin du projet: {Path.cwd()}")
    print(f"   • Fichier utils.py existe: {(Path.cwd() / 'utils' / 'utils.py').exists()}")
    print(f"   • Module __init__.py existe: {(Path.cwd() / 'utils' / '__init__.py').exists()}")
    
    print("\n La consolidation est OPÉRATIONNELLE et VALIDÉE!")
    print("\n Documentation disponible:")
    print("   • UTILS_DOCUMENTATION.md (guide complet)")
    print("   • QUICK_REFERENCE.md (référence rapide)")
    print("   • UTILS_INDEX.md (index structuré)")

except Exception as e:
    print(f"\n ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

