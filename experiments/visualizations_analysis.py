"""
Toutes les visualisations pour l'analyse - Westervelt Simulation

Ce script presente les visualisations disponibles pour analyser les resultats
du modele de Westervelt.

Parties:
1. Visualisations de base
2. Visualisations comparatives avec subplot_mosaic
3. Guide de selection rapide
"""

import pandas as pd

from core.solver import WesterveltSolver, WesterveltParams
from core.postprocessing import (
    plot_stability_detailed,
    plot_snapshots_energy_comparison,
    plot_scheme_comparison,
    plot_stability_scan,
)

print("Imports OK")

# =============================================================================
# SETUP INITIAL - Configuration commune du solver
# =============================================================================

COMMON_PARAMS = dict(
    c=1500,      # Vitesse du son (m/s)
    rho0=1000,   # Masse volumique de repos (kg/m^3)
    beta=4.8,    # Coefficient de non-linearite
    mu_v=1e-3,   # Viscosite de volume (Pa.s)
    dx=1e-4,     # Discretisation spatiale (m)
    dt=2e-8,     # Discretisation temporelle (s)
    nx=1200,     # Nombre de points spatiaux
    nt=1200,     # Nombre d'iterations temporelles
    bc="dirichlet",  # Conditions aux limites
)

params_explicit = WesterveltParams(**COMMON_PARAMS, scheme="explicit")
params_semi_implicit = WesterveltParams(**COMMON_PARAMS, scheme="semi_implicit")


def print_stability_criteria(params: WesterveltParams, label: str):
    """Affiche les nouveaux indicateurs de stabilite pour un schema donne."""
    solver = WesterveltSolver(params)
    cfl = params.c * params.dt / params.dx
    lambda_legacy = (params.c ** 2) * params.dt / (params.dx ** 2)

    if params.scheme == "explicit":
        margin = solver.explicit_stability_margin()
        theoretical_stable = solver.explicit_theoretical_stable()
    else:
        margin = solver.semi_implicit_stability_margin()
        theoretical_stable = solver.semi_implicit_theoretical_stable()

    print(f"{label}")
    print(f"  CFL = {cfl:.4f}")
    print(f"  lambda_legacy = {lambda_legacy:.4f}")
    print(f"  stability_margin = {margin:.6e}")
    print(f"  theoretical_stable = {theoretical_stable}")


print("Parametres configures")
print_stability_criteria(params_explicit, "Schema explicite")
print_stability_criteria(params_semi_implicit, "Schema semi-implicite")

# =============================================================================
# PARTIE 1 : VISUALISATIONS DE BASE
# Methodes simples et directes du solver pour une verification rapide.
# =============================================================================

print("\n" + "=" * 80)
print("PARTIE 1 : VISUALISATIONS DE BASE")
print("=" * 80)

# Visualisation 1A : Solution finale explicite
print("\n1A - Visualisation 1: Solution finale (explicite)")
print("-" * 40)

solver_explicit_solution = WesterveltSolver(params_explicit)
solver_explicit_solution.initialize(u0_type="gaussian")
solver_explicit_solution.run(store_energy=True)
solver_explicit_solution.plot_solution()

# Visualisation 1B : Solution finale semi-implicite
print("\n1B - Visualisation 1: Solution finale (semi-implicite)")
print("-" * 40)

solver_semi_solution = WesterveltSolver(params_semi_implicit)
solver_semi_solution.initialize(u0_type="gaussian")
solver_semi_solution.run(store_energy=True)
solver_semi_solution.plot_solution()

# Visualisation 2 : Snapshots spatiaux
print("\n2 - Visualisation 2: Snapshots spatiaux")
print("-" * 40)

solver2 = WesterveltSolver(params_semi_implicit)
solver2.initialize(u0_type="gaussian")

# Choisir des instants pour sauvegarder
times = [0.0, 1e-6, 2e-6, 4e-6, 6e-6, 8e-6]
snapshots = solver2.run_with_snapshots(times, store_energy=True)

print(f"Snapshots sauvegardes a {len(snapshots)} instants")
solver2.plot_snapshots(snapshots)

# Visualisation 3 : Historique d'energie
print("\n3 - Visualisation 3: Historique d'energie")
print("-" * 40)

print(f"Energie initiale: {solver2.energy_history[0]:.6e}")
print(f"Energie finale: {solver2.energy_history[-1]:.6e}")
solver2.plot_energy()

# =============================================================================
# PARTIE 2 : VISUALISATIONS COMPARATIVES AVEC SUBPLOT_MOSAIC
# Analyses connexes presentees cote a cote dans une seule figure.
# =============================================================================

print("\n" + "=" * 80)
print("PARTIE 2 : VISUALISATIONS COMPARATIVES AVEC SUBPLOT_MOSAIC")
print("=" * 80)

# Visualisation 4 : Snapshots vs energie (cote a cote)
print("\n4 - Visualisation 4: Snapshots vs Energie (subplot_mosaic)")
print("-" * 40)

solver3 = WesterveltSolver(params_semi_implicit)
solver3.initialize(u0_type="gaussian")
snapshots3 = solver3.run_with_snapshots(times, store_energy=True)

plot_snapshots_energy_comparison(
    solver3,
    snapshots3,
    title_prefix="Schema semi-implicite: ",
)

print("Avantages:")
print("  - Cote a cote: voir 2 analyses simultanement")
print("  - Correlation: identifier liens spatial/energetique")
print("  - Espace: 1 figure au lieu de 2")

# Visualisation 5 : Stabilite detaillee (nouveaux criteres)
print("\n5 - Visualisation 5: Stabilite detaillee (subplot_mosaic)")
print("-" * 40)

# Parametres pour le balayage de stabilite
dt_values = [1.0e-8, 1.5e-8, 2.0e-8, 2.5e-8, 3.0e-8]
amp_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

print(f"Balayage sur {len(dt_values)} x {len(amp_values)} = {len(dt_values) * len(amp_values)} configurations...")
print("Ceci peut prendre 1-2 minutes...")

solver4 = WesterveltSolver(params_semi_implicit)
stability_results = solver4.run_stability_scan(
    dt_values=dt_values,
    amplitude_values=amp_values,
    u0_type="gaussian",
    blowup_threshold=1e5,
)

# Statistiques de stabilite basees sur les nouveaux criteres
stable_count = sum(1 for r in stability_results if r["stable"])
theoretical_count = sum(1 for r in stability_results if r["theoretical_stable"])
nondegenerate_count = sum(1 for r in stability_results if r["nondegenerate"])
consistent_count = sum(
    1 for r in stability_results if bool(r["stable"]) == bool(r["theoretical_stable"])
)

print("\nBalayage termine")
print(f"  Configurations stables observees: {stable_count}/{len(stability_results)}")
print(f"  Configurations stables theoriques: {theoretical_count}/{len(stability_results)}")
print(f"  Configurations non degeneres (min_denom > 0): {nondegenerate_count}/{len(stability_results)}")
print(f"  Concordance observe/theorie: {consistent_count}/{len(stability_results)}")

plot_stability_detailed(stability_results)

print("Affiche 4 analyses connexes:")
print("  - Stabilite observee")
print("  - Marge de stabilite theorique")
print("  - Minimum de (1 - 2ku)")
print("  - Max|u| (blow-up)")

# Visualisation 6 : Carte simple de stabilite
print("\n6 - Visualisation 6: Carte simple de stabilite")
print("-" * 40)

plot_stability_scan(stability_results)

print("Caracteristiques:")
print("  - Type: Heatmap simple")
print("  - Utilite: Vue d'ensemble rapide")

# Visualisation 7 : Comparaison de deux schemas
print("\n7 - Visualisation 7: Comparaison de deux schemas (subplot_mosaic)")
print("-" * 40)

solver_explicit = WesterveltSolver(params_explicit)
solver_semi = WesterveltSolver(params_semi_implicit)

# Utiliser un subset de valeurs pour accelerer
dt_values_subset = [1.5e-8, 2.5e-8]
amp_values_subset = [1.0, 2.0, 3.0]

print("Balayage EXPLICITE...")
stability_explicit = solver_explicit.run_stability_scan(
    dt_values=dt_values_subset,
    amplitude_values=amp_values_subset,
    blowup_threshold=1e5,
)
print("Balayage EXPLICITE termine")

print("Balayage SEMI-IMPLICITE...")
stability_semi = solver_semi.run_stability_scan(
    dt_values=dt_values_subset,
    amplitude_values=amp_values_subset,
    blowup_threshold=1e5,
)
print("Balayage SEMI-IMPLICITE termine")

plot_scheme_comparison(stability_explicit, stability_semi)

print("Affiche:")
print("  - Schema explicite: stabilite observee")
print("  - Schema semi-implicite: stabilite observee")
print("  - Difference (Bleu=semi-implicite meilleur, Rouge=explicite meilleur)")

# =============================================================================
# PARTIE 3 : GUIDE DE SELECTION RAPIDE
# =============================================================================

print("\n" + "=" * 80)
print("PARTIE 3 : GUIDE DE SELECTION RAPIDE")
print("=" * 80)

summary_data = [
    {"#": "1A", "Fonction": "plot_solution()", "Affiche": "Solution finale explicite", "Priorite": "haute"},
    {"#": "1B", "Fonction": "plot_solution()", "Affiche": "Solution finale semi-implicite", "Priorite": "haute"},
    {"#": "2", "Fonction": "plot_snapshots()", "Affiche": "Snapshots", "Priorite": "haute"},
    {"#": "3", "Fonction": "plot_energy()", "Affiche": "Energie", "Priorite": "haute"},
    {"#": "4", "Fonction": "plot_snapshots_energy_comparison()", "Affiche": "Snapshots + Energie", "Priorite": "tres haute"},
    {"#": "5", "Fonction": "plot_stability_detailed()", "Affiche": "4 cartes", "Priorite": "tres haute"},
    {"#": "6", "Fonction": "plot_stability_scan()", "Affiche": "Carte simple", "Priorite": "haute"},
    {"#": "7", "Fonction": "plot_scheme_comparison()", "Affiche": "Explicite vs Semi-implicite", "Priorite": "tres haute"},
]

df = pd.DataFrame(summary_data)
print("\nRESUME DES VISUALISATIONS\n")
print(df.to_string(index=False))

guide = """
GUIDE DE SELECTION RAPIDE
=============================================================

Situation                                    Utiliser
-------------------------------------------------------------
Verification rapide apres simulation         Viz 1A + 1B + 3
Analyser l'evolution spatiale                Viz 2
Correler spatial et energetique              Viz 4 (subplot_mosaic)
Etudier la stabilite rapidement              Viz 6
Analyser la stabilite en detail              Viz 5 (subplot_mosaic)
Comparer deux schemas                        Viz 7 (subplot_mosaic)

RECOMMANDATIONS:
- Les visualisations 4, 5, 6 et 7 montrent des analyses connexes.
- Viz 5 utilise les nouveaux criteres: stability_margin, theoretical_stable,
  min_denom, nondegenerate.
"""

print(guide)

advantages = """
AVANTAGES DE subplot_mosaic
=============================================================

AVANT (figures separees)         APRES (subplot_mosaic)
-------------------------------------------------------------
Figure 1 + Figure 2              Une seule figure
Basculer entre fenetres          Analyses cote a cote
Comparer manuellement             Comparaison directe
Disposition heterogene            Axes alignes automatiquement

DETAILS:
1. Flexibilite des layouts
2. Comparaison visuelle immediate
3. Moins de fenetres ouvertes
4. Coherence des titres/axes/colorbars
5. Figures plus lisibles en rapport
"""

print(advantages)

workflow = """
WORKFLOWS TYPIQUES
=============================================================

WORKFLOW 1: Validation rapide (2-3 min)
1. Configurer le solver explicite et semi-implicite
2. run(store_energy=True)
3. plot_solution() pour chaque schema
4. plot_energy()

WORKFLOW 2: Analyse snapshots (5-10 min)
1. Definir times = [0.0, 1e-6, 2e-6, ...]
2. snapshots = run_with_snapshots(times, store_energy=True)
3. plot_snapshots_energy_comparison(...)

WORKFLOW 3: Etude de stabilite (10-30 min)
1. Definir dt_values et amp_values
2. results = run_stability_scan(...)
3. plot_stability_detailed(results)
4. Lire: stabilite observee + marge theorique + min_denom + max|u|

WORKFLOW 4: Comparaison de schemas (1-2 h)
1. Lancer run_stability_scan pour explicite et semi-implicite
2. plot_scheme_comparison(results_explicit, results_semi)
"""

print(workflow)

conclusion = """
IMPLEMENTATION COMPLETE
=============================================================

Le script fournit maintenant:
- Affichage separe des solutions explicite et semi-implicite
- Conservation des visualisations existantes
- Etude de stabilite mise a jour avec les nouveaux criteres
- Suppression des emoticones

Objectif atteint:
Utiliser subplot_mosaic pour les analyses connexes et maintenir
une lecture claire des resultats de stabilite.
"""

print(conclusion)

# =============================================================================
# FIN DU SCRIPT
# =============================================================================

print("\n" + "=" * 80)
print("SCRIPT COMPLET - TOUTES LES VISUALISATIONS EXECUTEES")
print("=" * 80)

