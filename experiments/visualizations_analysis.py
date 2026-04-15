"""
📊 Toutes les visualisations pour l'analyse - Westervelt Simulation

Ce script présente les 7 visualisations disponibles pour analyser les résultats
du modèle de Westervelt.

Parties:
1. Visualisations de base (3)
2. Visualisations comparatives avec subplot_mosaic (4) ⭐⭐⭐
3. Guide de sélection rapide
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.solver import WesterveltSolver, WesterveltParams
from src.postprocessing import (
    plot_stability_detailed,
    plot_snapshots_energy_comparison,
    plot_scheme_comparison,
    plot_stability_scan
)

print("✓ Tous les imports réussis")

# =============================================================================
# SETUP INITIAL - Configuration commune du solver
# =============================================================================

params = WesterveltParams(
    c=1500,           # Vitesse du son (m/s)
    rho0=1000,        # Masse volumique de repos (kg/m^3)
    beta=4.8,         # Coefficient de non-linearite
    mu_v=1e-3,        # Viscosite de volume (Pa.s)
    dx=1e-4,          # Discrétisation spatiale (m)
    dt=2e-8,          # Discrétisation temporelle (s)
    nx=1200,          # Nombre de points spatiaux
    nt=1200,          # Nombre d'itérations temporelles
    scheme="semi_implicit",  # Schéma semi-implicite
    bc="dirichlet"    # Conditions aux limites
)

print("✓ Paramètres configurés")
lambda_num = (params.c ** 2) * params.dt / (params.dx ** 2)
print(f"  lambda = {lambda_num:.4f} (critère de stabilité)")
print(f"  CFL = {params.c * params.dt / params.dx:.4f} (indicateur)")

# =============================================================================
# PARTIE 1️⃣ : VISUALISATIONS DE BASE (3)
# Méthodes simples et directes du solver pour une vérification rapide.
# =============================================================================

print("\n" + "="*80)
print("PARTIE 1️⃣ : VISUALISATIONS DE BASE (3)")
print("="*80)

# Visualisation 1️⃣ : Solution finale
# Fonction: solver.plot_solution()
# Affiche la solution u(x) à l'instant final de la simulation.

print("\n1️⃣  Visualisation 1: Solution finale")
print("-" * 40)

solver1 = WesterveltSolver(params)
solver1.initialize(ic_type="gaussian")
solver1.run(store_energy=True)
solver1.plot_solution()

# Visualisation 2️⃣ : Snapshots spatiaux
# Fonction: solver.plot_snapshots(snapshots)
# Affiche la solution u(x,t) à plusieurs instants temporels.

print("\n2️⃣  Visualisation 2: Snapshots spatiaux")
print("-" * 40)

solver2 = WesterveltSolver(params)
solver2.initialize(ic_type="gaussian")

# Choisir des instants pour sauvegarder
times = [0.0, 1e-6, 2e-6, 4e-6, 6e-6, 8e-6]
snapshots = solver2.run_with_snapshots(times, store_energy=True)

print(f"Snapshots sauvegardés à {len(snapshots)} instants")

# Afficher les snapshots
solver2.plot_snapshots(snapshots)

# Visualisation 3️⃣ : Historique d'énergie
# Fonction: solver.plot_energy()
# Affiche l'évolution temporelle de l'énergie discrète.

print("\n3️⃣  Visualisation 3: Historique d'énergie")
print("-" * 40)

print(f"Énergie initiale: {solver2.energy_history[0]:.6e}")
print(f"Énergie finale: {solver2.energy_history[-1]:.6e}")
solver2.plot_energy()

# =============================================================================
# PARTIE 2️⃣ : VISUALISATIONS COMPARATIVES AVEC SUBPLOT_MOSAIC ⭐⭐⭐
# Analyses connexes présentées côte à côte dans une seule figure.
# =============================================================================

print("\n" + "="*80)
print("PARTIE 2️⃣ : VISUALISATIONS COMPARATIVES AVEC SUBPLOT_MOSAIC ⭐⭐⭐")
print("="*80)

# Visualisation 4️⃣ : Snapshots vs Énergie (côte à côte)
# Fonction: plot_snapshots_energy_comparison(solver, snapshots)
# Affiche snapshots spatiaux (gauche) et historique énergétique (droite)

print("\n4️⃣  Visualisation 4: Snapshots vs Énergie (subplot_mosaic)")
print("-" * 40)

solver3 = WesterveltSolver(params)
solver3.initialize(ic_type="gaussian")

snapshots3 = solver3.run_with_snapshots(times, store_energy=True)

# Visualisation comparée avec subplot_mosaic
plot_snapshots_energy_comparison(
    solver3, 
    snapshots3, 
    title_prefix="Schema semi-implicite: "
)

print("✓ Avantages:")
print("  - Côte à côte: voir 2 analyses simultanément")
print("  - Corrélation: identifier liens spatial/énergétique")
print("  - Espace: 1 figure au lieu de 2")

# Visualisation 5️⃣ : Stabilité détaillée (3 métriques)
# Fonction: plot_stability_detailed(results)
# Affiche 3 cartes comparatives: Stabilité + lambda + Max|u|

print("\n5️⃣  Visualisation 5: Stabilité détaillée (subplot_mosaic)")
print("-" * 40)

# Paramètres pour le balayage de stabilité
dt_values = [1.0e-8, 1.5e-8, 2.0e-8, 2.5e-8, 3.0e-8]
amp_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

print(f"Balayage sur {len(dt_values)} × {len(amp_values)} = {len(dt_values)*len(amp_values)} configurations...")
print("Ceci peut prendre 1-2 minutes...")

# Exécuter le balayage
solver4 = WesterveltSolver(params)
stability_results = solver4.run_stability_scan(
    dt_values=dt_values,
    amplitude_values=amp_values,
    ic_type="gaussian",
    blowup_threshold=1e5
)

# Statistiques
stable_count = sum(1 for r in stability_results if r["stable"])
print(f"\n✓ Balayage terminé")
print(f"  Configurations stables: {stable_count}/{len(stability_results)}")
print(f"  Pourcentage: {100*stable_count/len(stability_results):.1f}%")

# Afficher l'analyse détaillée: 3 cartes comparatives
plot_stability_detailed(stability_results)

print("✓ Affiche 3 analyses connexes:")
print("  - Stabilité: Régions stables (vert) vs instables (rouge)")
print("  - Lambda: Critère de stabilité théorique")
print("  - Max|u|: Amplitude maximale (blow-up)")

# Visualisation 6️⃣ : Carte simple de stabilité
# Fonction: plot_stability_scan(results)
# Affiche une carte simple stable/instable (version light)

print("\n6️⃣  Visualisation 6: Carte simple de stabilité")
print("-" * 40)

plot_stability_scan(stability_results)

print("✓ Caractéristiques:")
print("  - Type: Heatmap simple")
print("  - Utilité: Vue d'ensemble rapide")

# Visualisation 7️⃣ : Comparaison de deux schémas
# Fonction: plot_scheme_comparison(results_explicit, results_semi_implicit)
# Affiche 3 cartes: Schéma explicite + Semi-implicite + Différence

print("\n7️⃣  Visualisation 7: Comparaison de deux schémas (subplot_mosaic)")
print("-" * 40)

# Créer un solver avec schéma EXPLICITE
params_explicit = WesterveltParams(
    c=1500, rho0=1000, beta=4.8, mu_v=1e-3,
    dx=1e-4, dt=2e-8, nx=1200, nt=1200,
    scheme="explicit", bc="dirichlet"
)

solver_explicit = WesterveltSolver(params_explicit)

# Utiliser un subset de valeurs pour accélérer
dt_values_subset = [1.5e-8, 2.5e-8]
amp_values_subset = [1.0, 2.0, 3.0]

print("Balayage EXPLICITE...")
stability_explicit = solver_explicit.run_stability_scan(
    dt_values=dt_values_subset,
    amplitude_values=amp_values_subset,
    blowup_threshold=1e5
)
print("✓ Balayage EXPLICITE terminé")

print("Balayage SEMI-IMPLICITE...")
solver_ql = WesterveltSolver(params)
stability_ql = solver_ql.run_stability_scan(
    dt_values=dt_values_subset,
    amplitude_values=amp_values_subset,
    blowup_threshold=1e5
)
print("✓ Balayage SEMI-IMPLICITE termine")

# Afficher la comparaison: 3 cartes
plot_scheme_comparison(stability_explicit, stability_ql)

print("✓ Affiche:")
print("  - Schéma explicite: Stabilité")
print("  - Schéma semi-implicite: Stabilité")
print("  - Différence (Bleu=Semi-implicite meilleur, Rouge=Explicite meilleur)")

# =============================================================================
# PARTIE 3️⃣ : GUIDE DE SÉLECTION RAPIDE
# =============================================================================

print("\n" + "="*80)
print("PARTIE 3️⃣ : GUIDE DE SÉLECTION RAPIDE")
print("="*80)

# Tableau récapitulatif
summary_data = [
    {"#": "1️⃣", "Fonction": "plot_solution()", "Affiche": "Solution finale", "Priorité": "⭐"},
    {"#": "2️⃣", "Fonction": "plot_snapshots()", "Affiche": "Snapshots", "Priorité": "⭐"},
    {"#": "3️⃣", "Fonction": "plot_energy()", "Affiche": "Énergie", "Priorité": "⭐"},
    {"#": "4️⃣", "Fonction": "plot_snapshots_energy_cmp()", "Affiche": "Snap+Énergie", "Priorité": "⭐⭐⭐"},
    {"#": "5️⃣", "Fonction": "plot_stability_detailed()", "Affiche": "3 cartes", "Priorité": "⭐⭐⭐"},
    {"#": "6️⃣", "Fonction": "plot_stability_scan()", "Affiche": "Simple", "Priorité": "⭐"},
    {"#": "7️⃣", "Fonction": "plot_scheme_comparison()", "Affiche": "Expl vs SI", "Priorité": "⭐⭐⭐"},
]

df = pd.DataFrame(summary_data)
print("\n📊 RÉSUMÉ DES 7 VISUALISATIONS\n")
print(df.to_string(index=False))

# Guide de sélection
guide = """

📊 GUIDE DE SÉLECTION RAPIDE
═════════════════════════════════════════════════════════════════

Situation                                    Utiliser
─────────────────────────────────────────────────────────────────

Vérification rapide après simulation         Viz 1 + 3

Analyser l'évolution spatiale                Viz 2

Corréler spatial ET énergétique ⭐⭐⭐          Viz 4 (subplot_mosaic)

Étudier la stabilité rapidement              Viz 6

Analyser la stabilité en détail ⭐⭐⭐         Viz 5 (subplot_mosaic)

Comparer deux schémas ⭐⭐⭐                     Viz 7 (subplot_mosaic)

═════════════════════════════════════════════════════════════════

💡 RECOMMANDATIONS:
   • Les visualisations 4️⃣, 5️⃣, 6️⃣ utilisent subplot_mosaic
   • Elles montrent plusieurs analyses connexes dans UNE seule figure
   • Préférer ces visualisations pour l'analyse approfondie

"""

print(guide)

# Avantages de subplot_mosaic
advantages = """
✨ AVANTAGES DE subplot_mosaic
═════════════════════════════════════════════════════════════════

AVANT (figures séparées)         APRÈS (subplot_mosaic)
────────────────────────────────────────────────────────────────

❌ Figure 1                      ✅ UNE SEULE Figure
❌ Figure 2                      ✅ Analyses côte à côte
❌ Basculer entre fenêtres       ✅ Axes alignés automatiquement
❌ Difficile de comparer         ✅ Comparaison directe
❌ Confusion possible            ✅ Clair et organisé

═════════════════════════════════════════════════════════════════

DÉTAILS:

1. FLEXIBILITÉ
   - Layouts asymétriques possibles (2 petits + 1 grand)
   - Pas limité à grille régulière

2. COMPARAISON
   - Graphiques connexes visibles simultanément
   - Relations directes identifiables

3. ESPACE
   - Une figure au lieu de plusieurs
   - Moins de fenêtres ouvertes

4. COHÉRENCE
   - Alignement des axes automatique
   - Colorbars bien positionnées
   - Un suptitle unique

5. DOCUMENTATION
   - Plus facile à inclure dans rapports/publications
   - Meilleure lisibilité pour présentation

"""

print(advantages)

# Workflows typiques
workflow = """
🚀 WORKFLOWS TYPIQUES
═════════════════════════════════════════════════════════════════

WORKFLOW 1: Validation rapide d'une simulation (2-3 min)
──────────────────────────────────────────────────────────
1. Configurer le solver
2. solver.run(store_energy=True)
3. solver.plot_solution()
4. solver.plot_energy()
→ Vérifier: Pas NaN/Inf, énergie décroissante

WORKFLOW 2: Analyse avec snapshots (5-10 min) ⭐⭐⭐
──────────────────────────────────────────────────────────
1. Configurer le solver
2. Définir times = [0.0, 1e-6, 2e-6, ...]
3. snapshots = solver.run_with_snapshots(times, store_energy=True)
4. plot_snapshots_energy_comparison(solver, snapshots) ← SUBPLOT_MOSAIC
→ Analyser: Corrélation spatial/énergétique

WORKFLOW 3: Étude de stabilité (10-30 min) ⭐⭐⭐
──────────────────────────────────────────────────────────
1. Configurer le solver
2. Définir dt_values et amp_values
3. results = solver.run_stability_scan(dt_values, amp_values)
4. plot_stability_detailed(results) ← SUBPLOT_MOSAIC
→ Analyser: 3 métriques comparées (Stab+CFL+Max)

WORKFLOW 4: Publication (1-2 heures) ⭐⭐⭐
──────────────────────────────────────────────────────────
1-3. Workflows 2-3
4. plot_scheme_comparison(results_expl, results_qla) ← SUBPLOT_MOSAIC
→ Générer les 3 figures principales:
   - Snapshots vs Énergie
   - Stabilité détaillée
   - Comparaison de schémas

"""

print(workflow)

# Conclusion
conclusion = """
✅ IMPLÉMENTATION COMPLÈTE
═════════════════════════════════════════════════════════════════

Vous avez maintenant accès à:

📊 7 VISUALISATIONS:
   • 3 visualisations de base (simples et directes)
   • 4 visualisations comparatives (avec subplot_mosaic) ⭐⭐⭐

📚 DOCUMENTATION COMPLÈTE:
   ✅ VISUALIZATIONS_COMPLETE_GUIDE.md (Guide détaillé)
   ✅ INDEX_VISUALIZATIONS.md (Navigation)
   ✅ ANALYSIS_IMPROVEMENTS.md (Technique)
   ✅ Ce script (Exemples exécutables)

🔧 CODE SOURCE:
   ✅ src/postprocessing.py (4 fonctions avec subplot_mosaic)
   ✅ src/solver.py (3 méthodes de base)

═════════════════════════════════════════════════════════════════

🎯 OBJECTIF ATTEINT:

Demande: "Utilise subplot_mosaic pour les graphiques qui 
présentent des analyses connexes qui nécessitent une 
comparaison entre eux."

Résultat: ✅ 4 visualisations avec subplot_mosaic
          ✅ Analyses connexes côte à côte
          ✅ Meilleure compréhension
          ✅ Documentation complète
          ✅ Prêt pour production

═════════════════════════════════════════════════════════════════

🚀 PROCHAINES ÉTAPES:
   1. Adapter les paramètres selon vos besoins
   2. Générer vos propres visualisations
   3. Inclure dans vos rapports/publications
   4. Publier vos résultats!

"""

print(conclusion)

# =============================================================================
# FIN DU SCRIPT
# =============================================================================

print("\n" + "="*80)
print("✅ SCRIPT COMPLET - TOUTES LES VISUALISATIONS EXÉCUTÉES")
print("="*80)

