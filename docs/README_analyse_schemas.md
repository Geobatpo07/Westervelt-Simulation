# Analyse des schemas Westervelt

Script: `experiments/analyse_schemas.py`

Ce script produit automatiquement:
- la solution finale pour `explicit`, `semi_implicit` et `splitting`;
- les cartes de stabilite observee;
- la comparaison theorie vs observe (pour `explicit` et `semi_implicit`);
- les comparaisons entre schemas;
- une analyse spectrale lineaire (rayon spectral).

Les sorties sont enregistrees dans:
- `outputs/analysis/schema_analysis/solutions`
- `outputs/analysis/schema_analysis/stability`
- `outputs/analysis/schema_analysis/comparisons`
- `outputs/analysis/schema_analysis/spectral_radius`
- `outputs/analysis/schema_analysis/data`

## Execution

Depuis la racine du projet:

```bash
python experiments/analyse_schemas.py --quick
```

Execution complete (grille plus dense):

```bash
python experiments/analyse_schemas.py
```

