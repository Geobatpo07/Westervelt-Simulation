# Westervelt-Simulation

Simulation numérique de l'équation de Westervelt pour la propagation d'ondes acoustiques non linéaires.

## Équation de Westervelt

L'équation modélisée est :
$$u_{tt} - c^2 \Delta u - \nu \epsilon \Delta u_t = \alpha \epsilon u u_{tt}$$
avec $\alpha = \frac{\gamma + 1}{c^2}$.

## Structure du projet

- `src/` : Code source du solver (explicite et splitting).
- `notebooks/` : Notebooks de démonstration et de test.
- `tests/` : Tests unitaires et validation du modèle.

## Utilisation

Consultez `notebooks/prresentation_01.ipynb` pour une présentation générale et `notebooks/test_solver.ipynb` pour des exemples d'utilisation du solver.
