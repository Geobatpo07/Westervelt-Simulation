# Westervelt-Simulation

Simulation numerique de l'equation de Westervelt pour la propagation d'ondes acoustiques non lineaires.

## Equation de Westervelt

L'equation modelisee est :
$$(1-2ku)u_{tt} - c^2 u_{xx} - b u_{xxt} = 2k(u_t)^2$$.

## Schemas implementes

Le projet utilise la variable auxiliaire $F$ et deux modes de simulation :

$$
\begin{cases}
F_i^{n+1}=F_i^n+\Delta t\,c^2\delta_{xx}u_i^n,\\[0.4em]
u_i^{n+1}=u_i^n+\Delta t\,\dfrac{F_i^{n+1}+b\,\delta_{xx}u_i^n}{1-2ku_i^n}.
\end{cases}
$$

$$
\begin{cases}
F_i^{n+1}=F_i^n+\Delta t\,c^2\delta_{xx}u_i^n,\\[0.4em]
u_i^{n+1}=u_i^n+\Delta t\,\dfrac{F_i^{n+1}+b\,\delta_{xx}u_i^{n+1}}{1-2ku_i^n}.
\end{cases}
$$

- `scheme="explicit"` : premier schema (entierement explicite en `u`).
- `scheme="semi_implicit"` : second schema (terme diffusif implicite en `u^{n+1}`).

## Stabilite numerique

Dans ce projet, le critere de stabilite decisionnel est base sur
$$\lambda = \frac{c^2\,\Delta t}{\Delta x^2}.$$

- `explicit` : stabilite conditionnelle (reference: `lambda <= 0.5`).
- `semi_implicit` : stabilite inconditionnelle (au sens du critere lambda).

Le nombre CFL est conserve comme indicateur physique utile au diagnostic,
mais n'est pas le critere decisionnel principal ici.

## Structure du projet

- `src/explicite.py` : pas de temps du schema explicite.
- `src/semi_implicite.py` : pas de temps du schema semi-implicite.
- `src/numerics.py` : operateurs discrets, mise a jour de `F`, systeme tri-diagonal.
- `src/solver.py` : solver unifie (`scheme="explicit"` ou `scheme="semi_implicit"`).
- `src/analysis.py` : outils d'analyse (snapshots, cartes de stabilite).
- `experiments/westervelt_sim.py` : script d'exemple complet.

## Analyse de Fourier du schema explicite

Le script `experiments/analyse_fourier_explicite.py` genere trois figures:

- module du facteur d'amplification et dispersion numerique,
- relation amplification vs dispersion,
- effet de la diffusion sur l'atténuation des hautes frequences.

Execution:

```bash
python experiments/analyse_fourier_explicite.py
```

## Utilisation rapide

```python
from core.solver import WesterveltSolver, WesterveltParams

params = WesterveltParams(
    c=1500,
    rho0=1000,
    beta=4.8,
    mu_v=6e-6,
    dx=2.8e-5,
    dt=1.8e-8,
    nx=2000,
    nt=1200,
    scheme="semi_implicit",
)
solver = WesterveltSolver(params)
solver.initialize(u0_type="gaussian", u1_type="zero")

snapshots = solver.run_with_snapshots([0, 1e-6, 2e-6, 4e-6], store_energy=True)
solver.plot_snapshots(snapshots)
solver.plot_energy()
```

Consultez aussi `notebooks/presentation_01.ipynb` et `notebooks/test_solver.ipynb`.
