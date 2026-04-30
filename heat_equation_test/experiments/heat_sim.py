import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import time

# Ajouter le répertoire core au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from heat_solver import solve_heat_explicit, solve_heat_implicit

def run_heat_experiment():
    L = 1.0        # Longueur du domaine
    nx = 100       # Nombre de points d'espace
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    alpha = 0.01   # Diffusivité
    dt = 0.001     # Pas de temps
    nt = 500       # Nombre de pas de temps

    r = alpha * dt / dx**2
    print(f"Stabilité (r = alpha * dt / dx^2): {r:.4f}")

    u0 = np.exp(-100 * (x - 0.5)**2)

    print("Exécution du solver explicite...")
    start = time.time()
    u_expl = solve_heat_explicit(u0.copy(), alpha, dx, dt, nt)
    print(f"Temps Explicite: {time.time() - start:.4f}s")

    print("Exécution du solver implicite...")
    start = time.time()
    u_impl = solve_heat_implicit(u0.copy(), alpha, dx, dt, nt)
    print(f"Temps Implicite: {time.time() - start:.4f}s")

    plt.figure(figsize=(10, 6))
    plt.plot(x, u0, '--', label='Initial')
    plt.plot(x, u_expl, label='Explicite')
    plt.plot(x, u_impl, ':', label='Implicite')
    plt.title("Comparaison des solveurs de chaleur")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_heat_experiment()
