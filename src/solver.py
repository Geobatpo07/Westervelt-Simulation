# src/solver.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from explicite import step_explicit

@dataclass
class WesterveltParams:
    """Classe pour enregistrer les paramètres du modèle de Westervelt."""
    c: float
    epsilon: float = 0.0
    nu: float = 0.0
    gamma: float = 0.0

    dx: float = 0.01
    dt: float = 0.001
    nx: int = 200
    nt: int = 1000
    nonlinear: bool = False
    bc: str = 'dirichlet'

    a: float = field(init=False)
    beta: float = field(init=False)

    def __post_init__(self):
        # a est lié à la dissipation
        self.a = self.nu * self.epsilon
        self.alpha = (self.gamma + 1) / (self.c**2)
        self.beta = self.alpha * self.epsilon

class WesterveltSolver:
    """Classe pour implémenter le solver du modèle de Westervelt."""
    def __init__(self, params: WesterveltParams):
        self.param = params
        self.x = np.linspace(0, self.param.dx*(self.param.nx-1), self.param.nx)
        self.u_prev = np.zeros(self.param.nx)
        self.u = np.zeros(self.param.nx)
        self.u_next = np.zeros(self.param.nx)
        self.check_cfl()

    def check_cfl(self):
        cfl = self.param.c*self.param.dt/self.param.dx
        print(f'CFL condition: {cfl:4f}')
        if cfl >= 1:
            print('Attention: La condition CFL est proche ou supérieure à 1. Risque d\'instabilité.')

    def initialize(self, ic_type='gaussian'):
        if ic_type == 'gaussian':
            self.u = np.exp(-((self.x - self.x.max()/4)/(self.x.max()/20))**2)
        elif ic_type == 'uniform':
            self.u = np.random.uniform(-0.1, 0.1, self.param.nx)
        else:
            raise ValueError('Type d\'initialisation non reconnu.')

        self.u_prev = self.u.copy()

    def step(self):
        bc_type = 0 if self.param.bc == 'dirichlet' else 1
        self.u_next = step_explicit(
            self.u, self.u_prev, self.param.c, self.param.a, self.param.beta,
            self.param.dt, self.param.dx, self.param.nonlinear, bc_type
        )
        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()

    def run(self):
        for _ in range(self.param.nt):
            self.step()

    def plot_solution(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.u)
        plt.title('Solution finale')
        plt.xlabel('x (m)')
        plt.ylabel('u')
        plt.grid(True)
        plt.show()

    def run_with_snapshots(self, times_to_save):
        if times_to_save is None:
            times_to_save = []

        dt = float(self.param.dt)
        nt = int(self.param.nt)

        indices_to_save = {}
        for t in times_to_save:
            n = int(round(float(t) / dt))
            if 0 <= n <= nt:
                indices_to_save[n] = t

        snapshots = {}
        for n in range(nt + 1):
            if n in indices_to_save:
                snapshots[indices_to_save[n]] = self.u.copy()
            if n < nt:
                self.step()

        return snapshots

    def plot_snapshots(self, snapshots):
        plt.figure(figsize=(10, 6))
        for t in sorted(snapshots.keys()):
            plt.plot(self.x, snapshots[t], label=f"t = {t*1e6:.2f} µs")
        plt.xlabel("x (m)")
        plt.ylabel("u(x,t)")
        plt.title("Évolution de l'onde de Westervelt")
        plt.legend()
        plt.grid(True)
        plt.show()

