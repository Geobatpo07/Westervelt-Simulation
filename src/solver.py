# src/solver.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from src.explicite import step_explicit
from src.semi_implicite import step_semi_implicit
from src.numerics import _apply_boundary, _laplacian_all, compute_energy


@dataclass
class WesterveltParams:
    """Paramètres du modèle de Westervelt."""

    c: float
    rho0: float = 1000.0
    beta: float = 3.5
    mu_v: float = 0.0

    # Parametres legacy optionnels (compatibilite)
    B_over_A: float | None = None
    nu: float | None = None

    dx: float = 0.01
    dt: float = 0.001
    nx: int = 200
    nt: int = 1000

    bc: str = "dirichlet"
    scheme: str = "explicit"  # "explicit" ou "semi_implicit"

    k: float = field(init=False)
    b: float = field(init=False)

    def __post_init__(self):
        if self.c <= 0.0:
            raise ValueError("c doit etre strictement positif.")
        if self.rho0 <= 0.0:
            raise ValueError("rho0 doit etre strictement positif.")

        # Compatibilite: si B/A est fourni, il definit beta.
        if self.B_over_A is not None:
            self.beta = 1.0 + 0.5 * self.B_over_A
        if self.beta <= 0.0:
            raise ValueError("beta doit etre strictement positif.")

        # Compatibilite: nu (viscosite cinematique) => mu_v = rho0 * nu.
        if self.nu is not None and self.mu_v == 0.0:
            self.mu_v = self.rho0 * self.nu
        if self.mu_v < 0.0:
            raise ValueError("mu_v doit etre positif ou nul.")

        # Coefficients physiques du modele.
        self.b = self.mu_v / self.rho0
        self.k = self.beta / (self.rho0 * self.c ** 2)

        self.scheme = self.scheme.lower()
        if self.scheme not in ("explicit", "semi_implicit"):
            raise ValueError("scheme doit valoir 'explicit' ou 'semi_implicit'.")


class WesterveltSolver:
    """Solver 1D pour l'équation de Westervelt (explicite ou semi-implicite)."""

    def __init__(self, params: WesterveltParams):
        self.param = params
        self.x = np.linspace(0, self.param.dx * (self.param.nx - 1), self.param.nx)

        self.u_prev = np.zeros(self.param.nx)
        self.u = np.zeros(self.param.nx)
        self.u_next = np.zeros(self.param.nx)
        self.F = np.zeros(self.param.nx)

        self.energy_history = []
        self.check_stability_indicators()

    def _lambda_number(self):
        """Nombre lambda = c^2 dt / dx^2."""
        return float((self.param.c ** 2) * self.param.dt / (self.param.dx ** 2))

    def _is_lambda_stable(self):
        """Critère de stabilité théorique basé sur lambda."""
        lam = self._lambda_number()
        if self.param.scheme == "explicit":
            return lam <= 0.5
        return True

    def check_stability_indicators(self):
        """Affiche les indicateurs de stabilité (lambda décisionnel, CFL informatif)."""
        cfl = self.param.c * self.param.dt / self.param.dx
        lam = self._lambda_number()
        print(f"Indicateurs: lambda={lam:.6g}, cfl={cfl:.6g}")
        if self.param.scheme == "explicit" and lam > 0.5:
            print("Attention: lambda > 0.5 en explicite (risque d'instabilité théorique).")

    def check_cfl(self):
        """Compatibilité: ancien nom, redirige vers check_stability_indicators."""
        self.check_stability_indicators()

    def initialize(self, ic_type="gaussian"):
        if ic_type == "gaussian":
            center = self.x.max() / 4.0
            width = max(self.x.max() / 20.0, 1e-12)
            self.u = np.exp(-((self.x - center) / width) ** 2)
        elif ic_type == "uniform":
            self.u = np.random.uniform(-0.1, 0.1, self.param.nx)
        else:
            raise ValueError("Type d'initialisation non reconnu.")

        self.u_prev = self.u.copy()
        # F^0 cohérent avec F=(1-2ku)u_t - b u_xx et u_t^0=0.
        self.F = -self.param.b * _laplacian_all(self.u, self.param.dx * self.param.dx)
        _apply_boundary(self.F, self.bc_type)
        self.energy_history = [self.compute_energy()]

    @property
    def bc_type(self):
        return 0 if self.param.bc == "dirichlet" else 1

    def compute_energy(self):
        return float(compute_energy(self.u, self.u_prev, self.param.c, self.param.dt, self.param.dx))

    def step(self):
        if self.param.scheme == "semi_implicit":
            self.u_next, F_next = step_semi_implicit(
                self.u,
                self.u_prev,
                self.F,
                self.param.c,
                self.param.b,
                self.param.k,
                self.param.dt,
                self.param.dx,
                self.bc_type,
            )
        else:
            self.u_next, F_next = step_explicit(
                self.u,
                self.u_prev,
                self.F,
                self.param.c,
                self.param.b,
                self.param.k,
                self.param.dt,
                self.param.dx,
                self.bc_type,
            )

        self.F = F_next.copy()

        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()

    def run(self, store_energy=True):
        if store_energy and len(self.energy_history) == 0:
            self.energy_history.append(self.compute_energy())

        for _ in range(self.param.nt):
            self.step()
            if store_energy:
                self.energy_history.append(self.compute_energy())

    def plot_solution(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.u)
        plt.title("Solution finale")
        plt.xlabel("x (m)")
        plt.ylabel("u")
        plt.grid(True)
        plt.show()

    def run_with_snapshots(self, times_to_save, store_energy=True):
        if times_to_save is None:
            times_to_save = []

        dt = float(self.param.dt)
        nt = int(self.param.nt)

        indices_to_save = {}
        for t in times_to_save:
            n = int(round(float(t) / dt))
            if 0 <= n <= nt:
                indices_to_save[n] = t

        if store_energy:
            self.energy_history = [self.compute_energy()]

        snapshots = {}
        for n in range(nt + 1):
            if n in indices_to_save:
                snapshots[indices_to_save[n]] = self.u.copy()
            if n < nt:
                self.step()
                if store_energy:
                    self.energy_history.append(self.compute_energy())

        return snapshots

    def run_stability_scan(self, dt_values, amplitude_values, ic_type="gaussian", blowup_threshold=1e6):
        """Balaye (dt, amplitude initiale) et renvoie un diagnostic de stabilité."""
        results = []

        for dt in dt_values:
            for amp in amplitude_values:
                test_params = WesterveltParams(
                    c=self.param.c,
                    rho0=self.param.rho0,
                    beta=self.param.beta,
                    mu_v=self.param.mu_v,
                    dx=self.param.dx,
                    dt=float(dt),
                    nx=self.param.nx,
                    nt=self.param.nt,
                    bc=self.param.bc,
                    scheme=self.param.scheme,
                )
                test_solver = WesterveltSolver(test_params)
                test_solver.initialize(ic_type=ic_type)
                test_solver.u *= float(amp)
                test_solver.u_prev = test_solver.u.copy()

                stable = True
                max_abs = 0.0
                for _ in range(test_params.nt):
                    test_solver.step()
                    cur = float(np.max(np.abs(test_solver.u)))
                    if not np.isfinite(cur) or cur > blowup_threshold:
                        stable = False
                        max_abs = cur
                        break
                    if cur > max_abs:
                        max_abs = cur

                results.append(
                    {
                        "dt": float(dt),
                        "amplitude": float(amp),
                        "stable": stable,
                        "lambda": float(test_solver._lambda_number()),
                        "lambda_stable": bool(test_solver._is_lambda_stable()),
                        "max_abs_u": float(max_abs),
                        "cfl": float(test_params.c * test_params.dt / test_params.dx),
                    }
                )

        return results

    def plot_snapshots(self, snapshots):
        plt.figure(figsize=(10, 6))
        for t in sorted(snapshots.keys()):
            plt.plot(self.x, snapshots[t], label=f"t = {t * 1e6:.2f} us")
        plt.xlabel("x (m)")
        plt.ylabel("u(x,t)")
        plt.title("Evolution de l'onde de Westervelt")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_energy(self):
        if not self.energy_history:
            print("Aucune energie stockee. Lancez run(..., store_energy=True).")
            return

        t = np.arange(len(self.energy_history)) * self.param.dt
        plt.figure(figsize=(10, 4))
        plt.plot(t, self.energy_history)
        plt.xlabel("t (s)")
        plt.ylabel("Energie discrete")
        plt.title("Evolution de l'energie")
        plt.grid(True)
        plt.show()
