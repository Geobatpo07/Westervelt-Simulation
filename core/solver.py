# core/solver.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from core.explicite import step_explicit
from core.semi_implicite import step_semi_implicit
from core.numerics import _apply_boundary, _laplacian_all, compute_energy
from utils.utils import timer, profile


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

        self.bc = self.bc.lower()
        if self.bc not in ("dirichlet", "neumann"):
            raise ValueError("bc doit valoir 'dirichlet' ou 'neumann'.")


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


    def explicit_stability_margin(self):
        """Indication de marge de stabilité théorique pour le schéma explicite."""
        return self.param.dx ** 2 - (self.param.c ** 2 * self.param.dt ** 2 + 2 * self.param.b * self.param.dt)


    def explicit_theoretical_stable(self):
        """Indication de stabilité théorique pour le schéma explicite."""
        return self.explicit_stability_margin() >= 0.0


    def semi_implicit_stability_margin(self, alpha=1.0):
        """Indication de marge de stabilité théorique pour le schéma semi-implicite."""
        return alpha * self.param.dx ** 2 - (self.param.c ** 2 * self.param.dt ** 2 - 2 * self.param.b * self.param.dt)


    def semi_implicit_theoretical_stable(self, alpha=1.0):
        """Indication de stabilité théorique pour le schéma semi-implicite."""
        return self.semi_implicit_stability_margin(alpha=alpha) >= 0.0


    def check_stability_indicators(self):
        """Affiche les indicateurs de stabilité (CFL, lambda) et les marges de stabilité pour le schéma choisi."""
        cfl = self.param.c * self.param.dt / self.param.dx
        lam_legacy = self._lambda_number()

        print(f"Indicateurs: CFL={cfl:.6g}, lambda={lam_legacy:.6g} (legacy)")

        if self.param.scheme == "explicit":
            margin = self.explicit_stability_margin()
            print(f"Marge de stabilité explicite: {margin:.6g}")
            if margin >= 0.0:
                print("Stable (marge positive)")
            else:
                print("Non stable.")
        elif self.param.scheme == "semi_implicit":
            margin = self.semi_implicit_stability_margin()
            print(f"Marge de stabilité semi-implicite: {margin:.6g}")
            if margin >= 0.0:
                print("Stable (marge positive)")
            else:
                print("Non stable.")
        else:
            print("Schéma inconnu pour l'analyse de stabilité.")


    def reset_auxiliary_field(self, u_t0=None):
        """
        Recalcule F à partir de u et d'une vitesse initiale u_t0.

        F = (1 - 2ku) u_t - b u_xx.

        Si u_t0=None, on approxime u_t par (u - u_prev)/dt.
        """
        if u_t0 is None:
            u_t0 = (self.u - self.u_prev) / self.param.dt

        denom = 1.0 - 2.0 * self.param.k * self.u
        self.F = denom * u_t0 - self.param.b * _laplacian_all(self.u, self.param.dx ** 2)

        _apply_boundary(self.F, self.bc_type)


    def _initial_profile(self, profile_type, amplitude=1.0, mu=None, sigma=None):
        """Construit un profil initial u0 ou u1."""

        if mu is None:
            mu = self.x.max() / 4.0

        if sigma is None:
            sigma = max(self.x.max() / 20.0, 1e-12)

        if profile_type == "zero":
            profile = np.zeros_like(self.x)

        elif profile_type == "gaussian":
            profile = np.exp(-((self.x - mu) ** 2) / (2.0 * sigma ** 2))

        elif profile_type == "gaussian_derivative":
            profile = (self.x - mu) * np.exp(-((self.x - mu) ** 2) / (2.0 * sigma ** 2))

        elif profile_type == "uniform":
            profile = np.random.uniform(-0.1, 0.1, self.param.nx)

        else:
            raise ValueError(f"Profil initial non reconnu : {profile_type}")

        return amplitude * profile


    def initialize(self, u0_type="gaussian", u1_type="zero", A1=1.0, A2=0.0, mu=None, sigma1=None, sigma2=None):
        """Initialise u^0 = u0 et u_t^0 = u1."""

        u0 = self._initial_profile(
            profile_type=u0_type,
            amplitude=A1,
            mu=mu,
            sigma=sigma1,
        )

        u1 = self._initial_profile(
            profile_type=u1_type,
            amplitude=A2,
            mu=mu,
            sigma=sigma2,
        )

        self.u = u0.copy()
        _apply_boundary(self.u, self.bc_type)

        _apply_boundary(u1, self.bc_type)

        self.u_prev = self.u - self.param.dt * u1
        _apply_boundary(self.u_prev, self.bc_type)

        self.reset_auxiliary_field(u_t0=u1)

        self.energy_history = [self.compute_energy()]


    @property
    def bc_type(self):
        """Retourne 0 pour Dirichlet et 1 pour Neumann, utilisé dans les fonctions de mise à jour."""
        return 0 if self.param.bc == "dirichlet" else 1


    def compute_energy(self):
        """Calcule l'energie totale de la solution."""
        return float(compute_energy(self.u, self.u_prev, self.param.c, self.param.dt, self.param.dx))


    def _evaluate_source(self, source, t):
        if source is None:
            return None

        if callable(source):
            return source(self.x, t)

        return source

    def step(self, source=None):
        """Effectue une étape de temps selon le schéma choisi."""
        if self.param.scheme == "semi_implicit":
            self.u_next, F_next = step_semi_implicit(
                self.u,
                self.F,
                self.param.c,
                self.param.b,
                self.param.k,
                self.param.dt,
                self.param.dx,
                self.bc_type,
                source=source,
            )
        else:
            self.u_next, F_next = step_explicit(
                self.u,
                self.F,
                self.param.c,
                self.param.b,
                self.param.k,
                self.param.dt,
                self.param.dx,
                self.bc_type,
                source=source,
            )

        self.F = F_next.copy()

        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()


    @profile
    def run(self, store_energy=True, source=None):
        """Fait tourner la simulation pour le nombre de pas de temps spécifié dans les paramètres.
        Si store_energy=True, stocke l'énergie à chaque pas de temps dans self.energy_history."""
        if store_energy and len(self.energy_history) == 0:
            self.energy_history.append(self.compute_energy())

        for n in range(self.param.nt):
            t_n = n * self.param.dt
            source_values = self._evaluate_source(source, t_n)
            self.step(source=source_values)
            if store_energy:
                self.energy_history.append(self.compute_energy())


    def plot_solution(self):
        """Affiche la solution finale."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.u)
        plt.title("Solution finale")
        plt.xlabel("x (m)")
        plt.ylabel("u")
        plt.grid(True)
        plt.show()


    @profile
    def run_with_snapshots(self, times_to_save, store_energy=True, source=None):
        """Fait tourner la simulation et sauvegarde des snapshots de u à des temps spécifiés dans times_to_save (en secondes).
         Renvoie un dictionnaire {t: u_snapshot} pour les temps demandés. Si store_energy=True, stocke aussi l'énergie à chaque pas de temps dans self.energy_history."""
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
            t_n = n * dt
            if n in indices_to_save:
                snapshots[indices_to_save[n]] = self.u.copy()
            if n < nt:
                source_values = self._evaluate_source(source, t_n)
                self.step(source=source_values)
                if store_energy:
                    self.energy_history.append(self.compute_energy())

        return snapshots


    @timer
    def run_stability_scan(
            self,
            dt_values,
            amplitude_values,
            u0_type="gaussian",
            u1_type="zero",
            velocity_amplitude=0.0,
            mu=None,
            sigma1=None,
            sigma2=None,
            blowup_threshold=1e6,
    ):
        """Balaye (dt, amplitude de u0) et renvoie un diagnostic de stabilité."""

        results = []
        tol = 1e-12

        for dt in dt_values:
            for amp in amplitude_values:
                min_denom = np.inf
                max_abs = 0.0
                stable = True

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

                test_solver.initialize(
                    u0_type=u0_type,
                    u1_type=u1_type,
                    A1=float(amp),
                    A2=float(velocity_amplitude),
                    mu=mu,
                    sigma1=sigma1,
                    sigma2=sigma2,
                )

                if test_solver.param.scheme == "explicit":
                    stability_margin = test_solver.explicit_stability_margin()
                    theoretical_stable = test_solver.explicit_theoretical_stable()
                elif test_solver.param.scheme == "semi_implicit":
                    stability_margin = test_solver.semi_implicit_stability_margin()
                    theoretical_stable = test_solver.semi_implicit_theoretical_stable()
                else:
                    stability_margin = np.nan
                    theoretical_stable = False

                for _ in range(test_params.nt):
                    denom = 1.0 - 2.0 * test_solver.param.k * test_solver.u
                    cur_min = float(np.min(denom))
                    min_denom = min(min_denom, cur_min)

                    if cur_min <= tol:
                        stable = False
                        break

                    test_solver.step()

                    cur = float(np.max(np.abs(test_solver.u)))

                    if not np.isfinite(cur) or cur > blowup_threshold:
                        stable = False
                        max_abs = cur
                        break

                    max_abs = max(max_abs, cur)

                results.append(
                    {
                        "dt": float(dt),
                        "amplitude": float(amp),
                        "amplitude_u0": float(amp),
                        "amplitude_u1": float(velocity_amplitude),

                        "u0_type": u0_type,
                        "u1_type": u1_type,

                        "stable": bool(stable),
                        "max_abs_u": float(max_abs),

                        "cfl": float(test_params.c * test_params.dt / test_params.dx),
                        "lambda_legacy": float(test_solver._lambda_number()),

                        "min_denom": float(min_denom),
                        "nondegenerate": bool(min_denom > tol),

                        "stability_margin": float(stability_margin),
                        "theoretical_stable": bool(theoretical_stable),
                    }
                )

        return results


    def plot_snapshots(self, snapshots):
        """Affiche les snapshots de la solution u(x) à différents temps."""
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


    def print_profiler_summary(self):
        """Affiche un résumé des temps et pics mémoire mesurés."""

        if not hasattr(self, "profiler") or not self.profiler:
            print("Aucune donnée de profiling disponible.")
            return

        print("\nRésumé du profiling")
        print("-" * 60)

        for func_name, records in self.profiler.items():
            durations = records["durations"]
            memories = records["peak_memory_mb"]

            n_calls = len(durations)

            print(f"{func_name}")
            print(f"  appels          : {n_calls}")
            print(f"  temps total     : {sum(durations):.4f} s")
            print(f"  temps moyen     : {sum(durations) / len(durations):.4f} s")
            print(f"  temps max       : {max(durations):.4f} s")
            print(f"  mémoire max     : {max(memories):.4f} MB")
            print()

