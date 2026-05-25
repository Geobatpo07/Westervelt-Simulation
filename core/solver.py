# core/solver.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Callable
from core.explicite import step_explicit
from core.semi_implicite import step_semi_implicit
from core.numerics import _apply_boundary, _laplacian_all, compute_energy
from utils import timer, profile, append_profiler_record_csv


@dataclass
class WesterveltParams:
    """
    Paramètres physiques et numériques pour le modèle de Westervelt.

    Attributes:
        c: Vitesse du son (m/s).
        rho0: Densité au repos (kg/m^3).
        beta: Coefficient de non-linéarité.
        mu_v: Viscosité volumique (Pa.s).
        B_over_A: Paramètre de non-linéarité (B/A), définit beta si fourni.
        nu: Viscosité cinématique (m^2/s), définit mu_v si fourni.
        dx: Pas d'espace (m).
        dt: Pas de temps (s).
        nx: Nombre de points spatiaux.
        nt: Nombre d'itérations temporelles.
        bc: Type de conditions aux limites ('dirichlet' ou 'neumann').
        scheme: Schéma numérique ('explicit' ou 'semi_implicit').
        k: Coefficient de non-linéarité (calculé).
        b: Coefficient de viscosité (calculé).
    """

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
        """
        Valide les paramètres et calcule les coefficients dérivés.

        Raises:
            ValueError: Si un paramètre est hors domaine valide ou si le schéma/BC est inconnu.
        """
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
        """
        Initialise le solveur avec les paramètres donnés.

        Args:
            params: Objet contenant les paramètres physiques et numériques.
        """
        self.param = params
        self.x = np.linspace(0, self.param.dx * (self.param.nx - 1), self.param.nx)

        self.u_prev = np.zeros(self.param.nx)
        self.u = np.zeros(self.param.nx)
        self.u_next = np.zeros(self.param.nx)
        self.F = np.zeros(self.param.nx)

        self.energy_history = []
        self.check_stability_indicators()


    def _lambda_number(self) -> float:
        """
        Calcule le nombre lambda (paramètre de stabilité de diffusion).

        Le nombre lambda est défini par c^2 * dt / dx^2.

        Returns:
            float: Valeur de c^2 * dt / dx^2.
        """
        return float((self.param.c ** 2) * self.param.dt / (self.param.dx ** 2))


    def explicit_stability_margin(self) -> float:
        """
        Calcule la marge de stabilité théorique pour le schéma explicite.

        La condition de stabilité pour le schéma explicite est liée à la positivité 
        de cette marge.

        Returns:
            float: Valeur de la marge (dx^2 - (c^2 * dt^2 + 2 * b * dt)).
        """
        return self.param.dx ** 2 - (self.param.c ** 2 * self.param.dt ** 2 + 2 * self.param.b * self.param.dt)


    def explicit_theoretical_stable(self) -> bool:
        """
        Vérifie si la condition de stabilité théorique du schéma explicite est remplie.

        Returns:
            bool: True si la marge de stabilité est positive ou nulle.
        """
        return self.explicit_stability_margin() >= 0.0


    def semi_implicit_stability_margin(self, alpha: float = 1.0) -> float:
        """
        Calcule la marge de stabilité théorique pour le schéma semi-implicite.

        Args:
            alpha: Paramètre de pondération du schéma (défaut 1.0).

        Returns:
            float: Valeur de la marge de stabilité.
        """
        return alpha * self.param.dx ** 2 - (self.param.c ** 2 * self.param.dt ** 2 - 2 * self.param.b * self.param.dt)


    def semi_implicit_theoretical_stable(self, alpha: float = 1.0) -> bool:
        """
        Vérifie si la condition de stabilité théorique du schéma semi-implicite est remplie.

        Args:
            alpha: Paramètre de pondération du schéma (défaut 1.0).

        Returns:
            bool: True si le schéma est théoriquement stable.
        """
        return self.semi_implicit_stability_margin(alpha=alpha) >= 0.0


    def check_stability_indicators(self) -> None:
        """
        Calcule et affiche les indicateurs de stabilité (CFL, lambda) et les marges.

        Affiche un diagnostic textuel dans la console basé sur le schéma
        configuré dans les paramètres.
        """
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


    def reset_auxiliary_field(self, u_t0: np.ndarray | None = None) -> None:
        """
        Recalcule le champ auxiliaire F à partir de u et d'une vitesse initiale.

        Le champ auxiliaire F est utilisé dans les schémas numériques pour 
        gérer les termes non-linéaires et de viscosité.
        Il est défini comme : F = (1 - 2ku) * u_t - b * u_xx.

        Args:
            u_t0: Dérivée temporelle initiale u_t(x, 0). Si None, elle est 
                estimée par différence finie arrière.
        """
        if u_t0 is None:
            u_t0 = (self.u - self.u_prev) / self.param.dt

        denom = 1.0 - 2.0 * self.param.k * self.u
        self.F = denom * u_t0 - self.param.b * _laplacian_all(self.u, self.param.dx ** 2)

        _apply_boundary(self.F, self.bc_type)


    def _initial_profile(
        self,
        profile_type: str,
        amplitude: float = 1.0,
        mu: float | None = None,
        sigma: float | None = None
    ) -> np.ndarray:
        """
        Génère un profil spatial initial pour u ou u_t.

        Supporte plusieurs types de profils (Gaussien, uniforme, etc.).

        Args:
            profile_type: Type de profil parmi {'zero', 'gaussian', 
                'gaussian_derivative', 'gaussian_zero_mean', 'uniform'}.
            amplitude: Facteur d'échelle appliqué au profil.
            mu: Centre du profil (pour les types Gaussiens).
            sigma: Largeur du profil (pour les types Gaussiens).

        Returns:
            np.ndarray: Vecteur de taille nx contenant les valeurs du profil.

        Raises:
            ValueError: Si le type de profil n'est pas reconnu.
        """

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

        elif profile_type == "gaussian_zero_mean":
            g = np.exp(-((self.x - mu) ** 2) / (2.0 * sigma ** 2))
            profile = g - np.mean(g)

        elif profile_type == "uniform":
            profile = np.random.uniform(-0.1, 0.1, self.param.nx)

        else:
            raise ValueError(f"Profil initial non reconnu : {profile_type}")

        return amplitude * profile


    def initialize(
        self,
        u0_type: str = "gaussian",
        u1_type: str = "zero",
        A1: float = 1.0,
        A2: float = 0.0,
        mu: float | None = None,
        sigma1: float | None = None,
        sigma2: float | None = None
    ) -> None:
        """
        Initialise les champs u, u_prev et F du solveur.

        Définit l'état initial à t=0 et t=-dt (pour le schéma à deux pas).

        Args:
            u0_type: Type de profil pour u(x, 0).
            u1_type: Type de profil pour la vitesse initiale u_t(x, 0).
            A1: Amplitude pour u0.
            A2: Amplitude pour u1.
            mu: Position centrale commune aux profils.
            sigma1: Écart-type pour le profil u0.
            sigma2: Écart-type pour le profil u1.
        """

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


    def compute_energy(self) -> float:
        """
        Calcule l'énergie totale actuelle de la solution numérique.

        L'énergie est calculée à partir des champs u et u_prev.

        Returns:
            float: Valeur de l'énergie discrète.
        """
        return float(compute_energy(self.u, self.u_prev, self.param.c, self.param.dt, self.param.dx))


    def _evaluate_source(self, source: Any, t: float) -> np.ndarray | None:
        """
        Évalue la fonction source au temps t sur toute la grille spatiale.

        Args:
            source: Soit un tableau (nx,), soit un callable(t, x), soit None.
            t: Temps actuel (s).

        Returns:
            np.ndarray | None: Valeurs de la source sur la grille ou None.
        """
        if source is None:
            return None

        if callable(source):
            return source(t, self.x)

        return source


    def step(self, source: np.ndarray | None = None) -> None:
        """
        Avance la simulation d'un pas de temps dt.

        Met à jour u_next, u, u_prev et le champ auxiliaire F en utilisant 
        le schéma spécifié (explicite ou semi-implicite).

        Args:
            source: Valeurs de la source externe au temps actuel (optionnel).
        """
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

        # self.u_prev, self.u, self.u_next = (
        #    self.u,
        #    self.u_next,
        #    self.u_prev,
        #)

        # self.F, self.F_next = (
        #    self.F_next,
        #    self.F,
        #)

        self.F = F_next.copy()

        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()


    @profile
    def run(self, store_energy: bool = True, source: Any = None) -> None:
        """
        Exécute la simulation temporelle complète.

        Boucle sur nt itérations temporelles.

        Args:
            store_energy: Si True, enregistre l'énergie à chaque pas de temps 
                dans `energy_history`.
            source: Fonction source externe callable(t, x) ou tableau (optionnel).
        """
        if store_energy and len(self.energy_history) == 0:
            self.energy_history.append(self.compute_energy())

        for n in range(self.param.nt):
            t_n = n * self.param.dt
            source_values = self._evaluate_source(source, t_n)
            self.step(source=source_values)
            if store_energy:
                self.energy_history.append(self.compute_energy())


    def plot_solution(self) -> None:
        """
        Affiche le profil spatial de la solution actuelle u(x).
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.u)
        plt.title("Solution finale")
        plt.xlabel("x (m)")
        plt.ylabel("u")
        plt.grid(True)
        plt.show()


    @profile
    def run_with_snapshots(
        self,
        times_to_save: List[float],
        store_energy: bool = True,
        source: Any = None
    ) -> Dict[float, np.ndarray]:
        """
        Exécute la simulation et capture l'état de u à des instants précis.

        Args:
            times_to_save: Liste des temps (s) auxquels sauvegarder u.
            store_energy: Si True, enregistre l'historique d'énergie.
            source: Fonction source (optionnel).

        Returns:
            Dict[float, np.ndarray]: Dictionnaire associant temps et snapshots (u).
        """
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
        dt_values: List[float] | np.ndarray,
        amplitude_values: List[float] | np.ndarray,
        u0_type: str = "gaussian",
        u1_type: str = "zero",
        velocity_amplitude: float = 0.0,
        mu: float | None = None,
        sigma1: float | None = None,
        sigma2: float | None = None,
        blowup_threshold: float = 1e6,
    ) -> List[Dict[str, Any]]:
        """
        Réalise un balayage paramétrique pour analyser la stabilité numérique.

        Teste différentes combinaisons de pas de temps et d'amplitudes initiales 
        pour détecter les zones d'instabilité ou de divergence.

        Args:
            dt_values: Liste des pas de temps à tester.
            amplitude_values: Liste des amplitudes initiales (u0) à tester.
            u0_type: Type de profil pour u0.
            u1_type: Type de profil pour u1.
            velocity_amplitude: Amplitude du profil u1.
            mu: Centre des profils.
            sigma1: Écart-type pour u0.
            sigma2: Écart-type pour u1.
            blowup_threshold: Seuil de valeur de u au-delà duquel on considère 
                qu'il y a divergence.

        Returns:
            List[Dict[str, Any]]: Résultats détaillés pour chaque point de la grille.
        """

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


    def plot_snapshots(self, snapshots: Dict[float, np.ndarray]) -> None:
        """
        Affiche les snapshots temporels de l'onde.

        Args:
            snapshots: Dictionnaire {temps: profil_u}.
        """
        plt.figure(figsize=(10, 6))
        for t in sorted(snapshots.keys()):
            plt.plot(self.x, snapshots[t], label=f"t = {t * 1e6:.2f} us")
        plt.xlabel("x (m)")
        plt.ylabel("u(x,t)")
        plt.title("Evolution de l'onde de Westervelt")
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_energy(self) -> None:
        """
        Affiche l'évolution temporelle de l'énergie totale stockée.
        """
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


    def print_profiler_summary(self) -> None:
        """
        Affiche un résumé statistique des performances (profilage) dans la console.

        Affiche le nombre d'appels, les temps total, moyen et maximum, 
        ainsi que le pic de mémoire pour chaque fonction décorée par `@profile`.
        """
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


    def save_profile_data(
        self,
        path: Path | str,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> None:
        """
        Sauvegarde les données de profilage dans un fichier CSV.

        Args:
            path: Chemin vers le fichier CSV de destination.
            extra_metadata: Métadonnées additionnelles à inclure dans chaque ligne.
        """
        if not hasattr(self, "profiler") or not self.profiler:
            print("Aucune donnée de profiling à sauvegarder.")
            return
        
        extra_metadata = extra_metadata or {}
        
        for func_name, records in self.profiler.items():
            durations = records["durations"]
            memories = records["peak_memory_mb"]
            
            if not durations: 
                continue
                
            record = {
                "function": func_name,
                "scheme": self.param.scheme,
                "bc": self.param.bc,
                "nx": self.param.nx,
                "nt": self.param.nt,
                "dx": self.param.dx,
                "dt": self.param.dt,
                "T_final": self.param.nt * self.param.dt,
                "c": self.param.c,
                "rho0": self.param.rho0,
                "beta": self.param.beta,
                "mu_v": self.param.mu_v,
                "k": self.param.k,
                "b": self.param.b,
                "n_calls": len(durations),
                "time_total_s": float(sum(durations)),
                "time_mean_s": float(sum(durations)) / len(durations),
                "time_max_s": float(max(durations)),
                "memory_max_mb": float(max(memories)),
                **extra_metadata,
            }
            
            append_profiler_record_csv(path=path, record=record)
            
        print(f"Données de profiling enregistrées dans : {path}")    
