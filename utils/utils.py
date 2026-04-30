# /utils/utils.py

"""
Fonctions utilitaires pour le projet Westervelt Simulation.

Contient:
- Gestion des fichiers et contrôle des versions
- Traitement de grilles de scans
- Sauvegarde de figures avec versioning
- Validation numérique et stabilité
- Analyse des schémas numériques
"""
import tracemalloc
from pathlib import Path
from datetime import datetime
import json
import re
import warnings
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Callable, Optional, Literal
import time
import functools


# =============================================================================
# Décorateurs
# =============================================================================

def timer(func: Callable) -> Callable:
    """
    Décorateur qui mesure et affiche le temps d'exécution d'une fonction.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start_time
            print(f"[{func.__name__}] Exécuté en {duration:.4f} secondes")
    return wrapper


def log_execution(func: Callable) -> Callable:
    """
    Décorateur qui logue l'appel de la fonction avec ses arguments.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # On évite de logger de gros arrays numpy
        def _repr_short(x):
            if isinstance(x, np.ndarray):
                return f"ndarray(shape={x.shape}, dtype={x.dtype})"
            return repr(x)

        args_repr = [_repr_short(a) for a in args]
        kwargs_repr = [f"{k}={_repr_short(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Appel de {func.__name__}({signature})")
        return func(*args, **kwargs)
    return wrapper


def deprecated(reason: str) -> Callable:
    """
    Décorateur pour marquer les fonctions comme obsolètes.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} est obsolète : {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_shape(func: Callable) -> Callable:
    """
    Décorateur pour valider que les arguments numpy ont des formes compatibles.
    Vérifie que tous les arrays ont la même forme ou sont scalaires.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        array_shapes = [arg.shape for arg in args if isinstance(arg, np.ndarray)]
        if array_shapes:
            first_shape = array_shapes[0]
            for shape in array_shapes[1:]:
                if shape != first_shape:
                    raise ValueError(f"Tous les arrays doivent avoir la même forme. Formes trouvées: {array_shapes}")
        return func(*args, **kwargs)
    return wrapper


def profile(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self_obj = args[0] if len(args) > 0 else None

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start_time
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_memory_mb = peak_memory / (1024 ** 2)

            print(
                f"[PROFILE] {func.__name__} | "
                f"temps: {duration:.4f}s | "
                f"pic mémoire: {peak_memory_mb:.2f} MB"
            )

            if self_obj is not None:
                if not hasattr(self_obj, "profiler"):
                    self_obj.profiler = {}

                if func.__name__ not in self_obj.profiler:
                    self_obj.profiler[func.__name__] = {
                        "durations": [],
                        "peak_memory_mb": []
                    }

                self_obj.profiler[func.__name__]["durations"].append(duration)
                self_obj.profiler[func.__name__]["peak_memory_mb"].append(peak_memory_mb)

        return result

    return wrapper


# =============================================================================
# Gestion des Fichiers et Versioning
# =============================================================================

def ensure_output_dir(base_dir: str = "outputs") -> Path:
    """
    Crée le répertoire de sortie s'il n'existe pas.
    
    Args:
        base_dir: Chemin du répertoire de base
        
    Returns:
        Path: Chemin du répertoire créé
    """
    output_path = Path(base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_next_version(filepath: Path) -> int:
    """
    Retourne le numéro de version suivant pour un fichier.
    
    Cherche les fichiers existants avec le pattern: filename_v1.ext, filename_v2.ext, etc.
    
    Args:
        filepath: Chemin du fichier (sans version)
        
    Returns:
        int: Numéro de version suivant
    """
    filepath = Path(filepath)
    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    
    existing_versions = []

    def _extract_version(path_obj: Path) -> Optional[int]:
        # Accepte: <stem>_v3.ext et <stem>_v3_metadata.json
        match = re.search(r"_v(\d+)(?:_metadata)?$", path_obj.stem)
        if not match:
            return None
        return int(match.group(1))

    for f in parent.glob(f"{stem}_v*{suffix}"):
        version = _extract_version(f)
        if version is not None:
            existing_versions.append(version)

    # Fallback robuste: si on passe un suffixe temporaire (.tmp), scanner tous les formats.
    if not existing_versions:
        for f in parent.glob(f"{stem}_v*.*"):
            version = _extract_version(f)
            if version is not None:
                existing_versions.append(version)
    
    return max(existing_versions) + 1 if existing_versions else 1


def save_figure_with_version(
    fig: plt.Figure,
    filename: str,
    output_dir: str = "outputs",
    formats: List[str] = None,
    dpi: int = 150,
    tight_layout: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """
    Enregistre une figure matplotlib avec contrôle des versions automatique.
    
    La fonction gère automatiquement la numérotation des versions pour éviter
    les conflits de fichiers. Supporte plusieurs formats d'export.
    
    Args:
        fig: Figure matplotlib à enregistrer
        filename: Nom du fichier (sans extension)
        output_dir: Répertoire de sortie
        formats: Liste des formats à enregistrer (default: ['png', 'pdf'])
        dpi: Résolution de la figure
        tight_layout: Appliquer tight_layout avant sauvegarde
        metadata: Dictionnaire de métadonnées à enregistrer en JSON
        
    Returns:
        Dict[str, Path]: Dictionnaire {format: chemin_fichier} des fichiers créés
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> paths = save_figure_with_version(
        ...     fig, 
        ...     "my_plot", 
        ...     metadata={"model": "Westervelt", "scheme": "semi-implicit"}
        ... )
    """
    if formats is None:
        formats = ["png", "pdf"]
    
    # Créer le répertoire s'il n'existe pas
    output_path = ensure_output_dir(output_dir)
    
    # Obtenir le numéro de version
    temp_file = output_path / f"{filename}.tmp"
    version = get_next_version(temp_file)
    
    # Appliquer tight_layout si demandé
    if tight_layout:
        fig.tight_layout()
    
    # Enregistrer dans les différents formats
    saved_paths = {}
    for fmt in formats:
        versioned_filename = f"{filename}_v{version}.{fmt}"
        filepath = output_path / versioned_filename
        fig.savefig(filepath, dpi=dpi, format=fmt, bbox_inches='tight')
        saved_paths[fmt] = filepath
        print(f"Figure enregistrée: {filepath}")
    
    # Enregistrer les métadonnées si fournies
    if metadata is not None:
        metadata_filename = f"{filename}_v{version}_metadata.json"
        metadata_path = output_path / metadata_filename
        metadata_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
            **metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_timestamp, f, indent=2)
        saved_paths["metadata"] = metadata_path
        print(f"Métadonnées enregistrées: {metadata_path}")
    
    return saved_paths


def save_data_with_version(
    data: Any,
    filename: str,
    output_dir: str = "outputs",
    fmt: str = "npy",
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Path, Optional[Path]]:
    """
    Enregistre des données NumPy/Python avec contrôle des versions.
    
    Args:
        data: Données à enregistrer (array NumPy ou dict pour JSON)
        filename: Nom du fichier (sans extension)
        output_dir: Répertoire de sortie
        fmt: Format ('npy' pour NumPy, 'npz' pour archive, 'json' pour JSON)
        metadata: Métadonnées optionnelles à enregistrer
        
    Returns:
        Tuple[Path, Optional[Path]]: (chemin_données, chemin_métadonnées)
    """
    output_path = ensure_output_dir(output_dir)
    
    # Obtenir la version
    temp_file = output_path / f"{filename}.tmp"
    version = get_next_version(temp_file)
    
    # Enregistrer les données
    versioned_filename = f"{filename}_v{version}.{fmt}"
    filepath = output_path / versioned_filename
    
    if fmt == "npy":
        np.save(filepath, data)
    elif fmt == "npz":
        np.savez(filepath, data=data)
    elif fmt == "json":
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Format non supporté: {fmt}")
    
    print(f"Données enregistrées: {filepath}")
    
    # Enregistrer les métadonnées
    metadata_path = None
    if metadata is not None:
        metadata_filename = f"{filename}_v{version}_metadata.json"
        metadata_path = output_path / metadata_filename
        metadata_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "format": fmt,
            **metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_timestamp, f, indent=2)
        print(f"Métadonnées enregistrées: {metadata_path}")
    
    return filepath, metadata_path


# =============================================================================
# Gestion des Grilles de Scans (Analyse de Stabilité)
# =============================================================================

def _get_scan_amplitude(result: Dict) -> float | None:
    """
    Extrait l'amplitude d'un résultat de scan.

    Le format recommandé est la clé "amplitude".
    La clé "amplitude_u0" est acceptée pour compatibilité avec les anciens résultats.
    """
    amplitude = result.get("amplitude")

    if amplitude is None:
        amplitude = result.get("amplitude_u0")

    if amplitude is None:
        return None

    return float(amplitude)


def build_scan_grid(
    results: List[Dict],
    dt_vals: List[float],
    amp_vals: List[float],
    extractor: Callable[[Dict], float],
    default: float = 0.0
) -> np.ndarray:
    """
    Construit une grille 2D à partir des résultats de scan.
    
    Crée une grille (amplitude x dt) avec une métrique custom.
    
    Args:
        results: Liste des résultats du scan
        dt_vals: Valeurs de dt uniques (triées)
        amp_vals: Valeurs d'amplitude uniques (triées)
        extractor: Fonction pour extraire la valeur à tracer
        default: Valeur par défaut pour les cellules vides
        
    Returns:
        np.ndarray: Grille 2D (amp_vals, dt_vals)
        
    Example:
        >>> results = [
        ...     {"dt": 1e-8, "amplitude": 0.1, "max_u": 0.5},
        ...     {"dt": 2e-8, "amplitude": 0.1, "max_u": 1.2}
        ... ]
        >>> grid = build_scan_grid(results, [1e-8, 2e-8], [0.1], lambda r: r["max_u"])
    """
    grid = np.full((len(amp_vals), len(dt_vals)), default, dtype=float)
    
    # Créer des dictionnaires d'indexation pour performance
    dt_index = {value: idx for idx, value in enumerate(dt_vals)}
    amp_index = {value: idx for idx, value in enumerate(amp_vals)}
    
    for result in results:
        dt_raw = result.get("dt")
        amp = _get_scan_amplitude(result)

        if dt_raw is None or amp is None:
            continue

        dt = float(dt_raw)

        i = amp_index.get(amp)
        j = dt_index.get(dt)

        if i is not None and j is not None:
            grid[i, j] = float(extractor(result))
    
    return grid


def get_scan_axes(results: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Extrait les axes uniques d'un scan (dt et amplitude).
    
    Retourne les axes triés (dt, amplitude) à partir d'un scan.
    
    Args:
        results: Liste des résultats du scan
        
    Returns:
        Tuple[List[float], List[float]]: (dt_vals, amp_vals) triés et uniques
        
    Example:
        >>> results = [
        ...     {"dt": 1e-8, "amplitude": 0.1, "stable": True},
        ...     {"dt": 2e-8, "amplitude": 0.1, "stable": False}
        ... ]
        >>> dt_vals, amp_vals = get_scan_axes(results)
    """
    dt_vals = sorted({float(r["dt"]) for r in results if r.get("dt") is not None})
    amp_vals = sorted({amp for r in results if (amp := _get_scan_amplitude(r)) is not None})
    
    return dt_vals, amp_vals


def compute_stable_ratio(results: List[Dict]) -> float:
    """
    Renvoie le ratio de configurations stables.
    
    Calcule le ratio (entre 0 et 1) des configurations stables dans l'ensemble
    des résultats.
    
    Args:
        results: Liste des résultats du scan avec clé 'stable' booléenne
        
    Returns:
        float: Ratio de configurations stables dans [0, 1]
        
    Example:
        >>> results = [
        ...     {"dt": 1e-8, "amplitude": 0.1, "stable": True},
        ...     {"dt": 2e-8, "amplitude": 0.1, "stable": False}
        ... ]
        >>> ratio = compute_stable_ratio(results)  # retourne 0.5
    """
    values = [1.0 if r.get("stable", False) else 0.0 for r in results]
    if not values:
        return 0.0
    return float(np.mean(values))


# =============================================================================
# Utilitaires Mathématiques et Statistiques
# =============================================================================

def compute_gradient(diff: np.ndarray, dx: float, bc_type: str) -> np.ndarray:
    """Générique pour calculer le gradient."""
    if bc_type == "dirichlet":
        # gradient centré (points intérieurs)
        return (diff[2:] - diff[:-2]) / (2 * dx)

    elif bc_type == "neumann":
        gradient = np.zeros_like(diff)
        # Bord gauche (forward difference)
        gradient[1:-1] = (diff[2:] - diff[:-2]) / (2 * dx)
        # bords
        gradient[0] = (diff[1] - diff[0]) / dx
        gradient[-1] = (diff[-1] - diff[-2]) / dx
        return gradient

    elif bc_type == "periodic":
        return (np.roll(diff, -1) - np.roll(diff, 1)) / (2 * dx)


def compute_error_metrics(
    solution: np.ndarray,
    reference: np.ndarray,
    dx: float = 1.0,
    compute_l2: bool = True,
    compute_h1: bool = True,
    compute_linf: bool = True,
    compute_rmse: bool = True,
    bc_type: str = "dirichlet"
) -> Dict[str, float]:
    """
    Calcule les métriques d'erreur entre deux solutions.
    
    Args:
        solution: Solution numérique
        reference: Solution de référence
        dx: le pas spatial pour la normalisation
        compute_l2: Calculer l'erreur L2
        compute_h1: Calculer l'erreur H1
        compute_linf: Calculer l'erreur L∞
        compute_rmse: Calculer la RMSE
        bc_type: conditions aux bords
        
    Returns:
        Dict[str, float]: Dictionnaire des métriques d'erreur
    """

    if solution.shape != reference.shape:
        raise ValueError("solution and reference doivent être de mêmes dimensions.")

    if bc_type not in ["dirichlet", "neumann", "periodic"]:
        raise ValueError("bc_type doit être 'dirichlet', 'neumann', ou 'periodic'")

    metrics: Dict[str, float] = {}
    diff = solution - reference
    l2_squared = dx * np.sum(diff**2)

    # Metrique de l'erreur L2
    if compute_l2:
        metrics["L2"] = float(np.sqrt(l2_squared))
    # Metrique de l'erreur H1
    elif compute_h1:
        gradient_diff = compute_gradient(diff, dx, bc_type=bc_type)
        gradient_squared = dx * np.sum(gradient_diff**2)
        metrics["H1"] = float(np.sqrt(l2_squared + gradient_squared))
    # Metrique Linf
    elif compute_linf:
        metrics["Linf"] = float(np.max(np.abs(diff)))
    # Metrque RMSE
    elif compute_rmse:
        metrics["RMSE"] = float(np.sqrt(np.mean(diff**2)))

    return metrics


def compute_linf_time_error(
        solution_time: np.ndarray,
        reference_time: np.ndarray,
        dx: float = 1.0,
        norm_type: str = "L2",
        bc_type: str = "dirichlet"
) -> float:
    """
    Calcule max_n ||solution_time[n] - reference_time[n]||_X.

    norm_type:
        - "L2"
        - "H1"
        - "grad"
        - "Linf"
        - "RMSE"
    """
    if solution_time.shape != reference_time.shape:
        raise ValueError("Solution et référence doivent être de mêmes dimensions.")

    if norm_type not in ["L2", "H1", "Linf", "grad", "RMSE"]:
        raise ValueError(f"Type de norme invalide: {norm_type}")

    errors = []

    for sol, ref in zip(solution_time, reference_time):
        if norm_type == "grad":
            grad_diff = compute_gradient(sol - ref, dx, bc_type=bc_type)
            error = np.sqrt(dx * np.sum(grad_diff**2))

        else:
            metrics = compute_error_metrics(
                sol,
                ref,
                dx=dx,
                compute_l2=(norm_type == "L2"),
                compute_h1=(norm_type == "H1"),
                compute_linf=(norm_type == "Linf"),
                compute_rmse=(norm_type == "RMSE"),
                bc_type=bc_type
            )
            error = metrics[norm_type]

        errors.append(error)

    return float(np.max(errors))


def compute_convergence_orders(error: Dict[str, float],) -> Dict[str, float]:
    """
    Calcule les ordres de convergence :
        ordre_N = log(e_{N-1} / e_N) / log(2)

    Args:
        errors: dictionnaire {N: erreur_N}

    Returns:
        dictionnaire {N: ordre_N}
    """
    orders = {}

    levels = sorted(error.keys())

    for i in range(1, len(levels)):
        N_prev = levels[i - 1]
        N = levels[i]

        e1 = error[N_prev]
        e2 = error[N]

        if e2 == 0 or e1 == 0:
            orders[N] = float('inf')
        else:
            orders[N] = float(np.log(e1 / e2) / np.log(2))

    return orders


def normalize_array(arr: np.ndarray, mode: Literal["minmax", "zscore", "robust"] = "minmax") -> np.ndarray:
    """
    Normalise un array NumPy.
    
    Args:
        arr: Array à normaliser
        mode: 'minmax' [0,1] ou 'zscore' (moyenne 0, std 1) ou 'robust' (Q1-Q3)
        
    Returns:
        np.ndarray: Array normalisé
    """
    arr = np.asarray(arr, dtype=float)

    if mode == "minmax":
        vmin = np.min(arr)
        vmax = np.max(arr)

        if vmax == vmin:
            return np.zeros_like(arr)

        return (arr - vmin) / (vmax - vmin)

    elif mode == "zscore":
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return np.zeros_like(arr)

        return (arr - mean) / std

    elif mode == "robust":
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1

        if iqr == 0:
            return np.zeros_like(arr)

        return (arr - q1) / iqr

    else:
        raise ValueError(f"Mode inconnu: {mode}")


# =============================================================================
# Utilitaires pour Matplotlib
# =============================================================================

def set_style(style: str = "seaborn-v0_8-darkgrid") -> None:
    """
    Configure le style de matplotlib.
    
    Args:
        style: Style matplotlib à utiliser
    """
    try:
        plt.style.use(style)
    except OSError:
        print(f"Style '{style}' non trouvé, utilisation du style par défaut")
        plt.style.use("default")


def create_comparison_figure(
    num_subplots: int,
    figsize: Tuple[int, int] = (14, 8)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Crée une figure avec subplots pour les comparaisons.
    
    Args:
        num_subplots: Nombre de subplots souhaités
        figsize: Taille de la figure
        
    Returns:
        Tuple[Figure, List[Axes]]: Figure et liste des axes
    """
    # Déterminer la grille appropriée
    cols = min(3, num_subplots)
    rows = (num_subplots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Assurer que axes est toujours une liste
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    return fig, axes[:num_subplots]


# =============================================================================
# Logging et Débogage
# =============================================================================

_PROGRESS_BARS: Dict[Tuple[str, int], tqdm] = {}

def log_computation_params(
    params: Dict[str, Any],
    output_file: Optional[str] = None
) -> None:
    """
    Enregistre les paramètres de calcul.
    
    Args:
        params: Dictionnaire des paramètres
        output_file: Fichier de sortie optionnel (sinon affiche dans la console)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_str = f"\n{'='*60}\n"
    log_str += f"Paramètres de calcul - {timestamp}\n"
    log_str += f"{'='*60}\n"
    
    for key, value in params.items():
        log_str += f"  {key:.<40} {value}\n"
    
    log_str += f"{'='*60}\n"
    
    print(log_str)
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write(log_str)


def print_progress(current: int, total: int, prefix: str = "Progression") -> None:
    """
    Affiche une barre de progression basée sur `tqdm`.
    
    Args:
        current: Itération actuelle
        total: Nombre total d'itérations
        prefix: Préfixe du message
    """
    if total <= 0:
        return

    key = (prefix, total)
    bar = _PROGRESS_BARS.get(key)

    if bar is None:
        bar = tqdm(total=total, desc=prefix, dynamic_ncols=True)
        _PROGRESS_BARS[key] = bar

    target = max(0, min(current, total))

    if target < bar.n:
        bar.close()
        bar = tqdm(total=total, desc=prefix, dynamic_ncols=True)
        _PROGRESS_BARS[key] = bar

    delta = target - bar.n
    if delta > 0:
        bar.update(delta)

    if target >= total:
        bar.close()
        _PROGRESS_BARS.pop(key, None)


# =============================================================================
# Validation Numérique et Stabilité
# =============================================================================

def compute_cfl_number(c: float, dt: float, dx: float) -> float:
    """
    Calcule le nombre de Courant-Friedrichs-Lewy (CFL).
    
    Important pour vérifier la stabilité des schémas explicites.
    CFL = c * dt / dx
    
    Args:
        c: Vitesse du son (m/s)
        dt: Discrétisation temporelle (s)
        dx: Discrétisation spatiale (m)
        
    Returns:
        float: Nombre CFL
        
    Example:
        >>> cfl = compute_cfl_number(1500, 2e-8, 1e-4)
    """
    return float(c * dt / dx)


def check_cfl_info(cfl: float, scheme: str = "explicit") -> str:
    """
    Retourne un message informatif sur CFL (non décisionnel pour la stabilité).

    La stabilité des schémas dans ce module est évaluée avec lambda,
    pas avec CFL.
    """
    _ = scheme  # Conservé pour compatibilité d'API
    return f"CFL = {cfl:.4f} (indicateur physique, stabilité évaluée via lambda)"


def compute_lambda_number(c: float, dt: float, dx: float) -> float:
    """
    Calcule le nombre de diffusion (Lambda).

    Lambda = c ** 2 * dt / dx^2.

    Args:
        c: Vitesse du son (m/s)
        dt: Discrétisation temporelle (s)
        dx: Discrétisation spatiale (m)

    Returns:
        float: Nombre de diffusion Lambda

    Example:
        >>> lambda_num = compute_lambda_number(1500, 2e-8, 1e-4)
    """
    return float(c ** 2 * dt / (dx ** 2))


def check_lambda_stability(lambda_num: float, scheme: str = "explicit") -> Tuple[bool, str]:
    """
    Vérifie la stabilité en fonction du nombre lambda.

    Args:
        lambda_num: Nombre de diffusion calculé
        scheme: Type de schéma ('explicit' ou 'semi_implicit')

    Returns:
        Tuple[bool, str]: (est_stable, message_detaille)

    Critères:
        - Explicite: conditionnel (lambda <= 0.5)
        - Semi-implicite: inconditionnel
    """
    normalized_scheme = scheme.lower().replace("-", "_")
    thresholds = {
        "explicit": 0.5,
        "semi_implicit": float("inf"),
    }

    if normalized_scheme not in thresholds:
        raise ValueError(f"Schéma inconnu: {scheme}")

    threshold = thresholds[normalized_scheme]
    is_stable = lambda_num <= threshold

    if normalized_scheme == "explicit":
        message = f"lambda = {lambda_num:.4f} (seuil explicit: 0.5) - "
        message += "Stable" if is_stable else "Instable (risque de divergence)"
    else:
        message = f"lambda = {lambda_num:.4f} - Stable (semi_implicit inconditionnel)"

    return is_stable, message


def check_cfl_stability(cfl: float, scheme: str = "explicit") -> Tuple[bool, str]:
    """
    Wrapper de compatibilité: redirige vers la logique de stabilité basee sur lambda.

    Note:
        Le paramètre `cfl` est interprété comme une valeur lambda pour conserver
        la compatibilité avec d'anciens appels. Utiliser directement
        `check_lambda_stability` dans le nouveau code.
    """
    return check_lambda_stability(cfl, scheme)


def compute_convergence_rate(errors: np.ndarray, mesh_sizes: np.ndarray) -> float:
    """
    Calcule le taux de convergence empirique.
    
    Utilise une régression log-log pour estimer l'ordre de convergence.
    
    Args:
        errors: Vecteur des erreurs (en fonction de h)
        mesh_sizes: Vecteur des tailles de maille h
        
    Returns:
        float: Taux de convergence estimé (pente en log-log)
        
    Example:
        >>> mesh_sizes = np.array([0.1, 0.05, 0.025, 0.0125])
        >>> errors = np.array([0.001, 0.00025, 0.0000625, 0.0000156])
        >>> rate = compute_convergence_rate(errors, mesh_sizes)
    """
    if len(errors) < 2 or len(mesh_sizes) < 2:
        return 0.0
    
    # Régression log-log
    log_h = np.log(mesh_sizes)
    log_e = np.log(errors)
    
    # Calcul de la pente
    coefficients = np.polyfit(log_h, log_e, 1)
    rate = coefficients[0]
    
    return float(rate)


def estimate_error_bounds(
    solution: np.ndarray,
    reference: np.ndarray = None,
    method: str = "absolute"
) -> Dict[str, float]:
    """
    Estime les bornes d'erreur d'une solution.
    
    Args:
        solution: Solution numérique
        reference: Solution de référence (optionnelle)
        method: 'absolute', 'relative', ou 'combined'
        
    Returns:
        Dict[str, float]: Bornes d'erreur estimées
    """
    result = {
        "max_abs_error": float(np.max(np.abs(solution))),
        "mean_abs_error": float(np.mean(np.abs(solution))),
        "std_error": float(np.std(solution))
    }
    
    if reference is not None:
        diff = solution - reference
        result["max_diff"] = float(np.max(np.abs(diff)))
        result["mean_diff"] = float(np.mean(np.abs(diff)))
        
        if method == "relative":
            ref_norm = np.max(np.abs(reference))
            if ref_norm > 1e-10:
                result["max_relative_error"] = result["max_diff"] / ref_norm
                result["mean_relative_error"] = result["mean_diff"] / ref_norm
    
    return result


# =============================================================================
# Développement et Analyse des Schémas
# =============================================================================

def analyze_scheme_properties(
    scheme_name: str,
    c: float = 1500,
    dt: float = 2e-8,
    dx: float = 1e-4
) -> Dict[str, Any]:
    """
    Analyse les propriétés théoriques d'un schéma numérique.
    
    Args:
        scheme_name: Nom du schéma ('explicit', 'semi_implicit')
        c: Vitesse du son
        dt: Discrétisation temporelle
        dx: Discrétisation spatiale
        
    Returns:
        Dict[str, Any]: Propriétés et caractéristiques du schéma
    """
    cfl_num = compute_cfl_number(c, dt, dx)
    lambda_num = compute_lambda_number(c, dt, dx)
    is_stable, stability_msg = check_lambda_stability(lambda_num, scheme_name)
    
    scheme_properties = {
        "explicit": {
            "order_space": 2,
            "order_time": 1,
            "stability": "Conditionnelle (lambda <= 0.5)",
            "computational_cost": "Faible",
            "use_case": "Domaines simples, haute résolution temporelle"
        },
        "semi_implicit": {
            "order_space": 2,
            "order_time": 1,
            "stability": "Inconditionnelle (basee sur lambda)",
            "computational_cost": "Moyen",
            "use_case": "Équilibre entre stabilité et coût"
        }
    }
    
    props = scheme_properties.get(scheme_name, {})
    
    return {
        "scheme": scheme_name,
        "cfl": cfl_num,
        "cfl_number": cfl_num,
        "lambda": lambda_num,
        "lambda_number": lambda_num,
        "is_stable": is_stable,
        "stability_message": stability_msg,
        **props
    }


def compare_schemes(
    c: float = 1500,
    dt: float = 2e-8,
    dx: float = 1e-4,
    schemes: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Compare les propriétés de plusieurs schémas numériques.
    
    Args:
        c: Vitesse du son
        dt: Discrétisation temporelle
        dx: Discrétisation spatiale
        schemes: Liste des schémas à comparer (default: tous)
        
    Returns:
        List[Dict[str, Any]]: Propriétés de chaque schéma
    """
    if schemes is None:
        schemes = ["explicit", "semi_implicit"]
    
    return [analyze_scheme_properties(s, c, dt, dx) for s in schemes]


def estimate_memory_usage(
    nx: int,
    nt: int,
    dtype: str = "float64"
) -> Dict[str, float]:
    """
    Estime l'utilisation mémoire d'une simulation.
    
    Args:
        nx: Nombre de points spatiaux
        nt: Nombre d'itérations temporelles
        dtype: Type de données NumPy (default: float64)
        
    Returns:
        Dict[str, float]: Estimations mémoire en MB
    """
    dtype_sizes = {
        "float32": 4,
        "float64": 8,
        "complex64": 8,
        "complex128": 16
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 8)
    
    # Estimations pour différentes grilles
    grids = {
        "Solution (u)": nx,
        "Vitesse (du/dt)": nx,
        "Snapshots complets (nt fois)": nx * nt,
        "Historique énergie (nt)": nt
    }
    
    return {
        name: (size * bytes_per_element) / (1024 ** 2)
        for name, size in grids.items()
    }


def print_simulation_summary(
    params: Dict[str, Any],
    scheme_name: str = "semi_implicit",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Affiche et retourne un résumé complet de la simulation.
    
    Args:
        params: Dictionnaire des paramètres (doit contenir c, dt, dx, nx, nt)
        scheme_name: Nom du schéma numérique
        verbose: Afficher le résumé dans la console
        
    Returns:
        Dict[str, Any]: Résumé de la simulation
    """
    c = params.get("c", 1500)
    dt = params.get("dt", 2e-8)
    dx = params.get("dx", 1e-4)
    nx = params.get("nx", 1200)
    nt = params.get("nt", 1200)
    
    lambda_num = compute_lambda_number(c, dt, dx)
    is_stable, stability_msg = check_lambda_stability(lambda_num, scheme_name)
    memory = estimate_memory_usage(nx, nt)
    
    summary = {
        "parameters": params,
        "stability_analysis": {
            "lambda": lambda_num,
            "is_stable": is_stable,
            "stability_message": stability_msg
        },
        "memory_estimation": memory,
        "scheme": analyze_scheme_properties(scheme_name, c, dt, dx)
    }
    
    if verbose:
        print("\n" + "="*70)
        print("RÉSUMÉ DE LA SIMULATION")
        print("="*70)
        print(f"\n Paramètres:")
        print(f"  • Vitesse du son (c)............ {c} m/s")
        print(f"  • Discrétisation spatiale (dx). {dx} m")
        print(f"  • Discrétisation temporelle (dt) {dt} s")
        print(f"  • Points spatiaux (nx)......... {nx}")
        print(f"  • Itérations temporelles (nt). {nt}")
        print(f"  • Schéma........................ {scheme_name}")
        
        print(f"\n Analyse stabilité:")
        print(f"  • {stability_msg}")
        
        print(f"\n Estimation mémoire:")
        for name, size in memory.items():
            print(f"  • {name:.<45} {size:.2f} MB")
        
        total_memory = sum(memory.values())
        print(f"  • {'Total estimé':.<45} {total_memory:.2f} MB")
        print("\n" + "="*70 + "\n")
    
    return summary


