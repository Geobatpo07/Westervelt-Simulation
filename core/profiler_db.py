# ./core/profiler_db.py

"""
Gestion SQLite des enregistrements de profiling pour les validations numériques.

Ce module fournit des outils pour stocker, gérer et analyser les données de 
profiling (temps d'exécution, utilisation mémoire) issues des simulations 
numériques de l'équation de Westervelt. Les données sont stockées dans une 
base de données SQLite pour faciliter les requêtes analytiques.

Caractéristiques principales :
    * Création et initialisation de la base de données SQLite.
    * Insertion d'enregistrements individuels ou en lot.
    * Importation de données depuis des fichiers CSV existants.
    * Lecture des données sous forme de DataFrames pandas.
    * Requêtes analytiques prédéfinies pour les rapports et le mémoire.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Tuple, Callable, Optional
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / Path("data/profiler_runs.sqlite")
DEFAULT_TABLE_NAME = "profiler_runs"

PROFILER_COLUMNS = [
    "timestamp",
    "function",
    "context",
    "validation_type",
    "scheme",
    "bc",
    "nx",
    "nt",
    "dx",
    "dt",
    "T_final",
    "L",
    "A",
    "A1",
    "A2",
    "c",
    "rho0",
    "beta",
    "mu_v",
    "b",
    "time_total_s",
    "memory_max_mb",
    "loaded_from_cache",
    "cache_path",
]


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    timestamp TEXT NOT NULL,

    function TEXT,
    context TEXT,
    validation_type TEXT,

    scheme TEXT,
    bc TEXT,

    nx INTEGER,
    nt INTEGER,

    dx REAL,
    dt REAL,
    T_final REAL,
    L REAL,

    A REAL,
    A1 REAL,
    A2 REAL,

    c REAL,
    rho0 REAL,
    beta REAL,
    mu_v REAL,
    b REAL,

    time_total_s REAL,
    memory_max_mb REAL,

    loaded_from_cache INTEGER,
    cache_path TEXT
);
"""


CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_{DEFAULT_TABLE_NAME}_scheme ON {DEFAULT_TABLE_NAME}(scheme);",
    f"CREATE INDEX IF NOT EXISTS idx_{DEFAULT_TABLE_NAME}_validation_type ON {DEFAULT_TABLE_NAME}(validation_type);",
    f"CREATE INDEX IF NOT EXISTS idx_{DEFAULT_TABLE_NAME}_nx ON {DEFAULT_TABLE_NAME}(nx);",
    f"CREATE INDEX IF NOT EXISTS idx_{DEFAULT_TABLE_NAME}_cache ON {DEFAULT_TABLE_NAME}(loaded_from_cache);",
    f"CREATE INDEX IF NOT EXISTS idx_{DEFAULT_TABLE_NAME}_timestamp ON {DEFAULT_TABLE_NAME}(timestamp);",
]

def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Établit une connexion à la base de données SQLite.

    Crée le répertoire parent si nécessaire.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers le fichier de base de données SQLite.
        Par défaut : DEFAULT_DB_PATH.

    Returns
    -------
    sqlite3.Connection
        Objet de connexion SQLite.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_profiler_database(
        db_path: str | Path = DEFAULT_DB_PATH,
        table_name: str = DEFAULT_TABLE_NAME,
) -> None:
    """
    Initialise la base de données du profiler et crée les tables et index.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers le fichier de base de données.
    table_name : str, optional
        Nom de la table à créer. Doit être 'profiler_runs'.

    Raises
    ------
    ValueError
        Si le nom de la table n'est pas celui attendu.
    """
    if table_name != DEFAULT_TABLE_NAME:
        raise ValueError("Table name must be 'profiler_runs'.")

    with get_connection(db_path) as conn:
        conn.executescript(CREATE_TABLE_SQL)
        for sql in CREATE_INDEXES_SQL:
            conn.execute(sql)
        conn.commit()


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise un dictionnaire d'enregistrement selon les colonnes attendues.

    Assure que toutes les colonnes requises sont présentes et convertit
    le flag de cache en entier pour SQLite.

    Parameters
    ----------
    record : Dict[str, Any]
        Dictionnaire brut contenant les données de profiling.

    Returns
    -------
    Dict[str, Any]
        Dictionnaire normalisé prêt pour l'insertion SQL.
    """
    normalized = {col: record.get(col, None) for col in PROFILER_COLUMNS}

    if normalized['loaded_from_cache'] is not None:
        normalized['loaded_from_cache'] = int(bool(normalized['loaded_from_cache']))

    return normalized


def insert_profiler_record(
        record: Dict[str, Any],
        db_path: str | Path = DEFAULT_DB_PATH,
) -> int | None:
    """
    Insère un seul enregistrement de profiling dans la base de données.

    Initialise la base de données si elle n'existe pas.

    Parameters
    ----------
    record : Dict[str, Any]
        Données de profiling à insérer.
    db_path : str | Path, optional
        Chemin vers la base de données.

    Returns
    -------
    int | None
        L'ID de la ligne insérée ou None.
    """
    initialize_profiler_database(db_path)

    normalized = _normalize_record(record)
    columns = list(normalized.keys())
    values = [normalized[col] for col in columns]

    placeholders = ", ".join(["?"] * len(columns))
    columns_sql = ", ".join(columns)

    sql = f"INSERT INTO {DEFAULT_TABLE_NAME} ({columns_sql}) VALUES ({placeholders})"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        return cursor.lastrowid


def insert_many_profiler_records(
        records: List[Dict[str, Any]],
        db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    Insère plusieurs enregistrements de profiling en une seule transaction.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Liste de dictionnaires d'enregistrements.
    db_path : str | Path, optional
        Chemin vers la base de données.

    Returns
    -------
    int
        Le nombre de lignes insérées.
    """
    initialize_profiler_database(db_path)

    normalized_records = [_normalize_record(record) for record in records]
    if not normalized_records:
        return 0

    columns = list(normalized_records[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    columns_sql = ", ".join(columns)

    sql = f"INSERT INTO {DEFAULT_TABLE_NAME} ({columns_sql}) VALUES ({placeholders})"

    values = [[record[col] for col in columns] for record in normalized_records]

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, values)
        conn.commit()

    return cursor.rowcount


def _validate_csv_field_counts(csv_path: Path) -> None:
    """
    Vérifie que chaque ligne du CSV contient le même nombre de champs que l'en-tête.

    Cette validation donne un message d'erreur plus clair que pandas lorsque le CSV
    contient une virgule non échappée ou une ligne incomplète/surnuméraire.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV vide : {csv_path}") from None

        expected_fields = len(header)

        for line_number, row in enumerate(reader, start=2):
            actual_fields = len(row)
            if actual_fields != expected_fields:
                raise ValueError(
                    f"CSV mal formé : {csv_path}\n"
                    f"Ligne {line_number} : attendu {expected_fields} champs, "
                    f"trouvé {actual_fields}.\n"
                    f"Cause probable : une virgule non protégée dans une valeur texte, "
                    f"ou une colonne en trop/manquante.\n"
                    f"Ligne lue : {row}"
                )


def import_profiler_csv(
    csv_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    validation_type: Optional[str] = None,
    if_exists: str = "append",
) -> int:
    """
    Importe un fichier CSV de profiling vers SQLite.

    Parameters
    ----------
    csv_path : str | Path
        Chemin du fichier CSV.
    db_path : str | Path, optional
        Chemin de la base SQLite.
    validation_type : str, optional
        Valeur à ajouter si la colonne validation_type n'existe pas dans le CSV.
    if_exists : str, optional
        "append" pour ajouter aux données existantes,
        "replace" pour supprimer et recréer la table avant l'import.

    Returns
    -------
    int
        Nombre de lignes importées avec succès.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV n'existe pas.
    ValueError
        Si `if_exists` n'est ni 'append' ni 'replace'.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except pd.errors.ParserError:
        # En cas d'erreur de parsing (ex: colonnes en trop), on réessaie plus souplement.
        df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines='skip')

    if if_exists == "replace":
        clear_profiler_runs(db_path)
    elif if_exists != "append":
        raise ValueError("if_exists must be 'append' or 'replace'")

    if "validation_type" not in df.columns:
        df["validation_type"] = validation_type

    # Les colonnes A1 et A2 sont souvent à la fin des nouveaux CSV, mais absentes des anciens.
    # Dans certains CSV mal formés, A1/A2 sont présentes mais sans en-tête correspondant.
    # On gère dynamiquement les colonnes supplémentaires non nommées si elles existent.
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed:')]
    if unnamed_cols:
        # Si on a 2 colonnes en trop, on suppose que c'est A1 et A2
        if len(unnamed_cols) == 2:
            df = df.rename(columns={unnamed_cols[0]: 'A1', unnamed_cols[1]: 'A2'})
        elif len(unnamed_cols) == 1:
            df = df.rename(columns={unnamed_cols[0]: 'A1'})

    # Les anciens CSV peuvent contenir A, mais pas A1/A2 (ou inversement).
    # On s'assure que toutes les colonnes attendues par SQLite existent dans le DataFrame.
    for col in PROFILER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Conversion des booléens pour SQLite
    if 'loaded_from_cache' in df.columns:
        df['loaded_from_cache'] = df['loaded_from_cache'].apply(_to_sqlite_bool)

    records = df.to_dict(orient="records")
    return insert_many_profiler_records(records, db_path)


def _to_sqlite_bool(value: Any) -> Optional[int]:
    """
    Convertit différentes représentations booléennes en format SQLite (0/1).

    Gère les booléens Python, les nombres, les chaînes de caractères (true/false,
    yes/no, oui/non) et les valeurs manquantes (NaN).

    Parameters
    ----------
    value : Any
        Valeur à convertir.

    Returns
    -------
    Optional[int]
        1 pour True, 0 pour False, ou None si la valeur est indéterminée.
    """
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(bool(value))

    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y", "oui"}:
        return 1
    if value_str in {"false", "0", "no", "n", "non"}:
        return 0

    return None


def read_profiler_runs(
    db_path: str | Path = DEFAULT_DB_PATH,
    where: str | None = None,
    params: tuple[Any, ...] = (),
    order_by: str = "timestamp",
) -> pd.DataFrame:
    """
    Lit les enregistrements de profiling sous forme de DataFrame pandas.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers la base de données.
    where : str, optional
        Clause WHERE SQL personnalisée (ex: "scheme = ?").
    params : tuple, optional
        Paramètres pour la clause WHERE.
    order_by : str, optional
        Colonne pour le tri des résultats.

    Returns
    -------
    pd.DataFrame
        Données de profiling résultant de la requête.

    Examples
    --------
    >>> df = read_profiler_runs(where="scheme = ?", params=("semi_implicit",))
    """
    initialize_profiler_database(db_path)

    sql = f"SELECT * FROM {DEFAULT_TABLE_NAME}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"

    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def delete_profiler_run(
    run_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """
    Supprime un enregistrement de profiling par son identifiant unique.

    Parameters
    ----------
    run_id : int
        Identifiant (ID) de la ligne à supprimer.
    db_path : str | Path, optional
        Chemin vers la base de données.
    """
    initialize_profiler_database(db_path)

    with get_connection(db_path) as conn:
        conn.execute(
            f"DELETE FROM {DEFAULT_TABLE_NAME} WHERE id = ?",
            (run_id,),
        )
        conn.commit()


def clear_profiler_runs(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """
    Vide intégralement la table de profiling.

    La structure de la table et ses index sont conservés.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers la base de données.
    """
    initialize_profiler_database(db_path)

    with get_connection(db_path) as conn:
        conn.execute(f"DELETE FROM {DEFAULT_TABLE_NAME}")
        conn.commit()


def summarize_by_scheme(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Agrège les statistiques de performance par schéma numérique.

    Calcule le nombre d'exécutions, ainsi que les moyennes et maxima
    du temps total et de l'utilisation mémoire pour chaque schéma.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers la base de données.

    Returns
    -------
    pd.DataFrame
        Résumé statistique par schéma (scheme, n_runs, avg_time, max_time, etc.).
    """
    initialize_profiler_database(db_path)

    sql = f"""
    SELECT
        scheme,
        COUNT(*) AS n_runs,
        AVG(time_total_s) AS avg_time_total_s,
        MAX(time_total_s) AS max_time_total_s,
        AVG(memory_max_mb) AS avg_memory_max_mb,
        MAX(memory_max_mb) AS max_memory_max_mb
    FROM {DEFAULT_TABLE_NAME}
    GROUP BY scheme
    ORDER BY scheme;
    """

    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def summarize_by_validation_type(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Agrège les performances par type de validation et par schéma.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers la base de données.

    Returns
    -------
    pd.DataFrame
        Statistiques groupées par type de validation et schéma.
    """
    initialize_profiler_database(db_path)

    sql = f"""
    SELECT
        validation_type,
        scheme,
        COUNT(*) AS n_runs,
        AVG(time_total_s) AS avg_time_total_s,
        MAX(time_total_s) AS max_time_total_s,
        AVG(memory_max_mb) AS avg_memory_max_mb,
        MAX(memory_max_mb) AS max_memory_max_mb
    FROM {DEFAULT_TABLE_NAME}
    GROUP BY validation_type, scheme
    ORDER BY validation_type, scheme;
    """

    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def performance_vs_mesh(
    db_path: str | Path = DEFAULT_DB_PATH,
    validation_type: str | None = None,
    scheme: str | None = None,
    exclude_cache: bool = True,
) -> pd.DataFrame:
    """
    Analyse l'évolution des performances en fonction de la finesse du maillage.

    Cette fonction est particulièrement utile pour générer des courbes de
    complexité (temps/mémoire en fonction de `nx`).

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers la base de données.
    validation_type : str, optional
        Filtrer par type de validation spécifique.
    scheme : str, optional
        Filtrer par schéma numérique spécifique.
    exclude_cache : bool, optional
        Si True, exclut les exécutions chargées depuis le cache pour ne pas
        fausser les mesures de performance réelles.

    Returns
    -------
    pd.DataFrame
        Données de performance (moyennes et maxima) groupées par paramètres
        de simulation et ordonnées par taille de maillage.
    """
    initialize_profiler_database(db_path)

    clauses = []
    params: list[Any] = []

    if validation_type is not None:
        clauses.append("validation_type = ?")
        params.append(validation_type)

    if scheme is not None:
        clauses.append("scheme = ?")
        params.append(scheme)

    if exclude_cache:
        clauses.append("COALESCE(loaded_from_cache, 0) = 0")

    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)

    sql = f"""
    SELECT
        validation_type,
        scheme,
        bc,
        nx,
        nt,
        dx,
        dt,
        T_final,
        L,
        A,
        A1,
        A2,
        AVG(time_total_s) AS avg_time_total_s,
        MAX(time_total_s) AS max_time_total_s,
        AVG(memory_max_mb) AS avg_memory_max_mb,
        MAX(memory_max_mb) AS max_memory_max_mb,
        COUNT(*) AS n_runs
    FROM {DEFAULT_TABLE_NAME}
    {where_sql}
    GROUP BY validation_type, scheme, bc, nx, nt, dx, dt, T_final, L, A, A1, A2
    ORDER BY nx, nt;
    """

    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=tuple(params))


def cache_statistics(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Calcule des statistiques sur l'utilisation et l'efficacité du cache.

    Compare les temps d'exécution et l'occupation mémoire des exécutions
    calculées vs celles récupérées du cache.

    Parameters
    ----------
    db_path : str | Path, optional
        Chemin vers la base de données.

    Returns
    -------
    pd.DataFrame
        Statistiques comparatives (n_runs, avg_time, avg_memory) pour les
        états de cache (0 ou 1).
    """
    initialize_profiler_database(db_path)

    sql = f"""
    SELECT
        COALESCE(loaded_from_cache, 0) AS loaded_from_cache,
        COUNT(*) AS n_runs,
        AVG(time_total_s) AS avg_time_total_s,
        AVG(memory_max_mb) AS avg_memory_max_mb
    FROM {DEFAULT_TABLE_NAME}
    GROUP BY COALESCE(loaded_from_cache, 0)
    ORDER BY loaded_from_cache;
    """

    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def export_query_to_csv(
    output_path: str | Path,
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    params: tuple[Any, ...] = (),
) -> Path:
    """
    Exécute une requête SQL personnalisée et exporte le résultat au format CSV.

    Parameters
    ----------
    output_path : str | Path
        Chemin du fichier CSV de sortie.
    query : str
        Requête SQL à exécuter.
    db_path : str | Path, optional
        Chemin vers la base de données.
    params : tuple, optional
        Paramètres pour la requête SQL.

    Returns
    -------
    Path
        Le chemin vers le fichier CSV généré.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    db = DEFAULT_DB_PATH
    initialize_profiler_database(db)
    import_profiler_csv(PROJECT_ROOT / "data/profiler_records.csv", db)
    print(f"Base initialisée : {db}")



