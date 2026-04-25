from __future__ import annotations

from datetime import datetime

import pandas as pd
from google.cloud import bigquery

# ── Constantes projet ────────────────────────────────────────────────────────
PROJECT_ID      = "tri-demandes-clients"
SOURCE_DATASET  = "Complaints"
SOURCE_TABLE    = "Signal_Conso"
MART_DATASET    = "signalconso_dev_marts"   # signalconso_prod_marts en prod
MART_TABLE      = "mart_signalconso"
GCS_BUCKET      = "clean_complaints"


# ── Client ───────────────────────────────────────────────────────────────────

def get_client(project_id: str = PROJECT_ID) -> bigquery.Client:
    return bigquery.Client(project=project_id)


# ── Lecture ──────────────────────────────────────────────────────────────────

def read_source_table(
    limit: int | None = None,
    project_id: str = PROJECT_ID,
) -> pd.DataFrame:
    """
    Lit la table source SignalConso depuis BigQuery et retourne un DataFrame.

    Args:
        limit: Nombre maximum de lignes (None = tout lire).
        project_id: ID du projet GCP.
    """
    client = get_client(project_id)
    table_ref = f"{project_id}.{SOURCE_DATASET}.{SOURCE_TABLE}"

    query = f"SELECT * FROM `{table_ref}`"
    if limit:
        query += f" LIMIT {limit}"

    df = client.query(query).to_dataframe()
    print(f"Lu {len(df)} lignes depuis {table_ref}")
    return df


def read_mart_table(
    project_id: str = PROJECT_ID,
    dataset_id: str = MART_DATASET,
    table_id: str = MART_TABLE,
    filters: str | None = None,
) -> pd.DataFrame:
    """
    Lit le mart dbt final depuis BigQuery.

    Args:
        project_id: ID du projet GCP.
        dataset_id: Dataset du mart dbt (dev ou prod).
        table_id: Nom de la table mart.
        filters: Clause WHERE optionnelle (ex: "year = 2024 AND dep_code = '75'").
    """
    client = get_client(project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    query = f"SELECT * FROM `{table_ref}`"
    if filters:
        query += f" WHERE {filters}"

    df = client.query(query).to_dataframe()
    print(f"Lu {len(df)} lignes depuis {table_ref}")
    return df


# ── Écriture ─────────────────────────────────────────────────────────────────

# Correspondance des types BigQuery → cast pandas à appliquer avant upload
_BQ_TYPE_CASTERS = {
    "STRING":    lambda s: s.apply(
                     lambda v: __import__('json').dumps(v, ensure_ascii=False)
                     if isinstance(v, (list, dict)) else (str(v) if pd.notna(v) else None)
                 ),
    "DATE":      lambda s: pd.to_datetime(s, errors="coerce").dt.date,
    "TIMESTAMP": lambda s: pd.to_datetime(s, errors="coerce"),
    "DATETIME":  lambda s: pd.to_datetime(s, errors="coerce"),
    "INTEGER":   lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
    "FLOAT":     lambda s: pd.to_numeric(s, errors="coerce"),
    "BOOLEAN":   lambda s: s.map(
                     lambda v: bool(v) if v not in (None, float("nan"), "") else None
                 ),
}


def _align_to_bq_schema(
    df: pd.DataFrame,
    bq_schema: list,
) -> pd.DataFrame:
    """
    Caste chaque colonne du DataFrame selon le type déclaré dans le schéma BigQuery.
    Colonnes absentes du schéma : converties en string par sécurité.
    Colonnes absentes du DataFrame : ignorées (BigQuery les remplira avec NULL).
    """
    import json

    df = df.copy()
    schema_map = {field.name: field.field_type for field in bq_schema}

    for col in df.columns:
        bq_type = schema_map.get(col)
        if bq_type and bq_type in _BQ_TYPE_CASTERS:
            try:
                df[col] = _BQ_TYPE_CASTERS[bq_type](df[col])
            except Exception as e:
                print(f"  ⚠ Cast échoué pour '{col}' ({bq_type}) : {e} — conversion en string")
                df[col] = df[col].astype(str).where(df[col].notna(), other=None)
        elif df[col].dtype == object:
            # Colonne hors schéma ou type inconnu : sérialise listes/dicts, str sinon
            df[col] = df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict))
                else (str(v) if pd.notna(v) else None)
            )

    return df


def upload_dataframe_to_bigquery(
    df: pd.DataFrame,
    project_id: str = PROJECT_ID,
    dataset_id: str = SOURCE_DATASET,
    table_id: str = SOURCE_TABLE,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    """
    Charge un DataFrame dans la table source BigQuery.
    Par défaut cible Complaints.Signal_Conso.

    Aligne automatiquement les types pandas sur le schéma BigQuery existant :
    - Listes/dicts → JSON string (category, subcategories, tags)
    - Dates string → DATE
    - Entiers string → INTEGER
    Évite toutes les erreurs PyArrow de conversion de type.

    Args:
        df: DataFrame à charger.
        project_id: ID du projet GCP.
        dataset_id: Dataset cible (défaut : 'Complaints').
        table_id: Table cible (défaut : 'Signal_Conso').
        write_disposition: 'WRITE_APPEND' ou 'WRITE_TRUNCATE'.
    """
    client    = get_client(project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Récupère le schéma de la table existante pour aligner les types
    try:
        bq_table = client.get_table(table_ref)
        full_schema = bq_table.schema

        # Aligne les types des colonnes présentes dans le DataFrame
        df = _align_to_bq_schema(df, full_schema)
        df["_ingested_at"] = datetime.utcnow().isoformat()

        # Ne passe à BigQuery que les champs présents dans le DataFrame
        # (clean_text, is_valid, token_count sont calculés par dbt — absents du raw)
        df_cols = set(df.columns)
        bq_schema = [f for f in full_schema if f.name in df_cols]
        print(f"  Schéma filtré : {len(bq_schema)}/{len(full_schema)} champs")

    except Exception:
        # Table inexistante : autodetect suffit
        print("  Table absente, autodetect activé")
        df["_ingested_at"] = datetime.utcnow().isoformat()
        bq_schema = None

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema=bq_schema if bq_schema else None,
        autodetect=(bq_schema is None),
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"Chargé {len(df)} lignes dans {table_ref}")


# ── Export GCS ───────────────────────────────────────────────────────────────

def export_mart_to_gcs(
    project_id: str = PROJECT_ID,
    dataset_id: str = MART_DATASET,
    table_id: str = MART_TABLE,
    bucket_name: str = GCS_BUCKET,
    file_format: str = "CSV",
) -> str:
    """
    Exporte le mart dbt vers clean_complaints/processed/ dans GCS.
    Le nom du fichier inclut la date du jour pour le partitionnement.

    Args:
        project_id: ID du projet GCP.
        dataset_id: Dataset BigQuery source de l'export.
        table_id: Table à exporter.
        bucket_name: Bucket GCS cible (défaut : 'clean_complaints').
        file_format: 'CSV' ou 'NEWLINE_DELIMITED_JSON'.

    Returns:
        URI GCS de destination.
    """
    client = get_client(project_id)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    destination_uri = f"gs://{bucket_name}/processed/{table_id}_{today}_*.csv"

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        job_config=bigquery.ExtractJobConfig(
            destination_format=(
                bigquery.DestinationFormat.CSV
                if file_format == "CSV"
                else bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
            ),
            print_header=True,
        ),
    )
    extract_job.result()
    print(f"Export de {table_ref} -> {destination_uri} terminé")
    return destination_uri


# ── Alias conservé pour rétrocompatibilité ───────────────────────────────────

def export_table_to_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    bucket_name: str,
    blob_prefix: str,
    file_format: str = "CSV",
) -> None:
    """Alias maintenu pour compatibilité avec l'ancien run_pipeline.py."""
    client = get_client(project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    destination_uri = f"gs://{bucket_name}/{blob_prefix}_*.csv"

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        job_config=bigquery.ExtractJobConfig(
            destination_format=(
                bigquery.DestinationFormat.CSV
                if file_format == "CSV"
                else bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
            ),
            print_header=True,
        ),
    )
    extract_job.result()
    print(f"Export de {table_ref} -> {destination_uri} terminé")