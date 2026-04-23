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

    Args:
        df: DataFrame à charger.
        project_id: ID du projet GCP.
        dataset_id: Dataset cible (défaut : 'Complaints').
        table_id: Table cible (défaut : 'Signal_Conso').
        write_disposition: 'WRITE_APPEND' ou 'WRITE_TRUNCATE'.
    """
    client = get_client(project_id)

    df = df.copy()
    df["_ingested_at"] = datetime.utcnow().isoformat()

    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
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