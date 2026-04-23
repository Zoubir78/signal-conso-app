from __future__ import annotations

from datetime import datetime

import pandas as pd
from google.cloud import bigquery


def get_client(project_id: str | None = None) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def upload_dataframe_to_bigquery(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    """
    Charge un DataFrame pandas dans BigQuery.

    Args:
        df: DataFrame à charger.
        project_id: ID du projet GCP.
        dataset_id: ID du dataset BigQuery (ex: 'signalconso_raw').
        table_id: Nom de la table (ex: 'signalconso').
        write_disposition: 'WRITE_APPEND' ou 'WRITE_TRUNCATE'.
    """
    client = get_client(project_id)

    # Ajout de la colonne d'ingestion pour la traçabilité dbt
    df = df.copy()
    df["_ingested_at"] = datetime.utcnow().isoformat()

    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,  # BigQuery infère le schéma automatiquement
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # attend la fin du job

    print(f"Chargé {len(df)} lignes dans {table_ref}")


def export_table_to_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    bucket_name: str,
    blob_prefix: str,
    file_format: str = "CSV",
) -> None:
    """
    Exporte une table BigQuery vers GCS (optionnel, remplace l'upload CSV manuel).

    Args:
        project_id: ID du projet GCP.
        dataset_id: ID du dataset BigQuery.
        table_id: Nom de la table.
        bucket_name: Nom du bucket GCS.
        blob_prefix: Préfixe du fichier de destination (ex: 'processed/mart_signalconso').
        file_format: 'CSV' ou 'NEWLINE_DELIMITED_JSON'.
    """
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
    print(f"Export de {table_ref} vers {destination_uri} terminé")
