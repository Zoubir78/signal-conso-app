from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.app.core.config import get_settings
from src.app.ingestion.extract import extract_from_signalconso_api
from src.app.services.bigquery_service import (
    export_table_to_gcs,
    upload_dataframe_to_bigquery,
)

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"

# Dataset BigQuery cible pour les données brutes
RAW_DATASET = "signalconso_raw"
RAW_TABLE = "signalconso"

# Dataset et table produits par dbt (mart final)
DBT_MART_DATASET = "signalconso_marts"
DBT_MART_TABLE = "mart_signalconso"

# Répertoire du projet dbt (relatif à ce script)
DBT_PROJECT_DIR = Path(__file__).parent / "dbt_signalconso"


def run_dbt(target: str = "dev") -> None:
    """Lance dbt run + dbt test via subprocess."""
    dbt_cmd_base = [
        "dbt",
        "--project-dir",
        str(DBT_PROJECT_DIR),
        "--profiles-dir",
        str(Path.home() / ".dbt"),
        "--target",
        target,
    ]

    print("--- dbt run ---")
    result = subprocess.run(
        dbt_cmd_base + ["run"],
        check=False,
        capture_output=False,
    )
    if result.returncode != 0:
        print("dbt run a échoué — arrêt du pipeline.", file=sys.stderr)
        sys.exit(result.returncode)

    print("--- dbt test ---")
    subprocess.run(
        dbt_cmd_base + ["test"],
        check=False,
        capture_output=False,
    )


def main(target: str = "dev", export_to_gcs: bool = True) -> None:
    settings = get_settings()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # ── 1. EXTRACT ──────────────────────────────────────────────────────────
    print(f"Extraction de l'API SignalConso ({today})…")
    raw_df = extract_from_signalconso_api(API_URL, limit=10000)
    print(f"  {len(raw_df)} enregistrements extraits")

    # ── 2. LOAD RAW → BigQuery ───────────────────────────────────────────────
    # Le DataFrame brut est chargé tel quel ; dbt se charge de la transformation.
    print(f"Chargement dans BigQuery ({RAW_DATASET}.{RAW_TABLE})…")
    upload_dataframe_to_bigquery(
        df=raw_df,
        project_id=settings.GCP_PROJECT_ID,
        dataset_id=RAW_DATASET,
        table_id=RAW_TABLE,
        write_disposition="WRITE_APPEND",  # WRITE_TRUNCATE pour repartir de zéro
    )

    # ── 3. TRANSFORM → dbt ──────────────────────────────────────────────────
    # dbt orchestre staging → intermediate → mart via SQL dans BigQuery.
    print("Lancement des modèles dbt…")
    run_dbt(target=target)

    # ── 4. EXPORT OPTIONNEL → GCS ───────────────────────────────────────────
    # Si besoin, exporte le mart final vers GCS pour d'autres consommateurs.
    if export_to_gcs and settings.GCS_BUCKET_NAME:
        print("Export du mart vers GCS…")
        export_table_to_gcs(
            project_id=settings.GCP_PROJECT_ID,
            dataset_id=DBT_MART_DATASET,
            table_id=DBT_MART_TABLE,
            bucket_name=settings.GCS_BUCKET_NAME,
            blob_prefix=f"processed/mart_signalconso_{today}",
        )

    print("Pipeline terminé.")


if __name__ == "__main__":
    # Usage : python run_pipeline.py [--target prod] [--no-gcs]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="dev", choices=["dev", "prod"])
    parser.add_argument("--no-gcs", action="store_true", help="Désactive l'export GCS")
    args = parser.parse_args()

    main(target=args.target, export_to_gcs=not args.no_gcs)
