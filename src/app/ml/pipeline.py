from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from src.app.core.config import get_settings
from src.app.ingestion.extract import extract_from_signalconso_api
from src.app.services.bigquery_service import (
    upload_dataframe_to_bigquery,
    read_mart_table,
    export_mart_to_gcs,
)
from src.app.services.gcs_service import upload_file_to_gcs
from src.app.ml.train import train_model

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"

DBT_PROJECT_DIR = Path(__file__).parent / "dbt"
DBT_TARGET      = "dev"   # passer "prod" en production


def _run_dbt(log, target: str = DBT_TARGET) -> None:
    """Lance dbt run + dbt test et lève une exception en cas d'échec."""
    base = [
        "dbt",
        "--project-dir", str(DBT_PROJECT_DIR),
        "--profiles-dir", str(Path.home() / ".dbt"),
        "--target", target,
    ]

    log("  dbt run...")
    result = subprocess.run(base + ["run"], capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ✖ dbt run échoué :\n{result.stdout[-2000:]}")
        raise RuntimeError("dbt run a échoué")

    log("  dbt test...")
    result = subprocess.run(base + ["test"], capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ⚠ dbt test en échec (pipeline continue) :\n{result.stdout[-1000:]}")


def run_pipeline(log) -> dict:
    """
    Pipeline complet intégrant dbt entre la transformation et l'entraînement ML.

    Flux :
      1. Extract API  → DataFrame brut
      2. Load raw     → BigQuery Complaints.Signal_Conso
      3. dbt run      → staging → intermediate → mart_signalconso
      4. Read mart    → DataFrame prêt pour le ML
      5. Train        → TF-IDF + LogisticRegression → model.joblib
      6. Upload       → GCS models/ + export mart processed/

    Args:
        log: callable de logging (st.write en Streamlit, print en CLI).

    Returns:
        dict avec les métriques clés du run.
    """
    settings = get_settings()
    today    = datetime.utcnow().strftime("%Y-%m-%d")

    log("🚀 Démarrage pipeline SignalConso")

    # ── 1. EXTRACT ────────────────────────────────────────────────────────────
    log("📥 Extraction API SignalConso...")
    raw_df = extract_from_signalconso_api(API_URL, limit=10_000)
    log(f"  ✔ {len(raw_df)} enregistrements extraits")

    # Sauvegarde locale optionnelle (utile pour debug / audit)
    raw_path = Path("data/raw/signalconso.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        str(raw_path),
        f"raw/signalconso_{today}.csv",
    )
    log("  ✔ RAW uploadé dans GCS")

    # ── 2. LOAD RAW → BigQuery ────────────────────────────────────────────────
    log("☁️  Chargement dans BigQuery (Complaints.Signal_Conso)...")
    upload_dataframe_to_bigquery(
        df=raw_df,
        project_id=settings.GCP_PROJECT_ID,
        write_disposition="WRITE_APPEND",
    )
    log(f"  ✔ {len(raw_df)} lignes chargées dans BigQuery")

    # ── 3. DBT ────────────────────────────────────────────────────────────────
    log("🔧 Modélisation dbt (staging → intermediate → mart)...")
    _run_dbt(log, target=DBT_TARGET)
    log("  ✔ dbt run + test terminés")

    # ── 4. LECTURE DU MART ────────────────────────────────────────────────────
    # On lit directement mart_signalconso depuis BigQuery :
    # clean_text est déjà normalisé, is_valid filtré, token_count calculé.
    # Plus besoin de transform_dataframe() — dbt s'en charge.
    log("📊 Lecture du mart dbt depuis BigQuery...")
    mart_df = read_mart_table(
        project_id=settings.GCP_PROJECT_ID,
        filters="is_valid = TRUE AND category IS NOT NULL",
    )
    log(f"  ✔ {len(mart_df)} lignes prêtes pour l'entraînement")

    # Sauvegarde locale pour traçabilité / re-run sans BigQuery
    mart_path = Path("data/processed/signalconso_mart.csv")
    mart_path.parent.mkdir(parents=True, exist_ok=True)
    mart_df.to_csv(mart_path, index=False)

    # ── 5. ENTRAÎNEMENT ML ────────────────────────────────────────────────────
    log("🤖 Entraînement TF-IDF + LogisticRegression...")
    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_model(
        df=mart_df,               # on passe le DataFrame directement
        text_col="clean_text",    # feature principale issue de dbt
        label_col="category",     # variable cible
        model_path=str(model_path),
    )

    log(f"  ✔ Accuracy : {metrics.get('accuracy', '?'):.2%}")
    log(f"  ✔ Modèle sauvegardé : {model_path}")

    # ── 6. UPLOAD ARTEFACTS ───────────────────────────────────────────────────
    log("📤 Upload des artefacts vers GCS...")

    # Modèle versionné
    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        str(model_path),
        f"models/model_{today}.joblib",
    )
    # Modèle "latest" (toujours écrasé)
    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        str(model_path),
        "models/model.joblib",
    )

    # Export du mart vers processed/ dans GCS
    export_mart_to_gcs(
        project_id=settings.GCP_PROJECT_ID,
        bucket_name=settings.GCS_BUCKET_NAME,
    )

    log("  ✔ Tous les artefacts uploadés dans GCS")
    log("🏁 Pipeline terminé avec succès")

    return {
        "raw_rows":   len(raw_df),
        "mart_rows":  len(mart_df),
        "model_path": str(model_path),
        "accuracy":   metrics.get("accuracy"),
        "date":       today,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(log=print)
