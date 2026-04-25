from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from src.app.core.config import get_settings
from src.app.ingestion.extract import extract_from_signalconso_api
from src.app.services.bigquery_service import (
    read_mart_table,
    export_mart_to_gcs,
)
from src.app.services.gcs_service import upload_file_to_gcs
from src.app.ml.train import train_model

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"

# Préfixe GCS lu par la table externe Complaints.Signal_Conso
# Doit correspondre au pattern configuré dans la table externe BigQuery
GCS_RAW_PREFIX = "raw/"

DBT_PROJECT_DIR = Path(__file__).parent / "dbt"
DBT_TARGET      = "dev"


def _run_dbt(log, target: str = DBT_TARGET) -> None:
    """Lance dbt run + dbt test et lève une exception en cas d'échec."""
    base = [
        "dbt",
        "--project-dir", str(DBT_PROJECT_DIR),
        "--profiles-dir", str(Path.home() / ".dbt"),
        "--target", target,
    ]

    log("  dbt run...")
    result = subprocess.run(base + ["run"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        log(f"  ✖ dbt run échoué :\n{output[-3000:]}")
        raise RuntimeError("dbt run a échoué")
    log(result.stdout.strip()[-500:])

    log("  dbt test...")
    result = subprocess.run(base + ["test"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        log(f"  ⚠ dbt test en échec (pipeline continue) :\n{output[-1000:]}")


def run_pipeline(log) -> dict:
    """
    Pipeline complet avec table externe BigQuery.

    Flux :
      1. Extract API   → DataFrame brut
      2. Upload GCS    → raw/signalconso_YYYY-MM-DD.csv
                         (la table externe Complaints.Signal_Conso lit ce fichier
                          automatiquement — pas besoin d'écrire dans BigQuery)
      3. dbt run       → staging → intermediate → mart_signalconso
      4. Read mart     → DataFrame prêt pour le ML (depuis BigQuery)
      5. Train         → TF-IDF + LogisticRegression → model.joblib
      6. Upload        → GCS models/ + export mart processed/

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

    # ── 2. UPLOAD GCS (= "Load" pour la table externe) ───────────────────────
    # Complaints.Signal_Conso est une table EXTERNE qui lit gs://clean_complaints/raw/*.csv
    # Uploader le fichier ici suffit — BigQuery le voit instantanément.
    # Aucun appel à upload_dataframe_to_bigquery n'est nécessaire.
    raw_path = Path("data/raw/signalconso.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    gcs_blob = f"{GCS_RAW_PREFIX}signalconso_{today}.csv"
    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        str(raw_path),
        gcs_blob,
    )
    log(f"  ✔ RAW uploadé → gs://{settings.GCS_BUCKET_NAME}/{gcs_blob}")
    log("  ℹ La table externe BigQuery lit ce fichier automatiquement")

    # ── 3. DBT ────────────────────────────────────────────────────────────────
    log("🔧 Modélisation dbt (staging → intermediate → mart)...")
    _run_dbt(log, target=DBT_TARGET)
    log("  ✔ dbt run + test terminés")

    # ── 4. LECTURE DU MART ────────────────────────────────────────────────────
    log("📊 Lecture du mart dbt depuis BigQuery...")
    mart_df = read_mart_table(
        project_id=settings.GCP_PROJECT_ID,
        filters="is_valid = TRUE AND category IS NOT NULL",
    )
    log(f"  ✔ {len(mart_df)} lignes prêtes pour l'entraînement")

    mart_path = Path("data/processed/signalconso_mart.csv")
    mart_path.parent.mkdir(parents=True, exist_ok=True)
    mart_df.to_csv(mart_path, index=False)

    # ── 5. ENTRAÎNEMENT ML ────────────────────────────────────────────────────
    log("🤖 Entraînement TF-IDF + LogisticRegression...")
    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_model(
        df=mart_df,
        text_col="clean_text",
        label_col="category",
        model_path=str(model_path),
    )

    log(f"  ✔ Accuracy   : {metrics.get('accuracy', 0):.2%}")
    log(f"  ✔ Classes    : {metrics.get('n_classes')}")
    log(f"  ✔ Train / Test : {metrics.get('n_train')} / {metrics.get('n_test')}")

    # ── 6. UPLOAD ARTEFACTS ───────────────────────────────────────────────────
    log("📤 Upload des artefacts vers GCS...")

    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        str(model_path),
        f"models/model_{today}.joblib",   # version datée
    )
    upload_file_to_gcs(
        settings.GCS_BUCKET_NAME,
        str(model_path),
        "models/model.joblib",            # latest (toujours écrasé)
    )

    export_mart_to_gcs(
        project_id=settings.GCP_PROJECT_ID,
        bucket_name=settings.GCS_BUCKET_NAME,
    )

    log("  ✔ Artefacts uploadés dans GCS")
    log("🏁 Pipeline terminé avec succès")

    return {
        "raw_rows":   len(raw_df),
        "mart_rows":  len(mart_df),
        "model_path": str(model_path),
        "accuracy":   metrics.get("accuracy"),
        "n_classes":  metrics.get("n_classes"),
        "date":       today,
    }


if __name__ == "__main__":
    run_pipeline(log=print)