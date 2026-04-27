from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

from src.app.core.config import get_settings
from src.app.ingestion.extract import extract_from_signalconso_api
from src.app.ml.train import AVAILABLE_MODELS, train_model
from src.app.services.bigquery_service import export_mart_to_gcs, read_mart_table
from src.app.services.gcs_service import upload_file_to_gcs, upload_json_to_gcs

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"

GCS_RAW_PREFIX = "raw/"
ROOT_DIR = Path(__file__).resolve().parents[1]
DBT_PROJECT_DIR = ROOT_DIR / "dbt"
DBT_TARGET = "dev"
DBT_PROFILES_DIR = DBT_PROJECT_DIR

# Modèles à entraîner — tous par défaut, peut être réduit pour accélérer
MODELS_TO_TRAIN = list(AVAILABLE_MODELS.keys())


def _run_dbt(log, target: str = DBT_TARGET) -> None:
    project_dir = str(DBT_PROJECT_DIR.resolve())
    profiles_dir = str(DBT_PROFILES_DIR.resolve())

    env = os.environ.copy()
    env["DBT_PROFILES_DIR"] = profiles_dir

    cmd = [
        "dbt",
        "run",
        "--profiles-dir",
        profiles_dir,
        "--target",
        target,
    ]

    log(f"  ▶ Commande: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log("----- STDOUT -----")
    log(result.stdout or "(vide)")
    log("----- STDERR -----")
    log(result.stderr or "(vide)")

    if result.returncode != 0:
        raise RuntimeError("dbt run a échoué")


def _print_leaderboard(results: list[dict], log) -> None:
    """Affiche un tableau comparatif des modèles triés par accuracy."""
    log("\n  ┌─────────────────────────┬──────────┬─────────┬────────┐")
    log("  │ Modèle                  │ Accuracy │  Train  │  Test  │")
    log("  ├─────────────────────────┼──────────┼─────────┼────────┤")
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        status = "🏆" if r == results[0] else "  "
        log(
            f"  │ {status}{r['model_name']:<22} │"
            f"  {r['accuracy']:.2%}  │"
            f" {r['n_train']:>6}  │"
            f" {r['n_test']:>5}  │"
        )
    log("  └─────────────────────────┴──────────┴─────────┴────────┘\n")


def run_pipeline(log) -> dict:
    """
    Pipeline complet avec entraînement multi-modèles et sélection du meilleur.

    Flux :
      1. Extract API   → DataFrame brut
      2. Upload GCS    → raw/  (table externe BigQuery lit automatiquement)
      3. dbt run       → staging → intermediate → mart_signalconso
      4. Read mart     → DataFrame prêt pour le ML
      5. Train ALL     → entraîne N modèles en parallèle sur le même split
      6. Evaluate      → leaderboard et sélection du meilleur
      7. Upload        → meilleur modèle → GCS models/ + rapport JSON

    Args:
        log: callable de logging (st.write en Streamlit, print en CLI).

    Returns:
        dict avec les métriques du meilleur modèle et le leaderboard complet.
    """
    settings = get_settings()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    log("🚀 Démarrage pipeline SignalConso")

    # ── 1. EXTRACT ────────────────────────────────────────────────────────────
    log("📥 Extraction API SignalConso...")
    raw_df = extract_from_signalconso_api(API_URL, limit=10_000)
    log(f"  ✔ {len(raw_df)} enregistrements extraits")

    # ── 2. UPLOAD GCS ─────────────────────────────────────────────────────────
    raw_path = Path("data/raw/signalconso.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    gcs_blob = f"{GCS_RAW_PREFIX}signalconso_{today}.csv"
    upload_file_to_gcs(settings.GCS_BUCKET_NAME, str(raw_path), gcs_blob)
    log(f"  ✔ RAW uploadé → gs://{settings.GCS_BUCKET_NAME}/{gcs_blob}")

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

    # ── 5. ENTRAÎNEMENT MULTI-MODÈLES ─────────────────────────────────────────
    log(f"🤖 Entraînement de {len(MODELS_TO_TRAIN)} modèles...")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for model_name in MODELS_TO_TRAIN:
        log(f"  ▶ {model_name}...")
        model_path = models_dir / f"{model_name}.joblib"

        try:
            metrics = train_model(
                df=mart_df,
                text_col="clean_text",
                label_col="category",
                model_name=model_name,
                model_path=str(model_path),
            )
            all_results.append(metrics)
            log(
                f"    ✔ Accuracy : {metrics['accuracy']:.2%}  "
                f"| F1-macro : {metrics.get('f1_macro', 0):.2%}"
                f"  ({metrics['n_train']} train / {metrics['n_test']} test)"
            )

        except Exception as e:
            log(f"    ✖ Échec : {e}")

    if not all_results:
        raise RuntimeError("Aucun modèle entraîné avec succès.")

    # ── 6. ÉVALUATION & SÉLECTION ─────────────────────────────────────────────
    log("\n📊 Leaderboard des modèles :")
    all_results.sort(key=lambda r: r["accuracy"], reverse=True)
    _print_leaderboard(all_results, log)

    best = all_results[0]
    log(
        f"🏆 Meilleur modèle : {best['model_name']}  "
        f"(accuracy={best['accuracy']:.2%}, f1-macro={best.get('f1_macro', 0):.2%})"
    )

    # ── 7. UPLOAD ARTEFACTS ───────────────────────────────────────────────────
    log("📤 Upload des artefacts vers GCS...")

    # On ne garde que les 2 meilleurs modèles
    top2_results = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)[:2]

    # Upload des 2 meilleurs modèles dans models/runs/<date>/
    for r in top2_results:
        local = models_dir / f"{r['model_name']}.joblib"
        if local.exists():
            try:
                upload_file_to_gcs(
                    settings.GCS_BUCKET_NAME,
                    str(local),
                    f"models/runs/{today}/{r['model_name']}.joblib",
                )
                log(f"  ✔ Uploaded {r['model_name']}")
            except Exception as e:
                log(f"  ⚠ Upload échoué ({r['model_name']}) : {e}")

    # Meilleur modèle → latest
    best = top2_results[0]
    best_local = models_dir / f"{best['model_name']}.joblib"

    try:
        upload_file_to_gcs(
            settings.GCS_BUCKET_NAME,
            str(best_local),
            "models/model.joblib",
        )
        upload_file_to_gcs(
            settings.GCS_BUCKET_NAME,
            str(best_local),
            f"models/model_{today}.joblib",
        )
        log("  ✔ Best model uploadé")
    except Exception as e:
        log(f"  ⚠ Upload best model échoué : {e}")

    # Rapport JSON avec seulement les 2 meilleurs modèles
    report = {
        "date": today,
        "best_model": best["model_name"],
        "leaderboard": [
            {
                "model": r["model_name"],
                "accuracy": round(r["accuracy"], 4),
                "f1_macro": round(r.get("f1_macro", 0), 4),
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            }
            for r in top2_results
        ],
    }

    try:
        upload_json_to_gcs(
            settings.GCS_BUCKET_NAME,
            f"models/runs/{today}/evaluation_report.json",
            report,
        )
        upload_json_to_gcs(
            settings.GCS_BUCKET_NAME,
            "models/evaluation_report.json",
            report,
        )
        log("  ✔ Report JSON uploadé")
    except Exception as e:
        log(f"  ⚠ Upload report échoué : {e}")

    # Export mart → processed/
    try:
        export_mart_to_gcs(
            project_id=settings.GCP_PROJECT_ID,
            bucket_name=settings.GCS_BUCKET_NAME,
        )
        log("  ✔ Mart exporté vers GCS")
    except Exception as e:
        log(f"  ⚠ Export mart échoué : {e}")

    log("🏁 Pipeline terminé avec succès")

    return {
        "raw_rows": len(raw_df),
        "mart_rows": len(mart_df),
        "best_model": best["model_name"],
        "accuracy": best["accuracy"],
        "f1_macro": best.get("f1_macro"),
        "n_classes": best["n_classes"],
        "leaderboard": report["leaderboard"],
        "date": today,
    }


if __name__ == "__main__":
    run_pipeline(log=print)
