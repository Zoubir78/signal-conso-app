from __future__ import annotations

import json
import time
from pathlib import Path

from google.api_core import retry as google_retry
from google.cloud import storage


def get_client() -> storage.Client:
    return storage.Client()


# ── Retry policy ─────────────────────────────────────────────────────────────
# Réessaie sur les erreurs transitoires SSL / réseau (max 3 fois, backoff 2s)
_RETRY = google_retry.Retry(
    initial=2.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=120.0,
)

# Seuil en octets sous lequel on force l'upload simple (non resumable)
# GCS utilise le resumable au-dessus de 8 MB par défaut
_MULTIPART_THRESHOLD = 8 * 1024 * 1024  # 8 MB


def upload_file_to_gcs(
    bucket_name: str,
    local_path: str,
    blob_name: str,
    max_attempts: int = 3,
) -> None:
    """
    Upload un fichier local vers GCS avec retry automatique sur erreur SSL/réseau.

    Pour les fichiers < 8 MB (joblib, JSON, CSV légers) : upload multipart simple.
    Pour les fichiers >= 8 MB : upload resumable (comportement GCS par défaut).

    Args:
        bucket_name:  Nom du bucket GCS.
        local_path:   Chemin local du fichier à uploader.
        blob_name:    Nom du blob de destination dans le bucket.
        max_attempts: Nombre de tentatives avant abandon.
    """
    file_size = Path(local_path).stat().st_size

    for attempt in range(1, max_attempts + 1):
        try:
            client = get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if file_size < _MULTIPART_THRESHOLD:
                # Upload simple (évite le resumable et ses problèmes SSL)
                with open(local_path, "rb") as f:
                    blob.upload_from_file(
                        f,
                        size=file_size,
                        retry=_RETRY,
                    )
            else:
                # Upload resumable pour les gros fichiers
                blob.upload_from_filename(local_path, retry=_RETRY)

            return  # succès

        except Exception as e:
            if attempt < max_attempts:
                wait = 2**attempt
                print(f"  ⚠ Upload GCS échoué (tentative {attempt}/{max_attempts}) : {e}")
                print(f"  ↻ Nouvelle tentative dans {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Upload GCS abandonné après {max_attempts} tentatives : {local_path} → {blob_name}"
                ) from e


def upload_json_to_gcs(bucket_name: str, blob_name: str, data: dict) -> None:
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False),
        content_type="application/json",
        retry=_RETRY,
    )


def download_json_from_gcs(bucket_name: str, blob_name: str) -> dict | None:
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    return json.loads(blob.download_as_text())


def find_prediction_in_bucket(bucket_name: str, prediction_id: str) -> dict | None:
    client = get_client()
    for blob in client.list_blobs(bucket_name, prefix="predictions/"):
        if prediction_id in blob.name:
            return json.loads(blob.download_as_text())
    return None


def get_latest_blob(bucket_name: str, prefix: str) -> str | None:
    client = get_client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        return None
    return max(blobs, key=lambda b: b.updated).name


def download_blob_to_file(
    bucket_name: str,
    blob_name: str,
    destination_path: str,
) -> None:
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path, retry=_RETRY)
