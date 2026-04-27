from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
from google.cloud import storage


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TicketModel:
    def __init__(
        self, model_path: str, bucket_name: str | None = None, blob_name: str | None = None
    ):
        self.model_path = model_path
        self.bucket_name = bucket_name
        self.blob_name = blob_name
        self.model = None

    def download_from_gcs(self) -> None:
        if not self.bucket_name or not self.blob_name:
            raise FileNotFoundError(f"Modèle introuvable localement : {self.model_path}")

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self.blob_name)

        if not blob.exists():
            raise FileNotFoundError(
                f"Modèle introuvable dans GCS : gs://{self.bucket_name}/{self.blob_name}"
            )

        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(path))

    def load(self) -> None:
        path = Path(self.model_path)

        if not path.exists():
            self.download_from_gcs()

        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.model = joblib.load(path)

    def predict(self, text: str) -> str:
        if self.model is None:
            self.load()

        clean_text = normalize_text(text)
        if not clean_text:
            raise ValueError("Le texte d'entrée est vide ou invalide.")

        return self.model.predict([clean_text])[0]

    def predict_many(self, texts: list[str]) -> list[str]:
        if self.model is None:
            self.load()

        clean_texts = [normalize_text(text) for text in texts]
        clean_texts = [text for text in clean_texts if text]

        if not clean_texts:
            raise ValueError("Aucun texte valide à prédire.")

        return list(self.model.predict(clean_texts))

    def predict_with_proba(self, text: str):
        if self.model is None:
            self.load()

        clean_text = normalize_text(text)
        if not clean_text:
            raise ValueError("Le texte d'entrée est vide ou invalide.")

        probs = self.model.predict_proba([clean_text])[0]
        idx = probs.argmax()

        return self.model.classes_[idx], float(probs[idx])
