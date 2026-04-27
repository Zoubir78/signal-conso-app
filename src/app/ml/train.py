from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Catalogue des modèles disponibles ────────────────────────────────────────
# Chaque entrée est un callable qui retourne un estimator sklearn non entraîné.
# TfidfVectorizer est partagé et ajouté automatiquement dans build_pipeline().

AVAILABLE_MODELS: dict[str, Any] = {
    "logreg": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        n_jobs=-1,
    ),
    "sgd": SGDClassifier(
        loss="modified_huber",  # supporte predict_proba
        max_iter=200,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "linearsvc": CalibratedClassifierCV(
        LinearSVC(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
        ),
        cv=3,
    ),
    "complementnb": ComplementNB(alpha=0.1),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
}


def _tfidf() -> TfidfVectorizer:
    """Retourne un TfidfVectorizer partagé par tous les pipelines."""
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )


def build_pipeline(model_name: str) -> Pipeline:
    """
    Construit le pipeline sklearn : TF-IDF → classificateur.

    Args:
        model_name: Clé dans AVAILABLE_MODELS.

    Returns:
        Pipeline sklearn non entraîné.

    Raises:
        ValueError: Si model_name est inconnu.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle inconnu : '{model_name}'. Disponibles : {list(AVAILABLE_MODELS)}")
    return Pipeline(
        [
            ("tfidf", _tfidf()),
            ("clf", AVAILABLE_MODELS[model_name]),
        ]
    )


def train_model(
    df: pd.DataFrame | None = None,
    data_path: str | None = None,
    text_col: str = "clean_text",
    label_col: str = "category",
    model_name: str = "logreg",
    model_path: str = "models/model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    min_class_samples: int = 5,
) -> dict[str, Any]:
    """
    Entraîne un pipeline TF-IDF + classificateur et le sérialise.

    Accepte un DataFrame (depuis dbt/BigQuery) ou un chemin CSV (legacy).

    Args:
        df:                 DataFrame avec colonnes text_col et label_col.
        data_path:          Chemin CSV alternatif si df est None.
        text_col:           Colonne de features (défaut : 'clean_text').
        label_col:          Colonne cible (défaut : 'category').
        model_name:         Clé dans AVAILABLE_MODELS (défaut : 'logreg').
        model_path:         Chemin de sauvegarde du modèle sérialisé.
        test_size:          Proportion jeu de test.
        random_state:       Graine aléatoire.
        min_class_samples:  Supprime les classes avec moins de N exemples.

    Returns:
        dict : model_name, accuracy, f1_macro, n_classes, n_train, n_test, report.
    """
    # ── Chargement ──────────────────────────────────────────────────────────
    if df is None:
        if data_path is None:
            raise ValueError("Fournir 'df' ou 'data_path'.")
        df = pd.read_csv(data_path)

    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Colonne absente : '{col}'")

    # ── Nettoyage ────────────────────────────────────────────────────────────
    df = df[[text_col, label_col]].dropna()
    df = df[df[text_col].str.strip().str.len() > 0]

    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= min_class_samples].index
    dropped = class_counts[class_counts < min_class_samples]
    if not dropped.empty:
        print(f"  Classes supprimées (< {min_class_samples} ex.) : {dropped.to_dict()}")
    df = df[df[label_col].isin(valid_classes)].reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Jeu de données trop petit : {len(df)} lignes.")

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    # ── Entraînement ─────────────────────────────────────────────────────────
    model = build_pipeline(model_name)
    model.fit(X_train, y_train)

    # ── Évaluation ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ── Sérialisation ─────────────────────────────────────────────────────────
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "n_classes": len(valid_classes),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "report": report,
        "model_path": model_path,
    }


def load_model(model_path: str = "models/model.joblib") -> Pipeline:
    """Charge un modèle sérialisé depuis le disque."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    return joblib.load(path)


def predict(texts: list[str], model_path: str = "models/model.joblib") -> list[dict]:
    """
    Prédit la catégorie et la probabilité maximale pour une liste de textes.

    Returns:
        [{"category": ..., "confidence": ...}, ...]
    """
    model = load_model(model_path)
    preds = model.predict(texts)
    probas = model.predict_proba(texts).max(axis=1)
    return [
        {"category": cat, "confidence": round(float(prob), 4)}
        for cat, prob in zip(preds, probas, strict=False)
    ]
