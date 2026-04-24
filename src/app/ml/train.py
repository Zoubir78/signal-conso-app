from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Construit le pipeline sklearn : TF-IDF → LogisticRegression.

    Args:
        max_features: Taille maximale du vocabulaire TF-IDF.
        ngram_range:  Unigrammes + bigrammes par défaut.
        C:            Inverse de la régularisation LR.
        max_iter:     Nombre max d'itérations LR.

    Returns:
        Pipeline sklearn non entraîné.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # atténue les très hautes fréquences
            min_df=2,                # ignore les tokens trop rares
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced", # compense les classes déséquilibrées
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1,
        )),
    ])


def train_model(
    df: pd.DataFrame | None = None,
    data_path: str | None = None,
    text_col: str = "clean_text",
    label_col: str = "category",
    model_path: str = "models/model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    min_class_samples: int = 5,
) -> dict[str, Any]:
    """
    Entraîne le pipeline TF-IDF + LogisticRegression et sérialise le modèle.

    Accepte soit un DataFrame (depuis dbt/BigQuery), soit un chemin CSV (legacy).

    Args:
        df:                 DataFrame avec colonnes text_col et label_col.
        data_path:          Chemin CSV alternatif (si df est None).
        text_col:           Colonne de features texte (défaut : 'clean_text').
        label_col:          Colonne cible (défaut : 'category').
        model_path:         Chemin de sauvegarde du modèle.
        test_size:          Proportion du jeu de test.
        random_state:       Graine aléatoire pour reproductibilité.
        min_class_samples:  Supprime les classes avec moins de N exemples.

    Returns:
        dict avec accuracy, n_classes, n_train, n_test, report.
    """
    # ── Chargement des données ──────────────────────────────────────────────
    if df is None:
        if data_path is None:
            raise ValueError("Fournir 'df' ou 'data_path'.")
        df = pd.read_csv(data_path)

    # ── Validation des colonnes ─────────────────────────────────────────────
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Colonne absente du DataFrame : '{col}'")

    # ── Nettoyage minimal ───────────────────────────────────────────────────
    df = df[[text_col, label_col]].dropna()
    df = df[df[text_col].str.strip().str.len() > 0]

    # Supprime les classes trop rares (évite les erreurs de stratification)
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= min_class_samples].index
    dropped = class_counts[class_counts < min_class_samples]
    if not dropped.empty:
        print(f"Classes supprimées (< {min_class_samples} exemples) : {dropped.to_dict()}")
    df = df[df[label_col].isin(valid_classes)].reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Jeu de données trop petit après filtrage : {len(df)} lignes.")

    # ── Split train / test ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    # ── Entraînement ────────────────────────────────────────────────────────
    model = build_pipeline()
    model.fit(X_train, y_train)

    # ── Évaluation ──────────────────────────────────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Classes   : {len(valid_classes)}")
    print(f"Train     : {len(X_train)}  |  Test : {len(X_test)}")

    # ── Sérialisation ───────────────────────────────────────────────────────
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    return {
        "accuracy":  accuracy,
        "n_classes": len(valid_classes),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "report":    report,
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
        Liste de dicts : [{"category": ..., "confidence": ...}, ...]
    """
    model  = load_model(model_path)
    preds  = model.predict(texts)
    probas = model.predict_proba(texts).max(axis=1)

    return [
        {"category": cat, "confidence": round(float(prob), 4)}
        for cat, prob in zip(preds, probas)
    ]
