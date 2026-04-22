# Signal-Conso-App

## Plateforme intelligente de tri et de priorisation des demandes clients

Plateforme IA de tri automatique de demandes clients, avec pipeline de données, API REST, modèle NLP et dashboard de suivi.

## Objectif

Le projet vise à :
- collecter et préparer des demandes clients ;
- les classer automatiquement par catégorie ;
- estimer leur priorité ;
- exposer les prédictions via une API ;
- suivre le pipeline et les performances dans une logique MLOps.

## Stack technique

- Python 3.11+
- FastAPI
- Streamlit
- PostgreSQL
- scikit-learn
- pandas
- SQLAlchemy
- Docker
- Google Cloud Storage

## Données

Le dépôt ne versionne pas les gros fichiers de données.

- `data/raw/` : données brutes locales ou temporaires
- `data/samples/` : échantillon réduit pour test
- `data/processed/` : données nettoyées
- `clean_complaints/` (GCS) : stockage des datasets, modèles et prédictions

Le fichier volumineux `complaints.csv` est chargé via un pipeline d’ingestion vers une table de staging PostgreSQL, puis transformé pour l’entraînement et l’inférence.

## Architecture

- **Ingestion** : lecture du CSV, contrôle qualité, chargement en staging
- **Transformation** : nettoyage, normalisation, création du texte d’entrée
- **ML** : entraînement et prédiction de catégories
- **API** : exposition des prédictions
- **Dashboard** : visualisation, évaluation et monitoring

## Démarrage rapide

```bash
git clone <repo-url>
cd plateforme-tri-demandes-clients
cp .env.example .env
pip install -r requirements.txt
uvicorn src.app.main:app --reload