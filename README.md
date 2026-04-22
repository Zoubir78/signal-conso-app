# Signal-Conso-App

## Plateforme intelligente de tri et de priorisation des demandes clients

Plateforme IA de tri automatique de demandes clients, avec pipeline de données, stockage analytique, API de prédiction et dashboard de suivi.

## Objectif

Le projet vise à :
- collecter et préparer des demandes clients ;
- les classer automatiquement par catégorie ;
- estimer leur priorité ;
- exposer les prédictions via une API ;
- suivre le pipeline et les performances dans une logique data / MLOps.

## Stack

| Catégorie | Outil |
|---|---|
| Versionning & CI/CD | GitHub + GitHub Actions |
| Qualité code | Ruff + SQLFluff + pre-commit |
| Containerisation | Docker + docker-compose |
| Infrastructure | GCP VM e2-small + GCS |
| Stockage & transformation | BigQuery + dbt-bigquery |
| Orchestration | Prefect Cloud |
| API ML Predictions | FastAPI |

## Architecture

- **Ingestion** : lecture des données et contrôle qualité
- **Transformation** : normalisation, nettoyage, préparation des jeux de données
- **Stockage analytique** : BigQuery
- **Modélisation** : entraînement et prédiction de catégories
- **API** : exposition des prédictions via FastAPI
- **Orchestration** : pipelines automatisés avec Prefect Cloud
- **Dashboard** : visualisation, évaluation et monitoring

## Setup

### Environnement Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Only if you don't have .env yet
cp -n .env.example .env
```

### dbt (Silver / Gold)

```bash
# Authenticate with GCP
gcloud auth application-default login

# Set up your local dbt profile
cp dbt/profiles.yml.example ~/.dbt/profiles.yml

# Verify connection
cd dbt && dbt debug

# Run Silver models
dbt run

# Run tests
dbt test
```

### Lancer l'API

```bash
uvicorn src.app.main:app --reload
```

### Lancer le dashboard Streamlit

```bash
streamlit run src/app/streamlit_dashboard.py
```

### Docker

```bash
docker compose up --build
```

## Données

Le dépôt ne versionne pas les gros fichiers de données.

- `data/raw/` : données brutes locales ou temporaires
- `data/samples/` : échantillon réduit pour test
- `data/processed/` : données nettoyées
- GCS : stockage des datasets, modèles et prédictions

Le fichier volumineux `complaints.csv` est chargé via un pipeline d’ingestion vers une table de staging, puis transformé pour l’entraînement et l’inférence.

## Pipeline principal

1. extraction des données
2. nettoyage et transformation
3. chargement dans BigQuery
4. modélisation via dbt
5. entraînement et évaluation du modèle
6. exposition via API
7. visualisation dans Streamlit

## Endpoints principaux

- `GET /health`
- `POST /predictions`
- `GET /predictions/{id}`

## Livrables attendus

- pipeline d’ingestion
- modèle IA
- API REST
- dashboard
- tests automatisés
- déploiement Docker
- orchestration Prefect

