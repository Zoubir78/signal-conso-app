FROM python:3.11-slim

# ── Variables d'environnement ────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    # dbt lit profiles.yml dans dbt/ du projet (pas dans ~/.dbt)
    DBT_PROFILES_DIR=/app/dbt \
    # pip : pas de cache, pas d'avertissement de version
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dépendances système ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python (couche cachée séparément du code source) ─────────────
# Copier uniquement les fichiers de dépendances en premier pour profiter
# du cache Docker — cette couche ne sera reconstruite que si setup.py/
# pyproject.toml/requirements.txt changent.
COPY setup.py* pyproject.toml* requirements*.txt ./
RUN pip install --upgrade pip \
    && pip install dbt-bigquery \
    && pip install -e ".[dev]" 2>/dev/null || pip install -r requirements.txt

# ── Code source ──────────────────────────────────────────────────────────────
COPY . .

# ── Dossiers runtime (modèles + données intermédiaires) ──────────────────────
RUN mkdir -p /app/models /app/data/raw /app/data/processed

EXPOSE 8501

CMD ["streamlit", "run", "src/app/dashboard/dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]