# ==============================
# VARIABLES
# ==============================
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
STREAMLIT=$(VENV)/bin/streamlit
UVICORN=$(VENV)/bin/uvicorn

PROJECT=src.app.main:app

# ==============================
# ENV SETUP
# ==============================
venv:
	python -m venv $(VENV)

install:
	$(PIP) install -e ".[dev]"

setup: venv install
	@echo "✅ Environnement prêt"

activate:
	@echo "source $(VENV)/bin/activate"

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .ruff_cache

# ==============================
# RUN
# ==============================
api:
	$(UVICORN) $(PROJECT) --reload

dashboard:
	$(STREAMLIT) run src/app/streamlit_dashboard.py

# ==============================
# QUALITY (LINT + FORMAT)
# ==============================
lint:
	ruff check .

format:
	ruff format .

sql-lint:
	sqlfluff lint .

sql-fix:
	sqlfluff fix .

precommit:
	pre-commit run --all-files

# ==============================
# TESTS
# ==============================
test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

# ==============================
# DBT (BigQuery)
# ==============================
dbt-debug:
	cd dbt && dbt debug

dbt-run:
	cd dbt && dbt run

dbt-test:
	cd dbt && dbt test

dbt-build:
	cd dbt && dbt build

# ==============================
# GCP
# ==============================
gcp-auth:
	gcloud auth application-default login

gcp-project:
	gcloud config set project $$GCP_PROJECT

# ==============================
# PIPELINE
# ==============================
pipeline:
	$(PYTHON) scripts/run_pipeline.py

train:
	$(PYTHON) scripts/train_model.py

predict-batch:
	$(PYTHON) scripts/export_predictions.py

# ==============================
# FULL WORKFLOWS
# ==============================
run-all: setup lint test dbt-run train
	@echo "🚀 Full pipeline exécuté"

mlops: lint test dbt-build pipeline train
	@echo "🤖 Pipeline MLOps complet OK"

ci: lint test
	@echo "✅ CI OK"

# ==============================
# DOCKER
# ==============================
docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

# ==============================
# HELP
# ==============================
help:
	@echo "📌 Commandes disponibles :"
	@echo "make setup           → setup environnement"
	@echo "make api             → lancer FastAPI"
	@echo "make dashboard       → lancer Streamlit"
	@echo "make lint            → lint code"
	@echo "make test            → tests"
	@echo "make dbt-run         → dbt run"
	@echo "make pipeline        → run pipeline"
	@echo "make train           → entrainement modèle"
	@echo "make mlops           → pipeline complet"
