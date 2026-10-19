# Pipeline entry points. Works with GNU make on Windows (Git Bash / chocolatey), macOS and Linux.
# Every target is one `python -m fraud.cli` call, so the CLI is the source of truth.
#   make all            full pipeline on the Kaggle data in data/raw/
#   make sample         full pipeline on generated synthetic data (no Kaggle account needed)

ifeq ($(OS),Windows_NT)
  PY ?= .venv/Scripts/python.exe
else
  PY ?= .venv/bin/python
endif
PROFILE ?= full
CLI = $(PY) -m fraud.cli
MODELS_DIR ?= models

.PHONY: setup data-check ingest features tune train evaluate policy explain backtest drift sql \
        report all sample notebooks dashboard api docker docker-run mlflow-ui test lint format clean-sample

setup:            ## create .venv and install dependencies + the package
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

data-check:       ## confirm the Kaggle CSVs are in data/raw/
	$(CLI) data-check --profile $(PROFILE)

ingest:           ## CSV -> DuckDB, LEFT JOIN identity
	$(CLI) ingest --profile $(PROFILE)

features:         ## SQL feature pipeline -> data/processed/features.parquet
	$(CLI) features --profile $(PROFILE)

tune:             ## Optuna search (writes configs/model_tuned.yaml)
	$(CLI) tune --profile $(PROFILE)

train:            ## rules, LR, LightGBM variants + ablations; calibrate; score
	$(CLI) train --profile $(PROFILE)

evaluate:         ## metrics table, bootstrap CIs, PR + calibration charts
	$(CLI) evaluate --profile $(PROFILE)

policy:           ## approve/review/decline policies + sensitivity grid
	$(CLI) policy --profile $(PROFILE)

explain:          ## TreeSHAP attributions + reason codes
	$(CLI) explain --profile $(PROFILE)

backtest:         ## rolling-origin backtest
	$(CLI) backtest --profile $(PROFILE)

drift:            ## PSI drift report
	$(CLI) drift --profile $(PROFILE)

sql:              ## analyst queries in sql/analysis/ -> reports/sql/
	$(CLI) sql --profile $(PROFILE)

report: evaluate policy explain backtest drift sql writeup

writeup:          ## refresh generated blocks in README / memo / model card
	$(CLI) writeup

all:              ## everything, in order
	$(CLI) all --profile $(PROFILE)

sample:           ## everything on synthetic data (fast, no Kaggle data needed)
	$(CLI) synth
	$(CLI) all --profile sample --n-trials 5

notebooks:        ## execute notebooks in place so GitHub renders their outputs
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

dashboard:        ## Streamlit policy simulator
	FRAUD_PROFILE=$(PROFILE) $(PY) -m streamlit run app/dashboard.py

api:              ## scoring API on http://localhost:8000 (docs at /docs)
	FRAUD_MODELS_DIR=$(MODELS_DIR) $(PY) -m uvicorn api.main:app --port 8000

docker:           ## build the scoring image from $(MODELS_DIR)
	docker build --build-arg MODELS_DIR=$(MODELS_DIR) -t fraud-api .

docker-run:
	docker run --rm -p 8000:8000 fraud-api

mlflow-ui:        ## browse experiment runs
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

clean-sample:
	rm -rf data/processed/sample models/sample reports/sample
