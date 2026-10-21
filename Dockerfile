# Scoring API image. Build after training so models/ holds the artifacts:
#   make docker                         (real-data model in models/)
#   make docker MODELS_DIR=models/sample (synthetic-sample model, works without Kaggle data)
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY configs/ configs/
COPY sql/ sql/
COPY api/ api/

ARG MODELS_DIR=models
COPY ${MODELS_DIR}/metadata.json ${MODELS_DIR}/policy.json ${MODELS_DIR}/serving.duckdb models/
COPY ${MODELS_DIR}/*.txt ${MODELS_DIR}/*.pkl models/

ENV FRAUD_ROOT=/app FRAUD_MODELS_DIR=/app/models FRAUD_MLFLOW=0 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
