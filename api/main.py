"""FastAPI scoring service.

    uvicorn api.main:app --port 8000            (models from models/)
    FRAUD_MODELS_DIR=models/sample uvicorn ...  (synthetic-sample model)

POST /score returns the calibrated fraud probability, the approve / review / decline
decision under the configured costs, and the top reason codes.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import ScoreRequest, ScoreResponse
from fraud.config import ROOT
from fraud.serving import Scorer

STATE: dict = {}


def models_dir() -> Path:
    return Path(os.environ.get("FRAUD_MODELS_DIR", ROOT / "models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["scorer"] = Scorer(models_dir())
    yield
    STATE.clear()


app = FastAPI(title="Fraud scoring API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    s: Scorer = STATE["scorer"]
    return {"status": "ok", "model_version": s.version, "n_features": len(s.features)}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> dict:
    s: Scorer = STATE["scorer"]
    txn = req.transaction.model_dump(exclude_none=True)
    try:
        return s.score(txn, req.costs.model_dump() if req.costs else None)
    except Exception as e:  # surface feature-pipeline problems as a 422, not a 500 stack trace
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}") from e
