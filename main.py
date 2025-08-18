import os, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db, Base, engine
from models import Prediction

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
INDEX_PATH = MODELS_DIR / "index.json"

SPAM_THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.80"))

if not INDEX_PATH.exists():
    raise RuntimeError(f"Model index not found at {INDEX_PATH}. Run train.py first.")
with open(INDEX_PATH) as f:
    index = json.load(f)
AVAILABLE_MODELS = index.get("models", [])
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", index.get("default", AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None))
if not AVAILABLE_MODELS:
    raise RuntimeError("No models available. Run train.py.")

_model_cache = {}

def load_model(name: str):
    if name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'. Options: {AVAILABLE_MODELS}")
    if name not in _model_cache:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise HTTPException(status_code=500, detail=f"Model file missing: {path}")
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]

Base.metadata.create_all(bind=engine)

class PredictRequest(BaseModel):
    text: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    model: Optional[str] = None  # choose model per request

class PredictResponse(BaseModel):
    id: int
    label: bool
    probability: float
    model_name: str

app = FastAPI(title="Spam API", version="0.2.0")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": SPAM_THRESHOLD,
        "default_model": DEFAULT_MODEL_NAME,
        "available_models": AVAILABLE_MODELS,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    model_name = (req.model or DEFAULT_MODEL_NAME)
    mdl = load_model(model_name)
    
    try:
        proba = float(mdl.predict_proba([req.text])[0][1])
    except AttributeError:
        score = float(mdl.decision_function([req.text])[0])
        proba = float(1.0 / (1.0 + np.exp(-score)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    label = proba >= SPAM_THRESHOLD

    rec = Prediction(
        sender=req.sender,
        subject=req.subject,
        text=req.text,
        label=label,
        probability=proba,
        model_name=model_name,
        created_at=datetime.now(timezone.utc),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return PredictResponse(
        id=rec.id,
        label=rec.label,
        probability=rec.probability,
        model_name=rec.model_name,
    )

@app.get("/messages")
def messages(
    limit: int = Query(50, ge=1, le=500),
    is_spam: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
):
    from sqlalchemy import select, desc
    q = select(Prediction).order_by(desc(Prediction.created_at)).limit(min(limit, 500))
    if is_spam is not None:
        q = q.where(Prediction.label == is_spam)
    rows = db.execute(q).scalars().all()
    return [
        {
            "id": r.id,
            "sender": r.sender,
            "subject": r.subject,
            "text": r.text,
            "label": r.label,
            "probability": r.probability,
            "model_name": r.model_name,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
