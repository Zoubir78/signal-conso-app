from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.app.core.config import get_settings
from src.app.ml.predict import TicketModel

router = APIRouter()

settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# Le best model est celui uploadé par le pipeline dans models/model.joblib
model = TicketModel(
    model_path=str(MODEL_PATH),
    bucket_name=settings.GCS_BUCKET_NAME,
    blob_name="models/model.joblib",
)


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    predicted_category: str
    confidence: float


@router.post("/predict", response_model=TicketResponse)
def predict_ticket(request: TicketRequest):
    try:
        if model.model is None:
            model.load()

        category, confidence = model.predict_with_proba(request.text)

        return TicketResponse(
            predicted_category=category,
            confidence=confidence,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
