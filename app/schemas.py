# app/schemas.py
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction_class: int
    confidence: float
    status: str
