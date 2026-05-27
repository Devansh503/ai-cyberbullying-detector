from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AnalyzeRequest(BaseModel):
    text: str
    source_lang: Optional[str] = Field(default=None, description="ISO code of the source language if known")

class Explanation(BaseModel):
    token: str
    weight: float

class AnalyzeResponse(BaseModel):
    text: str
    prediction: str
    category: str
    severity: str
    confidence: float
    probs: list[float] | float
    explanations: list[Explanation]
    translated: str | None = None


class RephraseRequest(BaseModel):
    text: str


class RephraseResponse(BaseModel):
    original: str
    censored: str
    suggestion: str


class FeedbackRequest(BaseModel):
    text: str
    correct: bool
