"""Health endpoints — no auth, used by uptime monitors and smoke tests."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from api.main import API_VERSION

    return HealthResponse(
        status="ok",
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc),
    )
