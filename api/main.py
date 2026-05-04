"""EdgeAcademy API entry point.

Run locally: `make api-run`
OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health

API_VERSION = "0.1.0"

app = FastAPI(
    title="EdgeAcademy API",
    description=(
        "Backend for EdgeAcademy — paper-betting Academy with collectible card "
        "mechanics + sportsbook affiliate funnel. See docs/api_design_v1.md."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


def _allowed_origins() -> list[str]:
    env = os.getenv("EDGEACADEMY_ENV", "dev")
    if env == "production":
        return [
            "https://lions-rater.streamlit.app",
            "https://edgeacademy.app",
            "https://www.edgeacademy.app",
        ]
    return [
        "https://lions-rater.streamlit.app",
        "http://localhost:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
