"""Shared FastAPI dependencies (auth, geo, pagination).

Stubs for now — wire to Supabase Auth + MaxMind in a follow-up commit.
"""

from __future__ import annotations

from fastapi import HTTPException


def get_current_user_stub() -> dict:
    """Placeholder user dependency. Replace with Supabase JWT verification."""
    raise HTTPException(
        status_code=501,
        detail={
            "error": {
                "code": "AUTH_NOT_WIRED",
                "message": "Authentication is not yet wired. Coming in a follow-up commit.",
            }
        },
    )


def require_pro_stub() -> dict:
    """Placeholder Pro-tier gate. Replace with subscription_tier check."""
    raise HTTPException(
        status_code=501,
        detail={
            "error": {
                "code": "AUTH_NOT_WIRED",
                "message": "Pro-tier checks not yet wired.",
            }
        },
    )
