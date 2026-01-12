"""Health check endpoint."""

from datetime import datetime
from fastapi import APIRouter, Response
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for container orchestration.

    Returns 200 OK if the service is healthy.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/ready")
async def readiness_check(response: Response) -> dict:
    """
    Readiness check endpoint.

    Checks if the service is ready to accept traffic.
    """
    # In the future, could check database connectivity, exchange API, etc.
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat(),
    }
