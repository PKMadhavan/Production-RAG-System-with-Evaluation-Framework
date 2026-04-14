"""Health check endpoint."""

from typing import Optional

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session, get_redis
from src.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    session: AsyncSession = Depends(get_db_session),
    redis: Optional[Redis] = Depends(get_redis),
) -> HealthResponse:
    """Check the health of the API and its dependencies."""
    pg_connected = False
    try:
        await session.execute(text("SELECT 1"))
        pg_connected = True
    except Exception:
        pass

    redis_connected = False
    if redis:
        try:
            await redis.ping()
            redis_connected = True
        except Exception:
            pass

    return HealthResponse(
        status="ok" if pg_connected else "degraded",
        version="0.1.0",
        postgres_connected=pg_connected,
        redis_connected=redis_connected,
    )
