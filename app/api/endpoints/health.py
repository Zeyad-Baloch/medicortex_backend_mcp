"""
Health check and status endpoints.
"""
from fastapi import APIRouter
from app.api.schemas import HealthCheckResponse
from app.dependencies import get_storage
from app.config import settings


router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint - verify service is running.
    """
    storage = get_storage()
    users = await storage.list_users()
    
    # Count total models (rough estimate)
    total_models = len(users) * 5  # 5 metrics per user
    
    return {
        "status": "healthy",
        "users_with_baselines": len(users),
        "total_models": total_models,
        "storage_type": settings.STORAGE_TYPE
    }
