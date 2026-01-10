"""
Baseline training and retrieval endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    BaselineTrainRequest,
    BaselineResponse
)
from app.services.baseline import BaselineService
from app.dependencies import get_baseline_service


router = APIRouter()


@router.post("/train", response_model=BaselineResponse)
async def train_baseline(request: BaselineTrainRequest):
    """
    Train a personalized health baseline for a user.
    
    Requires 7 days of health data to learn normal patterns.
    """
    try:
        service = get_baseline_service()
        
        # Convert Pydantic models to dicts
        data_dicts = [point.model_dump() for point in request.data]
        
        result = await service.train_baseline(
            user_id=request.user_id,
            data=data_dicts,
            days=request.days
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_baseline(user_id: str):
    """
    Retrieve a user's learned health baseline.
    """
    service = get_baseline_service()
    result = await service.get_baseline(user_id)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result
