"""
Anomaly detection endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    AnomalyDetectRequest,
    AnomalyResponse
)
from app.services.anomaly import AnomalyService
from app.dependencies import get_anomaly_service


router = APIRouter()


@router.post("/detect", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyDetectRequest):
    """
    Detect health anomalies based on user's learned baseline.
    
    User must have a trained baseline first.
    """
    try:
        service = get_anomaly_service()
        
        # Convert Pydantic models to dicts
        data_dicts = [point.model_dump() for point in request.data]
        
        result = await service.detect_anomalies(
            user_id=request.user_id,
            data=data_dicts
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
