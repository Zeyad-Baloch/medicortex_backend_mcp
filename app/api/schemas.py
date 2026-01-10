"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================================
# HEALTH DATA MODELS
# ============================================================================

class HealthDataPoint(BaseModel):
    """Single health data point with all metrics."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    heart_rate: float = Field(..., ge=0, le=300, description="Heart rate in BPM")
    steps: int = Field(..., ge=0, description="Step count")
    sleep_quality: float = Field(..., ge=0, le=100, description="Sleep quality score (0-100)")
    stress_level: float = Field(..., ge=0, le=100, description="Stress level (0-100)")
    calories: float = Field(..., ge=0, description="Calories burned")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-01T14:30:00",
                "heart_rate": 75.0,
                "steps": 8000,
                "sleep_quality": 85.0,
                "stress_level": 35.0,
                "calories": 2000.0
            }
        }


# ============================================================================
# BASELINE MODELS
# ============================================================================

class BaselineTrainRequest(BaseModel):
    """Request to train a user's health baseline."""
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    data: List[HealthDataPoint] = Field(..., min_items=1, description="Historical health data")
    days: int = Field(7, ge=1, le=30, description="Number of days for training")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "days": 7,
                "data": [
                    {
                        "timestamp": "2025-01-01T00:00:00",
                        "heart_rate": 70.0,
                        "steps": 500,
                        "sleep_quality": 80.0,
                        "stress_level": 30.0,
                        "calories": 90.0
                    }
                ]
            }
        }


class BaselineStats(BaseModel):
    """Statistical baseline for a single metric."""
    mean: float
    std: float
    median: float
    p10: float
    p90: float
    recommended_lower: float
    recommended_upper: float
    range_type: str


class BaselineResponse(BaseModel):
    """Response after training a baseline."""
    status: str
    user_id: str
    baselines: Dict[str, BaselineStats]
    model_performance: Optional[Dict[str, Dict[str, float]]] = None
    message: str


# ============================================================================
# ANOMALY DETECTION MODELS
# ============================================================================

class AnomalyDetectRequest(BaseModel):
    """Request to detect health anomalies."""
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    data: List[HealthDataPoint] = Field(..., min_items=1, description="Health data to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "data": [
                    {
                        "timestamp": "2026-01-09T14:00:00",
                        "heart_rate": 145.0,
                        "steps": 5,
                        "sleep_quality": 0.0,
                        "stress_level": 95.0,
                        "calories": 15.0
                    }
                ]
            }
        }


class AnomalyAlert(BaseModel):
    """Single anomaly alert."""
    timestamp: str
    metric: str
    current_value: float
    your_normal: float
    deviation_pct: float
    risk_level: str
    confidence: float


class AnomalyResponse(BaseModel):
    """Response with detected anomalies."""
    status: str
    user_id: str
    total_anomalies: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    alerts: List[AnomalyAlert]


# ============================================================================
# HEALTH CHECK MODELS
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    users_with_baselines: int
    total_models: int
    storage_type: str
