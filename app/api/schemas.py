"""
Pydantic models for API request/response validation.
All schemas in one place for easy reference.
"""
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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


# ============================================================================
# OCR MODELS
# ============================================================================



class ExtractedValue(BaseModel):
    """Single extracted health metric value."""
    value: Optional[float] = None
    unit: Optional[str] = None
    raw: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "value": 120.0,
                "unit": "mg/dL",
                "raw": "glucose: 120 mg/dL"
            }
        }


class BloodPressureValue(BaseModel):
    """Blood pressure with systolic/diastolic."""
    systolic: float
    diastolic: float
    unit: str = "mmHg"
    formatted: str

    class Config:
        json_schema_extra = {
            "example": {
                "systolic": 130.0,
                "diastolic": 85.0,
                "unit": "mmHg",
                "formatted": "130/85"
            }
        }


class HealthAlert(BaseModel):
    """Alert for abnormal health values."""
    metric: str
    value: Any
    severity: str = Field(..., description="low, medium, or high")
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "metric": "glucose",
                "value": 145.0,
                "severity": "high",
                "message": "Glucose elevated (145 vs normal 70-100 mg/dL)"
            }
        }


class PatientInfo(BaseModel):
    """Extracted patient demographic information."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    report_date: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "report_date": "January 12, 2026"
            }
        }


class OCRExtractResponse(BaseModel):
    """Response after extracting text from medical report - NOW WITH PARSING!"""
    status: str
    report_id: str
    user_id: str
    extracted_text: str
    confidence: float = Field(..., ge=0, le=1, description="OCR confidence score")
    report_type: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    created_at: str

    # NEW - Parsed data fields
    parsed_values: Dict[str, Any] = Field(default_factory=dict, description="Structured health values")
    patient_info: Dict[str, Any] = Field(default_factory=dict, description="Patient demographics")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Health alerts for abnormal values")
    metrics_found: int = Field(0, description="Number of health metrics extracted")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "report_id": "report_abc123xyz",
                "user_id": "user_123",
                "extracted_text": "Patient: John Doe\nGlucose: 120 mg/dL...",
                "confidence": 0.95,
                "report_type": "lab_report",
                "keywords": ["glucose", "blood_pressure"],
                "created_at": "2026-01-12T10:30:00",
                "parsed_values": {
                    "glucose": {"value": 120.0, "unit": "mg/dL"},
                    "blood_pressure": {"systolic": 130, "diastolic": 85, "formatted": "130/85"},
                    "heart_rate": {"value": 78.0, "unit": "bpm"}
                },
                "patient_info": {
                    "name": "John Doe",
                    "age": 45,
                    "gender": "Male"
                },
                "alerts": [
                    {
                        "metric": "glucose",
                        "value": 120.0,
                        "severity": "medium",
                        "message": "Glucose elevated (120 vs normal 70-100 mg/dL)"
                    }
                ],
                "metrics_found": 5
            }
        }


class OCRReportDetail(BaseModel):
    """Single OCR report details - WITH PARSED DATA."""
    report_id: str
    user_id: str
    extracted_text: str
    confidence: float
    report_type: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    created_at: str

    # NEW - Parsed data
    parsed_values: Dict[str, Any] = Field(default_factory=dict)
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    metrics_found: int = 0


class OCRReportsListResponse(BaseModel):
    """List of user's OCR reports."""
    status: str
    user_id: str
    total_reports: int
    reports: List[OCRReportDetail]

