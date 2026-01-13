"""
Anomaly service - handles anomaly detection business logic.
"""
from typing import List, Dict
from app.ml.pipeline import MLPipeline
from app.storage.base import BaseStorage


class AnomalyService:
    """Service for anomaly detection."""
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.pipeline = MLPipeline()
    
    async def detect_anomalies(self, user_id: str, data: List[Dict]) -> Dict:
        """
        Detect anomalies in user's health data.
        
        Args:
            user_id: Unique user identifier
            data: List of health data points to analyze
            
        Returns:
            Dict with detected anomalies and risk assessment
        """
        # Get user's baseline
        baseline = await self.storage.get_baseline(user_id)
        
        if not baseline:
            return {
                "status": "error",
                "message": f"No baseline found for user {user_id}. Train baseline first."
            }
        
        # Run anomaly detection pipeline
        detection_result = self.pipeline.detect_anomalies(baseline, data)
        
        return {
            "status": "success",
            "user_id": user_id,
            **detection_result
        }
