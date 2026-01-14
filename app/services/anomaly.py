"""
Anomaly service - handles anomaly detection and saves history.
"""
from typing import List, Dict
from app.ml.pipeline import MLPipeline
from app.storage.base import BaseStorage


class AnomalyService:
    """Service for anomaly detection and management."""

    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.pipeline = MLPipeline()

    async def detect_anomalies(self, user_id: str, data: List[Dict]) -> Dict:
        """
        Detect health anomalies for a user.

        Args:
            user_id: Unique user identifier
            data: List of current health data points (dicts)

        Returns:
            Dict with anomaly detection results
        """
        # Get user's baseline
        baseline = await self.storage.get_baseline(user_id)

        if not baseline:
            return {
                "status": "error",
                "message": f"No baseline found for user {user_id}. Please train baseline first."
            }

        # Run ML pipeline for anomaly detection
        result = self.pipeline.detect_anomalies(baseline, data)

        # ============================================================================
        # NEW - Save health data to history
        # ============================================================================
        await self.storage.save_health_data(user_id, data)

        # ============================================================================
        # NEW - Save anomalies to history
        # ============================================================================
        if result['alerts']:
            await self.storage.save_anomalies(user_id, result['alerts'])

        # Return response
        return {
            "status": "success",
            "user_id": user_id,
            "total_anomalies": result['total_anomalies'],
            "high_risk_count": result['high_risk_count'],
            "medium_risk_count": result['medium_risk_count'],
            "low_risk_count": result['low_risk_count'],
            "alerts": result['alerts']
        }