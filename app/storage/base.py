"""
Abstract storage interface for baseline and model persistence.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import pickle


class BaseStorage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save_baseline(self, user_id: str, baseline: Dict) -> bool:
        """Save user's baseline data."""
        pass

    @abstractmethod
    async def get_baseline(self, user_id: str) -> Optional[Dict]:
        """Retrieve user's baseline data."""
        pass

    @abstractmethod
    async def save_models(self, user_id: str, models: Dict, scalers: Dict) -> bool:
        """Save trained ML models and scalers."""
        pass

    @abstractmethod
    async def get_models(self, user_id: str) -> Optional[tuple]:
        """Retrieve trained models and scalers."""
        pass

    @abstractmethod
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user."""
        pass

    @abstractmethod
    async def list_users(self) -> list:
        """List all users with stored data."""
        pass

    # ============================================================================
    # OCR STORAGE METHODS
    # ============================================================================

    @abstractmethod
    async def save_ocr_report(self, user_id: str, report_id: str, report_data: Dict) -> bool:
        """Save OCR extracted report."""
        pass

    @abstractmethod
    async def get_ocr_report(self, report_id: str) -> Optional[Dict]:
        """Get specific OCR report by ID."""
        pass

    @abstractmethod
    async def get_user_ocr_reports(self, user_id: str) -> List[Dict]:
        """Get all OCR reports for a user."""
        pass

    @abstractmethod
    async def delete_ocr_report(self, report_id: str) -> bool:
        """Delete an OCR report."""
        pass

    # ============================================================================
    # NEW - HEALTH DATA HISTORY METHODS
    # ============================================================================

    @abstractmethod
    async def save_health_data(self, user_id: str, data_points: List[Dict]) -> bool:
        """Save multiple health data points to history."""
        pass

    @abstractmethod
    async def get_health_data_range(self, user_id: str, start_date: str = None,
                                   end_date: str = None, limit: int = None) -> List[Dict]:
        """Get health data for a user within a date range."""
        pass

    @abstractmethod
    async def get_latest_health_data(self, user_id: str) -> Optional[Dict]:
        """Get the most recent health data point for a user."""
        pass

    @abstractmethod
    async def get_health_data_stats(self, user_id: str, days: int = 30) -> Dict:
        """Get aggregated statistics for health data over a time period."""
        pass

    # ============================================================================
    # NEW - ANOMALY HISTORY METHODS
    # ============================================================================

    @abstractmethod
    async def save_anomalies(self, user_id: str, anomalies: List[Dict]) -> bool:
        """Save detected anomalies to history."""
        pass

    @abstractmethod
    async def get_anomaly_history(self, user_id: str, days: int = 30,
                                 risk_level: str = None) -> List[Dict]:
        """Get anomaly history for a user."""
        pass

    @abstractmethod
    async def get_anomaly_counts(self, user_id: str, days: int = 30) -> Dict:
        """Get count of anomalies by risk level."""
        pass

    # ============================================================================
    # SERIALIZATION METHODS
    # ============================================================================

    def serialize_models(self, models: Dict, scalers: Dict) -> bytes:
        """Serialize models and scalers to bytes."""
        return pickle.dumps({'models': models, 'scalers': scalers})

    def deserialize_models(self, data: bytes) -> tuple:
        """Deserialize models and scalers from bytes."""
        obj = pickle.loads(data)
        return obj['models'], obj['scalers']