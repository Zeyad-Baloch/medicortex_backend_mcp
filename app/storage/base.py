"""
Abstract storage interface for baseline and model persistence.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
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
    
    def serialize_models(self, models: Dict, scalers: Dict) -> bytes:
        """Serialize models and scalers to bytes."""
        return pickle.dumps({'models': models, 'scalers': scalers})
    
    def deserialize_models(self, data: bytes) -> tuple:
        """Deserialize models and scalers from bytes."""
        obj = pickle.loads(data)
        return obj['models'], obj['scalers']
