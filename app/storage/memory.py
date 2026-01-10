"""
In-memory storage implementation (no persistence).
Good for development and testing.
"""
from typing import Dict, Optional
from app.storage.base import BaseStorage


class MemoryStorage(BaseStorage):
    """In-memory storage - data lost on restart."""
    
    def __init__(self):
        self.baselines = {}
        self.models = {}
        self.scalers = {}
    
    async def save_baseline(self, user_id: str, baseline: Dict) -> bool:
        """Save baseline to memory."""
        self.baselines[user_id] = baseline
        return True
    
    async def get_baseline(self, user_id: str) -> Optional[Dict]:
        """Get baseline from memory."""
        return self.baselines.get(user_id)
    
    async def save_models(self, user_id: str, models: Dict, scalers: Dict) -> bool:
        """Save models to memory."""
        self.models[user_id] = models
        self.scalers[user_id] = scalers
        return True
    
    async def get_models(self, user_id: str) -> Optional[tuple]:
        """Get models from memory."""
        models = self.models.get(user_id)
        scalers = self.scalers.get(user_id)
        if models and scalers:
            return models, scalers
        return None
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete user data from memory."""
        deleted = False
        if user_id in self.baselines:
            del self.baselines[user_id]
            deleted = True
        if user_id in self.models:
            del self.models[user_id]
        if user_id in self.scalers:
            del self.scalers[user_id]
        return deleted
    
    async def list_users(self) -> list:
        """List all users."""
        return list(self.baselines.keys())
