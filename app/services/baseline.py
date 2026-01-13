"""
Baseline service - handles baseline training and retrieval business logic.
"""
from typing import List, Dict
from app.ml.pipeline import MLPipeline
from app.storage.base import BaseStorage


class BaselineService:
    """Service for baseline training and management."""
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.pipeline = MLPipeline()
    
    async def train_baseline(self, user_id: str, data: List[Dict], days: int = 7) -> Dict:
        """
        Train a user's personalized baseline.
        
        Args:
            user_id: Unique user identifier
            data: List of health data points (dicts)
            days: Number of days of data
            
        Returns:
            Dict with training results
        """
        # Run ML pipeline
        pipeline_result = self.pipeline.train_baseline(user_id, data)
        
        # Save baseline to storage
        await self.storage.save_baseline(user_id, pipeline_result['result']['baselines'])
        
        # Save trained models
        await self.storage.save_models(user_id, pipeline_result['models'], pipeline_result['scalers'])
        
        # Return response
        return {
            "status": "success",
            "user_id": user_id,
            "baselines": pipeline_result['result']['baselines'],
            "model_performance": pipeline_result['result']['model_performance'],
            "message": f"Successfully learned baselines for {user_id} using {len(data)} data points"
        }
    
    async def get_baseline(self, user_id: str) -> Dict:
        """
        Retrieve a user's baseline.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dict with baseline data or error
        """
        baseline = await self.storage.get_baseline(user_id)
        
        if not baseline:
            return {
                "status": "error",
                "message": f"No baseline found for user {user_id}"
            }
        
        return {
            "status": "success",
            "user_id": user_id,
            "baselines": baseline
        }
