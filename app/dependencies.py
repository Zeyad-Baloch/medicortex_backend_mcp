"""
Dependency injection for FastAPI.
Provides storage and service instances to routes.
"""
from app.storage.base import BaseStorage
from app.storage.memory import MemoryStorage
from app.storage.sqlite import SQLiteStorage
from app.services.baseline import BaselineService
from app.services.anomaly import AnomalyService
from app.config import settings


# Global storage instance (initialized once)
_storage_instance = None


def get_storage() -> BaseStorage:
    """
    Get storage backend based on configuration.
    Singleton pattern - one instance for the app.
    """
    global _storage_instance
    
    if _storage_instance is None:
        if settings.STORAGE_TYPE == "sqlite":
            _storage_instance = SQLiteStorage(settings.SQLITE_DB_PATH)
        elif settings.STORAGE_TYPE == "memory":
            _storage_instance = MemoryStorage()
        else:
            # Default to memory
            _storage_instance = MemoryStorage()
    
    return _storage_instance


def get_baseline_service(storage: BaseStorage = None) -> BaselineService:
    """Get baseline service with injected storage."""
    if storage is None:
        storage = get_storage()
    return BaselineService(storage)


def get_anomaly_service(storage: BaseStorage = None) -> AnomalyService:
    """Get anomaly service with injected storage."""
    if storage is None:
        storage = get_storage()
    return AnomalyService(storage)


# ============================================================================
# NEW - OCR SERVICE
# ============================================================================
from app.services.ocr import OCRService  # Add this import at top


def get_ocr_service(storage: BaseStorage = None) -> OCRService:
    """Get OCR service with injected storage."""
    if storage is None:
        storage = get_storage()
    return OCRService(storage)