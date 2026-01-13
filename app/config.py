"""
Application configuration and settings.
Uses environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # Project Info
    PROJECT_NAME: str = "MediCortex ML Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ML Configuration
    N_ESTIMATORS: int = 100
    MAX_DEPTH: int = 10
    CONTAMINATION: float = 0.05
    RANDOM_STATE: int = 42

    # Storage Configuration
    STORAGE_TYPE: str = "sqlite"  # Options: memory, sqlite, redis, postgres
    SQLITE_DB_PATH: str = "medicortex.db"
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    DATABASE_URL: Optional[str] = None

    # API Keys
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # ============================================================================
    # OCR CONFIGURATION - NEW
    # ============================================================================
    OCR_SPACE_API_KEY: Optional[str] = None  # Get free key from https://ocr.space/ocrapi
    OCR_LANGUAGE: str = "eng"  # eng, ara, urd, etc.
    OCR_MAX_FILE_SIZE_MB: int = 5  # Maximum file size in MB

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()