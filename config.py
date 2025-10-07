"""
Configuration management for the Forensic Attorney AI Backend.
Loads environment variables and provides centralized config access.
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type safety.
    """
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    
    # Application Settings
    APP_NAME: str = "Forensic Attorney AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Security
    API_KEY_HEADER: str = "X-API-Key"
    ROBLOX_API_KEY: Optional[str] = None  # Optional: Add API key validation
    
    # Rate Limiting (requests per minute)
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # CORS Settings
    ALLOWED_ORIGINS: list = ["*"]  # Roblox HttpService doesn't send Origin headers
    
    # Knowledge Base
    KNOWLEDGE_BASE_PATH: str = "knowledge_base"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures we only load settings once.
    """
    return Settings()
