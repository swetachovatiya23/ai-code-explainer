"""
Configuration Management for AI Code Explainer.

This module provides centralized configuration management using Pydantic settings,
with support for environment variables and .env files.

WARNING: GROQ_BASE_URL should NOT include /openai/v1 suffix - the SDK adds it automatically.
"""

import os
from typing import Optional, List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # ==========================================================================
    # GROQ API Configuration
    # ==========================================================================
    
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key for authentication"
    )
    
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com",
        description="Groq API base URL (WITHOUT /openai/v1 suffix - SDK adds it automatically)"
    )
    
    GROQ_MODEL_NAME: str = Field(
        default="llama-3.3-70b-versatile",
        description="AI model to use for code explanation"
    )
    
    # ==========================================================================
    # Application Configuration
    # ==========================================================================
    
    APP_NAME: str = Field(
        default="AI Code Explainer",
        description="Application name"
    )
    
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    MAX_CODE_LINES: int = Field(
        default=500,
        description="Maximum number of lines allowed for code input"
    )
    
    MAX_FILE_SIZE_MB: int = Field(
        default=5,
        description="Maximum file upload size in megabytes"
    )
    
    # ==========================================================================
    # API Mode Configuration
    # ==========================================================================
    
    API_HOST: str = Field(
        default="0.0.0.0",
        description="FastAPI server host"
    )
    
    API_PORT: int = Field(
        default=8000,
        description="FastAPI server port"
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    
    @field_validator("GROQ_BASE_URL")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL does not include /openai/v1 suffix."""
        v = v.rstrip("/")
        if v.endswith("/openai/v1"):
            v = v.replace("/openai/v1", "")
        if v.endswith("/openai"):
            v = v.replace("/openai", "")
        return v
    
    @field_validator("GROQ_MODEL_NAME")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that the model name is in the supported list."""
        valid_models = get_available_models()
        if v not in valid_models:
            # Return default if invalid
            return "llama-3.3-70b-versatile"
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


def get_available_models() -> List[str]:
    """
    Get list of available Groq models.
    
    These are validated, working model names from Groq's API.
    
    Returns:
        List of available model identifiers.
    """
    return [
        "llama-3.3-70b-versatile",      # Default/Recommended - Best quality
        "llama-3.1-8b-instant",          # Faster - Good for quick responses
        "llama-3.1-70b-versatile",       # High quality alternative
        "mixtral-8x7b-32768",            # Mixtral model with large context
        "gemma2-9b-it",                  # Google's Gemma 2 model
    ]


def get_model_descriptions() -> dict:
    """
    Get descriptions for available models.
    
    Returns:
        Dictionary mapping model names to their descriptions.
    """
    return {
        "llama-3.3-70b-versatile": "🌟 Llama 3.3 70B (Recommended) - Best quality for code explanation",
        "llama-3.1-8b-instant": "⚡ Llama 3.1 8B Instant - Faster responses, good for simple code",
        "llama-3.1-70b-versatile": "🔷 Llama 3.1 70B - High quality alternative",
        "mixtral-8x7b-32768": "🔀 Mixtral 8x7B - Large 32K context window",
        "gemma2-9b-it": "💎 Gemma 2 9B - Google's efficient model",
    }


def get_supported_languages() -> List[str]:
    """
    Get list of supported programming languages for analysis.
    
    Returns:
        List of supported language names.
    """
    return [
        "python",
        "javascript",
        "typescript",
        "java",
        "cpp",
        "c",
        "csharp",
        "go",
        "rust",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "r",
        "sql",
        "html",
        "css",
        "bash",
        "powershell",
    ]


def get_file_extensions() -> dict:
    """
    Get mapping of file extensions to programming languages.
    
    Returns:
        Dictionary mapping file extensions to language names.
    """
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".R": "r",
        ".sql": "sql",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".sh": "bash",
        ".bash": "bash",
        ".ps1": "powershell",
    }


def get_explanation_levels() -> List[dict]:
    """
    Get available explanation granularity levels.
    
    Returns:
        List of dictionaries with level details.
    """
    return [
        {
            "id": "high-level",
            "name": "High-Level Summary",
            "description": "What does this code do overall?",
            "icon": "📋"
        },
        {
            "id": "line-by-line",
            "name": "Line-by-Line Walkthrough",
            "description": "Step-by-step explanation of each line",
            "icon": "📝"
        },
        {
            "id": "eli5",
            "name": "ELI5 (Simple)",
            "description": "Explain using simple analogies",
            "icon": "🧒"
        },
    ]


def get_audience_levels() -> List[dict]:
    """
    Get available audience levels for explanations.
    
    Returns:
        List of dictionaries with audience details.
    """
    return [
        {
            "id": "beginner",
            "name": "Beginner",
            "description": "Avoids jargon, uses simple terms",
            "icon": "🌱"
        },
        {
            "id": "intermediate",
            "name": "Intermediate",
            "description": "Balanced technical depth",
            "icon": "🌿"
        },
        {
            "id": "expert",
            "name": "Expert",
            "description": "Focuses on architecture and patterns",
            "icon": "🌳"
        },
    ]


def get_analysis_modes() -> List[dict]:
    """
    Get available analysis modes.
    
    Returns:
        List of dictionaries with mode details.
    """
    return [
        {
            "id": "educational",
            "name": "Educational Mode",
            "description": "Focuses on teaching concepts (good for students)",
            "icon": "📚"
        },
        {
            "id": "auditor",
            "name": "Auditor Mode",
            "description": "Focuses on security and performance (good for developers)",
            "icon": "🔍"
        },
    ]


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings instance with current configuration.
    """
    return Settings()


def update_settings_from_session(session_state: dict) -> Settings:
    """
    Create settings instance updated with session state values.
    
    This allows runtime override of settings from Streamlit session state.
    
    Args:
        session_state: Streamlit session state dictionary.
    
    Returns:
        Settings instance with session values applied.
    """
    settings = Settings()
    
    # Override with session state values if present
    if session_state.get("groq_api_key"):
        settings.GROQ_API_KEY = session_state["groq_api_key"]
    if session_state.get("groq_base_url"):
        settings.GROQ_BASE_URL = session_state["groq_base_url"]
    if session_state.get("groq_model_name"):
        settings.GROQ_MODEL_NAME = session_state["groq_model_name"]
    
    return settings
