# Backend Package
"""
Backend module containing configuration, services, and API endpoints.
"""

from .config import Settings, get_settings
from .services import CodeExplainerService

__all__ = ["Settings", "get_settings", "CodeExplainerService"]
