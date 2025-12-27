"""
Configuration Package

This package provides:
- settings: Application configuration loaded from environment
- setup_logging: Initialize the logging system
- get_logger: Get a logger instance for any module

Usage:
    from src.config import settings, setup_logging, get_logger
    
    # At application startup
    setup_logging()
    
    # Access configuration
    print(settings.api_port)
    
    # Get a logger
    logger = get_logger(__name__)
    logger.info("Application started")
"""

from src.config.logging_config import get_logger, setup_logging
from src.config.settings import Settings, get_settings, settings

__all__ = [
    # Settings
    "Settings",
    "settings",
    "get_settings",
    # Logging
    "setup_logging",
    "get_logger",
]