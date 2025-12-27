"""
Logging Configuration

This module sets up structured logging using structlog.
It provides:
- Pretty colored output for development
- JSON output for production (easy to parse)
- Consistent log format across the application

Usage:
    from src.config.logging_config import setup_logging, get_logger
    
    # Setup at application start
    setup_logging()
    
    # Get a logger in any module
    logger = get_logger(__name__)
    logger.info("Something happened", user_id=123, action="login")
"""

import logging
import sys

import structlog

from src.config.settings import settings


def setup_logging() -> None:
    """
    Configure logging for the application.
    
    Call this once at application startup (in main.py).
    
    In development:
        - Pretty, colored console output
        - Easy to read for humans
    
    In production:
        - JSON format
        - Easy to parse by log aggregation tools
    """
    
    # Convert string log level to logging constant
    # "INFO" -> logging.INFO (which is 20)
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Processors are functions that transform log entries
    # They run in order, each one adding or modifying data
    shared_processors = [
        # Add log level (INFO, WARNING, etc.)
        structlog.stdlib.add_log_level,
        
        # Add logger name (usually the module name)
        structlog.stdlib.add_logger_name,
        
        # Add ISO format timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        
        # Add information about where the log was called from
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        
        # Handle exceptions nicely
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Choose output format based on environment
    if settings.is_development:
        # Development: Pretty colored output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,           # Enable colors
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]
    else:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging (for third-party libraries)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Reduce noise from chatty libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name, typically __name__ of the module.
              If None, returns the root logger.
    
    Returns:
        A configured logger instance
    
    Usage:
        # In any module
        from src.config.logging_config import get_logger
        
        logger = get_logger(__name__)
        
        # Basic logging
        logger.info("User logged in")
        
        # With structured data (recommended)
        logger.info("User logged in", user_id=123, ip="192.168.1.1")
        
        # Different levels
        logger.debug("Detailed info for debugging")
        logger.info("General information")
        logger.warning("Something might be wrong")
        logger.error("Something went wrong", error=str(e))
        
        # With exception info
        try:
            do_something()
        except Exception:
            logger.exception("Failed to do something")
    """
    return structlog.get_logger(name)