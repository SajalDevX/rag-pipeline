"""
API Package

This package contains the FastAPI application and routes.

Usage:
    from src.api import app
    
    # Run with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from src.api.app import app, create_app

__all__ = [
    "app",
    "create_app",
]