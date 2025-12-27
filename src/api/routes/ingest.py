"""
Ingestion API Routes

Endpoints for document ingestion:
- POST /ingest - Ingest a document from file or URL
- POST /ingest/upload - Upload and ingest a file
- DELETE /documents/{document_id} - Delete a document
"""

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.config import get_logger, settings
from src.models import IngestRequest, IngestResponse
from src.models.errors import RAGException
from src.services import IngestionService

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Ingestion"])

# Shared service instance
_ingestion_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """Get or create ingestion service instance."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Ingest a document from a file path or URL.
    
    This endpoint accepts a source path (local file or URL) and:
    1. Loads the document
    2. Splits it into chunks
    3. Generates embeddings
    4. Stores in the vector database
    
    Args:
        request: IngestRequest with source and optional metadata
    
    Returns:
        IngestResponse: Result of ingestion
    
    Example:
        POST /api/ingest
        {
            "source": "/path/to/document.pdf",
            "title": "My Document",
            "custom_metadata": {"department": "engineering"}
        }
    """
    logger.info("Ingest request received", source=request.source)
    
    try:
        service = get_ingestion_service()
        
        result = await service.ingest(
            source=request.source,
            document_type=request.document_type,
            title=request.title,
            custom_metadata=request.custom_metadata
        )
        
        return result
        
    except RAGException as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error during ingestion", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/ingest/upload", response_model=IngestResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    custom_metadata: str | None = Form(None)  # JSON string
) -> IngestResponse:
    """
    Upload a file and ingest it.
    
    This endpoint accepts a file upload and:
    1. Saves the file temporarily
    2. Ingests it into the system
    3. Cleans up the temporary file
    
    Args:
        file: Uploaded file
        title: Optional document title
        custom_metadata: Optional JSON string of metadata
    
    Returns:
        IngestResponse: Result of ingestion
    
    Example:
        POST /api/ingest/upload
        Content-Type: multipart/form-data
        file: <binary>
        title: "My Document"
    """
    logger.info("File upload received", filename=file.filename)
    
    # Validate file size
    # Note: For production, use streaming to handle large files
    
    # Create upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file with unique name
    file_id = str(uuid4())[:8]
    file_extension = Path(file.filename).suffix if file.filename else ""
    temp_path = upload_dir / f"{file_id}{file_extension}"
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.debug("File saved", path=str(temp_path))
        
        # Parse custom metadata if provided
        metadata = {}
        if custom_metadata:
            import json
            try:
                metadata = json.loads(custom_metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON in custom_metadata"
                )
        
        # Ingest the file
        service = get_ingestion_service()
        
        result = await service.ingest(
            source=str(temp_path),
            title=title or file.filename,
            custom_metadata=metadata
        )
        
        return result
        
    except RAGException as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
            logger.debug("Temporary file cleaned up", path=str(temp_path))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, Any]:
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: UUID of the document to delete
    
    Returns:
        dict: Deletion result
    
    Example:
        DELETE /api/documents/123e4567-e89b-12d3-a456-426614174000
    """
    logger.info("Delete request received", document_id=document_id)
    
    try:
        service = get_ingestion_service()
        deleted_count = await service.delete_document(document_id)
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_deleted": deleted_count
        }
        
    except RAGException as e:
        logger.error("Deletion failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")