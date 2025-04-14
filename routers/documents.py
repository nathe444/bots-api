import json
import logging
import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from typing import Optional
from database import get_db
from models import CustomKnowledge
from utils.file_processor import save_uploaded_file, extract_content
from utils.embeddings import process_embeddings

logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(tags=["documents"])

@router.get("/")
def read_root():
    return {"message": "Welcome to the Bot Training API"}

@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    bot_id: int = Form(...),
    title: Optional[str] = Form(None),
    created_by: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        # Convert created_by to integer or None
        created_by_int = None
        if created_by and created_by.strip():
            try:
                created_by_int = int(created_by)
            except ValueError:
                pass
                
        # Save the uploaded file
        file_path, unique_filename = save_uploaded_file(file, UPLOAD_DIR)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Extract content based on file type
        content, file_type = extract_content(file_path, file.content_type)
        
        # Use the filename as title if not provided
        if not title:
            title = os.path.splitext(file.filename)[0]
        
        # Create a new CustomKnowledge entry
        knowledge = CustomKnowledge(
            bot_id=bot_id,
            title=title,
            content=content,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            embedding_model="text-embedding-ada-002",
            is_embedded=False,
            embedding_status="pending",
            created_by=created_by_int,
            document_metadata=json.dumps({"original_filename": file.filename})
        )
        
        db.add(knowledge)
        db.commit()
        db.refresh(knowledge)
        
        # Process the document for embeddings and wait for it to complete
        await process_embeddings(knowledge.id, content, db)
        
        # Refresh the knowledge object to get the updated status
        db.refresh(knowledge)
        
        return {
            "message": "Document uploaded and processed successfully",
            "knowledge_id": knowledge.id,
            "title": title,
            "file_type": file_type,
            "file_size": file_size,
            "embedding_status": knowledge.embedding_status
        }
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")