import json
import logging
import os
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from models import CustomKnowledge, KnowledgeVector

logger = logging.getLogger(__name__)

async def process_embeddings(knowledge_id: int, content: str, db: Session):
    """Process document content and generate embeddings."""
    try:
        knowledge = db.query(CustomKnowledge).filter(CustomKnowledge.id == knowledge_id).first()
        if not knowledge:
            logger.error(f"Knowledge entry {knowledge_id} not found")
            return
        
        knowledge.embedding_status = "processing"
        db.commit()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(content)
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        for i, chunk in enumerate(chunks):
            try:
                embedding_vector = embeddings.embed_query(chunk)
                
                # Create KnowledgeVector entry
                vector = KnowledgeVector(
                    knowledge_id=knowledge_id,
                    chunk_index=i,
                    chunk_text=chunk,
                    embedding=json.dumps(embedding_vector),  # Serialize vector to JSON
                    embedding_model="text-embedding-ada-002",
                    dimensions=len(embedding_vector)
                )
                
                db.add(vector)
                
                if i % 5 == 0:  # Commit every 5 chunks
                    db.commit()
                    logger.info(f"Processed {i+1}/{len(chunks)} chunks for knowledge {knowledge_id}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i} for knowledge {knowledge_id}: {str(e)}", exc_info=True)
        
        # Final commit for any remaining chunks
        db.commit()
        
        # Update knowledge status
        knowledge.is_embedded = True
        knowledge.embedding_status = "completed"
        knowledge.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Embeddings generated for knowledge {knowledge_id}")
    except Exception as e:
        logger.error(f"Error processing embeddings for knowledge {knowledge_id}: {str(e)}", exc_info=True)
        try:
            knowledge.embedding_status = "failed"
            knowledge.updated_at = datetime.utcnow()
            db.commit()
        except Exception as commit_error:
            logger.error(f"Failed to update knowledge status: {str(commit_error)}")