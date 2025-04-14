import json
import logging
import os
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pydantic import BaseModel
from database import get_db
from models import CustomKnowledge, KnowledgeVector
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation store (in a production app, use a database)
conversation_store: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    session_id: str
    bot_id: int
    user_request: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[dict]] = None

@router.post("/chat", response_model=ChatResponse)
async def get_ai_response(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db)
):
    try:
        # Initialize or retrieve conversation history
        if request.session_id not in conversation_store:
            conversation_store[request.session_id] = []
        
        # Use provided history if available, otherwise use stored history
        conversation_history = request.history if request.history is not None else conversation_store[request.session_id]
        
        # 1. Retrieve relevant knowledge for the bot
        knowledge_entries = db.query(CustomKnowledge).filter(
            CustomKnowledge.bot_id == request.bot_id,
            CustomKnowledge.is_embedded == True,
            CustomKnowledge.embedding_status == "completed"
        ).all()
        
        if not knowledge_entries:
            logger.warning(f"No knowledge entries found for bot {request.bot_id}")
            # Fallback to general response if no knowledge is available
            return {"response": "I don't have any specific knowledge to answer that question."}
        
        # 2. Generate embedding for the user query
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        query_embedding = embeddings.embed_query(request.user_request)
        
        # 3. Find relevant chunks from the knowledge base
        relevant_chunks = []
        sources = []
        
        for knowledge in knowledge_entries:
            # Get all vectors for this knowledge entry
            vectors = db.query(KnowledgeVector).filter(
                KnowledgeVector.knowledge_id == knowledge.id
            ).all()
            
            if not vectors:
                continue
                
            # Calculate similarity scores
            for vector in vectors:
                chunk_embedding = json.loads(vector.embedding)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                if similarity > 0.7:  # Threshold for relevance
                    relevant_chunks.append({
                        "text": vector.chunk_text,
                        "similarity": similarity,
                        "source": knowledge.title
                    })
                    
                    # Add source information if not already added
                    source_titles = [s.get("title", "") for s in sources]
                    if knowledge.title not in source_titles:
                        sources.append({
                            "title": knowledge.title,
                            "id": knowledge.id
                        })
        
        # Sort chunks by similarity (highest first)
        relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 4. Prepare context from relevant chunks (limit to top 5)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks[:5]])
        
        # 5. Prepare messages for the API call
        messages = [{"role": "system", "content": f"""You are an AI assistant for a company. Answer the user's question based on the following context.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}
"""}]
        
        # Add conversation history to messages
        for msg in conversation_history:
            if "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the current user request
        messages.append({"role": "user", "content": request.user_request})
        
        # 6. Generate response using OpenAI with the new API syntax
        response = client.chat.completions.create(
            model="gpt-4",  # or another appropriate model
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content
        
        # Update conversation history
        conversation_store[request.session_id] = conversation_history + [
            {"role": "user", "content": request.user_request},
            {"role": "assistant", "content": ai_response}
        ]
        
        # Limit conversation history to last 10 messages to prevent context overflow
        if len(conversation_store[request.session_id]) > 10:
            conversation_store[request.session_id] = conversation_store[request.session_id][-10:]
        
        return {
            "response": ai_response,
            "sources": sources if sources else None
        }
        
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")