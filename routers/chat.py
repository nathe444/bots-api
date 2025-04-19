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
from langchain_community.utilities.zapier import ZapierNLAWrapper
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation store
conversation_store: Dict[str, List[Dict[str, str]]] = {}

# Zapier tools cache
zapier_tools_cache = []

# Helper functions for Zapier integration
def load_zapier_tools():
    """Load all available Zapier tools dynamically with metadata"""
    zapier_api_key = os.getenv("ZAPIER_API_KEY")
    if not zapier_api_key:
        logger.warning("Zapier API key not found in environment variables")
        return []
    
    try:
        zapier = ZapierNLAWrapper(zapier_nla_api_key=zapier_api_key)
        actions = zapier.list()
        
        tools = []
        for action in actions:
            tool = Tool(
                name=action["description"],
                description=action["description"],
                func=lambda query, action_id=action["id"]: zapier.run(action_id, query),
                metadata={
                    "action_id": action["id"],
                    "description": action["description"]
                }
            )
            tools.append(tool)
            
        logger.info(f"Successfully loaded {len(tools)} Zapier tools")
        return tools
    except Exception as e:
        logger.error(f"Error loading Zapier tools: {str(e)}")
        return []

def identify_zapier_action(user_input: str, tools: List[Tool]) -> Optional[Tool]:
    """Use LLM to identify the appropriate Zapier action"""
    prompt = f"""Match this request to a Zapier action:
User Request: {user_input}

Available Actions:
{", ".join([t.metadata['description'] for t in tools])}

Respond with JUST the action description that best matches."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        matched_description = response.choices[0].message.content.strip()
        return next((t for t in tools if t.metadata['description'] == matched_description), None)
    except Exception as e:
        logger.error(f"Action identification failed: {str(e)}")
        return None

def extract_action_parameters(user_input: str, action: Tool, history: List[dict]) -> dict:
    """Extract parameters needed for the action using LLM"""
    prompt = f"""Extract required parameters from this user input for the Zapier action:

Action Description: {action.metadata['description']}
User Input: {user_input}

Return JSON with:
- "params": key-value pairs of extracted parameters
- "missing": list of missing parameters

Special handling:
- For calendar events, infer 'primary' as default calendar ID
- For email actions, infer email from context
- Never ask for technical IDs (like calendar ID), use natural language"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        
        # Add default calendar ID handling
        if "calendar" in action.metadata['description'].lower():
            result["params"]["calendar_id"] = "primary"
            if "calendar_id" in result.get("missing", []):
                result["missing"].remove("calendar_id")
                
        return result
    except Exception as e:
        logger.error(f"Parameter extraction failed: {str(e)}")
        return {"params": {}, "missing": []}

def process_with_zapier(user_request: str, tools: List[Tool], conversation_history: List[dict]):
    """Process request with dynamic action detection and parameter extraction"""
    try:
        # Identify the appropriate action
        action = identify_zapier_action(user_request, tools)
        if not action:
            return None
        
        # Extract parameters
        extraction_result = extract_action_parameters(user_request, action, conversation_history)
        
        # Check for missing parameters
        if extraction_result.get("missing", []):
            return {
                "action": action,
                "params": extraction_result["params"],
                "missing": extraction_result["missing"],
                "prompt": f"Please provide: {', '.join(extraction_result['missing'])}"
            }
        
        # Execute the action
        query = ", ".join([f"{k}: {v}" for k, v in extraction_result["params"].items()])
        result = action.func(query)
        
        # Convert dictionary result to string if needed
        if isinstance(result, dict):
            result_str = f"Action completed successfully: {json.dumps(result, indent=2)}"
        else:
            result_str = str(result)
            
        return {
            "action": action,
            "result": result_str,
            "params": extraction_result["params"]
        }
        
    except Exception as e:
        logger.error(f"Zapier processing error: {str(e)}")
        return None

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    bot_id: int
    user_request: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    response: str = "I encountered an error processing your request."
    sources: Optional[List[dict]] = None
    zapier_action: Optional[bool] = None
    missing_fields: Optional[List[str]] = None

class ZapierTool(BaseModel):
    id: str
    name: str
    description: str

# Endpoints
@router.get("/zapier-tools", response_model=List[ZapierTool])
async def list_zapier_tools():
    """Endpoint to list all available Zapier tools"""
    global zapier_tools_cache
    
    if not zapier_tools_cache:
        zapier_tools_cache = load_zapier_tools()
    
    return [
        {
            "id": tool.metadata["action_id"],
            "name": tool.name,
            "description": tool.metadata["description"]
        }
        for tool in zapier_tools_cache
    ]

@router.post("/chat", response_model=ChatResponse)
async def get_ai_response(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db)
):
    try:
        # Initialize or retrieve conversation history
        if request.session_id not in conversation_store:
            conversation_store[request.session_id] = []
        
        conversation_history = request.history if request.history else conversation_store[request.session_id]
        
        # Check for Zapier actions first
        global zapier_tools_cache
        if not zapier_tools_cache:
            zapier_tools_cache = load_zapier_tools()
            
        if zapier_tools_cache:
            zapier_result = process_with_zapier(request.user_request, zapier_tools_cache, conversation_history)
            
            if zapier_result:
                if "missing" in zapier_result:
                    # Update conversation history with the prompt
                    conversation_store[request.session_id] = conversation_history + [
                        {"role": "user", "content": request.user_request},
                        {"role": "assistant", "content": zapier_result["prompt"]}
                    ]
                    return ChatResponse(
                        response=zapier_result["prompt"],
                        zapier_action=True,
                        missing_fields=zapier_result["missing"]
                    )
                else:
                    # Action executed successfully
                    conversation_store[request.session_id] = conversation_history + [
                        {"role": "user", "content": request.user_request},
                        {"role": "assistant", "content": zapier_result["result"]}
                    ]
                    return ChatResponse(
                        response=zapier_result["result"],
                        zapier_action=True
                    )
        
        # Fall back to knowledge-based response
        knowledge_entries = db.query(CustomKnowledge).filter(
            CustomKnowledge.bot_id == request.bot_id,
            CustomKnowledge.is_embedded == True,
            CustomKnowledge.embedding_status == "completed"
        ).all()
        
        if not knowledge_entries:
            return ChatResponse(response="I don't have relevant knowledge to answer that.")
        
        # Generate embedding for the query
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        query_embedding = embeddings.embed_query(request.user_request)
        
        # Find relevant chunks
        relevant_chunks = []
        sources = []
        for knowledge in knowledge_entries:
            vectors = db.query(KnowledgeVector).filter(
                KnowledgeVector.knowledge_id == knowledge.id
            ).all()
            
            for vector in vectors:
                chunk_embedding = json.loads(vector.embedding)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
                
                if similarity > 0.7:
                    relevant_chunks.append({
                        "text": vector.chunk_text,
                        "similarity": similarity,
                        "source": knowledge.title
                    })
                    if knowledge.title not in [s["title"] for s in sources]:
                        sources.append({"title": knowledge.title, "id": knowledge.id})
        
        # Prepare context
        context = "\n\n".join([chunk["text"] for chunk in sorted(
            relevant_chunks, key=lambda x: x["similarity"], reverse=True)[:5]])
        
        # Generate response
        messages = [
            {"role": "system", "content": f"Answer based on this context:\n{context}"},
            *[{"role": msg["role"], "content": msg["content"]} for msg in conversation_history],
            {"role": "user", "content": request.user_request}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        
        # Update conversation history
        conversation_store[request.session_id] = conversation_history + [
            {"role": "user", "content": request.user_request},
            {"role": "assistant", "content": ai_response}
        ]
        
        return ChatResponse(
            response=ai_response,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return ChatResponse(response="I encountered an error processing your request.")