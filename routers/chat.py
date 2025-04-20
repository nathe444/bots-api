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
from langchain_openai import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import ZapierToolkit
from langchain_community.utilities.zapier import ZapierNLAWrapper
from langchain.tools import Tool

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

def llm_identifies_zapier_action(llm, user_request, zapier_tools):
    """Ask the LLM if the user request matches any Zapier action."""
    tool_descriptions = "\n".join([f"- {tool['description']}" for tool in zapier_tools])
    prompt = (
        f"User request: \"{user_request}\"\n"
        f"Available Zapier actions:\n{tool_descriptions}\n"
        "Does the user request match any of the above Zapier actions? "
        "Respond with 'yes' or 'no' only."
    )
    response = llm(prompt)
    return response.strip().lower().startswith("yes")

def get_relevant_knowledge(db, bot_id, user_request, embeddings):
    """Helper function to get relevant knowledge chunks"""
    knowledge_entries = db.query(CustomKnowledge).filter(
        CustomKnowledge.bot_id == bot_id,
        CustomKnowledge.is_embedded == True,
        CustomKnowledge.embedding_status == "completed"
    ).all()
    
    query_embedding = embeddings.embed_query(user_request)
    relevant_chunks = []
    sources = []
    
    for knowledge in knowledge_entries:
        vectors = db.query(KnowledgeVector).filter(
            KnowledgeVector.knowledge_id == knowledge.id
        ).all()
        
        for vector in vectors:
            chunk_embedding = json.loads(vector.embedding)
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            if similarity > 0.7:
                relevant_chunks.append({
                    "text": vector.chunk_text,
                    "similarity": similarity,
                    "source": knowledge.title
                })
                if knowledge.title not in [s.get("title", "") for s in sources]:
                    sources.append({"title": knowledge.title, "id": knowledge.id})
    
    relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return relevant_chunks[:5], sources

def select_zapier_tool(user_request, zapier_tools, llm):
    """Select the most appropriate Zapier tool based on the request"""
    tools_prompt = (
        f"User request: '{user_request}'\n"
        "Available tools:\n"
        f"{json.dumps([{'id': t['id'], 'description': t['description']} for t in zapier_tools], indent=2)}\n"
        "Return only the ID of the most appropriate tool for this request, or 'none' if no tool matches."
    )
    response = llm.invoke(tools_prompt).strip()
    matching_tools = [t for t in zapier_tools if t['id'] in response]
    return matching_tools[0] if matching_tools else None

@router.post("/chat", response_model=ChatResponse)
async def get_ai_response(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db)
):
    try:
        # Initialize conversation store
        if request.session_id not in conversation_store:
            conversation_store[request.session_id] = []
        conversation_history = request.history if request.history is not None else conversation_store[request.session_id]

        # Setup LLM and Zapier
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        zapier = ZapierNLAWrapper(zapier_nla_api_key=os.getenv("ZAPIER_API_KEY"))
        zapier_tools = zapier.list()
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Get relevant knowledge first
        relevant_chunks, sources = get_relevant_knowledge(db, request.bot_id, request.user_request, embeddings)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        # Select appropriate Zapier tool
        selected_tool = select_zapier_tool(request.user_request, zapier_tools, llm)

        if selected_tool:
            # Create a combined prompt that uses both knowledge and action
            action_prompt = (
                f"Context from knowledge base:\n{context}\n\n"
                f"User request: {request.user_request}\n\n"
                f"Based on this context and the user's request, "
                f"generate appropriate instructions for the following action: {selected_tool['description']}"
            )
            
            action_instructions = llm.invoke(action_prompt)
            
            # Execute the Zapier action with proper error handling
            try:
                tool = Tool(
                    name=selected_tool['id'],
                    description=selected_tool['description'],
                    func=lambda x: zapier.run(selected_tool['id'], x)
                )
                
                action_result = tool.run(action_instructions)
                
                response = (
                    f"Based on the available information, I've taken the following action: {action_result}\n\n"
                    f"Additional context from our knowledge base: {context}"
                )
                
                conversation_store[request.session_id] = conversation_history + [
                    {"role": "user", "content": request.user_request},
                    {"role": "assistant", "content": response}
                ]
                return {"response": response, "sources": sources if sources else None}
            except Exception as tool_error:
                logger.error(f"Error executing Zapier action: {str(tool_error)}")
                # Fallback to knowledge-based response
                selected_tool = None

        # Knowledge-based response (when no Zapier tool is selected or Zapier action fails)
        if not selected_tool:
            messages = [{"role": "system", "content": f"""You are an AI assistant for a company. Answer the user's question based on the following context.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}
"""}]
            
            for msg in conversation_history:
                if "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": request.user_request})

            # Use LangChain OpenAI LLM for chat completion
            llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            ai_response = llm.invoke(prompt)

            conversation_store[request.session_id] = conversation_history + [
                {"role": "user", "content": request.user_request},
                {"role": "assistant", "content": ai_response}
            ]
            if len(conversation_store[request.session_id]) > 10:
                conversation_store[request.session_id] = conversation_store[request.session_id][-10:]
            return {
                "response": ai_response,
                "sources": sources if sources else None
            }
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")