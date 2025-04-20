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

@router.post("/chat", response_model=ChatResponse)
async def get_ai_response(
    request: ChatRequest = Body(...),
    db: Session = Depends(get_db)
):
    try:
        if request.session_id not in conversation_store:
            conversation_store[request.session_id] = []
        conversation_history = request.history if request.history is not None else conversation_store[request.session_id]

        # --- Zapier agent integration with LLM intent detection ---
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        zapier = ZapierNLAWrapper(zapier_nla_api_key=os.getenv("ZAPIER_API_KEY"))
        # toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        zapier_tools = zapier.list()  # Get available Zapier actions
        # logger.info(zapier_tools)
        # Use LLM to decide if this is a Zapier action
        if llm_identifies_zapier_action(llm, request.user_request, zapier_tools):
            from langchain.tools import Tool
            
            # First, check if this is a hybrid request (knowledge + action)
            is_hybrid = False
            knowledge_keywords = ["summarize", "summary", "extract", "analyze", "review"]
            action_keywords = ["email", "send", "share", "forward"]
            
            if any(kw in request.user_request.lower() for kw in knowledge_keywords) and \
               any(aw in request.user_request.lower() for aw in action_keywords):
                is_hybrid = True
            
            if is_hybrid:
                # 1. First get the knowledge content
                knowledge_entries = db.query(CustomKnowledge).filter(
                    CustomKnowledge.bot_id == request.bot_id,
                    CustomKnowledge.is_embedded == True,
                    CustomKnowledge.embedding_status == "completed"
                ).all()
                
                if not knowledge_entries:
                    return {"response": "I don't have any specific knowledge to summarize."}
                
                # Get relevant content using embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                query_embedding = embeddings.embed_query(request.user_request)
                relevant_chunks = []
                
                for knowledge in knowledge_entries:
                    vectors = db.query(KnowledgeVector).filter(
                        KnowledgeVector.knowledge_id == knowledge.id
                    ).all()
                    if not vectors:
                        continue
                    for vector in vectors:
                        chunk_embedding = json.loads(vector.embedding)
                        similarity = np.dot(query_embedding, chunk_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                        )
                        if similarity > 0.7:
                            relevant_chunks.append({
                                "text": vector.chunk_text,
                                "similarity": similarity
                            })
                
                relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
                context = "\n\n".join([chunk["text"] for chunk in relevant_chunks[:5]])
                
                # 2. Generate summary
                summary_prompt = f"Please summarize the following content concisely:\n\n{context}"
                summary = llm.invoke(summary_prompt)
                
                # 3. Now use Zapier to email the summary
                email_tools = [tool for tool in zapier_tools if "email" in tool["description"].lower()]
                
                if email_tools:
                    email_action = email_tools[0]
                    email_tool = Tool(
                        name=email_action["id"],
                        description=email_action["description"],
                        func=lambda instructions, action_id=email_action["id"]: zapier.run(
                            action_id, 
                            f"Send this email to {request.user_request.split('to ')[-1].strip()}. Subject: Document Summary. Body: {summary}"
                        )
                    )
                    
                    email_result = email_tool.run("")
                    response_message = f"I've summarized the document and sent it to {request.user_request.split('to ')[-1].strip()}.\n\nSummary:\n{summary}"
                    
                    conversation_store[request.session_id] = conversation_history + [
                        {"role": "user", "content": request.user_request},
                        {"role": "assistant", "content": response_message}
                    ]
                    return {"response": response_message, "sources": None}
            
            # If not hybrid or no email tools, proceed with regular agent
            tools = []
            for action in zapier_tools:
                # Fix: Create a proper closure for the lambda function
                def create_tool_func(action_id):
                    return lambda instructions: zapier.run(action_id, instructions)
                
                tool_func = create_tool_func(action["id"])
                
                tools.append(
                    Tool(
                        name=action["id"],
                        description=action["description"],
                        func=tool_func
                    )
                )
            
            # Add more specific system instructions for the agent
            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,  # Set to True for debugging
                handle_parsing_errors=True
            )
            
            # Provide more context to the agent
            agent_prompt = f"""
            You need to help with this request: "{request.user_request}"
            
            If this involves sending an email, make sure to:
            1. Specify the recipient email address clearly
            2. Include a subject line
            3. Write appropriate email content
            
            Choose the most appropriate tool for this task.
            """
            
            # Run the agent to perform the action
            agent_response = agent.run(agent_prompt)
            
            # Post-process the agent's response to make it more human-friendly
            if "final answer" in agent_response.lower():
                # Extract recipient from the original request
                recipient = ""
                if "to " in request.user_request:
                    recipient = request.user_request.split("to ")[-1].strip()
                
                # Create a more natural response
                if "email" in agent_response.lower() and recipient:
                    humanized_response = f"I've sent an email to {recipient} with the requested information."
                else:
                    humanized_response = "I've completed your request. The action has been performed successfully."
            else:
                humanized_response = agent_response
            
            conversation_store[request.session_id] = conversation_history + [
                {"role": "user", "content": request.user_request},
                {"role": "assistant", "content": humanized_response}
            ]
            if len(conversation_store[request.session_id]) > 10:
                conversation_store[request.session_id] = conversation_store[request.session_id][-10:]
            return {
                "response": humanized_response,
                "sources": None
            }

        # 1. Retrieve relevant knowledge for the bot
        knowledge_entries = db.query(CustomKnowledge).filter(
            CustomKnowledge.bot_id == request.bot_id,
            CustomKnowledge.is_embedded == True,
            CustomKnowledge.embedding_status == "completed"
        ).all()
        
        if not knowledge_entries:
            logger.warning(f"No knowledge entries found for bot {request.bot_id}")
            return {"response": "I don't have any specific knowledge to answer that question."}
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        query_embedding = embeddings.embed_query(request.user_request)
        relevant_chunks = []
        sources = []
        for knowledge in knowledge_entries:
            vectors = db.query(KnowledgeVector).filter(
                KnowledgeVector.knowledge_id == knowledge.id
            ).all()
            if not vectors:
                continue
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
                    source_titles = [s.get("title", "") for s in sources]
                    if knowledge.title not in source_titles:
                        sources.append({
                            "title": knowledge.title,
                            "id": knowledge.id
                        })
        relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks[:5]])
        messages = [{"role": "system", "content": f"""You are an AI assistant for a company. Answer the user's question based on the following context.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}
"""}]
        for msg in conversation_history:
            if "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": request.user_request})

        # --- FIX: Use LangChain OpenAI LLM for chat completion ---
        llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        # Convert messages to a single prompt string
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