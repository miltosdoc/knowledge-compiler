"""FROZEN v1 module - kept for reference only.

Superseded by the kc2/ package. Paths here are the original hardcoded
macOS paths and the stages do not connect (sieve emits .txt, distiller
reads .json). Do not build on this; see PLAN_v2.md.
The API key formerly hardcoded here was leaked publicly and must be
treated as compromised; credentials now come from the environment.
"""
import os

"""
Knowledge Compiler - Phase 4: The Gateway
FastAPI OpenAI-compatible proxy that injects compiled clinical intuition
into the system prompt before forwarding to any LLM.
"""
import json
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx

from compiler import compile_prompt, load_graph

BASE_DIR = Path("/Users/meditalks/knowledge-compiler")

# --- Configuration ---
API_KEY = os.environ["XSILICO_API_KEY"]
UPSTREAM_URL = "https://staging.xsilico.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.1"

app = FastAPI(title="Knowledge Compiler Gateway", version="0.1.0")

# Load graph at startup
graph = None

@app.on_event("startup")
async def startup():
    global graph
    try:
        graph = load_graph()
        print(f"Knowledge graph loaded: {graph['metadata']['total_notes']} notes")
    except Exception as e:
        print(f"Warning: Could not load graph: {e}")
        graph = None

# --- OpenAI-Compatible Models ---

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Custom field for Knowledge Compiler
    intuition_seed: Optional[str] = Field(None, description="Seed concept for clinical intuition compilation")

class Choice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": DEFAULT_MODEL, "object": "model", "owned_by": "knowledge-compiler"},
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global graph
    
    # 1. Determine the intuition seed
    seed = request.intuition_seed
    
    if not seed and request.messages:
        # Auto-detect seed from the last user message
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                seed = msg.content[:100]  # Use first 100 chars as seed
                break
    
    # 2. Compile the intuition prompt (if graph is available)
    compiled_intuition = ""
    if seed and graph:
        try:
            compiled_intuition = compile_prompt(seed, graph, max_tokens=8000)
        except Exception as e:
            compiled_intuition = f"[Intuition compilation error: {e}]"
    
    # 3. Inject compiled intuition into system prompt
    messages = []
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content or ""})
    
    # Prepend or merge with existing system message
    if compiled_intuition:
        system_content = compiled_intuition
        existing_system = None
        other_messages = []
        for m in messages:
            if m["role"] == "system":
                existing_system = m["content"]
            else:
                other_messages.append(m)
        
        if existing_system:
            system_content = existing_system + "\n\n" + compiled_intuition
        
        messages = [{"role": "system", "content": system_content}] + other_messages
    
    # 4. Forward to upstream LLM
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            payload = {
                "model": request.model,
                "messages": messages,
                "temperature": request.temperature,
            }
            if request.max_tokens:
                payload["max_tokens"] = request.max_tokens
            
            resp = await client.post(
                f"{UPSTREAM_URL}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            resp.raise_for_status()
            result = resp.json()
            
            # Add metadata about intuition compilation
            if compiled_intuition and "choices" in result and result["choices"]:
                result["choices"][0].get("message", {})["intuition_injected"] = bool(seed and graph)
            
            return result
            
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upstream error: {e}")

@app.get("/v1/compile/{seed}")
async def compile_endpoint(seed: str, max_tokens: int = 10000):
    """Direct endpoint to compile intuition for a given seed."""
    if not graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    
    compiled = compile_prompt(seed, graph, max_tokens=max_tokens)
    return {
        "seed": seed,
        "compiled_prompt": compiled,
        "estimated_tokens": len(compiled) // 4
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graph_loaded": graph is not None,
        "total_notes": graph["metadata"]["total_notes"] if graph else 0,
        "total_links": graph["metadata"]["total_links"] if graph else 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8199)
