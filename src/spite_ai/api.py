from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .jaden_ai import SpiteAI
from .config import Config


app = FastAPI(title="Spite AI API", version="0.1.0")

# Initialize the AI system once at startup
config = Config.from_env()
ai_system = SpiteAI(config)


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[str] = ""
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    # model: Optional[str] = None  # optional per-request model override - removed for simplicity


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Retrieve RAG context and generate a reply using the AI system."""
    try:
        # Use the AI system's methods directly
        context = ai_system.get_context(req.query, k=req.k, similarity_threshold=req.similarity_threshold)
        result = ai_system.generate_response(req.query, context, req.chat_history or "")
        
        return ChatResponse(response=result)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")



