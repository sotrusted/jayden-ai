from typing import Optional
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spite AI API", version="0.1.0")

# Global AI system - will be initialized on first request
_ai_system = None
_config = None

def get_ai_system():
    """Get or initialize the AI system singleton."""
    global _ai_system, _config
    if _ai_system is None:
        logger.info("Initializing Spite AI system...")
        _config = Config.from_env()
        
        # Import here to avoid circular imports and double initialization
        from .jaden_ai import SpiteAI
        _ai_system = SpiteAI(_config)
        logger.info("Spite AI system initialized successfully")
    
    return _ai_system, _config

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[str] = ""
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Generate a response using RAG over Spite Magazine corpus."""
    try:
        ai_system, config = get_ai_system()
        
        # Get context using the AI system's method
        context = ai_system.get_context(req.query, k=req.k, similarity_threshold=req.similarity_threshold)
        
        # Generate response
        result = ai_system.generate_response(req.query, context, req.chat_history or "")
        
        return ChatResponse(response=result)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
