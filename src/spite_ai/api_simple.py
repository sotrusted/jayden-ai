from typing import Optional
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spite AI API", version="0.1.0")

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[str] = ""
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
def health() -> dict:
    """Health check endpoint - no AI initialization."""
    return {"status": "ok", "message": "Spite AI API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Generate a response using RAG over Spite Magazine corpus."""
    try:
        # Import and initialize only when needed
        from .config import Config
        from .jaden_ai import get_context, generate_with_mistral
        
        logger.info(f"Processing chat request: {req.query[:50]}...")
        
        # Get context
        context = get_context(req.query, k=req.k, similarity_threshold=req.similarity_threshold)
        logger.info("Context retrieved successfully")
        
        # Generate response - use the global client if available
        config = Config.from_env()
        if config.API_MODE:
            # For API mode, we need to create a client
            import groq
            client = groq.Groq(api_key=config.GROQ_API_KEY)
            result = generate_with_mistral(req.query, context, req.chat_history or "", client)
        else:
            result = generate_with_mistral(req.query, context, req.chat_history or "")
        
        logger.info("Response generated successfully")
        return ChatResponse(response=result)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(response=f"Sorry, I encountered an error: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with basic info."""
    return {
        "message": "Spite AI API", 
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
