from typing import Optional, Generator, List
import logging
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spite AI API", version="0.1.0")

# Initialize the AI system once at startup
ai_system = None
config = None

@app.on_event("startup")
async def startup_event():
    """Initialize AI system on startup."""
    global ai_system, config
    try:
        from .config import Config
        from .jaden_ai import SpiteAI
        
        logger.info("Initializing AI system...")
        config = Config.from_env()
        ai_system = SpiteAI(config)
        logger.info("AI system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up AI system on shutdown."""
    global ai_system
    if ai_system is not None:
        try:
            ai_system.cleanup()
            logger.info("AI system cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[str] = ""
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    citations: Optional[List[dict]] = None

@app.get("/health")
def health() -> dict:
    """Health check endpoint - no AI initialization."""
    return {"status": "ok", "message": "Spite AI API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Generate a response using RAG over Spite Magazine corpus."""
    try:
        if ai_system is None:
            raise HTTPException(status_code=503, detail="AI system not initialized")
        
        logger.info(f"Processing chat request: {req.query[:50]}...")
        
        # Use the pre-initialized AI system with metadata
        context, citation_metadata = ai_system.get_context_with_metadata(req.query, k=req.k, similarity_threshold=req.similarity_threshold)
        logger.info("Context retrieved successfully")
        
        # Generate response using the AI system (non-streaming)
        result = ai_system.generate_response(req.query, context, req.chat_history or "", stream=False)
        
        # Add citation numbers to metadata
        citations = []
        for i, meta in enumerate(citation_metadata, 1):
            citation = {
                "citation_number": i,
                "type": meta["type"],  # "post" or "comment"
                "id": meta["id"],
                "url": f"/{meta['type']}/{meta['id']}"  # Frontend can customize this
            }
            # Add additional fields based on type
            if meta["type"] == "post":
                citation.update({
                    "title": meta.get("title", ""),
                    "author": meta.get("author"),
                    "display_name": meta.get("display_name", "")
                })
            elif meta["type"] == "comment":
                citation.update({
                    "name": meta.get("name", ""),
                    "post_id": meta.get("post_id")
                })
            citations.append(citation)
        
        logger.info("Response generated successfully")
        return ChatResponse(response=result, citations=citations)
    
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
            "chat": "/chat (POST)",
            "chat_stream": "/chat/stream (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def generate_stream_response(stream_generator) -> Generator[str, None, None]:
    """Convert streaming response to Server-Sent Events format."""
    try:
        for chunk in stream_generator:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    # Format as Server-Sent Events
                    data = json.dumps({"content": delta.content})
                    yield f"data: {data}\n\n"
        
        # Signal end of stream
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"


def generate_stream_response_with_citations(stream_generator, citations: List[dict]) -> Generator[str, None, None]:
    """Convert streaming response to Server-Sent Events format with citations."""
    try:
        # First, send the citations
        if citations:
            citations_data = json.dumps({"citations": citations})
            yield f"data: {citations_data}\n\n"
        
        # Validate stream generator
        if stream_generator is None:
            yield f"data: {json.dumps({'error': 'No stream generator provided'})}\n\n"
            return
            
        logger.info("Starting to iterate over stream generator")
        chunk_count = 0
        
        # Then stream the content
        for chunk in stream_generator:
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count}, type: {type(chunk)}")
            
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    # Format as Server-Sent Events
                    data = json.dumps({"content": delta.content})
                    yield f"data: {data}\n\n"
                    logger.info(f"Yielded content chunk: {delta.content[:50]}...")
            else:
                logger.warning(f"Chunk {chunk_count} has no valid content: {chunk}")
        
        logger.info(f"Stream completed after {chunk_count} chunks")
        # Signal end of stream
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generator: {e}")
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream a response using RAG over Spite Magazine corpus."""
    try:
        if ai_system is None:
            error_data = json.dumps({"error": "AI system not initialized"})
            error_stream = iter([f"data: {error_data}\n\n"])
            return StreamingResponse(
                error_stream,
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive", 
                    "Content-Type": "text/event-stream",
                }
            )
        
        logger.info(f"Processing streaming chat request: {req.query[:50]}...")
        
        # Use the pre-initialized AI system with metadata
        context, citation_metadata = ai_system.get_context_with_metadata(req.query, k=req.k, similarity_threshold=req.similarity_threshold)
        logger.info("Context retrieved successfully")
        
        # Prepare citation metadata
        citations = []
        for i, meta in enumerate(citation_metadata, 1):
            citation = {
                "citation_number": i,
                "type": meta["type"],  # "post" or "comment"
                "id": meta["id"],
                "url": f"/{meta['type']}/{meta['id']}"  # Frontend can customize this
            }
            # Add additional fields based on type
            if meta["type"] == "post":
                citation.update({
                    "title": meta.get("title", ""),
                    "author": meta.get("author"),
                    "display_name": meta.get("display_name", "")
                })
            elif meta["type"] == "comment":
                citation.update({
                    "name": meta.get("name", ""),
                    "post_id": meta.get("post_id")
                })
            citations.append(citation)
        
        # Generate streaming response using the AI system
        logger.info("Requesting streaming response from AI system")
        stream_generator = ai_system.generate_response(req.query, context, req.chat_history or "", stream=True)
        
        logger.info("Starting streaming response")
        logger.info(f"Stream generator type: {type(stream_generator)}")
        
        # Return streaming response with citations
        return StreamingResponse(
            generate_stream_response_with_citations(stream_generator, citations),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {e}")
        # Return error as stream
        error_data = json.dumps({"error": f"Sorry, I encountered an error: {str(e)}"})
        error_stream = iter([f"data: {error_data}\n\n"])
        return StreamingResponse(
            error_stream,
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive", 
                "Content-Type": "text/event-stream",
            }
        )

