# Spite AI API Usage Guide

## Quick Start

### 1. Environment Setup

Set your Groq API key:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Get your API key from: https://console.groq.com/

### 2. Start the Server

```bash
uv run uvicorn src.spite_ai.api_simple:app --host 0.0.0.0 --port 8080
```

The server will start and load:
- Lorebook for retrieval boosting
- Sentence transformer model (all-mpnet-base-v2)
- ~36K corpus entries from Spite Magazine
- Style profile and system prompt

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"ok"}`

**Chat Endpoint:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is Jayden?",
    "chat_history": ""
  }'
```

**Advanced Chat with Parameters:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Dimes Square?",
    "chat_history": "User: Hi\nAssistant: Hello! What would you like to know about Spite Magazine?",
    "k": 10,
    "similarity_threshold": 0.7
  }'
```

## API Endpoints

### GET /health
Returns server health status.

**Response:**
```json
{"status": "ok"}
```

### POST /chat
Generate a response using RAG over Spite Magazine corpus.

**Request Body:**
```json
{
  "query": "string",           // Required: Your question
  "chat_history": "string",    // Optional: Previous conversation
  "k": "int",                  // Optional: Number of passages to retrieve
  "similarity_threshold": "float", // Optional: Minimum similarity score
  "model": "string"            // Optional: Override model for this request
}
```

**Response:**
```json
{
  "response": "string"         // Generated response from Spite AI
}
```

### POST /chat/stream
Generate a streaming response using RAG over Spite Magazine corpus. This endpoint returns a Server-Sent Events (SSE) stream that can be listened to for real-time response generation.

**Request Body:**
```json
{
  "query": "string",           // Required: Your question
  "chat_history": "string",    // Optional: Previous conversation
  "k": "int",                  // Optional: Number of passages to retrieve
  "similarity_threshold": "float" // Optional: Minimum similarity score
}
```

**Response:**
Server-Sent Events stream with `Content-Type: text/event-stream`. Each event contains:
```
data: {"content": "partial response text"}

data: {"content": "more response text"}

data: {"done": true}
```

**Example Usage:**
```bash
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is Jayden?",
    "chat_history": ""
  }' \
  --no-buffer
```

**JavaScript Example:**
```javascript
const response = await fetch('http://localhost:8080/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Who is Jayden?',
    chat_history: ''
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.content) {
        console.log(data.content);
      } else if (data.done) {
        console.log('Stream finished');
      }
    }
  }
}
```

## Configuration

The API uses environment variables for configuration:

```bash
# Required
export GROQ_API_KEY="your_api_key"

# Optional overrides
export SPITE_OLLAMA_MODEL="llama-3.3-70b-versatile"
export SPITE_OLLAMA_TEMPERATURE="0.2"
export SPITE_SLANG_DENSITY="mid"
export SPITE_USE_CITATIONS="true"
export SPITE_DEFAULT_K="8"
export SPITE_SIMILARITY_THRESHOLD="0.65"
```

## Deployment

### Local Development
```bash
uv run uvicorn src.spite_ai.api:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uv run uvicorn src.spite_ai.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Note: Use only 1 worker if you plan to use per-request model overrides.

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.spite_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

1. **Server won't start**: Ensure `GROQ_API_KEY` is set
2. **Empty responses**: Check that data files exist in `/data` directory
3. **Memory issues**: The model loads ~36K embeddings, ensure adequate RAM
4. **Slow responses**: First request may be slower due to model loading

## Features

- **RAG-powered responses**: Searches through Spite Magazine corpus
- **Terminally online persona**: Uses in-jokes, slang, and community references
- **Lorebook integration**: Boosts retrieval for Spite-specific terms
- **Citation support**: Optional passage citations in responses
- **Configurable parameters**: Adjust retrieval and generation settings
- **Chat history support**: Maintains conversation context
