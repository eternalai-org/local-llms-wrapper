import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import httpx
import asyncio
from fastapi.responses import JSONResponse
import uuid
from typing import AsyncGenerator

app = FastAPI()

# -------------------------
# 🚀 Define Default Models
# -------------------------
DEFAULT_TEXT_MODEL = "gpt-4-turbo"
DEFAULT_VISION_MODEL = "gpt-4-vision-preview"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# -------------------------
# 🚀 Define Request Schemas
# -------------------------

class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class ToolCall(BaseModel):
    type: str  # e.g., "function"
    function: Dict[str, str]  # {"name": "...", "arguments": "..."}

class ChatCompletionRequest(BaseModel):
    model: str  = DEFAULT_TEXT_MODEL
    messages: List[Message]
    stream: Optional[bool] = False
    tools: Optional[List[ToolCall]] = None

    def fix_message_order(self):
        """Auto-corrects message order issues by inserting missing roles with empty content for consecutive user or assistant messages."""
        if not self.messages:
            return

        fixed_messages = []
        last_role = None

        for msg in self.messages:
            role = msg.role.strip() if msg.role else ""  # Ensure non-null role
            content = msg.content.strip() if msg.content else ""  # Ensure non-null content

            # Check for consecutive user or assistant messages
            if last_role in {"user", "assistant"} and role == last_role:
                # Insert opposite role with empty content
                opposite_role = "assistant" if last_role == "user" else "user"
                fixed_messages.append(Message(role=opposite_role, content=""))

            fixed_messages.append(Message(role=role, content=content))
            last_role = role  # Update last_role to the current role

        self.messages = fixed_messages

class VisionChatCompletionRequest(BaseModel):
    model: str = DEFAULT_VISION_MODEL
    messages: List[Message]
    stream: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    model: str = DEFAULT_EMBEDDING_MODEL
    input: List[str] = Field(..., description="List of text inputs for embedding")


async def generate_text_response(request: ChatCompletionRequest):
    """Simulates a text model response (replace with real model call)."""
    try:
        port = app.state.service_info["port"]
    except KeyError:
        raise HTTPException(status_code=503, detail="Port not found in service info")
    if request.stream:
        async def stream_generator():
            async with httpx.AsyncClient() as client:
               async with client.stream("POST", f"http://localhost:{port}/v1/chat/completions", json=request.dict()) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    chat_completions = requests.post(f"http://localhost:{port}/v1/chat/completions", json=request.dict())
    return chat_completions.json()

async def generate_vision_response(messages: List[Message], stream: bool):
    """Simulates a vision model response."""
    response_text = "dummy result"
    if stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            for word in response_text.split():
                yield f"data: {word}\n\n"
                await asyncio.sleep(0.1)
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    return JSONResponse({
        "id": f"visioncmpl-{uuid.uuid4()}",
        "object": "vision.completion",
        "choices": [{"message": {"role": "assistant", "content": response_text}}]
    })

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/update")
async def update(request: dict):
    app.state.service_info = request
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handles text-based chat completions, including streaming and tool calls."""
    request.fix_message_order()
    print(app.state.service_info)
    print(request)
    return await generate_text_response(request)


@app.post("/v1/chat/completions/vision")
async def vision_chat_completions(request: VisionChatCompletionRequest):
    """Handles vision-based chat completions, including streaming."""
    return await generate_vision_response(request.messages, request.stream)


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Handles embedding requests and returns a dummy embedding vector."""
    dummy_vector_size = 768  # Example: OpenAI's embedding dimension
    embeddings = [[0.01] * dummy_vector_size for _ in request.input]  

    return JSONResponse({
        "object": "list",
        "data": [{"embedding": e} for e in embeddings]
    })