import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, constr, validator
from typing import List, Dict, Optional
import asyncio
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Default Models
DEFAULT_TEXT_MODEL = "gpt-4-turbo"
DEFAULT_VISION_MODEL = "gpt-4-vision-preview"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

# Request Schemas
class Message(BaseModel):
    role: str
    content: str

class ToolCall(BaseModel):
    type: str
    function: Dict[str, str]

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_TEXT_MODEL
    messages: List[Message]
    stream: Optional[bool] = False
    tools: Optional[List[ToolCall]] = None

    @validator("messages")
    def check_messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v

    def fix_message_order(self):
        if not self.messages:
            return
        fixed_messages = []
        last_role = None
        for msg in self.messages:
            role = msg.role.strip()
            content = msg.content.strip()
            if last_role in {"user", "assistant"} and role == last_role:
                opposite_role = "assistant" if last_role == "user" else "user"
                fixed_messages.append(Message(role=opposite_role, content="."))
            fixed_messages.append(Message(role=role, content=content))
            last_role = role
        self.messages = fixed_messages

class VisionChatCompletionRequest(BaseModel):
    model: str = DEFAULT_VISION_MODEL
    messages: List[Message]
    stream: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    model: str = DEFAULT_EMBEDDING_MODEL
    input: List[str] = Field(..., description="List of text inputs for embedding")

# Response Generators
async def generate_text_response(request: ChatCompletionRequest):
    if not hasattr(app.state, "service_info") or "port" not in app.state.service_info:
        logger.error("Service information not set")
        raise HTTPException(status_code=503, detail="Service information not set")
    port = app.state.service_info["port"]

    if request.stream:
        async def stream_generator():
            try:
                async with app.state.client.stream("POST", f"http://localhost:{port}/v1/chat/completions", json=request.dict()) as response:
                    if response.status_code != 200:
                        error_msg = f"data: Error: {response.status_code} - {await response.text()}\n\n"
                        logger.error(f"Streaming error: {response.status_code} - {await response.text()}")
                        yield error_msg
                        return
                    async for line in response.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield f"data: Error: {str(e)}\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    response = await app.state.client.post(f"http://localhost:{port}/v1/chat/completions", json=request.dict())
    logger.info(f"Received response with status code: {response.status_code}")
    if response.status_code != 200:
        logger.error(f"Non-streaming error: {response.status_code} - {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

async def generate_vision_response(messages: List[Message], stream: bool):
    response_text = "dummy result"
    if stream:
        async def stream_generator():
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

# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/update")
async def update(request: dict):
    app.state.service_info = request
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"Received chat completion request: {request}")
    request.fix_message_order()
    return await generate_text_response(request)

@app.post("/v1/chat/completions/vision")
async def vision_chat_completions(request: VisionChatCompletionRequest):
    return await generate_vision_response(request.messages, request.stream)

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    dummy_vector_size = 768
    embeddings = [[0.01] * dummy_vector_size for _ in request.input]
    return JSONResponse({
        "object": "list",
        "data": [{"embedding": e} for e in embeddings]
    })