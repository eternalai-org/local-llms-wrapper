import os
import json
import pickle
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import asyncio

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

# Load service info at startup
@app.on_event("startup")
async def startup_event():
    running_service_file = os.getenv("RUNNING_SERVICE_FILE")
    if not running_service_file:
        raise Exception("Environment variable RUNNING_SERVICE_FILE not set")
    try:
        with open(running_service_file, "rb") as f:
            app.state.service_info = pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load service info: {str(e)}")

# # Asynchronous image download function
# async def download_image(url: str, path: str):
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.get(url) as response:
#                 if response.status == 200:
#                     async with aiofiles.open(path, 'wb') as f:
#                         await f.write(await response.read())
#                 else:
#                     raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status}")
#     except aiohttp.ClientError as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

# @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
# async def chat_completions(request: ChatCompletionRequest):
#     service_info = app.state.service_info
    

#     multimodal = service_info.get("multimodal", False)
#     if not multimodal:
#         raise HTTPException(status_code=400, detail="Only support multimodal for this API endpoint")

#     if len(request.messages) > 1:
#         raise HTTPException(status_code=400, detail="Only support single message for now")

#     family = service_info["family"]
#     local_text_path = service_info["local_text_path"]
#     local_projector_path = service_info["local_projector_path"]

#     # Create temporary file for image
#     temp_file = tempfile.NamedTemporaryFile(delete=False)
#     local_image_path = temp_file.name
#     prompt = ""

#     # Process message content
#     for message_content in request.messages[0].content:
#         if message_content.type == "text":
#             prompt = message_content.text
#         elif message_content.type == "image_url":
#             url = message_content.image_url["url"]
#             await download_image(url, local_image_path)

#     cli = os.getenv(family)
#     if not cli:
#         raise HTTPException(status_code=503, detail=f"CLI for family '{family}' not found in environment variables")

#     if not request.stream:
#         # Handle non-streaming case (unimplemented as in original)
#         pass
#     else:
#         async def stream_output():
#             try:
#                 start_return = False
#                 cmd = [
#                     cli, "--model", local_text_path, "--mmproj", local_projector_path,
#                     "--image", local_image_path, "-p", prompt
#                 ]
#                 print(f"Running command: {' '.join(cmd)}")
#                 proc = await asyncio.create_subprocess_exec(
#                     *cmd,
#                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
#                 )
#                 while True:
#                     line = await proc.stdout.readline()
#                     if not line:
#                         break
#                     output = line.decode("utf-8").strip()
#                     if "Image decoded in" in output:
#                         start_return = True
#                     if start_return:
#                         yield f"data: {json.dumps({'choices': [{'delta': {'content': output}}]})}\n\n"
#                 yield "data: [DONE]\n\n"
#                 await proc.wait()
#             finally:
#                 # Ensure temporary file is deleted
#                 os.remove(local_image_path)

#         return StreamingResponse(stream_output(), media_type="text/event-stream")

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

# --------------------------------------
# 🚀 Dummy Function to Simulate Response
# --------------------------------------

async def generate_text_response(request: ChatCompletionRequest):
    """Simulates a text model response (replace with real model call)."""
    try:
        port = app.state.service_info["port"]
    except KeyError:
        raise HTTPException(status_code=503, detail="Port not found in service info")
    if request.stream:
        async def stream_generator():
            async with httpx.AsyncClient() as client:
               async with client.stream("POST", "http://localhost:11535/v1/chat/completions", json=request.dict()) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    chat_completions = requests.post(f"http://localhost:{port}/v1/chat/completions", json=request.dict())
    return chat_completions.json()

async def generate_vision_response(messages: List[Message], stream: bool):
    """Simulates a vision model response."""

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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handles text-based chat completions, including streaming and tool calls."""


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