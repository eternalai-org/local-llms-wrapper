"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""

import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any

# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "gpt-4-turbo"          # Default model for text-based chat completions
    VISION_MODEL = "gpt-4-vision-preview"  # Model used for vision-based requests
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Model used for generating embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Data Models
class Message(BaseModel):
    """
    Represents a single message in a chat completion request.
    """
    role: str      # The role of the message sender (e.g., 'user', 'assistant')
    content: Optional[Union[str, List[Dict[str, str]]]]  # The content of the message
 

class ToolCall(BaseModel):
    """
    Represents a tool call within a chat completion request.
    """
    type: str             # The type of tool call
    function: Dict[str, str]  # Details of the function to be called

class ChatCompletionRequest(BaseModel):
    """
    Model for chat completion requests, including messages, streaming option, and tools.
    """
    model: str = Config.TEXT_MODEL          # Model to use, defaults to text model
    messages: List[Message]                 # List of messages in the chat
    stream: Optional[bool] = False          # Whether to stream the response
    tools: Optional[Any] = None # Optional list of tools to use

    @validator("messages")
    def check_messages_not_empty(cls, v):
        """
        Ensure that the messages list is not empty.
        """
        if not v:
            raise ValueError("messages cannot be empty")
        return v
    
    def is_vision_request(self) -> bool:
        """
        Check if the request includes image content, indicating a vision-based request.
        If so, switch the model to the vision model.
        """
        for message in self.messages:
            content = message.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        self.model = Config.VISION_MODEL    
                        return True        
        return False

    def fix_message_order(self) -> None:
        """
        Ensure that messages alternate between 'user' and 'assistant' roles.
        If consecutive messages have the same role, insert a dummy message with the opposite role.
        """
        if not self.messages:
            return
            
        fixed_messages = []
        last_role = None
        
        for msg in self.messages:
            role = msg.role.strip()
            content = msg.content or " "
            
            # Insert opposite role if needed
            if (last_role in ("user", "assistant")) and role == last_role:
                fixed_messages.append(Message(
                    role="assistant" if last_role == "user" else "user",
                    content=" "
                ))
            
            fixed_messages.append(Message(role=role, content=content))
            last_role = role
            
        self.messages = fixed_messages

class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Config.EMBEDDING_MODEL     # Model to use, defaults to embedding model
    input: List[str] = Field(..., description="List of text inputs for embedding")  # Text inputs to embed

# Service Functions
class ServiceHandler:
    """
    Handler class for making requests to the underlying service.
    """
    @staticmethod
    async def get_service_port() -> int:
        """
        Retrieve the port of the underlying service from the app's state.
        """
        if not hasattr(app.state, "service_info") or "port" not in app.state.service_info:
            logger.error("Service information not set")
            raise HTTPException(status_code=503, detail="Service information not set")
        return app.state.service_info["port"]
    
    @staticmethod
    async def generate_text_response(request: ChatCompletionRequest):
        """
        Generate a response for chat completion requests, supporting both streaming and non-streaming.
        """
        port = await ServiceHandler.get_service_port()
        
        # Convert to dict, supporting both Pydantic v1 and v2
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()

        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        return await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_dict)

    @staticmethod
    async def generate_vision_response(request: ChatCompletionRequest):
        """
        Generate a response for vision-based chat completion requests.
        Supports a single message with exactly one text prompt and one image (base64 or URL).
        """
        # Check if the service supports multimodal inputs
        multimodal = app.state.service_info.get("multimodal", False)
        if not multimodal:
            raise HTTPException(status_code=400, detail="This model does not support vision-based requests")

        # Retrieve configuration values
        family = app.state.service_info["family"]
        cli = os.getenv(family)
        local_text_path = app.state.service_info["local_text_path"]
        local_projector_path = app.state.service_info["local_projector_path"]

        # Enforce a single message
        if len(request.messages) != 1:
            raise HTTPException(status_code=400, detail="Vision-based requests must contain exactly one message")

        # Process the content of the single message
        content = request.messages[0].content
        text = None
        image_url = None

        # Extract exactly one text and one image_url from the content list
        for item in content:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="Content items must be dictionaries")
            item_type = item.get("type")
            if item_type == "text":
                if text is not None:
                    raise HTTPException(status_code=400, detail="Only one text prompt is allowed in vision-based requests")
                text = item.get("text")
            elif item_type == "image_url":
                if image_url is not None:
                    raise HTTPException(status_code=400, detail="Only one image is allowed in vision-based requests")
                image_url = item.get("image_url", {}).get("url")
            else:
                raise HTTPException(status_code=400, detail="Invalid content type in vision-based request")

        # Validate that both text and image_url are present
        if text is None or image_url is None:
            raise HTTPException(status_code=400, detail="Vision-based requests must include one text prompt and one image")

        # Handle the image_url: base64 or URL
        if image_url.startswith("data:image/"):
            # Base64-encoded image
            try:
                header, encoded = image_url.split(",", 1)
                data = base64.b64decode(encoded)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(data)
                    image_path = temp_file.name
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
        else:
            # Regular URL
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url)
                    if response.status_code != 200:
                        raise HTTPException(status_code=400, detail="Failed to download image")
                    data = response.content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        temp_file.write(data)
                        image_path = temp_file.name
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

        # Construct and execute the command
        command = [
            cli,
            "--model", local_text_path,
            "--mmproj", local_projector_path,
            "--image", image_path,
            "--prompt", text
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_message = stderr.decode().strip() if stderr else "Unknown error"
                raise HTTPException(status_code=500, detail=f"Command failed: {error_message}")

            response = stdout.decode().strip()
            return response
        finally:
            # Clean up the temporary file
            if os.path.exists(image_path):
                os.remove(image_path)
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest):
        """
        Generate a response for embedding requests.
        """
        port = await ServiceHandler.get_service_port()
        return await ServiceHandler._make_api_call(port, "/v1/embeddings", request.dict())
        
    
    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: dict) -> dict:
        """
        Make a non-streaming API call to the specified endpoint and return the JSON response.
        """
        try:
            logger.info(f"Making API call to endpoint: {endpoint}")
            response = await app.state.client.post(
                f"http://localhost:{port}{endpoint}", 
                json=data,
                timeout=None  # Wait indefinitely for a response
            )
            logger.info(f"Received response with status code: {response.status_code}")

            
            if response.status_code != 200:
                logger.error(f"Error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
            return response.json()
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def _stream_generator(port: int, data: dict):
        """
        Generator for streaming responses from the service.
        Yields chunks of data as they are received, formatted for SSE (Server-Sent Events).
        """
        try:
            async with app.state.client.stream(
                "POST", 
                f"http://localhost:{port}/v1/chat/completions", 
                json=data
            ) as response:
                if response.status_code != 200:
                    error_msg = f"data: Error: {response.status_code} - {await response.text()}\n\n"
                    logger.error(f"Streaming error: {response.status_code} - {await response.text()}")
                    yield error_msg
                    return
                    
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: Error: {str(e)}\n\n"

# Request Processor
class RequestProcessor:
    """
    Class for processing requests asynchronously using a queue.
    Currently not utilized in the provided endpoints, possibly intended for future use.
    """
    queue = asyncio.Queue()  # Queue for asynchronous request processing
    endpoint_handlers = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
    }  # Mapping of endpoints to their request models and handlers
    
    # Global worker function
    @staticmethod
    async def worker():
        """
        Worker function to process requests from the queue asynchronously.
        Currently not actively used in the provided code.
        """
        while True:
            try:
                endpoint, request_data, future = await RequestProcessor.queue.get()
                
                if endpoint in RequestProcessor.endpoint_handlers:
                    model_cls, handler = RequestProcessor.endpoint_handlers[endpoint]
                    try:
                        request_obj = model_cls(**request_data)
                        result = await handler(request_obj)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                else:
                    future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
            except asyncio.CancelledError:
                break  # Exit the loop when the task is canceled
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler: initialize the HTTP client and start the worker task.
    """
    app.state.client = httpx.AsyncClient()  # Create an asynchronous HTTP client
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())  # Start the worker

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler: close the HTTP client and cancel the worker task.
    """
    await app.state.client.aclose()  # Close the HTTP client
    if hasattr(app.state, "worker_task"):
        app.state.worker_task.cancel()
        try:
            await app.state.worker_task  # Wait for the worker to finish
        except asyncio.CancelledError:
            pass  # Handle cancellation gracefully

# API Endpoints
@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns a simple status to indicate the service is running.
    """
    return {"status": "ok"}

# API Endpoints
@app.get("/v1/health")
async def health():
    """
    Health check endpoint.
    Returns a simple status to indicate the service is running.
    """
    return {"status": "ok"}

@app.post("/update")
async def update(request: dict):
    """
    Update the service information in the app's state.
    Stores the provided request data for use in determining the service port.
    """
    app.state.service_info = request
    return {"status": "ok"}

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests.
    Processes the request, checks for vision content, fixes message order, and generates the response.
    """
    logger.info(f"Received chat completion request: {request}")
    if not request.is_vision_request():  # Updates model if vision content is detected
        request.fix_message_order()   # Ensures proper user-assistant alternation
        return await ServiceHandler.generate_text_response(request)
    return await ServiceHandler.generate_vision_response(request)

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests.
    Generates embeddings using the specified model.
    """
    return await ServiceHandler.generate_embeddings_response(request)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests.
    Processes the request, checks for vision content, fixes message order, and generates the response.
    """
    logger.info(f"Received chat completion request: {request}")
    if not request.is_vision_request():  # Updates model if vision content is detected
        request.fix_message_order()   # Ensures proper user-assistant alternation
        return await ServiceHandler.generate_text_response(request)
    return await ServiceHandler.generate_vision_response(request)


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests.
    Generates embeddings using the specified model.
    """
    return await ServiceHandler.generate_embeddings_response(request)