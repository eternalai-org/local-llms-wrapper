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
from typing import List, Dict, Optional

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
    content: str   # The content of the message

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
    tools: Optional[List[ToolCall]] = None  # Optional list of tools to use

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
            content = msg.content.strip()
            if last_role in {"user", "assistant"} and role == last_role:
                opposite_role = "assistant" if last_role == "user" else "user"
                fixed_messages.append(Message(role=opposite_role, content=" "))
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

        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request.dict()),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        return await ServiceHandler._make_api_call(port, "/v1/chat/completions", request.dict())
    
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
                        yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"  # Signal the end of the stream
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

@app.post("/v1/update")
async def update(request: dict):
    """
    Update the service information in the app's state.
    Stores the provided request data for use in determining the service port.
    """
    app.state.service_info = request
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests.
    Processes the request, checks for vision content, fixes message order, and generates the response.
    """
    logger.info(f"Received chat completion request: {request}")
    request.is_vision_request()  # Updates model if vision content is detected
    request.fix_message_order()   # Ensures proper user-assistant alternation
    return await ServiceHandler.generate_text_response(request)

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests.
    Generates embeddings using the specified model.
    """
    return await ServiceHandler.generate_embeddings_response(request)