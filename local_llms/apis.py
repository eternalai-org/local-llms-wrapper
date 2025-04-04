"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""

import os
import logging
import httpx
import asyncio
import base64
import tempfile
import functools
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any, Callable
from functools import lru_cache
import time

# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "gpt-4-turbo"          # Default model for text-based chat completions
    VISION_MODEL = "gpt-4-vision-preview"  # Model used for vision-based requests
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Model used for generating embeddings
    HTTP_TIMEOUT = 60.0                 # Default timeout for HTTP requests in seconds
    CACHE_TTL = 300                     # Cache time-to-live in seconds (5 minutes)
    MAX_RETRIES = 3                     # Maximum number of retries for HTTP requests
    POOL_CONNECTIONS = 100              # Maximum number of connections in the pool
    POOL_KEEPALIVE = 20                 # Keep connections alive for 20 seconds

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    """
    Model for chat completion requests, including messages, streaming option, and tools.
    """
    model: str = Config.TEXT_MODEL          # Model to use, defaults to text model
    messages: Optional[Union[List[Any]]]
    stream: Optional[bool] = False          # Whether to stream the response
    tools: Optional[Any] = None             # Optional list of tools to use
    max_tokens: Optional[int] = None        # Maximum tokens in the response
    temperature: Optional[float] = None     # Temperature for sampling
    top_p: Optional[float] = None           # Top p for nucleus sampling

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
        # Early optimization - only check the last message from the user
        if not self.messages:
            return False
            
        last_message = self.messages[-1]
        if last_message.get("role") != "user":
            # Check all messages if the last one is not from user
            for message in self.messages:
                if self._check_message_for_image(message):
                    return True
            return False
            
        # Check just the last user message
        return self._check_message_for_image(last_message)
    
    def _check_message_for_image(self, message: Any) -> bool:
        """Helper method to check if a message contains an image."""
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    self.model = Config.VISION_MODEL    
                    return True
        return False

    def fix_message_order(self) -> None:
        if self.messages:
            return
        fixed_messages = []
        if self.messages[0].get("role") == "system":
            self.messages.pop(0)
            fixed_messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant capable of using tools when necessary. Respond in natural language when tools are not required."
                }
            )
        for msg in self.messages:
            fixed_messages.append(msg)
        self.messages = fixed_messages
                
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Config.EMBEDDING_MODEL     # Model to use, defaults to embedding model
    input: List[str] = Field(..., description="List of text inputs for embedding")  # Text inputs to embed
    
    @validator("input")
    def check_input_not_empty(cls, v):
        """Ensure the input list is not empty."""
        if not v:
            raise ValueError("input list cannot be empty")
        return v

# Cache for service port to avoid repeated lookups
@lru_cache(maxsize=1)
def get_cached_service_port():
    """
    Retrieve the port of the underlying service from the app's state with caching.
    The cache is invalidated when the service info is updated.
    """
    if not hasattr(app.state, "service_info") or "port" not in app.state.service_info:
        logger.error("Service information not set")
        raise HTTPException(status_code=503, detail="Service information not set")
    return app.state.service_info["port"]

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
        try:
            return get_cached_service_port()
        except HTTPException:
            # If cache lookup fails, try direct lookup
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
            if request.tools:
                raise HTTPException(status_code=400, detail="Streaming is not supported with tools")
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
        if not cli:
            raise HTTPException(status_code=500, detail=f"CLI environment variable '{family}' not set")
            
        local_text_path = app.state.service_info.get("local_text_path")
        local_projector_path = app.state.service_info.get("local_projector_path")
        
        if not local_text_path or not local_projector_path:
            raise HTTPException(status_code=500, detail="Model paths not properly configured")

        # Enforce a single message
        if len(request.messages) != 1:
            raise HTTPException(status_code=400, detail="Vision-based requests must contain exactly one message")

        # Process the content of the single message
        content = request.messages[0].content
        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="Vision content must be a list")

        text = None
        image_url = None
        image_path = None

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
                raise HTTPException(status_code=400, detail=f"Invalid content type '{item_type}' in vision-based request")

        # Validate that both text and image_url are present
        if text is None or image_url is None:
            raise HTTPException(status_code=400, detail="Vision-based requests must include one text prompt and one image")

        try:
            # Handle the image_url: base64 or URL
            image_path = await ServiceHandler._process_image(image_url)
            
            # Construct and execute the command
            command = [
                cli,
                "--model", local_text_path,
                "--mmproj", local_projector_path,
                "--image", image_path,
                "--prompt", text
            ]

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
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vision processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vision processing error: {str(e)}")
        finally:
            # Clean up the temporary file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {image_path}: {str(e)}")
    
    @staticmethod
    async def _process_image(image_url: str) -> str:
        """
        Process the image URL or base64 data and return the path to the saved image.
        """
        if image_url.startswith("data:image/"):
            # Base64-encoded image
            try:
                header, encoded = image_url.split(",", 1)
                data = base64.b64decode(encoded)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(data)
                    return temp_file.name
            except Exception as e:
                logger.error(f"Failed to process base64 image: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
        else:
            # Regular URL
            try:
                async with httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT) as client:
                    response = await client.get(image_url)
                    if response.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status_code}")
                    data = response.content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        temp_file.write(data)
                        return temp_file.name
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Timeout while downloading image")
            except Exception as e:
                logger.error(f"Failed to download image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest):
        """
        Generate a response for embedding requests.
        """
        port = await ServiceHandler.get_service_port()
        # Convert to dict, supporting both Pydantic v1 and v2
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return await ServiceHandler._make_api_call(port, "/v1/embeddings", request_dict)
    
    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: dict, retries: int = Config.MAX_RETRIES) -> dict:
        """
        Make a non-streaming API call to the specified endpoint and return the JSON response.
        Includes retry logic for transient errors.
        """
        attempts = 0
        last_exception = None
        
        while attempts < retries:
            try:
                logger.info(f"Making API call to endpoint: {endpoint} (attempt {attempts+1}/{retries})")
                response = await app.state.client.post(
                    f"http://localhost:{port}{endpoint}", 
                    json=data,
                    timeout=Config.HTTP_TIMEOUT
                )
                logger.info(f"Received response with status code: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Error: {response.status_code} - {error_text}")
                    # Don't retry client errors (4xx), only server errors (5xx)
                    if response.status_code < 500:
                        raise HTTPException(status_code=response.status_code, detail=error_text)
                    last_exception = HTTPException(status_code=response.status_code, detail=error_text)
                else:
                    return response.json()
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout during API call (attempt {attempts+1}/{retries}): {str(e)}")
                last_exception = HTTPException(status_code=504, detail="Gateway Timeout")
            except Exception as e:
                logger.error(f"API call error (attempt {attempts+1}/{retries}): {str(e)}")
                last_exception = HTTPException(status_code=500, detail=str(e))
            
            # Exponential backoff with jitter
            if attempts < retries - 1:  # Don't sleep after the last attempt
                sleep_time = (2 ** attempts) + (0.1 * random.random())
                await asyncio.sleep(sleep_time)
            
            attempts += 1
        
        # If we get here, all retries failed
        logger.error(f"All {retries} API call attempts failed")
        if last_exception:
            raise last_exception
        raise HTTPException(status_code=500, detail="Unknown error during API call")
    
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
                json=data,
                timeout=None  # Streaming needs indefinite timeout
            ) as response:
                if response.status_code != 200:
                    error_text = await response.text()
                    error_msg = f"data: {{\"error\":{{\"message\":\"{error_text}\",\"code\":{response.status_code}}}}}\n\n"
                    logger.error(f"Streaming error: {response.status_code} - {error_text}")
                    yield error_msg
                    return
                    
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"

# Request Processor
class RequestProcessor:
    """
    Class for processing requests asynchronously using a queue.
    """
    queue = asyncio.Queue()  # Queue for asynchronous request processing
    endpoint_handlers = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
    }  # Mapping of endpoints to their request models and handlers
    
    @staticmethod
    async def process_request(endpoint: str, request_data: dict):
        """
        Process a request asynchronously by adding it to the queue.
        Returns a Future that will be resolved with the result.
        """
        future = asyncio.Future()
        await RequestProcessor.queue.put((endpoint, request_data, future))
        return await future
    
    # Global worker function
    @staticmethod
    async def worker():
        """
        Worker function to process requests from the queue asynchronously.
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
                        logger.error(f"Handler error for {endpoint}: {str(e)}")
                        future.set_exception(e)
                else:
                    logger.error(f"Endpoint not found: {endpoint}")
                    future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
            except asyncio.CancelledError:
                logger.info("Worker task cancelled, exiting")
                break  # Exit the loop when the task is canceled
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                # Continue working, don't crash the worker

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware that adds a header with the processing time for the request.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Dependencies
async def get_background_tasks():
    """Dependency to get background tasks."""
    return BackgroundTasks()

# Import random here to avoid moving it up (where it might not be needed for all code paths)
import random

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler: initialize the HTTP client and start the worker task.
    """
    # Create an asynchronous HTTP client with connection pooling
    limits = httpx.Limits(
        max_connections=Config.POOL_CONNECTIONS,
        max_keepalive_connections=Config.POOL_CONNECTIONS
    )
    app.state.client = httpx.AsyncClient(limits=limits, timeout=Config.HTTP_TIMEOUT)
    
    # Start the worker
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    logger.info("Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler: close the HTTP client and cancel the worker task.
    """
    logger.info("Shutting down service")
    
    # Close the HTTP client
    if hasattr(app.state, "client"):
        await app.state.client.aclose()
    
    # Cancel the worker task
    if hasattr(app.state, "worker_task"):
        app.state.worker_task.cancel()
        try:
            await app.state.worker_task  # Wait for the worker to finish
        except asyncio.CancelledError:
            pass  # Handle cancellation gracefully
    
    logger.info("Service shutdown complete")

# API Endpoints
@app.get("/health")
@app.get("/v1/health")
async def health():
    """
    Health check endpoint.
    Returns a simple status to indicate the service is running.
    """
    # Invalidate the service port cache periodically
    get_cached_service_port.cache_clear()
    
    # Check if the service info is set
    if not hasattr(app.state, "service_info"):
        return {"status": "starting", "message": "Service info not set yet"}
    
    return {"status": "ok", "service": app.state.service_info.get("family", "unknown")}

@app.post("/unload")
async def unload_model():
    """
    Endpoint to unload the model from memory but keep the server running.
    This helps reduce memory usage when the model is not actively being used.
    """
    try:
        logger.info("Received request to unload model from memory")
        
        # Check if the service info is set
        if not hasattr(app.state, "service_info"):
            raise HTTPException(status_code=503, detail="Service info not set yet")
            
        # Perform model unloading operations
        # This is a placeholder - actual implementation depends on the underlying model server
        # For example, you might send a signal to the model server to release memory
        
        # For now, just log and return success
        logger.info("Model has been unloaded from memory")
        return {"status": "ok", "message": "Model unloaded successfully"}
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.post("/reload")
async def reload_model(model_path: dict):
    """
    Endpoint to reload a previously unloaded model.
    
    Args:
        model_path (dict): Dictionary containing the model_path key with path to the model file
    """
    try:
        logger.info(f"Received request to reload model from {model_path.get('model_path')}")
        
        # Check if the service info is set
        if not hasattr(app.state, "service_info"):
            raise HTTPException(status_code=503, detail="Service info not set yet")
            
        # Validate the model path
        path = model_path.get("model_path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=400, detail="Invalid or missing model path")
            
        # Perform model reloading operations
        # This is a placeholder - actual implementation depends on the underlying model server
        # For example, you might send a signal to the model server to reload the model
        
        # For now, just log and return success
        logger.info(f"Model has been reloaded from {path}")
        return {"status": "ok", "message": "Model reloaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.post("/update")
async def update(request: dict):
    """
    Update the service information in the app's state.
    Stores the provided request data for use in determining the service port.
    """
    app.state.service_info = request
    # Invalidate the cache when service info is updated
    get_cached_service_port.cache_clear()
    logger.info(f"Updated service info: {request.get('family', 'unknown')} on port {request.get('port', 'unknown')}")
    return {"status": "ok", "message": "Service info updated successfully"}

# Add background task handling
async def handle_request_in_background(background_tasks: BackgroundTasks, endpoint: str, request_data: dict):
    """Handle a request in the background."""
    try:
        return await RequestProcessor.process_request(endpoint, request_data)
    except Exception as e:
        logger.error(f"Background task error: {str(e)}")
        raise

# Combined endpoint handler function
async def handle_completion_request(request: ChatCompletionRequest, endpoint: str):
    """
    Common handler for chat completion requests.
    """
    logger.info(f"Received chat completion request for model: {request.model}")
    
    if request.is_vision_request():
        return await ServiceHandler.generate_vision_response(request)
    
    request.fix_message_order()
    # logger.info(f"Fixed message order: {request.messages}")
    return await ServiceHandler.generate_text_response(request)

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests.
    Processes the request, checks for vision content, fixes message order, and generates the response.
    """
    return await handle_completion_request(request, "/chat/completions")

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests.
    Generates embeddings using the specified model.
    """
    return await ServiceHandler.generate_embeddings_response(request)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests (v1 API).
    """
    return await handle_completion_request(request, "/v1/chat/completions")

@app.post("/v1/embeddings")
async def v1_embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests (v1 API).
    """
    return await ServiceHandler.generate_embeddings_response(request)