"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""
import os
import tempfile
import logging
import httpx
import asyncio
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator, confloat, conint
from typing import List, Dict, Optional, Union, Any, Tuple, Set
from loguru import logger
import pickle
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("local_llms_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of active requests",
    ["endpoint"]
)
QUEUE_SIZE = Gauge(
    "queue_size",
    "Current size of the request queue"
)

# Configuration
class Config:
    """
    Configuration class holding the default model names and rate limiting settings.
    """
    TEXT_MODEL = "gpt-4-turbo"
    VISION_MODEL = "gpt-4-vision-preview"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    RATE_LIMIT_REQUESTS = 100  # requests per minute
    RATE_LIMIT_WINDOW = 60  # seconds
    MAX_TOKENS = 4096
    MAX_MESSAGES = 100
    REQUEST_TIMEOUT = 30  # seconds
    MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent requests
    QUEUE_TIMEOUT = 300  # Maximum time to wait in queue (5 minutes)
    MAX_RETRIES = 3  # Maximum number of retries for failed requests
    RETRY_DELAY = 1  # Initial delay between retries in seconds
    MAX_RETRY_DELAY = 10  # Maximum delay between retries in seconds
    HEALTH_CHECK_INTERVAL = 30  # seconds
    METRICS_ENABLED = True

# Custom exceptions
class ServiceUnavailableError(Exception):
    """Raised when the underlying service is unavailable."""
    pass

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    pass

class RequestTimeoutError(Exception):
    """Raised when a request times out."""
    pass

class ValidationError(Exception):
    """Raised when request validation fails."""
    pass

app = FastAPI(
    title="Local LLM API",
    description="API for interacting with local Large Language Models",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize Prometheus metrics
if Config.METRICS_ENABLED:
    Instrumentator().instrument(app).expose(app)

# Request Queue Management
class RequestQueue:
    """Manages the request queue and processing."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_requests = 0
        self.lock = asyncio.Lock()
        self.workers = []
        self.is_running = True
        self._health_check_task = None
        self._last_health_check = 0

    async def add_request(self, request_id: str, request_data: dict) -> asyncio.Future:
        """Add a request to the queue and return a future for its result."""
        future = asyncio.Future()
        await self.queue.put((request_id, request_data, future))
        QUEUE_SIZE.set(self.queue.qsize())
        return future

    async def process_request(self, request_id: str, request_data: dict, future: asyncio.Future):
        """Process a single request."""
        try:
            async with self.lock:
                self.active_requests += 1
                ACTIVE_REQUESTS.labels(endpoint=request_data.get("type", "unknown")).inc()

            # Process the request based on its type
            if "messages" in request_data:
                result = await self._process_chat_request(request_data)
            else:
                result = await self._process_embedding_request(request_data)

            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            async with self.lock:
                self.active_requests -= 1
                ACTIVE_REQUESTS.labels(endpoint=request_data.get("type", "unknown")).dec()

    async def _process_chat_request(self, request_data: dict) -> dict:
        """Process a chat completion request with retries."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                port = await ServiceHandler.get_service_port()
                if request_data.get("stream", False):
                    return StreamingResponse(
                        ServiceHandler._stream_generator(port, request_data),
                        media_type="text/event-stream"
                    )
                return await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_data)
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                delay = min(Config.RETRY_DELAY * (2 ** attempt), Config.MAX_RETRY_DELAY)
                logger.warning(f"Retry {attempt + 1}/{Config.MAX_RETRIES} after {delay}s: {str(e)}")
                await asyncio.sleep(delay)

    async def _process_embedding_request(self, request_data: dict) -> dict:
        """Process an embedding request with retries."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                port = await ServiceHandler.get_service_port()
                return await ServiceHandler._make_api_call(port, "/v1/embeddings", request_data)
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                delay = min(Config.RETRY_DELAY * (2 ** attempt), Config.MAX_RETRY_DELAY)
                logger.warning(f"Retry {attempt + 1}/{Config.MAX_RETRIES} after {delay}s: {str(e)}")
                await asyncio.sleep(delay)

    async def _health_check(self):
        """Periodically check service health."""
        while self.is_running:
            try:
                port = await ServiceHandler.get_service_port()
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get(f"http://localhost:{port}/health")
                    response.raise_for_status()
                    self._last_health_check = time.time()
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
            await asyncio.sleep(Config.HEALTH_CHECK_INTERVAL)

    async def worker(self):
        """Worker function to process requests from the queue."""
        while self.is_running:
            try:
                request_id, request_data, future = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=Config.QUEUE_TIMEOUT
                )
                start_time = time.time()
                
                try:
                    await self.process_request(request_id, request_data, future)
                finally:
                    duration = time.time() - start_time
                    REQUEST_LATENCY.labels(endpoint=request_data.get("type", "unknown")).observe(duration)
                    REQUEST_COUNT.labels(
                        endpoint=request_data.get("type", "unknown"),
                        status="success"
                    ).inc()
                
                self.queue.task_done()
                QUEUE_SIZE.set(self.queue.qsize())
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                REQUEST_COUNT.labels(
                    endpoint=request_data.get("type", "unknown"),
                    status="error"
                ).inc()

    async def start_workers(self, num_workers: int = 3):
        """Start worker tasks to process requests."""
        self.workers = [
            asyncio.create_task(self.worker())
            for _ in range(num_workers)
        ]
        self._health_check_task = asyncio.create_task(self._health_check())

    async def stop_workers(self):
        """Stop all worker tasks."""
        self.is_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)

# Initialize request queue
request_queue = RequestQueue()

# Rate limiting with improved tracking
class RateLimiter:
    def __init__(self, requests_per_minute: int, window_seconds: int):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_rate_limited(self, client_id: str) -> bool:
        async with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
            
            # Check if rate limit exceeded
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return True
                
            self.requests[client_id].append(now)
            return False

rate_limiter = RateLimiter(Config.RATE_LIMIT_REQUESTS, Config.RATE_LIMIT_WINDOW)

async def get_client_id(request: Request) -> str:
    """Get client identifier from request headers or IP."""
    return request.headers.get("X-Client-ID", request.client.host)

async def check_rate_limit(client_id: str = Depends(get_client_id)):
    """Dependency to check rate limiting."""
    if await rate_limiter.is_rate_limited(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later."
        )
    return client_id

# Data Models with improved validation
class Message(BaseModel):
    """Represents a single message in a chat completion request."""
    role: str
    content: Optional[Union[str, List[Dict[str, str]]]]

    @validator("role")
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("role must be one of: user, assistant, system")
        return v

    @validator("content")
    def validate_content(cls, v):
        if v is None:
            return " "
        return v

class ChatCompletionRequest(BaseModel):
    """Model for chat completion requests with improved validation."""
    model: str = Config.TEXT_MODEL
    messages: List[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[conint(le=Config.MAX_TOKENS)] = None
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    top_p: Optional[confloat(ge=0.0, le=1.0)] = None
    presence_penalty: Optional[confloat(ge=-2.0, le=2.0)] = None
    frequency_penalty: Optional[confloat(ge=-2.0, le=2.0)] = None
    stop: Optional[Union[str, List[str]]] = None

    @validator("messages")
    def check_messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > Config.MAX_MESSAGES:
            raise ValueError(f"Too many messages. Maximum allowed: {Config.MAX_MESSAGES}")
        return v

    def is_vision_request(self) -> bool:
        """Check if the request includes image content."""
        for message in self.messages:
            content = message.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        self.model = Config.VISION_MODEL
                        return True
        return False

    def fix_message_order(self) -> None:
        """Ensure messages alternate between user and assistant roles."""
        if not self.messages:
            return

        fixed_messages = []
        last_role = None

        for msg in self.messages:
            role = msg.role.strip()
            content = msg.content or " "

            if (last_role in ("user", "assistant")) and role == last_role:
                fixed_messages.append(Message(
                    role="assistant" if last_role == "user" else "user",
                    content=" "
                ))

            fixed_messages.append(Message(role=role, content=content))
            last_role = role

        self.messages = fixed_messages

class EmbeddingRequest(BaseModel):
    """Model for embedding requests with improved validation."""
    model: str = Config.EMBEDDING_MODEL
    input: List[str] = Field(..., description="List of text inputs for embedding")

    @validator("input")
    def validate_input(cls, v):
        if not v:
            raise ValueError("input cannot be empty")
        if len(v) > 100:  # Limit batch size
            raise ValueError("Too many inputs. Maximum allowed: 100")
        return v

# Service Handler with improved error handling
class ServiceHandler:
    """Handles communication with the underlying LLM service."""

    @staticmethod
    async def get_service_port() -> int:
        """Get the port number of the running LLM service."""
        try:
            with open("running_service.pkl", "rb") as f:
                service_info = pickle.load(f)
                port = service_info.get("app_port")
                if not port:
                    raise ServiceUnavailableError("Service port not found")
                return port
        except Exception as e:
            logger.error(f"Error getting service port: {str(e)}")
            raise ServiceUnavailableError("LLM service is not running")

    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: dict) -> dict:
        """Make an API call to the LLM service with improved error handling."""
        url = f"http://localhost:{port}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=data)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            raise RequestTimeoutError("Request timed out")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitExceededError("Rate limit exceeded")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Service error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Service error: {str(e)}"
            )

    @staticmethod
    async def _stream_generator(port: int, data: dict):
        """Generate streaming response from the LLM service with error handling."""
        url = f"http://localhost:{port}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
                async with client.stream("POST", url, json=data) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting Local LLM API service")
    await request_queue.start_workers()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Local LLM API service")
    await request_queue.stop_workers()

# API Endpoints with improved error handling and metrics
@app.get("/health")
async def health():
    """Health check endpoint with detailed status."""
    try:
        port = await ServiceHandler.get_service_port()
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"http://localhost:{port}/health")
            response.raise_for_status()
            return {
                "status": "ok",
                "queue_size": request_queue.queue.qsize(),
                "active_requests": request_queue.active_requests,
                "last_health_check": request_queue._last_health_check,
                "uptime": time.time() - request_queue._last_health_check
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(check_rate_limit)
):
    """Handle chat completion requests with improved error handling."""
    try:
        # Fix message order and prepare request data
        request.fix_message_order()
        request_data = request.dict()
        request_data["type"] = "chat"
        
        # Add request to queue and wait for result
        future = await request_queue.add_request(client_id, request_data)
        return await future
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(check_rate_limit)
):
    """Handle embedding requests with improved error handling."""
    try:
        # Add request to queue and wait for result
        request_data = request.dict()
        request_data["type"] = "embedding"
        future = await request_queue.add_request(client_id, request_data)
        return await future
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/v1/chat/completions")
async def chat_completions_v1(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(check_rate_limit)
):
    """Handle v1 chat completion requests."""
    return await chat_completions(request, background_tasks, client_id)

@app.post("/v1/embeddings")
async def embeddings_v1(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(check_rate_limit)
):
    """Handle v1 embedding requests."""
    return await embeddings(request, background_tasks, client_id)

# Error handlers
@app.exception_handler(ServiceUnavailableError)
async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)}
    )

@app.exception_handler(RateLimitExceededError)
async def rate_limit_handler(request: Request, exc: RateLimitExceededError):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)}
    )

@app.exception_handler(RequestTimeoutError)
async def timeout_handler(request: Request, exc: RequestTimeoutError):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": str(exc)}
    )

@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )