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
import random
import time
import json
import uuid
import subprocess
import signal
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, Union, Any, Callable, Tuple
from functools import lru_cache

# Import schemas from schema.py
from local_llms.schema import (
    Config, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants for dynamic unload feature
IDLE_TIMEOUT = 600  # 10 minutes in seconds
UNLOAD_CHECK_INTERVAL = 60  # Check every 60 seconds

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
    async def kill_llama_server():
        """
        Kill the llama-server process if it's running.
        """
        try:
            # Get the PID from the service info
            if not hasattr(app.state, "service_info") or "pid" not in app.state.service_info:
                logger.warning("No PID found in service info, cannot kill llama-server")
                return False
                
            pid = app.state.service_info["pid"]
            logger.info(f"Attempting to kill llama-server with PID {pid}")
            
            # Try to kill the process
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Successfully sent SIGTERM to llama-server (PID: {pid})")
            
            # Wait a moment and check if the process is still running
            await asyncio.sleep(2)
            try:
                os.kill(pid, 0)  # Check if process exists
                # If we get here, the process is still running, try SIGKILL
                logger.warning(f"Process {pid} still running after SIGTERM, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                # Process is already gone, which is good
                pass
                
            # Remove the PID from service info
            if hasattr(app.state, "service_info"):
                app.state.service_info.pop("pid", None)
                
            return True
        except Exception as e:
            logger.error(f"Error killing llama-server: {str(e)}")
            return False
    
    @staticmethod
    async def reload_llama_server():
        """
        Reload the llama-server process.
        """
        try:
            # Get the command to start llama-server from the service info
            if not hasattr(app.state, "service_info") or "running_llm_command" not in app.state.service_info:
                logger.error("No running_llm_command found in service info, cannot reload llama-server")
                return False
                
            command = app.state.service_info["running_llm_command"]
            logger.info(f"Reloading llama-server with command: {command}")
            
            # Start the process in the background
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setpgrp  # Run in a new process group
            )
            
            # Wait a moment for the process to start
            await asyncio.sleep(2)
            
            # Check if the process is running
            if process.poll() is None:
                # Process is running, update the PID in service info
                if hasattr(app.state, "service_info"):
                    app.state.service_info["pid"] = process.pid
                logger.info(f"Successfully reloaded llama-server with PID {process.pid}")
                return True
            else:
                # Process failed to start
                stdout, stderr = process.communicate()
                logger.error(f"Failed to reload llama-server: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Error reloading llama-server: {str(e)}")
            return False
    
    @staticmethod
    def _detect_and_fix_tool_calls_in_content(response_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Detect and fix malformed tool calls that are encoded as JSON strings in content.
        
        Args:
            response_data: The response data from the LLM
            
        Returns:
            Tuple[Dict, bool]: The fixed response data and a boolean indicating if a retry is needed
        """
        need_retry = False
        
        # Handle non-dict responses
        if not isinstance(response_data, dict):
            return response_data, need_retry
        
        # Make a copy to avoid modifying the original
        fixed_response = response_data.copy()
        
        # Process only if the response has choices
        if "choices" not in fixed_response:
            return fixed_response, need_retry
        
        choices = fixed_response.get("choices", [])
        
        # Iterate through all choices
        for i, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            
            message = choice.get("message", {})
            if not message or "content" not in message or not message["content"]:
                continue
            
            content = message["content"]
            
            # Check if content contains a JSON string with tool_calls
            if not isinstance(content, str) or "tool_calls" not in content:
                continue
            
            # Try to parse as JSON
            try:
                parsed_content = json.loads(content)
                
                # Check if parsed content has tool_calls
                if not isinstance(parsed_content, dict) or "tool_calls" not in parsed_content:
                    continue
                
                tool_calls = parsed_content.get("tool_calls", [])
                
                # Only proceed if tool_calls is a non-empty list
                if not isinstance(tool_calls, list) or not tool_calls:
                    logger.warning("Found tool_calls in content but it's not a valid list")
                    need_retry = True
                    continue
                
                # Found valid tool_calls, let's fix the response
                logger.warning(f"Found tool_calls in content: {tool_calls}")
                
                # Create fixed message structure
                fixed_message = message.copy()
                fixed_message["tool_calls"] = tool_calls
                fixed_message["content"] = ""  # Set content to empty string instead of null
                
                # Update the choice with fixed message
                fixed_choice = choice.copy()
                fixed_choice["message"] = fixed_message
                
                # Update choices list
                new_choices = choices.copy() 
                new_choices[i] = fixed_choice
                fixed_response["choices"] = new_choices
                
                logger.info("Successfully fixed tool_calls in response")
                return fixed_response, False  # No need to retry as we fixed it
                
            except (json.JSONDecodeError, ValueError) as e:
                # JSON parsing failed
                logger.warning(f"Content contains 'tool_calls' but couldn't parse JSON: {e}")
                need_retry = True
        
        # Return the possibly modified response and retry flag
        return fixed_response, need_retry
    
    @staticmethod
    async def generate_text_response(request: ChatCompletionRequest):
        """
        Generate a response for chat completion requests, supporting both streaming and non-streaming.
        """
        port = await ServiceHandler.get_service_port()

        request.fix_messages()
        if request.is_vision_request():
            raise HTTPException(status_code=400, detail="This model does not support vision-based requests")
        
        # Convert to dict, supporting both Pydantic v1 and v2
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()

        if request.stream:
            if request.tools:
                # For streaming with tools, we need to get the non-streaming response first
                # and then simulate streaming from it
                stream_request = request_dict.copy()
                stream_request["stream"] = False  # Get non-streaming response first
                
                # Make a non-streaming API call
                response_data = await ServiceHandler._make_api_call(port, "/v1/chat/completions", stream_request)
                
                # Return a simulated streaming response
                return StreamingResponse(
                    ServiceHandler._fake_stream_with_tools(response_data, request.model),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            
            # Return a streaming response for non-tool requests
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        response_data = await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_dict)
        
        # Format the response according to OpenAI's schema
        if not isinstance(response_data, dict):
            # If the response is a string or other non-dict type, wrap it
            response = ChatCompletionResponse.create_from_content(response_data, request.model)
            return response.model_dump() if hasattr(response, "model_dump") else response.dict()
        
        # If response is already in OpenAI format, return it
        if "choices" in response_data and "object" in response_data:
            return response_data
        
        # Otherwise, format it
        response = ChatCompletionResponse.create_from_dict(response_data, request.model)
        return response.model_dump() if hasattr(response, "model_dump") else response.dict()

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

            # Create formatted response
            response = ChatCompletionResponse.create_from_content(stdout.decode().strip(), request.model)
            return response.model_dump() if hasattr(response, "model_dump") else response.dict()
            
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
        response_data = await ServiceHandler._make_api_call(port, "/v1/embeddings", request_dict)
        
        # Handle if the response is already in the OpenAI format
        if isinstance(response_data, dict) and "data" in response_data and "object" in response_data:
            return response_data
        
        # Handle when the response is a raw list of embeddings or a single embedding
        if isinstance(response_data, list):
            # List of embeddings
            input_texts = request.input if isinstance(request.input, list) else [request.input]
            response = EmbeddingResponse.create_from_embeddings(response_data, request.model, input_texts)
            return response.model_dump() if hasattr(response, "model_dump") else response.dict()
            
        elif isinstance(response_data, dict) and "embedding" in response_data:
            # Single embedding in a dict
            response = EmbeddingResponse.create_from_single_embedding(
                response_data["embedding"], 
                request.model, 
                response_data.get("usage")
            )
            return response.model_dump() if hasattr(response, "model_dump") else response.dict()
        else:
            # Unexpected format, return as is
            return response_data
    
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
                    response_data = response.json()
                    
                    # Check for tool calls in content and fix if needed
                    fixed_data, need_retry = ServiceHandler._detect_and_fix_tool_calls_in_content(response_data)
                    
                    # If we need to retry but we're not on the last attempt
                    if need_retry and attempts < retries - 1:
                        logger.info("Detected issue with tool calls, will retry")
                        attempts += 1
                        # Use exponential backoff before retry
                        sleep_time = (2 ** attempts) + (0.1 * random.random())
                        await asyncio.sleep(sleep_time)
                        continue
                    
                    return fixed_data
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

    @staticmethod
    async def _fake_stream_with_tools(formatted_response: dict, model: str):
        """
        Generate a fake streaming response for tool-based chat completions.
        This method simulates the streaming behavior by breaking a complete response into chunks.
        
        Args:
            formatted_response: The complete response to stream in chunks
            model: The model name
        """
        
        # Base structure for each chunk
        base_chunk = {
            "id": formatted_response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion.chunk",
            "created": formatted_response.get("created", int(time.time())),
            "model": formatted_response.get("model", model),
        }
        
        # Add system_fingerprint only if it exists in the response
        if "system_fingerprint" in formatted_response:
            base_chunk["system_fingerprint"] = formatted_response["system_fingerprint"]

        choices = formatted_response.get("choices", [])
        if not choices:
            # If no choices, return empty response and DONE
            yield f"data: {json.dumps({**base_chunk, 'choices': []})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Step 1: Initial chunk with role for all choices
        initial_choices = [
            {
                "index": choice["index"],
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None
            }
            for choice in choices
        ]
        yield f"data: {json.dumps({**base_chunk, 'choices': initial_choices})}\n\n"

        # Step 2: Chunk with content or tool_calls for all choices
        content_choices = []
        for choice in choices:
            message = choice.get("message", {})
            delta = {}
            
            # For tool calls responses
            if "tool_calls" in message:
                updated_tool_calls = []
                for idx, tool_call in enumerate(message["tool_calls"]):
                    updated_tool_calls.append(tool_call)
                    updated_tool_calls[idx]["index"] = str(idx)
                delta["tool_calls"] = updated_tool_calls
                delta["reasoning_content"] = ""
            # For content responses
            elif message.get("content"):
                delta["content"] = message["content"]
                delta["reasoning_content"] = ""
            else:
                # Empty content/null case
                delta["reasoning_content"] = ""
                
            if delta:  # Only include choices with content
                content_choices.append({
                    "index": choice["index"],
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": None
                })
                
        if content_choices:
            yield f"data: {json.dumps({**base_chunk, 'choices': content_choices})}\n\n"

        # Step 3: Final chunk with finish reason for all choices
        finish_choices = [
            {
                "index": choice["index"],
                "delta": {},
                "logprobs": None,
                "finish_reason": choice["finish_reason"]
            }
            for choice in choices
        ]
        yield f"data: {json.dumps({**base_chunk, 'choices': finish_choices})}\n\n"

        # Step 4: End of stream
        yield "data: [DONE]\n\n"

# Request Processor
class RequestProcessor:
    """
    Class for processing requests sequentially using a queue.
    Ensures that only one request is processed at a time to accommodate limitations
    of backends like llama-server that can only handle one request at a time.
    """
    queue = asyncio.Queue()  # Queue for sequential request processing
    processing_lock = asyncio.Lock()  # Lock to ensure only one request is processed at a time
    
    # Define which endpoints need to be processed sequentially
    MODEL_ENDPOINTS = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
    }  # Mapping of endpoints to their request models and handlers
    
    @staticmethod
    async def process_request(endpoint: str, request_data: dict):
        """
        Process a request by adding it to the queue and waiting for the result.
        This ensures requests are processed in order, one at a time.
        Returns a Future that will be resolved with the result.
        """
        request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
        queue_size = RequestProcessor.queue.qsize()
        
        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint} (queue size: {queue_size})")
        
        # Update the last request time
        app.state.last_request_time = time.time()
        
        # Check if we need to reload the llama-server
        if hasattr(app.state, "service_info") and "pid" not in app.state.service_info:
            logger.info(f"[{request_id}] Llama-server not running, reloading...")
            await ServiceHandler.reload_llama_server()
        
        start_wait_time = time.time()
        future = asyncio.Future()
        await RequestProcessor.queue.put((endpoint, request_data, future, request_id, start_wait_time))
        
        # Wait for the future to be resolved
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def process_direct(endpoint: str, request_data: dict):
        """
        Process a request directly without queueing.
        Use this for administrative endpoints that don't require model access.
        """
        request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
        logger.info(f"[{request_id}] Processing direct request for endpoint {endpoint}")
        
        # Update the last request time
        app.state.last_request_time = time.time()
        
        # Check if we need to reload the llama-server
        if hasattr(app.state, "service_info") and "pid" not in app.state.service_info:
            logger.info(f"[{request_id}] Llama-server not running, reloading...")
            await ServiceHandler.reload_llama_server()
        
        start_time = time.time()
        if endpoint in RequestProcessor.MODEL_ENDPOINTS:
            model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
            request_obj = model_cls(**request_data)
            result = await handler(request_obj)
            
            process_time = time.time() - start_time
            logger.info(f"[{request_id}] Direct request completed for endpoint {endpoint} (time: {process_time:.2f}s)")
            
            return result
        else:
            logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
            raise HTTPException(status_code=404, detail="Endpoint not found")
    
    # Global worker function
    @staticmethod
    async def worker():
        """
        Worker function to process requests from the queue sequentially.
        Only one request is processed at a time.
        """
        logger.info("Request processor worker started")
        processed_count = 0
        
        while True:
            try:
                endpoint, request_data, future, request_id, start_wait_time = await RequestProcessor.queue.get()
                
                wait_time = time.time() - start_wait_time
                queue_size = RequestProcessor.queue.qsize()
                processed_count += 1
                
                logger.info(f"[{request_id}] Processing request from queue for endpoint {endpoint} "
                           f"(wait time: {wait_time:.2f}s, queue size: {queue_size}, processed: {processed_count})")
                
                # Use the lock to ensure only one request is processed at a time
                async with RequestProcessor.processing_lock:
                    processing_start = time.time()
                    
                    if endpoint in RequestProcessor.MODEL_ENDPOINTS:
                        model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
                        try:
                            request_obj = model_cls(**request_data)
                            result = await handler(request_obj)
                            future.set_result(result)
                            
                            processing_time = time.time() - processing_start
                            total_time = time.time() - start_wait_time
                            
                            logger.info(f"[{request_id}] Completed request for endpoint {endpoint} "
                                       f"(processing: {processing_time:.2f}s, total: {total_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"[{request_id}] Handler error for {endpoint}: {str(e)}")
                            future.set_exception(e)
                    else:
                        logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
                        future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
                
                # Log periodic status about queue health
                if processed_count % 10 == 0:
                    logger.info(f"Queue status: current size={queue_size}, processed={processed_count}")
                
            except asyncio.CancelledError:
                logger.info("Worker task cancelled, exiting")
                break  # Exit the loop when the task is canceled
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                # Continue working, don't crash the worker

# Unload checker task
async def unload_checker():
    """
    Periodically check if the llama-server has been idle for too long and unload it if needed.
    """
    logger.info("Unload checker task started")
    
    while True:
        try:
            # Wait for the check interval
            await asyncio.sleep(UNLOAD_CHECK_INTERVAL)
            logger.info(f"Unload checker task running at {time.time()}")
            
            # Check if the service is running and has been idle for too long
            if (hasattr(app.state, "service_info") and 
                "pid" in app.state.service_info and 
                hasattr(app.state, "last_request_time")):
                
                idle_time = time.time() - app.state.last_request_time
                
                if idle_time > IDLE_TIMEOUT:
                    logger.info(f"Llama-server has been idle for {idle_time:.2f}s, unloading...")
                    await ServiceHandler.kill_llama_server()
            
        except asyncio.CancelledError:
            logger.info("Unload checker task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in unload checker task: {str(e)}")
            # Continue running despite errors

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
    
    # Initialize the last request time
    app.state.last_request_time = time.time()
    
    # Start the worker
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    
    # Start the unload checker task
    app.state.unload_checker_task = asyncio.create_task(unload_checker())
    
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
    
    # Cancel the unload checker task
    if hasattr(app.state, "unload_checker_task"):
        app.state.unload_checker_task.cancel()
        try:
            await app.state.unload_checker_task  # Wait for the task to finish
        except asyncio.CancelledError:
            pass  # Handle cancellation gracefully
    
    # Kill the llama-server if it's running
    if hasattr(app.state, "service_info") and "pid" in app.state.service_info:
        await ServiceHandler.kill_llama_server()
    
    logger.info("Service shutdown complete")

# API Endpoints
@app.get("/health")
@app.get("/v1/health")
async def health():
    """
    Health check endpoint.
    Returns a simple status to indicate the service is running.
    This endpoint bypasses the request queue for immediate response.
    """
    # Invalidate the service port cache periodically
    get_cached_service_port.cache_clear()
    
    # Check if the service info is set
    if not hasattr(app.state, "service_info"):
        return {"status": "starting", "message": "Service info not set yet"}
    
    # Update the last request time
    app.state.last_request_time = time.time()
    
    return {"status": "ok", "service": app.state.service_info.get("family", "unknown")}


@app.post("/update")
async def update(request: dict):
    """
    Update the service information in the app's state.
    Stores the provided request data for use in determining the service port.
    This endpoint bypasses the request queue for immediate response.
    """
    app.state.service_info = request
    # Invalidate the cache when service info is updated
    get_cached_service_port.cache_clear()
    logger.info(f"Updated service info: {request.get('family', 'unknown')} on port {request.get('port', 'unknown')}")
    return {"status": "ok", "message": "Service info updated successfully"}

# Modified endpoint handlers for model-based endpoints
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests.
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/chat/completions", request_dict)

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests.
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/embeddings", request_dict)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """
    Endpoint for chat completion requests (v1 API).
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict)

@app.post("/v1/embeddings")
async def v1_embeddings(request: EmbeddingRequest):
    """
    Endpoint for embedding requests (v1 API).
    Uses the request queue to ensure only one model request is processed at a time.
    """
    # Convert to dict, supporting both Pydantic v1 and v2
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    return await RequestProcessor.process_request("/v1/embeddings", request_dict)