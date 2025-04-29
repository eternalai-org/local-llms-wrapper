"""
Schema definitions for API requests and responses following OpenAI's API standard.
"""

import time
import re
import random
import os
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any, ClassVar

MAX_CONTEXT_LENGTH = 30000

# Precompile regex patterns for better performance
UNICODE_BOX_PATTERN = re.compile(r'\\u25[0-9a-fA-F]{2}')
NEWLINE_PATTERN = re.compile(r'\\n')
BACKTICKS_START_PATTERN = re.compile(r'```(?:\\n)?(?:\w+)?')
BACKTICKS_END_PATTERN = re.compile(r'```')

# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "gpt-4-turbo"          # Default model for text-based chat completions
    VISION_MODEL = "gpt-4-vision-preview"  # Model used for vision-based requests
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Model used for generating embeddings
    HTTP_TIMEOUT = 600.0                 # Default timeout for HTTP requests in seconds
    CACHE_TTL = 300                     # Cache time-to-live in seconds (5 minutes)
    MAX_RETRIES = 3                     # Maximum number of retries for HTTP requests
    POOL_CONNECTIONS = 100              # Maximum number of connections in the pool
    POOL_KEEPALIVE = 20                 # Keep connections alive for 20 seconds

# Service configuration
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", 8000))      # Port for the service
SERVICE_START_TIMEOUT = int(os.environ.get("SERVICE_START_TIMEOUT", 60))  # Timeout for service start
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", 300))       # Timeout for idle service (5 minutes)
UNLOAD_CHECK_INTERVAL = int(os.environ.get("UNLOAD_CHECK_INTERVAL", 60))  # Interval for checking idle status (1 minute)

# Helper functions
def _extract_reasoning_content(content: str) -> tuple[str, str]:
    """
    Extract reasoning content from <think> tags in the message content.
    
    Args:
        content: The message content that may contain <think> tags
        
    Returns:
        tuple: (updated_content, reasoning_content)
            updated_content: The content with <think> tags removed
            reasoning_content: The content inside <think> tags
    """
    if not content or "<think>" not in content:
        return content, ""
        
    # Extract reasoning content from <think> tags
    reasoning_content = ""
    updated_content = content
    
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    matches = think_pattern.findall(content)
    
    if matches:
        reasoning_content = "\n".join(matches)
        # Remove <think> tags from the content
        updated_content = think_pattern.sub('', content).strip()
        
    return updated_content, reasoning_content

# Chat completion models
class ChatCompletionRequest(BaseModel):
    """
    Model for chat completion requests, matching OpenAI's API schema.
    """
    model: str = Config.TEXT_MODEL          # Model to use, defaults to text model
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False          # Whether to stream the response
    tools: Optional[List[Dict[str, Any]]] = None   # Optional list of tools to use
    max_tokens: Optional[int] = None        # Maximum tokens in the response
    temperature: Optional[float] = None     # Temperature for sampling
    top_p: Optional[float] = None           # Top p for nucleus sampling
    frequency_penalty: Optional[float] = None  # Frequency penalty
    presence_penalty: Optional[float] = None   # Presence penalty
    stop: Optional[Union[str, List[str]]] = None  # Stop sequences
    seed: Optional[int] = 0                 # Seed for reproducibility
    
    @validator("messages")
    def validate_messages(cls, v):
        """
        Validate that messages list is not empty.
        """    
        if not v:
            raise ValueError("messages cannot be empty")
        return v
    
    def fix_messages(self) -> None:
        """
        Fix the messages list to ensure it starts with a system message if it exists.
        Also replaces null values with empty strings and cleans special box text.
        """
        def clean_special_box_text(input_text):
            if not isinstance(input_text, str):
                return ""
            # Apply all regex substitutions in one pass
            text = UNICODE_BOX_PATTERN.sub('', input_text)
            text = NEWLINE_PATTERN.sub('', text)
            text = BACKTICKS_START_PATTERN.sub('', text)
            text = BACKTICKS_END_PATTERN.sub('', text)
            return text.strip()
        
        # Process all messages at once
        system_messages = []
        non_system_messages = []
        
        for message in self.messages:
            # Replace null values with empty strings
            for key in message:
                if message[key] is None:
                    message[key] = ""
            
            # Clean content
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = clean_special_box_text(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = clean_special_box_text(item.get("text", ""))
            
            # Sort messages by role
            if message.get("role") == "system":
                system_messages.append(message)
            else:
                non_system_messages.append(message)
        
        # Reorder messages to ensure system message comes first if it exists
        if system_messages:
            self.messages = [system_messages[0]] + non_system_messages
    
    def is_vision_request(self) -> bool:
        """
        Check if the request includes image content, indicating a vision-based request.
        If so, switch the model to the vision model.
        """
        # Check if messages is empty first
        if not self.messages:
            return False
            
        # Optimization: Check only the last user message first
        last_message = self.messages[-1]
        if last_message.get("role") == "user" and self._check_message_for_image(last_message):
            return True
        
        # Fall back to checking all messages
        for message in self.messages:
            if message.get("role") == "user" and self._check_message_for_image(message):
                return True
                
        return False
    
    def _check_message_for_image(self, message: Dict[str, Any]) -> bool:
        """Helper method to check if a message contains an image."""
        content = message.get("content")
        if not isinstance(content, list):
            return False
            
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                self.model = Config.VISION_MODEL    
                return True
        return False

class ChatCompletionResponseChoice(BaseModel):
    """Model for a single choice in a chat completion response."""
    index: int
    message: Dict[str, Any]
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    """Model for chat completion responses following OpenAI's schema."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]
    
    @classmethod
    def create_from_content(cls, content: Any, model: str):
        """Create a standard response from content."""
        timestamp = int(time.time())
        # Handle case where content is a string but might contain <think> tags
        if isinstance(content, str) and "<think>" in content:
            updated_content, reasoning_content = _extract_reasoning_content(content)
            message = {
                "role": "assistant", 
                "content": updated_content
            }
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
        else:
            message = {"role": "assistant", "content": content if isinstance(content, str) else str(content)}
            
        return cls(
            id=f"chatcmpl-{timestamp}{random.randint(10000, 99999)}",
            created=timestamp,
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=message,
                    finish_reason="stop"
                )
            ],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any], model: str):
        """Create a standard response from dictionary data."""
        timestamp = int(time.time())
        
        message = {
            "role": "assistant", 
            "content": data.get("content", "")
        }
        
        # Add reasoning_content if it exists
        if "reasoning_content" in data:
            message["reasoning_content"] = data["reasoning_content"]
            
        return cls(
            id=data.get("id", f"chatcmpl-{timestamp}{random.randint(10000, 99999)}"),
            created=data.get("created", timestamp),
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=message,
                    finish_reason=data.get("finish_reason", "stop")
                )
            ],
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        )

# Embedding models
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Config.EMBEDDING_MODEL     # Model to use, defaults to embedding model
    input: Union[str, List[str]] = Field(..., description="Text input(s) for embedding")  # Text inputs to embed
    encoding_format: Optional[str] = "float"  # The format of the output data
    
    @validator("input")
    def check_input_not_empty(cls, v):
        """Ensure the input is not empty."""
        if isinstance(v, list) and not v:
            raise ValueError("input list cannot be empty")
        if isinstance(v, str) and not v.strip():
            raise ValueError("input string cannot be empty")
        return v

class EmbeddingResponseData(BaseModel):
    """Model for a single embedding in a response."""
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    """Model for embedding responses following OpenAI's schema."""
    object: str = "list"
    data: List[EmbeddingResponseData]
    model: str
    usage: Dict[str, int]
    
    # Default empty usage dict
    DEFAULT_USAGE: ClassVar[Dict[str, int]] = {"prompt_tokens": 0, "total_tokens": 0}

    @classmethod
    def create_from_embeddings(cls, embeddings: List[List[float]], model: str, input_texts: List[str]):
        """Create a standard response from a list of embeddings."""
        data = [
            EmbeddingResponseData(embedding=embedding, index=i)
            for i, embedding in enumerate(embeddings)
            if i < len(input_texts)  # Guard against index errors
        ]
                
        return cls(
            data=data,
            model=model,
            usage=cls.DEFAULT_USAGE
        )

    @classmethod
    def create_from_single_embedding(cls, embedding: List[float], model: str, usage: Optional[Dict[str, int]] = None):
        """Create a standard response from a single embedding."""
        return cls(
            data=[EmbeddingResponseData(embedding=embedding, index=0)],
            model=model,
            usage=usage or cls.DEFAULT_USAGE
        ) 