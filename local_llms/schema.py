"""
Schema definitions for API requests and responses following OpenAI's API standard.
"""

import time
import random
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
    HTTP_TIMEOUT = 600.0                # Default timeout for HTTP requests in seconds (increased to 10 minutes)
    CACHE_TTL = 300                     # Cache time-to-live in seconds (5 minutes)
    MAX_RETRIES = 5                     # Maximum number of retries for HTTP requests (increased from 3)
    POOL_CONNECTIONS = 100              # Maximum number of connections in the pool
    POOL_KEEPALIVE = 20                 # Keep connections alive for 20 seconds

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
        if not self.messages:
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
        if isinstance(content, str):
            message = {"role": "assistant", "content": content}
        else:
            message = {"role": "assistant", "content": str(content)}
            
        return cls(
            id=f"chatcmpl-{int(time.time())}{random.randint(10000, 99999)}",
            created=int(time.time()),
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
        return cls(
            id=data.get("id", f"chatcmpl-{int(time.time())}{random.randint(10000, 99999)}"),
            created=data.get("created", int(time.time())),
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": data.get("content", "")},
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

    @classmethod
    def create_from_embeddings(cls, embeddings: List[List[float]], model: str, input_texts: List[str]):
        """Create a standard response from a list of embeddings."""
        data = []
        for i, embedding in enumerate(embeddings):
            if i < len(input_texts):  # Guard against index errors
                data.append(EmbeddingResponseData(
                    embedding=embedding,
                    index=i
                ))
                
        return cls(
            data=data,
            model=model,
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    @classmethod
    def create_from_single_embedding(cls, embedding: List[float], model: str, usage: Optional[Dict[str, int]] = None):
        """Create a standard response from a single embedding."""
        return cls(
            data=[EmbeddingResponseData(
                embedding=embedding,
                index=0
            )],
            model=model,
            usage=usage or {"prompt_tokens": 0, "total_tokens": 0}
        ) 