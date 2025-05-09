"""
Schema definitions for API requests and responses following OpenAI's API standard.
"""

import time
import re
import random
from typing_extensions import Literal
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any, ClassVar

# Precompile regex patterns for better performance
UNICODE_BOX_PATTERN = re.compile(r'\\u25[0-9a-fA-F]{2}')

# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "gpt-4-turbo"          # Default model for text-based chat completions
    VISION_MODEL = "gpt-4-vision-preview"  # Model used for vision-based requests
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Model used for generating embeddings

# Common models used in both streaming and non-streaming contexts
class ImageUrl(BaseModel):
    """
    Represents an image URL in a message.
    """
    url: str

class VisionContentItem(BaseModel):
    """
    Represents a single content item in a message (text or image).
    """
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class FunctionCall(BaseModel):
    """
    Represents a function call in a message.
    """
    arguments: str
    name: str

class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call in a message.
    """
    id: str
    function: FunctionCall
    type: Literal["function"]

class Message(BaseModel):
    """
    Represents a message in a chat completion.
    """
    content: Union[str, List[VisionContentItem]]
    refusal: Optional[str] = None
    role: Literal["system", "user", "assistant", "tool"]
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(BaseModel):
    """
    Base model for chat completion requests.
    """
    model: str = Config.TEXT_MODEL
    messages: List[Message]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    n: Optional[int] = 1
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    enable_thinking: Optional[bool] = False

    @validator("messages")
    def check_messages_not_empty(cls, v):
        """
        Ensure that the messages list is not empty and validate message structure.
        """
        if not v:
            raise ValueError("messages cannot be empty")
        
        # Validate message history length
        if len(v) > 100:  # OpenAI's limit is typically around 100 messages
            raise ValueError("message history too long")
            
        # Validate message roles
        valid_roles = {"user", "assistant", "system", "tool"}
        for msg in v:
            if msg.role not in valid_roles:
                raise ValueError(f"invalid role: {msg.role}")
                
        return v

    @validator("temperature")
    def check_temperature(cls, v):
        """
        Validate temperature is between 0 and 2.
        """
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    def is_vision_request(self) -> bool:
        """
        Check if the request includes image content, indicating a vision-based request.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        for message in self.messages:
            if isinstance(message.content, list):
                for item in message.content:
                    if item.type == "image_url":
                        logger.debug(f"Detected vision request with image: {item.image_url.url[:30]}...")
                        return True
        
        logger.debug(f"No images detected, treating as text-only request")
        return False
    
    def fix_messages(self) -> None:
        """
        Fix the messages list to ensure it starts with a system message if it exists.
        Also replaces null values with empty strings and cleans special box text.
        """
        def clean_special_box_text(input_text):
            text = UNICODE_BOX_PATTERN.sub('', input_text)
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
        if not self.enable_thinking:
            final_message = non_system_messages[-1]
            final_message["content"] = final_message["content"] + " /no_think"
            non_system_messages[-1] = final_message
        # Reorder messages to ensure system message comes first if it exists
        if system_messages:
            self.messages = [system_messages[0]] + non_system_messages
        else:
            self.messages = non_system_messages

# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for non-streaming chat completion requests.
    """
    stream: bool = False
    enable_thinking: bool = True

class Choice(BaseModel):
    """
    Represents a choice in a chat completion response.
    """
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    index: int
    message: Message

class ChatCompletionResponse(BaseModel):
    """
    Represents a complete chat completion response.
    """
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]

# Embedding models
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Config.EMBEDDING_MODEL
    input: List[str] = Field(..., description="List of text inputs for embedding")
    image_url: Optional[str] = Field(default=None, description="Image URL to embed")

class Embedding(BaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="The index of the embedding in the list")
    object: str = Field(default="embedding", description="The object type")

class EmbeddingResponse(BaseModel):
    """
    Represents an embedding response.
    """
    object: str = "list"
    data: List[Embedding]
    model: str