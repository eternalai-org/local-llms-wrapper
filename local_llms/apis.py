from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
import requests
from PIL import Image
import io
import base64

app = FastAPI(title="Custom OpenAI-Compatible API with Vision")

# Define OpenAI-like request structures
class MessageContent(BaseModel):
    type: str = Field(..., description="Type of content: 'text' or 'image_url'")
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: Union[str, List[MessageContent]] = Field(..., description="Message content, can be string or list for multimodal input")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g., 'llava-13b'")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-custom-id"
    object: str = "chat.completion"
    created: int = 1623456789  # Placeholder timestamp
    model: str
    choices: List[Choice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Replace with your OpenLLM server URL
OPENLLM_URL = "http://localhost:3000/v1/chat/completions"

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    # Prepare payload for OpenLLM
    payload = {
        "model": request.model,
        "messages": [],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream
    }

    # Process messages, handling text and images
    for msg in request.messages:
        if isinstance(msg.content, str):
            payload["messages"].append({"role": msg.role, "content": msg.content})
        else:
            # Handle multimodal content (text + image)
            content_parts = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    content_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    # Download and encode image as base64 (OpenLLM may expect this)
                    image_url = part.image_url["url"]
                    try:
                        response = requests.get(image_url, timeout=5)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        content_parts.append({"type": "image", "data": f"data:image/png;base64,{img_base64}"})
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
            payload["messages"].append({"role": msg.role, "content": content_parts})

    # Forward request to OpenLLM server
    try:
        response = requests.post(OPENLLM_URL, json=payload, timeout=30)
        response.raise_for_status()
        openllm_data = response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenLLM server error: {str(e)}")

    # Transform OpenLLM response to OpenAI format
    choices = [
        Choice(
            index=0,
            message=Message(
                role="assistant",
                content=openllm_data["choices"][0]["message"]["content"]
            ),
            finish_reason=openllm_data["choices"][0].get("finish_reason", "stop")
        )
    ]
    return ChatCompletionResponse(
        model=request.model,
        choices=choices,
        usage=openllm_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)