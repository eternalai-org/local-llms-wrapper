import os
import pickle
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
import requests
import subprocess


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
    model: str = "gemma3"
    choices: List[Choice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        running_service_file = os.getenv("RUNNING_SERVICE_FILE")
        with open(running_service_file, "rb") as f:
            service_info = pickle.load(f)
        print(f"Service info: {service_info}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load service info: {str(e)}")
    if multimodal:
        local_text_path = service_info["local_text_path"]
        local_projector_path = service_info["local_projector_path"]
        family = service_info["family"] 
        # only support gemma3 for multimodal
        cli = os.environ.get(family)
        # Prepare payload for OpenLLM
        payload = {
            "model": request.model,
            "messages": [],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }

        print(f"Processing {len(request.messages)} messages")
        # Process messages, handling multimodal content only
        for msg in request.messages:
            # Expect msg.content to be a list for multimodal input only
            if not isinstance(msg.content, list):
                raise HTTPException(status_code=400, detail="Multimodal messages must be a list of content parts")
            content_parts = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    content_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    image_url = part.image_url["url"]
                    try:
                        response = requests.get(image_url, timeout=5)
                        response.raise_for_status()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            tmp_file.write(response.content)
                            tmp_file_path = tmp_file.name
                            print(f"Running llama-gemma3-cli for image: {tmp_file_path}")
                            command_to_run = [
                                cli,
                                "--model", local_text_path,
                                "--mmproj", local_projector_path,
                                "--image", tmp_file_path,
                                "-p", "Describe this image",
                            ]
                            print(f"Running command: {command_to_run}")
                            result = subprocess.run(
                                command_to_run,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            output = result.stdout.decode("utf-8")
                            # Catch the output from the command
                            print(output)
                            return output
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)